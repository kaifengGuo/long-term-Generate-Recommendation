import argparse
import json
from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from eval_tiger_phase2_blend_env import TigerPhase2BlendPolicy
from model.reward import get_immediate_reward
from tiger_phase2_blend_common import (
    build_history_tokens,
    build_iid2sid_tokens,
    infer_model_size_args,
    load_tiger_model,
)
import utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulator preference pairs for TIGER-SPO.")
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--eval_log_path", type=str, default="")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate_pool_size", type=int, default=16)
    parser.add_argument("--num_candidate_slates", type=int, default=6)
    parser.add_argument("--rollout_horizon", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min_reward_gap", type=float, default=0.05)
    parser.add_argument("--min_overlap_items", type=int, default=0)
    parser.add_argument("--min_diff_items", type=int, default=0)
    parser.add_argument("--max_diff_items", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def clone_observation_row(observation: Dict[str, Dict[str, torch.Tensor]], row_idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {"user_profile": {}, "user_history": {}}
    for section in out.keys():
        for key, value in observation[section].items():
            out[section][key] = value[row_idx:row_idx + 1].detach().clone()
    return out


@contextmanager
def temporary_env_state(
    env: KREnvironment_WholeSession_GPU,
    *,
    observation: Dict[str, Dict[str, torch.Tensor]],
    temper: torch.Tensor,
    step: torch.Tensor,
    sum_reward: torch.Tensor,
):
    saved_observation = env.current_observation
    saved_temper = env.current_temper
    saved_step = env.current_step
    saved_sum_reward = getattr(env, "current_sum_reward", None)
    saved_batch_size = env.episode_batch_size
    try:
        env.current_observation = deepcopy(observation)
        env.current_temper = temper.detach().clone()
        env.current_step = step.detach().clone()
        if saved_sum_reward is not None:
            env.current_sum_reward = sum_reward.detach().clone()
        env.episode_batch_size = 1
        yield
    finally:
        env.current_observation = saved_observation
        env.current_temper = saved_temper
        env.current_step = saved_step
        if saved_sum_reward is not None:
            env.current_sum_reward = saved_sum_reward
        env.episode_batch_size = saved_batch_size


def build_policy_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        beam_width=int(args.beam_width),
        max_hist_items=int(args.max_hist_items),
        phase2_blend_scale=0.0,
        phase2_decode_topk=0,
        phase4_prefix_scale=0.0,
        phase3_actor_scale=0.0,
        phase5_token_actor_scale=0.0,
        phase6_prefix_scale=0.0,
        phase6_token_actor_scale=0.0,
        online_slate_allocator_scale=0.0,
        slate_value_scale=0.0,
        slate_rerank_pool=max(int(args.slate_size), 8),
        slate_greedy_candidates=max(int(args.slate_size), 6),
        fast_base_generate=True,
        disable_decoder_kv_cache=False,
    )


@torch.no_grad()
def get_ranked_item_candidates(
    policy: TigerPhase2BlendPolicy,
    observation: Dict[str, Dict[str, torch.Tensor]],
    candidate_info: Dict[str, torch.Tensor],
    row_idx: int,
) -> Tuple[List[int], Dict[int, float]]:
    hist_iids = observation["user_history"]["history"][row_idx:row_idx + 1].long().to(policy.device)
    cand_iids = candidate_info["item_id"].long().detach().cpu().numpy()
    global_to_local = {int(gid): idx for idx, gid in enumerate(cand_iids)}
    user_hist_items = set(int(x) for x in hist_iids[0].detach().cpu().numpy().tolist())

    input_ids, attention_mask = build_history_tokens(
        hist_iids,
        policy.iid2sid_tok,
        int(policy.max_hist_items),
        int(policy.sid_depth),
    )
    item_score_pairs: List[Tuple[int, float]] = []
    gen = policy.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=int(policy.beam_width),
        num_return_sequences=int(policy.beam_width),
        max_length=int(policy.sid_depth) + 1,
        early_stopping=True,
        do_sample=False,
    )
    gen = gen[:, 1:1 + int(policy.sid_depth)]
    beams = gen.view(1, int(policy.beam_width), int(policy.sid_depth)).detach().cpu().numpy()[0]
    for rank in range(int(policy.beam_width)):
        key = tuple(int(x) for x in beams[rank].tolist())
        gid = policy.sid2iid_map.get(key)
        if gid is None:
            continue
        if gid not in global_to_local or gid in user_hist_items:
            continue
        score = float(int(policy.beam_width) - rank)
        item_score_pairs.append((int(global_to_local[gid]), score))

    item_score_pairs.sort(key=lambda x: x[1], reverse=True)
    score_by_idx: Dict[int, float] = {}
    candidate_idxs: List[int] = []
    for idx_local, score in item_score_pairs:
        if int(idx_local) in score_by_idx:
            continue
        score_by_idx[int(idx_local)] = float(score)
        candidate_idxs.append(int(idx_local))
    return candidate_idxs, score_by_idx


def _fill_slate(base: List[int], pool: List[int], slate_size: int) -> List[int]:
    out = []
    seen = set()
    for x in base + pool:
        if int(x) in seen:
            continue
        seen.add(int(x))
        out.append(int(x))
        if len(out) >= int(slate_size):
            break
    return out[: int(slate_size)]


def build_candidate_slates(
    candidate_idxs: Sequence[int],
    score_by_idx: Dict[int, float],
    *,
    slate_size: int,
    candidate_pool_size: int,
    num_candidate_slates: int,
) -> List[List[int]]:
    pool = [int(x) for x in candidate_idxs[: max(int(candidate_pool_size), int(slate_size))]]
    if len(pool) < int(slate_size):
        return []
    slates: List[List[int]] = []
    top = _fill_slate(pool[: int(slate_size)], pool, int(slate_size))
    if len(top) == int(slate_size):
        slates.append(top)

    for start in range(1, max(1, len(pool) - int(slate_size) + 1)):
        slate = _fill_slate(pool[start:start + int(slate_size)], pool, int(slate_size))
        if len(slate) == int(slate_size):
            slates.append(slate)
        if len(slates) >= int(num_candidate_slates):
            break

    if len(slates) < int(num_candidate_slates):
        top_slate = slates[0] if slates else top
        for repl_idx in range(int(slate_size)):
            for pool_idx in range(int(slate_size), len(pool)):
                candidate = list(top_slate)
                candidate[repl_idx] = int(pool[pool_idx])
                slate = _fill_slate(candidate, pool, int(slate_size))
                if len(slate) == int(slate_size):
                    slates.append(slate)
                if len(slates) >= int(num_candidate_slates):
                    break
            if len(slates) >= int(num_candidate_slates):
                break

    uniq: List[List[int]] = []
    seen = set()
    for slate in slates:
        key = tuple(int(x) for x in slate)
        if key in seen:
            continue
        seen.add(key)
        uniq.append([int(x) for x in slate])
        if len(uniq) >= int(num_candidate_slates):
            break
    return uniq


def score_slate_with_branch_rollout(
    env: KREnvironment_WholeSession_GPU,
    policy: TigerPhase2BlendPolicy,
    *,
    row_observation: Dict[str, Dict[str, torch.Tensor]],
    row_temper: torch.Tensor,
    row_step: torch.Tensor,
    row_sum_reward: torch.Tensor,
    action_local: Sequence[int],
    horizon: int,
    gamma: float,
) -> Tuple[float, List[float], bool]:
    total = 0.0
    reward_trace: List[float] = []
    action_tensor = torch.tensor([list(action_local)], dtype=torch.long, device=env.device)

    with temporary_env_state(
        env,
        observation=row_observation,
        temper=row_temper,
        step=row_step,
        sum_reward=row_sum_reward,
    ):
        for t in range(int(max(1, horizon))):
            response_dict = env.get_response({"action": action_tensor})
            response_dict["immediate_response_weight"] = env.response_weights
            reward = float(get_immediate_reward(response_dict).item())
            reward_trace.append(reward)
            total += (float(gamma) ** t) * reward
            done_mask = env.get_leave_signal(None, None, response_dict)
            done = bool(done_mask[0].item())
            env.update_observation(None, action_tensor, response_dict["immediate_response"], done_mask, update_current=True)
            env.current_step = env.current_step + 1
            if done or t >= int(horizon) - 1:
                return total, reward_trace, done
            cand = env.get_candidate_info()
            action_tensor = policy.act(env.current_observation, cand)[:1]
    return total, reward_trace, False


def local_to_global(candidate_info: Dict[str, torch.Tensor], slate_local: Sequence[int]) -> List[int]:
    cand_iids = candidate_info["item_id"].detach().cpu().numpy().tolist()
    out = []
    for idx in slate_local:
        if 0 <= int(idx) < len(cand_iids):
            out.append(int(cand_iids[int(idx)]))
    return out


def global_to_sid_tokens(iid2sid_tok: torch.Tensor, item_ids: Sequence[int]) -> List[List[int]]:
    out: List[List[int]] = []
    for iid in item_ids:
        if 0 <= int(iid) < int(iid2sid_tok.shape[0]):
            out.append([int(x) for x in iid2sid_tok[int(iid)].detach().cpu().tolist()])
        else:
            out.append([])
    return out


def slate_overlap_stats(chosen_item_ids: Sequence[int], rejected_item_ids: Sequence[int]) -> Tuple[int, int]:
    chosen = set(int(x) for x in chosen_item_ids)
    rejected = set(int(x) for x in rejected_item_ids)
    overlap = len(chosen & rejected)
    diff = max(len(chosen), len(rejected)) - overlap
    return int(overlap), int(diff)


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(args.device)

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = KREnvironment_WholeSession_GPU(args)
    if hasattr(env, "set_seed"):
        env.set_seed(int(args.seed))
    size_args = infer_model_size_args(str(args.model_size))
    tiger, sid_depth, _codebook_size = load_tiger_model(
        tiger_ckpt=str(args.tiger_ckpt),
        sid_mapping_path=str(args.sid_mapping_path),
        num_layers=int(size_args["num_layers"]),
        num_decoder_layers=int(size_args["num_decoder_layers"]),
        d_model=int(size_args["d_model"]),
        d_ff=int(size_args["d_ff"]),
        num_heads=int(size_args["num_heads"]),
        d_kv=int(size_args["d_kv"]),
        dropout_rate=0.1,
        feed_forward_proj="relu",
        device=device,
    )
    iid2sid_tok, sid2iid_map_tok = build_iid2sid_tokens(env.reader, str(args.sid_mapping_path), int(sid_depth), device)
    policy = TigerPhase2BlendPolicy(
        model=tiger,
        iid2sid_tok=iid2sid_tok,
        sid2iid_map_tok=sid2iid_map_tok,
        sid_depth=int(sid_depth),
        device=device,
        slate_size=int(args.slate_size),
        reader=env.reader,
        args=build_policy_args(args),
        phase2_head=None,
        phase4_prefix_head=None,
        phase3_actor_head=None,
        phase5_token_actor_head=None,
        phase6_plan_head=None,
        phase6_prefix_head=None,
        phase6_token_actor_head=None,
        online_slate_allocator_head=None,
        slate_value_head=None,
    )

    observation = env.reset({"batch_size": int(args.episode_batch_size), "empty_history": False})
    finished = 0
    pair_count = 0
    page_index_by_row = np.ones(int(args.episode_batch_size), dtype=np.int64)
    with output_path.open("w", encoding="utf-8") as out:
        while finished < int(args.num_episodes):
            cand = env.get_candidate_info()
            action = policy.act(observation, cand)
            bsz = int(action.shape[0])
            cand_iids = cand["item_id"].detach().cpu().numpy()

            for row_idx in range(bsz):
                if int(args.max_pairs) > 0 and pair_count >= int(args.max_pairs):
                    break
                candidate_idxs, score_by_idx = get_ranked_item_candidates(policy, observation, cand, row_idx)
                candidate_slates = build_candidate_slates(
                    candidate_idxs,
                    score_by_idx,
                    slate_size=int(args.slate_size),
                    candidate_pool_size=int(args.candidate_pool_size),
                    num_candidate_slates=int(args.num_candidate_slates),
                )
                if len(candidate_slates) < 2:
                    continue

                row_observation = clone_observation_row(observation, row_idx)
                row_temper = env.current_temper[row_idx:row_idx + 1].detach().clone()
                row_step = env.current_step[row_idx:row_idx + 1].detach().clone()
                row_sum_reward = env.current_sum_reward[row_idx:row_idx + 1].detach().clone()

                scored: List[Dict[str, Any]] = []
                for slate_local in candidate_slates:
                    score, reward_trace, done = score_slate_with_branch_rollout(
                        env,
                        policy,
                        row_observation=row_observation,
                        row_temper=row_temper,
                        row_step=row_step,
                        row_sum_reward=row_sum_reward,
                        action_local=slate_local,
                        horizon=int(args.rollout_horizon),
                        gamma=float(args.gamma),
                    )
                    scored.append(
                        {
                            "slate_local": [int(x) for x in slate_local],
                            "score": float(score),
                            "reward_trace": [float(x) for x in reward_trace],
                            "done": bool(done),
                        }
                    )

                scored.sort(key=lambda x: x["score"], reverse=True)
                best = scored[0]
                worst = scored[-1]
                gap = float(best["score"] - worst["score"])
                if gap < float(args.min_reward_gap):
                    continue

                chosen_item_ids = local_to_global(cand, best["slate_local"])
                rejected_item_ids = local_to_global(cand, worst["slate_local"])
                if len(chosen_item_ids) != int(args.slate_size) or len(rejected_item_ids) != int(args.slate_size):
                    continue
                overlap_items, diff_items = slate_overlap_stats(chosen_item_ids, rejected_item_ids)
                if int(args.min_overlap_items) > 0 and int(overlap_items) < int(args.min_overlap_items):
                    continue
                if int(args.min_diff_items) > 0 and int(diff_items) < int(args.min_diff_items):
                    continue
                if int(args.max_diff_items) > 0 and int(diff_items) > int(args.max_diff_items):
                    continue
                payload = {
                    "episode_progress_index": int(finished),
                    "page_index": int(page_index_by_row[row_idx]),
                    "user_id": int(observation["user_profile"]["user_id"][row_idx].item()),
                    "history_items": [int(x) for x in observation["user_history"]["history"][row_idx].detach().cpu().numpy().tolist() if int(x) > 0],
                    "chosen_item_ids": chosen_item_ids,
                    "chosen_sid_tokens_list": global_to_sid_tokens(iid2sid_tok, chosen_item_ids),
                    "rejected_item_ids": rejected_item_ids,
                    "rejected_sid_tokens_list": global_to_sid_tokens(iid2sid_tok, rejected_item_ids),
                    "chosen_score": float(best["score"]),
                    "rejected_score": float(worst["score"]),
                    "reward_gap": float(gap),
                    "chosen_reward_trace": [float(x) for x in best["reward_trace"]],
                    "rejected_reward_trace": [float(x) for x in worst["reward_trace"]],
                    "chosen_done": bool(best["done"]),
                    "rejected_done": bool(worst["done"]),
                    "candidate_count": int(len(scored)),
                    "overlap_items": int(overlap_items),
                    "diff_items": int(diff_items),
                    "candidate_pool_item_ids": [int(cand_iids[int(idx)]) for idx in candidate_idxs[: int(args.candidate_pool_size)] if 0 <= int(idx) < len(cand_iids)],
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                pair_count += 1

            observation, response_dict, _update_info = env.step({"action": action})
            done_mask = response_dict["done"]
            for row_idx in range(int(done_mask.shape[0])):
                if bool(done_mask[row_idx].item()):
                    finished += 1
                    page_index_by_row[row_idx] = 1
                else:
                    page_index_by_row[row_idx] += 1
            if finished % max(1, int(args.log_every)) == 0:
                print(f"[pairs] finished={finished}/{args.num_episodes} pairs={pair_count}")
            if int(args.max_pairs) > 0 and pair_count >= int(args.max_pairs):
                break

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = {
        "method": "TIGER SPO Pair Builder",
        "output_path": str(output_path.resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "num_episodes": int(args.num_episodes),
        "num_pairs": int(pair_count),
        "slate_size": int(args.slate_size),
        "episode_batch_size": int(args.episode_batch_size),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "candidate_pool_size": int(args.candidate_pool_size),
        "num_candidate_slates": int(args.num_candidate_slates),
        "rollout_horizon": int(args.rollout_horizon),
        "gamma": float(args.gamma),
        "min_reward_gap": float(args.min_reward_gap),
        "min_overlap_items": int(args.min_overlap_items),
        "min_diff_items": int(args.min_diff_items),
        "max_diff_items": int(args.max_diff_items),
        "single_response": bool(args.single_response),
        "initial_temper": float(args.initial_temper),
        "temper_consume_mode": str(args.temper_consume_mode),
        "temper_consume_decay": float(args.temper_consume_decay),
        "temper_cost_offset": float(args.temper_cost_offset),
        "temper_max_drop": float(args.temper_max_drop),
        "history_consume_mode": str(args.history_consume_mode),
        "history_consume_decay": float(args.history_consume_decay),
        "seed": int(args.seed),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pairs] saved {pair_count} pairs to {output_path}")
    print(f"[pairs] meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
