import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_tiger_phase3_credit_chain import build_transport_chain, transform_episode_credits  # noqa: E402
from tiger_phase2_blend_common import build_iid2sid_tokens  # noqa: E402
from tiger_phase7_welfare_common import compute_welfare_step_reward, resolve_reward_weights  # noqa: E402
from tiger_slate_allocator_common import build_bootstrap_target_shares  # noqa: E402

from tiger_hcla_rl.common import (  # noqa: E402
    build_history_state,
    build_item_features,
    build_page_features,
    compute_support_strength,
    load_reader_from_uirm_log,
    normalize_scores,
    set_random_seed,
    weighted_response_strength,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build long-term counterfactual HCLA chain for TIGER.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--credit_mode", type=str, default="centered", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--reward_preset", type=str, default="click_longview")
    parser.add_argument("--reward_weights_json", type=str, default="")
    parser.add_argument("--use_trace_step_reward", action="store_true")
    parser.add_argument("--hazard_lambda", type=float, default=0.35)
    parser.add_argument("--item_response_scale", type=float, default=1.0)
    parser.add_argument("--item_reward_scale", type=float, default=0.50)
    parser.add_argument("--item_support_scale", type=float, default=0.35)
    parser.add_argument("--item_share_mode", type=str, default="bootstrap", choices=["bootstrap", "heuristic"])
    parser.add_argument("--share_heuristic_mix", type=float, default=0.60)
    parser.add_argument("--share_support_mix", type=float, default=0.25)
    parser.add_argument("--share_response_mix", type=float, default=0.15)
    parser.add_argument("--transport_epsilon", type=float, default=0.35)
    parser.add_argument("--transport_iter", type=int, default=16)
    parser.add_argument("--cf_topk_history", type=int, default=5)
    parser.add_argument("--cf_smooth", type=float, default=0.05)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def load_trace_rows(trace_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "selected_item_ids" not in payload:
                continue
            rows.append(payload)
    if not rows:
        raise ValueError(f"No usable trace rows in {trace_path}")
    return rows


def history_sid_matrix(history_items: Sequence[int], iid2sid_tok_cpu: torch.Tensor) -> np.ndarray:
    hist = [int(x) for x in history_items if 0 <= int(x) < int(iid2sid_tok_cpu.shape[0])]
    if not hist:
        return np.zeros((0, 0), dtype=np.int64)
    hist_sid = iid2sid_tok_cpu[torch.tensor(hist, dtype=torch.long)].detach().cpu().numpy()
    valid = np.any(hist_sid > 0, axis=1)
    return hist_sid[valid]


def baseline_by_page_index(page_infos: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for row in page_infos:
        buckets.setdefault(int(row["page_index"]), []).append(float(row["lt_page_value_raw"]))
    return {int(k): float(np.mean(v)) if v else 0.0 for k, v in buckets.items()}


def allocate_signed_item_adv(
    *,
    page_adv: float,
    item_rewards: np.ndarray,
    response_strengths: np.ndarray,
    support_strengths: np.ndarray,
    response_scale: float,
    reward_scale: float,
    support_scale: float,
) -> np.ndarray:
    if item_rewards.size <= 0:
        return np.zeros((0,), dtype=np.float32)
    reward_centered = item_rewards - float(item_rewards.mean())
    response_centered = response_strengths - float(response_strengths.mean()) if response_strengths.size > 0 else np.zeros_like(item_rewards)
    support_centered = support_strengths - float(support_strengths.mean()) if support_strengths.size > 0 else np.zeros_like(item_rewards)
    signal = (
        float(response_scale) * response_centered
        + float(reward_scale) * reward_centered
        + float(support_scale) * support_centered
    )
    if abs(float(page_adv)) <= 1e-8:
        return np.zeros_like(item_rewards, dtype=np.float32)
    if float(page_adv) >= 0.0:
        shares = normalize_scores(np.maximum(signal, 0.0) + 0.05 * np.maximum(reward_centered, 0.0))
        return (abs(float(page_adv)) * shares).astype(np.float32)
    shares = normalize_scores(np.maximum(-signal, 0.0) + 0.05 * np.maximum(-reward_centered, 0.0))
    return (-abs(float(page_adv)) * shares).astype(np.float32)


def main() -> int:
    args = parse_args()
    set_random_seed(int(args.seed))

    trace_path = Path(args.trace_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace_rows = load_trace_rows(trace_path)
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(args.device))
    response_names = list(reader.get_statistics()["feedback_type"])
    reward_weights, reward_weight_map = resolve_reward_weights(
        response_names,
        preset=str(args.reward_preset),
        reward_weights_json=str(args.reward_weights_json),
    )
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in trace_rows:
        grouped.setdefault(str(rec["episode_id"]), []).append(rec)

    page_infos: List[Dict[str, Any]] = []
    for episode_id, pages in sorted(grouped.items(), key=lambda x: int(x[0])):
        pages = sorted(pages, key=lambda x: int(x.get("page_index", 0)))
        if int(args.max_pages) > 0:
            pages = pages[: int(args.max_pages)]
        lt_step_rewards: List[float] = []
        for rec in pages:
            base_reward = float(rec.get("step_reward", 0.0)) if bool(args.use_trace_step_reward) else compute_welfare_step_reward(
                rec.get("selected_responses", []),
                reward_weights,
            )
            hazard = 1.0 if bool(rec.get("done", False)) else 0.0
            lt_step_reward = float(base_reward - float(args.hazard_lambda) * hazard)
            lt_step_rewards.append(float(lt_step_reward))

        returns = [0.0 for _ in pages]
        running = 0.0
        for idx in range(len(pages) - 1, -1, -1):
            running = float(lt_step_rewards[idx]) + float(args.gamma) * running
            returns[idx] = float(running)

        for idx, rec in enumerate(pages):
            page_infos.append(
                {
                    "episode_id": int(episode_id),
                    "page_index": int(rec.get("page_index", idx + 1)),
                    "trace": rec,
                    "lt_step_reward": float(lt_step_rewards[idx]),
                    "lt_page_value_raw": float(returns[idx]),
                }
            )

    page_baselines = baseline_by_page_index(page_infos)
    raw_adv_by_episode: Dict[int, List[float]] = {}
    for row in page_infos:
        episode_id = int(row["episode_id"])
        raw_adv = float(row["lt_page_value_raw"] - page_baselines.get(int(row["page_index"]), 0.0))
        row["lt_page_adv_raw"] = float(raw_adv)
        raw_adv_by_episode.setdefault(episode_id, []).append(float(raw_adv))

    adv_transform_by_episode: Dict[int, List[float]] = {}
    for episode_id, raw_adv in raw_adv_by_episode.items():
        adv_transform_by_episode[int(episode_id)] = transform_episode_credits(
            raw_adv,
            str(args.credit_mode),
            float(args.credit_clip),
        )

    episode_page_ptr: Dict[int, int] = {}
    page_feature_names: List[str] = []
    item_feature_names: List[str] = []
    n_rows = 0
    max_page_index = max(int(r["page_index"]) for r in page_infos) if page_infos else 1
    max_slate_size = max(int(len(r["trace"].get("selected_item_ids", []))) for r in page_infos) if page_infos else 1
    with output_path.open("w", encoding="utf-8") as out:
        for row in page_infos:
            episode_id = int(row["episode_id"])
            page_pos = int(episode_page_ptr.get(episode_id, 0))
            episode_page_ptr[episode_id] = page_pos + 1
            page_adv = float(adv_transform_by_episode[episode_id][page_pos])
            rec = row["trace"]
            history_items = [int(x) for x in rec.get("history_items", [])][-int(args.max_hist_items):]
            input_ids, attention_mask = build_history_state(
                history_items,
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(args.max_hist_items),
                sid_depth=int(sid_depth),
            )

            selected_item_ids = [int(x) for x in rec.get("selected_item_ids", [])]
            selected_tokens_list = [[int(v) for v in seq] for seq in rec.get("selected_sid_tokens_list", [])]
            selected_responses = [[float(v) for v in resp] for resp in rec.get("selected_responses", [])]
            selected_item_rewards = [float(x) for x in rec.get("selected_item_rewards", [])]
            slate_size = int(len(selected_item_ids))
            if slate_size <= 0:
                continue
            if len(selected_tokens_list) != slate_size:
                selected_tokens_list = []
                for iid in selected_item_ids:
                    if 0 <= int(iid) < int(iid2sid_tok_cpu.shape[0]):
                        selected_tokens_list.append([int(x) for x in iid2sid_tok_cpu[int(iid)].tolist()])
                    else:
                        selected_tokens_list.append([0 for _ in range(int(sid_depth))])

            history_sid = history_sid_matrix(history_items, iid2sid_tok_cpu)
            response_strengths = np.asarray(
                [weighted_response_strength(resp, reward_weights) for resp in selected_responses],
                dtype=np.float32,
            )
            support_strengths = np.asarray(
                [compute_support_strength(history_sid, tgt) for tgt in selected_tokens_list],
                dtype=np.float32,
            )
            item_rewards = np.asarray(selected_item_rewards, dtype=np.float32)
            if item_rewards.size != slate_size:
                item_rewards = np.zeros((slate_size,), dtype=np.float32)
            reward_centered = item_rewards - float(item_rewards.mean()) if item_rewards.size > 0 else np.zeros((slate_size,), dtype=np.float32)
            response_centered = response_strengths - float(response_strengths.mean()) if response_strengths.size > 0 else np.zeros((slate_size,), dtype=np.float32)
            support_centered = support_strengths - float(support_strengths.mean()) if support_strengths.size > 0 else np.zeros((slate_size,), dtype=np.float32)
            bootstrap_shares, heuristic_shares, _centered_rewards_bootstrap, _response_strength_bootstrap = build_bootstrap_target_shares(
                page_credit=float(page_adv),
                item_rewards=item_rewards.tolist(),
                responses=selected_responses,
                support_strengths=support_strengths.tolist(),
                response_weights=reward_weights,
                heuristic_mix=float(args.share_heuristic_mix),
                support_mix=float(args.share_support_mix),
                response_mix=float(args.share_response_mix),
            )
            if str(args.item_share_mode) == "bootstrap":
                share_abs = np.asarray(bootstrap_shares, dtype=np.float32).reshape(-1)
                if share_abs.shape[0] != slate_size:
                    share_abs = normalize_scores(np.ones((slate_size,), dtype=np.float32))
                item_adv = float(page_adv) * share_abs
            else:
                item_adv = allocate_signed_item_adv(
                    page_adv=float(page_adv),
                    item_rewards=item_rewards,
                    response_strengths=response_strengths,
                    support_strengths=support_strengths,
                    response_scale=float(args.item_response_scale),
                    reward_scale=float(args.item_reward_scale),
                    support_scale=float(args.item_support_scale),
                )
                share_abs = normalize_scores(np.abs(item_adv)) if item_adv.size > 0 and float(np.abs(item_adv).sum()) > 1e-8 else np.zeros((slate_size,), dtype=np.float32)
            heuristic_shares = np.asarray(heuristic_shares, dtype=np.float32).reshape(-1)
            if heuristic_shares.shape[0] != slate_size:
                heuristic_shares = np.zeros((slate_size,), dtype=np.float32)

            page_features, page_feature_names = build_page_features(
                page_index=int(row["page_index"]),
                max_page_index=int(max_page_index),
                history_len=len(history_items),
                max_hist_items=int(args.max_hist_items),
                slate_size=int(slate_size),
                max_slate_size=int(max_slate_size),
                selected_item_rewards=selected_item_rewards,
                selected_responses=selected_responses,
                step_reward=float(row["lt_step_reward"]),
                done=bool(rec.get("done", False)),
                response_strengths=response_strengths.tolist(),
                support_strengths=support_strengths.tolist(),
            )

            for item_pos, item_id in enumerate(selected_item_ids):
                target_tokens = selected_tokens_list[item_pos] if item_pos < len(selected_tokens_list) else []
                if len(target_tokens) != int(sid_depth):
                    continue
                item_feature, item_feature_names = build_item_features(
                    slate_item_index=int(item_pos),
                    slate_size=int(slate_size),
                    item_reward=float(item_rewards[item_pos]) if item_pos < item_rewards.size else 0.0,
                    item_reward_centered=float(reward_centered[item_pos]) if item_pos < reward_centered.size else 0.0,
                    response_strength=float(response_strengths[item_pos]) if item_pos < response_strengths.size else 0.0,
                    response_strength_centered=float(response_centered[item_pos]) if item_pos < response_centered.size else 0.0,
                    support_strength=float(support_strengths[item_pos]) if item_pos < support_strengths.size else 0.0,
                    support_strength_centered=float(support_centered[item_pos]) if item_pos < support_centered.size else 0.0,
                    item_share=float(share_abs[item_pos]) if item_pos < share_abs.size else 0.0,
                    page_adv=float(page_adv),
                    done=bool(rec.get("done", False)),
                )
                transport = build_transport_chain(
                    history_items=history_items,
                    target_tokens=target_tokens,
                    item_credit=float(item_adv[item_pos]),
                    iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                    epsilon=float(args.transport_epsilon),
                    n_iter=int(args.transport_iter),
                    cf_topk_history=int(args.cf_topk_history),
                    cf_smooth=float(args.cf_smooth),
                )
                token_adv = [float(x) for x in transport["token_credit_calibrated"]]
                token_prefix = np.cumsum(np.asarray(token_adv, dtype=np.float32), dtype=np.float32).tolist()
                payload = {
                    "episode_id": int(episode_id),
                    "page_index": int(row["page_index"]),
                    "user_id": int(rec.get("user_id", -1)),
                    "slate_size": int(slate_size),
                    "slate_item_index": int(item_pos),
                    "selected_item_id": int(item_id),
                    "selected_item_ids": [int(x) for x in selected_item_ids],
                    "history_items": [int(x) for x in history_items],
                    "input_ids": [int(x) for x in input_ids],
                    "attention_mask": [int(x) for x in attention_mask],
                    "selected_sid_tokens": [int(x) for x in target_tokens],
                    "selected_sid_tokens_list": [[int(v) for v in seq] for seq in selected_tokens_list],
                    "selected_item_rewards": [float(x) for x in item_rewards.tolist()],
                    "selected_responses": [[float(v) for v in resp] for resp in selected_responses],
                    "selected_item_credit_shares": [float(x) for x in share_abs.tolist()],
                    "selected_item_credit_shares_heuristic": [float(x) for x in heuristic_shares.tolist()],
                    "selected_item_cf_adv_list": [float(x) for x in item_adv.tolist()],
                    "selected_response": [float(x) for x in selected_responses[item_pos]] if item_pos < len(selected_responses) else [],
                    "selected_item_reward": float(item_rewards[item_pos]) if item_pos < item_rewards.size else 0.0,
                    "lt_page_reward": float(row["lt_step_reward"]),
                    "lt_page_value_raw": float(row["lt_page_value_raw"]),
                    "lt_page_baseline": float(page_baselines.get(int(row["page_index"]), 0.0)),
                    "lt_page_adv_raw": float(row["lt_page_adv_raw"]),
                    "lt_page_adv": float(page_adv),
                    "lt_page_hazard": float(1.0 if bool(rec.get("done", False)) else 0.0),
                    "lt_page_survival": float(0.0 if bool(rec.get("done", False)) else 1.0),
                    "lt_item_cf_adv": float(item_adv[item_pos]) if item_pos < item_adv.size else 0.0,
                    "lt_item_cf_value": float(row["lt_page_value_raw"] - item_adv[item_pos]) if item_pos < item_adv.size else float(row["lt_page_value_raw"]),
                    "lt_item_share": float(share_abs[item_pos]) if item_pos < share_abs.size else 0.0,
                    "lt_token_cf_adv": [float(x) for x in token_adv],
                    "lt_token_cf_prefix": [float(x) for x in token_prefix],
                    "response_strength": float(response_strengths[item_pos]) if item_pos < response_strengths.size else 0.0,
                    "support_strength": float(support_strengths[item_pos]) if item_pos < support_strengths.size else 0.0,
                    "page_features": [float(x) for x in page_features.tolist()],
                    "item_features": [float(x) for x in item_feature.tolist()],
                    "history_credit_calibrated": [float(x) for x in transport["history_credit_calibrated"]],
                    "token_transport_adv": [float(x) for x in transport["token_credit_calibrated"]],
                    "done": bool(rec.get("done", False)),
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n_rows += 1

    meta = {
        "method": "TIGER HCLA-RL chain",
        "trace_path": str(trace_path.resolve()),
        "output_path": str(output_path.resolve()),
        "n_trace_rows": int(len(trace_rows)),
        "n_chain_rows": int(n_rows),
        "gamma": float(args.gamma),
        "credit_mode": str(args.credit_mode),
        "credit_clip": float(args.credit_clip),
        "hazard_lambda": float(args.hazard_lambda),
        "item_share_mode": str(args.item_share_mode),
        "share_heuristic_mix": float(args.share_heuristic_mix),
        "share_support_mix": float(args.share_support_mix),
        "share_response_mix": float(args.share_response_mix),
        "reward_preset": str(args.reward_preset),
        "reward_weights": reward_weight_map,
        "sid_depth": int(sid_depth),
        "page_feature_names": page_feature_names,
        "item_feature_names": item_feature_names,
    }
    write_json(str(output_path) + ".meta.json", meta)
    print(f"[hcla-chain] saved {n_rows} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
