import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import utils  # noqa: E402
from reader import *  # noqa: F401,F403,E402

from tiger_phase2_blend_common import build_history_tokens, build_iid2sid_tokens, build_sid_prefix_to_next, decoder_input_ids_from_targets, infer_model_size_args, load_tiger_model  # noqa: E402

from tiger_page_sid_rl.common import (  # noqa: E402
    build_page_scalar_features,
    build_user_feature_vector,
    get_user_feature_layout,
    pooled_history_summary,
    write_json,
)
from tiger_page_sid_rl.models import load_page_sid_qcritic_bundle  # noqa: E402


def load_reader_from_uirm_log(uirm_log_path: str, device: str):
    with open(uirm_log_path, "r", encoding="utf-8") as infile:
        class_args = eval(infile.readline(), {"Namespace": Namespace})
        training_args = eval(infile.readline(), {"Namespace": Namespace})
    training_args.val_holdout_per_user = 0
    training_args.test_holdout_per_user = 0
    training_args.device = device
    reader_class = eval("{0}.{0}".format(class_args.reader))
    return reader_class(training_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped candidate sets for TIGER-HCA-GRPO actor training.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--critic_bundle_path", type=str, required=True)
    parser.add_argument("--critic_meta_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--critic_eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--reward_field",
        type=str,
        default="page_q_value",
        choices=[
            "page_q_value",
            "page_q_mean",
            "page_q_pess",
            "item_advantage",
            "item_advantage_mean",
            "item_advantage_pess",
            "adaptive_support_pess",
        ],
    )
    parser.add_argument("--critic_pessimism_beta", type=float, default=0.0)
    parser.add_argument("--support_penalty_scale", type=float, default=0.0)
    parser.add_argument("--support_gap_temperature", type=float, default=1.0)
    parser.add_argument("--support_gap_clip", type=float, default=0.0)
    parser.add_argument("--adaptive_beta_unc_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_beta_support_scale", type=float, default=0.0)
    parser.add_argument(
        "--reward_transform",
        type=str,
        default="raw",
        choices=["raw", "centered", "clipped_margin", "tanh_margin"],
    )
    parser.add_argument("--reward_margin_clip", type=float, default=0.0)
    parser.add_argument("--reward_margin_temperature", type=float, default=1.0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default="")
    return parser.parse_args()


def load_chain_rows(chain_path: Path, max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with chain_path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "selected_item_ids" not in payload or "slate_item_index" not in payload:
                continue
            rows.append(payload)
    if not rows:
        raise ValueError(f"No usable rows in {chain_path}")
    return rows


def decode_ranked_sequences(
    tiger,
    *,
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    sid_prefix_to_next: Dict[Tuple[int, ...], List[int]],
    sid_depth: int,
    beam_width: int,
    device: torch.device,
) -> List[Tuple[List[int], float]]:
    beams: List[Tuple[List[int], float]] = [([], 0.0)]
    for _step in range(int(sid_depth)):
        prefixes = [[0] + seq for seq, _score in beams]
        max_len = max(len(prefix) for prefix in prefixes)
        dec = np.zeros((len(prefixes), max_len), dtype=np.int64)
        for row_idx, prefix in enumerate(prefixes):
            dec[row_idx, -len(prefix):] = np.asarray(prefix, dtype=np.int64)
        dec_input = torch.tensor(dec, dtype=torch.long, device=device)
        enc_hidden = encoder_hidden.expand(len(prefixes), -1, -1)
        enc_attn = attention_mask.expand(len(prefixes), -1)
        logits, _hidden = tiger.decode_with_hidden_from_encoded(
            encoder_hidden=enc_hidden,
            attention_mask=enc_attn,
            decoder_input_ids=dec_input,
        )
        step_scores = torch.log_softmax(logits[:, -1, :], dim=-1)
        step_scores[:, 0] = -1e9

        candidates: List[Tuple[List[int], float]] = []
        for beam_idx, (seq, base_score) in enumerate(beams):
            allowed = sid_prefix_to_next.get(tuple(seq), [])
            if not allowed:
                continue
            allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
            allowed_scores = step_scores[beam_idx, allowed_tensor]
            topk = min(int(beam_width), int(allowed_scores.shape[0]))
            vals, idxs = torch.topk(allowed_scores, k=max(topk, 1), dim=0)
            for local_pos in range(topk):
                token = int(allowed[int(idxs[local_pos].item())])
                score = float(base_score + vals[local_pos].item())
                candidates.append((seq + [token], score))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[1], reverse=True)
        next_beams: List[Tuple[List[int], float]] = []
        seen = set()
        for seq, score in candidates:
            key = tuple(int(x) for x in seq)
            if key in seen:
                continue
            seen.add(key)
            next_beams.append((seq, score))
            if len(next_beams) >= int(beam_width):
                break
        beams = next_beams
    return beams


def evaluate_q_variants(
    critic,
    *,
    pre_summary: torch.Tensor,
    page_features: torch.Tensor,
    user_features: torch.Tensor,
    variant_token_ids: torch.Tensor,
    eval_batch_size: int,
    device: torch.device,
    pessimism_beta: float,
) -> Dict[str, torch.Tensor]:
    q_means: List[torch.Tensor] = []
    q_stds: List[torch.Tensor] = []
    total = int(variant_token_ids.shape[0])
    for start in range(0, total, int(eval_batch_size)):
        chunk_tokens = variant_token_ids[start:start + int(eval_batch_size)].to(device)
        chunk_items = (chunk_tokens > 0).any(dim=-1)
        chunk_pre = pre_summary.expand(chunk_tokens.shape[0], -1)
        chunk_page = page_features.expand(chunk_tokens.shape[0], -1)
        chunk_user = user_features.expand(chunk_tokens.shape[0], -1)
        outputs = critic(
            pre_summary=chunk_pre,
            token_ids=chunk_tokens,
            item_mask=chunk_items,
            page_features=chunk_page,
            user_features=chunk_user,
        )
        q_means.append(outputs.get("q_mean", outputs["q_value"]).detach().cpu())
        q_stds.append(outputs.get("q_std", torch.zeros_like(outputs["q_value"])).detach().cpu())
    q_mean = torch.cat(q_means, dim=0)
    q_std = torch.cat(q_stds, dim=0)
    q_pess = q_mean - float(pessimism_beta) * q_std
    return {"q_mean": q_mean, "q_std": q_std, "q_pess": q_pess}


def evaluate_candidate_support(
    tiger,
    *,
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_tokens: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    target_tokens = candidate_tokens.to(device)
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)
    enc_hidden = encoder_hidden.expand(target_tokens.shape[0], -1, -1)
    enc_attn = attention_mask.expand(target_tokens.shape[0], -1)
    logits, _hidden = tiger.decode_with_hidden_from_encoded(
        encoder_hidden=enc_hidden,
        attention_mask=enc_attn,
        decoder_input_ids=decoder_input_ids,
    )
    token_log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = token_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    valid_mask = (target_tokens > 0).float()
    seq_logprob_sum = (target_log_probs * valid_mask).sum(dim=-1)
    seq_logprob_mean = seq_logprob_sum / valid_mask.sum(dim=-1).clamp_min(1.0)
    return {
        "support_logprob_sum": seq_logprob_sum.detach().cpu(),
        "support_logprob_mean": seq_logprob_mean.detach().cpu(),
    }


def build_candidate_variants(
    base_page_tokens: torch.Tensor,
    candidate_tokens: Sequence[int],
    slot_index: int,
) -> Tuple[torch.Tensor, int]:
    sid_depth = int(base_page_tokens.shape[1])
    candidate = torch.tensor([int(x) for x in candidate_tokens], dtype=torch.long)
    valid_len = int((candidate > 0).sum().item())
    variants = []
    full_variant = base_page_tokens.clone()
    full_variant[int(slot_index)] = candidate
    variants.append(full_variant)
    for sid_idx in range(valid_len):
        prefix_variant = base_page_tokens.clone()
        prefix_variant[int(slot_index)].zero_()
        prefix_variant[int(slot_index), : sid_idx + 1] = candidate[: sid_idx + 1]
        variants.append(prefix_variant)
    return torch.stack(variants, dim=0), valid_len


def build_batched_candidate_variants(
    candidates: Sequence[Dict[str, Any]],
    *,
    base_page_tokens: torch.Tensor,
    slot_index: int,
) -> Tuple[torch.Tensor | None, Dict[int, Dict[str, int]]]:
    variant_batches: List[torch.Tensor] = []
    variant_meta: Dict[int, Dict[str, int]] = {}
    offset = 0
    for cand_idx, candidate in enumerate(candidates):
        if str(candidate.get("source")) == "behavior":
            continue
        variants, valid_len = build_candidate_variants(
            base_page_tokens=base_page_tokens,
            candidate_tokens=candidate["tokens"],
            slot_index=int(slot_index),
        )
        count = int(variants.shape[0])
        variant_batches.append(variants)
        variant_meta[int(cand_idx)] = {
            "start": int(offset),
            "count": int(count),
            "valid_len": int(valid_len),
        }
        offset += count
    if not variant_batches:
        return None, variant_meta
    return torch.cat(variant_batches, dim=0), variant_meta


def transform_reward(
    reward_value: float,
    behavior_value: float,
    *,
    transform: str,
    margin_clip: float,
    margin_temperature: float,
) -> Tuple[float, float]:
    margin = float(reward_value - behavior_value)
    mode = str(transform).lower()
    if mode == "raw":
        return float(reward_value), margin
    if mode == "centered":
        return margin, margin
    if mode == "clipped_margin":
        clip_value = abs(float(margin_clip))
        if clip_value <= 0.0:
            clip_value = 1.0
        return float(np.clip(margin, -clip_value, clip_value)), margin
    if mode == "tanh_margin":
        scale = abs(float(margin_clip))
        if scale <= 0.0:
            scale = 1.0
        temperature = max(abs(float(margin_temperature)), 1e-6)
        return float(scale * np.tanh(margin / temperature)), margin
    raise ValueError(f"Unsupported reward_transform: {transform}")


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(str(args.device))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chain_rows = load_chain_rows(Path(args.chain_path), int(args.max_rows))
    max_page_index = max(int(row.get("page_index", 1)) for row in chain_rows)
    max_slate_size = max(len(row.get("selected_item_ids", [])) for row in chain_rows)

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    user_feature_keys, user_feature_sizes = get_user_feature_layout(reader)
    user_feature_cache: Dict[int, List[float]] = {}
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, sid2iid_map_tok = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    sid_prefix_to_next = build_sid_prefix_to_next(sid2iid_map_tok)

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, _codebook_size = load_tiger_model(
        tiger_ckpt=str(args.tiger_ckpt),
        sid_mapping_path=str(args.sid_mapping_path),
        num_layers=int(size_cfg["num_layers"]),
        num_decoder_layers=int(size_cfg["num_decoder_layers"]),
        d_model=int(size_cfg["d_model"]),
        d_ff=int(size_cfg["d_ff"]),
        num_heads=int(size_cfg["num_heads"]),
        d_kv=int(size_cfg["d_kv"]),
        dropout_rate=0.1,
        feed_forward_proj="relu",
        device=device,
    )
    for param in tiger.parameters():
        param.requires_grad = False
    tiger.eval()

    critic, critic_meta = load_page_sid_qcritic_bundle(
        str(args.critic_bundle_path),
        str(args.critic_meta_path),
        device,
    )
    critic.eval()

    n_groups = 0
    n_rows = 0
    group_sizes: List[int] = []
    raw_rewards: List[float] = []
    norm_rewards: List[float] = []
    item_advs: List[float] = []
    token_advs: List[float] = []
    behavior_rows = 0
    behavior_source_rewards: List[float] = []
    beam_source_rewards: List[float] = []
    behavior_reward_raws: List[float] = []
    beam_reward_raws: List[float] = []
    best_source_margins: List[float] = []
    best_reward_margins: List[float] = []
    top_behavior_source = 0
    top_behavior_reward = 0
    support_logprob_means: List[float] = []
    support_gap_scaled_means: List[float] = []
    adaptive_beta_values: List[float] = []
    with output_path.open("w", encoding="utf-8") as out:
        for row in chain_rows:
            history_items = [int(x) for x in row.get("history_items", [])][-int(args.max_hist_items):]
            selected_item_ids = [int(x) for x in row.get("selected_item_ids", [])]
            user_id = int(row.get("user_id", -1))
            slot_index = int(row.get("slate_item_index", 0))
            if not selected_item_ids or slot_index < 0 or slot_index >= len(selected_item_ids):
                continue
            if any(int(item_id) >= int(iid2sid_tok_cpu.shape[0]) or int(item_id) < 0 for item_id in selected_item_ids):
                continue

            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids_cpu, attention_mask_cpu = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(args.max_hist_items),
                int(sid_depth),
            )
            input_ids = input_ids_cpu.to(device)
            attention_mask = attention_mask_cpu.to(device)
            with torch.inference_mode():
                encoder_hidden = tiger.encode(input_ids, attention_mask)
                pre_summary = pooled_history_summary(tiger, input_ids, attention_mask)
            if int(user_id) not in user_feature_cache:
                user_feature_cache[int(user_id)] = build_user_feature_vector(
                    reader,
                    int(user_id),
                    user_feature_keys=user_feature_keys,
                    user_feature_sizes=user_feature_sizes,
                ).tolist()
            user_features_tensor = torch.tensor(
                user_feature_cache[int(user_id)],
                dtype=torch.float32,
                device=device,
            ).view(1, -1)

            history_set = set(history_items)
            page_item_set = set(selected_item_ids)
            behavior_item_id = int(row.get("selected_item_id", selected_item_ids[slot_index]))
            with torch.inference_mode():
                ranked = decode_ranked_sequences(
                    tiger,
                    encoder_hidden=encoder_hidden,
                    attention_mask=attention_mask,
                    sid_prefix_to_next=sid_prefix_to_next,
                    sid_depth=int(sid_depth),
                    beam_width=max(int(args.beam_width), int(args.group_size) * 2),
                    device=device,
                )

            candidate_pool: List[Dict[str, Any]] = []
            seen_items = set()
            behavior_tokens = [int(x) for x in row.get("selected_sid_tokens", [])]
            if len(behavior_tokens) == int(sid_depth):
                candidate_pool.append(
                    {
                        "item_id": int(behavior_item_id),
                        "tokens": list(behavior_tokens),
                        "source": "behavior",
                        "rank": -1,
                        "page_q_value": float(row.get("page_q_value", 0.0)),
                        "page_q_mean": float(row.get("page_q_mean", row.get("page_q_value", 0.0))),
                        "page_q_std": float(row.get("page_q_std", 0.0)),
                        "page_q_pess": float(row.get("page_q_pess", row.get("page_q_value", 0.0))),
                        "item_advantage": float(row.get("item_advantage", 0.0)),
                        "item_advantage_mean": float(row.get("item_advantage_mean", row.get("item_advantage", 0.0))),
                        "item_advantage_pess": float(row.get("item_advantage_pess", row.get("item_advantage", 0.0))),
                        "sid_advantage": [float(x) for x in row.get("sid_advantage", [])],
                        "sid_advantage_pess": [float(x) for x in row.get("sid_advantage_pess", row.get("sid_advantage", []))],
                    }
                )
                seen_items.add(int(behavior_item_id))

            for rank, (sid_seq, score) in enumerate(ranked):
                key = tuple(int(x) for x in sid_seq)
                item_id = sid2iid_map_tok.get(key)
                if item_id is None:
                    continue
                if int(item_id) in history_set:
                    continue
                if int(item_id) in page_item_set and int(item_id) != int(behavior_item_id):
                    continue
                if int(item_id) in seen_items:
                    continue
                candidate_pool.append(
                    {
                        "item_id": int(item_id),
                        "tokens": [int(x) for x in sid_seq],
                        "source": "beam",
                        "rank": int(rank),
                        "beam_score": float(score),
                    }
                )
                seen_items.add(int(item_id))
                if len(candidate_pool) >= int(args.group_size):
                    break

            if len(candidate_pool) < 2:
                continue

            base_page_tokens = iid2sid_tok_cpu[torch.tensor(selected_item_ids, dtype=torch.long)].clone()
            page_features = build_page_scalar_features(
                page_index=int(row.get("page_index", 1)),
                max_page_index=int(max_page_index),
                history_len=len(history_items),
                max_hist_items=int(args.max_hist_items),
                slate_size=len(selected_item_ids),
                max_slate_size=int(max_slate_size),
            )
            page_features_tensor = torch.tensor(page_features, dtype=torch.float32, device=device).view(1, -1)
            q_without_item = float(row.get("page_q_without_item", 0.0))
            q_without_item_mean = float(row.get("page_q_without_item_mean", q_without_item))
            q_without_item_pess = float(row.get("page_q_without_item_pess", q_without_item))

            group_candidates = candidate_pool[: int(args.group_size)]
            candidate_tokens_tensor = torch.tensor(
                [[int(x) for x in candidate["tokens"]] for candidate in group_candidates],
                dtype=torch.long,
            )
            with torch.inference_mode():
                support_outputs = evaluate_candidate_support(
                    tiger,
                    encoder_hidden=encoder_hidden,
                    attention_mask=attention_mask,
                    candidate_tokens=candidate_tokens_tensor,
                    device=device,
                )
            batched_variants, variant_meta = build_batched_candidate_variants(
                group_candidates,
                base_page_tokens=base_page_tokens,
                slot_index=int(slot_index),
            )
            batched_q_outputs: Dict[str, torch.Tensor] | None = None
            if batched_variants is not None:
                with torch.inference_mode():
                    batched_q_outputs = evaluate_q_variants(
                        critic,
                        pre_summary=pre_summary,
                        page_features=page_features_tensor,
                        user_features=user_features_tensor,
                        variant_token_ids=batched_variants,
                        eval_batch_size=int(args.critic_eval_batch_size),
                        device=device,
                        pessimism_beta=float(args.critic_pessimism_beta),
                    )

            group_records: List[Dict[str, Any]] = []
            input_ids_list = input_ids_cpu.squeeze(0).tolist()
            attention_mask_list = attention_mask_cpu.squeeze(0).tolist()
            for cand_idx, candidate in enumerate(group_candidates):
                support_logprob_sum = float(support_outputs["support_logprob_sum"][cand_idx].item())
                support_logprob_mean = float(support_outputs["support_logprob_mean"][cand_idx].item())
                if str(candidate.get("source")) == "behavior":
                    full_q = float(candidate.get("page_q_mean", candidate.get("page_q_value", 0.0)))
                    full_q_std = float(candidate.get("page_q_std", 0.0))
                    full_q_pess = float(candidate.get("page_q_pess", full_q))
                    item_adv = float(candidate.get("item_advantage_mean", candidate.get("item_advantage", 0.0)))
                    item_adv_pess = float(candidate.get("item_advantage_pess", item_adv))
                    sid_adv = [float(x) for x in candidate.get("sid_advantage", [])]
                    sid_adv_pess = [float(x) for x in candidate.get("sid_advantage_pess", sid_adv)]
                else:
                    if batched_q_outputs is None:
                        raise RuntimeError("Missing batched critic outputs for non-behavior candidate.")
                    meta = variant_meta.get(int(cand_idx))
                    if meta is None:
                        raise RuntimeError(f"Missing variant metadata for candidate index {cand_idx}.")
                    start = int(meta["start"])
                    end = start + int(meta["count"])
                    valid_len = int(meta["valid_len"])
                    q_mean_values = batched_q_outputs["q_mean"][start:end]
                    q_std_values = batched_q_outputs["q_std"][start:end]
                    q_pess_values = batched_q_outputs["q_pess"][start:end]
                    full_q = float(q_mean_values[0].item())
                    full_q_std = float(q_std_values[0].item())
                    full_q_pess = float(q_pess_values[0].item())
                    item_adv = float(full_q - q_without_item_mean)
                    item_adv_pess = float(full_q_pess - q_without_item_pess)
                    sid_adv = [0.0 for _ in range(int(sid_depth))]
                    sid_adv_pess = [0.0 for _ in range(int(sid_depth))]
                    prev_q = float(q_without_item)
                    prev_q_pess = float(q_without_item_pess)
                    for sid_idx in range(valid_len):
                        prefix_q = float(q_mean_values[1 + sid_idx].item())
                        prefix_q_pess = float(q_pess_values[1 + sid_idx].item())
                        sid_adv[sid_idx] = float(prefix_q - prev_q)
                        sid_adv_pess[sid_idx] = float(prefix_q_pess - prev_q_pess)
                        prev_q = prefix_q
                        prev_q_pess = prefix_q_pess

                group_records.append(
                    {
                        "group_id": f"{int(row.get('episode_id', -1))}:{int(row.get('page_index', -1))}:{int(slot_index)}",
                        "episode_id": int(row.get("episode_id", -1)),
                        "user_id": int(row.get("user_id", -1)),
                        "page_index": int(row.get("page_index", -1)),
                        "slot_index": int(slot_index),
                        "history_items": list(history_items),
                        "input_ids": input_ids_list,
                        "attention_mask": attention_mask_list,
                        "selected_item_ids": list(selected_item_ids),
                        "target_tokens": [int(x) for x in candidate["tokens"]],
                        "candidate_item_id": int(candidate.get("item_id", -1)),
                        "candidate_source": str(candidate.get("source", "beam")),
                        "candidate_rank": int(candidate.get("rank", cand_idx)),
                        "is_behavior": bool(str(candidate.get("source")) == "behavior"),
                        "page_q_value": float(full_q),
                        "page_q_mean": float(full_q),
                        "page_q_std": float(full_q_std),
                        "page_q_pess": float(full_q_pess),
                        "page_q_without_item": float(q_without_item_mean),
                        "page_q_without_item_mean": float(q_without_item_mean),
                        "page_q_without_item_pess": float(q_without_item_pess),
                        "item_advantage": float(item_adv),
                        "item_advantage_mean": float(item_adv),
                        "item_advantage_pess": float(item_adv_pess),
                        "sid_advantage": [float(x) for x in sid_adv],
                        "sid_advantage_pess": [float(x) for x in sid_adv_pess],
                        "support_logprob_sum": float(support_logprob_sum),
                        "support_logprob_mean": float(support_logprob_mean),
                    }
                )

            behavior_record = next((record for record in group_records if bool(record.get("is_behavior"))), None)
            if behavior_record is None:
                continue

            behavior_support = float(behavior_record.get("support_logprob_mean", 0.0))
            group_q_std_mean = max(
                float(np.mean([float(record["page_q_std"]) for record in group_records])),
                1e-6,
            )
            support_gap_temperature = max(abs(float(args.support_gap_temperature)), 1e-6)
            support_gap_clip = abs(float(args.support_gap_clip))
            for record in group_records:
                support_gap = max(0.0, behavior_support - float(record.get("support_logprob_mean", 0.0)))
                support_gap_scaled = float(support_gap / support_gap_temperature)
                if support_gap_clip > 0.0:
                    support_gap_scaled = float(np.clip(support_gap_scaled, 0.0, support_gap_clip))
                uncertainty_ratio = float(record["page_q_std"]) / group_q_std_mean
                adaptive_beta = (
                    float(args.critic_pessimism_beta)
                    + float(args.adaptive_beta_unc_scale) * uncertainty_ratio
                    + float(args.adaptive_beta_support_scale) * support_gap_scaled
                )
                adaptive_support_pess = (
                    float(record["page_q_mean"])
                    - adaptive_beta * float(record["page_q_std"])
                    - float(args.support_penalty_scale) * support_gap_scaled
                )
                record["support_gap_vs_behavior"] = float(support_gap)
                record["support_gap_scaled"] = float(support_gap_scaled)
                record["uncertainty_ratio"] = float(uncertainty_ratio)
                record["adaptive_beta"] = float(adaptive_beta)
                record["adaptive_support_pess"] = float(adaptive_support_pess)
                record["reward_model_value"] = float(
                    {
                        "page_q_value": float(record["page_q_value"]),
                        "page_q_mean": float(record["page_q_mean"]),
                        "page_q_pess": float(record["page_q_pess"]),
                        "item_advantage": float(record["item_advantage"]),
                        "item_advantage_mean": float(record["item_advantage_mean"]),
                        "item_advantage_pess": float(record["item_advantage_pess"]),
                        "adaptive_support_pess": float(record["adaptive_support_pess"]),
                    }[str(args.reward_field)]
                )

            behavior_source_value = float(behavior_record["reward_model_value"])
            behavior_reward_raw = float("nan")
            for record in group_records:
                reward_raw, reward_margin = transform_reward(
                    float(record["reward_model_value"]),
                    behavior_source_value,
                    transform=str(args.reward_transform),
                    margin_clip=float(args.reward_margin_clip),
                    margin_temperature=float(args.reward_margin_temperature),
                )
                record["reward_behavior_value"] = float(behavior_source_value)
                record["reward_margin_vs_behavior"] = float(reward_margin)
                record["reward_raw"] = float(reward_raw)
                if bool(record.get("is_behavior")):
                    behavior_reward_raw = float(reward_raw)

            if not np.isfinite(behavior_reward_raw):
                continue

            rewards = np.asarray([float(record["reward_raw"]) for record in group_records], dtype=np.float32)
            mean = float(rewards.mean())
            std = float(rewards.std())
            denom = max(std, 1e-6)
            best_source_record = max(group_records, key=lambda item: float(item["reward_model_value"]))
            best_reward_record = max(group_records, key=lambda item: float(item["reward_raw"]))
            top_behavior_source += int(bool(best_source_record.get("is_behavior")))
            top_behavior_reward += int(bool(best_reward_record.get("is_behavior")))
            best_source_margins.append(float(best_source_record["reward_model_value"]) - float(behavior_source_value))
            best_reward_margins.append(float(best_reward_record["reward_raw"]) - float(behavior_reward_raw))
            for record in group_records:
                record["reward_mean"] = float(mean)
                record["reward_std"] = float(std)
                record["group_advantage"] = float((float(record["reward_raw"]) - mean) / denom)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_rows += 1
                raw_rewards.append(float(record["reward_raw"]))
                norm_rewards.append(float(record["group_advantage"]))
                item_advs.append(float(record["item_advantage"]))
                token_advs.extend(float(x) for x in record["sid_advantage"])
                support_logprob_means.append(float(record.get("support_logprob_mean", 0.0)))
                support_gap_scaled_means.append(float(record.get("support_gap_scaled", 0.0)))
                adaptive_beta_values.append(float(record.get("adaptive_beta", float(args.critic_pessimism_beta))))
                behavior_rows += int(bool(record["is_behavior"]))
                if bool(record["is_behavior"]):
                    behavior_source_rewards.append(float(record["reward_model_value"]))
                    behavior_reward_raws.append(float(record["reward_raw"]))
                else:
                    beam_source_rewards.append(float(record["reward_model_value"]))
                    beam_reward_raws.append(float(record["reward_raw"]))
            n_groups += 1
            group_sizes.append(len(group_records))

    summary = {
        "method": "TIGER-HCA-GRPO Group Builder",
        "chain_path": str(Path(args.chain_path).resolve()),
        "critic_bundle_path": str(Path(args.critic_bundle_path).resolve()),
        "critic_meta_path": str(Path(args.critic_meta_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "output_path": str(output_path.resolve()),
        "sid_depth": int(sid_depth),
        "critic_hidden_size": int(critic_meta["hidden_size"]),
        "critic_ensemble_size": int(critic_meta.get("ensemble_size", 1)),
        "critic_pessimism_beta": float(args.critic_pessimism_beta),
        "reward_field": str(args.reward_field),
        "support_penalty_scale": float(args.support_penalty_scale),
        "support_gap_temperature": float(args.support_gap_temperature),
        "support_gap_clip": float(args.support_gap_clip),
        "adaptive_beta_unc_scale": float(args.adaptive_beta_unc_scale),
        "adaptive_beta_support_scale": float(args.adaptive_beta_support_scale),
        "reward_transform": str(args.reward_transform),
        "reward_margin_clip": float(args.reward_margin_clip),
        "reward_margin_temperature": float(args.reward_margin_temperature),
        "n_input_rows": int(len(chain_rows)),
        "n_groups": int(n_groups),
        "n_rows": int(n_rows),
        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "behavior_ratio": float(behavior_rows / max(n_rows, 1)),
        "reward_raw_mean": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
        "reward_raw_std": float(np.std(raw_rewards)) if raw_rewards else 0.0,
        "group_adv_abs_mean": float(np.mean(np.abs(norm_rewards))) if norm_rewards else 0.0,
        "item_adv_abs_mean": float(np.mean(np.abs(item_advs))) if item_advs else 0.0,
        "sid_adv_abs_mean": float(np.mean(np.abs(token_advs))) if token_advs else 0.0,
        "support_logprob_mean": float(np.mean(support_logprob_means)) if support_logprob_means else 0.0,
        "support_gap_scaled_mean": float(np.mean(support_gap_scaled_means)) if support_gap_scaled_means else 0.0,
        "adaptive_beta_mean": float(np.mean(adaptive_beta_values)) if adaptive_beta_values else float(args.critic_pessimism_beta),
        "behavior_source_reward_mean": float(np.mean(behavior_source_rewards)) if behavior_source_rewards else 0.0,
        "beam_source_reward_mean": float(np.mean(beam_source_rewards)) if beam_source_rewards else 0.0,
        "behavior_reward_mean": float(np.mean(behavior_reward_raws)) if behavior_reward_raws else 0.0,
        "beam_reward_mean": float(np.mean(beam_reward_raws)) if beam_reward_raws else 0.0,
        "beam_source_minus_behavior": (
            float(np.mean(beam_source_rewards) - np.mean(behavior_source_rewards))
            if behavior_source_rewards and beam_source_rewards
            else 0.0
        ),
        "beam_reward_minus_behavior": (
            float(np.mean(beam_reward_raws) - np.mean(behavior_reward_raws))
            if behavior_reward_raws and beam_reward_raws
            else 0.0
        ),
        "avg_best_source_minus_behavior": float(np.mean(best_source_margins)) if best_source_margins else 0.0,
        "median_best_source_minus_behavior": float(np.median(best_source_margins)) if best_source_margins else 0.0,
        "avg_best_reward_minus_behavior": float(np.mean(best_reward_margins)) if best_reward_margins else 0.0,
        "median_best_reward_minus_behavior": float(np.median(best_reward_margins)) if best_reward_margins else 0.0,
        "pos_source_margin_frac": float(np.mean(np.asarray(best_source_margins) > 0.0)) if best_source_margins else 0.0,
        "pos_reward_margin_frac": float(np.mean(np.asarray(best_reward_margins) > 0.0)) if best_reward_margins else 0.0,
        "top_behavior_source_frac": float(top_behavior_source / max(n_groups, 1)),
        "top_beam_source_frac": float(1.0 - (top_behavior_source / max(n_groups, 1))) if n_groups > 0 else 0.0,
        "top_behavior_reward_frac": float(top_behavior_reward / max(n_groups, 1)),
        "top_beam_reward_frac": float(1.0 - (top_behavior_reward / max(n_groups, 1))) if n_groups > 0 else 0.0,
    }
    summary_out = Path(args.summary_out) if str(args.summary_out).strip() else output_path.with_name("hca_grpo_group_summary.json")
    write_json(summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
