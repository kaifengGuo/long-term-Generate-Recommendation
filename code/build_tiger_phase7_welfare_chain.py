import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from build_tiger_phase3_credit_chain import (
    build_transport_chain,
    load_reader_from_uirm_log,
    transform_episode_credits,
)
from tiger_phase2_blend_common import (
    build_history_tokens,
    build_iid2sid_tokens,
    infer_model_size_args,
    load_tiger_model,
)
from tiger_phase7_welfare_common import (
    aggregate_history_credit_features,
    build_welfare_page_features,
    compute_welfare_step_reward,
    load_welfare_head,
    pooled_history_summary,
    resolve_reward_weights,
)
from tiger_slate_allocator_common import build_slate_allocator_inputs, load_slate_allocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build welfare-aware TIGER chain with explicit page value estimates.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--welfare_head_path", type=str, default="")
    parser.add_argument("--welfare_meta_path", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_page_index", type=int, default=20)
    parser.add_argument("--max_slate_size", type=int, default=6)
    parser.add_argument("--credit_mode", type=str, default="centered", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0)
    parser.add_argument("--transport_epsilon", type=float, default=0.35)
    parser.add_argument("--transport_iter", type=int, default=16)
    parser.add_argument("--cf_topk_history", type=int, default=5)
    parser.add_argument("--cf_smooth", type=float, default=0.05)
    parser.add_argument("--reward_preset", type=str, default="")
    parser.add_argument("--reward_weights_json", type=str, default="")
    parser.add_argument("--credit_source", type=str, default="pred", choices=["pred", "target", "blend"])
    parser.add_argument("--credit_blend_alpha", type=float, default=0.5)
    parser.add_argument("--allocator_head_path", type=str, default="")
    parser.add_argument("--allocator_meta_path", type=str, default="")
    parser.add_argument("--allocator_device", type=str, default="cpu")
    parser.add_argument("--allocator_blend_alpha", type=float, default=0.7)
    parser.add_argument("--allocator_keep_topk", type=int, default=2)
    parser.add_argument("--heuristic_mix", type=float, default=0.60)
    parser.add_argument("--support_mix", type=float, default=0.25)
    parser.add_argument("--response_mix", type=float, default=0.15)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def load_trace_records(trace_path: Path) -> List[Dict[str, Any]]:
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
        raise ValueError(f"No usable trace records found in {trace_path}")
    return rows


def infer_welfare_values(
    rows: Sequence[Dict[str, Any]],
    *,
    reader,
    tiger,
    sid_mapping_path: str,
    welfare_head,
    device: torch.device,
    max_hist_items: int,
    max_page_index: int,
    max_slate_size: int,
) -> Tuple[List[float], List[List[float]], List[str]]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])

    values: List[float] = []
    feat_rows: List[List[float]] = []
    feature_names: List[str] = []
    if welfare_head is None:
        return values, feat_rows, feature_names
    for rec in rows:
        history_items = [int(x) for x in rec.get("history_items", [])][-int(max_hist_items):]
        hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
        input_ids, attention_mask = build_history_tokens(
            hist_tensor,
            iid2sid_tok_cpu,
            int(max_hist_items),
            int(sid_depth),
        )
        page_feat, feat_names = build_welfare_page_features(
            page_index=int(rec.get("page_index", 0)),
            max_page_index=int(max_page_index),
            history_len=len(history_items),
            max_hist_items=int(max_hist_items),
            slate_size=len(rec.get("selected_item_ids", [])),
            max_slate_size=int(max_slate_size),
            selected_item_rewards=rec.get("selected_item_rewards", []),
            selected_responses=rec.get("selected_responses", []),
        )
        feature_names = feat_names
        with torch.no_grad():
            summary = pooled_history_summary(tiger, input_ids.to(device), attention_mask.to(device))
            page_tensor = torch.tensor(page_feat, dtype=torch.float32, device=device).unsqueeze(0)
            pred = welfare_head(summary, page_tensor)
        values.append(float(pred.item()))
        feat_rows.append(page_feat.astype(np.float32).tolist())
    return values, feat_rows, feature_names


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))

    raw_records = load_trace_records(trace_path)
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(args.device))
    response_names = list(reader.get_statistics()["feedback_type"])

    preset = str(args.reward_preset).strip()
    reward_weights_json = str(args.reward_weights_json).strip()
    reward_weight_map: Dict[str, float] = {}
    if str(args.welfare_meta_path).strip():
        welfare_meta = json.loads(Path(args.welfare_meta_path).read_text(encoding="utf-8"))
        if not preset:
            preset = str(welfare_meta.get("reward_preset", "click_longview"))
        if not reward_weights_json and "reward_weights" in welfare_meta:
            reward_weights_json = json.dumps(welfare_meta["reward_weights"])
    reward_weights, reward_weight_map = resolve_reward_weights(
        response_names,
        preset=preset or "click_longview",
        reward_weights_json=reward_weights_json,
    )

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
    for p in tiger.parameters():
        p.requires_grad = False
    tiger.eval()

    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    token_vocab_size = int(iid2sid_tok_cpu.max().item()) + 1

    welfare_head = None
    welfare_meta = {}
    if str(args.welfare_head_path).strip():
        welfare_head, welfare_meta = load_welfare_head(str(args.welfare_head_path), str(args.welfare_meta_path), device)

    allocator = None
    allocator_meta: Dict[str, Any] | None = None
    allocator_device = torch.device(str(args.allocator_device))
    if str(args.allocator_head_path).strip() and str(args.allocator_meta_path).strip():
        allocator, allocator_meta = load_slate_allocator(
            str(args.allocator_head_path),
            str(args.allocator_meta_path),
            allocator_device,
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in raw_records:
        grouped.setdefault(str(rec["episode_id"]), []).append(rec)

    n_pages = 0
    n_items = 0
    with output_path.open("w", encoding="utf-8") as out:
        for episode_id, pages in grouped.items():
            pages = sorted(pages, key=lambda x: int(x.get("page_index", 0)))
            target_rewards = [
                float(compute_welfare_step_reward(x.get("selected_responses", []), reward_weights))
                for x in pages
            ]
            target_returns = [0.0 for _ in pages]
            running = 0.0
            for idx in range(len(pages) - 1, -1, -1):
                running = float(target_rewards[idx]) + float(args.gamma) * running
                target_returns[idx] = float(running)

            pred_values, feat_rows, feature_names = infer_welfare_values(
                pages,
                reader=reader,
                tiger=tiger,
                sid_mapping_path=str(args.sid_mapping_path),
                welfare_head=welfare_head,
                device=device,
                max_hist_items=int(args.max_hist_items),
                max_page_index=int(args.max_page_index),
                max_slate_size=int(args.max_slate_size),
            )
            if not pred_values:
                pred_values = list(target_returns)
                feat_rows = [[] for _ in pages]
                feature_names = []

            if str(args.credit_source) == "pred":
                credit_source_values = np.asarray(pred_values, dtype=np.float32)
            elif str(args.credit_source) == "target":
                credit_source_values = np.asarray(target_returns, dtype=np.float32)
            else:
                alpha = float(np.clip(float(args.credit_blend_alpha), 0.0, 1.0))
                credit_source_values = alpha * np.asarray(pred_values, dtype=np.float32) + (1.0 - alpha) * np.asarray(target_returns, dtype=np.float32)
            slate_credits = transform_episode_credits(credit_source_values.tolist(), str(args.credit_mode), float(args.credit_clip))
            session_credit = float(sum(target_rewards))

            for page_idx, rec in enumerate(pages):
                selected_item_ids = [int(x) for x in rec.get("selected_item_ids", [])]
                selected_sid_tokens_list = [[int(v) for v in seq] for seq in rec.get("selected_sid_tokens_list", [])]
                selected_responses = [[float(v) for v in resp] for resp in rec.get("selected_responses", [])]
                selected_item_rewards = [float(x) for x in rec.get("selected_item_rewards", [])]
                slate_size = len(selected_item_ids)
                if slate_size == 0:
                    continue
                if len(selected_sid_tokens_list) != slate_size:
                    selected_sid_tokens_list = []
                    for iid in selected_item_ids:
                        if 0 <= int(iid) < int(iid2sid_tok_cpu.shape[0]):
                            selected_sid_tokens_list.append([int(x) for x in iid2sid_tok_cpu[int(iid)].tolist()])
                        else:
                            selected_sid_tokens_list.append([])
                if len(selected_item_rewards) != slate_size:
                    selected_item_rewards = [0.0] * slate_size
                history_items = [int(x) for x in rec.get("history_items", [])][-int(args.max_hist_items):]
                slate_credit = float(slate_credits[page_idx])
                welfare_return_target = float(target_returns[page_idx])
                welfare_return_pred = float(pred_values[page_idx])
                credit_source_value = float(credit_source_values[page_idx])
                allocator_inputs = build_slate_allocator_inputs(
                    history_items=history_items,
                    selected_item_ids=selected_item_ids,
                    selected_sid_tokens_list=selected_sid_tokens_list,
                    selected_responses=selected_responses,
                    selected_item_rewards=selected_item_rewards,
                    response_weights=reward_weights.tolist(),
                    page_credit=float(slate_credit),
                    slate_return_raw=float(credit_source_value),
                    step_reward=float(target_rewards[page_idx]),
                    page_index=int(rec.get("page_index", page_idx + 1)),
                    iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                    max_hist_items=int(args.max_hist_items),
                    token_vocab_size=int(token_vocab_size),
                    heuristic_mix=float(args.heuristic_mix),
                    support_mix=float(args.support_mix),
                    response_mix=float(args.response_mix),
                )
                heuristic_shares = allocator_inputs["heuristic_shares"]
                centered_rewards = allocator_inputs["centered_rewards"]
                learned_shares = None
                if allocator is not None:
                    item_feat = torch.tensor(allocator_inputs["item_features"], dtype=torch.float32, device=allocator_device).unsqueeze(0)
                    page_feat = torch.tensor(allocator_inputs["page_features"], dtype=torch.float32, device=allocator_device).unsqueeze(0)
                    mask = torch.ones((1, slate_size), dtype=torch.bool, device=allocator_device)
                    with torch.no_grad():
                        pred = allocator.predict_shares(item_feat, page_feat, mask=mask)
                    learned_shares = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    alpha = float(np.clip(float(args.allocator_blend_alpha), 0.0, 1.0))
                    shares = alpha * learned_shares + (1.0 - alpha) * heuristic_shares
                    shares = shares / max(float(shares.sum()), 1e-8)
                else:
                    shares = heuristic_shares
                keep_topk = int(args.allocator_keep_topk)
                if keep_topk > 0 and keep_topk < int(len(shares)):
                    keep = np.argsort(-shares)[:keep_topk]
                    sparse = np.zeros_like(shares, dtype=np.float32)
                    sparse[keep] = shares[keep]
                    shares = sparse / max(float(sparse.sum()), 1e-8)

                n_pages += 1
                for item_pos, item_id in enumerate(selected_item_ids):
                    target_tokens = selected_sid_tokens_list[item_pos] if item_pos < len(selected_sid_tokens_list) else []
                    if len(target_tokens) != sid_depth or not any(int(x) > 0 for x in target_tokens):
                        continue
                    item_share = float(shares[item_pos]) if item_pos < len(shares) else 0.0
                    item_credit = float(slate_credit * item_share)
                    item_credit_raw = float(credit_source_value * item_share)
                    chain = build_transport_chain(
                        history_items=history_items,
                        target_tokens=target_tokens,
                        item_credit=item_credit,
                        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                        epsilon=float(args.transport_epsilon),
                        n_iter=int(args.transport_iter),
                        cf_topk_history=int(args.cf_topk_history),
                        cf_smooth=float(args.cf_smooth),
                    )
                    history_stats = aggregate_history_credit_features(chain["history_credit_calibrated"])
                    payload = {
                        "episode_id": int(episode_id),
                        "page_index": int(rec.get("page_index", page_idx + 1)),
                        "user_id": int(rec.get("user_id", -1)),
                        "slate_size": int(slate_size),
                        "session_credit": float(session_credit),
                        "welfare_reward_session": float(sum(target_rewards)),
                        "welfare_step_reward": float(target_rewards[page_idx]),
                        "welfare_return_target": float(welfare_return_target),
                        "welfare_return_pred": float(welfare_return_pred),
                        "welfare_credit_source_value": float(credit_source_value),
                        "slate_credit": float(slate_credit),
                        "slate_credit_raw": float(credit_source_value),
                        "selected_item_ids": [int(x) for x in selected_item_ids],
                        "selected_item_rewards": [float(x) for x in selected_item_rewards],
                        "selected_item_reward_centered": [float(x) for x in centered_rewards.tolist()],
                        "selected_item_credit_shares": [float(x) for x in shares.tolist()],
                        "selected_item_credit_shares_bootstrap": [float(x) for x in allocator_inputs["target_shares"].tolist()],
                        "selected_item_credit_shares_heuristic": [float(x) for x in heuristic_shares.tolist()],
                        "selected_item_credit_shares_model": [float(x) for x in learned_shares.tolist()] if learned_shares is not None else [],
                        "selected_item_support_strengths": [float(x) for x in allocator_inputs["support_strengths"].tolist()],
                        "selected_item_response_strengths": [float(x) for x in allocator_inputs["response_strengths"].tolist()],
                        "slate_item_index": int(item_pos),
                        "selected_item_id": int(item_id),
                        "selected_sid_tokens": list(target_tokens),
                        "selected_response": list(selected_responses[item_pos]) if item_pos < len(selected_responses) else [],
                        "history_items": list(history_items),
                        "item_credit": float(item_credit),
                        "item_credit_raw": float(item_credit_raw),
                        "item_reward": float(selected_item_rewards[item_pos]) if item_pos < len(selected_item_rewards) else 0.0,
                        "item_reward_centered": float(centered_rewards[item_pos]) if item_pos < len(centered_rewards) else 0.0,
                        "history_items_explicit": chain["history_items"],
                        "history_credit": chain["history_credit"],
                        "history_cf_drop": chain["history_cf_drop"],
                        "history_credit_calibrated": chain["history_credit_calibrated"],
                        "history_pos_mass": float(history_stats["history_pos_mass"]),
                        "history_neg_mass": float(history_stats["history_neg_mass"]),
                        "history_pos_ratio": float(history_stats["history_pos_ratio"]),
                        "history_neg_ratio": float(history_stats["history_neg_ratio"]),
                        "history_balance": float(history_stats["history_balance"]),
                        "block_credit": chain["block_credit"],
                        "block_credit_calibrated": chain["block_credit_calibrated"],
                        "token_credit": chain["token_credit"],
                        "token_credit_calibrated": chain["token_credit_calibrated"],
                        "base_support": float(chain["base_support"]),
                        "step_reward": float(rec.get("step_reward", 0.0)),
                        "credit_source": str(args.credit_source),
                        "welfare_feature_names": feature_names,
                        "welfare_page_features": feat_rows[page_idx],
                        "done": bool(rec.get("done", False)),
                    }
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    n_items += 1

    meta = {
        "trace_path": str(trace_path.resolve()),
        "output_path": str(output_path.resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "welfare_head_path": str(Path(args.welfare_head_path).resolve()) if str(args.welfare_head_path).strip() else "",
        "welfare_meta_path": str(Path(args.welfare_meta_path).resolve()) if str(args.welfare_meta_path).strip() else "",
        "credit_source": str(args.credit_source),
        "credit_blend_alpha": float(args.credit_blend_alpha),
        "credit_mode": str(args.credit_mode),
        "gamma": float(args.gamma),
        "reward_weights": reward_weight_map,
        "response_names": response_names,
        "n_trace_records": int(len(raw_records)),
        "n_episodes": int(len(grouped)),
        "n_pages": int(n_pages),
        "n_item_rows": int(n_items),
        "sid_depth": int(sid_depth),
        "allocator_mode": "learned" if allocator is not None else "heuristic",
        "allocator_head_path": str(Path(args.allocator_head_path).resolve()) if str(args.allocator_head_path).strip() else "",
        "allocator_meta_path": str(Path(args.allocator_meta_path).resolve()) if str(args.allocator_meta_path).strip() else "",
        "allocator_blend_alpha": float(args.allocator_blend_alpha),
        "allocator_keep_topk": int(args.allocator_keep_topk),
        "heuristic_mix": float(args.heuristic_mix),
        "support_mix": float(args.support_mix),
        "response_mix": float(args.response_mix),
        "welfare_meta": welfare_meta,
    }
    if allocator_meta is not None:
        meta["allocator_meta"] = allocator_meta
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase7-chain] saved welfare chain to {output_path}")
    print(f"[phase7-chain] meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
