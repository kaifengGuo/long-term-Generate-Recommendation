import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from reader import *  # noqa: F401,F403

from tiger_phase2_blend_common import build_iid2sid_tokens, sinkhorn_transport


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
    parser = argparse.ArgumentParser(description="Build explicit Phase3 attribution chain for TIGER traces.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--credit_mode", type=str, default="return", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0)
    parser.add_argument("--transport_epsilon", type=float, default=0.35)
    parser.add_argument("--transport_iter", type=int, default=16)
    parser.add_argument("--cf_topk_history", type=int, default=5)
    parser.add_argument("--cf_smooth", type=float, default=0.05)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def transform_episode_credits(raw_returns: Sequence[float], mode: str, clip: float) -> List[float]:
    vals = np.asarray(list(raw_returns), dtype=np.float32)
    if vals.size == 0:
        return []
    mode_name = str(mode).lower()
    if mode_name == "return":
        out = vals.copy()
    elif mode_name == "centered":
        out = vals - float(vals.mean())
    elif mode_name == "zscore":
        mean = float(vals.mean())
        std = float(vals.std())
        out = vals - mean if std < 1e-6 else (vals - mean) / std
    else:
        raise ValueError(f"Unsupported credit_mode: {mode}")
    if float(clip) > 0.0:
        out = np.clip(out, -float(clip), float(clip))
    return out.astype(np.float32).tolist()


def load_trace_records(trace_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "selected_item_id" not in payload:
                continue
            records.append(payload)
    if not records:
        raise ValueError(f"No trace records found in {trace_path}")
    return records


def longest_prefix_ratio(hist_sid: np.ndarray, target_sid: np.ndarray, end_idx: int) -> float:
    width = int(end_idx) + 1
    if width <= 0:
        return 0.0
    match = float(np.mean(hist_sid[:width] == target_sid[:width]))
    return float(np.clip(match, 0.0, 1.0))


def aggregate_support(affinity: np.ndarray) -> float:
    if affinity.size == 0:
        return 0.0
    return float(np.mean(np.max(affinity, axis=0)))


def _normalize(scores: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
    scores = np.maximum(scores, 0.0)
    if float(scores.sum()) <= 1e-8:
        scores = np.maximum(fallback, 0.0)
    if float(scores.sum()) <= 1e-8:
        scores = np.ones_like(scores, dtype=np.float32)
    return scores / max(float(scores.sum()), 1e-8)


def build_transport_chain(
    history_items: Sequence[int],
    target_tokens: Sequence[int],
    item_credit: float,
    iid2sid_tok_cpu: torch.Tensor,
    epsilon: float,
    n_iter: int,
    cf_topk_history: int,
    cf_smooth: float,
) -> Dict[str, Any]:
    target = np.asarray(target_tokens, dtype=np.int64).reshape(-1)
    depth = int(target.shape[0])
    hist = [int(i) for i in history_items if 0 <= int(i) < int(iid2sid_tok_cpu.shape[0])]
    hist = hist[-50:]
    if depth <= 0:
        return {
            "history_items": [],
            "history_credit": [],
            "history_cf_drop": [],
            "history_credit_calibrated": [],
            "block_credit": [],
            "block_credit_calibrated": [],
            "token_credit": [],
            "token_credit_calibrated": [],
            "base_support": 0.0,
            "slate_credit": float(item_credit),
        }
    if not hist:
        uniform = np.ones(depth, dtype=np.float32) * (float(item_credit) / max(depth, 1))
        return {
            "history_items": [],
            "history_credit": [],
            "history_cf_drop": [],
            "history_credit_calibrated": [],
            "block_credit": uniform.tolist(),
            "block_credit_calibrated": uniform.tolist(),
            "token_credit": uniform.tolist(),
            "token_credit_calibrated": uniform.tolist(),
            "base_support": 0.0,
            "slate_credit": float(item_credit),
        }

    hist_tensor = torch.tensor(hist, dtype=torch.long)
    hist_sid = iid2sid_tok_cpu[hist_tensor].numpy()
    valid_mask = np.any(hist_sid > 0, axis=1)
    hist_sid = hist_sid[valid_mask]
    hist = [hist[i] for i in range(len(hist)) if bool(valid_mask[i])]
    if hist_sid.size == 0:
        uniform = np.ones(depth, dtype=np.float32) * (float(item_credit) / max(depth, 1))
        return {
            "history_items": [],
            "history_credit": [],
            "history_cf_drop": [],
            "history_credit_calibrated": [],
            "block_credit": uniform.tolist(),
            "block_credit_calibrated": uniform.tolist(),
            "token_credit": uniform.tolist(),
            "token_credit_calibrated": uniform.tolist(),
            "base_support": 0.0,
            "slate_credit": float(item_credit),
        }

    n_hist = int(hist_sid.shape[0])
    affinity = np.zeros((n_hist, depth), dtype=np.float32)
    for r in range(n_hist):
        for c in range(depth):
            affinity[r, c] = longest_prefix_ratio(hist_sid[r], target, c)

    row_mass = np.linspace(1.0, 2.0, n_hist, dtype=np.float32)
    col_mass = np.ones(depth, dtype=np.float32)
    pos_cost = 1.0 - affinity
    neg_cost = affinity
    pos_plan = sinkhorn_transport(row_mass, col_mass, pos_cost, epsilon=float(epsilon), n_iter=int(n_iter))
    neg_plan = sinkhorn_transport(row_mass, col_mass, neg_cost, epsilon=float(epsilon), n_iter=int(n_iter))
    pos_mass = max(float(item_credit), 0.0)
    neg_mass = max(-float(item_credit), 0.0)
    signed_plan = pos_mass * pos_plan - neg_mass * neg_plan
    history_credit = signed_plan.sum(axis=1)
    block_credit = signed_plan.sum(axis=0)
    residual = float(item_credit) - float(block_credit.sum())
    if depth > 0:
        block_credit[-1] += residual
        signed_plan[:, -1] += residual / max(float(n_hist), 1.0)

    base_support = aggregate_support(affinity)
    cf_drop = np.zeros((n_hist,), dtype=np.float32)
    order = np.argsort(-np.abs(history_credit))
    for ridx in order[: min(int(cf_topk_history), int(n_hist))]:
        if n_hist <= 1:
            cf_drop[int(ridx)] = 0.0
            continue
        keep_mask = np.ones((n_hist,), dtype=bool)
        keep_mask[int(ridx)] = False
        aff_wo = affinity[keep_mask]
        cf_drop[int(ridx)] = float(base_support - aggregate_support(aff_wo))

    if pos_mass > 0.0:
        base_row = pos_plan.sum(axis=1)
        cf_score = np.abs(cf_drop) + float(cf_smooth)
        row_mass_cal = _normalize(base_row * cf_score, base_row)
        row_cond = pos_plan / np.maximum(pos_plan.sum(axis=1, keepdims=True), 1e-8)
        plan_cal = row_mass_cal[:, None] * row_cond
        block_credit_cal = pos_mass * plan_cal.sum(axis=0)
        history_credit_cal = pos_mass * row_mass_cal
    elif neg_mass > 0.0:
        base_row = neg_plan.sum(axis=1)
        cf_score = np.abs(cf_drop) + float(cf_smooth)
        row_mass_cal = _normalize(base_row * cf_score, base_row)
        row_cond = neg_plan / np.maximum(neg_plan.sum(axis=1, keepdims=True), 1e-8)
        plan_cal = row_mass_cal[:, None] * row_cond
        block_credit_cal = -neg_mass * plan_cal.sum(axis=0)
        history_credit_cal = -neg_mass * row_mass_cal
    else:
        block_credit_cal = np.zeros((depth,), dtype=np.float32)
        history_credit_cal = np.zeros((n_hist,), dtype=np.float32)
    if depth > 0:
        residual_cal = float(item_credit) - float(block_credit_cal.sum())
        block_credit_cal[-1] += residual_cal

    return {
        "history_items": [int(x) for x in hist],
        "history_credit": history_credit.astype(np.float32).tolist(),
        "history_cf_drop": cf_drop.astype(np.float32).tolist(),
        "history_credit_calibrated": history_credit_cal.astype(np.float32).tolist(),
        "block_credit": block_credit.astype(np.float32).tolist(),
        "block_credit_calibrated": block_credit_cal.astype(np.float32).tolist(),
        "token_credit": block_credit.astype(np.float32).tolist(),
        "token_credit_calibrated": block_credit_cal.astype(np.float32).tolist(),
        "base_support": float(base_support),
        "slate_credit": float(item_credit),
    }


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_records = load_trace_records(trace_path)
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(args.device))
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in raw_records:
        grouped.setdefault(str(rec["episode_id"]), []).append(rec)

    with output_path.open("w", encoding="utf-8") as out:
        for episode_id, pages in grouped.items():
            pages = sorted(pages, key=lambda x: int(x["page_index"]))
            rewards = [float(x.get("step_reward", 0.0)) for x in pages]
            returns = [0.0 for _ in pages]
            running = 0.0
            for idx in range(len(pages) - 1, -1, -1):
                running = rewards[idx] + float(args.gamma) * running
                returns[idx] = running
            credits = transform_episode_credits(returns, str(args.credit_mode), float(args.credit_clip))
            session_credit = float(sum(rewards))
            for idx, rec in enumerate(pages):
                target_item = int(rec.get("selected_item_id", -1))
                target_tokens = [int(x) for x in rec.get("selected_sid_tokens", [])]
                history_items = [int(x) for x in rec.get("history_items", [])][-int(args.max_hist_items):]
                if target_item <= 0 or len(target_tokens) != sid_depth or not any(int(x) > 0 for x in target_tokens):
                    continue
                chain = build_transport_chain(
                    history_items=history_items,
                    target_tokens=target_tokens,
                    item_credit=float(credits[idx]),
                    iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                    epsilon=float(args.transport_epsilon),
                    n_iter=int(args.transport_iter),
                    cf_topk_history=int(args.cf_topk_history),
                    cf_smooth=float(args.cf_smooth),
                )
                payload = {
                    "episode_id": int(episode_id),
                    "page_index": int(rec.get("page_index", idx + 1)),
                    "user_id": int(rec.get("user_id", -1)),
                    "session_credit": float(session_credit),
                    "session_return_to_go": float(returns[0]) if returns else 0.0,
                    "slate_credit": float(chain["slate_credit"]),
                    "item_credit": float(credits[idx]),
                    "item_credit_raw": float(returns[idx]),
                    "selected_item_id": int(target_item),
                    "selected_sid_tokens": list(target_tokens),
                    "history_items": list(history_items),
                    "history_items_explicit": chain["history_items"],
                    "history_credit": chain["history_credit"],
                    "history_cf_drop": chain["history_cf_drop"],
                    "history_credit_calibrated": chain["history_credit_calibrated"],
                    "block_credit": chain["block_credit"],
                    "block_credit_calibrated": chain["block_credit_calibrated"],
                    "token_credit": chain["token_credit"],
                    "token_credit_calibrated": chain["token_credit_calibrated"],
                    "base_support": float(chain["base_support"]),
                    "step_reward": float(rec.get("step_reward", 0.0)),
                    "done": bool(rec.get("done", False)),
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"[phase3] saved explicit chain to {output_path}")
    print(f"[phase3] n_trace_records={len(raw_records)} n_episodes={len(grouped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
