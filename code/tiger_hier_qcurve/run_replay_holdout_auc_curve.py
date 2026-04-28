import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log
from tiger_hier_prefix_common import ItemPrefixValueHead, load_item_prefix_head, load_page_prefix_head
from tiger_phase2_blend_common import (
    TokenPrefixValueHead,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_prefix_value_head,
    load_tiger_model,
)
from tiger_phase6_joint_common import SlateCreditHead
from tiger_hier_qcurve.run_userbatch_incremental_loss_curve import (
    build_eval_bundle,
    train_item_one_pass,
    train_page_one_pass,
    train_token_one_pass,
)
from train_tiger_item_prefix_critic import collate_rows as collate_item_rows
from train_tiger_page_prefix_critic import collate_rows as collate_page_rows, pool_history_summary
from train_tiger_prefix_critic import collate_prefix, prefix_to_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay incremental critic updates from saved steps and evaluate holdout ROC-AUC curves."
    )
    parser.add_argument("--source_output_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--base_tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--page_credit_mode", type=str, default="centered")
    parser.add_argument("--page_lr", type=float, default=1e-4)
    parser.add_argument("--item_lr", type=float, default=1e-4)
    parser.add_argument("--token_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--page_batch_size", type=int, default=64)
    parser.add_argument("--item_batch_size", type=int, default=128)
    parser.add_argument("--token_batch_size", type=int, default=64)
    parser.add_argument("--page_train_passes", type=int, default=1)
    parser.add_argument("--item_train_passes", type=int, default=1)
    parser.add_argument("--token_train_passes", type=int, default=1)
    parser.add_argument("--token_target_clip", type=float, default=0.0)
    parser.add_argument("--token_prefix_loss_scale", type=float, default=1.0)
    parser.add_argument("--token_delta_loss_scale", type=float, default=1.0)
    parser.add_argument("--token_item_loss_scale", type=float, default=0.5)
    parser.add_argument("--page_init_head_path", type=str, default="")
    parser.add_argument("--page_init_meta_path", type=str, default="")
    parser.add_argument("--item_init_head_path", type=str, default="")
    parser.add_argument("--item_init_meta_path", type=str, default="")
    parser.add_argument("--token_init_head_path", type=str, default="")
    parser.add_argument("--token_init_meta_path", type=str, default="")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--max_step", type=int, default=0)
    parser.add_argument("--label_eps", type=float, default=1e-8)
    return parser.parse_args()


def load_source_summary(source_output_root: Path) -> List[Dict[str, Any]]:
    summary_path = source_output_root / "userbatch_incremental_loss_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing source summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def safe_roc_auc(scores: np.ndarray, targets: np.ndarray, eps: float) -> Dict[str, float]:
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must share shape")
    mask = np.abs(targets) > float(eps)
    if not np.any(mask):
        return {"auc": float("nan"), "n_eval": 0, "n_pos": 0, "n_neg": 0}
    y_score = scores[mask]
    y_true = (targets[mask] > float(eps)).astype(np.int64)
    pos = int(y_true.sum())
    neg = int(y_true.shape[0] - pos)
    if pos == 0 or neg == 0:
        return {"auc": float("nan"), "n_eval": int(y_true.shape[0]), "n_pos": pos, "n_neg": neg}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "n_eval": int(y_true.shape[0]),
        "n_pos": pos,
        "n_neg": neg,
    }


def evaluate_page_auc(tiger, head, loader, device: torch.device, eps: float) -> Dict[str, float]:
    head.eval()
    score_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            summary = pool_history_summary(tiger, input_ids, attention_mask)
            pred = head(summary.detach())
            score_list.append(pred.detach().cpu().numpy().astype(np.float64))
            target_list.append(targets.detach().cpu().numpy().astype(np.float64))
    scores = np.concatenate(score_list, axis=0) if score_list else np.zeros((0,), dtype=np.float64)
    targets = np.concatenate(target_list, axis=0) if target_list else np.zeros((0,), dtype=np.float64)
    return safe_roc_auc(scores, targets, eps)


def evaluate_item_auc(head, loader, device: torch.device, eps: float) -> Dict[str, float]:
    head.eval()
    score_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = head(
                batch["item_features"].to(device),
                batch["page_features"].to(device),
                mask=batch["mask"].to(device),
                prefix_len=batch["prefix_len"].to(device),
                total_items=batch["total_items"].to(device),
            )
            targets = batch["targets"].to(device)
            score_list.append(pred.detach().cpu().numpy().astype(np.float64))
            target_list.append(targets.detach().cpu().numpy().astype(np.float64))
    scores = np.concatenate(score_list, axis=0) if score_list else np.zeros((0,), dtype=np.float64)
    targets = np.concatenate(target_list, axis=0) if target_list else np.zeros((0,), dtype=np.float64)
    return safe_roc_auc(scores, targets, eps)


def evaluate_token_delta_auc(tiger, head, loader, device: torch.device, eps: float) -> Dict[str, float]:
    head.eval()
    score_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            token_credit = batch["token_credit"].to(device)
            _logits, hidden = tiger.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
            )
            pred_prefix = head(hidden.detach(), target_tokens)
            pred_delta = prefix_to_delta(pred_prefix)
            score_list.append(pred_delta.detach().cpu().numpy().astype(np.float64).reshape(-1))
            target_list.append(token_credit.detach().cpu().numpy().astype(np.float64).reshape(-1))
    scores = np.concatenate(score_list, axis=0) if score_list else np.zeros((0,), dtype=np.float64)
    targets = np.concatenate(target_list, axis=0) if target_list else np.zeros((0,), dtype=np.float64)
    return safe_roc_auc(scores, targets, eps)


def build_plots(summary: List[Dict[str, Any]], output_root: Path) -> None:
    def draw(metric_key: str, title: str, file_name: str, color: str) -> None:
        points = [(int(row["step"]), float(row[metric_key])) for row in summary if metric_key in row and pd.notna(row[metric_key])]
        if not points:
            return
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=2.0, color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_root / file_name, dpi=180)
        plt.close(fig)

    draw("page_holdout_auc", "Page Holdout ROC-AUC vs Step", "page_holdout_auc_curve.png", "#1d4ed8")
    draw("item_holdout_auc", "Item Holdout ROC-AUC vs Step", "item_holdout_auc_curve.png", "#d9480f")
    draw("token_holdout_delta_auc", "Token Delta Holdout ROC-AUC vs Step", "token_holdout_auc_curve.png", "#0f766e")


def write_summary(summary: List[Dict[str, Any]], output_root: Path) -> None:
    json_path = output_root / "holdout_auc_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_root / "holdout_auc_summary.csv"
    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    build_plots(summary, output_root)


def resolve_holdout_paths(source_summary: List[Dict[str, Any]], source_output_root: Path) -> Tuple[Path, Path]:
    first = source_summary[0] if source_summary else {}
    trace_path = first.get("holdout_trace_path", "")
    chain_path = first.get("holdout_chain_path", "")
    if trace_path and chain_path:
        return Path(trace_path), Path(chain_path)
    fallback_dir = source_output_root / f"fixed_holdout_{int(first.get('holdout_users', 0))}users"
    return fallback_dir / "holdout_trace.jsonl", fallback_dir / "holdout_chain.jsonl"


def main() -> int:
    args = parse_args()
    source_output_root = Path(args.source_output_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_summary = load_source_summary(source_output_root)
    if not source_summary:
        raise ValueError("Source summary is empty.")
    max_step_available = int(source_summary[-1]["step"])
    max_step = int(args.max_step) if int(args.max_step) > 0 else max_step_available
    max_step = min(max_step, max_step_available)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(str(args.device))
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))

    size_args = infer_model_size_args(str(args.model_size))
    tiger, sid_depth, codebook_size = load_tiger_model(
        tiger_ckpt=str(args.base_tiger_ckpt),
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
    for p in tiger.parameters():
        p.requires_grad = False
    tiger.eval()

    if str(args.page_init_head_path).strip() and str(args.page_init_meta_path).strip():
        page_head, _page_meta = load_page_prefix_head(str(args.page_init_head_path), str(args.page_init_meta_path), device)
    else:
        page_head = SlateCreditHead(hidden_size=int(size_args["d_model"]), mlp_dim=128).to(device)
    item_head = None
    if str(args.item_init_head_path).strip() and str(args.item_init_meta_path).strip():
        item_head, _item_meta = load_item_prefix_head(str(args.item_init_head_path), str(args.item_init_meta_path), device)
    token_head = None
    if str(args.token_init_head_path).strip() and str(args.token_init_meta_path).strip():
        token_head, _token_meta = load_prefix_value_head(str(args.token_init_head_path), str(args.token_init_meta_path), device)

    page_optimizer = torch.optim.AdamW(page_head.parameters(), lr=float(args.page_lr), weight_decay=float(args.weight_decay))
    item_optimizer = (
        torch.optim.AdamW(item_head.parameters(), lr=float(args.item_lr), weight_decay=float(args.weight_decay))
        if item_head is not None
        else None
    )
    token_optimizer = (
        torch.optim.AdamW(token_head.parameters(), lr=float(args.token_lr), weight_decay=float(args.weight_decay))
        if token_head is not None
        else None
    )

    holdout_trace_path, holdout_chain_path = resolve_holdout_paths(source_summary, source_output_root)
    holdout_bundle = build_eval_bundle(holdout_trace_path, holdout_chain_path, iid2sid_tok_cpu, sid_depth, args)

    summary: List[Dict[str, Any]] = []
    started = time.time()
    for step in range(1, max_step + 1):
        step_dir = source_output_root / f"step_{step:02d}"
        trace_path = step_dir / "rollout_trace.jsonl"
        chain_path = step_dir / "rollout_chain.jsonl"
        if not trace_path.exists() or not chain_path.exists():
            break

        step_bundle = build_eval_bundle(trace_path, chain_path, iid2sid_tok_cpu, sid_depth, args)

        page_loader = step_bundle["page_loader"]
        for _ in range(int(args.page_train_passes)):
            train_page_one_pass(tiger, page_head, page_optimizer, page_loader, device)

        item_dataset = step_bundle["item_dataset"]
        if item_head is None:
            first_row = item_dataset[0]
            item_head = ItemPrefixValueHead(
                item_dim=int(first_row["item_features"].shape[1]),
                page_dim=int(first_row["page_features"].shape[0]),
                hidden_dim=96,
                dropout=0.10,
                stats_dim=3,
            ).to(device)
            item_optimizer = torch.optim.AdamW(item_head.parameters(), lr=float(args.item_lr), weight_decay=float(args.weight_decay))
        item_loader = step_bundle["item_loader"]
        for _ in range(int(args.item_train_passes)):
            train_item_one_pass(item_head, item_optimizer, item_loader, device)

        if token_head is None:
            token_head = TokenPrefixValueHead(
                hidden_size=int(size_args["d_model"]),
                vocab_size=int(codebook_size) + 1,
                token_dim=32,
                mlp_dim=128,
            ).to(device)
            token_optimizer = torch.optim.AdamW(token_head.parameters(), lr=float(args.token_lr), weight_decay=float(args.weight_decay))
        token_loader = step_bundle["token_loader"]
        for _ in range(int(args.token_train_passes)):
            train_token_one_pass(tiger, token_head, token_optimizer, token_loader, device, args)

        should_eval = (step % int(args.eval_every) == 0) or (step == max_step)
        if not should_eval:
            continue

        eval_start = time.time()
        page_auc = evaluate_page_auc(tiger, page_head, holdout_bundle["page_loader"], device, float(args.label_eps))
        item_auc = evaluate_item_auc(item_head, holdout_bundle["item_loader"], device, float(args.label_eps))
        token_auc = evaluate_token_delta_auc(tiger, token_head, holdout_bundle["token_loader"], device, float(args.label_eps))
        record = {
            "step": int(step),
            "page_holdout_auc": page_auc["auc"],
            "page_n_eval": page_auc["n_eval"],
            "page_n_pos": page_auc["n_pos"],
            "page_n_neg": page_auc["n_neg"],
            "item_holdout_auc": item_auc["auc"],
            "item_n_eval": item_auc["n_eval"],
            "item_n_pos": item_auc["n_pos"],
            "item_n_neg": item_auc["n_neg"],
            "token_holdout_delta_auc": token_auc["auc"],
            "token_n_eval": token_auc["n_eval"],
            "token_n_pos": token_auc["n_pos"],
            "token_n_neg": token_auc["n_neg"],
            "elapsed_minutes": float((time.time() - started) / 60.0),
            "eval_seconds": float(time.time() - eval_start),
        }
        summary.append(record)
        write_summary(summary, output_root)
        print(json.dumps(record, ensure_ascii=False))

    final_path = output_root / "holdout_auc_summary.json"
    print(json.dumps({"output_root": str(output_root), "summary_path": str(final_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
