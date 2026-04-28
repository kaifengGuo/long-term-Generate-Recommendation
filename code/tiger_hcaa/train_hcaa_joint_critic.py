import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_phase2_blend_common import (  # noqa: E402
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_tiger_model,
)
from tiger_hcla_rl.common import pooled_history_summary  # noqa: E402

from tiger_hcaa.common import (  # noqa: E402
    EpisodeDataset,
    build_episode_samples,
    collate_episodes,
    load_jsonl_rows,
    load_reader_from_uirm_log,
    masked_huber_loss,
    masked_kl_div,
    masked_mae,
    safe_correlation,
    set_random_seed,
    split_groups,
    write_json,
)
from tiger_hcaa.models import HCAAJointCritic, load_hcaa_bundle, save_hcaa_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TIGER-HCAA joint critic.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--pre_loss_scale", type=float, default=1.0)
    parser.add_argument("--post_loss_scale", type=float, default=1.0)
    parser.add_argument("--bias_loss_scale", type=float, default=0.5)
    parser.add_argument("--page_loss_scale", type=float, default=1.0)
    parser.add_argument("--item_share_loss_scale", type=float, default=0.5)
    parser.add_argument("--item_loss_scale", type=float, default=1.0)
    parser.add_argument("--token_share_loss_scale", type=float, default=0.5)
    parser.add_argument("--token_loss_scale", type=float, default=1.0)
    parser.add_argument("--page_cons_scale", type=float, default=1.0)
    parser.add_argument("--item_cons_scale", type=float, default=1.0)
    parser.add_argument("--token_cons_scale", type=float, default=1.0)
    parser.add_argument("--init_bundle_path", type=str, default="")
    parser.add_argument("--init_meta_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


def build_loaders(
    samples: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
    valid_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    dataset = EpisodeDataset(samples)
    groups = [str(row["group"]) for row in samples]
    if float(valid_ratio) > 0.0 and len(samples) > 1:
        train_idx, valid_idx = split_groups(groups, float(valid_ratio), int(seed))
    else:
        all_idx = np.arange(len(samples), dtype=np.int64)
        train_idx = all_idx
        valid_idx = all_idx
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_episodes,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_episodes,
    )
    return train_loader, valid_loader, train_idx, valid_idx


def build_samples_from_files(
    *,
    trace_path: str,
    chain_path: str,
    uirm_log_path: str,
    sid_mapping_path: str,
    max_hist_items: int,
    gamma: float,
    hazard_lambda: float,
    max_episodes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], torch.Tensor]:
    trace_rows = load_jsonl_rows(trace_path)
    chain_rows = load_jsonl_rows(chain_path)
    reader = load_reader_from_uirm_log(str(uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    samples, data_meta = build_episode_samples(
        trace_rows=trace_rows,
        chain_rows=chain_rows,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(max_hist_items),
        gamma=float(gamma),
        hazard_lambda=float(hazard_lambda),
        max_episodes=int(max_episodes),
    )
    return samples, data_meta, iid2sid_tok_cpu.cpu()


@torch.no_grad()
def compute_batch_encodings(
    tiger,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_mask = batch["page_mask"].to(device)
    pre_input_ids = batch["pre_input_ids"].to(device)
    pre_attention_mask = batch["pre_attention_mask"].to(device)
    post_input_ids = batch["post_input_ids"].to(device)
    post_attention_mask = batch["post_attention_mask"].to(device)
    token_ids = batch["token_ids"].to(device)
    item_mask = batch["item_mask"].to(device)

    bsz, max_pages, hist_len = pre_input_ids.shape
    max_items = token_ids.shape[2]
    token_len = token_ids.shape[3]
    hidden_size = int(tiger.model.config.d_model)

    flat_page_mask = page_mask.view(-1)
    flat_pre_ids = pre_input_ids.view(-1, hist_len)[flat_page_mask]
    flat_pre_attn = pre_attention_mask.view(-1, hist_len)[flat_page_mask]
    flat_post_ids = post_input_ids.view(-1, hist_len)[flat_page_mask]
    flat_post_attn = post_attention_mask.view(-1, hist_len)[flat_page_mask]
    pre_summary = pooled_history_summary(tiger, flat_pre_ids, flat_pre_attn)
    post_summary = pooled_history_summary(tiger, flat_post_ids, flat_post_attn)

    flat_token_ids = token_ids.view(-1, max_items, token_len)[flat_page_mask]
    flat_item_mask = item_mask.view(-1, max_items)[flat_page_mask]
    token_hidden = torch.zeros(
        (flat_token_ids.shape[0], max_items, token_len, hidden_size),
        dtype=pre_summary.dtype,
        device=device,
    )
    if int(flat_item_mask.sum().item()) > 0:
        page_idx, item_idx = flat_item_mask.nonzero(as_tuple=True)
        item_hist_ids = flat_pre_ids[page_idx]
        item_hist_attn = flat_pre_attn[page_idx]
        item_token_ids = flat_token_ids[page_idx, item_idx]
        _logits, hidden = tiger.decode_with_hidden(
            input_ids=item_hist_ids,
            attention_mask=item_hist_attn,
            decoder_input_ids=decoder_input_ids_from_targets(item_token_ids),
        )
        token_hidden[page_idx, item_idx] = hidden
    return pre_summary, post_summary, token_hidden


def page_conservation_loss(
    page_adv_pred: torch.Tensor,
    pre_value_pred: torch.Tensor,
    post_value_pred: torch.Tensor,
    page_mask: torch.Tensor,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for batch_idx in range(int(page_mask.shape[0])):
        valid = torch.nonzero(page_mask[batch_idx], as_tuple=False).reshape(-1)
        if valid.numel() <= 0:
            continue
        first_idx = int(valid[0].item())
        last_idx = int(valid[-1].item())
        lhs = page_adv_pred[batch_idx, valid].sum()
        rhs = post_value_pred[batch_idx, last_idx] - pre_value_pred[batch_idx, first_idx]
        losses.append((lhs - rhs).pow(2))
    if not losses:
        return page_adv_pred.new_zeros(())
    return torch.stack(losses).mean()


def forward_batch(
    tiger,
    model: HCAAJointCritic,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    page_mask = batch["page_mask"].to(device)
    item_mask = batch["item_mask"].to(device)
    token_mask = batch["token_mask"].to(device)
    page_features = batch["page_features"].to(device)
    item_features = batch["item_features"].to(device)
    token_ids = batch["token_ids"].to(device)

    pre_value_target = batch["pre_value_target"].to(device)
    post_value_target = batch["post_value_target"].to(device)
    page_bias_target = batch["page_bias_target"].to(device)
    page_adv_target = batch["page_adv_target"].to(device)
    item_share_target = batch["item_share_target"].to(device)
    item_adv_target = batch["item_adv_target"].to(device)
    token_share_target = batch["token_share_target"].to(device)
    token_adv_target = batch["token_adv_target"].to(device)

    pre_summary_flat, post_summary_flat, token_hidden_flat = compute_batch_encodings(tiger, batch, device)
    flat_page_mask = page_mask.view(-1)
    flat_item_mask = item_mask.view(-1, item_mask.shape[-1])[flat_page_mask]
    flat_token_mask = token_mask.view(-1, token_mask.shape[-2], token_mask.shape[-1])[flat_page_mask]
    flat_page_features = page_features.view(-1, page_features.shape[-1])[flat_page_mask]
    flat_item_features = item_features.view(-1, item_features.shape[-2], item_features.shape[-1])[flat_page_mask]
    flat_token_ids = token_ids.view(-1, token_ids.shape[-2], token_ids.shape[-1])[flat_page_mask]

    outputs = model(
        pre_summary=pre_summary_flat,
        post_summary=post_summary_flat,
        page_features=flat_page_features,
        item_features=flat_item_features,
        item_mask=flat_item_mask,
        token_hidden=token_hidden_flat,
        token_ids=flat_token_ids,
        token_mask=flat_token_mask,
    )

    pre_pred = torch.zeros_like(pre_value_target)
    post_pred = torch.zeros_like(post_value_target)
    page_bias_pred = torch.zeros_like(page_bias_target)
    page_adv_pred = torch.zeros_like(page_adv_target)
    item_share_pred = torch.zeros_like(item_share_target)
    item_adv_pred = torch.zeros_like(item_adv_target)
    token_share_pred = torch.zeros_like(token_share_target)
    token_adv_pred = torch.zeros_like(token_adv_target)

    pre_pred.view(-1)[flat_page_mask] = outputs["pre_value"]
    post_pred.view(-1)[flat_page_mask] = outputs["post_value"]
    page_bias_pred.view(-1)[flat_page_mask] = outputs["page_bias"]
    page_adv_pred.view(-1)[flat_page_mask] = outputs["page_adv"]
    item_share_pred.view(-1, item_mask.shape[-1])[flat_page_mask] = outputs["item_shares"]
    item_adv_pred.view(-1, item_mask.shape[-1])[flat_page_mask] = outputs["item_adv"]
    token_share_pred.view(-1, token_mask.shape[-2], token_mask.shape[-1])[flat_page_mask] = outputs["token_shares"]
    token_adv_pred.view(-1, token_mask.shape[-2], token_mask.shape[-1])[flat_page_mask] = outputs["token_adv"]

    losses = {
        "pre_loss": masked_huber_loss(pre_pred, pre_value_target, page_mask),
        "post_loss": masked_huber_loss(post_pred, post_value_target, page_mask),
        "bias_loss": masked_huber_loss(page_bias_pred, page_bias_target, page_mask),
        "page_loss": masked_huber_loss(page_adv_pred, page_adv_target, page_mask),
        "item_share_loss": masked_kl_div(item_share_pred, item_share_target, item_mask, dim=-1),
        "item_loss": masked_huber_loss(item_adv_pred, item_adv_target, item_mask),
        "token_share_loss": masked_kl_div(token_share_pred, token_share_target, token_mask, dim=-1),
        "token_loss": masked_huber_loss(token_adv_pred, token_adv_target, token_mask),
        "page_cons_loss": page_conservation_loss(page_adv_pred, pre_pred, post_pred, page_mask),
        "item_cons_loss": masked_huber_loss(item_adv_pred.sum(dim=-1), page_adv_pred, page_mask),
        "token_cons_loss": masked_huber_loss(token_adv_pred.sum(dim=-1), item_adv_pred, item_mask),
    }
    total_loss = (
        float(args.pre_loss_scale) * losses["pre_loss"]
        + float(args.post_loss_scale) * losses["post_loss"]
        + float(args.bias_loss_scale) * losses["bias_loss"]
        + float(args.page_loss_scale) * losses["page_loss"]
        + float(args.item_share_loss_scale) * losses["item_share_loss"]
        + float(args.item_loss_scale) * losses["item_loss"]
        + float(args.token_share_loss_scale) * losses["token_share_loss"]
        + float(args.token_loss_scale) * losses["token_loss"]
        + float(args.page_cons_scale) * losses["page_cons_loss"]
        + float(args.item_cons_scale) * losses["item_cons_loss"]
        + float(args.token_cons_scale) * losses["token_cons_loss"]
    )
    stats = {
        "loss": float(total_loss.item()),
        "pre_loss": float(losses["pre_loss"].item()),
        "post_loss": float(losses["post_loss"].item()),
        "bias_loss": float(losses["bias_loss"].item()),
        "page_loss": float(losses["page_loss"].item()),
        "item_share_loss": float(losses["item_share_loss"].item()),
        "item_loss": float(losses["item_loss"].item()),
        "token_share_loss": float(losses["token_share_loss"].item()),
        "token_loss": float(losses["token_loss"].item()),
        "page_cons_loss": float(losses["page_cons_loss"].item()),
        "item_cons_loss": float(losses["item_cons_loss"].item()),
        "token_cons_loss": float(losses["token_cons_loss"].item()),
        "pre_mae": float(masked_mae(pre_pred, pre_value_target, page_mask).item()),
        "post_mae": float(masked_mae(post_pred, post_value_target, page_mask).item()),
        "bias_mae": float(masked_mae(page_bias_pred, page_bias_target, page_mask).item()),
        "page_mae": float(masked_mae(page_adv_pred, page_adv_target, page_mask).item()),
        "item_share_mae": float(masked_mae(item_share_pred, item_share_target, item_mask).item()),
        "item_mae": float(masked_mae(item_adv_pred, item_adv_target, item_mask).item()),
        "token_share_mae": float(masked_mae(token_share_pred, token_share_target, token_mask).item()),
        "token_mae": float(masked_mae(token_adv_pred, token_adv_target, token_mask).item()),
        "page_corr": safe_correlation(page_adv_pred, page_adv_target, page_mask),
        "item_corr": safe_correlation(item_adv_pred, item_adv_target, item_mask),
        "token_corr": safe_correlation(token_adv_pred, token_adv_target, token_mask),
    }
    return total_loss, stats


@torch.no_grad()
def evaluate(
    tiger,
    model: HCAAJointCritic,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_batch(tiger, model, batch, device, args)
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def train_one_epoch(
    tiger,
    model: HCAAJointCritic,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, stats = forward_batch(tiger, model, batch, device, args)
        loss.backward()
        if float(args.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    set_random_seed(int(args.seed))
    device = torch.device(str(args.device))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    samples, data_meta, _iid2sid_tok_cpu = build_samples_from_files(
        trace_path=str(args.trace_path),
        chain_path=str(args.chain_path),
        uirm_log_path=str(args.uirm_log_path),
        sid_mapping_path=str(args.sid_mapping_path),
        max_hist_items=int(args.max_hist_items),
        gamma=float(args.gamma),
        hazard_lambda=float(args.hazard_lambda),
        max_episodes=int(args.max_episodes),
    )
    if not samples:
        raise ValueError("No usable HCAA samples were built from trace/chain.")
    train_loader, valid_loader, train_idx, valid_idx = build_loaders(
        samples,
        batch_size=int(args.batch_size),
        valid_ratio=float(args.valid_ratio),
        seed=int(args.seed),
    )

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, codebook_size_model = load_tiger_model(
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

    if str(args.init_bundle_path).strip() and str(args.init_meta_path).strip():
        model, init_meta = load_hcaa_bundle(str(args.init_bundle_path), str(args.init_meta_path), device)
        page_dim = int(init_meta["page_dim"])
        item_dim = int(init_meta["item_dim"])
    else:
        page_dim = int(data_meta["page_dim"])
        item_dim = int(data_meta["item_dim"])
        model = HCAAJointCritic(
            hidden_size=int(size_cfg["d_model"]),
            page_dim=page_dim,
            item_dim=item_dim,
                vocab_size=int(codebook_size_model) + 1,
            mlp_dim=int(args.mlp_dim),
            token_dim=int(args.token_dim),
            dropout=float(args.dropout),
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: List[Dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    bundle_path = save_dir / "hcaa_joint_bundle.pt"
    meta_path = save_dir / "hcaa_joint_meta.json"

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(tiger, model, optimizer, train_loader, device, args)
        valid_metrics = evaluate(tiger, model, valid_loader, device, args)
        history.append({"epoch": int(epoch), "train": train_metrics, "valid": valid_metrics})
        if float(valid_metrics.get("loss", float("inf"))) < float(best_valid_loss):
            best_valid_loss = float(valid_metrics["loss"])
            best_epoch = int(epoch)
            meta = {
                "method": "TIGER-HCAA joint critic",
                "hidden_size": int(size_cfg["d_model"]),
                "page_dim": int(page_dim),
                "item_dim": int(item_dim),
                "vocab_size": int(codebook_size_model) + 1,
                "mlp_dim": int(args.mlp_dim),
                "token_dim": int(args.token_dim),
                "dropout": float(args.dropout),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid_loss),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "data_meta": data_meta,
            }
            save_hcaa_bundle(bundle_path, meta_path, model, meta)

    metrics = {
        "method": "TIGER-HCAA joint critic",
        "trace_path": str(Path(args.trace_path).resolve()),
        "chain_path": str(Path(args.chain_path).resolve()),
        "bundle_path": str(bundle_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "history": history,
        "data_meta": data_meta,
    }
    metrics_out = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "hcaa_joint_metrics.json"
    write_json(metrics_out, metrics)
    return metrics


def main() -> int:
    args = parse_args()
    metrics = run_training(args)
    print(json.dumps({"best_epoch": int(metrics["best_epoch"]), "best_valid_loss": float(metrics["best_valid_loss"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
