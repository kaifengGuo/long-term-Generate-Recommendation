import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_phase2_blend_common import build_iid2sid_tokens, infer_model_size_args, load_tiger_model  # noqa: E402

from tiger_page_sid_rl.common import (  # noqa: E402
    TracePageDataset,
    build_page_samples,
    collate_trace_pages,
    load_jsonl_rows,
    load_reader_from_uirm_log,
    masked_mean,
    masked_huber_loss,
    masked_mae,
    pooled_history_summary,
    safe_correlation,
    set_random_seed,
    split_groups,
    write_json,
)
from tiger_page_sid_rl.models import (  # noqa: E402
    PageSIDQCritic,
    PageSIDQCriticV8,
    PageSIDQCriticEnsemble,
    build_page_sid_qcritic_base,
    load_page_sid_qcritic_bundle,
    save_page_sid_qcritic_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train page-level Q critic for the TIGER Page-SID closed loop.")
    parser.add_argument("--trace_path", type=str, required=True)
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--item_dim", type=int, default=128)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--critic_arch", type=str, default="base", choices=["base", "v8", "v9add"])
    parser.add_argument("--critic_num_heads", type=int, default=4)
    parser.add_argument("--critic_num_layers", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--critic_target_heuristic_mix", type=float, default=0.60)
    parser.add_argument("--critic_target_support_mix", type=float, default=0.25)
    parser.add_argument("--critic_target_response_mix", type=float, default=0.15)
    parser.add_argument("--critic_page_loss_scale", type=float, default=1.0)
    parser.add_argument("--critic_item_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_prefix_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_page_huber_beta", type=float, default=1.0)
    parser.add_argument("--critic_item_huber_beta", type=float, default=1.0)
    parser.add_argument("--critic_prefix_huber_beta", type=float, default=1.0)
    parser.add_argument("--critic_rank_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_monotonic_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_rank_min_gap", type=float, default=0.05)
    parser.add_argument("--init_bundle_path", type=str, default="")
    parser.add_argument("--init_meta_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, default="")
    parser.add_argument("--step_metrics_csv_out", type=str, default="")
    parser.add_argument("--step_metrics_jsonl_out", type=str, default="")
    return parser.parse_args()


def extract_q_outputs(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_mean = outputs.get("q_mean", outputs["q_value"])
    q_std = outputs.get("q_std", torch.zeros_like(q_mean))
    q_values = outputs.get("q_values")
    if q_values is None:
        q_values = q_mean.unsqueeze(-1)
    return q_mean, q_std, q_values


def build_loaders(
    samples: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
    valid_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    dataset = TracePageDataset(samples)
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
        collate_fn=collate_trace_pages,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_trace_pages,
    )
    return train_loader, valid_loader, train_idx, valid_idx


def build_samples_from_trace(
    *,
    trace_path: str,
    uirm_log_path: str,
    sid_mapping_path: str,
    max_hist_items: int,
    gamma: float,
    hazard_lambda: float,
    max_episodes: int,
    critic_target_heuristic_mix: float,
    critic_target_support_mix: float,
    critic_target_response_mix: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], torch.Tensor]:
    trace_rows = load_jsonl_rows(trace_path)
    reader = load_reader_from_uirm_log(str(uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    samples, data_meta = build_page_samples(
        trace_rows=trace_rows,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        reader=reader,
        max_hist_items=int(max_hist_items),
        gamma=float(gamma),
        hazard_lambda=float(hazard_lambda),
        max_episodes=int(max_episodes),
        aux_target_heuristic_mix=float(critic_target_heuristic_mix),
        aux_target_support_mix=float(critic_target_support_mix),
        aux_target_response_mix=float(critic_target_response_mix),
    )
    return samples, data_meta, iid2sid_tok_cpu.cpu()


def has_auxiliary_losses(args: argparse.Namespace) -> bool:
    return any(
        float(scale) > 0.0
        for scale in [
            args.critic_item_loss_scale,
            args.critic_prefix_loss_scale,
            args.critic_rank_loss_scale,
            args.critic_monotonic_loss_scale,
        ]
    )


def build_variant_batch(
    token_ids: torch.Tensor,
    item_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    batch_size, max_items, sid_depth = token_ids.shape
    variant_tokens: List[torch.Tensor] = []
    variant_owner: List[int] = []
    item_null_indices = torch.full((batch_size, max_items), -1, dtype=torch.long, device=token_ids.device)
    prefix_indices = torch.full((batch_size, max_items, sid_depth), -1, dtype=torch.long, device=token_ids.device)

    for batch_idx in range(batch_size):
        base_tokens = token_ids[batch_idx]
        for item_idx in range(max_items):
            if not bool(item_mask[batch_idx, item_idx].item()):
                continue
            null_variant = base_tokens.clone()
            null_variant[item_idx].zero_()
            item_null_indices[batch_idx, item_idx] = len(variant_tokens)
            variant_tokens.append(null_variant)
            variant_owner.append(int(batch_idx))
            valid_len = int((base_tokens[item_idx] > 0).sum().item())
            for sid_idx in range(valid_len):
                prefix_variant = base_tokens.clone()
                prefix_variant[item_idx, sid_idx + 1:].zero_()
                prefix_indices[batch_idx, item_idx, sid_idx] = len(variant_tokens)
                variant_tokens.append(prefix_variant)
                variant_owner.append(int(batch_idx))

    if not variant_tokens:
        return None

    variant_token_ids = torch.stack(variant_tokens, dim=0)
    variant_item_mask = (variant_token_ids > 0).any(dim=-1)
    variant_owner_idx = torch.tensor(variant_owner, dtype=torch.long, device=token_ids.device)
    return variant_token_ids, variant_item_mask, variant_owner_idx, item_null_indices, prefix_indices


def pairwise_rank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    min_gap: float,
) -> Tuple[torch.Tensor, float]:
    loss_terms: List[torch.Tensor] = []
    acc_terms: List[torch.Tensor] = []
    for row_idx in range(int(scores.shape[0])):
        valid_idx = torch.nonzero(mask[row_idx], as_tuple=False).reshape(-1)
        if valid_idx.numel() <= 1:
            continue
        row_scores = scores[row_idx, valid_idx]
        row_targets = targets[row_idx, valid_idx]
        target_diff = row_targets.unsqueeze(1) - row_targets.unsqueeze(0)
        pair_mask = torch.triu(target_diff.abs() > float(min_gap), diagonal=1)
        if not bool(pair_mask.any()):
            continue
        score_diff = row_scores.unsqueeze(1) - row_scores.unsqueeze(0)
        signed_margin = torch.sign(target_diff) * score_diff
        pair_terms = signed_margin[pair_mask]
        loss_terms.append(F.softplus(-pair_terms).mean())
        acc_terms.append((pair_terms > 0).float().mean())
    if not loss_terms:
        return scores.new_zeros(()), 0.0
    return torch.stack(loss_terms).mean(), float(torch.stack(acc_terms).mean().item())


def forward_batch(
    tiger,
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pre_input_ids = batch["pre_input_ids"].to(device)
    pre_attention_mask = batch["pre_attention_mask"].to(device)
    page_features = batch["page_features"].to(device)
    user_features = batch["user_features"].to(device)
    token_ids = batch["token_ids"].to(device)
    item_mask = batch["item_mask"].to(device)
    q_target = batch["q_target"].to(device)
    item_share_target = batch["item_share_target"].to(device)
    item_adv_target = batch["item_adv_target"].to(device)
    sid_adv_target = batch["sid_adv_target"].to(device)
    sid_mask = (token_ids > 0) & item_mask.unsqueeze(-1)

    with torch.no_grad():
        pre_summary = pooled_history_summary(tiger, pre_input_ids, pre_attention_mask)
    outputs = model(
        pre_summary=pre_summary,
        token_ids=token_ids,
        item_mask=item_mask,
        page_features=page_features,
        user_features=user_features,
    )
    q_pred, q_std, q_values = extract_q_outputs(outputs)
    q_target_all = q_target.unsqueeze(-1).expand_as(q_values)
    page_loss = masked_huber_loss(
        q_values,
        q_target_all,
        None,
        beta=float(args.critic_page_huber_beta),
    )
    item_loss = q_pred.new_zeros(())
    prefix_loss = q_pred.new_zeros(())
    rank_loss = q_pred.new_zeros(())
    monotonic_loss = q_pred.new_zeros(())
    item_rank_acc = 0.0
    item_delta_mae = 0.0
    item_delta_corr = 0.0
    prefix_delta_mae = 0.0
    prefix_delta_corr = 0.0
    item_delta_abs_mean = 0.0
    prefix_delta_abs_mean = 0.0
    variant_rows = 0

    if has_auxiliary_losses(args):
        variant_pack = build_variant_batch(token_ids, item_mask)
        if variant_pack is not None:
            variant_token_ids, variant_item_mask, variant_owner_idx, item_null_indices, prefix_indices = variant_pack
            variant_rows = int(variant_token_ids.shape[0])
            variant_outputs = model(
                pre_summary=pre_summary.index_select(0, variant_owner_idx),
                token_ids=variant_token_ids,
                item_mask=variant_item_mask,
                page_features=page_features.index_select(0, variant_owner_idx),
                user_features=user_features.index_select(0, variant_owner_idx),
            )
            variant_q_mean, _variant_q_std, variant_q_values = extract_q_outputs(variant_outputs)
            ensemble_size = int(variant_q_values.shape[-1])

            item_null_q_mean = torch.zeros_like(item_adv_target)
            item_null_q_values = torch.zeros(
                item_adv_target.shape + (ensemble_size,),
                dtype=q_values.dtype,
                device=device,
            )
            prefix_q_mean = torch.zeros_like(sid_adv_target)
            prefix_q_values = torch.zeros(
                sid_adv_target.shape + (ensemble_size,),
                dtype=q_values.dtype,
                device=device,
            )

            item_null_mask = item_null_indices >= 0
            if bool(item_null_mask.any()):
                flat_idx = item_null_indices[item_null_mask]
                item_null_q_mean[item_null_mask] = variant_q_mean.index_select(0, flat_idx)
                item_null_q_values[item_null_mask] = variant_q_values.index_select(0, flat_idx)

            prefix_variant_mask = prefix_indices >= 0
            if bool(prefix_variant_mask.any()):
                flat_idx = prefix_indices[prefix_variant_mask]
                prefix_q_mean[prefix_variant_mask] = variant_q_mean.index_select(0, flat_idx)
                prefix_q_values[prefix_variant_mask] = variant_q_values.index_select(0, flat_idx)

            item_delta_mean = q_pred.unsqueeze(1) - item_null_q_mean
            item_delta_values = q_values.unsqueeze(1) - item_null_q_values
            item_loss = masked_huber_loss(
                item_delta_values,
                item_adv_target.unsqueeze(-1).expand_as(item_delta_values),
                item_mask.unsqueeze(-1).expand_as(item_delta_values),
                beta=float(args.critic_item_huber_beta),
            )

            sid_delta_mean = torch.zeros_like(sid_adv_target)
            sid_delta_values = torch.zeros_like(prefix_q_values)
            prev_q_mean = item_null_q_mean
            prev_q_values = item_null_q_values
            for sid_idx in range(int(token_ids.shape[-1])):
                cur_q_mean = prefix_q_mean[:, :, sid_idx]
                cur_q_values = prefix_q_values[:, :, sid_idx, :]
                sid_delta_mean[:, :, sid_idx] = cur_q_mean - prev_q_mean
                sid_delta_values[:, :, sid_idx, :] = cur_q_values - prev_q_values
                cur_mask = sid_mask[:, :, sid_idx]
                prev_q_mean = torch.where(cur_mask, cur_q_mean, prev_q_mean)
                prev_q_values = torch.where(cur_mask.unsqueeze(-1), cur_q_values, prev_q_values)

            prefix_loss = masked_huber_loss(
                sid_delta_values,
                sid_adv_target.unsqueeze(-1).expand_as(sid_delta_values),
                sid_mask.unsqueeze(-1).expand_as(sid_delta_values),
                beta=float(args.critic_prefix_huber_beta),
            )
            rank_loss, item_rank_acc = pairwise_rank_loss(
                item_delta_mean,
                item_share_target,
                item_mask,
                float(args.critic_rank_min_gap),
            )
            sign_mask = sid_mask & (item_adv_target.abs().unsqueeze(-1) > 1e-8)
            sign_target = torch.sign(item_adv_target).unsqueeze(-1)
            monotonic_loss = masked_mean(F.relu(-(sid_delta_mean * sign_target)), sign_mask)
            item_delta_mae = float(masked_mae(item_delta_mean, item_adv_target, item_mask).item())
            item_delta_corr = safe_correlation(item_delta_mean, item_adv_target, item_mask)
            prefix_delta_mae = float(masked_mae(sid_delta_mean, sid_adv_target, sid_mask).item())
            prefix_delta_corr = safe_correlation(sid_delta_mean, sid_adv_target, sid_mask)
            item_delta_abs_mean = float(masked_mean(item_delta_mean.abs(), item_mask).item())
            prefix_delta_abs_mean = float(masked_mean(sid_delta_mean.abs(), sid_mask).item())

    loss = (
        float(args.critic_page_loss_scale) * page_loss
        + float(args.critic_item_loss_scale) * item_loss
        + float(args.critic_prefix_loss_scale) * prefix_loss
        + float(args.critic_rank_loss_scale) * rank_loss
        + float(args.critic_monotonic_loss_scale) * monotonic_loss
    )
    stats = {
        "loss": float(loss.item()),
        "page_loss": float(page_loss.item()),
        "item_loss": float(item_loss.item()),
        "prefix_loss": float(prefix_loss.item()),
        "rank_loss": float(rank_loss.item()),
        "monotonic_loss": float(monotonic_loss.item()),
        "q_mae": float(masked_mae(q_pred, q_target, None).item()),
        "q_corr": safe_correlation(q_pred, q_target, None),
        "q_mean": float(q_pred.mean().item()),
        "target_mean": float(q_target.mean().item()),
        "q_std_mean": float(q_std.mean().item()),
        "item_delta_mae": float(item_delta_mae),
        "item_delta_corr": float(item_delta_corr),
        "prefix_delta_mae": float(prefix_delta_mae),
        "prefix_delta_corr": float(prefix_delta_corr),
        "item_delta_abs_mean": float(item_delta_abs_mean),
        "prefix_delta_abs_mean": float(prefix_delta_abs_mean),
        "item_rank_acc": float(item_rank_acc),
        "variant_rows": float(variant_rows),
    }
    return loss, stats


@torch.no_grad()
def evaluate(
    tiger,
    model,
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
    model,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int,
    global_step_start: int,
    step_csv_writer=None,
    step_jsonl_fp=None,
) -> Dict[str, float]:
    model.train()
    metrics: Dict[str, List[float]] = {}
    global_step = int(global_step_start)
    total_steps = max(len(loader), 1)
    for step_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss, stats = forward_batch(tiger, model, batch, device, args)
        loss.backward()
        if float(args.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()
        global_step += 1
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
        if step_csv_writer is not None:
            row = {
                "event_type": "train_step",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "step_in_epoch": int(step_idx),
                "steps_per_epoch": int(total_steps),
                "loss": float(stats.get("loss", 0.0)),
                "page_loss": float(stats.get("page_loss", 0.0)),
                "item_loss": float(stats.get("item_loss", 0.0)),
                "prefix_loss": float(stats.get("prefix_loss", 0.0)),
                "rank_loss": float(stats.get("rank_loss", 0.0)),
                "monotonic_loss": float(stats.get("monotonic_loss", 0.0)),
                "q_mae": float(stats.get("q_mae", 0.0)),
                "q_corr": float(stats.get("q_corr", 0.0)),
                "q_mean": float(stats.get("q_mean", 0.0)),
                "target_mean": float(stats.get("target_mean", 0.0)),
                "q_std_mean": float(stats.get("q_std_mean", 0.0)),
                "item_delta_mae": float(stats.get("item_delta_mae", 0.0)),
                "prefix_delta_mae": float(stats.get("prefix_delta_mae", 0.0)),
                "item_rank_acc": float(stats.get("item_rank_acc", 0.0)),
            }
            step_csv_writer.writerow(row)
            if step_jsonl_fp is not None:
                step_jsonl_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    set_random_seed(int(args.seed))
    device = torch.device(str(args.device))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    samples, data_meta, iid2sid_tok_cpu = build_samples_from_trace(
        trace_path=str(args.trace_path),
        uirm_log_path=str(args.uirm_log_path),
        sid_mapping_path=str(args.sid_mapping_path),
        max_hist_items=int(args.max_hist_items),
        gamma=float(args.gamma),
        hazard_lambda=float(args.hazard_lambda),
        max_episodes=int(args.max_episodes),
        critic_target_heuristic_mix=float(args.critic_target_heuristic_mix),
        critic_target_support_mix=float(args.critic_target_support_mix),
        critic_target_response_mix=float(args.critic_target_response_mix),
    )
    if not samples:
        raise ValueError("No usable page samples were built from trace.")
    train_loader, valid_loader, train_idx, valid_idx = build_loaders(
        samples,
        batch_size=int(args.batch_size),
        valid_ratio=float(args.valid_ratio),
        seed=int(args.seed),
    )

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, _codebook_size_model = load_tiger_model(
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

    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    if str(args.init_bundle_path).strip() and str(args.init_meta_path).strip():
        model, init_meta = load_page_sid_qcritic_bundle(str(args.init_bundle_path), str(args.init_meta_path), device)
        page_feat_dim = int(init_meta["page_feat_dim"])
        critic_arch = str(init_meta.get("arch", "base")).lower()
        user_feat_dim = int(init_meta.get("user_feat_dim", int(data_meta.get("user_feat_dim", 0))))
        critic_num_heads = int(init_meta.get("num_heads", int(args.critic_num_heads)))
        critic_num_layers = int(init_meta.get("num_layers", int(args.critic_num_layers)))
        ensemble_size = int(init_meta.get("ensemble_size", int(args.ensemble_size)))
    else:
        page_feat_dim = int(data_meta["page_feat_dim"])
        critic_arch = str(args.critic_arch).lower()
        user_feat_dim = int(data_meta.get("user_feat_dim", 0))
        critic_num_heads = int(args.critic_num_heads)
        critic_num_layers = int(args.critic_num_layers)
        ensemble_size = int(args.ensemble_size)
        base_kwargs = {
            "arch": str(critic_arch),
            "hidden_size": int(size_cfg["d_model"]),
            "vocab_size": int(vocab_size),
            "page_feat_dim": int(page_feat_dim),
            "user_feat_dim": int(user_feat_dim),
            "sid_depth": int(data_meta.get("sid_depth", 4)),
            "item_dim": int(args.item_dim),
            "model_dim": int(args.model_dim),
            "num_heads": int(critic_num_heads),
            "num_layers": int(critic_num_layers),
            "dropout": float(args.dropout),
        }
        if int(ensemble_size) > 1:
            model = PageSIDQCriticEnsemble(
                [build_page_sid_qcritic_base(**base_kwargs) for _ in range(int(ensemble_size))]
            ).to(device)
        else:
            model = build_page_sid_qcritic_base(**base_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    step_csv_path = Path(args.step_metrics_csv_out) if str(args.step_metrics_csv_out).strip() else save_dir / "page_sid_qcritic_step_metrics.csv"
    step_jsonl_path = Path(args.step_metrics_jsonl_out) if str(args.step_metrics_jsonl_out).strip() else save_dir / "page_sid_qcritic_step_metrics.jsonl"
    step_csv_path.parent.mkdir(parents=True, exist_ok=True)
    step_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    bundle_path = save_dir / "page_sid_qcritic_bundle.pt"
    meta_path = save_dir / "page_sid_qcritic_meta.json"
    global_step = 0
    with step_csv_path.open("w", encoding="utf-8", newline="") as step_csv_fp, step_jsonl_path.open("w", encoding="utf-8") as step_jsonl_fp:
        csv_fieldnames = [
            "event_type",
            "epoch",
            "global_step",
            "step_in_epoch",
            "steps_per_epoch",
            "loss",
            "page_loss",
            "item_loss",
            "prefix_loss",
            "rank_loss",
            "monotonic_loss",
            "q_mae",
            "q_corr",
            "q_mean",
            "target_mean",
            "q_std_mean",
            "item_delta_mae",
            "prefix_delta_mae",
            "item_rank_acc",
        ]
        step_csv_writer = csv.DictWriter(step_csv_fp, fieldnames=csv_fieldnames)
        step_csv_writer.writeheader()

        for epoch in range(1, int(args.epochs) + 1):
            train_metrics = train_one_epoch(
                tiger,
                model,
                optimizer,
                train_loader,
                device,
                args,
                epoch=int(epoch),
                global_step_start=int(global_step),
                step_csv_writer=step_csv_writer,
                step_jsonl_fp=step_jsonl_fp,
            )
            global_step += int(len(train_loader))
            valid_metrics = evaluate(tiger, model, valid_loader, device, args)
            history.append({"epoch": int(epoch), "train": train_metrics, "valid": valid_metrics})
            epoch_row = {
                "event_type": "valid_epoch",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "step_in_epoch": int(len(train_loader)),
                "steps_per_epoch": int(len(train_loader)),
                "loss": float(valid_metrics.get("loss", 0.0)),
                "page_loss": float(valid_metrics.get("page_loss", 0.0)),
                "item_loss": float(valid_metrics.get("item_loss", 0.0)),
                "prefix_loss": float(valid_metrics.get("prefix_loss", 0.0)),
                "rank_loss": float(valid_metrics.get("rank_loss", 0.0)),
                "monotonic_loss": float(valid_metrics.get("monotonic_loss", 0.0)),
                "q_mae": float(valid_metrics.get("q_mae", 0.0)),
                "q_corr": float(valid_metrics.get("q_corr", 0.0)),
                "q_mean": float(valid_metrics.get("q_mean", 0.0)),
                "target_mean": float(valid_metrics.get("target_mean", 0.0)),
                "q_std_mean": float(valid_metrics.get("q_std_mean", 0.0)),
                "item_delta_mae": float(valid_metrics.get("item_delta_mae", 0.0)),
                "prefix_delta_mae": float(valid_metrics.get("prefix_delta_mae", 0.0)),
                "item_rank_acc": float(valid_metrics.get("item_rank_acc", 0.0)),
            }
            step_csv_writer.writerow(epoch_row)
            step_jsonl_fp.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")
            if float(valid_metrics.get("loss", float("inf"))) < float(best_valid_loss):
                best_valid_loss = float(valid_metrics["loss"])
                best_epoch = int(epoch)
                meta = {
                    "method": "TIGER Page-SID Q Critic",
                    "arch": str(critic_arch),
                    "hidden_size": int(size_cfg["d_model"]),
                    "vocab_size": int(vocab_size),
                    "page_feat_dim": int(page_feat_dim),
                    "user_feat_dim": int(user_feat_dim),
                    "sid_depth": int(data_meta.get("sid_depth", 4)),
                    "item_dim": int(args.item_dim),
                    "model_dim": int(args.model_dim),
                    "num_heads": int(critic_num_heads),
                    "num_layers": int(critic_num_layers),
                    "ensemble_size": int(ensemble_size),
                    "dropout": float(args.dropout),
                    "critic_target_heuristic_mix": float(args.critic_target_heuristic_mix),
                    "critic_target_support_mix": float(args.critic_target_support_mix),
                    "critic_target_response_mix": float(args.critic_target_response_mix),
                    "critic_page_loss_scale": float(args.critic_page_loss_scale),
                    "critic_item_loss_scale": float(args.critic_item_loss_scale),
                    "critic_prefix_loss_scale": float(args.critic_prefix_loss_scale),
                    "critic_page_huber_beta": float(args.critic_page_huber_beta),
                    "critic_item_huber_beta": float(args.critic_item_huber_beta),
                    "critic_prefix_huber_beta": float(args.critic_prefix_huber_beta),
                    "critic_rank_loss_scale": float(args.critic_rank_loss_scale),
                    "critic_monotonic_loss_scale": float(args.critic_monotonic_loss_scale),
                    "critic_rank_min_gap": float(args.critic_rank_min_gap),
                    "best_epoch": int(best_epoch),
                    "best_valid_loss": float(best_valid_loss),
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                    "step_metrics_csv_path": str(step_csv_path.resolve()),
                    "step_metrics_jsonl_path": str(step_jsonl_path.resolve()),
                    "data_meta": data_meta,
                }
                save_page_sid_qcritic_bundle(bundle_path, meta_path, model, meta)

    metrics = {
        "method": "TIGER Page-SID Q Critic",
        "critic_arch": str(critic_arch),
        "trace_path": str(Path(args.trace_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "bundle_path": str(bundle_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "critic_target_heuristic_mix": float(args.critic_target_heuristic_mix),
        "critic_target_support_mix": float(args.critic_target_support_mix),
        "critic_target_response_mix": float(args.critic_target_response_mix),
        "critic_page_loss_scale": float(args.critic_page_loss_scale),
        "critic_item_loss_scale": float(args.critic_item_loss_scale),
        "critic_prefix_loss_scale": float(args.critic_prefix_loss_scale),
        "critic_page_huber_beta": float(args.critic_page_huber_beta),
        "critic_item_huber_beta": float(args.critic_item_huber_beta),
        "critic_prefix_huber_beta": float(args.critic_prefix_huber_beta),
        "critic_rank_loss_scale": float(args.critic_rank_loss_scale),
        "critic_monotonic_loss_scale": float(args.critic_monotonic_loss_scale),
        "critic_rank_min_gap": float(args.critic_rank_min_gap),
        "step_metrics_csv_path": str(step_csv_path.resolve()),
        "step_metrics_jsonl_path": str(step_jsonl_path.resolve()),
        "history": history,
        "data_meta": data_meta,
    }
    metrics_out = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "page_sid_qcritic_metrics.json"
    write_json(metrics_out, metrics)
    return metrics


def main() -> int:
    args = parse_args()
    metrics = run_training(args)
    print(
        json.dumps(
            {
                "best_epoch": int(metrics["best_epoch"]),
                "best_valid_loss": float(metrics["best_valid_loss"]),
                "bundle_path": str(metrics["bundle_path"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
