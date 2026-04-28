import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils

from tiger_phase2_blend_common import decoder_input_ids_from_targets, infer_model_size_args, load_tiger_model, write_json
from tiger_page_sid_rl.common import iter_jsonl_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TIGER-HCA-GRPO: grouped relative policy optimization with hierarchical token weighting."
    )
    parser.add_argument("--group_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Old-policy checkpoint used to build group relative targets.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group_adv_field", type=str, default="group_advantage")
    parser.add_argument("--token_adv_field", type=str, default="sid_advantage")
    parser.add_argument("--item_adv_field", type=str, default="item_advantage")
    parser.add_argument("--page_reward_field", type=str, default="reward_raw")
    parser.add_argument("--min_abs_group_adv", type=float, default=0.0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--item_adv_scale", type=float, default=0.10)
    parser.add_argument("--page_gate_scale", type=float, default=0.10)
    parser.add_argument("--page_gate_min", type=float, default=0.85)
    parser.add_argument("--page_gate_max", type=float, default=1.15)
    parser.add_argument(
        "--page_gate_mode",
        type=str,
        default="abs_tanh",
        choices=["abs_tanh", "signed_tanh", "positive_tanh", "none"],
    )
    parser.add_argument("--positive_topk", type=int, default=2)
    parser.add_argument("--positive_floor", type=float, default=0.0)
    parser.add_argument("--negative_topk", type=int, default=2)
    parser.add_argument("--negative_floor", type=float, default=0.0)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--clip_eps", type=float, default=0.20)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--adaptive_kl_support_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_kl_unc_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_clip_support_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_clip_unc_scale", type=float, default=0.0)
    parser.add_argument("--min_clip_eps", type=float, default=0.02)
    parser.add_argument("--trust_support_field", type=str, default="support_gap_scaled")
    parser.add_argument("--trust_unc_field", type=str, default="uncertainty_ratio")
    parser.add_argument("--entropy_scale", type=float, default=0.0)
    parser.add_argument("--sft_scale", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class HCAGRPOGroupedDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], dim=0),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], dim=0),
        "target_tokens": torch.stack([torch.tensor(x["target_tokens"], dtype=torch.long) for x in batch], dim=0),
        "token_adv": torch.stack([torch.tensor(x["token_adv"], dtype=torch.float32) for x in batch], dim=0),
        "item_adv": torch.tensor([float(x["item_adv"]) for x in batch], dtype=torch.float32),
        "group_adv": torch.tensor([float(x["group_adv"]) for x in batch], dtype=torch.float32),
        "page_reward": torch.tensor([float(x["page_reward"]) for x in batch], dtype=torch.float32),
        "trust_support": torch.tensor([float(x.get("trust_support", 0.0)) for x in batch], dtype=torch.float32),
        "trust_unc": torch.tensor([float(x.get("trust_unc", 0.0)) for x in batch], dtype=torch.float32),
        "groups": [x["group"] for x in batch],
    }


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, group in enumerate(groups):
        (valid_idx if group in valid_groups else train_idx).append(idx)
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def load_group_rows(
    group_path: Path,
    *,
    group_adv_field: str,
    token_adv_field: str,
    item_adv_field: str,
    page_reward_field: str,
    trust_support_field: str,
    trust_unc_field: str,
    min_abs_group_adv: float,
    max_rows: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for payload in iter_jsonl_records(group_path, max_rows=int(max_rows)):
        target_tokens = [int(x) for x in payload.get("target_tokens", [])]
        token_adv = [float(x) for x in payload.get(str(token_adv_field), [])]
        group_adv = float(payload.get(str(group_adv_field), 0.0))
        item_adv = float(payload.get(str(item_adv_field), 0.0))
        page_reward = float(payload.get(str(page_reward_field), payload.get("reward_raw", 0.0)))
        trust_support = float(payload.get(str(trust_support_field), 0.0))
        trust_unc = float(payload.get(str(trust_unc_field), 0.0))
        if not target_tokens or len(target_tokens) != len(token_adv):
            continue
        if abs(float(group_adv)) < float(min_abs_group_adv):
            continue
        rows.append(
            {
                "group": str(payload.get("group_id", payload.get("episode_id", "na"))),
                "input_ids": [int(x) for x in payload.get("input_ids", [])],
                "attention_mask": [int(x) for x in payload.get("attention_mask", [])],
                "target_tokens": target_tokens,
                "token_adv": token_adv,
                "item_adv": item_adv,
                "group_adv": group_adv,
                "page_reward": page_reward,
                "trust_support": trust_support,
                "trust_unc": trust_unc,
            }
        )
    if not rows:
        raise ValueError(f"No usable grouped rows in {group_path}")
    return rows


def set_train_scope(tiger, scope: str) -> int:
    for param in tiger.parameters():
        param.requires_grad = False
    if str(scope) == "full":
        for param in tiger.parameters():
            param.requires_grad = True
    elif str(scope) == "decoder_only":
        for param in tiger.model.decoder.parameters():
            param.requires_grad = True
        for param in tiger.model.lm_head.parameters():
            param.requires_grad = True
    elif str(scope) == "last_decoder_block":
        for param in tiger.model.decoder.block[-1].parameters():
            param.requires_grad = True
        for param in tiger.model.decoder.final_layer_norm.parameters():
            param.requires_grad = True
        for param in tiger.model.lm_head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported train_scope: {scope}")
    return sum(param.numel() for param in tiger.parameters() if param.requires_grad)


def renorm_signal(values: torch.Tensor, mode: str) -> torch.Tensor:
    if str(mode) == "none":
        return values
    denom = values.abs().mean().clamp_min(1e-6)
    return values / denom


def build_sparse_mask(scores: torch.Tensor, topk: int, floor: float) -> torch.Tensor:
    positive = scores > float(floor)
    if int(topk) <= 0:
        return torch.zeros_like(scores, dtype=torch.float32)
    if int(scores.shape[-1]) <= int(topk):
        return positive.float()
    idx = torch.topk(scores, k=min(int(topk), int(scores.shape[-1])), dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return (mask & positive).float()


def normalize_weights(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores * mask
    denom = masked.sum(dim=-1, keepdim=True)
    fallback = mask.sum(dim=-1, keepdim=True)
    return torch.where(
        denom > 1e-8,
        masked / denom.clamp_min(1e-8),
        torch.where(fallback > 0.0, mask / fallback.clamp_min(1.0), torch.zeros_like(mask)),
    )


def build_effective_advantages(
    token_adv: torch.Tensor,
    item_adv: torch.Tensor,
    group_adv: torch.Tensor,
    page_reward: torch.Tensor,
    *,
    item_adv_scale: float,
    page_gate_scale: float,
    page_gate_min: float,
    page_gate_max: float,
    page_gate_mode: str,
    positive_topk: int,
    positive_floor: float,
    negative_topk: int,
    negative_floor: float,
    credit_clip: float,
    renorm_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(credit_clip) > 0.0:
        token_adv = token_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        item_adv = item_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        group_adv = group_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        page_reward = page_reward.clamp(min=-float(credit_clip), max=float(credit_clip))

    token_adv = renorm_signal(token_adv, str(renorm_mode))
    item_adv = renorm_signal(item_adv.unsqueeze(-1), str(renorm_mode)).squeeze(-1)
    group_adv = renorm_signal(group_adv.unsqueeze(-1), str(renorm_mode)).squeeze(-1)
    page_reward = renorm_signal(page_reward.unsqueeze(-1), str(renorm_mode)).squeeze(-1)

    pos_scores = torch.relu(token_adv) + float(item_adv_scale) * torch.relu(item_adv).unsqueeze(-1)
    neg_scores = torch.relu(-token_adv) + float(item_adv_scale) * torch.relu(-item_adv).unsqueeze(-1)
    pos_mask = build_sparse_mask(pos_scores, int(positive_topk), float(positive_floor))
    neg_mask = build_sparse_mask(neg_scores, int(negative_topk), float(negative_floor))

    use_pos = group_adv >= 0.0
    weight_scores = torch.where(use_pos.unsqueeze(-1), pos_scores, neg_scores)
    weight_mask = torch.where(use_pos.unsqueeze(-1), pos_mask, neg_mask)
    abs_scores = token_adv.abs() + float(item_adv_scale) * item_adv.abs().unsqueeze(-1)
    missing = weight_mask.sum(dim=-1) <= 0
    if bool(missing.any()):
        fallback_idx = abs_scores[missing].argmax(dim=-1, keepdim=True)
        weight_mask[missing] = 0.0
        weight_mask[missing].scatter_(1, fallback_idx, 1.0)
        weight_scores[missing] = abs_scores[missing]

    weights = normalize_weights(weight_scores.clamp_min(0.0), weight_mask)
    gate_mode = str(page_gate_mode).strip().lower()
    if gate_mode == "none":
        gate_signal = torch.zeros_like(page_reward)
    elif gate_mode == "signed_tanh":
        gate_signal = torch.tanh(page_reward)
    elif gate_mode == "positive_tanh":
        gate_signal = torch.tanh(torch.relu(page_reward))
    else:
        gate_signal = torch.tanh(page_reward.abs())
    page_gate = 1.0 + float(page_gate_scale) * gate_signal
    page_gate = page_gate.clamp(min=float(page_gate_min), max=float(page_gate_max)).unsqueeze(-1)
    effective_adv = group_adv.unsqueeze(-1) * page_gate * weights
    return effective_adv, weights, pos_mask, neg_mask, page_gate


def compute_grpo_loss(
    actor_logits: torch.Tensor,
    old_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    token_adv: torch.Tensor,
    item_adv: torch.Tensor,
    group_adv: torch.Tensor,
    page_reward: torch.Tensor,
    trust_support: torch.Tensor,
    trust_unc: torch.Tensor,
    *,
    item_adv_scale: float,
    page_gate_scale: float,
    page_gate_min: float,
    page_gate_max: float,
    page_gate_mode: str,
    positive_topk: int,
    positive_floor: float,
    negative_topk: int,
    negative_floor: float,
    credit_clip: float,
    renorm_mode: str,
    clip_eps: float,
    kl_scale: float,
    adaptive_kl_support_scale: float,
    adaptive_kl_unc_scale: float,
    adaptive_clip_support_scale: float,
    adaptive_clip_unc_scale: float,
    min_clip_eps: float,
    entropy_scale: float,
    sft_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    actor_log_probs = torch.log_softmax(actor_logits, dim=-1)
    old_log_probs = torch.log_softmax(old_logits.detach(), dim=-1)
    actor_target_logp = actor_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    old_target_logp = old_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

    effective_adv, weights, pos_mask, neg_mask, page_gate = build_effective_advantages(
        token_adv,
        item_adv,
        group_adv,
        page_reward,
        item_adv_scale=float(item_adv_scale),
        page_gate_scale=float(page_gate_scale),
        page_gate_min=float(page_gate_min),
        page_gate_max=float(page_gate_max),
        page_gate_mode=str(page_gate_mode),
        positive_topk=int(positive_topk),
        positive_floor=float(positive_floor),
        negative_topk=int(negative_topk),
        negative_floor=float(negative_floor),
        credit_clip=float(credit_clip),
        renorm_mode=str(renorm_mode),
    )
    active_mask = (weights > 0.0).float()
    trust_support = trust_support.clamp_min(0.0).unsqueeze(-1)
    trust_unc = trust_unc.clamp_min(0.0).unsqueeze(-1)
    trust_multiplier = (
        1.0
        + float(adaptive_kl_support_scale) * trust_support
        + float(adaptive_kl_unc_scale) * trust_unc
    ).clamp_min(1e-6)
    clip_multiplier = (
        1.0
        + float(adaptive_clip_support_scale) * trust_support
        + float(adaptive_clip_unc_scale) * trust_unc
    ).clamp_min(1e-6)
    effective_clip_eps = (float(clip_eps) / clip_multiplier).clamp_min(float(min_clip_eps))

    ratio = torch.exp(actor_target_logp - old_target_logp)
    clipped_ratio = torch.minimum(
        torch.maximum(ratio, 1.0 - effective_clip_eps),
        1.0 + effective_clip_eps,
    )
    surrogate1 = ratio * effective_adv
    surrogate2 = clipped_ratio * effective_adv
    policy_obj = torch.minimum(surrogate1, surrogate2)
    policy_loss = -(policy_obj * active_mask).sum() / active_mask.sum().clamp_min(1e-8)

    old_probs = old_log_probs.exp()
    token_kl = (old_probs * (old_log_probs - actor_log_probs)).sum(dim=-1)
    kl_loss = (token_kl * active_mask * trust_multiplier).sum() / active_mask.sum().clamp_min(1e-8)
    entropy = -(actor_log_probs.exp() * actor_log_probs).sum(dim=-1)
    entropy_bonus = (entropy * active_mask).sum() / active_mask.sum().clamp_min(1e-8)
    ce_loss = F.cross_entropy(
        actor_logits.reshape(-1, actor_logits.shape[-1]),
        target_tokens.reshape(-1),
        reduction="mean",
    )
    loss = policy_loss + float(kl_scale) * kl_loss - float(entropy_scale) * entropy_bonus + float(sft_scale) * ce_loss

    clipped = ((ratio - clipped_ratio).abs() > 1e-8).float()
    stats = {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "entropy_bonus": float(entropy_bonus.item()),
        "sft_loss": float(ce_loss.item()),
        "target_gain": float(((actor_target_logp.exp() - old_target_logp.exp()) * active_mask).sum().item() / active_mask.sum().item()),
        "approx_kl": float(((old_target_logp - actor_target_logp) * active_mask).sum().item() / active_mask.sum().item()),
        "clip_frac": float((clipped * active_mask).sum().item() / active_mask.sum().item()),
        "ratio_mean": float((ratio * active_mask).sum().item() / active_mask.sum().item()),
        "clip_eps_mean": float((effective_clip_eps * active_mask).sum().item() / active_mask.sum().item()),
        "kl_multiplier_mean": float((trust_multiplier * active_mask).sum().item() / active_mask.sum().item()),
        "trust_support_mean": float((trust_support * active_mask).sum().item() / active_mask.sum().item()),
        "trust_unc_mean": float((trust_unc * active_mask).sum().item() / active_mask.sum().item()),
        "signed_adv_mean": float((effective_adv * active_mask).sum().item() / active_mask.sum().item()),
        "signed_adv_abs": float((effective_adv.abs() * active_mask).sum().item() / active_mask.sum().item()),
        "page_gate_mean": float(page_gate.mean().item()),
        "active_frac": float(active_mask.mean().item()),
        "pos_active_frac": float(pos_mask.mean().item()),
        "neg_active_frac": float(neg_mask.mean().item()),
        "pos_selected_per_row": float(pos_mask.sum(dim=-1).mean().item()),
        "neg_selected_per_row": float(neg_mask.sum(dim=-1).mean().item()),
        "group_adv_mean": float(group_adv.mean().item()),
        "group_adv_abs": float(group_adv.abs().mean().item()),
        "token_weight_mean": float(weights.mean().item()),
        "page_reward_mean": float(page_reward.mean().item()),
    }
    return loss, stats


def forward_actor(
    actor_tiger,
    old_tiger,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_tokens = batch["target_tokens"].to(device)
    token_adv = batch["token_adv"].to(device)
    item_adv = batch["item_adv"].to(device)
    group_adv = batch["group_adv"].to(device)
    page_reward = batch["page_reward"].to(device)
    trust_support = batch["trust_support"].to(device)
    trust_unc = batch["trust_unc"].to(device)
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)

    with torch.no_grad():
        old_logits, _ = old_tiger.decode_with_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    actor_logits, _ = actor_tiger.decode_with_hidden(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    return compute_grpo_loss(
        actor_logits,
        old_logits,
        target_tokens,
        token_adv,
        item_adv,
        group_adv,
        page_reward,
        trust_support,
        trust_unc,
        item_adv_scale=float(args.item_adv_scale),
        page_gate_scale=float(args.page_gate_scale),
        page_gate_min=float(args.page_gate_min),
        page_gate_max=float(args.page_gate_max),
        page_gate_mode=str(args.page_gate_mode),
        positive_topk=int(args.positive_topk),
        positive_floor=float(args.positive_floor),
        negative_topk=int(args.negative_topk),
        negative_floor=float(args.negative_floor),
        credit_clip=float(args.credit_clip),
        renorm_mode=str(args.renorm_mode),
        clip_eps=float(args.clip_eps),
        kl_scale=float(args.kl_scale),
        adaptive_kl_support_scale=float(getattr(args, "adaptive_kl_support_scale", 0.0)),
        adaptive_kl_unc_scale=float(getattr(args, "adaptive_kl_unc_scale", 0.0)),
        adaptive_clip_support_scale=float(getattr(args, "adaptive_clip_support_scale", 0.0)),
        adaptive_clip_unc_scale=float(getattr(args, "adaptive_clip_unc_scale", 0.0)),
        min_clip_eps=float(getattr(args, "min_clip_eps", 0.02)),
        entropy_scale=float(args.entropy_scale),
        sft_scale=float(args.sft_scale),
    )


@torch.no_grad()
def evaluate_actor(
    actor_tiger,
    old_tiger,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    actor_tiger.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_actor(actor_tiger, old_tiger, batch, device, args)
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    grouped_rows = load_group_rows(
        Path(args.group_path),
        group_adv_field=str(args.group_adv_field),
        token_adv_field=str(args.token_adv_field),
        item_adv_field=str(args.item_adv_field),
        page_reward_field=str(args.page_reward_field),
        trust_support_field=str(args.trust_support_field),
        trust_unc_field=str(args.trust_unc_field),
        min_abs_group_adv=float(args.min_abs_group_adv),
        max_rows=int(args.max_rows),
    )

    size_cfg = infer_model_size_args(str(args.model_size))
    old_tiger, sid_depth, _codebook_size = load_tiger_model(
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
    for param in old_tiger.parameters():
        param.requires_grad = False
    old_tiger.eval()

    actor_init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, _sid_depth2, _codebook_size2 = load_tiger_model(
        tiger_ckpt=str(actor_init_ckpt),
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
    n_trainable = set_train_scope(actor_tiger, str(args.train_scope))

    dataset = HCAGRPOGroupedDataset(grouped_rows)
    groups = [row["group"] for row in grouped_rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_rows)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_rows)

    params = [param for param in actor_tiger.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_epoch = 0
    best_key = float("inf")
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        actor_tiger.train()
        train_losses: List[float] = []
        train_target_gain: List[float] = []
        train_clip_frac: List[float] = []
        for batch in train_loader:
            loss, stats = forward_actor(actor_tiger, old_tiger, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_target_gain.append(float(stats["target_gain"]))
            train_clip_frac.append(float(stats["clip_frac"]))

        valid_metrics = evaluate_actor(actor_tiger, old_tiger, valid_loader, device, args)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        valid_metrics["train_target_gain"] = float(np.mean(train_target_gain)) if train_target_gain else 0.0
        valid_metrics["train_clip_frac"] = float(np.mean(train_clip_frac)) if train_clip_frac else 0.0
        history.append(dict(valid_metrics))
        if float(valid_metrics["loss"]) < float(best_key):
            best_key = float(valid_metrics["loss"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu() for key, value in actor_tiger.state_dict().items()}
            best_metrics = dict(valid_metrics)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"target_gain={valid_metrics['target_gain']:.4f} "
            f"approx_kl={valid_metrics['approx_kl']:.4f} "
            f"clip_frac={valid_metrics['clip_frac']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("TIGER-HCA-GRPO training produced no checkpoint.")

    save_dir = Path(args.save_dir) if str(args.save_dir).strip() else Path(args.tiger_ckpt).resolve().parent / "tiger_hca_grpo_actor"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "tiger_hca_grpo_actor_tiger.pth"
    meta_path = save_dir / "tiger_hca_grpo_actor_meta.json"
    metrics_path = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "tiger_hca_grpo_actor_metrics.json"
    torch.save(best_state, ckpt_path)
    meta = {
        "method": "TIGER-HCA-GRPO Actor",
        "group_path": str(Path(args.group_path).resolve()),
        "old_policy_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "train_scope": str(args.train_scope),
        "n_trainable": int(n_trainable),
        "group_adv_field": str(args.group_adv_field),
        "token_adv_field": str(args.token_adv_field),
        "item_adv_field": str(args.item_adv_field),
        "page_reward_field": str(args.page_reward_field),
        "page_gate_mode": str(args.page_gate_mode),
        "clip_eps": float(args.clip_eps),
        "kl_scale": float(args.kl_scale),
        "adaptive_kl_support_scale": float(args.adaptive_kl_support_scale),
        "adaptive_kl_unc_scale": float(args.adaptive_kl_unc_scale),
        "adaptive_clip_support_scale": float(args.adaptive_clip_support_scale),
        "adaptive_clip_unc_scale": float(args.adaptive_clip_unc_scale),
        "min_clip_eps": float(args.min_clip_eps),
        "trust_support_field": str(args.trust_support_field),
        "trust_unc_field": str(args.trust_unc_field),
        "entropy_scale": float(args.entropy_scale),
        "sft_scale": float(args.sft_scale),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_rows": int(len(grouped_rows)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
    }
    write_json(meta_path, meta)
    write_json(
        metrics_path,
        {
            "tiger_ckpt": str(ckpt_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "history": history,
        },
    )
    print(f"[hca-grpo] saved fine-tuned TIGER to {ckpt_path}")
    print(f"[hca-grpo] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
