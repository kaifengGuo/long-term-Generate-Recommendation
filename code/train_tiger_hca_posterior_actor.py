import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils

from tiger_phase2_blend_common import decoder_input_ids_from_targets, infer_model_size_args, load_tiger_model, write_json
from tiger_page_sid_rl.common import iter_jsonl_records


POSTERIOR_SCORE_FIELDS = [
    "reward_model_value",
    "reward_raw",
    "reward_margin_vs_behavior",
    "page_q_value",
    "page_q_mean",
    "page_q_pess",
    "item_advantage",
    "item_advantage_mean",
    "item_advantage_pess",
    "adaptive_support_pess",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribution-aware pessimistic posterior distillation over grouped TIGER-HCA candidates."
    )
    parser.add_argument("--group_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Reference / rollout TIGER checkpoint used to build grouped candidates.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--min_candidates", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--score_field", type=str, default="reward_model_value", choices=POSTERIOR_SCORE_FIELDS)
    parser.add_argument("--prior_field", type=str, default="support_logprob_mean", choices=["none", "support_logprob_sum", "support_logprob_mean"])
    parser.add_argument("--score_scale", type=float, default=1.0)
    parser.add_argument("--prior_scale", type=float, default=0.20)
    parser.add_argument("--posterior_temperature", type=float, default=1.0)
    parser.add_argument("--teacher_logit_clip", type=float, default=20.0)
    parser.add_argument("--teacher_safe_support_gap_max", type=float, default=-1.0)
    parser.add_argument("--teacher_safe_uncertainty_ratio_max", type=float, default=0.0)
    parser.add_argument("--teacher_topk", type=int, default=0)
    parser.add_argument("--teacher_behavior_mix", type=float, default=0.0)
    parser.add_argument(
        "--reference_kl_mode",
        type=str,
        default="safe_uniform",
        choices=["teacher", "safe_uniform", "all_uniform"],
    )
    parser.add_argument("--reference_kl_scale", type=float, default=0.0)
    parser.add_argument("--score_normalization", type=str, default="mean_token", choices=["sum", "mean_token"])
    parser.add_argument("--attr_adv_mode", type=str, default="pess", choices=["raw", "pess"])
    parser.add_argument("--attr_item_scale", type=float, default=0.10)
    parser.add_argument("--attr_temperature", type=float, default=1.0)
    parser.add_argument("--attr_mix", type=float, default=0.50)
    parser.add_argument("--attr_credit_clip", type=float, default=3.0)
    parser.add_argument("--attr_renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class PosteriorGroupedDataset(Dataset):
    def __init__(self, groups: Sequence[Dict[str, Any]]):
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.groups[int(idx)]


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


def load_grouped_candidates(
    group_path: Path,
    *,
    score_field: str,
    prior_field: str,
    attr_adv_mode: str,
    max_groups: int,
    min_candidates: int,
) -> List[Dict[str, Any]]:
    groups: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    use_pess = str(attr_adv_mode).strip().lower() == "pess"
    token_key = "sid_advantage_pess" if use_pess else "sid_advantage"
    item_key = "item_advantage_pess" if use_pess else "item_advantage"
    for line_idx, payload in enumerate(iter_jsonl_records(group_path)):
        group_id = str(payload.get("group_id", payload.get("episode_id", line_idx)))
        target_tokens = [int(x) for x in payload.get("target_tokens", [])]
        token_adv = [float(x) for x in payload.get(token_key, [])]
        if not target_tokens:
            continue
        if len(token_adv) != len(target_tokens):
            token_adv = [0.0 for _ in target_tokens]
        if group_id not in groups:
            if int(max_groups) > 0 and len(groups) >= int(max_groups):
                continue
            groups[group_id] = {
                "group": group_id,
                "input_ids": [int(x) for x in payload.get("input_ids", [])],
                "attention_mask": [int(x) for x in payload.get("attention_mask", [])],
                "candidates": [],
            }
        prior_value = 0.0 if str(prior_field) == "none" else float(payload.get(str(prior_field), 0.0))
        groups[group_id]["candidates"].append(
            {
                "target_tokens": target_tokens,
                "token_adv": token_adv,
                "item_adv": float(payload.get(str(item_key), 0.0)),
                "score": float(payload.get(str(score_field), 0.0)),
                "prior": float(prior_value),
                "support_gap_scaled": float(payload.get("support_gap_scaled", 0.0)),
                "uncertainty_ratio": float(payload.get("uncertainty_ratio", 1.0)),
                "is_behavior": bool(payload.get("is_behavior", False)),
            }
        )

    rows: List[Dict[str, Any]] = []
    for group_id, payload in groups.items():
        candidates = payload.get("candidates", [])
        if len(candidates) < int(min_candidates):
            continue
        rows.append(
            {
                "group": group_id,
                "input_ids": payload["input_ids"],
                "attention_mask": payload["attention_mask"],
                "candidates": candidates,
            }
        )
    if not rows:
        raise ValueError(f"No usable grouped candidates in {group_path}")
    return rows


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_candidates = max(len(row["candidates"]) for row in batch)
    input_len = max(len(row["input_ids"]) for row in batch)
    target_len = max(len(candidate["target_tokens"]) for row in batch for candidate in row["candidates"])

    input_ids = torch.zeros((batch_size, input_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, input_len), dtype=torch.long)
    target_tokens = torch.zeros((batch_size, max_candidates, target_len), dtype=torch.long)
    token_adv = torch.zeros((batch_size, max_candidates, target_len), dtype=torch.float32)
    item_adv = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    score = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    prior = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    support_gap_scaled = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    uncertainty_ratio = torch.ones((batch_size, max_candidates), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    is_behavior = torch.zeros((batch_size, max_candidates), dtype=torch.bool)

    for row_idx, row in enumerate(batch):
        row_input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        row_attention = torch.tensor(row["attention_mask"], dtype=torch.long)
        input_ids[row_idx, : row_input_ids.shape[0]] = row_input_ids
        attention_mask[row_idx, : row_attention.shape[0]] = row_attention
        for cand_idx, candidate in enumerate(row["candidates"]):
            cand_tokens = torch.tensor(candidate["target_tokens"], dtype=torch.long)
            cand_adv = torch.tensor(candidate["token_adv"], dtype=torch.float32)
            target_tokens[row_idx, cand_idx, : cand_tokens.shape[0]] = cand_tokens
            token_adv[row_idx, cand_idx, : cand_adv.shape[0]] = cand_adv
            item_adv[row_idx, cand_idx] = float(candidate["item_adv"])
            score[row_idx, cand_idx] = float(candidate["score"])
            prior[row_idx, cand_idx] = float(candidate["prior"])
            support_gap_scaled[row_idx, cand_idx] = float(candidate.get("support_gap_scaled", 0.0))
            uncertainty_ratio[row_idx, cand_idx] = float(candidate.get("uncertainty_ratio", 1.0))
            candidate_mask[row_idx, cand_idx] = True
            is_behavior[row_idx, cand_idx] = bool(candidate["is_behavior"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_tokens": target_tokens,
        "token_adv": token_adv,
        "item_adv": item_adv,
        "score": score,
        "prior": prior,
        "support_gap_scaled": support_gap_scaled,
        "uncertainty_ratio": uncertainty_ratio,
        "candidate_mask": candidate_mask,
        "is_behavior": is_behavior,
        "groups": [row["group"] for row in batch],
    }


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=-1) * mask.float()
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def masked_uniform(mask: torch.Tensor) -> torch.Tensor:
    probs = mask.float()
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0)


def build_attr_weights(
    token_adv: torch.Tensor,
    item_adv: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    item_scale: float,
    temperature: float,
    credit_clip: float,
    renorm_mode: str,
) -> torch.Tensor:
    if float(credit_clip) > 0.0:
        token_adv = token_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        item_adv = item_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
    token_adv = renorm_signal(token_adv * valid_mask.float(), str(renorm_mode))
    item_adv = renorm_signal(item_adv.unsqueeze(-1), str(renorm_mode)).squeeze(-1)
    logits = token_adv + float(item_scale) * item_adv.unsqueeze(-1)
    logits = logits / max(abs(float(temperature)), 1e-6)
    return masked_softmax(logits, valid_mask.bool())


def sequence_stats(
    tiger,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    score_normalization: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)
    logits, _hidden = tiger.decode_with_hidden(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logp = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    valid_mask = (target_tokens > 0).float()
    seq_logp_sum = (target_logp * valid_mask).sum(dim=-1)
    if str(score_normalization) == "sum":
        seq_logp = seq_logp_sum
    else:
        seq_logp = seq_logp_sum / valid_mask.sum(dim=-1).clamp_min(1.0)
    return logits, log_probs, target_logp, valid_mask, seq_logp


def build_teacher_weights(
    *,
    score: torch.Tensor,
    prior: torch.Tensor,
    support_gap_scaled: torch.Tensor,
    uncertainty_ratio: torch.Tensor,
    candidate_mask: torch.Tensor,
    is_behavior: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_logits = float(args.score_scale) * score + float(args.prior_scale) * prior
    clip_value = abs(float(args.teacher_logit_clip))
    if clip_value > 0.0:
        teacher_logits = teacher_logits.clamp(min=-clip_value, max=clip_value)
    teacher_logits = teacher_logits / max(abs(float(args.posterior_temperature)), 1e-6)

    safe_mask = candidate_mask.bool()
    support_limit = float(args.teacher_safe_support_gap_max)
    unc_limit = float(args.teacher_safe_uncertainty_ratio_max)
    if support_limit >= 0.0:
        safe_mask = safe_mask & (support_gap_scaled <= support_limit)
    if unc_limit > 0.0:
        safe_mask = safe_mask & (uncertainty_ratio <= unc_limit)
    safe_mask = safe_mask | (candidate_mask.bool() & is_behavior.bool())
    safe_available = safe_mask.any(dim=-1, keepdim=True)
    safe_mask = torch.where(safe_available, safe_mask, candidate_mask.bool())

    teacher_mask = safe_mask
    if int(args.teacher_topk) > 0:
        masked_logits = teacher_logits.masked_fill(~teacher_mask, -1e9)
        topk = min(int(args.teacher_topk), int(masked_logits.shape[-1]))
        topk_idx = torch.topk(masked_logits, k=max(topk, 1), dim=-1).indices
        topk_mask = torch.zeros_like(teacher_mask, dtype=torch.bool)
        topk_mask.scatter_(dim=1, index=topk_idx, value=True)
        topk_mask = (topk_mask & teacher_mask) | (candidate_mask.bool() & is_behavior.bool())
        topk_available = topk_mask.any(dim=-1, keepdim=True)
        teacher_mask = torch.where(topk_available, topk_mask, teacher_mask)

    teacher_weights = masked_softmax(teacher_logits, teacher_mask)
    behavior_mix = float(min(max(float(args.teacher_behavior_mix), 0.0), 1.0))
    if behavior_mix > 0.0:
        behavior_mask = candidate_mask.bool() & is_behavior.bool()
        behavior_present = behavior_mask.any(dim=-1, keepdim=True)
        behavior_dist = masked_uniform(behavior_mask)
        mixed = (1.0 - behavior_mix) * teacher_weights + behavior_mix * behavior_dist
        teacher_weights = torch.where(behavior_present, mixed, teacher_weights)
    return teacher_logits, teacher_weights, safe_mask, teacher_mask


def candidate_sequence_kl(
    actor_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    token_kl = F.kl_div(actor_log_probs, ref_log_probs.exp(), reduction="none", log_target=False).sum(dim=-1)
    return (token_kl * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1.0)


def forward_actor(
    actor_tiger,
    ref_tiger,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_tokens = batch["target_tokens"].to(device)
    token_adv = batch["token_adv"].to(device)
    item_adv = batch["item_adv"].to(device)
    score = batch["score"].to(device)
    prior = batch["prior"].to(device)
    support_gap_scaled = batch["support_gap_scaled"].to(device)
    uncertainty_ratio = batch["uncertainty_ratio"].to(device)
    candidate_mask = batch["candidate_mask"].to(device)
    is_behavior = batch["is_behavior"].to(device)

    batch_size, max_candidates, target_len = target_tokens.shape
    flat_mask = candidate_mask.reshape(-1)
    expanded_input_ids = input_ids.unsqueeze(1).expand(-1, max_candidates, -1).reshape(batch_size * max_candidates, -1)
    expanded_attention = attention_mask.unsqueeze(1).expand(-1, max_candidates, -1).reshape(batch_size * max_candidates, -1)
    flat_targets = target_tokens.reshape(batch_size * max_candidates, target_len)

    flat_input_ids = expanded_input_ids[flat_mask]
    flat_attention = expanded_attention[flat_mask]
    flat_targets = flat_targets[flat_mask]
    flat_token_adv = token_adv.reshape(batch_size * max_candidates, target_len)[flat_mask]
    flat_item_adv = item_adv.reshape(batch_size * max_candidates)[flat_mask]

    actor_logits, actor_log_probs, flat_target_logp, flat_valid_mask, flat_plain_logp = sequence_stats(
        actor_tiger,
        flat_input_ids,
        flat_attention,
        flat_targets,
        score_normalization=str(args.score_normalization),
    )
    flat_attr_weights = build_attr_weights(
        flat_token_adv,
        flat_item_adv,
        flat_valid_mask > 0.0,
        item_scale=float(args.attr_item_scale),
        temperature=float(args.attr_temperature),
        credit_clip=float(args.attr_credit_clip),
        renorm_mode=str(args.attr_renorm_mode),
    )
    flat_attr_logp = (flat_target_logp * flat_attr_weights).sum(dim=-1)
    if str(args.score_normalization) == "sum":
        flat_attr_logp = flat_attr_logp * flat_valid_mask.sum(dim=-1).clamp_min(1.0)
    attr_mix = float(min(max(float(args.attr_mix), 0.0), 1.0))
    flat_candidate_logp = (1.0 - attr_mix) * flat_plain_logp + attr_mix * flat_attr_logp

    candidate_logp = torch.zeros((batch_size, max_candidates), dtype=torch.float32, device=device)
    candidate_logp[candidate_mask] = flat_candidate_logp

    teacher_logits, teacher_weights, safe_mask, teacher_mask = build_teacher_weights(
        score=score,
        prior=prior,
        support_gap_scaled=support_gap_scaled,
        uncertainty_ratio=uncertainty_ratio,
        candidate_mask=candidate_mask,
        is_behavior=is_behavior,
        args=args,
    )

    group_loss = -(teacher_weights * candidate_logp).sum(dim=-1)
    reference_kl_loss = group_loss.new_tensor(0.0)
    if float(args.reference_kl_scale) > 0.0:
        with torch.no_grad():
            _ref_logits, ref_log_probs, _ref_target_logp, _ref_valid_mask, _ref_seq_logp = sequence_stats(
                ref_tiger,
                flat_input_ids,
                flat_attention,
                flat_targets,
                score_normalization=str(args.score_normalization),
            )
        flat_candidate_kl = candidate_sequence_kl(actor_log_probs, ref_log_probs, flat_valid_mask)
        candidate_kl = torch.zeros((batch_size, max_candidates), dtype=torch.float32, device=device)
        candidate_kl[candidate_mask] = flat_candidate_kl
        if str(args.reference_kl_mode) == "teacher":
            kl_weights = teacher_weights
        elif str(args.reference_kl_mode) == "all_uniform":
            kl_weights = masked_uniform(candidate_mask.bool())
        else:
            kl_weights = masked_uniform(safe_mask.bool())
        reference_kl_loss = (kl_weights * candidate_kl).sum(dim=-1).mean()
    loss = group_loss.mean() + float(args.reference_kl_scale) * reference_kl_loss

    teacher_entropy = -(teacher_weights * torch.log(teacher_weights.clamp_min(1e-8))).sum(dim=-1)
    teacher_top1_prob, teacher_top1_idx = teacher_weights.max(dim=-1)
    teacher_top1_behavior = is_behavior.gather(dim=1, index=teacher_top1_idx.unsqueeze(-1)).float().squeeze(-1)
    behavior_teacher_prob = (teacher_weights * is_behavior.float()).sum(dim=-1)

    weighted_candidate_mass = teacher_weights[candidate_mask]
    weighted_attr_entropy = (flat_attr_weights * torch.log(flat_attr_weights.clamp_min(1e-8))).sum(dim=-1).neg()
    attr_entropy_mean = float((weighted_candidate_mass * weighted_attr_entropy).sum().item() / max(float(batch_size), 1.0))
    attr_active_tokens = float((weighted_candidate_mass * flat_valid_mask.sum(dim=-1)).sum().item() / max(float(batch_size), 1.0))

    stats = {
        "loss": float(loss.item()),
        "group_loss": float(group_loss.mean().item()),
        "reference_kl_loss": float(reference_kl_loss.item()),
        "expected_logp": float((teacher_weights * candidate_logp).sum(dim=-1).mean().item()),
        "plain_logp": float(flat_plain_logp.mean().item()),
        "attr_logp": float(flat_attr_logp.mean().item()),
        "teacher_entropy": float(teacher_entropy.mean().item()),
        "teacher_top1_prob": float(teacher_top1_prob.mean().item()),
        "teacher_behavior_prob": float(behavior_teacher_prob.mean().item()),
        "teacher_top1_behavior_frac": float(teacher_top1_behavior.mean().item()),
        "safe_candidate_frac": float((safe_mask.float() * candidate_mask.float()).sum().item() / candidate_mask.float().sum().clamp_min(1.0).item()),
        "teacher_candidate_frac": float((teacher_mask.float() * candidate_mask.float()).sum().item() / candidate_mask.float().sum().clamp_min(1.0).item()),
        "candidate_count": float(candidate_mask.sum(dim=-1).float().mean().item()),
        "score_mean": float(score[candidate_mask].mean().item()),
        "score_std": float(score[candidate_mask].std(unbiased=False).item()),
        "prior_mean": float(prior[candidate_mask].mean().item()),
        "teacher_logit_mean": float(teacher_logits[candidate_mask].mean().item()),
        "attr_entropy": float(attr_entropy_mean),
        "attr_active_tokens": float(attr_active_tokens),
        "attr_mix": float(attr_mix),
    }
    return loss, stats


@torch.no_grad()
def evaluate_actor(
    actor_tiger,
    ref_tiger,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    actor_tiger.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_actor(actor_tiger, ref_tiger, batch, device, args)
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    group_rows = load_grouped_candidates(
        Path(args.group_path),
        score_field=str(args.score_field),
        prior_field=str(args.prior_field),
        attr_adv_mode=str(args.attr_adv_mode),
        max_groups=int(args.max_groups),
        min_candidates=int(args.min_candidates),
    )

    size_cfg = infer_model_size_args(str(args.model_size))
    actor_init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, sid_depth, _codebook_size = load_tiger_model(
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
    ref_tiger, _ref_sid_depth, _ref_codebook_size = load_tiger_model(
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
    for param in ref_tiger.parameters():
        param.requires_grad = False
    ref_tiger.eval()

    dataset = PosteriorGroupedDataset(group_rows)
    groups = [row["group"] for row in group_rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_rows,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rows,
    )

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
        train_expected_logp: List[float] = []
        train_teacher_entropy: List[float] = []
        train_reference_kl: List[float] = []
        for batch in train_loader:
            loss, stats = forward_actor(actor_tiger, ref_tiger, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_expected_logp.append(float(stats["expected_logp"]))
            train_teacher_entropy.append(float(stats["teacher_entropy"]))
            train_reference_kl.append(float(stats["reference_kl_loss"]))

        valid_metrics = evaluate_actor(actor_tiger, ref_tiger, valid_loader, device, args)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        valid_metrics["train_expected_logp"] = float(np.mean(train_expected_logp)) if train_expected_logp else 0.0
        valid_metrics["train_teacher_entropy"] = float(np.mean(train_teacher_entropy)) if train_teacher_entropy else 0.0
        valid_metrics["train_reference_kl_loss"] = float(np.mean(train_reference_kl)) if train_reference_kl else 0.0
        history.append(dict(valid_metrics))
        if float(valid_metrics["loss"]) < float(best_key):
            best_key = float(valid_metrics["loss"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu() for key, value in actor_tiger.state_dict().items()}
            best_metrics = dict(valid_metrics)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"expected_logp={valid_metrics['expected_logp']:.4f} "
            f"teacher_top1={valid_metrics['teacher_top1_prob']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Posterior actor training produced no checkpoint.")

    save_dir = Path(args.save_dir) if str(args.save_dir).strip() else Path(args.tiger_ckpt).resolve().parent / "tiger_hca_posterior_actor"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "tiger_hca_posterior_actor_tiger.pth"
    meta_path = save_dir / "tiger_hca_posterior_actor_meta.json"
    metrics_path = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "tiger_hca_posterior_actor_metrics.json"
    torch.save(best_state, ckpt_path)

    n_candidates = sum(len(row["candidates"]) for row in group_rows)
    meta = {
        "method": "TIGER-HCA Posterior Actor",
        "group_path": str(Path(args.group_path).resolve()),
        "reference_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "train_scope": str(args.train_scope),
        "n_trainable": int(n_trainable),
        "score_field": str(args.score_field),
        "prior_field": str(args.prior_field),
        "score_scale": float(args.score_scale),
        "prior_scale": float(args.prior_scale),
        "posterior_temperature": float(args.posterior_temperature),
        "teacher_logit_clip": float(args.teacher_logit_clip),
        "teacher_safe_support_gap_max": float(args.teacher_safe_support_gap_max),
        "teacher_safe_uncertainty_ratio_max": float(args.teacher_safe_uncertainty_ratio_max),
        "teacher_topk": int(args.teacher_topk),
        "teacher_behavior_mix": float(args.teacher_behavior_mix),
        "reference_kl_mode": str(args.reference_kl_mode),
        "reference_kl_scale": float(args.reference_kl_scale),
        "score_normalization": str(args.score_normalization),
        "attr_adv_mode": str(args.attr_adv_mode),
        "attr_item_scale": float(args.attr_item_scale),
        "attr_temperature": float(args.attr_temperature),
        "attr_mix": float(args.attr_mix),
        "attr_credit_clip": float(args.attr_credit_clip),
        "attr_renorm_mode": str(args.attr_renorm_mode),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_groups": int(len(group_rows)),
        "n_candidates": int(n_candidates),
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
    print(f"[hca-post] saved fine-tuned TIGER to {ckpt_path}")
    print(f"[hca-post] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
