import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_phase2_blend_common import (  # noqa: E402
    build_iid2sid_tokens,
    build_sid_prefix_to_next,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_tiger_model,
)
from tiger_slate_online_common import build_online_slate_inputs, load_online_slate_allocator  # noqa: E402

from tiger_hcla_rl.common import (  # noqa: E402
    build_sparse_mask,
    calibrate_token_delta,
    load_reader_from_uirm_log,
    pooled_history_summary,
    ppo_clipped_surrogate,
    prefix_to_delta,
    renorm_signal,
    set_random_seed,
    set_train_scope,
    split_groups,
    write_json,
)
from tiger_hcla_rl.models import load_critic_bundle  # noqa: E402


class HCLAActorDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TIGER with full HCLA-RL actor updates.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Old policy checkpoint for PPO-style reference.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, default="")
    parser.add_argument("--init_tiger_ckpt", type=str, default="", help="Actor init checkpoint. Defaults to --tiger_ckpt.")
    parser.add_argument("--critic_bundle_path", type=str, default="")
    parser.add_argument("--critic_meta_path", type=str, default="")
    parser.add_argument("--item_allocator_head_path", type=str, default="")
    parser.add_argument("--item_allocator_meta_path", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--adv_blend_alpha", type=float, default=0.50)
    parser.add_argument("--item_share_blend_alpha", type=float, default=0.40)
    parser.add_argument("--item_adv_scale", type=float, default=0.15)
    parser.add_argument("--page_adv_scale", type=float, default=0.15)
    parser.add_argument("--hazard_gate_scale", type=float, default=0.10)
    parser.add_argument("--hazard_gate_min", type=float, default=0.85)
    parser.add_argument("--hazard_gate_max", type=float, default=1.15)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--positive_topk", type=int, default=2)
    parser.add_argument("--positive_floor", type=float, default=0.0)
    parser.add_argument("--negative_topk", type=int, default=1)
    parser.add_argument("--negative_floor", type=float, default=0.0)
    parser.add_argument("--clip_eps", type=float, default=0.20)
    parser.add_argument("--ppo_loss_scale", type=float, default=1.00)
    parser.add_argument("--sibling_loss_scale", type=float, default=0.35)
    parser.add_argument("--sibling_adv_beta", type=float, default=0.75)
    parser.add_argument("--sibling_temperature", type=float, default=1.00)
    parser.add_argument("--sibling_score_clip", type=float, default=6.0)
    parser.add_argument("--sibling_center_mode", type=str, default="old_policy", choices=["old_policy", "mean"])
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--entropy_scale", type=float, default=0.0)
    parser.add_argument("--sft_scale", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


def load_chain_rows(chain_path: Path, max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with chain_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
            token_adv = [float(x) for x in payload.get("lt_token_cf_adv", [])]
            if len(target_tokens) <= 0 or len(target_tokens) != len(token_adv):
                continue
            rows.append(
                {
                    "group": str(payload.get("episode_id", "na")),
                    "page_group": f"{int(payload.get('episode_id', -1))}:{int(payload.get('page_index', -1))}",
                    "input_ids": [int(x) for x in payload.get("input_ids", [])],
                    "attention_mask": [int(x) for x in payload.get("attention_mask", [])],
                    "target_tokens": target_tokens,
                    "page_features": [float(x) for x in payload.get("page_features", [])],
                    "item_features": [float(x) for x in payload.get("item_features", [])],
                    "page_baseline": float(payload.get("lt_page_baseline", 0.0)),
                    "page_adv": float(payload.get("lt_page_adv", 0.0)),
                    "item_adv": float(payload.get("lt_item_cf_adv", 0.0)),
                    "item_share": float(payload.get("lt_item_share", 0.0)),
                    "token_adv": token_adv,
                    "hazard": float(payload.get("lt_page_hazard", 0.0)),
                    "history_items": [int(x) for x in payload.get("history_items", [])],
                    "slate_item_index": int(payload.get("slate_item_index", 0)),
                    "selected_item_ids": [int(x) for x in payload.get("selected_item_ids", [])],
                    "selected_sid_tokens_list": [[int(v) for v in seq] for seq in payload.get("selected_sid_tokens_list", [])],
                    "selected_item_credit_shares": [float(x) for x in payload.get("selected_item_credit_shares", [])],
                }
            )
    if not rows:
        raise ValueError(f"No usable rows in {chain_path}")
    return rows


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], dim=0),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], dim=0),
        "target_tokens": torch.stack([torch.tensor(x["target_tokens"], dtype=torch.long) for x in batch], dim=0),
        "page_features": torch.stack([torch.tensor(x["page_features"], dtype=torch.float32) for x in batch], dim=0),
        "item_features": torch.stack([torch.tensor(x["item_features"], dtype=torch.float32) for x in batch], dim=0),
        "page_baseline": torch.tensor([float(x["page_baseline"]) for x in batch], dtype=torch.float32),
        "page_adv": torch.tensor([float(x["page_adv"]) for x in batch], dtype=torch.float32),
        "item_adv": torch.tensor([float(x["item_adv"]) for x in batch], dtype=torch.float32),
        "item_share": torch.tensor([float(x["item_share"]) for x in batch], dtype=torch.float32),
        "token_adv": torch.stack([torch.tensor(x["token_adv"], dtype=torch.float32) for x in batch], dim=0),
        "hazard": torch.tensor([float(x["hazard"]) for x in batch], dtype=torch.float32),
        "slate_item_index": torch.tensor([int(x["slate_item_index"]) for x in batch], dtype=torch.long),
        "groups": [x["group"] for x in batch],
        "page_groups": [x["page_group"] for x in batch],
        "history_items_list": [x["history_items"] for x in batch],
        "selected_item_ids_list": [x["selected_item_ids"] for x in batch],
        "selected_sid_tokens_list": [x["selected_sid_tokens_list"] for x in batch],
        "selected_item_credit_shares_list": [x["selected_item_credit_shares"] for x in batch],
    }


def build_allowed_token_mask(
    target_tokens: torch.Tensor,
    sid_prefix_to_next: Dict[Tuple[int, ...], List[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    bsz, length = target_tokens.shape
    mask = torch.zeros((bsz, length, int(vocab_size)), dtype=torch.bool, device=device)
    for row_idx in range(bsz):
        prefix: List[int] = []
        for pos in range(length):
            tok = int(target_tokens[row_idx, pos].item())
            allowed = list(sid_prefix_to_next.get(tuple(prefix), []))
            if tok not in allowed:
                allowed.append(tok)
            if not allowed:
                allowed = [tok]
            mask[row_idx, pos, allowed] = True
            prefix.append(tok)
    return mask


@torch.no_grad()
def predict_item_shares(
    *,
    allocator_head,
    batch: Dict[str, Any],
    iid2sid_tok_cpu: torch.Tensor,
    token_vocab_size: int,
    max_hist_items: int,
    device: torch.device,
) -> torch.Tensor:
    raw_share = batch["item_share"].to(device)
    item_features: List[torch.Tensor] = []
    page_features: List[torch.Tensor] = []
    item_indices: List[int] = []
    valid_rows: List[int] = []

    for row_idx, item_ids in enumerate(batch["selected_item_ids_list"]):
        if not item_ids:
            continue
        item_pos = int(batch["slate_item_index"][row_idx].item())
        if item_pos < 0 or item_pos >= len(item_ids):
            continue
        online = build_online_slate_inputs(
            history_items=batch["history_items_list"][row_idx],
            candidate_item_ids=item_ids,
            candidate_sid_tokens_list=batch["selected_sid_tokens_list"][row_idx],
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            max_hist_items=int(max_hist_items),
            token_vocab_size=int(token_vocab_size),
            base_scores=None,
        )
        if int(online["item_features"].shape[0]) <= item_pos:
            continue
        item_features.append(torch.tensor(online["item_features"], dtype=torch.float32))
        page_features.append(torch.tensor(online["page_features"], dtype=torch.float32))
        item_indices.append(int(item_pos))
        valid_rows.append(int(row_idx))

    if not valid_rows:
        return raw_share

    max_items = max(int(x.shape[0]) for x in item_features)
    item_dim = int(item_features[0].shape[1])
    page_dim = int(page_features[0].shape[0])
    item_tensor = torch.zeros((len(valid_rows), max_items, item_dim), dtype=torch.float32, device=device)
    page_tensor = torch.zeros((len(valid_rows), page_dim), dtype=torch.float32, device=device)
    mask_tensor = torch.zeros((len(valid_rows), max_items), dtype=torch.bool, device=device)
    gather_index = torch.tensor(item_indices, dtype=torch.long, device=device)
    for local_idx in range(len(valid_rows)):
        n_items = int(item_features[local_idx].shape[0])
        item_tensor[local_idx, :n_items] = item_features[local_idx].to(device)
        page_tensor[local_idx] = page_features[local_idx].to(device)
        mask_tensor[local_idx, :n_items] = True

    pred = allocator_head.predict_shares(item_tensor, page_tensor, mask=mask_tensor)
    gathered = pred.gather(dim=-1, index=gather_index.unsqueeze(-1)).squeeze(-1)
    out = raw_share.clone()
    for local_idx, row_idx in enumerate(valid_rows):
        out[row_idx] = gathered[local_idx]
    return out


def prepare_advantages(
    *,
    actor_tiger,
    critic_bundle: Tuple[Any, Any, Any, Dict[str, Any]] | None,
    item_allocator_bundle: Tuple[Any, torch.Tensor, int] | None,
    batch: Dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_page_adv = batch["page_adv"].to(device)
    raw_item_adv = batch["item_adv"].to(device)
    raw_item_share = batch["item_share"].to(device)
    raw_token_adv = batch["token_adv"].to(device)
    if critic_bundle is None or float(args.adv_blend_alpha) <= 0.0:
        page_adv = raw_page_adv
        token_adv = raw_token_adv
        item_value_pred = raw_item_adv
    else:
        page_head, item_head, token_head, _meta = critic_bundle
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        page_features = batch["page_features"].to(device)
        item_features = batch["item_features"].to(device)
        page_baseline = batch["page_baseline"].to(device)

        with torch.no_grad():
            summary = pooled_history_summary(actor_tiger, input_ids, attention_mask)
            decoder_input_ids = decoder_input_ids_from_targets(target_tokens)
            _logits, hidden = actor_tiger.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            page_value_pred = page_head(summary, page_features)
            page_adv_pred = page_value_pred - page_baseline
            item_value_pred = item_head(summary, page_features, item_features)
            token_prefix_pred = token_head(hidden, target_tokens)
            token_adv_pred = prefix_to_delta(token_prefix_pred)
            token_adv_pred = calibrate_token_delta(token_adv_pred, item_value_pred)

        blend = float(np.clip(float(args.adv_blend_alpha), 0.0, 1.0))
        page_adv = (1.0 - blend) * raw_page_adv + blend * page_adv_pred
        token_adv = (1.0 - blend) * raw_token_adv + blend * token_adv_pred

    if item_allocator_bundle is not None:
        allocator_head, iid2sid_tok_cpu, token_vocab_size = item_allocator_bundle
        share_pred = predict_item_shares(
            allocator_head=allocator_head,
            batch=batch,
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            token_vocab_size=int(token_vocab_size),
            max_hist_items=int(args.max_hist_items),
            device=device,
        )
        share_blend = (1.0 - float(args.item_share_blend_alpha)) * raw_item_share + float(args.item_share_blend_alpha) * share_pred
    else:
        share_blend = raw_item_share

    if critic_bundle is None or float(args.adv_blend_alpha) <= 0.0:
        item_adv = page_adv * share_blend
    else:
        item_adv_from_share = page_adv * share_blend
        blend = float(np.clip(float(args.adv_blend_alpha), 0.0, 1.0))
        item_adv = (1.0 - blend) * item_adv_from_share + blend * item_value_pred
    return page_adv, item_adv, token_adv, share_blend


def compute_sibling_loss(
    *,
    actor_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    target_tokens: torch.Tensor,
    token_scores_all: torch.Tensor,
    allowed_mask: torch.Tensor,
    token_adv: torch.Tensor,
    item_adv: torch.Tensor,
    page_adv: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if float(args.sibling_loss_scale) <= 0.0:
        zero = actor_log_probs.new_zeros(())
        return zero, {
            "sibling_loss": 0.0,
            "choice_frac": 0.0,
            "chosen_target_prob": 0.0,
            "target_entropy": 0.0,
        }

    mag = (
        token_adv.abs()
        + float(args.item_adv_scale) * item_adv.abs().unsqueeze(-1)
        + float(args.page_adv_scale) * page_adv.abs().unsqueeze(-1)
    )
    mag = renorm_signal(mag, str(args.renorm_mode))
    choice_mask = (allowed_mask.sum(dim=-1) > 1).float()

    masked_scores = token_scores_all.masked_fill(~allowed_mask, 0.0)
    old_allowed_log_probs = old_log_probs.masked_fill(~allowed_mask, -1e9)
    old_allowed_probs = torch.softmax(old_allowed_log_probs, dim=-1)
    if str(args.sibling_center_mode) == "mean":
        denom = allowed_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        baseline = masked_scores.sum(dim=-1, keepdim=True) / denom
    else:
        baseline = (old_allowed_probs * masked_scores).sum(dim=-1, keepdim=True)
    adv_scores = masked_scores - baseline
    if float(args.sibling_score_clip) > 0.0:
        adv_scores = adv_scores.clamp(min=-float(args.sibling_score_clip), max=float(args.sibling_score_clip))

    scale = float(args.sibling_adv_beta) / max(float(args.sibling_temperature), 1e-6)
    target_logits = old_allowed_log_probs + scale * adv_scores
    target_logits = target_logits.masked_fill(~allowed_mask, -1e9)
    target_probs = torch.softmax(target_logits, dim=-1)

    actor_allowed_log_probs = actor_log_probs.masked_fill(~allowed_mask, -1e9)
    actor_allowed_log_probs = actor_allowed_log_probs - torch.logsumexp(actor_allowed_log_probs, dim=-1, keepdim=True)
    per_pos_ce = -(target_probs * actor_allowed_log_probs).sum(dim=-1)

    weight = choice_mask * mag
    denom = weight.sum()
    if float(denom.item()) <= 1e-8:
        sibling_loss = actor_log_probs.new_zeros(())
    else:
        sibling_loss = (weight * per_pos_ce).sum() / denom

    chosen_target_prob = target_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    target_entropy = -(target_probs * torch.log(target_probs.clamp_min(1e-8))).sum(dim=-1)
    stats = {
        "sibling_loss": float(sibling_loss.item()),
        "choice_frac": float(choice_mask.mean().item()),
        "chosen_target_prob": float(chosen_target_prob.mean().item()),
        "target_entropy": float(target_entropy.mean().item()),
    }
    return sibling_loss, stats


def forward_batch(
    actor_tiger,
    old_tiger,
    critic_bundle,
    item_allocator_bundle,
    sid_prefix_to_next: Dict[Tuple[int, ...], List[int]],
    batch: Dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_tokens = batch["target_tokens"].to(device)
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)

    page_adv, item_adv, token_adv, item_share = prepare_advantages(
        actor_tiger=actor_tiger,
        critic_bundle=critic_bundle,
        item_allocator_bundle=item_allocator_bundle,
        batch=batch,
        device=device,
        args=args,
    )

    if float(args.credit_clip) > 0.0:
        page_adv = page_adv.clamp(min=-float(args.credit_clip), max=float(args.credit_clip))
        item_adv = item_adv.clamp(min=-float(args.credit_clip), max=float(args.credit_clip))
        token_adv = token_adv.clamp(min=-float(args.credit_clip), max=float(args.credit_clip))

    page_adv = renorm_signal(page_adv, str(args.renorm_mode))
    item_adv = renorm_signal(item_adv, str(args.renorm_mode))
    token_adv = renorm_signal(token_adv, str(args.renorm_mode))

    with torch.no_grad():
        old_logits, old_hidden = old_tiger.decode_with_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    actor_logits, _hidden = actor_tiger.decode_with_hidden(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    actor_log_probs = torch.log_softmax(actor_logits, dim=-1)
    old_log_probs = torch.log_softmax(old_logits.detach(), dim=-1)
    actor_target_logp = actor_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    old_target_logp = old_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(actor_target_logp - old_target_logp)

    hazard = batch["hazard"].to(device)
    page_gate = 1.0 + float(args.hazard_gate_scale) * torch.tanh(page_adv - hazard)
    page_gate = page_gate.clamp(min=float(args.hazard_gate_min), max=float(args.hazard_gate_max)).unsqueeze(-1)
    signed_adv = page_gate * (
        token_adv
        + float(args.item_adv_scale) * item_adv.unsqueeze(-1)
        + float(args.page_adv_scale) * page_adv.unsqueeze(-1)
    )
    signed_adv = renorm_signal(signed_adv, str(args.renorm_mode))

    pos_mask = build_sparse_mask(torch.relu(signed_adv), int(args.positive_topk), float(args.positive_floor))
    neg_mask = build_sparse_mask(torch.relu(-signed_adv), int(args.negative_topk), float(args.negative_floor))
    active_mask = (pos_mask + neg_mask).clamp(max=1.0)
    missing = active_mask.sum(dim=-1) <= 0
    if bool(missing.any()):
        fallback_idx = signed_adv[missing].abs().argmax(dim=-1, keepdim=True)
        fallback_mask = torch.zeros_like(signed_adv[missing], dtype=torch.float32)
        fallback_mask.scatter_(1, fallback_idx, 1.0)
        active_mask[missing] = fallback_mask

    surrogate = ppo_clipped_surrogate(ratio, signed_adv, float(args.clip_eps))
    pg_loss = -(active_mask * surrogate).sum() / (active_mask.sum() + 1e-8)
    sibling_loss = actor_log_probs.new_zeros(())
    sibling_stats = {
        "sibling_loss": 0.0,
        "choice_frac": 0.0,
        "chosen_target_prob": 0.0,
        "target_entropy": 0.0,
    }
    if critic_bundle is not None and float(args.sibling_loss_scale) > 0.0:
        _page_head, _item_head, token_head, _meta = critic_bundle
        vocab_size = int(actor_logits.shape[-1])
        allowed_mask = build_allowed_token_mask(target_tokens, sid_prefix_to_next, vocab_size, device)
        with torch.no_grad():
            token_scores_all = token_head.score_all_tokens(old_hidden.reshape(-1, old_hidden.shape[-1]))
            if int(token_scores_all.shape[-1]) < vocab_size:
                pad = torch.zeros(
                    (token_scores_all.shape[0], vocab_size - int(token_scores_all.shape[-1])),
                    dtype=token_scores_all.dtype,
                    device=token_scores_all.device,
                )
                token_scores_all = torch.cat([token_scores_all, pad], dim=-1)
            elif int(token_scores_all.shape[-1]) > vocab_size:
                token_scores_all = token_scores_all[:, :vocab_size]
            token_scores_all = token_scores_all.view(old_hidden.shape[0], old_hidden.shape[1], vocab_size)
        sibling_loss, sibling_stats = compute_sibling_loss(
            actor_log_probs=actor_log_probs,
            old_log_probs=old_log_probs,
            target_tokens=target_tokens,
            token_scores_all=token_scores_all,
            allowed_mask=allowed_mask,
            token_adv=token_adv,
            item_adv=item_adv,
            page_adv=page_adv,
            args=args,
        )
    kl_loss = F.kl_div(actor_log_probs, old_log_probs.exp(), reduction="batchmean", log_target=False)
    entropy = -(actor_log_probs.exp() * actor_log_probs).sum(dim=-1).mean()
    ce = F.cross_entropy(actor_logits.reshape(-1, actor_logits.shape[-1]), target_tokens.reshape(-1), reduction="none").view_as(target_tokens)
    pos_weight = torch.relu(signed_adv)
    sft_loss = (pos_weight * ce).sum() / (pos_weight.sum() + 1e-8)

    loss = (
        float(args.ppo_loss_scale) * pg_loss
        + float(args.sibling_loss_scale) * sibling_loss
        + float(args.kl_scale) * kl_loss
        - float(args.entropy_scale) * entropy
        + float(args.sft_scale) * sft_loss
    )
    stats = {
        "loss": float(loss.item()),
        "pg_loss": float(pg_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "entropy": float(entropy.item()),
        "sft_loss": float(sft_loss.item()),
        "target_prob": float(actor_target_logp.exp().mean().item()),
        "old_target_prob": float(old_target_logp.exp().mean().item()),
        "target_gain": float((actor_target_logp.exp() - old_target_logp.exp()).mean().item()),
        "signed_adv_mean": float(signed_adv.mean().item()),
        "page_gate_mean": float(page_gate.mean().item()),
        "active_frac": float(active_mask.mean().item()),
        "item_share_mean": float(item_share.mean().item()),
        "item_adv_mean": float(item_adv.mean().item()),
    }
    stats.update(sibling_stats)
    return loss, stats


@torch.no_grad()
def evaluate(
    actor_tiger,
    old_tiger,
    critic_bundle,
    item_allocator_bundle,
    sid_prefix_to_next: Dict[Tuple[int, ...], List[int]],
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    actor_tiger.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_batch(
            actor_tiger,
            old_tiger,
            critic_bundle,
            item_allocator_bundle,
            sid_prefix_to_next,
            batch,
            device,
            args,
        )
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def main() -> int:
    args = parse_args()
    set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    chain_path = Path(args.chain_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = load_chain_rows(chain_path, int(args.max_rows))
    groups = [str(x["group"]) for x in rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(HCLAActorDataset(rows), train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_rows)
    valid_loader = DataLoader(Subset(HCLAActorDataset(rows), valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_rows)

    size_cfg = infer_model_size_args(str(args.model_size))
    old_tiger, _sid_depth, _vocab_size = load_tiger_model(
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
    init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, _sid_depth_actor, _vocab_size_actor = load_tiger_model(
        tiger_ckpt=init_ckpt,
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
    old_tiger.eval()
    trainable_params = set_train_scope(actor_tiger, str(args.train_scope))
    critic_bundle = None
    if str(args.critic_bundle_path).strip() and str(args.critic_meta_path).strip():
        critic_bundle = load_critic_bundle(str(args.critic_bundle_path), str(args.critic_meta_path), device)

    sid_prefix_to_next: Dict[Tuple[int, ...], List[int]] = {}
    item_allocator_bundle = None
    if float(args.sibling_loss_scale) > 0.0 or (
        str(args.item_allocator_head_path).strip() and str(args.item_allocator_meta_path).strip()
    ):
        if not str(args.uirm_log_path).strip():
            raise ValueError("--uirm_log_path is required for sibling-distribution updates and item allocator features.")
        reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
        sid_df = pd.read_csv(str(args.sid_mapping_path))
        sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
        iid2sid_tok_cpu, sid2iid_map_tok = build_iid2sid_tokens(
            reader,
            str(args.sid_mapping_path),
            int(sid_depth_cfg),
            torch.device("cpu"),
        )
        sid_prefix_to_next = build_sid_prefix_to_next(sid2iid_map_tok)
        token_vocab_size = int(max(int(iid2sid_tok_cpu.max().item()) + 1, int(_vocab_size_actor)))
        if str(args.item_allocator_head_path).strip() and str(args.item_allocator_meta_path).strip():
            item_allocator_head, _allocator_meta = load_online_slate_allocator(
                str(args.item_allocator_head_path),
                str(args.item_allocator_meta_path),
                device,
            )
            item_allocator_bundle = (item_allocator_head, iid2sid_tok_cpu.cpu(), int(token_vocab_size))

    optimizer = torch.optim.AdamW(
        [p for p in actor_tiger.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_loss = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    best_ckpt = save_dir / "hcla_actor_tiger.pth"
    for epoch in range(1, int(args.epochs) + 1):
        actor_tiger.train()
        train_metrics: Dict[str, List[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = forward_batch(
                actor_tiger,
                old_tiger,
                critic_bundle,
                item_allocator_bundle,
                sid_prefix_to_next,
                batch,
                device,
                args,
            )
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_([p for p in actor_tiger.parameters() if p.requires_grad], max_norm=float(args.grad_clip_norm))
            optimizer.step()
            for key, value in stats.items():
                train_metrics.setdefault(key, []).append(float(value))
        train_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in train_metrics.items()}
        valid_mean = evaluate(
            actor_tiger,
            old_tiger,
            critic_bundle,
            item_allocator_bundle,
            sid_prefix_to_next,
            valid_loader,
            device,
            args,
        )
        history.append({"epoch": int(epoch), "train": train_mean, "valid": valid_mean})
        if float(valid_mean.get("loss", float("inf"))) < float(best_loss):
            best_loss = float(valid_mean["loss"])
            best_epoch = int(epoch)
            torch.save(actor_tiger.state_dict(), best_ckpt)

    meta = {
        "method": "TIGER HCLA-RL actor",
        "variant": "item-share + sibling-distribution",
        "chain_path": str(chain_path.resolve()),
        "reference_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "actor_init_tiger_ckpt": str(Path(init_ckpt).resolve()),
        "actor_ckpt_path": str(best_ckpt.resolve()),
        "critic_bundle_path": str(Path(args.critic_bundle_path).resolve()) if str(args.critic_bundle_path).strip() else "",
        "critic_meta_path": str(Path(args.critic_meta_path).resolve()) if str(args.critic_meta_path).strip() else "",
        "item_allocator_head_path": str(Path(args.item_allocator_head_path).resolve()) if str(args.item_allocator_head_path).strip() else "",
        "item_allocator_meta_path": str(Path(args.item_allocator_meta_path).resolve()) if str(args.item_allocator_meta_path).strip() else "",
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_loss),
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "trainable_params": int(trainable_params),
        "history": history,
    }
    metrics_out = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "hcla_actor_metrics.json"
    write_json(metrics_out, meta)
    write_json(save_dir / "hcla_actor_meta.json", meta)
    print(f"[hcla-actor] best_valid_loss={best_loss:.6f} epoch={best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
