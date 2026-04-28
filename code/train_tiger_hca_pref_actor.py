import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils

from tiger_phase2_blend_common import decoder_input_ids_from_targets, infer_model_size_args, load_tiger_model, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preference-style actor post-training over HCA support-constrained pairs.")
    parser.add_argument("--pair_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Frozen reference TIGER checkpoint.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--pref_beta", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--sft_scale", type=float, default=0.05)
    parser.add_argument("--gap_scale", type=float, default=1.0)
    parser.add_argument("--gap_clip", type=float, default=2.0)
    parser.add_argument("--score_normalization", type=str, default="mean_token", choices=["sum", "mean_token"])
    parser.add_argument("--attr_adv_mode", type=str, default="pess", choices=["raw", "pess"])
    parser.add_argument("--attr_pair_scale", type=float, default=0.0)
    parser.add_argument("--attr_item_scale", type=float, default=0.10)
    parser.add_argument("--attr_credit_clip", type=float, default=3.0)
    parser.add_argument("--attr_renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--attr_topk", type=int, default=2)
    parser.add_argument("--attr_floor", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class TigerHCAPreferenceDataset(Dataset):
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
        "chosen_tokens": torch.stack([torch.tensor(x["chosen_tokens"], dtype=torch.long) for x in batch], dim=0),
        "rejected_tokens": torch.stack([torch.tensor(x["rejected_tokens"], dtype=torch.long) for x in batch], dim=0),
        "chosen_token_adv": torch.stack([torch.tensor(x["chosen_token_adv"], dtype=torch.float32) for x in batch], dim=0),
        "rejected_token_adv": torch.stack([torch.tensor(x["rejected_token_adv"], dtype=torch.float32) for x in batch], dim=0),
        "chosen_item_adv": torch.tensor([float(x["chosen_item_adv"]) for x in batch], dtype=torch.float32),
        "rejected_item_adv": torch.tensor([float(x["rejected_item_adv"]) for x in batch], dtype=torch.float32),
        "reward_gap": torch.tensor([float(x["reward_gap"]) for x in batch], dtype=torch.float32),
        "groups": [str(x["group"]) for x in batch],
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


def load_pair_rows(pair_path: Path, max_rows: int, attr_adv_mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    use_pess = str(attr_adv_mode).strip().lower() == "pess"
    chosen_token_key = "chosen_sid_advantage_pess" if use_pess else "chosen_sid_advantage"
    rejected_token_key = "rejected_sid_advantage_pess" if use_pess else "rejected_sid_advantage"
    chosen_item_key = "chosen_item_advantage_pess" if use_pess else "chosen_item_advantage"
    rejected_item_key = "rejected_item_advantage_pess" if use_pess else "rejected_item_advantage"
    with pair_path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chosen_tokens = [int(x) for x in payload.get("chosen_tokens", [])]
            rejected_tokens = [int(x) for x in payload.get("rejected_tokens", [])]
            if not chosen_tokens or len(chosen_tokens) != len(rejected_tokens):
                continue
            chosen_token_adv = [float(x) for x in payload.get(chosen_token_key, [])]
            rejected_token_adv = [float(x) for x in payload.get(rejected_token_key, [])]
            if len(chosen_token_adv) != len(chosen_tokens):
                chosen_token_adv = [0.0 for _ in chosen_tokens]
            if len(rejected_token_adv) != len(rejected_tokens):
                rejected_token_adv = [0.0 for _ in rejected_tokens]
            rows.append(
                {
                    "group": str(payload.get("group", payload.get("pair_id", line_idx))),
                    "input_ids": [int(x) for x in payload.get("input_ids", [])],
                    "attention_mask": [int(x) for x in payload.get("attention_mask", [])],
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "chosen_token_adv": chosen_token_adv,
                    "rejected_token_adv": rejected_token_adv,
                    "chosen_item_adv": float(payload.get(chosen_item_key, 0.0)),
                    "rejected_item_adv": float(payload.get(rejected_item_key, 0.0)),
                    "reward_gap": float(payload.get("reward_gap", 0.0)),
                }
            )
    if not rows:
        raise ValueError(f"No usable preference pairs in {pair_path}")
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


def build_sparse_positive_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
    floor: float,
) -> torch.Tensor:
    positive = (scores > float(floor)) & valid_mask
    if int(topk) <= 0 or int(scores.shape[-1]) <= int(topk):
        return positive.float()
    masked_scores = scores.masked_fill(~valid_mask, -1e9)
    idx = torch.topk(masked_scores, k=min(int(topk), int(scores.shape[-1])), dim=-1).indices
    mask = torch.zeros_like(valid_mask, dtype=torch.bool)
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


def build_attr_weights(
    token_adv: torch.Tensor,
    item_adv: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    item_scale: float,
    credit_clip: float,
    renorm_mode: str,
    topk: int,
    floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if float(credit_clip) > 0.0:
        token_adv = token_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        item_adv = item_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
    token_adv = renorm_signal(token_adv * valid_mask.float(), str(renorm_mode))
    item_adv = renorm_signal(item_adv.unsqueeze(-1), str(renorm_mode)).squeeze(-1)
    pos_scores = (torch.relu(token_adv) + float(item_scale) * torch.relu(item_adv).unsqueeze(-1)) * valid_mask.float()
    pos_mask = build_sparse_positive_mask(pos_scores, valid_mask.bool(), int(topk), float(floor))
    abs_scores = (token_adv.abs() + float(item_scale) * item_adv.abs().unsqueeze(-1)) * valid_mask.float()
    missing = pos_mask.sum(dim=-1) <= 0
    if bool(missing.any()):
        fallback_idx = abs_scores[missing].argmax(dim=-1, keepdim=True)
        pos_mask[missing] = 0.0
        pos_mask[missing].scatter_(1, fallback_idx, 1.0)
        pos_scores[missing] = abs_scores[missing]
    weights = normalize_weights(pos_scores.clamp_min(0.0), pos_mask)
    return weights, pos_mask


def sequence_logp_and_ce(
    tiger,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    score_normalization: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)
    logits, _hidden = tiger.decode_with_hidden(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logp = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    valid_mask = (target_tokens > 0)
    seq_logp_sum = (target_logp * valid_mask.float()).sum(dim=-1)
    if str(score_normalization) == "sum":
        seq_logp = seq_logp_sum
    else:
        seq_logp = seq_logp_sum / valid_mask.float().sum(dim=-1).clamp_min(1.0)
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_tokens.reshape(-1),
        reduction="mean",
    )
    return seq_logp, ce_loss, target_logp, valid_mask.float()


def weighted_sequence_logp(target_logp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (target_logp * weights).sum(dim=-1)


def compute_preference_loss(
    *,
    actor_chosen: torch.Tensor,
    actor_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    reward_gap: torch.Tensor,
    chosen_ce: torch.Tensor,
    pref_beta: float,
    label_smoothing: float,
    sft_scale: float,
    gap_scale: float,
    gap_clip: float,
    attr_actor_chosen: Optional[torch.Tensor],
    attr_actor_rejected: Optional[torch.Tensor],
    attr_ref_chosen: Optional[torch.Tensor],
    attr_ref_rejected: Optional[torch.Tensor],
    attr_pair_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pref_actor = actor_chosen - actor_rejected
    pref_ref = ref_chosen - ref_rejected
    margin = float(pref_beta) * (pref_actor - pref_ref)
    pos_loss = -F.logsigmoid(margin)
    if float(label_smoothing) > 0.0:
        neg_loss = -F.logsigmoid(-margin)
        pos_loss = (1.0 - float(label_smoothing)) * pos_loss + float(label_smoothing) * neg_loss
    gap_weight = 1.0 + float(gap_scale) * torch.clamp(reward_gap, min=0.0, max=float(gap_clip))
    pair_loss = (gap_weight * pos_loss).mean()
    attr_pair_loss = pair_loss.new_tensor(0.0)
    attr_margin_mean = 0.0
    attr_pair_acc = 0.0
    if (
        float(attr_pair_scale) > 0.0
        and attr_actor_chosen is not None
        and attr_actor_rejected is not None
        and attr_ref_chosen is not None
        and attr_ref_rejected is not None
    ):
        attr_pref_actor = attr_actor_chosen - attr_actor_rejected
        attr_pref_ref = attr_ref_chosen - attr_ref_rejected
        attr_margin = float(pref_beta) * (attr_pref_actor - attr_pref_ref)
        attr_pos_loss = -F.logsigmoid(attr_margin)
        if float(label_smoothing) > 0.0:
            attr_neg_loss = -F.logsigmoid(-attr_margin)
            attr_pos_loss = (1.0 - float(label_smoothing)) * attr_pos_loss + float(label_smoothing) * attr_neg_loss
        attr_pair_loss = (gap_weight * attr_pos_loss).mean()
        attr_margin_mean = float(attr_margin.mean().item())
        attr_pair_acc = float((attr_margin > 0).float().mean().item())
    loss = pair_loss + float(attr_pair_scale) * attr_pair_loss + float(sft_scale) * chosen_ce
    stats = {
        "loss": float(loss.item()),
        "pair_loss": float(pair_loss.item()),
        "attr_pair_loss": float(attr_pair_loss.item()),
        "sft_loss": float(chosen_ce.item()),
        "pref_actor": float(pref_actor.mean().item()),
        "pref_ref": float(pref_ref.mean().item()),
        "pref_gain": float((pref_actor - pref_ref).mean().item()),
        "margin": float(margin.mean().item()),
        "pair_acc": float((margin > 0).float().mean().item()),
        "attr_margin": float(attr_margin_mean),
        "attr_pair_acc": float(attr_pair_acc),
        "attr_pair_scale": float(attr_pair_scale),
        "gap_weight": float(gap_weight.mean().item()),
        "reward_gap": float(reward_gap.mean().item()),
    }
    return loss, stats


def forward_actor(
    actor_tiger,
    ref_tiger,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    chosen_tokens = batch["chosen_tokens"].to(device)
    rejected_tokens = batch["rejected_tokens"].to(device)
    chosen_token_adv = batch["chosen_token_adv"].to(device)
    rejected_token_adv = batch["rejected_token_adv"].to(device)
    chosen_item_adv = batch["chosen_item_adv"].to(device)
    rejected_item_adv = batch["rejected_item_adv"].to(device)
    reward_gap = batch["reward_gap"].to(device)

    actor_chosen, chosen_ce, actor_chosen_token_logp, chosen_valid_mask = sequence_logp_and_ce(
        actor_tiger,
        input_ids,
        attention_mask,
        chosen_tokens,
        score_normalization=str(args.score_normalization),
    )
    actor_rejected, _rejected_ce, actor_rejected_token_logp, rejected_valid_mask = sequence_logp_and_ce(
        actor_tiger,
        input_ids,
        attention_mask,
        rejected_tokens,
        score_normalization=str(args.score_normalization),
    )
    attr_actor_chosen = None
    attr_actor_rejected = None
    attr_ref_chosen = None
    attr_ref_rejected = None
    chosen_attr_mask = None
    rejected_attr_mask = None
    if float(args.attr_pair_scale) > 0.0:
        chosen_attr_weights, chosen_attr_mask = build_attr_weights(
            chosen_token_adv,
            chosen_item_adv,
            chosen_valid_mask > 0.0,
            item_scale=float(args.attr_item_scale),
            credit_clip=float(args.attr_credit_clip),
            renorm_mode=str(args.attr_renorm_mode),
            topk=int(args.attr_topk),
            floor=float(args.attr_floor),
        )
        rejected_attr_weights, rejected_attr_mask = build_attr_weights(
            rejected_token_adv,
            rejected_item_adv,
            rejected_valid_mask > 0.0,
            item_scale=float(args.attr_item_scale),
            credit_clip=float(args.attr_credit_clip),
            renorm_mode=str(args.attr_renorm_mode),
            topk=int(args.attr_topk),
            floor=float(args.attr_floor),
        )
        attr_actor_chosen = weighted_sequence_logp(actor_chosen_token_logp, chosen_attr_weights)
        attr_actor_rejected = weighted_sequence_logp(actor_rejected_token_logp, rejected_attr_weights)
    with torch.no_grad():
        ref_chosen, _, ref_chosen_token_logp, _ = sequence_logp_and_ce(
            ref_tiger,
            input_ids,
            attention_mask,
            chosen_tokens,
            score_normalization=str(args.score_normalization),
        )
        ref_rejected, _, ref_rejected_token_logp, _ = sequence_logp_and_ce(
            ref_tiger,
            input_ids,
            attention_mask,
            rejected_tokens,
            score_normalization=str(args.score_normalization),
        )
        if float(args.attr_pair_scale) > 0.0:
            attr_ref_chosen = weighted_sequence_logp(ref_chosen_token_logp, chosen_attr_weights)
            attr_ref_rejected = weighted_sequence_logp(ref_rejected_token_logp, rejected_attr_weights)
    loss, stats = compute_preference_loss(
        actor_chosen=actor_chosen,
        actor_rejected=actor_rejected,
        ref_chosen=ref_chosen,
        ref_rejected=ref_rejected,
        reward_gap=reward_gap,
        chosen_ce=chosen_ce,
        pref_beta=float(args.pref_beta),
        label_smoothing=float(args.label_smoothing),
        sft_scale=float(args.sft_scale),
        gap_scale=float(args.gap_scale),
        gap_clip=float(args.gap_clip),
        attr_actor_chosen=attr_actor_chosen,
        attr_actor_rejected=attr_actor_rejected,
        attr_ref_chosen=attr_ref_chosen,
        attr_ref_rejected=attr_ref_rejected,
        attr_pair_scale=float(args.attr_pair_scale),
    )
    if chosen_attr_mask is not None and rejected_attr_mask is not None:
        stats["chosen_attr_active_frac"] = float(chosen_attr_mask.mean().item())
        stats["rejected_attr_active_frac"] = float(rejected_attr_mask.mean().item())
        stats["chosen_item_adv_mean"] = float(chosen_item_adv.mean().item())
        stats["rejected_item_adv_mean"] = float(rejected_item_adv.mean().item())
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
    ref_tiger.eval()
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

    pair_rows = load_pair_rows(Path(args.pair_path), int(args.max_rows), str(args.attr_adv_mode))

    size_cfg = infer_model_size_args(str(args.model_size))
    ref_tiger, sid_depth, _codebook_size = load_tiger_model(
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

    dataset = TigerHCAPreferenceDataset(pair_rows)
    groups = [row["group"] for row in pair_rows]
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
        train_pair_acc: List[float] = []
        train_pref_gain: List[float] = []
        for batch in train_loader:
            loss, stats = forward_actor(actor_tiger, ref_tiger, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_pair_acc.append(float(stats["pair_acc"]))
            train_pref_gain.append(float(stats["pref_gain"]))

        valid_metrics = evaluate_actor(actor_tiger, ref_tiger, valid_loader, device, args)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        valid_metrics["train_pair_acc"] = float(np.mean(train_pair_acc)) if train_pair_acc else 0.0
        valid_metrics["train_pref_gain"] = float(np.mean(train_pref_gain)) if train_pref_gain else 0.0
        history.append(dict(valid_metrics))
        if float(valid_metrics["loss"]) < float(best_key):
            best_key = float(valid_metrics["loss"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu() for key, value in actor_tiger.state_dict().items()}
            best_metrics = dict(valid_metrics)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"pair_acc={valid_metrics['pair_acc']:.4f} "
            f"pref_gain={valid_metrics['pref_gain']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Preference actor training produced no checkpoint.")

    save_dir = Path(args.save_dir) if str(args.save_dir).strip() else Path(args.tiger_ckpt).resolve().parent / "tiger_hca_pref_actor"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "tiger_hca_pref_actor_tiger.pth"
    meta_path = save_dir / "tiger_hca_pref_actor_meta.json"
    metrics_path = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "tiger_hca_pref_actor_metrics.json"
    torch.save(best_state, ckpt_path)
    meta = {
        "method": "TIGER-HCA Preference Actor",
        "pair_path": str(Path(args.pair_path).resolve()),
        "old_policy_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "train_scope": str(args.train_scope),
        "n_trainable": int(n_trainable),
        "pref_beta": float(args.pref_beta),
        "label_smoothing": float(args.label_smoothing),
        "sft_scale": float(args.sft_scale),
        "gap_scale": float(args.gap_scale),
        "gap_clip": float(args.gap_clip),
        "score_normalization": str(args.score_normalization),
        "attr_adv_mode": str(args.attr_adv_mode),
        "attr_pair_scale": float(args.attr_pair_scale),
        "attr_item_scale": float(args.attr_item_scale),
        "attr_credit_clip": float(args.attr_credit_clip),
        "attr_renorm_mode": str(args.attr_renorm_mode),
        "attr_topk": int(args.attr_topk),
        "attr_floor": float(args.attr_floor),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_pairs": int(len(pair_rows)),
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
    print(f"[hca-pref] saved fine-tuned TIGER to {ckpt_path}")
    print(f"[hca-pref] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
