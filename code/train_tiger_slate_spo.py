import argparse
import json
import random
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils
from reader import *  # noqa: F401,F403

from tiger_phase2_blend_common import (
    load_prefix_value_head,
    build_history_tokens,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_tiger_model,
    write_json,
)
from tiger_hier_prefix_common import load_item_prefix_head
from tiger_slate_online_common import SlateValueHead, build_online_slate_inputs


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
    parser = argparse.ArgumentParser(description="TIGER Slate-SPO: actor + slate scorer preference optimization.")
    parser.add_argument("--pair_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Frozen reference TIGER checkpoint.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="", help="Actor init checkpoint. Defaults to --tiger_ckpt.")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--slate_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument(
        "--train_scope",
        type=str,
        default="last_decoder_block",
        choices=["decoder_only", "last_decoder_block", "full"],
    )
    parser.add_argument("--spo_beta", type=float, default=1.0)
    parser.add_argument("--slate_margin_scale", type=float, default=1.0)
    parser.add_argument("--slate_reg_scale", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--sft_scale", type=float, default=0.05)
    parser.add_argument("--gap_scale", type=float, default=0.75)
    parser.add_argument("--gap_clip", type=float, default=2.0)
    parser.add_argument("--slate_hidden_dim", type=int, default=96)
    parser.add_argument("--slate_dropout", type=float, default=0.10)
    parser.add_argument("--item_prefix_head_path", type=str, default="")
    parser.add_argument("--item_prefix_meta_path", type=str, default="")
    parser.add_argument("--item_teacher_scale", type=float, default=0.0)
    parser.add_argument("--item_teacher_mode", type=str, default="both", choices=["both", "chosen_only"])
    parser.add_argument("--token_prefix_head_path", type=str, default="")
    parser.add_argument("--token_prefix_meta_path", type=str, default="")
    parser.add_argument("--token_teacher_scale", type=float, default=0.0)
    parser.add_argument("--token_teacher_mode", type=str, default="both", choices=["both", "chosen_only"])
    parser.add_argument("--teacher_warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--score_normalization",
        type=str,
        default="mean_token",
        choices=["sum", "mean_token", "mean_item"],
    )
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class TigerSlateSPODataset(Dataset):
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
        "chosen_item_ids": torch.stack([torch.tensor(x["chosen_item_ids"], dtype=torch.long) for x in batch], dim=0),
        "rejected_item_ids": torch.stack([torch.tensor(x["rejected_item_ids"], dtype=torch.long) for x in batch], dim=0),
        "reward_gap": torch.tensor([float(x["reward_gap"]) for x in batch], dtype=torch.float32),
        "chosen_score": torch.tensor([float(x["chosen_score"]) for x in batch], dtype=torch.float32),
        "rejected_score": torch.tensor([float(x["rejected_score"]) for x in batch], dtype=torch.float32),
        "history_items": [x["history_items"] for x in batch],
        "groups": [x["group"] for x in batch],
    }


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, g in enumerate(groups):
        (valid_idx if g in valid_groups else train_idx).append(idx)
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def load_pair_rows(
    pair_path: Path,
    reader,
    sid_mapping_path: str,
    max_hist_items: int,
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], int, int, int, torch.Tensor]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    rows: List[Dict[str, Any]] = []
    slate_size = 0

    with pair_path.open("r", encoding="utf-8") as infile:
        for line_idx, line in enumerate(infile):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chosen_tokens = [[int(x) for x in seq] for seq in payload.get("chosen_sid_tokens_list", [])]
            rejected_tokens = [[int(x) for x in seq] for seq in payload.get("rejected_sid_tokens_list", [])]
            chosen_item_ids = [int(x) for x in payload.get("chosen_item_ids", [])]
            rejected_item_ids = [int(x) for x in payload.get("rejected_item_ids", [])]
            if not chosen_tokens or not rejected_tokens:
                continue
            if len(chosen_tokens) != len(rejected_tokens):
                continue
            if len(chosen_item_ids) != len(chosen_tokens) or len(rejected_item_ids) != len(rejected_tokens):
                continue
            if any(len(seq) != sid_depth for seq in chosen_tokens):
                continue
            if any(len(seq) != sid_depth for seq in rejected_tokens):
                continue
            history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(max_hist_items),
                int(sid_depth),
            )
            group = f"{payload.get('user_id', 'na')}:{payload.get('episode_progress_index', 'na')}"
            slate_size = max(slate_size, len(chosen_tokens))
            rows.append(
                {
                    "group": group,
                    "input_ids": input_ids.squeeze(0).tolist(),
                    "attention_mask": attention_mask.squeeze(0).tolist(),
                    "history_items": history_items,
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "chosen_item_ids": chosen_item_ids,
                    "rejected_item_ids": rejected_item_ids,
                    "reward_gap": float(payload.get("reward_gap", 0.0)),
                    "chosen_score": float(payload.get("chosen_score", 0.0)),
                    "rejected_score": float(payload.get("rejected_score", 0.0)),
                }
            )
    if not rows:
        raise ValueError(f"No usable Slate-SPO rows in {pair_path}")
    return rows, sid_depth, vocab_size, int(slate_size), iid2sid_tok_cpu


def set_train_scope(tiger, scope: str) -> int:
    for p in tiger.parameters():
        p.requires_grad = False
    name = str(scope)
    if name == "full":
        for p in tiger.parameters():
            p.requires_grad = True
    elif name == "decoder_only":
        for p in tiger.model.decoder.parameters():
            p.requires_grad = True
        for p in tiger.model.lm_head.parameters():
            p.requires_grad = True
    elif name == "last_decoder_block":
        for p in tiger.model.decoder.block[-1].parameters():
            p.requires_grad = True
        for p in tiger.model.decoder.final_layer_norm.parameters():
            p.requires_grad = True
        for p in tiger.model.lm_head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported train_scope: {scope}")
    return sum(p.numel() for p in tiger.parameters() if p.requires_grad)


def repeat_encoder_hidden(encoder_hidden: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if int(repeat_count) <= 1:
        return encoder_hidden
    return encoder_hidden.unsqueeze(1).expand(-1, int(repeat_count), -1, -1).reshape(
        encoder_hidden.shape[0] * int(repeat_count),
        encoder_hidden.shape[1],
        encoder_hidden.shape[2],
    )


def repeat_attention_mask(attention_mask: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if int(repeat_count) <= 1:
        return attention_mask
    return attention_mask.unsqueeze(1).expand(-1, int(repeat_count), -1).reshape(
        attention_mask.shape[0] * int(repeat_count),
        attention_mask.shape[1],
    )


def score_mode_divisor(token_count: int, slate_size: int, score_normalization: str) -> float:
    if str(score_normalization) == "sum":
        return 1.0
    if str(score_normalization) == "mean_item":
        return float(max(1, slate_size))
    return float(max(1, token_count))


def compute_slate_logp_components(
    tiger,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    score_normalization: str,
    encoder_no_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, slate_size, sid_depth = target_tokens.shape
    flat_targets = target_tokens.reshape(batch_size * slate_size, sid_depth)
    decoder_input_ids = decoder_input_ids_from_targets(flat_targets)

    if bool(encoder_no_grad):
        with torch.no_grad():
            encoder_hidden = tiger.encode(input_ids, attention_mask)
    else:
        encoder_hidden = tiger.encode(input_ids, attention_mask)

    encoder_hidden = repeat_encoder_hidden(encoder_hidden, slate_size)
    attention_mask_rep = repeat_attention_mask(attention_mask, slate_size)
    logits, hidden = tiger.decode_with_hidden_from_encoded(
        encoder_hidden,
        attention_mask=attention_mask_rep,
        decoder_input_ids=decoder_input_ids,
    )
    token_log_probs = F.log_softmax(logits, dim=-1)
    token_logp = token_log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
    item_logp = token_logp.sum(dim=-1).view(batch_size, slate_size)
    denom = score_mode_divisor(int(slate_size * sid_depth), int(slate_size), str(score_normalization))
    slate_scores = item_logp.sum(dim=-1) / float(denom)
    chosen_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        flat_targets.reshape(-1),
        reduction="mean",
    )
    hidden = hidden.view(batch_size, slate_size, sid_depth, hidden.shape[-1])
    token_logp = token_logp.view(batch_size, slate_size, sid_depth)
    return slate_scores, item_logp, chosen_ce, hidden, token_logp


def build_batch_slate_features(
    *,
    history_items_batch: Sequence[Sequence[int]],
    slate_item_ids: torch.Tensor,
    slate_item_scores: torch.Tensor,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    token_vocab_size: int,
    device: torch.device,
    use_scores: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    item_features_list: List[torch.Tensor] = []
    page_features_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    item_dim = 0
    page_dim = 0

    item_ids_np = slate_item_ids.detach().cpu().numpy()
    item_scores_np = slate_item_scores.detach().cpu().numpy()
    for row_idx, history_items in enumerate(history_items_batch):
        online = build_online_slate_inputs(
            history_items=[int(x) for x in history_items],
            candidate_item_ids=[int(x) for x in item_ids_np[row_idx].tolist()],
            candidate_sid_tokens_list=None,
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            max_hist_items=int(max_hist_items),
            token_vocab_size=int(token_vocab_size),
            base_scores=[float(x) for x in item_scores_np[row_idx].tolist()] if bool(use_scores) else None,
        )
        item_features = torch.tensor(online["item_features"], dtype=torch.float32)
        page_features = torch.tensor(online["page_features"], dtype=torch.float32)
        mask = torch.ones((item_features.shape[0],), dtype=torch.bool)
        item_features_list.append(item_features)
        page_features_list.append(page_features)
        mask_list.append(mask)
        item_dim = int(item_features.shape[1])
        page_dim = int(page_features.shape[0])

    item_features = torch.stack(item_features_list, dim=0).to(device)
    page_features = torch.stack(page_features_list, dim=0).to(device)
    mask = torch.stack(mask_list, dim=0).to(device)
    return item_features, page_features, mask, item_dim, page_dim


def prefix_to_delta(prefix_values: torch.Tensor) -> torch.Tensor:
    first = prefix_values[..., :1]
    if prefix_values.size(-1) <= 1:
        return first
    rest = prefix_values[..., 1:] - prefix_values[..., :-1]
    return torch.cat([first, rest], dim=-1)


def normalize_rowwise(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std


def compute_item_teacher_loss(
    item_head,
    actor_item_scores: torch.Tensor,
    item_features: torch.Tensor,
    page_features: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_items = mask.sum(dim=-1).long()
    prefix_values = []
    for prefix_len in range(1, int(item_features.shape[1]) + 1):
        prefix_len_tensor = torch.full(
            (item_features.shape[0],),
            int(prefix_len),
            dtype=torch.long,
            device=item_features.device,
        )
        value = item_head(
            item_features,
            page_features,
            mask=mask,
            prefix_len=prefix_len_tensor,
            total_items=total_items,
        )
        prefix_values.append(value)
    prefix_values = torch.stack(prefix_values, dim=-1)
    teacher_delta = prefix_to_delta(prefix_values).detach()
    loss = F.smooth_l1_loss(normalize_rowwise(actor_item_scores), normalize_rowwise(teacher_delta))
    stats = {
        "item_teacher_loss": float(loss.item()),
        "item_teacher_mean": float(teacher_delta.mean().item()),
    }
    return loss, stats


def compute_token_teacher_loss(
    token_head,
    actor_hidden: torch.Tensor,
    actor_token_logp: torch.Tensor,
    target_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, slate_size, sid_depth, hidden_dim = actor_hidden.shape
    hidden_flat = actor_hidden.reshape(bsz * slate_size, sid_depth, hidden_dim)
    target_flat = target_tokens.reshape(bsz * slate_size, sid_depth)
    teacher_prefix = token_head(hidden_flat.detach(), target_flat)
    teacher_delta = prefix_to_delta(teacher_prefix).view(bsz, slate_size, sid_depth).detach()
    actor_token_logp_flat = actor_token_logp.reshape(bsz * slate_size, sid_depth)
    teacher_delta_flat = teacher_delta.reshape(bsz * slate_size, sid_depth)
    loss = F.smooth_l1_loss(normalize_rowwise(actor_token_logp_flat), normalize_rowwise(teacher_delta_flat))
    stats = {
        "token_teacher_loss": float(loss.item()),
        "token_teacher_mean": float(teacher_delta.mean().item()),
    }
    return loss, stats


def compute_joint_loss(
    *,
    actor_chosen: torch.Tensor,
    actor_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    value_chosen: torch.Tensor,
    value_rejected: torch.Tensor,
    reward_gap: torch.Tensor,
    chosen_score: torch.Tensor,
    rejected_score: torch.Tensor,
    chosen_ce: torch.Tensor,
    item_teacher_loss: torch.Tensor | None,
    token_teacher_loss: torch.Tensor | None,
    spo_beta: float,
    slate_margin_scale: float,
    slate_reg_scale: float,
    item_teacher_scale: float,
    token_teacher_scale: float,
    label_smoothing: float,
    sft_scale: float,
    gap_scale: float,
    gap_clip: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    actor_margin = float(spo_beta) * ((actor_chosen - actor_rejected) - (ref_chosen - ref_rejected))
    value_margin = float(slate_margin_scale) * (value_chosen - value_rejected)
    margin = actor_margin + value_margin
    pos_loss = -F.logsigmoid(margin)
    if float(label_smoothing) > 0:
        neg_loss = -F.logsigmoid(-margin)
        pos_loss = (1.0 - float(label_smoothing)) * pos_loss + float(label_smoothing) * neg_loss
    gap_weight = 1.0 + float(gap_scale) * torch.clamp(reward_gap, min=0.0, max=float(gap_clip))
    pair_loss = (gap_weight * pos_loss).mean()
    slate_reg = 0.5 * (
        F.mse_loss(value_chosen, chosen_score, reduction="mean") +
        F.mse_loss(value_rejected, rejected_score, reduction="mean")
    )
    item_reg = item_teacher_loss if item_teacher_loss is not None else torch.zeros_like(pair_loss)
    token_reg = token_teacher_loss if token_teacher_loss is not None else torch.zeros_like(pair_loss)
    loss = (
        pair_loss
        + float(slate_reg_scale) * slate_reg
        + float(item_teacher_scale) * item_reg
        + float(token_teacher_scale) * token_reg
        + float(sft_scale) * chosen_ce
    )
    stats = {
        "loss": float(loss.item()),
        "pair_loss": float(pair_loss.item()),
        "slate_reg": float(slate_reg.item()),
        "item_reg": float(item_reg.item()),
        "token_reg": float(token_reg.item()),
        "sft_loss": float(chosen_ce.item()),
        "margin": float(margin.mean().item()),
        "pair_acc": float((margin > 0).float().mean().item()),
        "actor_pref_gain": float(((actor_chosen - actor_rejected) - (ref_chosen - ref_rejected)).mean().item()),
        "value_pref_gain": float((value_chosen - value_rejected).mean().item()),
        "value_chosen_mae": float(torch.abs(value_chosen - chosen_score).mean().item()),
        "value_rejected_mae": float(torch.abs(value_rejected - rejected_score).mean().item()),
        "gap_weight": float(gap_weight.mean().item()),
    }
    return loss, stats


def run_epoch(
    *,
    actor_tiger,
    ref_tiger,
    slate_head: SlateValueHead,
    item_head,
    token_head,
    loader: DataLoader,
    iid2sid_tok_cpu: torch.Tensor,
    token_vocab_size: int,
    device: torch.device,
    args: argparse.Namespace,
    epoch_idx: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    actor_tiger.train(mode=train_mode)
    if not train_mode:
        actor_tiger.eval()
    slate_head.train(mode=train_mode)
    if not train_mode:
        slate_head.eval()
    ref_tiger.eval()
    teacher_active = bool(train_mode) and (
        int(args.teacher_warmup_epochs) <= 0 or int(epoch_idx) <= int(args.teacher_warmup_epochs)
    )

    encoder_no_grad = not any(p.requires_grad for p in actor_tiger.model.encoder.parameters())
    meter: Dict[str, List[float]] = {
        "loss": [],
        "pair_loss": [],
        "slate_reg": [],
        "item_reg": [],
        "token_reg": [],
        "sft_loss": [],
        "margin": [],
        "pair_acc": [],
        "actor_pref_gain": [],
        "value_pref_gain": [],
        "value_chosen_mae": [],
        "value_rejected_mae": [],
        "gap_weight": [],
    }
    inferred_item_dim = None
    inferred_page_dim = None

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chosen_tokens = batch["chosen_tokens"].to(device)
        rejected_tokens = batch["rejected_tokens"].to(device)
        chosen_item_ids = batch["chosen_item_ids"].to(device)
        rejected_item_ids = batch["rejected_item_ids"].to(device)
        reward_gap = batch["reward_gap"].to(device)
        chosen_score = batch["chosen_score"].to(device)
        rejected_score = batch["rejected_score"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        actor_chosen, actor_item_scores_chosen, chosen_ce, chosen_hidden, chosen_token_logp = compute_slate_logp_components(
            actor_tiger,
            input_ids,
            attention_mask,
            chosen_tokens,
            score_normalization=str(args.score_normalization),
            encoder_no_grad=bool(encoder_no_grad),
        )
        actor_rejected, actor_item_scores_rejected, _rejected_ce, rejected_hidden, rejected_token_logp = compute_slate_logp_components(
            actor_tiger,
            input_ids,
            attention_mask,
            rejected_tokens,
            score_normalization=str(args.score_normalization),
            encoder_no_grad=bool(encoder_no_grad),
        )
        with torch.no_grad():
            ref_chosen, _ref_item_scores_chosen, _ref_ce1, _ref_hidden1, _ref_toklogp1 = compute_slate_logp_components(
                ref_tiger,
                input_ids,
                attention_mask,
                chosen_tokens,
                score_normalization=str(args.score_normalization),
                encoder_no_grad=False,
            )
            ref_rejected, _ref_item_scores_rejected, _ref_ce2, _ref_hidden2, _ref_toklogp2 = compute_slate_logp_components(
                ref_tiger,
                input_ids,
                attention_mask,
                rejected_tokens,
                score_normalization=str(args.score_normalization),
                encoder_no_grad=False,
            )

        chosen_item_features, chosen_page_features, chosen_mask, item_dim, page_dim = build_batch_slate_features(
            history_items_batch=batch["history_items"],
            slate_item_ids=chosen_item_ids,
            slate_item_scores=actor_item_scores_chosen.detach(),
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            max_hist_items=int(args.max_hist_items),
            token_vocab_size=int(token_vocab_size),
            device=device,
        )
        rejected_item_features, rejected_page_features, rejected_mask, _item_dim2, _page_dim2 = build_batch_slate_features(
            history_items_batch=batch["history_items"],
            slate_item_ids=rejected_item_ids,
            slate_item_scores=actor_item_scores_rejected.detach(),
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            max_hist_items=int(args.max_hist_items),
            token_vocab_size=int(token_vocab_size),
            device=device,
        )
        inferred_item_dim = int(item_dim)
        inferred_page_dim = int(page_dim)
        value_chosen = slate_head(chosen_item_features, chosen_page_features, mask=chosen_mask)
        value_rejected = slate_head(rejected_item_features, rejected_page_features, mask=rejected_mask)

        item_teacher_loss = None
        if teacher_active and item_head is not None and float(args.item_teacher_scale) > 0:
            chosen_item_teacher_features, chosen_item_teacher_page, chosen_item_teacher_mask, _id3, _pd3 = build_batch_slate_features(
                history_items_batch=batch["history_items"],
                slate_item_ids=chosen_item_ids,
                slate_item_scores=actor_item_scores_chosen.detach(),
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(args.max_hist_items),
                token_vocab_size=int(token_vocab_size),
                device=device,
                use_scores=False,
            )
            rejected_item_teacher_features, rejected_item_teacher_page, rejected_item_teacher_mask, _id4, _pd4 = build_batch_slate_features(
                history_items_batch=batch["history_items"],
                slate_item_ids=rejected_item_ids,
                slate_item_scores=actor_item_scores_rejected.detach(),
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(args.max_hist_items),
                token_vocab_size=int(token_vocab_size),
                device=device,
                use_scores=False,
            )
            item_loss_chosen, _item_stats_chosen = compute_item_teacher_loss(
                item_head,
                actor_item_scores_chosen,
                chosen_item_teacher_features,
                chosen_item_teacher_page,
                chosen_item_teacher_mask,
            )
            if str(args.item_teacher_mode) == "chosen_only":
                item_teacher_loss = item_loss_chosen
            else:
                item_loss_rejected, _item_stats_rejected = compute_item_teacher_loss(
                    item_head,
                    actor_item_scores_rejected,
                    rejected_item_teacher_features,
                    rejected_item_teacher_page,
                    rejected_item_teacher_mask,
                )
                item_teacher_loss = 0.5 * (item_loss_chosen + item_loss_rejected)

        token_teacher_loss = None
        if teacher_active and token_head is not None and float(args.token_teacher_scale) > 0:
            token_loss_chosen, _tok_stats_chosen = compute_token_teacher_loss(
                token_head,
                chosen_hidden,
                chosen_token_logp,
                chosen_tokens,
            )
            if str(args.token_teacher_mode) == "chosen_only":
                token_teacher_loss = token_loss_chosen
            else:
                token_loss_rejected, _tok_stats_rejected = compute_token_teacher_loss(
                    token_head,
                    rejected_hidden,
                    rejected_token_logp,
                    rejected_tokens,
                )
                token_teacher_loss = 0.5 * (token_loss_chosen + token_loss_rejected)

        loss, stats = compute_joint_loss(
            actor_chosen=actor_chosen,
            actor_rejected=actor_rejected,
            ref_chosen=ref_chosen,
            ref_rejected=ref_rejected,
            value_chosen=value_chosen,
            value_rejected=value_rejected,
            reward_gap=reward_gap,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            chosen_ce=chosen_ce,
            item_teacher_loss=item_teacher_loss,
            token_teacher_loss=token_teacher_loss,
            spo_beta=float(args.spo_beta),
            slate_margin_scale=float(args.slate_margin_scale),
            slate_reg_scale=float(args.slate_reg_scale),
            item_teacher_scale=float(args.item_teacher_scale),
            token_teacher_scale=float(args.token_teacher_scale),
            label_smoothing=float(args.label_smoothing),
            sft_scale=float(args.sft_scale),
            gap_scale=float(args.gap_scale),
            gap_clip=float(args.gap_clip),
        )

        if train_mode:
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in actor_tiger.parameters() if p.requires_grad] + list(slate_head.parameters()),
                    max_norm=float(args.grad_clip_norm),
                )
            optimizer.step()

        for key in meter.keys():
            meter[key].append(float(stats[key]))

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in meter.items()}
    result["teacher_active"] = 1.0 if teacher_active else 0.0
    if inferred_item_dim is not None:
        result["item_dim"] = float(inferred_item_dim)
    if inferred_page_dim is not None:
        result["page_dim"] = float(inferred_page_dim)
    return result


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(args.device)

    pair_path = Path(args.pair_path)
    if not pair_path.is_absolute():
        pair_path = pair_path.resolve()

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    rows, sid_depth, vocab_size, slate_size, iid2sid_tok_cpu = load_pair_rows(
        pair_path=pair_path,
        reader=reader,
        sid_mapping_path=str(args.sid_mapping_path),
        max_hist_items=int(args.max_hist_items),
        max_rows=int(args.max_rows),
    )

    groups = [str(x["group"]) for x in rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    dataset = TigerSlateSPODataset(rows)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_rows,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_rows,
    )

    size_args = infer_model_size_args(str(args.model_size))
    ref_tiger, sid_depth_model, codebook_size = load_tiger_model(
        tiger_ckpt=str(args.tiger_ckpt),
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
    ref_tiger.eval()
    for p in ref_tiger.parameters():
        p.requires_grad = False

    actor_init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, sid_depth_model2, _codebook_size2 = load_tiger_model(
        tiger_ckpt=actor_init_ckpt,
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
    trainable_params = set_train_scope(actor_tiger, str(args.train_scope))

    if int(sid_depth_model) != int(sid_depth) or int(sid_depth_model2) != int(sid_depth):
        raise ValueError(
            f"SID depth mismatch: pair={sid_depth}, ref_model={sid_depth_model}, actor_model={sid_depth_model2}"
        )
    if int(codebook_size + 1) != int(vocab_size):
        raise ValueError(f"Vocab mismatch: pair={vocab_size}, model={codebook_size + 1}")

    sample_online = build_online_slate_inputs(
        history_items=rows[0]["history_items"],
        candidate_item_ids=rows[0]["chosen_item_ids"],
        candidate_sid_tokens_list=rows[0]["chosen_tokens"],
        iid2sid_tok_cpu=iid2sid_tok_cpu,
        max_hist_items=int(args.max_hist_items),
        token_vocab_size=int(vocab_size),
        base_scores=[0.0 for _ in rows[0]["chosen_item_ids"]],
    )
    item_dim = int(sample_online["item_features"].shape[1])
    page_dim = int(sample_online["page_features"].shape[0])
    slate_head = SlateValueHead(
        item_dim=item_dim,
        page_dim=page_dim,
        hidden_dim=int(args.slate_hidden_dim),
        dropout=float(args.slate_dropout),
    ).to(device)

    item_head = None
    if str(args.item_prefix_head_path).strip():
        item_meta_path = str(args.item_prefix_meta_path).strip()
        if not item_meta_path:
            item_meta_path = str(Path(args.item_prefix_head_path).with_name("item_prefix_meta.json"))
        item_head, _item_meta = load_item_prefix_head(str(args.item_prefix_head_path), item_meta_path, device)
        for p in item_head.parameters():
            p.requires_grad = False
        item_head.eval()

    token_head = None
    if str(args.token_prefix_head_path).strip():
        token_meta_path = str(args.token_prefix_meta_path).strip()
        if not token_meta_path:
            token_meta_path = str(Path(args.token_prefix_head_path).with_name("prefix_critic_meta.json"))
        token_head, _token_meta = load_prefix_value_head(str(args.token_prefix_head_path), token_meta_path, device)
        for p in token_head.parameters():
            p.requires_grad = False
        token_head.eval()

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in actor_tiger.parameters() if p.requires_grad], "lr": float(args.lr)},
            {"params": list(slate_head.parameters()), "lr": float(args.slate_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    best_state = None
    best_valid = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(
            actor_tiger=actor_tiger,
            ref_tiger=ref_tiger,
            slate_head=slate_head,
            item_head=item_head,
            token_head=token_head,
            loader=train_loader,
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            token_vocab_size=int(vocab_size),
            device=device,
            args=args,
            epoch_idx=epoch,
            optimizer=optimizer,
        )
        valid_metrics = run_epoch(
            actor_tiger=actor_tiger,
            ref_tiger=ref_tiger,
            slate_head=slate_head,
            item_head=item_head,
            token_head=token_head,
            loader=valid_loader,
            iid2sid_tok_cpu=iid2sid_tok_cpu,
            token_vocab_size=int(vocab_size),
            device=device,
            args=args,
            epoch_idx=epoch,
            optimizer=None,
        )
        record = {
            "epoch": int(epoch),
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(record)
        print(
            f"[tiger-slate-spo] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"teacher={'on' if bool(train_metrics.get('teacher_active', 0.0)) else 'off'} "
            f"valid_acc={valid_metrics['pair_acc']:.4f} "
            f"valid_actor_gain={valid_metrics['actor_pref_gain']:.4f} "
            f"valid_value_gain={valid_metrics['value_pref_gain']:.4f}"
        )
        if best_valid is None or float(valid_metrics["loss"]) < float(best_valid):
            best_valid = float(valid_metrics["loss"])
            best_state = {
                "actor": deepcopy(actor_tiger.state_dict()),
                "slate_head": {k: v.detach().cpu() for k, v in slate_head.state_dict().items()},
            }

    if best_state is None:
        raise RuntimeError("TIGER Slate-SPO training produced no checkpoint.")

    save_dir = Path(args.save_dir) if args.save_dir else Path(actor_init_ckpt).resolve().parent / "tiger_slate_spo"
    save_dir.mkdir(parents=True, exist_ok=True)
    actor_ckpt_path = save_dir / "tiger_slate_spo_tiger.pth"
    head_path = save_dir / "slate_value_head.pt"
    meta_path = save_dir / "tiger_slate_spo_meta.json"
    head_meta_path = save_dir / "slate_value_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "tiger_slate_spo_metrics.json"
    torch.save(best_state["actor"], actor_ckpt_path)
    torch.save(best_state["slate_head"], head_path)

    head_meta = {
        "method": "TIGER Slate-SPO SlateValueHead",
        "item_dim": int(item_dim),
        "page_dim": int(page_dim),
        "hidden_dim": int(args.slate_hidden_dim),
        "dropout": float(args.slate_dropout),
        "max_hist_items": int(args.max_hist_items),
        "sid_depth": int(sid_depth),
        "seed": int(args.seed),
    }
    meta = {
        "method": "TIGER Slate-SPO",
        "pair_path": str(pair_path.resolve()),
        "reference_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "actor_init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "actor_ckpt_path": str(actor_ckpt_path.resolve()),
        "slate_head_path": str(head_path.resolve()),
        "slate_head_meta_path": str(head_meta_path.resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "vocab_size": int(vocab_size),
        "slate_size": int(slate_size),
        "train_scope": str(args.train_scope),
        "trainable_actor_params": int(trainable_params),
        "num_rows": int(len(rows)),
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "slate_lr": float(args.slate_lr),
        "weight_decay": float(args.weight_decay),
        "spo_beta": float(args.spo_beta),
        "slate_margin_scale": float(args.slate_margin_scale),
        "slate_reg_scale": float(args.slate_reg_scale),
        "item_prefix_head_path": str(Path(args.item_prefix_head_path).resolve()) if str(args.item_prefix_head_path).strip() else "",
        "item_teacher_scale": float(args.item_teacher_scale),
        "item_teacher_mode": str(args.item_teacher_mode),
        "token_prefix_head_path": str(Path(args.token_prefix_head_path).resolve()) if str(args.token_prefix_head_path).strip() else "",
        "token_teacher_scale": float(args.token_teacher_scale),
        "token_teacher_mode": str(args.token_teacher_mode),
        "teacher_warmup_epochs": int(args.teacher_warmup_epochs),
        "label_smoothing": float(args.label_smoothing),
        "sft_scale": float(args.sft_scale),
        "gap_scale": float(args.gap_scale),
        "gap_clip": float(args.gap_clip),
        "score_normalization": str(args.score_normalization),
        "best_valid_loss": float(best_valid),
        "history": history,
    }
    write_json(head_meta_path, head_meta)
    write_json(meta_path, meta)
    write_json(metrics_path, meta)
    print(f"[tiger-slate-spo] saved actor checkpoint to {actor_ckpt_path}")
    print(f"[tiger-slate-spo] saved slate head to {head_path}")
    print(f"[tiger-slate-spo] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
