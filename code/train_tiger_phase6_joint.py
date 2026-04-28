import argparse
import json
import random
from argparse import Namespace
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
    build_history_tokens,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_tiger_model,
    write_json,
)
from tiger_phase6_joint_common import (
    PLAN_NAMES,
    HistoryPlanHead,
    PlanConditionedPrefixValueHead,
    PlanConditionedTokenActorHead,
    SlateCreditHead,
    derive_plan_target,
)


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
    parser = argparse.ArgumentParser(description="Joint hierarchical training for TIGER slate/plan/item/token credit heads.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--token_credit_field", type=str, default="token_credit_calibrated")
    parser.add_argument("--item_credit_field", type=str, default="item_credit")
    parser.add_argument("--target_clip", type=float, default=0.0)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--plan_dim", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--plan_mlp_dim", type=int, default=128)
    parser.add_argument("--slate_mlp_dim", type=int, default=128)
    parser.add_argument("--plan_cls_loss_scale", type=float, default=0.2)
    parser.add_argument("--plan_credit_loss_scale", type=float, default=0.5)
    parser.add_argument("--slate_loss_scale", type=float, default=0.5)
    parser.add_argument("--prefix_loss_scale", type=float, default=1.0)
    parser.add_argument("--delta_loss_scale", type=float, default=1.0)
    parser.add_argument("--item_loss_scale", type=float, default=0.5)
    parser.add_argument("--page_cons_scale", type=float, default=0.5)
    parser.add_argument("--item_cons_scale", type=float, default=0.25)
    parser.add_argument("--actor_loss_scale", type=float, default=0.5)
    parser.add_argument("--actor_neg_scale", type=float, default=0.15)
    parser.add_argument("--actor_anchor_scale", type=float, default=0.05)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--residual_clip", type=float, default=2.0)
    parser.add_argument("--init_joint_head_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


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


class JointPageDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_pages(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bsz = len(batch)
    max_items = max(int(len(x["target_tokens"])) for x in batch)
    sid_depth = int(len(batch[0]["target_tokens"][0]))
    input_ids = torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], dim=0)
    attention_mask = torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], dim=0)
    target_tokens = torch.zeros((bsz, max_items, sid_depth), dtype=torch.long)
    token_credit = torch.zeros((bsz, max_items, sid_depth), dtype=torch.float32)
    prefix_credit = torch.zeros((bsz, max_items, sid_depth), dtype=torch.float32)
    item_credit = torch.zeros((bsz, max_items), dtype=torch.float32)
    item_mask = torch.zeros((bsz, max_items), dtype=torch.bool)
    for row_idx, row in enumerate(batch):
        n_items = int(len(row["target_tokens"]))
        target_tokens[row_idx, :n_items] = torch.tensor(row["target_tokens"], dtype=torch.long)
        token_credit[row_idx, :n_items] = torch.tensor(row["token_credit"], dtype=torch.float32)
        prefix_credit[row_idx, :n_items] = torch.tensor(row["prefix_credit"], dtype=torch.float32)
        item_credit[row_idx, :n_items] = torch.tensor(row["item_credit"], dtype=torch.float32)
        item_mask[row_idx, :n_items] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_tokens": target_tokens,
        "token_credit": token_credit,
        "prefix_credit": prefix_credit,
        "item_credit": item_credit,
        "item_mask": item_mask,
        "slate_credit": torch.tensor([float(x["slate_credit"]) for x in batch], dtype=torch.float32),
        "plan_credit": torch.tensor([float(x["plan_credit"]) for x in batch], dtype=torch.float32),
        "plan_label": torch.tensor([int(x["plan_label"]) for x in batch], dtype=torch.long),
        "plan_ratio": torch.tensor([float(x["plan_ratio"]) for x in batch], dtype=torch.float32),
        "groups": [x["group"] for x in batch],
    }


def load_page_rows(
    chain_path: Path,
    reader,
    sid_mapping_path: str,
    max_hist_items: int,
    token_credit_field: str,
    item_credit_field: str,
    target_clip: float,
    max_pages: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = f"{int(payload['episode_id'])}:{int(payload['page_index'])}"
            grouped.setdefault(key, []).append(payload)

    page_rows: List[Dict[str, Any]] = []
    for page_idx, (group, rows) in enumerate(sorted(grouped.items(), key=lambda x: tuple(int(v) for v in x[0].split(":")))):
        if int(max_pages) > 0 and page_idx >= int(max_pages):
            break
        rows = sorted(rows, key=lambda x: int(x.get("slate_item_index", 0)))
        first = rows[0]
        history_items = [int(x) for x in first.get("history_items", [])][-int(max_hist_items):]
        hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
        input_ids, attention_mask = build_history_tokens(
            hist_tensor,
            iid2sid_tok_cpu,
            int(max_hist_items),
            int(sid_depth),
        )
        slate_credit = float(first.get("slate_credit", 0.0))
        support_arr = np.asarray(first.get("selected_item_support_strengths", []), dtype=np.float32)
        response_arr = np.asarray(first.get("selected_item_response_strengths", []), dtype=np.float32)
        share_arr = np.asarray(first.get("selected_item_credit_shares", []), dtype=np.float32)
        mean_support = float(support_arr.mean()) if support_arr.size > 0 else 0.0
        mean_response = float(response_arr.mean()) if response_arr.size > 0 else 0.0
        top_share = float(np.max(np.abs(share_arr))) if share_arr.size > 0 else 0.0
        plan_label, plan_ratio = derive_plan_target(
            slate_credit=slate_credit,
            mean_support=mean_support,
            mean_response=mean_response,
            top_share=top_share,
        )
        residual_scale = 1.0 - float(plan_ratio)
        item_targets: List[float] = []
        token_targets: List[List[float]] = []
        prefix_targets: List[List[float]] = []
        target_tokens_list: List[List[int]] = []
        for row in rows:
            target_tokens = [int(x) for x in row.get("selected_sid_tokens", [])]
            token_credit = [float(x) for x in row.get(str(token_credit_field), row.get("token_credit_calibrated", []))]
            item_credit = float(row.get(str(item_credit_field), row.get("item_credit", 0.0)))
            if len(target_tokens) != sid_depth or len(token_credit) != sid_depth:
                continue
            token_credit_arr = residual_scale * np.asarray(token_credit, dtype=np.float32)
            if float(target_clip) > 0.0:
                token_credit_arr = np.clip(token_credit_arr, -float(target_clip), float(target_clip))
            prefix_credit_arr = np.cumsum(token_credit_arr, dtype=np.float32)
            item_credit_scaled = float(prefix_credit_arr[-1]) if prefix_credit_arr.size > 0 else float(residual_scale * item_credit)
            target_tokens_list.append(target_tokens)
            token_targets.append(token_credit_arr.tolist())
            prefix_targets.append(prefix_credit_arr.tolist())
            item_targets.append(float(item_credit_scaled))
        if not target_tokens_list:
            continue
        page_rows.append(
            {
                "group": str(group),
                "input_ids": input_ids.squeeze(0).tolist(),
                "attention_mask": attention_mask.squeeze(0).tolist(),
                "target_tokens": target_tokens_list,
                "token_credit": token_targets,
                "prefix_credit": prefix_targets,
                "item_credit": item_targets,
                "slate_credit": float(slate_credit),
                "plan_credit": float(plan_ratio * slate_credit),
                "plan_label": int(plan_label),
                "plan_ratio": float(plan_ratio),
                "plan_name": PLAN_NAMES[int(plan_label)],
            }
        )
    if not page_rows:
        raise ValueError(f"No usable page rows in {chain_path}")
    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    return page_rows, sid_depth, vocab_size


def pooled_history_summary(tiger, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    enc_out = tiger.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = enc_out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


def prefix_to_delta(prefix_values: torch.Tensor) -> torch.Tensor:
    first = prefix_values[:, :1]
    if prefix_values.size(1) <= 1:
        return first
    rest = prefix_values[:, 1:] - prefix_values[:, :-1]
    return torch.cat([first, rest], dim=1)


def flatten_item_batch(batch: Dict[str, torch.Tensor], plan_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    item_mask = batch["item_mask"]
    bsz, max_items, sid_depth = batch["target_tokens"].shape
    flat_input_ids = batch["input_ids"].unsqueeze(1).expand(-1, max_items, -1)[item_mask]
    flat_attention_mask = batch["attention_mask"].unsqueeze(1).expand(-1, max_items, -1)[item_mask]
    flat_target_tokens = batch["target_tokens"][item_mask]
    flat_token_credit = batch["token_credit"][item_mask]
    flat_prefix_credit = batch["prefix_credit"][item_mask]
    flat_item_credit = batch["item_credit"][item_mask]
    flat_plan_probs = plan_probs.unsqueeze(1).expand(-1, max_items, -1)[item_mask]
    flat_page_index = torch.arange(bsz, device=batch["input_ids"].device).unsqueeze(1).expand(-1, max_items)[item_mask]
    return {
        "input_ids": flat_input_ids,
        "attention_mask": flat_attention_mask,
        "target_tokens": flat_target_tokens,
        "token_credit": flat_token_credit,
        "prefix_credit": flat_prefix_credit,
        "item_credit": flat_item_credit,
        "plan_probs": flat_plan_probs,
        "page_index": flat_page_index,
        "n_items": int(flat_target_tokens.size(0)),
        "sid_depth": int(sid_depth),
    }


def compute_actor_loss(
    base_logits: torch.Tensor,
    residual_scores: torch.Tensor,
    target_tokens: torch.Tensor,
    token_credit: torch.Tensor,
    item_credit: torch.Tensor,
    *,
    neg_scale: float,
    anchor_scale: float,
    credit_clip: float,
    residual_clip: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    base_log_probs = torch.log_softmax(base_logits.detach(), dim=-1)
    residual_scores = residual_scores - residual_scores.mean(dim=-1, keepdim=True)
    if float(residual_clip) > 0.0:
        residual_scores = residual_scores.clamp(min=-float(residual_clip), max=float(residual_clip))
    combined_logits = base_log_probs + residual_scores
    combined_log_probs = torch.log_softmax(combined_logits, dim=-1)
    target_logp = combined_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    target_prob = target_logp.exp()
    base_target_prob = base_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).exp()

    if float(credit_clip) > 0.0:
        token_credit = token_credit.clamp(min=-float(credit_clip), max=float(credit_clip))
        item_credit = item_credit.clamp(min=-float(credit_clip), max=float(credit_clip))
    pos_w = torch.relu(token_credit) + 0.20 * torch.relu(item_credit).unsqueeze(-1)
    neg_w = torch.relu(-token_credit)
    pos_loss = -(pos_w * target_logp).sum() / (pos_w.sum() + 1e-8)
    neg_excess = torch.relu(target_prob - base_target_prob)
    neg_loss = (neg_w * neg_excess).sum() / (neg_w.sum() + 1e-8)
    anchor_loss = residual_scores.pow(2).mean()
    loss = pos_loss + float(neg_scale) * neg_loss + float(anchor_scale) * anchor_loss
    stats = {
        "pos_loss": float(pos_loss.item()),
        "neg_loss": float(neg_loss.item()),
        "anchor_loss": float(anchor_loss.item()),
        "target_gain": float((target_prob - base_target_prob).mean().item()),
    }
    return loss, stats


def forward_joint(
    tiger,
    plan_head: HistoryPlanHead,
    slate_head: SlateCreditHead,
    prefix_head: PlanConditionedPrefixValueHead,
    token_actor_head: PlanConditionedTokenActorHead,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    slate_credit = batch["slate_credit"].to(device)
    plan_credit = batch["plan_credit"].to(device)
    plan_label = batch["plan_label"].to(device)
    item_mask = batch["item_mask"].to(device)
    item_credit_full = batch["item_credit"].to(device)

    with torch.no_grad():
        history_summary = pooled_history_summary(tiger, input_ids, attention_mask)
    plan_logits, plan_credit_pred = plan_head(history_summary.detach())
    slate_credit_pred = slate_head(history_summary.detach())
    plan_probs = torch.softmax(plan_logits, dim=-1)

    flat = flatten_item_batch(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_tokens": batch["target_tokens"].to(device),
            "token_credit": batch["token_credit"].to(device),
            "prefix_credit": batch["prefix_credit"].to(device),
            "item_credit": item_credit_full,
            "item_mask": item_mask,
        },
        plan_probs,
    )
    if flat["n_items"] <= 0:
        raise RuntimeError("No valid items in joint batch.")

    with torch.no_grad():
        base_logits, hidden = tiger.decode_with_hidden(
            input_ids=flat["input_ids"],
            attention_mask=flat["attention_mask"],
            decoder_input_ids=decoder_input_ids_from_targets(flat["target_tokens"]),
        )

    pred_prefix = prefix_head(hidden.detach(), flat["target_tokens"], flat["plan_probs"])
    pred_delta = prefix_to_delta(pred_prefix)
    pred_item = pred_prefix[:, -1]
    seq_len = int(hidden.shape[1])
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    plan_seq = flat["plan_probs"].unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, flat["plan_probs"].shape[-1])
    residual_scores = token_actor_head.score_all_tokens(hidden_flat, plan_seq).view(hidden.shape[0], seq_len, -1)

    prefix_loss = F.smooth_l1_loss(pred_prefix, flat["prefix_credit"])
    delta_loss = F.smooth_l1_loss(pred_delta, flat["token_credit"])
    item_loss = F.smooth_l1_loss(pred_item, flat["item_credit"])
    cons_item_loss = F.smooth_l1_loss(pred_delta.sum(dim=1), flat["item_credit"])
    actor_loss, actor_stats = compute_actor_loss(
        base_logits,
        residual_scores,
        flat["target_tokens"],
        flat["token_credit"],
        flat["item_credit"],
        neg_scale=float(args.actor_neg_scale),
        anchor_scale=float(args.actor_anchor_scale),
        credit_clip=float(args.credit_clip),
        residual_clip=float(args.residual_clip),
    )

    bsz = int(input_ids.shape[0])
    max_items = int(item_mask.shape[1])
    pred_item_full = torch.zeros((bsz, max_items), dtype=torch.float32, device=device)
    pred_item_full[item_mask] = pred_item
    page_cons_loss = F.smooth_l1_loss(plan_credit_pred + pred_item_full.sum(dim=1), slate_credit)
    slate_loss = F.smooth_l1_loss(slate_credit_pred, slate_credit)
    plan_cls_loss = F.cross_entropy(plan_logits, plan_label)
    plan_credit_loss = F.smooth_l1_loss(plan_credit_pred, plan_credit)

    total_loss = (
        float(args.plan_cls_loss_scale) * plan_cls_loss
        + float(args.plan_credit_loss_scale) * plan_credit_loss
        + float(args.slate_loss_scale) * slate_loss
        + float(args.prefix_loss_scale) * prefix_loss
        + float(args.delta_loss_scale) * delta_loss
        + float(args.item_loss_scale) * item_loss
        + float(args.page_cons_scale) * page_cons_loss
        + float(args.item_cons_scale) * cons_item_loss
        + float(args.actor_loss_scale) * actor_loss
    )
    stats = {
        "loss": float(total_loss.item()),
        "plan_cls_loss": float(plan_cls_loss.item()),
        "plan_credit_loss": float(plan_credit_loss.item()),
        "slate_loss": float(slate_loss.item()),
        "prefix_loss": float(prefix_loss.item()),
        "delta_loss": float(delta_loss.item()),
        "item_loss": float(item_loss.item()),
        "page_cons_loss": float(page_cons_loss.item()),
        "item_cons_loss": float(cons_item_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "target_gain": float(actor_stats["target_gain"]),
        "plan_acc": float((plan_logits.argmax(dim=-1) == plan_label).float().mean().item()),
    }
    return total_loss, stats


@torch.no_grad()
def evaluate_joint(
    tiger,
    plan_head: HistoryPlanHead,
    slate_head: SlateCreditHead,
    prefix_head: PlanConditionedPrefixValueHead,
    token_actor_head: PlanConditionedTokenActorHead,
    loader: DataLoader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    plan_head.eval()
    slate_head.eval()
    prefix_head.eval()
    token_actor_head.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_joint(
            tiger,
            plan_head,
            slate_head,
            prefix_head,
            token_actor_head,
            batch,
            device,
            args,
        )
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(device))
    page_rows, sid_depth, vocab_size = load_page_rows(
        Path(args.chain_path),
        reader,
        str(args.sid_mapping_path),
        int(args.max_hist_items),
        str(args.token_credit_field),
        str(args.item_credit_field),
        float(args.target_clip),
        int(args.max_pages),
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

    dataset = JointPageDataset(page_rows)
    groups = [row["group"] for row in page_rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_pages)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_pages)

    n_plans = int(len(PLAN_NAMES))
    plan_head = HistoryPlanHead(int(size_cfg["d_model"]), n_plans, mlp_dim=int(args.plan_mlp_dim)).to(device)
    slate_head = SlateCreditHead(int(size_cfg["d_model"]), mlp_dim=int(args.slate_mlp_dim)).to(device)
    prefix_head = PlanConditionedPrefixValueHead(
        int(size_cfg["d_model"]),
        int(vocab_size),
        n_plans,
        token_dim=int(args.token_dim),
        plan_dim=int(args.plan_dim),
        mlp_dim=int(args.mlp_dim),
    ).to(device)
    token_actor_head = PlanConditionedTokenActorHead(
        int(size_cfg["d_model"]),
        int(vocab_size),
        n_plans,
        token_dim=int(args.token_dim),
        plan_dim=int(args.plan_dim),
        mlp_dim=int(args.mlp_dim),
    ).to(device)

    if str(args.init_joint_head_path).strip():
        payload = torch.load(str(args.init_joint_head_path), map_location="cpu")
        if "plan_head_state_dict" in payload:
            plan_head.load_state_dict(payload["plan_head_state_dict"], strict=False)
        if "slate_head_state_dict" in payload:
            slate_head.load_state_dict(payload["slate_head_state_dict"], strict=False)
        if "prefix_head_state_dict" in payload:
            prefix_head.load_state_dict(payload["prefix_head_state_dict"], strict=False)
        if "token_actor_head_state_dict" in payload:
            token_actor_head.load_state_dict(payload["token_actor_head_state_dict"], strict=False)
        print(f"[phase6] warm-started from {args.init_joint_head_path}")

    optimizer = torch.optim.AdamW(
        list(plan_head.parameters())
        + list(slate_head.parameters())
        + list(prefix_head.parameters())
        + list(token_actor_head.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_key = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        plan_head.train()
        slate_head.train()
        prefix_head.train()
        token_actor_head.train()
        train_losses: List[float] = []
        train_plan_acc: List[float] = []
        for batch in train_loader:
            loss, stats = forward_joint(
                tiger,
                plan_head,
                slate_head,
                prefix_head,
                token_actor_head,
                batch,
                device,
                args,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_plan_acc.append(float(stats["plan_acc"]))

        valid_metrics = evaluate_joint(
            tiger,
            plan_head,
            slate_head,
            prefix_head,
            token_actor_head,
            valid_loader,
            device,
            args,
        )
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        valid_metrics["train_plan_acc"] = float(np.mean(train_plan_acc)) if train_plan_acc else 0.0
        history.append(dict(valid_metrics))
        key = float(valid_metrics["loss"])
        if key < best_key:
            best_key = key
            best_state = {
                "plan_head_state_dict": {k: v.detach().cpu() for k, v in plan_head.state_dict().items()},
                "slate_head_state_dict": {k: v.detach().cpu() for k, v in slate_head.state_dict().items()},
                "prefix_head_state_dict": {k: v.detach().cpu() for k, v in prefix_head.state_dict().items()},
                "token_actor_head_state_dict": {k: v.detach().cpu() for k, v in token_actor_head.state_dict().items()},
            }
            best_metrics = dict(valid_metrics)
            best_epoch = int(epoch)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_plan_acc={valid_metrics['plan_acc']:.4f} "
            f"valid_target_gain={valid_metrics['target_gain']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Joint training produced no checkpoint.")

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.tiger_ckpt).resolve().parent / "phase6_joint"
    save_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_dir / "phase6_joint_heads.pt"
    meta_path = save_dir / "phase6_joint_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "phase6_joint_metrics.json"

    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Phase6 Joint Hierarchical",
        "chain_path": str(Path(args.chain_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "hidden_size": int(size_cfg["d_model"]),
        "vocab_size": int(vocab_size),
        "sid_depth": int(sid_depth),
        "n_plans": int(n_plans),
        "plan_names": PLAN_NAMES,
        "token_dim": int(args.token_dim),
        "plan_dim": int(args.plan_dim),
        "mlp_dim": int(args.mlp_dim),
        "plan_mlp_dim": int(args.plan_mlp_dim),
        "slate_mlp_dim": int(args.slate_mlp_dim),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_pages": int(len(page_rows)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "init_joint_head_path": str(Path(args.init_joint_head_path).resolve()) if str(args.init_joint_head_path).strip() else "",
    }
    write_json(meta_path, meta)
    write_json(
        metrics_path,
        {
            "head_path": str(head_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "history": history,
            "n_pages": int(len(page_rows)),
        },
    )
    print(f"[phase6] saved joint heads to {head_path}")
    print(f"[phase6] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
