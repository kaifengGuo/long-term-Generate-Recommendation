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
    parser = argparse.ArgumentParser(
        description="TIGER-HCA-WSFT: conservative weighted SFT / unlikelihood actor update from hierarchical advantages."
    )
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--token_adv_field", type=str, default="sid_advantage")
    parser.add_argument("--item_adv_field", type=str, default="item_advantage")
    parser.add_argument("--page_adv_field", type=str, default="page_q_value")
    parser.add_argument("--min_abs_adv", type=float, default=1e-6)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--item_adv_scale", type=float, default=0.10)
    parser.add_argument("--page_adv_scale", type=float, default=0.10)
    parser.add_argument("--page_gate_scale", type=float, default=0.10)
    parser.add_argument("--page_gate_min", type=float, default=0.85)
    parser.add_argument("--page_gate_max", type=float, default=1.15)
    parser.add_argument("--positive_topk", type=int, default=2)
    parser.add_argument("--positive_floor", type=float, default=0.0)
    parser.add_argument("--negative_topk", type=int, default=1)
    parser.add_argument("--negative_floor", type=float, default=0.0)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--entropy_scale", type=float, default=0.0)
    parser.add_argument("--mle_scale", type=float, default=0.10)
    parser.add_argument("--neg_scale", type=float, default=0.25)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class HCAWSFTDataset(Dataset):
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
        "page_adv": torch.tensor([float(x["page_adv"]) for x in batch], dtype=torch.float32),
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


def load_chain_rows(
    chain_path: Path,
    reader,
    sid_mapping_path: str,
    max_hist_items: int,
    token_adv_field: str,
    item_adv_field: str,
    page_adv_field: str,
    min_abs_adv: float,
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    rows: List[Dict[str, Any]] = []
    with chain_path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
            token_adv = [float(x) for x in payload.get(str(token_adv_field), [])]
            item_adv = float(payload.get(str(item_adv_field), 0.0))
            page_adv = float(payload.get(str(page_adv_field), 0.0))
            if len(target_tokens) != int(sid_depth) or len(token_adv) != int(sid_depth):
                continue
            if max(
                abs(float(item_adv)),
                abs(float(page_adv)),
                max((abs(float(x)) for x in token_adv), default=0.0),
            ) < float(min_abs_adv):
                continue
            history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(max_hist_items),
                int(sid_depth),
            )
            rows.append(
                {
                    "group": str(payload.get("episode_id", "na")),
                    "input_ids": input_ids.squeeze(0).tolist(),
                    "attention_mask": attention_mask.squeeze(0).tolist(),
                    "target_tokens": target_tokens,
                    "token_adv": token_adv,
                    "item_adv": item_adv,
                    "page_adv": page_adv,
                }
            )
    if not rows:
        raise ValueError(f"No usable chain rows in {chain_path}")
    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    return rows, sid_depth, vocab_size


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


def normalize_weights(scores: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    masked = scores * active_mask
    denom = masked.sum(dim=-1, keepdim=True)
    fallback = active_mask.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        denom > 1e-8,
        masked / denom.clamp_min(1e-8),
        torch.where(fallback > 0.0, active_mask / fallback.clamp_min(1.0), torch.zeros_like(active_mask)),
    )
    return normalized


def compute_wsft_loss(
    actor_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    token_adv: torch.Tensor,
    item_adv: torch.Tensor,
    page_adv: torch.Tensor,
    *,
    item_adv_scale: float,
    page_adv_scale: float,
    page_gate_scale: float,
    page_gate_min: float,
    page_gate_max: float,
    positive_topk: int,
    positive_floor: float,
    negative_topk: int,
    negative_floor: float,
    credit_clip: float,
    renorm_mode: str,
    kl_scale: float,
    entropy_scale: float,
    mle_scale: float,
    neg_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if float(credit_clip) > 0.0:
        token_adv = token_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        item_adv = item_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
        page_adv = page_adv.clamp(min=-float(credit_clip), max=float(credit_clip))
    token_adv = renorm_signal(token_adv, str(renorm_mode))
    item_adv = renorm_signal(item_adv.unsqueeze(-1), str(renorm_mode)).squeeze(-1)
    page_adv = renorm_signal(page_adv.unsqueeze(-1), str(renorm_mode)).squeeze(-1)

    actor_log_probs = torch.log_softmax(actor_logits, dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits.detach(), dim=-1)
    ref_probs = ref_log_probs.exp()
    actor_target_logp = actor_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    ref_target_logp = ref_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    target_prob = actor_target_logp.exp()
    ce = F.cross_entropy(
        actor_logits.reshape(-1, actor_logits.shape[-1]),
        target_tokens.reshape(-1),
        reduction="none",
    ).view_as(target_tokens)

    page_gate = 1.0 + float(page_gate_scale) * torch.tanh(page_adv)
    page_gate = page_gate.clamp(min=float(page_gate_min), max=float(page_gate_max)).unsqueeze(-1)
    item_bias_pos = float(item_adv_scale) * torch.relu(item_adv).unsqueeze(-1)
    item_bias_neg = float(item_adv_scale) * torch.relu(-item_adv).unsqueeze(-1)
    page_bias_pos = float(page_adv_scale) * torch.relu(page_adv).unsqueeze(-1)
    page_bias_neg = float(page_adv_scale) * torch.relu(-page_adv).unsqueeze(-1)

    pos_scores = torch.relu(token_adv) + item_bias_pos + page_bias_pos
    neg_scores = torch.relu(-token_adv) + item_bias_neg + page_bias_neg
    pos_mask = build_sparse_mask(pos_scores, int(positive_topk), float(positive_floor))
    neg_mask = build_sparse_mask(neg_scores, int(negative_topk), float(negative_floor))
    active_mask = (pos_mask + neg_mask).clamp(max=1.0)

    missing = active_mask.sum(dim=-1) <= 0
    if bool(missing.any()):
        fallback_idx = token_adv[missing].abs().argmax(dim=-1, keepdim=True)
        pos_mask[missing] = 0.0
        neg_mask[missing] = 0.0
        active_mask[missing] = 0.0
        active_mask[missing].scatter_(1, fallback_idx, 1.0)
        pos_mask[missing].scatter_(1, fallback_idx, 1.0)

    pos_weights = normalize_weights(pos_scores, pos_mask)
    neg_weights = normalize_weights(neg_scores, neg_mask)
    pos_weighted = page_gate * pos_weights
    neg_weighted = page_gate * neg_weights

    pos_loss = (pos_weighted * ce).sum() / pos_weighted.sum().clamp_min(1e-8)
    if float(neg_weighted.sum().item()) > 0.0 and float(neg_scale) > 0.0:
        neg_loss = (neg_weighted * target_prob).sum() / neg_weighted.sum().clamp_min(1e-8)
    else:
        neg_loss = torch.zeros((), dtype=actor_logits.dtype, device=actor_logits.device)
    mle_loss = ce.mean()
    kl_loss = F.kl_div(actor_log_probs, ref_probs, reduction="batchmean", log_target=False)
    entropy = -(actor_log_probs.exp() * actor_log_probs).sum(dim=-1)
    entropy_bonus = (entropy * active_mask).sum() / active_mask.sum().clamp_min(1e-8)
    loss = pos_loss + float(neg_scale) * neg_loss + float(mle_scale) * mle_loss + float(kl_scale) * kl_loss - float(entropy_scale) * entropy_bonus

    target_gain = ((actor_target_logp.exp() - ref_target_logp.exp()) * active_mask).sum() / active_mask.sum().clamp_min(1e-8)
    approx_kl = ((ref_target_logp - actor_target_logp) * active_mask).sum() / active_mask.sum().clamp_min(1e-8)
    stats = {
        "loss": float(loss.item()),
        "pos_loss": float(pos_loss.item()),
        "neg_loss": float(neg_loss.item()),
        "mle_loss": float(mle_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "entropy_bonus": float(entropy_bonus.item()),
        "target_gain": float(target_gain.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_frac": 0.0,
        "ratio_mean": 1.0,
        "signed_adv_mean": float((token_adv * active_mask).sum().item() / active_mask.sum().item()),
        "signed_adv_abs": float((token_adv.abs() * active_mask).sum().item() / active_mask.sum().item()),
        "page_gate_mean": float(page_gate.mean().item()),
        "active_frac": float(active_mask.mean().item()),
        "pos_active_frac": float(pos_mask.mean().item()),
        "neg_active_frac": float(neg_mask.mean().item()),
        "pos_selected_per_row": float(pos_mask.sum(dim=-1).mean().item()),
        "neg_selected_per_row": float(neg_mask.sum(dim=-1).mean().item()),
        "target_prob": float((target_prob * active_mask).sum().item() / active_mask.sum().item()),
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
    target_tokens = batch["target_tokens"].to(device)
    token_adv = batch["token_adv"].to(device)
    item_adv = batch["item_adv"].to(device)
    page_adv = batch["page_adv"].to(device)
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)

    with torch.no_grad():
        ref_logits, _ = ref_tiger.decode_with_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    actor_logits, _ = actor_tiger.decode_with_hidden(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    return compute_wsft_loss(
        actor_logits,
        ref_logits,
        target_tokens,
        token_adv,
        item_adv,
        page_adv,
        item_adv_scale=float(args.item_adv_scale),
        page_adv_scale=float(args.page_adv_scale),
        page_gate_scale=float(args.page_gate_scale),
        page_gate_min=float(args.page_gate_min),
        page_gate_max=float(args.page_gate_max),
        positive_topk=int(args.positive_topk),
        positive_floor=float(args.positive_floor),
        negative_topk=int(args.negative_topk),
        negative_floor=float(args.negative_floor),
        credit_clip=float(args.credit_clip),
        renorm_mode=str(args.renorm_mode),
        kl_scale=float(args.kl_scale),
        entropy_scale=float(args.entropy_scale),
        mle_scale=float(args.mle_scale),
        neg_scale=float(args.neg_scale),
    )


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

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(device))
    chain_rows, sid_depth, vocab_size = load_chain_rows(
        Path(args.chain_path),
        reader,
        str(args.sid_mapping_path),
        int(args.max_hist_items),
        str(args.token_adv_field),
        str(args.item_adv_field),
        str(args.page_adv_field),
        float(args.min_abs_adv),
        int(args.max_rows),
    )

    size_cfg = infer_model_size_args(str(args.model_size))
    ref_tiger, _sid_depth_model, _codebook_size = load_tiger_model(
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
    actor_tiger, _sid_depth_model2, _codebook_size2 = load_tiger_model(
        tiger_ckpt=actor_init_ckpt,
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

    dataset = HCAWSFTDataset(chain_rows)
    groups = [row["group"] for row in chain_rows]
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
        train_approx_kl: List[float] = []
        for batch in train_loader:
            loss, stats = forward_actor(actor_tiger, ref_tiger, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_target_gain.append(float(stats["target_gain"]))
            train_approx_kl.append(float(stats["approx_kl"]))

        valid_metrics = evaluate_actor(actor_tiger, ref_tiger, valid_loader, device, args)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        valid_metrics["train_target_gain"] = float(np.mean(train_target_gain)) if train_target_gain else 0.0
        valid_metrics["train_approx_kl"] = float(np.mean(train_approx_kl)) if train_approx_kl else 0.0
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
            f"active_frac={valid_metrics['active_frac']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("TIGER-HCA-WSFT training produced no checkpoint.")

    save_dir = Path(args.save_dir) if str(args.save_dir).strip() else Path(args.tiger_ckpt).resolve().parent / "tiger_hca_wsft_actor"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "tiger_hca_wsft_actor_tiger.pth"
    meta_path = save_dir / "tiger_hca_wsft_actor_meta.json"
    metrics_path = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "tiger_hca_wsft_actor_metrics.json"
    torch.save(best_state, ckpt_path)
    meta = {
        "method": "TIGER-HCA-WSFT Actor",
        "chain_path": str(Path(args.chain_path).resolve()),
        "old_policy_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "vocab_size": int(vocab_size),
        "token_adv_field": str(args.token_adv_field),
        "item_adv_field": str(args.item_adv_field),
        "page_adv_field": str(args.page_adv_field),
        "train_scope": str(args.train_scope),
        "n_trainable": int(n_trainable),
        "mle_scale": float(args.mle_scale),
        "neg_scale": float(args.neg_scale),
        "kl_scale": float(args.kl_scale),
        "entropy_scale": float(args.entropy_scale),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_rows": int(len(chain_rows)),
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
    print(f"[hca-wsft] saved fine-tuned TIGER to {ckpt_path}")
    print(f"[hca-wsft] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
