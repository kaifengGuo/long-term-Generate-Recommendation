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
    TokenLongTermActorHead,
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
    parser = argparse.ArgumentParser(description="Train Phase3 actor head for TIGER from explicit credit chains.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--target_credit_field", type=str, default="token_credit_calibrated")
    parser.add_argument("--item_credit_field", type=str, default="item_credit")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--mlp_dim", type=int, default=256)
    parser.add_argument("--item_weight_scale", type=float, default=0.2)
    parser.add_argument("--neg_scale", type=float, default=0.25)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class Phase3ActorDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[int(idx)]
        return (
            torch.tensor(row["input_ids"], dtype=torch.long),
            torch.tensor(row["attention_mask"], dtype=torch.long),
            torch.tensor(row["target_tokens"], dtype=torch.long),
            torch.tensor(row["token_credit"], dtype=torch.float32),
            torch.tensor(float(row["item_credit"]), dtype=torch.float32),
            row["group"],
        )


def collate_actor(batch):
    input_ids = torch.stack([x[0] for x in batch], dim=0)
    attention_mask = torch.stack([x[1] for x in batch], dim=0)
    target_tokens = torch.stack([x[2] for x in batch], dim=0)
    token_credit = torch.stack([x[3] for x in batch], dim=0)
    item_credit = torch.stack([x[4] for x in batch], dim=0)
    groups = [x[5] for x in batch]
    return input_ids, attention_mask, target_tokens, token_credit, item_credit, groups


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


def load_chain_rows(
    chain_path: Path,
    reader,
    sid_mapping_path: str,
    max_hist_items: int,
    target_credit_field: str,
    item_credit_field: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    rows: List[Dict[str, Any]] = []
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
            history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
            token_credit = [float(x) for x in payload.get(str(target_credit_field), [])]
            item_credit = float(payload.get(str(item_credit_field), 0.0))
            if len(target_tokens) != sid_depth or len(token_credit) != sid_depth:
                continue
            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(max_hist_items),
                int(sid_depth),
            )
            rows.append(
                {
                    "group": str(payload["episode_id"]),
                    "input_ids": input_ids.squeeze(0).tolist(),
                    "attention_mask": attention_mask.squeeze(0).tolist(),
                    "target_tokens": target_tokens,
                    "token_credit": token_credit,
                    "item_credit": item_credit,
                }
            )
    if not rows:
        raise ValueError(f"No usable chain rows in {chain_path}")
    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    return rows, sid_depth, vocab_size


def compute_actor_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    token_credit: torch.Tensor,
    item_credit: torch.Tensor,
    *,
    item_weight_scale: float,
    neg_scale: float,
    credit_clip: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    vocab = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab)
    flat_target = target_tokens.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_target, reduction="none").view_as(target_tokens)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logp = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    target_prob = target_logp.exp()

    token_credit = token_credit.clamp(min=-float(credit_clip), max=float(credit_clip))
    pos_w = torch.relu(token_credit) + float(item_weight_scale) * torch.relu(item_credit).unsqueeze(-1)
    neg_w = torch.relu(-token_credit)
    pos_loss = (pos_w * ce).sum() / (pos_w.sum() + 1e-8)
    neg_loss = (neg_w * target_prob).sum() / (neg_w.sum() + 1e-8)
    loss = pos_loss + float(neg_scale) * neg_loss
    stats = {
        "pos_loss": float(pos_loss.item()),
        "neg_loss": float(neg_loss.item()),
        "pos_weight": float(pos_w.mean().item()),
        "neg_weight": float(neg_w.mean().item()),
        "target_prob": float(target_prob.mean().item()),
    }
    return loss, stats


def evaluate_actor(tiger, actor_head, loader: DataLoader, device: torch.device, args) -> Dict[str, float]:
    actor_head.eval()
    losses: List[float] = []
    probs: List[float] = []
    with torch.no_grad():
        for input_ids, attention_mask, target_tokens, token_credit, item_credit, _groups in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            token_credit = token_credit.to(device)
            item_credit = item_credit.to(device)
            _logits, hidden = tiger.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
            )
            logits = actor_head(hidden)
            loss, stats = compute_actor_loss(
                logits,
                target_tokens,
                token_credit,
                item_credit,
                item_weight_scale=float(args.item_weight_scale),
                neg_scale=float(args.neg_scale),
                credit_clip=float(args.credit_clip),
            )
            losses.append(float(loss.item()))
            probs.append(float(stats["target_prob"]))
    return {
        "actor_loss": float(np.mean(losses)) if losses else 0.0,
        "target_prob": float(np.mean(probs)) if probs else 0.0,
    }


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
        str(args.target_credit_field),
        str(args.item_credit_field),
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

    dataset = Phase3ActorDataset(chain_rows)
    groups = [row["group"] for row in chain_rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_actor)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_actor)

    actor_head = TokenLongTermActorHead(
        hidden_size=int(size_cfg["d_model"]),
        vocab_size=int(vocab_size),
        mlp_dim=int(args.mlp_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(actor_head.parameters(), lr=float(args.lr))

    best_key = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        actor_head.train()
        train_losses: List[float] = []
        train_probs: List[float] = []
        for input_ids, attention_mask, target_tokens, token_credit, item_credit, _groups in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            token_credit = token_credit.to(device)
            item_credit = item_credit.to(device)
            with torch.no_grad():
                _logits, hidden = tiger.decode_with_hidden(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
                )
            logits = actor_head(hidden.detach())
            loss, stats = compute_actor_loss(
                logits,
                target_tokens,
                token_credit,
                item_credit,
                item_weight_scale=float(args.item_weight_scale),
                neg_scale=float(args.neg_scale),
                credit_clip=float(args.credit_clip),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_probs.append(float(stats["target_prob"]))

        valid_metrics = evaluate_actor(tiger, actor_head, valid_loader, device, args)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        valid_metrics["train_target_prob"] = float(np.mean(train_probs)) if train_probs else 0.0
        history.append(dict(valid_metrics))
        key = float(valid_metrics["actor_loss"])
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in actor_head.state_dict().items()}
            best_metrics = dict(valid_metrics)
            best_epoch = int(epoch)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"valid_actor_loss={valid_metrics['actor_loss']:.4f} "
            f"valid_target_prob={valid_metrics['target_prob']:.4f}"
        )

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in actor_head.state_dict().items()}
        best_metrics = evaluate_actor(tiger, actor_head, valid_loader, device, args)
        best_epoch = int(args.epochs)

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.tiger_ckpt).resolve().parent / "phase3_actor"
    save_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_dir / "phase3_actor_head.pt"
    meta_path = save_dir / "phase3_actor_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "phase3_actor_metrics.json"

    torch.save({"model_state_dict": best_state}, head_path)
    meta = {
        "method": "TIGER Phase3 Calibrated Actor",
        "chain_path": str(Path(args.chain_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "hidden_size": int(size_cfg["d_model"]),
        "vocab_size": int(vocab_size),
        "sid_depth": int(sid_depth),
        "mlp_dim": int(args.mlp_dim),
        "target_credit_field": str(args.target_credit_field),
        "item_credit_field": str(args.item_credit_field),
        "item_weight_scale": float(args.item_weight_scale),
        "neg_scale": float(args.neg_scale),
        "credit_clip": float(args.credit_clip),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_rows": int(len(chain_rows)),
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
            "n_rows": int(len(chain_rows)),
        },
    )
    print(f"[phase3] saved actor head to {head_path}")
    print(f"[phase3] saved actor meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
