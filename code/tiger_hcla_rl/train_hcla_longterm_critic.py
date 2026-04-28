import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_phase2_blend_common import decoder_input_ids_from_targets, infer_model_size_args, load_tiger_model  # noqa: E402

from tiger_hcla_rl.common import (  # noqa: E402
    calibrate_token_delta,
    pooled_history_summary,
    prefix_to_delta,
    set_random_seed,
    split_groups,
    write_json,
)
from tiger_hcla_rl.models import (  # noqa: E402
    ItemLongTermCritic,
    PageLongTermCritic,
    TokenLongTermCritic,
    save_critic_bundle,
)


class HCLARowDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train page/item/token long-term critics for TIGER HCLA-RL.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--page_loss_scale", type=float, default=1.0)
    parser.add_argument("--item_loss_scale", type=float, default=1.0)
    parser.add_argument("--prefix_loss_scale", type=float, default=0.5)
    parser.add_argument("--delta_loss_scale", type=float, default=1.0)
    parser.add_argument("--item_cons_scale", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


def load_chain_rows(chain_path: Path, max_rows: int) -> Tuple[List[Dict[str, Any]], int, int, int]:
    rows: List[Dict[str, Any]] = []
    page_dim = 0
    item_dim = 0
    vocab_size = 0
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
            page_features = [float(x) for x in payload.get("page_features", [])]
            item_features = [float(x) for x in payload.get("item_features", [])]
            rows.append(
                {
                    "group": str(payload.get("episode_id", "na")),
                    "input_ids": [int(x) for x in payload.get("input_ids", [])],
                    "attention_mask": [int(x) for x in payload.get("attention_mask", [])],
                    "target_tokens": target_tokens,
                    "page_features": page_features,
                    "item_features": item_features,
                    "page_value": float(payload.get("lt_page_value_raw", 0.0)),
                    "item_adv": float(payload.get("lt_item_cf_adv", 0.0)),
                    "token_adv": token_adv,
                    "token_prefix": [float(x) for x in payload.get("lt_token_cf_prefix", [])],
                }
            )
            page_dim = max(page_dim, len(page_features))
            item_dim = max(item_dim, len(item_features))
            vocab_size = max(vocab_size, max(target_tokens) + 1)
    if not rows:
        raise ValueError(f"No usable rows in {chain_path}")
    return rows, int(page_dim), int(item_dim), int(vocab_size)


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], dim=0),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], dim=0),
        "target_tokens": torch.stack([torch.tensor(x["target_tokens"], dtype=torch.long) for x in batch], dim=0),
        "page_features": torch.stack([torch.tensor(x["page_features"], dtype=torch.float32) for x in batch], dim=0),
        "item_features": torch.stack([torch.tensor(x["item_features"], dtype=torch.float32) for x in batch], dim=0),
        "page_value": torch.tensor([float(x["page_value"]) for x in batch], dtype=torch.float32),
        "item_adv": torch.tensor([float(x["item_adv"]) for x in batch], dtype=torch.float32),
        "token_adv": torch.stack([torch.tensor(x["token_adv"], dtype=torch.float32) for x in batch], dim=0),
        "token_prefix": torch.stack([torch.tensor(x["token_prefix"], dtype=torch.float32) for x in batch], dim=0),
        "groups": [x["group"] for x in batch],
    }


def forward_batch(
    tiger,
    page_head: PageLongTermCritic,
    item_head: ItemLongTermCritic,
    token_head: TokenLongTermCritic,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_tokens = batch["target_tokens"].to(device)
    page_features = batch["page_features"].to(device)
    item_features = batch["item_features"].to(device)
    page_value = batch["page_value"].to(device)
    item_adv = batch["item_adv"].to(device)
    token_adv = batch["token_adv"].to(device)
    token_prefix = batch["token_prefix"].to(device)

    with torch.no_grad():
        history_summary = pooled_history_summary(tiger, input_ids, attention_mask)
        decoder_input_ids = decoder_input_ids_from_targets(target_tokens)
        _logits, hidden = tiger.decode_with_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    page_pred = page_head(history_summary.detach(), page_features)
    item_pred = item_head(history_summary.detach(), page_features, item_features)
    token_prefix_pred = token_head(hidden.detach(), target_tokens)
    token_delta_pred = prefix_to_delta(token_prefix_pred)
    token_delta_pred = calibrate_token_delta(token_delta_pred, item_pred)

    page_loss = F.smooth_l1_loss(page_pred, page_value)
    item_loss = F.smooth_l1_loss(item_pred, item_adv)
    prefix_loss = F.smooth_l1_loss(token_prefix_pred, token_prefix)
    delta_loss = F.smooth_l1_loss(token_delta_pred, token_adv)
    item_cons_loss = F.smooth_l1_loss(token_delta_pred.sum(dim=-1), item_adv)
    loss = (
        float(args.page_loss_scale) * page_loss
        + float(args.item_loss_scale) * item_loss
        + float(args.prefix_loss_scale) * prefix_loss
        + float(args.delta_loss_scale) * delta_loss
        + float(args.item_cons_scale) * item_cons_loss
    )
    stats = {
        "loss": float(loss.item()),
        "page_loss": float(page_loss.item()),
        "item_loss": float(item_loss.item()),
        "prefix_loss": float(prefix_loss.item()),
        "delta_loss": float(delta_loss.item()),
        "item_cons_loss": float(item_cons_loss.item()),
        "page_mae": float((page_pred - page_value).abs().mean().item()),
        "item_mae": float((item_pred - item_adv).abs().mean().item()),
        "token_mae": float((token_delta_pred - token_adv).abs().mean().item()),
    }
    return loss, stats


@torch.no_grad()
def evaluate(
    tiger,
    page_head: PageLongTermCritic,
    item_head: ItemLongTermCritic,
    token_head: TokenLongTermCritic,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    page_head.eval()
    item_head.eval()
    token_head.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_batch(tiger, page_head, item_head, token_head, batch, device, args)
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

    rows, page_dim, item_dim, vocab_size_from_data = load_chain_rows(chain_path, int(args.max_rows))
    groups = [str(x["group"]) for x in rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(HCLARowDataset(rows), train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_rows)
    valid_loader = DataLoader(Subset(HCLARowDataset(rows), valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_rows)

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth, vocab_size_model = load_tiger_model(
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

    hidden_size = int(size_cfg["d_model"])
    page_head = PageLongTermCritic(hidden_size=hidden_size, page_dim=int(page_dim), mlp_dim=int(args.mlp_dim)).to(device)
    item_head = ItemLongTermCritic(hidden_size=hidden_size, page_dim=int(page_dim), item_dim=int(item_dim), mlp_dim=int(args.mlp_dim)).to(device)
    token_head = TokenLongTermCritic(hidden_size=hidden_size, vocab_size=max(int(vocab_size_model), int(vocab_size_from_data)), token_dim=int(args.token_dim), mlp_dim=int(args.mlp_dim)).to(device)

    optimizer = torch.optim.AdamW(
        list(page_head.parameters()) + list(item_head.parameters()) + list(token_head.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    bundle_path = save_dir / "critic_bundle.pt"
    meta_path = save_dir / "critic_bundle_meta.json"

    for epoch in range(1, int(args.epochs) + 1):
        page_head.train()
        item_head.train()
        token_head.train()
        train_metrics: Dict[str, List[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = forward_batch(tiger, page_head, item_head, token_head, batch, device, args)
            loss.backward()
            optimizer.step()
            for key, value in stats.items():
                train_metrics.setdefault(key, []).append(float(value))
        train_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in train_metrics.items()}
        valid_mean = evaluate(tiger, page_head, item_head, token_head, valid_loader, device, args)
        history.append({"epoch": int(epoch), "train": train_mean, "valid": valid_mean})
        if float(valid_mean.get("loss", float("inf"))) < float(best_score):
            best_score = float(valid_mean["loss"])
            best_epoch = int(epoch)
            meta = {
                "method": "TIGER HCLA-RL critic",
                "hidden_size": int(hidden_size),
                "page_dim": int(page_dim),
                "item_dim": int(item_dim),
                "vocab_size": max(int(vocab_size_model), int(vocab_size_from_data)),
                "mlp_dim": int(args.mlp_dim),
                "token_dim": int(args.token_dim),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_score),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
            save_critic_bundle(
                bundle_path,
                meta_path,
                page_head=page_head,
                item_head=item_head,
                token_head=token_head,
                meta=meta,
            )

    metrics = {
        "method": "TIGER HCLA-RL critic",
        "chain_path": str(chain_path.resolve()),
        "critic_bundle_path": str(bundle_path.resolve()),
        "critic_meta_path": str(meta_path.resolve()),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_score),
        "history": history,
    }
    metrics_out = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "critic_metrics.json"
    write_json(metrics_out, metrics)
    print(f"[hcla-critic] best_valid_loss={best_score:.6f} epoch={best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
