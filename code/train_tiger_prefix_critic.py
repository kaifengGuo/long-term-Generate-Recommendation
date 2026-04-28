import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log
from tiger_phase2_blend_common import (
    TokenPrefixValueHead,
    build_iid2sid_tokens,
    build_history_tokens,
    decoder_input_ids_from_targets,
    load_tiger_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TIGER token-prefix critic on explicit phase3 chain records.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size", type=str, default="base", choices=["mini", "small", "base", "large"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--token_credit_field", type=str, default="token_credit_calibrated")
    parser.add_argument("--item_credit_field", type=str, default="item_credit")
    parser.add_argument("--target_clip", type=float, default=0.0)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prefix_loss_scale", type=float, default=1.0)
    parser.add_argument("--delta_loss_scale", type=float, default=1.0)
    parser.add_argument("--item_loss_scale", type=float, default=0.5)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    return parser.parse_args()


def apply_size_defaults(args: argparse.Namespace) -> None:
    size = str(args.size).lower()
    if size == "mini":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif size == "small":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif size == "base":
        args.num_layers = 4
        args.num_decoder_layers = 4
        args.d_model = 128
        args.d_ff = 1024
        args.num_heads = 6
        args.d_kv = 64
    elif size == "large":
        args.num_layers = 6
        args.num_decoder_layers = 6
        args.d_model = 192
        args.d_ff = 1536
        args.num_heads = 8
        args.d_kv = 24


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, g in enumerate(groups):
        if g in valid_groups:
            valid_idx.append(idx)
        else:
            train_idx.append(idx)
    if not train_idx and valid_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx and train_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


class PrefixCriticDataset(Dataset):
    def __init__(
        self,
        chain_path: Path,
        *,
        iid2sid_tok_cpu: torch.Tensor,
        sid_depth: int,
        max_hist_items: int,
        token_credit_field: str,
        item_credit_field: str,
        target_clip: float,
        max_records: int,
    ):
        self.rows: List[Dict[str, Any]] = []
        with chain_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
                token_credit = payload.get(str(token_credit_field), payload.get("token_credit_calibrated", payload.get("token_credit", [])))
                history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
                item_credit = float(payload.get(str(item_credit_field), payload.get("item_credit", 0.0)))
                if len(target_tokens) != int(sid_depth) or len(token_credit) != int(sid_depth):
                    continue
                if not any(int(x) > 0 for x in target_tokens):
                    continue
                token_credit_arr = np.asarray(token_credit, dtype=np.float32)
                prefix_credit = np.cumsum(token_credit_arr, dtype=np.float32)
                if float(target_clip) > 0.0:
                    token_credit_arr = np.clip(token_credit_arr, -float(target_clip), float(target_clip))
                    prefix_credit = np.cumsum(token_credit_arr, dtype=np.float32)
                    item_credit = float(prefix_credit[-1])
                hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
                input_ids, attention_mask = build_history_tokens(
                    hist_tensor,
                    iid2sid_tok_cpu,
                    int(max_hist_items),
                    int(sid_depth),
                )
                self.rows.append(
                    {
                        "input_ids": input_ids.squeeze(0),
                        "attention_mask": attention_mask.squeeze(0),
                        "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
                        "token_credit": torch.tensor(token_credit_arr, dtype=torch.float32),
                        "prefix_credit": torch.tensor(prefix_credit, dtype=torch.float32),
                        "item_credit": torch.tensor(float(item_credit), dtype=torch.float32),
                        "group": str(payload.get("episode_id", len(self.rows))),
                    }
                )
                if int(max_records) > 0 and len(self.rows) >= int(max_records):
                    break
        if not self.rows:
            raise ValueError(f"No usable rows found in {chain_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_prefix(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "target_tokens": torch.stack([x["target_tokens"] for x in batch], dim=0),
        "token_credit": torch.stack([x["token_credit"] for x in batch], dim=0),
        "prefix_credit": torch.stack([x["prefix_credit"] for x in batch], dim=0),
        "item_credit": torch.stack([x["item_credit"] for x in batch], dim=0),
        "groups": [x["group"] for x in batch],
    }


def prefix_to_delta(prefix_values: torch.Tensor) -> torch.Tensor:
    first = prefix_values[:, :1]
    if prefix_values.size(1) <= 1:
        return first
    rest = prefix_values[:, 1:] - prefix_values[:, :-1]
    return torch.cat([first, rest], dim=1)


def evaluate_prefix_head(
    tiger,
    head: TokenPrefixValueHead,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    head.eval()
    prefix_losses: List[float] = []
    delta_losses: List[float] = []
    item_losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            prefix_credit = batch["prefix_credit"].to(device)
            token_credit = batch["token_credit"].to(device)
            item_credit = batch["item_credit"].to(device)
            _logits, hidden = tiger.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
            )
            pred_prefix = head(hidden.detach(), target_tokens)
            pred_delta = prefix_to_delta(pred_prefix)
            pred_item = pred_prefix[:, -1]
            prefix_losses.append(float(F.smooth_l1_loss(pred_prefix, prefix_credit).item()))
            delta_losses.append(float(F.smooth_l1_loss(pred_delta, token_credit).item()))
            item_losses.append(float(F.mse_loss(pred_item, item_credit).item()))
    return {
        "prefix_loss": float(np.mean(prefix_losses)) if prefix_losses else 0.0,
        "delta_loss": float(np.mean(delta_losses)) if delta_losses else 0.0,
        "item_loss": float(np.mean(item_losses)) if item_losses else 0.0,
    }


def main() -> None:
    args = parse_args()
    apply_size_defaults(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reader = load_reader_from_uirm_log(args.uirm_log_path, str(device))
    tiger, sid_depth, codebook_size = load_tiger_model(
        tiger_ckpt=args.tiger_ckpt,
        sid_mapping_path=args.sid_mapping_path,
        num_layers=int(args.num_layers),
        num_decoder_layers=int(args.num_decoder_layers),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        num_heads=int(args.num_heads),
        d_kv=int(args.d_kv),
        dropout_rate=float(args.dropout_rate),
        feed_forward_proj=str(args.feed_forward_proj),
        device=device,
    )
    iid2sid_tok, _sid2iid_map = build_iid2sid_tokens(reader, args.sid_mapping_path, sid_depth, device)
    dataset = PrefixCriticDataset(
        Path(args.chain_path),
        iid2sid_tok_cpu=iid2sid_tok.cpu(),
        sid_depth=int(sid_depth),
        max_hist_items=int(args.max_hist_items),
        token_credit_field=str(args.token_credit_field),
        item_credit_field=str(args.item_credit_field),
        target_clip=float(args.target_clip),
        max_records=int(args.max_records),
    )
    groups = [row["group"] for row in dataset.rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_prefix,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_prefix,
    )

    head = TokenPrefixValueHead(
        hidden_size=int(args.d_model),
        vocab_size=int(codebook_size) + 1,
        token_dim=int(args.token_dim),
        mlp_dim=int(args.mlp_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_valid = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        train_losses: List[float] = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            token_credit = batch["token_credit"].to(device)
            prefix_credit = batch["prefix_credit"].to(device)
            item_credit = batch["item_credit"].to(device)
            with torch.no_grad():
                _logits, hidden = tiger.decode_with_hidden(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
                )
            pred_prefix = head(hidden.detach(), target_tokens)
            pred_delta = prefix_to_delta(pred_prefix)
            pred_item = pred_prefix[:, -1]
            loss_prefix = F.smooth_l1_loss(pred_prefix, prefix_credit)
            loss_delta = F.smooth_l1_loss(pred_delta, token_credit)
            loss_item = F.mse_loss(pred_item, item_credit)
            loss = (
                float(args.prefix_loss_scale) * loss_prefix
                + float(args.delta_loss_scale) * loss_delta
                + float(args.item_loss_scale) * loss_item
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_metrics = evaluate_prefix_head(tiger, head, valid_loader, device, args)
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "valid_prefix_loss": float(valid_metrics["prefix_loss"]),
            "valid_delta_loss": float(valid_metrics["delta_loss"]),
            "valid_item_loss": float(valid_metrics["item_loss"]),
        }
        history.append(record)
        score = float(valid_metrics["prefix_loss"] + valid_metrics["delta_loss"] + 0.5 * valid_metrics["item_loss"])
        print(json.dumps(record, ensure_ascii=True))
        if score < best_valid:
            best_valid = score
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            best_epoch = int(epoch)

    if best_state is None:
        raise RuntimeError("Prefix critic training produced no checkpoint.")

    head_path = save_dir / "prefix_critic_head.pt"
    meta_path = save_dir / "prefix_critic_meta.json"
    metrics_path = save_dir / "prefix_critic_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Prefix Critic",
        "chain_path": str(Path(args.chain_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "head_path": str(head_path.resolve()),
        "token_credit_field": str(args.token_credit_field),
        "item_credit_field": str(args.item_credit_field),
        "max_hist_items": int(args.max_hist_items),
        "sid_depth": int(sid_depth),
        "vocab_size": int(codebook_size) + 1,
        "token_dim": int(args.token_dim),
        "mlp_dim": int(args.mlp_dim),
        "size": str(args.size),
        "num_layers": int(args.num_layers),
        "num_decoder_layers": int(args.num_decoder_layers),
        "d_model": int(args.d_model),
        "d_ff": int(args.d_ff),
        "num_heads": int(args.num_heads),
        "d_kv": int(args.d_kv),
        "dropout_rate": float(args.dropout_rate),
        "feed_forward_proj": str(args.feed_forward_proj),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_valid),
        "n_rows": int(len(dataset)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
    }
    metrics = {
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_valid),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Saved prefix critic to {head_path}")
    print(f"[Done] Meta: {meta_path}")
    print(f"[Done] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
