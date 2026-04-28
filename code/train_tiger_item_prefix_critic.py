import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log
from tiger_hier_prefix_common import ItemPrefixValueHead, load_item_prefix_head
from tiger_phase2_blend_common import build_iid2sid_tokens
from tiger_slate_online_common import build_online_slate_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an item-prefix value critic for TIGER hierarchical advantages.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--item_credit_field", type=str, default="item_credit")
    parser.add_argument("--valid_ratio", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--init_head_path", type=str, default="")
    parser.add_argument("--init_meta_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, required=True)
    return parser.parse_args()


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, group in enumerate(groups):
        if group in valid_groups:
            valid_idx.append(idx)
        else:
            train_idx.append(idx)
    if not train_idx and valid_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx and train_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def load_trace_rows(trace_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "selected_item_ids" not in payload:
                continue
            key = f"{int(payload['episode_id'])}:{int(payload.get('page_index', 0))}"
            rows[key] = payload
    if not rows:
        raise ValueError(f"No usable trace rows in {trace_path}")
    return rows


def load_chain_groups(chain_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "page_index" not in payload:
                continue
            key = f"{int(payload['episode_id'])}:{int(payload['page_index'])}"
            grouped.setdefault(key, []).append(payload)
    if not grouped:
        raise ValueError(f"No usable chain rows in {chain_path}")
    return grouped


class ItemPrefixDataset(Dataset):
    def __init__(
        self,
        trace_map: Dict[str, Dict[str, Any]],
        chain_groups: Dict[str, List[Dict[str, Any]]],
        *,
        iid2sid_tok_cpu: torch.Tensor,
        max_hist_items: int,
        item_credit_field: str,
        max_pages: int,
    ):
        token_vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
        self.rows: List[Dict[str, Any]] = []
        n_pages_seen = 0
        for key in sorted(chain_groups.keys(), key=lambda x: tuple(int(v) for v in x.split(":"))):
            if key not in trace_map:
                continue
            trace_row = trace_map[key]
            page_rows = sorted(chain_groups[key], key=lambda x: int(x.get("slate_item_index", 0)))
            selected_item_ids = [int(x) for x in trace_row.get("selected_item_ids", [])]
            if not selected_item_ids:
                continue
            selected_sid_tokens_list = [[int(v) for v in seq] for seq in trace_row.get("selected_sid_tokens_list", [])]
            online = build_online_slate_inputs(
                history_items=[int(x) for x in trace_row.get("history_items", [])],
                candidate_item_ids=selected_item_ids,
                candidate_sid_tokens_list=selected_sid_tokens_list,
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(max_hist_items),
                token_vocab_size=int(token_vocab_size),
                base_scores=None,
            )
            item_features = np.asarray(online["item_features"], dtype=np.float32)
            page_features = np.asarray(online["page_features"], dtype=np.float32)
            if item_features.ndim != 2 or item_features.shape[0] != len(selected_item_ids):
                continue
            item_credit = np.zeros((len(selected_item_ids),), dtype=np.float32)
            for row in page_rows:
                pos = int(row.get("slate_item_index", -1))
                if 0 <= pos < len(selected_item_ids):
                    item_credit[pos] = float(row.get(str(item_credit_field), row.get("item_credit", 0.0)))
            prefix_values = np.cumsum(item_credit, dtype=np.float32)
            for prefix_len in range(len(selected_item_ids) + 1):
                target = float(prefix_values[prefix_len - 1]) if prefix_len > 0 else 0.0
                self.rows.append(
                    {
                        "item_features": torch.tensor(item_features, dtype=torch.float32),
                        "page_features": torch.tensor(page_features, dtype=torch.float32),
                        "prefix_len": int(prefix_len),
                        "total_items": int(len(selected_item_ids)),
                        "target": torch.tensor(target, dtype=torch.float32),
                        "group": str(trace_row.get("episode_id", key)),
                    }
                )
            n_pages_seen += 1
            if int(max_pages) > 0 and n_pages_seen >= int(max_pages):
                break
        if not self.rows:
            raise ValueError("No usable item-prefix states constructed.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_items = max(int(x["item_features"].shape[0]) for x in batch)
    item_dim = int(batch[0]["item_features"].shape[1])
    batch_size = len(batch)
    item_features = torch.zeros((batch_size, max_items, item_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    page_features = torch.stack([x["page_features"] for x in batch], dim=0)
    prefix_len = torch.tensor([int(x["prefix_len"]) for x in batch], dtype=torch.long)
    total_items = torch.tensor([int(x["total_items"]) for x in batch], dtype=torch.long)
    targets = torch.stack([x["target"] for x in batch], dim=0)
    groups: List[str] = []
    for row_idx, row in enumerate(batch):
        n_items = int(row["item_features"].shape[0])
        item_features[row_idx, :n_items] = row["item_features"]
        keep = min(int(row["prefix_len"]), n_items)
        if keep > 0:
            mask[row_idx, :keep] = True
        groups.append(str(row["group"]))
    return {
        "item_features": item_features,
        "page_features": page_features,
        "mask": mask,
        "prefix_len": prefix_len,
        "total_items": total_items,
        "targets": targets,
        "groups": groups,
    }


def evaluate_head(model: ItemPrefixValueHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    maes: List[float] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["item_features"].to(device),
                batch["page_features"].to(device),
                mask=batch["mask"].to(device),
                prefix_len=batch["prefix_len"].to(device),
                total_items=batch["total_items"].to(device),
            )
            targets = batch["targets"].to(device)
            loss = F.mse_loss(pred, targets)
            mae = torch.abs(pred - targets).mean()
            losses.append(float(loss.item()))
            maes.append(float(mae.item()))
    return {
        "mse": float(np.mean(losses)) if losses else 0.0,
        "mae": float(np.mean(maes)) if maes else 0.0,
    }


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))

    dataset = ItemPrefixDataset(
        load_trace_rows(Path(args.trace_path)),
        load_chain_groups(Path(args.chain_path)),
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        item_credit_field=str(args.item_credit_field),
        max_pages=int(args.max_pages),
    )
    groups = [row["group"] for row in dataset.rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_rows)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_rows)

    item_dim = int(dataset.rows[0]["item_features"].shape[1])
    page_dim = int(dataset.rows[0]["page_features"].shape[0])
    head = ItemPrefixValueHead(
        item_dim=int(item_dim),
        page_dim=int(page_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    if str(args.init_head_path).strip() and str(args.init_meta_path).strip():
        init_head, _ = load_item_prefix_head(str(args.init_head_path), str(args.init_meta_path), device)
        head.load_state_dict(init_head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        train_losses: List[float] = []
        for batch in train_loader:
            pred = head(
                batch["item_features"].to(device),
                batch["page_features"].to(device),
                mask=batch["mask"].to(device),
                prefix_len=batch["prefix_len"].to(device),
                total_items=batch["total_items"].to(device),
            )
            targets = batch["targets"].to(device)
            loss = F.mse_loss(pred, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        valid = evaluate_head(head, valid_loader, device)
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "valid_mse": float(valid["mse"]),
            "valid_mae": float(valid["mae"]),
        }
        print(json.dumps(record, ensure_ascii=True))
        history.append(record)
        if float(valid["mse"]) < best_score:
            best_score = float(valid["mse"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No item-prefix checkpoint produced.")

    head_path = save_dir / "item_prefix_head.pt"
    meta_path = save_dir / "item_prefix_meta.json"
    metrics_path = save_dir / "item_prefix_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Item Prefix Critic",
        "trace_path": str(Path(args.trace_path).resolve()),
        "chain_path": str(Path(args.chain_path).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "item_credit_field": str(args.item_credit_field),
        "max_hist_items": int(args.max_hist_items),
        "item_dim": int(item_dim),
        "page_dim": int(page_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "stats_dim": 3,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "n_states": int(len(dataset)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
    }
    metrics = {
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Saved item prefix critic to {head_path}")
    print(f"[Done] Meta: {meta_path}")
    print(f"[Done] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
