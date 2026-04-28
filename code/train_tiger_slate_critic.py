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

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log, transform_episode_credits
from tiger_phase2_blend_common import build_iid2sid_tokens
from tiger_slate_online_common import SlateValueHead, build_online_slate_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a slate-level critic from TIGER slate traces.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--credit_mode", type=str, default="return", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0)
    parser.add_argument("--target_field", type=str, default="return", choices=["return", "credit"])
    parser.add_argument("--valid_ratio", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max_pages", type=int, default=0)
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


def load_trace_records(trace_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "selected_item_ids" not in payload:
                continue
            rows.append(payload)
    if not rows:
        raise ValueError(f"No usable trace rows in {trace_path}")
    return rows


class SlateCriticDataset(Dataset):
    def __init__(
        self,
        trace_path: Path,
        *,
        iid2sid_tok_cpu: torch.Tensor,
        max_hist_items: int,
        gamma: float,
        credit_mode: str,
        credit_clip: float,
        target_field: str,
        max_pages: int,
    ):
        token_vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
        raw_rows = load_trace_records(trace_path)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            grouped.setdefault(str(int(row["episode_id"])), []).append(row)
        self.rows: List[Dict[str, Any]] = []
        for episode_id, pages in grouped.items():
            pages = sorted(pages, key=lambda x: int(x.get("page_index", 0)))
            rewards = [float(x.get("step_reward", 0.0)) for x in pages]
            returns = [0.0 for _ in pages]
            running = 0.0
            for idx in range(len(pages) - 1, -1, -1):
                running = rewards[idx] + float(gamma) * running
                returns[idx] = running
            credits = transform_episode_credits(returns, str(credit_mode), float(credit_clip))
            for page_idx, rec in enumerate(pages):
                selected_item_ids = [int(x) for x in rec.get("selected_item_ids", [])]
                if not selected_item_ids:
                    continue
                online = build_online_slate_inputs(
                    history_items=[int(x) for x in rec.get("history_items", [])],
                    candidate_item_ids=selected_item_ids,
                    candidate_sid_tokens_list=rec.get("selected_sid_tokens_list", None),
                    iid2sid_tok_cpu=iid2sid_tok_cpu,
                    max_hist_items=int(max_hist_items),
                    token_vocab_size=int(token_vocab_size),
                    base_scores=None,
                )
                target = float(returns[page_idx]) if str(target_field) == "return" else float(credits[page_idx])
                self.rows.append(
                    {
                        "item_features": torch.tensor(online["item_features"], dtype=torch.float32),
                        "page_features": torch.tensor(online["page_features"], dtype=torch.float32),
                        "target": torch.tensor(target, dtype=torch.float32),
                        "group": str(episode_id),
                    }
                )
                if int(max_pages) > 0 and len(self.rows) >= int(max_pages):
                    return
        if not self.rows:
            raise ValueError(f"No usable critic rows in {trace_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_pages(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_items = max(int(x["item_features"].shape[0]) for x in batch)
    item_dim = int(batch[0]["item_features"].shape[1])
    bsz = len(batch)
    item_features = torch.zeros((bsz, max_items, item_dim), dtype=torch.float32)
    mask = torch.zeros((bsz, max_items), dtype=torch.bool)
    page_features = torch.stack([x["page_features"] for x in batch], dim=0)
    targets = torch.stack([x["target"] for x in batch], dim=0)
    groups: List[str] = []
    for row_idx, row in enumerate(batch):
        n_items = int(row["item_features"].shape[0])
        item_features[row_idx, :n_items] = row["item_features"]
        mask[row_idx, :n_items] = True
        groups.append(str(row["group"]))
    return {
        "item_features": item_features,
        "page_features": page_features,
        "targets": targets,
        "mask": mask,
        "groups": groups,
    }


def evaluate_head(model: SlateValueHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    mae_vals: List[float] = []
    with torch.no_grad():
        for batch in loader:
            item_features = batch["item_features"].to(device)
            page_features = batch["page_features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            pred = model(item_features, page_features, mask=mask)
            loss = F.mse_loss(pred, targets)
            mae = torch.abs(pred - targets).mean()
            losses.append(float(loss.item()))
            mae_vals.append(float(mae.item()))
    return {
        "mse": float(np.mean(losses)) if losses else 0.0,
        "mae": float(np.mean(mae_vals)) if mae_vals else 0.0,
    }


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reader = load_reader_from_uirm_log(args.uirm_log_path, "cpu")
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _sid2iid_map = build_iid2sid_tokens(reader, args.sid_mapping_path, int(sid_depth), torch.device("cpu"))

    dataset = SlateCriticDataset(
        Path(args.trace_path),
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        gamma=float(args.gamma),
        credit_mode=str(args.credit_mode),
        credit_clip=float(args.credit_clip),
        target_field=str(args.target_field),
        max_pages=int(args.max_pages),
    )
    groups = [row["group"] for row in dataset.rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_pages)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_pages)

    item_dim = int(dataset.rows[0]["item_features"].shape[1])
    page_dim = int(dataset.rows[0]["page_features"].shape[0])
    model = SlateValueHead(item_dim=item_dim, page_dim=page_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            item_features = batch["item_features"].to(device)
            page_features = batch["page_features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            pred = model(item_features, page_features, mask=mask)
            loss = F.mse_loss(pred, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        valid = evaluate_head(model, valid_loader, device)
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "valid_mse": float(valid["mse"]),
            "valid_mae": float(valid["mae"]),
        }
        print(json.dumps(record, ensure_ascii=True))
        history.append(record)
        score = float(valid["mse"])
        if score < best_score:
            best_score = score
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Slate critic training produced no checkpoint.")

    head_path = save_dir / "slate_value_head.pt"
    meta_path = save_dir / "slate_value_meta.json"
    metrics_path = save_dir / "slate_value_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Slate Critic",
        "trace_path": str(Path(args.trace_path).resolve()),
        "target_field": str(args.target_field),
        "item_dim": int(item_dim),
        "page_dim": int(page_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "max_hist_items": int(args.max_hist_items),
        "gamma": float(args.gamma),
        "credit_mode": str(args.credit_mode),
        "credit_clip": float(args.credit_clip),
        "sid_depth": int(sid_depth),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "n_pages": int(len(dataset)),
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
    print(f"[Done] Saved slate critic to {head_path}")
    print(f"[Done] Meta: {meta_path}")
    print(f"[Done] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
