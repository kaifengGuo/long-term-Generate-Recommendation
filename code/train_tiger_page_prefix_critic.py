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
from tiger_hier_prefix_common import load_page_prefix_head
from tiger_phase2_blend_common import (
    build_history_tokens,
    build_iid2sid_tokens,
    infer_model_size_args,
    load_tiger_model,
)
from tiger_phase6_joint_common import SlateCreditHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a page-prefix value critic for TIGER hierarchical advantages.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--page_delta_field", type=str, default="credit", choices=["credit", "reward"])
    parser.add_argument("--credit_mode", type=str, default="return", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0)
    parser.add_argument("--valid_ratio", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--max_states", type=int, default=0)
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


def load_trace_rows(trace_path: Path) -> List[Dict[str, Any]]:
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
        raise ValueError(f"No usable rows in {trace_path}")
    return rows


def pool_history_summary(tiger, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    enc_out = tiger.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = enc_out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


class PagePrefixDataset(Dataset):
    def __init__(
        self,
        trace_path: Path,
        *,
        iid2sid_tok_cpu: torch.Tensor,
        max_hist_items: int,
        gamma: float,
        page_delta_field: str,
        credit_mode: str,
        credit_clip: float,
        max_states: int,
    ):
        raw_rows = load_trace_rows(trace_path)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            grouped.setdefault(str(int(row["episode_id"])), []).append(row)

        sid_depth = int(iid2sid_tok_cpu.shape[1])
        self.rows: List[Dict[str, Any]] = []
        n_states = 0
        for episode_id, pages in grouped.items():
            pages = sorted(pages, key=lambda x: int(x.get("page_index", 0)))
            rewards = [float(x.get("step_reward", 0.0)) for x in pages]
            returns = [0.0 for _ in pages]
            running = 0.0
            for idx in range(len(pages) - 1, -1, -1):
                running = rewards[idx] + float(gamma) * running
                returns[idx] = running
            if str(page_delta_field) == "reward":
                deltas = rewards
            else:
                deltas = transform_episode_credits(returns, str(credit_mode), float(credit_clip))

            prefix_history = [int(x) for x in pages[0].get("history_items", [])]
            prefix_value = 0.0
            for state_idx, history_items in enumerate([prefix_history] + [None] * len(pages)):
                if state_idx > 0:
                    page = pages[state_idx - 1]
                    prefix_history = prefix_history + [int(x) for x in page.get("selected_item_ids", [])]
                    prefix_value += float(deltas[state_idx - 1])
                    history_items = prefix_history
                hist_tensor = torch.tensor(history_items[-int(max_hist_items):], dtype=torch.long).view(1, -1)
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
                        "target": torch.tensor(float(prefix_value), dtype=torch.float32),
                        "group": str(episode_id),
                        "state_index": int(state_idx),
                    }
                )
                n_states += 1
                if int(max_states) > 0 and n_states >= int(max_states):
                    return
        if not self.rows:
            raise ValueError(f"No usable prefix states in {trace_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "targets": torch.stack([x["target"] for x in batch], dim=0),
        "groups": [x["group"] for x in batch],
    }


def evaluate_head(tiger, head: SlateCreditHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    head.eval()
    losses: List[float] = []
    maes: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            summary = pool_history_summary(tiger, input_ids, attention_mask)
            pred = head(summary.detach())
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
    size_args = infer_model_size_args(str(args.model_size))
    tiger, sid_depth, _codebook_size = load_tiger_model(
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
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))

    dataset = PagePrefixDataset(
        Path(args.trace_path),
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        gamma=float(args.gamma),
        page_delta_field=str(args.page_delta_field),
        credit_mode=str(args.credit_mode),
        credit_clip=float(args.credit_clip),
        max_states=int(args.max_states),
    )
    groups = [row["group"] for row in dataset.rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_rows)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_rows)

    head = SlateCreditHead(hidden_size=int(size_args["d_model"]), mlp_dim=int(args.mlp_dim)).to(device)
    if str(args.init_head_path).strip() and str(args.init_meta_path).strip():
        init_head, _ = load_page_prefix_head(str(args.init_head_path), str(args.init_meta_path), device)
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            with torch.no_grad():
                summary = pool_history_summary(tiger, input_ids, attention_mask)
            pred = head(summary.detach())
            loss = F.mse_loss(pred, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        valid = evaluate_head(tiger, head, valid_loader, device)
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
        raise RuntimeError("No page-prefix checkpoint produced.")

    head_path = save_dir / "page_prefix_head.pt"
    meta_path = save_dir / "page_prefix_meta.json"
    metrics_path = save_dir / "page_prefix_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Page Prefix Critic",
        "trace_path": str(Path(args.trace_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "hidden_size": int(size_args["d_model"]),
        "mlp_dim": int(args.mlp_dim),
        "sid_depth": int(sid_depth),
        "page_delta_field": str(args.page_delta_field),
        "credit_mode": str(args.credit_mode),
        "credit_clip": float(args.credit_clip),
        "gamma": float(args.gamma),
        "max_hist_items": int(args.max_hist_items),
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
    print(f"[Done] Saved page prefix critic to {head_path}")
    print(f"[Done] Meta: {meta_path}")
    print(f"[Done] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
