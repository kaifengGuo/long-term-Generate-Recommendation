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
    infer_model_size_args,
    load_tiger_model,
    write_json,
)
from tiger_phase7_welfare_common import (
    WelfareValueHead,
    build_welfare_page_features,
    compute_welfare_step_reward,
    pooled_history_summary,
    resolve_reward_weights,
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
    parser = argparse.ArgumentParser(description="Train an explicit welfare value head for TIGER page prefixes.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_page_index", type=int, default=20)
    parser.add_argument("--max_slate_size", type=int, default=6)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--reward_preset", type=str, default="click_longview")
    parser.add_argument("--reward_weights_json", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class WelfareDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_pages(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], dim=0),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], dim=0),
        "page_features": torch.stack([torch.tensor(x["page_features"], dtype=torch.float32) for x in batch], dim=0),
        "target_return": torch.tensor([float(x["target_return"]) for x in batch], dtype=torch.float32),
        "target_reward": torch.tensor([float(x["target_reward"]) for x in batch], dtype=torch.float32),
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
        raise ValueError(f"No usable trace rows in {trace_path}")
    return rows


def build_training_rows(
    *,
    trace_rows: Sequence[Dict[str, Any]],
    reader,
    sid_mapping_path: str,
    max_hist_items: int,
    max_page_index: int,
    max_slate_size: int,
    gamma: float,
    reward_weights: np.ndarray,
    max_pages: int,
) -> Tuple[List[Dict[str, Any]], int, int, List[str], Dict[str, float]]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    response_names = list(reader.get_statistics()["feedback_type"])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in trace_rows:
        grouped.setdefault(str(rec["episode_id"]), []).append(rec)

    rows: List[Dict[str, Any]] = []
    feature_names: List[str] = []
    for episode_idx, (episode_id, pages) in enumerate(sorted(grouped.items(), key=lambda x: int(x[0]))):
        if int(max_pages) > 0 and episode_idx >= int(max_pages):
            break
        pages = sorted(pages, key=lambda x: int(x.get("page_index", 0)))
        welfare_rewards: List[float] = []
        page_features_cache: List[np.ndarray] = []
        history_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for rec in pages:
            history_items = [int(x) for x in rec.get("history_items", [])][-int(max_hist_items):]
            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(max_hist_items),
                int(sid_depth),
            )
            selected_responses = rec.get("selected_responses", [])
            selected_item_rewards = rec.get("selected_item_rewards", [])
            welfare_step = compute_welfare_step_reward(selected_responses, reward_weights)
            page_feat, feat_names = build_welfare_page_features(
                page_index=int(rec.get("page_index", 0)),
                max_page_index=int(max_page_index),
                history_len=len(history_items),
                max_hist_items=int(max_hist_items),
                slate_size=len(rec.get("selected_item_ids", [])),
                max_slate_size=int(max_slate_size),
                selected_item_rewards=selected_item_rewards,
                selected_responses=selected_responses,
            )
            feature_names = feat_names
            history_cache.append((input_ids.squeeze(0), attention_mask.squeeze(0)))
            page_features_cache.append(page_feat)
            welfare_rewards.append(float(welfare_step))

        returns = [0.0 for _ in pages]
        running = 0.0
        for idx in range(len(pages) - 1, -1, -1):
            running = float(welfare_rewards[idx]) + float(gamma) * running
            returns[idx] = float(running)

        for idx, rec in enumerate(pages):
            rows.append(
                {
                    "group": str(episode_id),
                    "input_ids": history_cache[idx][0].tolist(),
                    "attention_mask": history_cache[idx][1].tolist(),
                    "page_features": page_features_cache[idx].tolist(),
                    "target_return": float(returns[idx]),
                    "target_reward": float(welfare_rewards[idx]),
                    "page_index": int(rec.get("page_index", 0)),
                }
            )
    if not rows:
        raise ValueError("No welfare training rows were built.")
    return rows, sid_depth, int(iid2sid_tok_cpu.max().item()) + 1, feature_names, {
        "reward_mean": float(np.mean([float(x["target_reward"]) for x in rows])),
        "return_mean": float(np.mean([float(x["target_return"]) for x in rows])),
    }


def forward_welfare(
    tiger,
    welfare_head: WelfareValueHead,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    page_features = batch["page_features"].to(device)
    target_return = batch["target_return"].to(device)
    target_reward = batch["target_reward"].to(device)
    with torch.no_grad():
        history_summary = pooled_history_summary(tiger, input_ids, attention_mask)
    pred = welfare_head(history_summary.detach(), page_features)
    return_loss = F.smooth_l1_loss(pred, target_return)
    reward_pred = pred - target_reward
    reward_loss = F.smooth_l1_loss(reward_pred, target_return - target_reward)
    loss = return_loss + 0.25 * reward_loss
    stats = {
        "loss": float(loss.item()),
        "return_loss": float(return_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "pred_mean": float(pred.mean().item()),
        "target_mean": float(target_return.mean().item()),
        "mae": float((pred - target_return).abs().mean().item()),
    }
    return loss, stats


@torch.no_grad()
def evaluate_welfare(tiger, welfare_head: WelfareValueHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    welfare_head.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _loss, stats = forward_welfare(tiger, welfare_head, batch, device)
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(device))
    response_names = list(reader.get_statistics()["feedback_type"])
    reward_weights, reward_weight_map = resolve_reward_weights(
        response_names,
        preset=str(args.reward_preset),
        reward_weights_json=str(args.reward_weights_json),
    )
    trace_rows = load_trace_rows(Path(args.trace_path))
    train_rows, sid_depth, vocab_size, feature_names, data_stats = build_training_rows(
        trace_rows=trace_rows,
        reader=reader,
        sid_mapping_path=str(args.sid_mapping_path),
        max_hist_items=int(args.max_hist_items),
        max_page_index=int(args.max_page_index),
        max_slate_size=int(args.max_slate_size),
        gamma=float(args.gamma),
        reward_weights=reward_weights,
        max_pages=int(args.max_pages),
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

    dataset = WelfareDataset(train_rows)
    groups = [row["group"] for row in train_rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_pages)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_pages)

    welfare_head = WelfareValueHead(
        hidden_size=int(size_cfg["d_model"]),
        page_dim=int(len(feature_names)),
        mlp_dim=int(args.mlp_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(welfare_head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_key = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        welfare_head.train()
        train_losses: List[float] = []
        for batch in train_loader:
            loss, _stats = forward_welfare(tiger, welfare_head, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_metrics = evaluate_welfare(tiger, welfare_head, valid_loader, device)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(dict(valid_metrics))
        key = float(valid_metrics["loss"])
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in welfare_head.state_dict().items()}
            best_metrics = dict(valid_metrics)
            best_epoch = int(epoch)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_mae={valid_metrics['mae']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Welfare head training produced no checkpoint.")

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.tiger_ckpt).resolve().parent / "phase7_welfare"
    save_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_dir / "phase7_welfare_head.pt"
    meta_path = save_dir / "phase7_welfare_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "phase7_welfare_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Phase7 Welfare Head",
        "trace_path": str(Path(args.trace_path).resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "hidden_size": int(size_cfg["d_model"]),
        "page_dim": int(len(feature_names)),
        "feature_names": feature_names,
        "response_names": response_names,
        "reward_preset": str(args.reward_preset),
        "reward_weights": reward_weight_map,
        "gamma": float(args.gamma),
        "sid_depth": int(sid_depth),
        "vocab_size": int(vocab_size),
        "mlp_dim": int(args.mlp_dim),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_pages": int(len(train_rows)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "data_stats": data_stats,
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
        },
    )
    print(f"[phase7] saved welfare head to {head_path}")
    print(f"[phase7] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
