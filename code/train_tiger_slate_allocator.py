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
from tiger_slate_allocator_common import SlateItemAllocator, build_slate_allocator_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned slate->item allocator for TIGER slate attribution.")
    parser.add_argument("--trace_paths", type=str, nargs="+", required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--credit_mode", type=str, default="return", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0)
    parser.add_argument("--heuristic_mix", type=float, default=0.60)
    parser.add_argument("--support_mix", type=float, default=0.25)
    parser.add_argument("--response_mix", type=float, default=0.15)
    parser.add_argument("--valid_ratio", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--share_loss_scale", type=float, default=1.0)
    parser.add_argument("--reward_loss_scale", type=float, default=0.30)
    parser.add_argument("--support_loss_scale", type=float, default=0.20)
    parser.add_argument("--entropy_reg", type=float, default=0.0)
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
        raise ValueError(f"No usable trace rows found in {trace_path}")
    return rows


class SlateAllocatorDataset(Dataset):
    def __init__(
        self,
        trace_paths: Sequence[str],
        *,
        iid2sid_tok_cpu: torch.Tensor,
        max_hist_items: int,
        gamma: float,
        credit_mode: str,
        credit_clip: float,
        heuristic_mix: float,
        support_mix: float,
        response_mix: float,
        max_pages: int,
    ):
        self.rows: List[Dict[str, Any]] = []
        token_vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
        for trace_idx, trace_str in enumerate(trace_paths):
            trace_path = Path(trace_str)
            raw_rows = load_trace_records(trace_path)
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for row in raw_rows:
                group_key = f"{trace_idx}:{int(row['episode_id'])}"
                grouped.setdefault(group_key, []).append(row)
            for episode_key, pages in grouped.items():
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
                    selected_sid_tokens_list = [[int(v) for v in seq] for seq in rec.get("selected_sid_tokens_list", [])]
                    selected_responses = [[float(v) for v in resp] for resp in rec.get("selected_responses", [])]
                    selected_item_rewards = [float(x) for x in rec.get("selected_item_rewards", [])]
                    if not selected_item_ids:
                        continue
                    features = build_slate_allocator_inputs(
                        history_items=[int(x) for x in rec.get("history_items", [])],
                        selected_item_ids=selected_item_ids,
                        selected_sid_tokens_list=selected_sid_tokens_list,
                        selected_responses=selected_responses,
                        selected_item_rewards=selected_item_rewards,
                        response_weights=[float(x) for x in rec.get("response_weights", [])],
                        page_credit=float(credits[page_idx]),
                        slate_return_raw=float(returns[page_idx]),
                        step_reward=float(rec.get("step_reward", 0.0)),
                        page_index=int(rec.get("page_index", page_idx + 1)),
                        iid2sid_tok_cpu=iid2sid_tok_cpu,
                        max_hist_items=int(max_hist_items),
                        token_vocab_size=int(token_vocab_size),
                        heuristic_mix=float(heuristic_mix),
                        support_mix=float(support_mix),
                        response_mix=float(response_mix),
                    )
                    if features["item_features"].size == 0:
                        continue
                    self.rows.append(
                        {
                            "item_features": torch.tensor(features["item_features"], dtype=torch.float32),
                            "page_features": torch.tensor(features["page_features"], dtype=torch.float32),
                            "target_shares": torch.tensor(features["target_shares"], dtype=torch.float32),
                            "heuristic_shares": torch.tensor(features["heuristic_shares"], dtype=torch.float32),
                            "item_rewards": torch.tensor(features["item_rewards"], dtype=torch.float32),
                            "support_strengths": torch.tensor(features["support_strengths"], dtype=torch.float32),
                            "group": str(episode_key),
                            "trace_path": str(trace_path.resolve()),
                            "page_credit": float(credits[page_idx]),
                            "page_index": int(rec.get("page_index", page_idx + 1)),
                            "slate_size": int(len(selected_item_ids)),
                        }
                    )
                    if int(max_pages) > 0 and len(self.rows) >= int(max_pages):
                        return
        if not self.rows:
            raise ValueError("No usable slate pages found for allocator training.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_pages(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_items = max(int(x["item_features"].shape[0]) for x in batch)
    item_dim = int(batch[0]["item_features"].shape[1])
    bsz = len(batch)
    item_features = torch.zeros((bsz, max_items, item_dim), dtype=torch.float32)
    target_shares = torch.zeros((bsz, max_items), dtype=torch.float32)
    heuristic_shares = torch.zeros((bsz, max_items), dtype=torch.float32)
    item_rewards = torch.zeros((bsz, max_items), dtype=torch.float32)
    support_strengths = torch.zeros((bsz, max_items), dtype=torch.float32)
    mask = torch.zeros((bsz, max_items), dtype=torch.bool)
    page_features = torch.stack([x["page_features"] for x in batch], dim=0)
    groups: List[str] = []
    for row_idx, row in enumerate(batch):
        n_items = int(row["item_features"].shape[0])
        item_features[row_idx, :n_items] = row["item_features"]
        target_shares[row_idx, :n_items] = row["target_shares"]
        heuristic_shares[row_idx, :n_items] = row["heuristic_shares"]
        item_rewards[row_idx, :n_items] = row["item_rewards"]
        support_strengths[row_idx, :n_items] = row["support_strengths"]
        mask[row_idx, :n_items] = True
        groups.append(str(row["group"]))
    return {
        "item_features": item_features,
        "page_features": page_features,
        "target_shares": target_shares,
        "heuristic_shares": heuristic_shares,
        "item_rewards": item_rewards,
        "support_strengths": support_strengths,
        "mask": mask,
        "groups": groups,
    }


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(logits, dim=-1)


def evaluate_allocator(
    model: SlateItemAllocator,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    share_losses: List[float] = []
    reward_losses: List[float] = []
    support_losses: List[float] = []
    l1_losses: List[float] = []
    top1_hits = 0.0
    top1_total = 0.0
    with torch.no_grad():
        for batch in loader:
            item_features = batch["item_features"].to(device)
            page_features = batch["page_features"].to(device)
            target_shares = batch["target_shares"].to(device)
            item_rewards = batch["item_rewards"].to(device)
            support_strengths = batch["support_strengths"].to(device)
            mask = batch["mask"].to(device)
            logits = model(item_features, page_features, mask=mask)
            pred_shares = masked_softmax(logits, mask)
            share_loss = -(target_shares * torch.log(pred_shares.clamp_min(1e-8))).sum(dim=-1).mean()
            pred_reward = (pred_shares * item_rewards).sum(dim=-1)
            target_reward = (target_shares * item_rewards).sum(dim=-1)
            pred_support = (pred_shares * support_strengths).sum(dim=-1)
            target_support = (target_shares * support_strengths).sum(dim=-1)
            reward_loss = F.mse_loss(pred_reward, target_reward)
            support_loss = F.mse_loss(pred_support, target_support)
            l1_loss = torch.abs(pred_shares - target_shares).sum(dim=-1).mean()
            share_losses.append(float(share_loss.item()))
            reward_losses.append(float(reward_loss.item()))
            support_losses.append(float(support_loss.item()))
            l1_losses.append(float(l1_loss.item()))
            top1_hits += float((pred_shares.argmax(dim=-1) == target_shares.argmax(dim=-1)).sum().item())
            top1_total += float(pred_shares.shape[0])
    return {
        "share_loss": float(np.mean(share_losses)) if share_losses else 0.0,
        "reward_loss": float(np.mean(reward_losses)) if reward_losses else 0.0,
        "support_loss": float(np.mean(support_losses)) if support_losses else 0.0,
        "l1_loss": float(np.mean(l1_losses)) if l1_losses else 0.0,
        "top1_acc": float(top1_hits / max(top1_total, 1.0)),
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

    dataset = SlateAllocatorDataset(
        args.trace_paths,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        gamma=float(args.gamma),
        credit_mode=str(args.credit_mode),
        credit_clip=float(args.credit_clip),
        heuristic_mix=float(args.heuristic_mix),
        support_mix=float(args.support_mix),
        response_mix=float(args.response_mix),
        max_pages=int(args.max_pages),
    )

    groups = [row["group"] for row in dataset.rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pages,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pages,
    )

    sample_item_dim = int(dataset.rows[0]["item_features"].shape[1])
    sample_page_dim = int(dataset.rows[0]["page_features"].shape[0])
    model = SlateItemAllocator(
        item_dim=sample_item_dim,
        page_dim=sample_page_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
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
            target_shares = batch["target_shares"].to(device)
            item_rewards = batch["item_rewards"].to(device)
            support_strengths = batch["support_strengths"].to(device)
            mask = batch["mask"].to(device)

            logits = model(item_features, page_features, mask=mask)
            pred_shares = masked_softmax(logits, mask)
            share_loss = -(target_shares * torch.log(pred_shares.clamp_min(1e-8))).sum(dim=-1).mean()
            pred_reward = (pred_shares * item_rewards).sum(dim=-1)
            target_reward = (target_shares * item_rewards).sum(dim=-1)
            pred_support = (pred_shares * support_strengths).sum(dim=-1)
            target_support = (target_shares * support_strengths).sum(dim=-1)
            reward_loss = F.mse_loss(pred_reward, target_reward)
            support_loss = F.mse_loss(pred_support, target_support)
            entropy = -(pred_shares * torch.log(pred_shares.clamp_min(1e-8))).sum(dim=-1).mean()
            loss = (
                float(args.share_loss_scale) * share_loss
                + float(args.reward_loss_scale) * reward_loss
                + float(args.support_loss_scale) * support_loss
                - float(args.entropy_reg) * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_metrics = evaluate_allocator(model, valid_loader, device)
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "valid_share_loss": float(valid_metrics["share_loss"]),
            "valid_reward_loss": float(valid_metrics["reward_loss"]),
            "valid_support_loss": float(valid_metrics["support_loss"]),
            "valid_l1_loss": float(valid_metrics["l1_loss"]),
            "valid_top1_acc": float(valid_metrics["top1_acc"]),
        }
        print(json.dumps(record, ensure_ascii=True))
        history.append(record)
        score = (
            float(valid_metrics["share_loss"])
            + 0.30 * float(valid_metrics["reward_loss"])
            + 0.20 * float(valid_metrics["support_loss"])
            + 0.10 * float(valid_metrics["l1_loss"])
        )
        if score < best_score:
            best_score = score
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Slate allocator training produced no checkpoint.")

    head_path = save_dir / "slate_allocator_head.pt"
    meta_path = save_dir / "slate_allocator_meta.json"
    metrics_path = save_dir / "slate_allocator_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Learned Slate Item Allocator",
        "trace_paths": [str(Path(x).resolve()) for x in args.trace_paths],
        "item_dim": int(sample_item_dim),
        "page_dim": int(sample_page_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "max_hist_items": int(args.max_hist_items),
        "gamma": float(args.gamma),
        "credit_mode": str(args.credit_mode),
        "credit_clip": float(args.credit_clip),
        "heuristic_mix": float(args.heuristic_mix),
        "support_mix": float(args.support_mix),
        "response_mix": float(args.response_mix),
        "sid_depth": int(sid_depth),
        "token_vocab_size": int(iid2sid_tok_cpu.max().item()) + 1,
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
    print(f"[Done] Saved slate allocator to {head_path}")
    print(f"[Done] Meta: {meta_path}")
    print(f"[Done] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
