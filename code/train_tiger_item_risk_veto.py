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
from tiger_phase2_blend_common import build_iid2sid_tokens
from tiger_slate_online_common import OnlineSlateAllocatorHead, build_online_slate_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Phase7c item-level risk veto reranker.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--valid_ratio", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--l1_loss_scale", type=float, default=0.20)
    parser.add_argument("--support_loss_scale", type=float, default=0.05)
    parser.add_argument("--entropy_reg_scale", type=float, default=0.00)
    parser.add_argument("--sample_weight_power", type=float, default=0.50)
    parser.add_argument("--min_page_neg_mass", type=float, default=0.00)
    parser.add_argument("--zero_page_weight", type=float, default=0.10)
    parser.add_argument("--include_zero_pages", action="store_true")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="local_neg_mass",
        choices=["local_neg_mass", "item_plus_page_neg", "page_item_neg_only"],
    )
    parser.add_argument("--base_score_field", type=str, default="")
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


def safe_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for x in values:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)
    return out


def extract_item_neg_mass(payload: Dict[str, Any], target_mode: str) -> float:
    name = str(target_mode)
    if name == "page_item_neg_only":
        return max(float(payload.get("phase7b_page_item_neg_credit", 0.0)), 0.0)
    if name == "item_plus_page_neg":
        item_neg = max(float(payload.get("phase7b_item_neg_credit", 0.0)), 0.0)
        page_neg = max(float(payload.get("phase7b_page_item_neg_credit", 0.0)), 0.0)
        return float(item_neg + page_neg)
    token_neg = sum(max(float(x), 0.0) for x in safe_float_list(payload.get("phase7b_token_neg_credit", [])))
    item_neg = max(float(payload.get("phase7b_item_neg_credit", 0.0)), 0.0)
    page_neg = max(float(payload.get("phase7b_page_item_neg_credit", 0.0)), 0.0)
    local_neg = float(payload.get("phase7b_local_neg_mass", token_neg + item_neg + page_neg))
    return max(local_neg, 0.0)


class ItemRiskVetoDataset(Dataset):
    def __init__(
        self,
        chain_path: Path,
        *,
        iid2sid_tok_cpu: torch.Tensor,
        max_hist_items: int,
        max_pages: int,
        target_mode: str,
        min_page_neg_mass: float,
        include_zero_pages: bool,
        zero_page_weight: float,
        sample_weight_power: float,
        base_score_field: str,
    ):
        token_vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
        grouped: Dict[Tuple[int, int], Dict[str, Any]] = {}
        with chain_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                key = (int(payload["episode_id"]), int(payload["page_index"]))
                if key not in grouped:
                    if int(max_pages) > 0 and len(grouped) >= int(max_pages):
                        continue
                    grouped[key] = {"page_payload": payload, "item_rows": {}}
                idx = int(payload.get("slate_item_index", -1))
                if idx >= 0:
                    grouped[key]["item_rows"][idx] = payload
        if not grouped:
            raise ValueError(f"No usable page groups found in {chain_path}")

        self.rows: List[Dict[str, Any]] = []
        skipped_low_signal = 0
        skipped_invalid = 0
        for (episode_id, _page_index), bundle in grouped.items():
            payload = bundle["page_payload"]
            item_rows = bundle["item_rows"]
            item_ids = [int(x) for x in payload.get("selected_item_ids", [])]
            n_items = len(item_ids)
            if n_items <= 0:
                skipped_invalid += 1
                continue
            neg_mass = np.zeros((n_items,), dtype=np.float32)
            for idx in range(n_items):
                row = item_rows.get(idx)
                if row is None:
                    continue
                neg_mass[idx] = float(extract_item_neg_mass(row, target_mode))
            page_neg_mass = float(np.maximum(neg_mass, 0.0).sum())
            if page_neg_mass < float(min_page_neg_mass):
                skipped_low_signal += 1
                continue
            if page_neg_mass <= 1e-8:
                if not bool(include_zero_pages):
                    skipped_low_signal += 1
                    continue
                target_shares = np.full((n_items,), 1.0 / float(max(n_items, 1)), dtype=np.float32)
                sample_weight = float(zero_page_weight)
            else:
                target_shares = neg_mass / max(page_neg_mass, 1e-8)
                sample_weight = float(max(page_neg_mass, 1e-8) ** float(sample_weight_power))

            base_scores = None
            if str(base_score_field).strip():
                maybe_scores = safe_float_list(payload.get(str(base_score_field), []))
                if len(maybe_scores) == n_items:
                    base_scores = maybe_scores

            online = build_online_slate_inputs(
                history_items=[int(x) for x in payload.get("history_items", [])],
                candidate_item_ids=item_ids,
                candidate_sid_tokens_list=payload.get("selected_sid_tokens_list", None),
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(max_hist_items),
                token_vocab_size=int(token_vocab_size),
                base_scores=base_scores,
            )
            self.rows.append(
                {
                    "item_features": torch.tensor(online["item_features"], dtype=torch.float32),
                    "page_features": torch.tensor(online["page_features"], dtype=torch.float32),
                    "target_shares": torch.tensor(target_shares, dtype=torch.float32),
                    "support_strengths": torch.tensor(online["support_strengths"], dtype=torch.float32),
                    "sample_weight": float(sample_weight),
                    "page_neg_mass": float(page_neg_mass),
                    "group": str(episode_id),
                }
            )
        if not self.rows:
            raise ValueError(f"No usable risk-veto rows in {chain_path}")
        self.stats = {
            "n_pages": int(len(self.rows)),
            "skipped_low_signal": int(skipped_low_signal),
            "skipped_invalid": int(skipped_invalid),
            "page_neg_mass_mean": float(np.mean([float(x["page_neg_mass"]) for x in self.rows])),
            "page_neg_mass_p50": float(np.median([float(x["page_neg_mass"]) for x in self.rows])),
            "page_neg_mass_p90": float(np.quantile([float(x["page_neg_mass"]) for x in self.rows], 0.90)),
            "sample_weight_mean": float(np.mean([float(x["sample_weight"]) for x in self.rows])),
        }

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
    support_strengths = torch.zeros((bsz, max_items), dtype=torch.float32)
    sample_weights = torch.zeros((bsz,), dtype=torch.float32)
    page_neg_mass = torch.zeros((bsz,), dtype=torch.float32)
    mask = torch.zeros((bsz, max_items), dtype=torch.bool)
    page_features = torch.stack([x["page_features"] for x in batch], dim=0)
    groups: List[str] = []
    for row_idx, row in enumerate(batch):
        n_items = int(row["item_features"].shape[0])
        item_features[row_idx, :n_items] = row["item_features"]
        target_shares[row_idx, :n_items] = row["target_shares"]
        support_strengths[row_idx, :n_items] = row["support_strengths"]
        mask[row_idx, :n_items] = True
        sample_weights[row_idx] = float(row["sample_weight"])
        page_neg_mass[row_idx] = float(row["page_neg_mass"])
        groups.append(str(row["group"]))
    return {
        "item_features": item_features,
        "page_features": page_features,
        "target_shares": target_shares,
        "support_strengths": support_strengths,
        "sample_weights": sample_weights,
        "page_neg_mass": page_neg_mass,
        "mask": mask,
        "groups": groups,
    }


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def evaluate_head(model: OnlineSlateAllocatorHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    share_losses: List[float] = []
    l1_losses: List[float] = []
    support_losses: List[float] = []
    weight_sum = 0.0
    top1_hits = 0.0
    top1_weight = 0.0
    with torch.no_grad():
        for batch in loader:
            item_features = batch["item_features"].to(device)
            page_features = batch["page_features"].to(device)
            target_shares = batch["target_shares"].to(device)
            support_strengths = batch["support_strengths"].to(device)
            sample_weights = batch["sample_weights"].to(device)
            mask = batch["mask"].to(device)
            logits = model(item_features, page_features, mask=mask)
            pred = masked_softmax(logits, mask)
            share_loss = -(target_shares * torch.log(pred.clamp_min(1e-8))).sum(dim=-1)
            l1_loss = torch.abs(pred - target_shares).sum(dim=-1)
            pred_support = (pred * support_strengths).sum(dim=-1)
            target_support = (target_shares * support_strengths).sum(dim=-1)
            support_loss = F.mse_loss(pred_support, target_support, reduction="none")
            share_losses.append(float(weighted_mean(share_loss, sample_weights).item()))
            l1_losses.append(float(weighted_mean(l1_loss, sample_weights).item()))
            support_losses.append(float(weighted_mean(support_loss, sample_weights).item()))
            matches = (pred.argmax(dim=-1) == target_shares.argmax(dim=-1)).float()
            top1_hits += float((matches * sample_weights).sum().item())
            top1_weight += float(sample_weights.sum().item())
            weight_sum += float(sample_weights.sum().item())
    return {
        "share_loss": float(np.mean(share_losses)) if share_losses else 0.0,
        "l1_loss": float(np.mean(l1_losses)) if l1_losses else 0.0,
        "support_loss": float(np.mean(support_losses)) if support_losses else 0.0,
        "top1_acc": float(top1_hits / max(top1_weight, 1e-8)),
        "weight_sum": float(weight_sum),
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

    dataset = ItemRiskVetoDataset(
        Path(args.chain_path),
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        max_pages=int(args.max_pages),
        target_mode=str(args.target_mode),
        min_page_neg_mass=float(args.min_page_neg_mass),
        include_zero_pages=bool(args.include_zero_pages),
        zero_page_weight=float(args.zero_page_weight),
        sample_weight_power=float(args.sample_weight_power),
        base_score_field=str(args.base_score_field),
    )
    groups = [row["group"] for row in dataset.rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_pages)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_pages)

    item_dim = int(dataset.rows[0]["item_features"].shape[1])
    page_dim = int(dataset.rows[0]["page_features"].shape[0])
    model = OnlineSlateAllocatorHead(item_dim=item_dim, page_dim=page_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
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
            support_strengths = batch["support_strengths"].to(device)
            sample_weights = batch["sample_weights"].to(device)
            mask = batch["mask"].to(device)
            logits = model(item_features, page_features, mask=mask)
            pred = masked_softmax(logits, mask)
            share_loss = -(target_shares * torch.log(pred.clamp_min(1e-8))).sum(dim=-1)
            l1_loss = torch.abs(pred - target_shares).sum(dim=-1)
            pred_support = (pred * support_strengths).sum(dim=-1)
            target_support = (target_shares * support_strengths).sum(dim=-1)
            support_loss = F.mse_loss(pred_support, target_support, reduction="none")
            entropy = -(pred * torch.log(pred.clamp_min(1e-8))).sum(dim=-1)
            loss = weighted_mean(share_loss, sample_weights)
            loss = loss + float(args.l1_loss_scale) * weighted_mean(l1_loss, sample_weights)
            loss = loss + float(args.support_loss_scale) * weighted_mean(support_loss, sample_weights)
            loss = loss - float(args.entropy_reg_scale) * weighted_mean(entropy, sample_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        valid = evaluate_head(model, valid_loader, device)
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "valid_share_loss": float(valid["share_loss"]),
            "valid_l1_loss": float(valid["l1_loss"]),
            "valid_support_loss": float(valid["support_loss"]),
            "valid_top1_acc": float(valid["top1_acc"]),
        }
        print(json.dumps(record, ensure_ascii=True))
        history.append(record)
        score = float(valid["share_loss"] + float(args.l1_loss_scale) * valid["l1_loss"])
        if score < best_score:
            best_score = score
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Risk veto training produced no checkpoint.")

    head_path = save_dir / "online_slate_allocator_head.pt"
    meta_path = save_dir / "online_slate_allocator_meta.json"
    metrics_path = save_dir / "online_slate_allocator_metrics.json"
    torch.save(best_state, head_path)
    meta = {
        "method": "TIGER Phase7c Item Risk Veto",
        "head_role": "risk_veto_allocator",
        "chain_path": str(Path(args.chain_path).resolve()),
        "target_mode": str(args.target_mode),
        "item_dim": int(item_dim),
        "page_dim": int(page_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "max_hist_items": int(args.max_hist_items),
        "sid_depth": int(sid_depth),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "n_pages": int(len(dataset)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "min_page_neg_mass": float(args.min_page_neg_mass),
        "include_zero_pages": bool(args.include_zero_pages),
        "zero_page_weight": float(args.zero_page_weight),
        "sample_weight_power": float(args.sample_weight_power),
        "base_score_field": str(args.base_score_field),
        "recommended_eval_scales": [-0.02, -0.05, -0.10],
        "dataset_stats": dataset.stats,
    }
    metrics = {
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Saved item risk veto head to {head_path}")
    print(f"[Done] Meta: {meta_path}")
    print(f"[Done] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
