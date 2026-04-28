import argparse
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils
from reader import *  # noqa: F401,F403

from tiger_phase2_blend_common import (
    TokenCreditTransportHead,
    build_history_tokens,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_tiger_model,
    sinkhorn_transport,
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
    parser = argparse.ArgumentParser(description="Train Phase2 Blend Decode head for TIGER.")
    parser.add_argument("--trace_dir", type=str, required=True, help="Trace JSONL file or directory dumped by eval_tiger_phase2_blend_env.py.")
    parser.add_argument("--chain_path", type=str, default="", help="Optional explicit phase3 chain JSONL. If set, train directly from chain credits.")
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--credit_mode", type=str, default="return", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0, help="If > 0, clip transformed item credit into [-credit_clip, credit_clip].")
    parser.add_argument("--chain_block_field", type=str, default="block_credit", help="Block target field when chain_path is provided.")
    parser.add_argument("--chain_item_field", type=str, default="item_credit", help="Item credit field when chain_path is provided.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--conservation_scale", type=float, default=0.5)
    parser.add_argument("--sign_scale", type=float, default=0.2)
    parser.add_argument("--transport_epsilon", type=float, default=0.35)
    parser.add_argument("--transport_iter", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


def transform_episode_credits(raw_returns: Sequence[float], mode: str, clip: float) -> List[float]:
    vals = np.asarray(list(raw_returns), dtype=np.float32)
    if vals.size == 0:
        return []
    mode_name = str(mode).lower()
    if mode_name == "return":
        out = vals.copy()
    elif mode_name == "centered":
        out = vals - float(vals.mean())
    elif mode_name == "zscore":
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-6:
            out = vals - mean
        else:
            out = (vals - mean) / std
    else:
        raise ValueError(f"Unsupported credit_mode: {mode}")
    if float(clip) > 0.0:
        out = np.clip(out, -float(clip), float(clip))
    return out.astype(np.float32).tolist()


def load_trace_records(trace_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    trace_paths = [trace_dir] if trace_dir.is_file() else sorted(trace_dir.glob("*.jsonl"))
    for path in trace_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "episode_id" not in payload or "selected_item_id" not in payload:
                    continue
                records.append(payload)
    if not records:
        raise ValueError(f"No trace records found under {trace_dir}")
    return records


def load_chain_records(chain_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "selected_item_id" not in payload:
                continue
            records.append(payload)
    if not records:
        raise ValueError(f"No chain records found under {chain_path}")
    return records


def longest_prefix_ratio(hist_sid: np.ndarray, target_sid: np.ndarray, end_idx: int) -> float:
    width = int(end_idx) + 1
    if width <= 0:
        return 0.0
    match = float(np.mean(hist_sid[:width] == target_sid[:width]))
    return float(np.clip(match, 0.0, 1.0))


def build_block_credit(
    history_items: Sequence[int],
    target_tokens: Sequence[int],
    item_credit: float,
    iid2sid_tok_cpu: torch.Tensor,
    epsilon: float,
    n_iter: int,
) -> np.ndarray:
    target = np.asarray(target_tokens, dtype=np.int64).reshape(-1)
    depth = int(target.shape[0])
    if depth <= 0:
        return np.zeros((0,), dtype=np.float32)
    hist = [int(i) for i in history_items if 0 <= int(i) < int(iid2sid_tok_cpu.shape[0])]
    hist = hist[-50:]
    if not hist:
        return np.ones(depth, dtype=np.float32) * (float(item_credit) / max(depth, 1))

    hist_tensor = torch.tensor(hist, dtype=torch.long)
    hist_sid = iid2sid_tok_cpu[hist_tensor].numpy()
    valid_mask = np.any(hist_sid > 0, axis=1)
    hist_sid = hist_sid[valid_mask]
    if hist_sid.size == 0:
        return np.ones(depth, dtype=np.float32) * (float(item_credit) / max(depth, 1))

    n_hist = int(hist_sid.shape[0])
    affinity = np.zeros((n_hist, depth), dtype=np.float32)
    for r in range(n_hist):
        for c in range(depth):
            affinity[r, c] = longest_prefix_ratio(hist_sid[r], target, c)

    row_mass = np.linspace(1.0, 2.0, n_hist, dtype=np.float32)
    col_mass = np.ones(depth, dtype=np.float32)
    pos_cost = 1.0 - affinity
    neg_cost = affinity
    pos_plan = sinkhorn_transport(row_mass, col_mass, pos_cost, epsilon=float(epsilon), n_iter=int(n_iter))
    neg_plan = sinkhorn_transport(row_mass, col_mass, neg_cost, epsilon=float(epsilon), n_iter=int(n_iter))
    pos_mass = max(float(item_credit), 0.0)
    neg_mass = max(-float(item_credit), 0.0)
    block_credit = pos_mass * pos_plan.sum(axis=0) - neg_mass * neg_plan.sum(axis=0)
    residual = float(item_credit) - float(block_credit.sum())
    block_credit[-1] += residual
    return block_credit.astype(np.float32)


class Phase2RecordDataset(Dataset):
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
            torch.tensor(row["block_credit"], dtype=torch.float32),
            torch.tensor(float(row["item_credit"]), dtype=torch.float32),
            row["group"],
        )


def collate_phase2(batch):
    input_ids = torch.stack([x[0] for x in batch], dim=0)
    attention_mask = torch.stack([x[1] for x in batch], dim=0)
    target_tokens = torch.stack([x[2] for x in batch], dim=0)
    block_credit = torch.stack([x[3] for x in batch], dim=0)
    item_credit = torch.stack([x[4] for x in batch], dim=0)
    groups = [x[5] for x in batch]
    return input_ids, attention_mask, target_tokens, block_credit, item_credit, groups


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
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def evaluate_head(
    tiger,
    head: TokenCreditTransportHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    head.eval()
    block_losses: List[float] = []
    item_losses: List[float] = []
    sign_accs: List[float] = []
    with torch.no_grad():
        for input_ids, attention_mask, target_tokens, block_credit, item_credit, _groups in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            block_credit = block_credit.to(device)
            item_credit = item_credit.to(device)
            _logits, hidden = tiger.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
            )
            pred = head(hidden, target_tokens)
            pred_item = pred.sum(dim=1)
            block_losses.append(float(F.mse_loss(pred, block_credit).item()))
            item_losses.append(float(F.mse_loss(pred_item, item_credit).item()))
            sign_accs.append(float(((pred >= 0.0) == (block_credit >= 0.0)).float().mean().item()))
    return {
        "block_mse": float(np.mean(block_losses)) if block_losses else 0.0,
        "item_mse": float(np.mean(item_losses)) if item_losses else 0.0,
        "sign_acc": float(np.mean(sign_accs)) if sign_accs else 0.0,
    }


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    trace_dir = Path(args.trace_dir)
    chain_path = Path(args.chain_path) if str(args.chain_path).strip() else None
    raw_records = load_trace_records(trace_dir) if chain_path is None else load_chain_records(chain_path)
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(device))

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, sid_depth, codebook_size = load_tiger_model(
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
    for param in tiger.parameters():
        param.requires_grad = False
    tiger.eval()

    iid2sid_tok_cpu, _sid2iid = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth), torch.device("cpu"))

    train_rows: List[Dict[str, Any]] = []
    keep = 0
    if chain_path is not None:
        for rec in raw_records:
            target_item = int(rec.get("selected_item_id", -1))
            target_tokens = [int(x) for x in rec.get("selected_sid_tokens", [])]
            history_items = [int(x) for x in rec.get("history_items", [])]
            block_credit = [float(x) for x in rec.get(str(args.chain_block_field), [])]
            item_credit = float(rec.get(str(args.chain_item_field), 0.0))
            if (
                target_item <= 0
                or len(target_tokens) != int(sid_depth)
                or len(block_credit) != int(sid_depth)
                or not any(int(x) > 0 for x in target_tokens)
            ):
                continue
            hist_tensor = torch.tensor(history_items[-int(args.max_hist_items):], dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(args.max_hist_items),
                int(sid_depth),
            )
            train_rows.append(
                {
                    "group": str(rec["episode_id"]),
                    "input_ids": input_ids.squeeze(0).tolist(),
                    "attention_mask": attention_mask.squeeze(0).tolist(),
                    "target_tokens": list(target_tokens),
                    "raw_return": float(rec.get("item_credit_raw", item_credit)),
                    "item_credit": float(item_credit),
                    "block_credit": list(block_credit),
                    "selected_item_id": int(target_item),
                }
            )
            keep += 1
    else:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in raw_records:
            grouped.setdefault(str(rec["episode_id"]), []).append(rec)

        for episode_id, pages in grouped.items():
            pages = sorted(pages, key=lambda x: int(x["page_index"]))
            rewards = [float(x.get("step_reward", 0.0)) for x in pages]
            returns = [0.0 for _ in pages]
            running = 0.0
            for idx in range(len(pages) - 1, -1, -1):
                running = rewards[idx] + float(args.gamma) * running
                returns[idx] = running
            credits = transform_episode_credits(
                returns,
                mode=str(args.credit_mode),
                clip=float(args.credit_clip),
            )
            for idx, rec in enumerate(pages):
                target_item = int(rec.get("selected_item_id", -1))
                target_tokens = [int(x) for x in rec.get("selected_sid_tokens", [])]
                history_items = [int(x) for x in rec.get("history_items", [])]
                if target_item <= 0 or len(target_tokens) != int(sid_depth) or not any(int(x) > 0 for x in target_tokens):
                    continue
                hist_tensor = torch.tensor(history_items[-int(args.max_hist_items):], dtype=torch.long).view(1, -1)
                input_ids, attention_mask = build_history_tokens(
                    hist_tensor,
                    iid2sid_tok_cpu,
                    int(args.max_hist_items),
                    int(sid_depth),
                )
                item_credit = float(credits[idx])
                block_credit = build_block_credit(
                    history_items=history_items,
                    target_tokens=target_tokens,
                    item_credit=item_credit,
                    iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                    epsilon=float(args.transport_epsilon),
                    n_iter=int(args.transport_iter),
                )
                train_rows.append(
                    {
                        "group": str(episode_id),
                        "input_ids": input_ids.squeeze(0).tolist(),
                        "attention_mask": attention_mask.squeeze(0).tolist(),
                        "target_tokens": list(target_tokens),
                        "raw_return": float(returns[idx]),
                        "item_credit": float(item_credit),
                        "block_credit": block_credit.tolist(),
                        "selected_item_id": int(target_item),
                    }
                )
                keep += 1

    if not train_rows:
        raise ValueError("No usable phase2 rows built from traces.")

    dataset = Phase2RecordDataset(train_rows)
    groups = [row["group"] for row in train_rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_phase2,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_phase2,
    )

    head = TokenCreditTransportHead(
        hidden_size=int(size_cfg["d_model"]),
        vocab_size=int(codebook_size) + 1,
        token_dim=int(args.token_dim),
        mlp_dim=int(args.mlp_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr))

    best_key = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        train_losses: List[float] = []
        for input_ids, attention_mask, target_tokens, block_credit, item_credit, _groups in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            block_credit = block_credit.to(device)
            item_credit = item_credit.to(device)
            with torch.no_grad():
                _logits, hidden = tiger.decode_with_hidden(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids_from_targets(target_tokens),
                )
            pred = head(hidden.detach(), target_tokens)
            pred_item = pred.sum(dim=1)
            loss_block = F.smooth_l1_loss(pred, block_credit)
            loss_cons = F.mse_loss(pred_item, item_credit)
            loss_sign = F.binary_cross_entropy_with_logits(pred, (block_credit >= 0.0).float())
            loss = loss_block + float(args.conservation_scale) * loss_cons + float(args.sign_scale) * loss_sign
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_metrics = evaluate_head(tiger, head, valid_loader, device)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(dict(valid_metrics))
        key = float(valid_metrics["item_mse"] + 0.5 * valid_metrics["block_mse"])
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            best_metrics = dict(valid_metrics)
            best_epoch = int(epoch)
        print(
            f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
            f"item_mse={valid_metrics['item_mse']:.4f} block_mse={valid_metrics['block_mse']:.4f} "
            f"sign_acc={valid_metrics['sign_acc']:.4f}"
        )

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        best_metrics = evaluate_head(tiger, head, valid_loader, device)
        best_epoch = int(args.epochs)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(args.tiger_ckpt).resolve().parent / "phase2_blend"
    save_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_dir / "phase2_blend_head.pt"
    meta_path = save_dir / "phase2_blend_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "phase2_blend_metrics.json"
    torch.save({"model_state_dict": best_state}, head_path)
    meta = {
        "method": "TIGER Base + Phase2 Blend Decode",
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "trace_dir": str(trace_dir.resolve()),
        "chain_path": str(chain_path.resolve()) if chain_path is not None else "",
        "model_size": str(args.model_size),
        "hidden_size": int(size_cfg["d_model"]),
        "vocab_size": int(codebook_size) + 1,
        "sid_depth": int(sid_depth),
        "token_dim": int(args.token_dim),
        "mlp_dim": int(args.mlp_dim),
        "gamma": float(args.gamma),
        "credit_mode": str(args.credit_mode),
        "credit_clip": float(args.credit_clip),
        "chain_block_field": str(args.chain_block_field),
        "chain_item_field": str(args.chain_item_field),
        "transport_epsilon": float(args.transport_epsilon),
        "transport_iter": int(args.transport_iter),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_trace_records": int(len(raw_records)),
        "n_train_rows": int(len(train_rows)),
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
            "n_trace_records": int(len(raw_records)),
            "n_train_rows": int(len(train_rows)),
        },
    )
    print(f"[phase2] saved head to {head_path}")
    print(f"[phase2] saved meta to {meta_path}")
    print(f"[phase2] saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
