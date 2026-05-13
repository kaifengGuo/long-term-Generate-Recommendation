# -*- coding: utf-8 -*-
"""
Train a BERT4Rec sequential baseline under the same data/mapping convention as
the existing SASRec baseline.

Compliance notes:
- Uses the exact same global video_id -> item_id mapping builder as SASRec.
- Uses click-only user sequences, matching SASRec/GRU4Rec history semantics.
- Saves a checkpoint that can be evaluated in KREnvironment_WholeSession_GPU
  with the same slate_size/candidate pool/reward aggregation as other baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.bert4rec import BERT4Rec, BERT4RecConfig
from sasrec_utils import SASRecTrainDataset, SASRecValDataset, evaluate_full_ranking, evaluate_sampled_ranking, set_seed
from train_sasrec_baseline import build_global_mapping_and_click_sequences, save_json


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_mlm_batch(
    seq: torch.Tensor,
    target: torch.Tensor,
    *,
    n_items: int,
    mask_id: int,
    mask_prob: float,
    mask_token_prob: float,
    random_token_prob: float,
    force_last_mask: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SASRecTrainDataset returns left-padded prefix and next target. For BERT4Rec
    we form a full training sequence by appending target at the final position,
    then apply Cloze masking. The final target position is masked by default so
    the offline objective is aligned with online next-item scoring.
    """
    device = seq.device
    full = torch.cat([seq[:, 1:], target.view(-1, 1)], dim=1).long()
    labels = torch.full_like(full, -100)
    nonpad = full.gt(0)
    sampled = torch.rand(full.shape, device=device).lt(float(mask_prob)) & nonpad
    if bool(force_last_mask):
        sampled[:, -1] = target.gt(0)
    no_mask_rows = sampled.sum(dim=1).eq(0)
    if no_mask_rows.any():
        sampled[no_mask_rows, -1] = target[no_mask_rows].gt(0)

    labels[sampled] = full[sampled]
    input_ids = full.clone()

    replace_draw = torch.rand(full.shape, device=device)
    use_mask = sampled & replace_draw.lt(float(mask_token_prob))
    use_random = sampled & replace_draw.ge(float(mask_token_prob)) & replace_draw.lt(float(mask_token_prob) + float(random_token_prob))
    input_ids[use_mask] = int(mask_id)
    if use_random.any():
        input_ids[use_random] = torch.randint(1, int(n_items) + 1, size=(int(use_random.sum().item()),), device=device)
    return input_ids, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument(
        "--mapping_order", type=str, default="sorted", choices=["sorted", "first_seen"],
        help="Item-id mapping order. first_seen matches KRMBSeqReader / online env candidate ids."
    )
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_neg", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=2026)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_metric_k", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--mask_prob", type=float, default=0.20)
    parser.add_argument("--mask_token_prob", type=float, default=0.80)
    parser.add_argument("--random_token_prob", type=float, default=0.10)
    parser.add_argument("--force_last_mask", action="store_true", default=True)
    parser.add_argument("--no_force_last_mask", action="store_false", dest="force_last_mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    os.makedirs(args.save_dir, exist_ok=True)

    cache_npz = ""
    cache_meta = ""
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_npz = os.path.join(args.cache_dir, "bert4rec_cache.npz")
        cache_meta = os.path.join(args.cache_dir, "bert4rec_meta.json")

    if cache_npz and os.path.exists(cache_npz) and os.path.exists(cache_meta):
        print(f"[Data] Loading cached sequences: {cache_npz}")
        pack = np.load(cache_npz, allow_pickle=True)
        user_seqs = pack["user_seqs"].tolist()
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n_items_total = int(meta["n_items_total"])
    else:
        print(f"[Data] Building GLOBAL mapping + click sequences from CSV: {args.csv_path}")
        print(f"[Data] mapping_order={args.mapping_order}")
        user_seqs, vid2iid, uid2uix, n_items_total = build_global_mapping_and_click_sequences(
            args.csv_path,
            mapping_order=args.mapping_order,
        )
        print(f"[Data] n_users={len(user_seqs)} n_items_total={n_items_total} clicks_total={sum(len(s) for s in user_seqs)}")
        if cache_npz:
            np.savez_compressed(cache_npz, user_seqs=np.array(user_seqs, dtype=object))
            save_json({
                "vid2iid": vid2iid,
                "uid2uix": uid2uix,
                "n_items_total": n_items_total,
                "mapping_order": args.mapping_order,
            }, cache_meta)
            print(f"[Data] Saved cache to {cache_npz} and {cache_meta}")

    mx = max((max(s) for s in user_seqs if len(s) > 0), default=0)
    mn = min((min(s) for s in user_seqs if len(s) > 0), default=0)
    print(f"[Data] Summary: n_users={len(user_seqs)} n_items_total={n_items_total} clicks_total={sum(len(s) for s in user_seqs)}")
    print(f"[Data] Sanity: seq_id_min={mn} seq_id_max={mx}")
    if mx > n_items_total or mn < 1:
        raise RuntimeError(f"[Data] ID out of range: min={mn} max={mx} n_items_total={n_items_total}")

    train_ds = SASRecTrainDataset(user_seqs=user_seqs, max_len=int(args.max_len))
    val_ds = SASRecValDataset(user_seqs=user_seqs, max_len=int(args.max_len))
    print(f"[Data] Train samples={len(train_ds)} Val users={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    cfg = BERT4RecConfig(
        n_items=int(n_items_total),
        max_len=int(args.max_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    )
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    model = BERT4Rec(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[Model] {cfg}")
    print(f"[Model] mask_id={model.mask_id}")
    print(f"[Model] Total parameters: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_metric = -1.0
    bad_epochs = 0
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    save_json(
        {
            "cfg": asdict(cfg),
            "args": vars(args),
            "n_params": n_params,
            "n_users": len(user_seqs),
            "n_items_total": int(n_items_total),
            "baseline": "BERT4Rec",
            "mapping_order": args.mapping_order,
            "compliance": "same CSV, global item mapping, click-only histories, same strict env eval wrapper",
        },
        os.path.join(args.save_dir, "run_meta.json"),
    )

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_step = 0

        for seq, target in train_loader:
            s_max = int(seq.max().item())
            s_min = int(seq.min().item())
            t_max = int(target.max().item())
            t_min = int(target.min().item())
            if s_min < 0 or s_max > n_items_total or t_min < 1 or t_max > n_items_total:
                raise RuntimeError(f"[OOB] seq_min={s_min} seq_max={s_max} target_min={t_min} target_max={t_max} n_items={n_items_total}")

            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            input_ids, labels = build_mlm_batch(
                seq,
                target,
                n_items=int(n_items_total),
                mask_id=int(model.mask_id),
                mask_prob=float(args.mask_prob),
                mask_token_prob=float(args.mask_token_prob),
                random_token_prob=float(args.random_token_prob),
                force_last_mask=bool(args.force_last_mask),
            )
            loss = model.compute_mlm_loss(input_ids, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optim.step()

            loss_sum += float(loss.item())
            n_step += 1

        train_loss = loss_sum / max(n_step, 1)
        dt = time.time() - t0

        if args.eval_neg and int(args.eval_neg) > 0:
            val_metrics = evaluate_sampled_ranking(
                model,
                val_loader,
                device=device,
                n_items=int(n_items_total),
                n_neg=int(args.eval_neg),
                seed=int(args.eval_seed),
            )
        else:
            val_metrics = evaluate_full_ranking(model, val_loader, device=device)

        flat = " ".join([f"hit@{k}={val_metrics[f'hit@{k}']:.4f} ndcg@{k}={val_metrics[f'ndcg@{k}']:.4f}" for k in (1, 5, 10, 20)])
        print(f"[Epoch {ep}] Train loss: {train_loss:.4f}  dt={dt:.1f}s")
        print(f"[Epoch {ep}] Val: {flat}")

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep}, last_path)

        key = f"hit@{int(args.save_metric_k)}"
        cur = float(val_metrics.get(key, -1.0))
        improved = cur == cur and cur > best_metric + float(args.early_stop_min_delta)
        if improved:
            best_metric = cur
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep}, best_path)
            print(f"[Save] best.pt updated by {key}={best_metric:.6f}")
        else:
            bad_epochs += 1

        if int(args.early_stop_patience) > 0 and bad_epochs >= int(args.early_stop_patience):
            print(f"[EarlyStop] {key} did not improve for {args.early_stop_patience} consecutive epochs. Stop at epoch {ep}.")
            break

    print("[Done]")


if __name__ == "__main__":
    main()
