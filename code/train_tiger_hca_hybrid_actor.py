import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train_tiger_hca_grpo_actor as grpo_actor
import train_tiger_hca_pref_actor as pref_actor
import utils

from tiger_phase2_blend_common import infer_model_size_args, load_tiger_model, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid TIGER-HCA actor training: GRPO main objective with DPO-style conservative anchoring."
    )
    parser.add_argument("--group_path", type=str, required=True)
    parser.add_argument("--pair_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Frozen rollout/reference TIGER checkpoint.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group_adv_field", type=str, default="group_advantage")
    parser.add_argument("--token_adv_field", type=str, default="sid_advantage")
    parser.add_argument("--item_adv_field", type=str, default="item_advantage")
    parser.add_argument("--page_reward_field", type=str, default="reward_raw")
    parser.add_argument("--min_abs_group_adv", type=float, default=0.0)
    parser.add_argument("--group_max_rows", type=int, default=0)
    parser.add_argument("--pair_max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--item_adv_scale", type=float, default=0.10)
    parser.add_argument("--page_gate_scale", type=float, default=0.10)
    parser.add_argument("--page_gate_min", type=float, default=0.85)
    parser.add_argument("--page_gate_max", type=float, default=1.15)
    parser.add_argument(
        "--page_gate_mode",
        type=str,
        default="abs_tanh",
        choices=["abs_tanh", "signed_tanh", "positive_tanh", "none"],
    )
    parser.add_argument("--positive_topk", type=int, default=2)
    parser.add_argument("--positive_floor", type=float, default=0.0)
    parser.add_argument("--negative_topk", type=int, default=2)
    parser.add_argument("--negative_floor", type=float, default=0.0)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--clip_eps", type=float, default=0.20)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--adaptive_kl_support_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_kl_unc_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_clip_support_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_clip_unc_scale", type=float, default=0.0)
    parser.add_argument("--min_clip_eps", type=float, default=0.02)
    parser.add_argument("--trust_support_field", type=str, default="support_gap_scaled")
    parser.add_argument("--trust_unc_field", type=str, default="uncertainty_ratio")
    parser.add_argument("--entropy_scale", type=float, default=0.0)
    parser.add_argument("--grpo_sft_scale", type=float, default=0.0)
    parser.add_argument("--pref_anchor_scale", type=float, default=0.50)
    parser.add_argument("--pref_beta", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--pref_sft_scale", type=float, default=0.05)
    parser.add_argument("--gap_scale", type=float, default=1.0)
    parser.add_argument("--gap_clip", type=float, default=2.0)
    parser.add_argument("--score_normalization", type=str, default="mean_token", choices=["sum", "mean_token"])
    parser.add_argument("--attr_adv_mode", type=str, default="pess", choices=["raw", "pess"])
    parser.add_argument("--attr_pair_scale", type=float, default=0.0)
    parser.add_argument("--attr_item_scale", type=float, default=0.10)
    parser.add_argument("--attr_credit_clip", type=float, default=3.0)
    parser.add_argument("--attr_renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--attr_topk", type=int, default=2)
    parser.add_argument("--attr_floor", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    parser.add_argument("--step_metrics_csv_out", type=str, default="")
    parser.add_argument("--step_metrics_jsonl_out", type=str, default="")
    return parser.parse_args()


def cycle_loader(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def build_grpo_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        item_adv_scale=float(args.item_adv_scale),
        page_gate_scale=float(args.page_gate_scale),
        page_gate_min=float(args.page_gate_min),
        page_gate_max=float(args.page_gate_max),
        page_gate_mode=str(args.page_gate_mode),
        positive_topk=int(args.positive_topk),
        positive_floor=float(args.positive_floor),
        negative_topk=int(args.negative_topk),
        negative_floor=float(args.negative_floor),
        credit_clip=float(args.credit_clip),
        renorm_mode=str(args.renorm_mode),
        clip_eps=float(args.clip_eps),
        kl_scale=float(args.kl_scale),
        adaptive_kl_support_scale=float(args.adaptive_kl_support_scale),
        adaptive_kl_unc_scale=float(args.adaptive_kl_unc_scale),
        adaptive_clip_support_scale=float(args.adaptive_clip_support_scale),
        adaptive_clip_unc_scale=float(args.adaptive_clip_unc_scale),
        min_clip_eps=float(args.min_clip_eps),
        entropy_scale=float(args.entropy_scale),
        sft_scale=float(args.grpo_sft_scale),
    )


def build_pref_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pref_beta=float(args.pref_beta),
        label_smoothing=float(args.label_smoothing),
        sft_scale=float(args.pref_sft_scale),
        gap_scale=float(args.gap_scale),
        gap_clip=float(args.gap_clip),
        score_normalization=str(args.score_normalization),
        attr_pair_scale=float(args.attr_pair_scale),
        attr_item_scale=float(args.attr_item_scale),
        attr_credit_clip=float(args.attr_credit_clip),
        attr_renorm_mode=str(args.attr_renorm_mode),
        attr_topk=int(args.attr_topk),
        attr_floor=float(args.attr_floor),
    )


def evaluate_hybrid(
    actor_tiger,
    old_tiger,
    group_loader: DataLoader,
    pair_loader: DataLoader,
    device: torch.device,
    *,
    grpo_args: SimpleNamespace,
    pref_args: SimpleNamespace,
    pref_anchor_scale: float,
) -> Dict[str, float]:
    grpo_metrics = grpo_actor.evaluate_actor(actor_tiger, old_tiger, group_loader, device, grpo_args)
    pref_metrics = pref_actor.evaluate_actor(actor_tiger, old_tiger, pair_loader, device, pref_args)
    combined = float(grpo_metrics["loss"]) + float(pref_anchor_scale) * float(pref_metrics["loss"])
    metrics: Dict[str, float] = {
        "loss": combined,
        "pref_anchor_scale": float(pref_anchor_scale),
    }
    for key, value in grpo_metrics.items():
        metrics[f"grpo_{key}"] = float(value)
    for key, value in pref_metrics.items():
        metrics[f"pref_{key}"] = float(value)
    return metrics


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))

    grouped_rows = grpo_actor.load_group_rows(
        Path(args.group_path),
        group_adv_field=str(args.group_adv_field),
        token_adv_field=str(args.token_adv_field),
        item_adv_field=str(args.item_adv_field),
        page_reward_field=str(args.page_reward_field),
        trust_support_field=str(args.trust_support_field),
        trust_unc_field=str(args.trust_unc_field),
        min_abs_group_adv=float(args.min_abs_group_adv),
        max_rows=int(args.group_max_rows),
    )
    pair_rows = pref_actor.load_pair_rows(
        Path(args.pair_path),
        int(args.pair_max_rows),
        str(args.attr_adv_mode),
    )

    size_cfg = infer_model_size_args(str(args.model_size))
    old_tiger, sid_depth, _codebook_size = load_tiger_model(
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
    for param in old_tiger.parameters():
        param.requires_grad = False
    old_tiger.eval()

    actor_init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, _sid_depth2, _codebook_size2 = load_tiger_model(
        tiger_ckpt=str(actor_init_ckpt),
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
    n_trainable = grpo_actor.set_train_scope(actor_tiger, str(args.train_scope))

    group_dataset = grpo_actor.HCAGRPOGroupedDataset(grouped_rows)
    group_ids = [row["group"] for row in grouped_rows]
    group_train_idx, group_valid_idx = grpo_actor.split_groups(group_ids, float(args.valid_ratio), int(args.seed))
    group_train_loader = DataLoader(
        Subset(group_dataset, group_train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=grpo_actor.collate_rows,
    )
    group_valid_loader = DataLoader(
        Subset(group_dataset, group_valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=grpo_actor.collate_rows,
    )

    pair_dataset = pref_actor.TigerHCAPreferenceDataset(pair_rows)
    pair_ids = [row["group"] for row in pair_rows]
    pair_train_idx, pair_valid_idx = pref_actor.split_groups(pair_ids, float(args.valid_ratio), int(args.seed))
    pair_train_loader = DataLoader(
        Subset(pair_dataset, pair_train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=pref_actor.collate_rows,
    )
    pair_valid_loader = DataLoader(
        Subset(pair_dataset, pair_valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=pref_actor.collate_rows,
    )

    params = [param for param in actor_tiger.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    grpo_args = build_grpo_args(args)
    pref_args = build_pref_args(args)

    best_state = None
    best_epoch = 0
    best_key = float("inf")
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    save_dir = Path(args.save_dir) if str(args.save_dir).strip() else Path(args.tiger_ckpt).resolve().parent / "tiger_hca_hybrid_actor"
    save_dir.mkdir(parents=True, exist_ok=True)
    step_csv_path = Path(args.step_metrics_csv_out) if str(args.step_metrics_csv_out).strip() else save_dir / "tiger_hca_hybrid_actor_step_metrics.csv"
    step_jsonl_path = Path(args.step_metrics_jsonl_out) if str(args.step_metrics_jsonl_out).strip() else save_dir / "tiger_hca_hybrid_actor_step_metrics.jsonl"
    step_csv_path.parent.mkdir(parents=True, exist_ok=True)
    step_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0

    with step_csv_path.open("w", encoding="utf-8", newline="") as step_csv_fp, step_jsonl_path.open("w", encoding="utf-8") as step_jsonl_fp:
        csv_fieldnames = [
            "event_type",
            "epoch",
            "global_step",
            "step_in_epoch",
            "steps_per_epoch",
            "train_loss",
            "train_grpo_loss",
            "train_pref_loss",
            "train_clip_frac",
            "train_pair_acc",
        ]
        step_csv_writer = csv.DictWriter(step_csv_fp, fieldnames=csv_fieldnames)
        step_csv_writer.writeheader()

        for epoch in range(1, int(args.epochs) + 1):
            actor_tiger.train()
            train_total_losses: List[float] = []
            train_grpo_losses: List[float] = []
            train_pref_losses: List[float] = []
            train_clip_fracs: List[float] = []
            train_pair_accs: List[float] = []
            group_iter = cycle_loader(group_train_loader)
            pair_iter = cycle_loader(pair_train_loader)
            n_steps = max(len(group_train_loader), len(pair_train_loader))
            for step_idx in range(1, int(n_steps) + 1):
                group_batch = next(group_iter)
                pair_batch = next(pair_iter)
                grpo_loss, grpo_stats = grpo_actor.forward_actor(actor_tiger, old_tiger, group_batch, device, grpo_args)
                pref_loss, pref_stats = pref_actor.forward_actor(actor_tiger, old_tiger, pair_batch, device, pref_args)
                loss = grpo_loss + float(args.pref_anchor_scale) * pref_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
                optimizer.step()
                global_step += 1
                train_total_losses.append(float(loss.item()))
                train_grpo_losses.append(float(grpo_loss.item()))
                train_pref_losses.append(float(pref_loss.item()))
                train_clip_fracs.append(float(grpo_stats["clip_frac"]))
                train_pair_accs.append(float(pref_stats["pair_acc"]))

                step_row = {
                    "event_type": "train_step",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "step_in_epoch": int(step_idx),
                    "steps_per_epoch": int(n_steps),
                    "train_loss": float(loss.item()),
                    "train_grpo_loss": float(grpo_loss.item()),
                    "train_pref_loss": float(pref_loss.item()),
                    "train_clip_frac": float(grpo_stats.get("clip_frac", 0.0)),
                    "train_pair_acc": float(pref_stats.get("pair_acc", 0.0)),
                }
                step_csv_writer.writerow(step_row)
                step_jsonl_fp.write(json.dumps(step_row, ensure_ascii=False) + "\n")

            valid_metrics = evaluate_hybrid(
                actor_tiger,
                old_tiger,
                group_valid_loader,
                pair_valid_loader,
                device,
                grpo_args=grpo_args,
                pref_args=pref_args,
                pref_anchor_scale=float(args.pref_anchor_scale),
            )
            valid_metrics["epoch"] = float(epoch)
            valid_metrics["train_loss"] = float(np.mean(train_total_losses)) if train_total_losses else 0.0
            valid_metrics["train_grpo_loss"] = float(np.mean(train_grpo_losses)) if train_grpo_losses else 0.0
            valid_metrics["train_pref_loss"] = float(np.mean(train_pref_losses)) if train_pref_losses else 0.0
            valid_metrics["train_clip_frac"] = float(np.mean(train_clip_fracs)) if train_clip_fracs else 0.0
            valid_metrics["train_pair_acc"] = float(np.mean(train_pair_accs)) if train_pair_accs else 0.0
            history.append(dict(valid_metrics))
            epoch_row = {
                "event_type": "valid_epoch",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "step_in_epoch": int(n_steps),
                "steps_per_epoch": int(n_steps),
                "train_loss": float(valid_metrics["train_loss"]),
                "train_grpo_loss": float(valid_metrics["train_grpo_loss"]),
                "train_pref_loss": float(valid_metrics["train_pref_loss"]),
                "train_clip_frac": float(valid_metrics["train_clip_frac"]),
                "train_pair_acc": float(valid_metrics["train_pair_acc"]),
            }
            step_csv_writer.writerow(epoch_row)
            step_jsonl_fp.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")
            if float(valid_metrics["loss"]) < float(best_key):
                best_key = float(valid_metrics["loss"])
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu() for key, value in actor_tiger.state_dict().items()}
                best_metrics = dict(valid_metrics)
            print(
                f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} "
                f"valid_loss={valid_metrics['loss']:.4f} "
                f"grpo_valid={valid_metrics['grpo_loss']:.4f} "
                f"pref_valid={valid_metrics['pref_loss']:.4f} "
                f"clip_frac={valid_metrics['grpo_clip_frac']:.4f} "
                f"pair_acc={valid_metrics['pref_pair_acc']:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Hybrid actor training produced no checkpoint.")

    ckpt_path = save_dir / "tiger_hca_hybrid_actor_tiger.pth"
    meta_path = save_dir / "tiger_hca_hybrid_actor_meta.json"
    metrics_path = Path(args.metrics_out) if str(args.metrics_out).strip() else save_dir / "tiger_hca_hybrid_actor_metrics.json"
    torch.save(best_state, ckpt_path)
    meta = {
        "method": "TIGER-HCA Hybrid Actor",
        "group_path": str(Path(args.group_path).resolve()),
        "pair_path": str(Path(args.pair_path).resolve()),
        "old_policy_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "train_scope": str(args.train_scope),
        "n_trainable": int(n_trainable),
        "group_adv_field": str(args.group_adv_field),
        "token_adv_field": str(args.token_adv_field),
        "item_adv_field": str(args.item_adv_field),
        "page_reward_field": str(args.page_reward_field),
        "clip_eps": float(args.clip_eps),
        "kl_scale": float(args.kl_scale),
        "entropy_scale": float(args.entropy_scale),
        "grpo_sft_scale": float(args.grpo_sft_scale),
        "pref_anchor_scale": float(args.pref_anchor_scale),
        "pref_beta": float(args.pref_beta),
        "label_smoothing": float(args.label_smoothing),
        "pref_sft_scale": float(args.pref_sft_scale),
        "gap_scale": float(args.gap_scale),
        "gap_clip": float(args.gap_clip),
        "score_normalization": str(args.score_normalization),
        "attr_adv_mode": str(args.attr_adv_mode),
        "attr_pair_scale": float(args.attr_pair_scale),
        "attr_item_scale": float(args.attr_item_scale),
        "attr_credit_clip": float(args.attr_credit_clip),
        "attr_renorm_mode": str(args.attr_renorm_mode),
        "attr_topk": int(args.attr_topk),
        "attr_floor": float(args.attr_floor),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "step_metrics_csv_path": str(step_csv_path.resolve()),
        "step_metrics_jsonl_path": str(step_jsonl_path.resolve()),
        "n_group_rows": int(len(grouped_rows)),
        "n_pair_rows": int(len(pair_rows)),
        "n_group_train": int(len(group_train_idx)),
        "n_group_valid": int(len(group_valid_idx)),
        "n_pair_train": int(len(pair_train_idx)),
        "n_pair_valid": int(len(pair_valid_idx)),
    }
    write_json(meta_path, meta)
    write_json(
        metrics_path,
        {
            "tiger_ckpt": str(ckpt_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "step_metrics_csv_path": str(step_csv_path.resolve()),
            "step_metrics_jsonl_path": str(step_jsonl_path.resolve()),
            "history": history,
        },
    )
    print(f"[hca-hybrid] saved fine-tuned TIGER to {ckpt_path}")
    print(f"[hca-hybrid] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
