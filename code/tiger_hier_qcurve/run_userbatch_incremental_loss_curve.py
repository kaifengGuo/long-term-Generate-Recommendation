import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log
from tiger_hier_prefix_common import ItemPrefixValueHead, load_item_prefix_head, load_page_prefix_head
from tiger_phase2_blend_common import (
    TokenPrefixValueHead,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_prefix_value_head,
    load_tiger_model,
)
from tiger_phase6_joint_common import SlateCreditHead
from train_tiger_item_prefix_critic import (
    ItemPrefixDataset,
    collate_rows as collate_item_rows,
    evaluate_head as evaluate_item_head,
    load_chain_groups,
    load_trace_rows,
)
from train_tiger_page_prefix_critic import (
    PagePrefixDataset,
    collate_rows as collate_page_rows,
    evaluate_head as evaluate_page_head,
    pool_history_summary,
)
from train_tiger_prefix_critic import (
    PrefixCriticDataset,
    collate_prefix,
    evaluate_prefix_head,
    prefix_to_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental three-level Q loss curve with one step = one batch of rollout users."
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--base_tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--users_per_step", type=int, default=8)
    parser.add_argument("--holdout_users", type=int, default=0)
    parser.add_argument("--holdout_eval_every", type=int, default=0)
    parser.add_argument("--holdout_seed", type=int, default=12026)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--chain_credit_mode", type=str, default="centered")
    parser.add_argument("--page_credit_mode", type=str, default="centered")
    parser.add_argument("--rollout_log_every", type=int, default=8)
    parser.add_argument("--allocator_head_path", type=str, default="")
    parser.add_argument("--allocator_meta_path", type=str, default="")
    parser.add_argument("--allocator_blend_alpha", type=float, default=0.7)
    parser.add_argument("--allocator_keep_topk", type=int, default=2)
    parser.add_argument("--page_lr", type=float, default=1e-4)
    parser.add_argument("--item_lr", type=float, default=1e-4)
    parser.add_argument("--token_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--page_batch_size", type=int, default=64)
    parser.add_argument("--item_batch_size", type=int, default=128)
    parser.add_argument("--token_batch_size", type=int, default=64)
    parser.add_argument("--page_train_passes", type=int, default=1)
    parser.add_argument("--item_train_passes", type=int, default=1)
    parser.add_argument("--token_train_passes", type=int, default=1)
    parser.add_argument("--token_target_clip", type=float, default=0.0)
    parser.add_argument("--token_prefix_loss_scale", type=float, default=1.0)
    parser.add_argument("--token_delta_loss_scale", type=float, default=1.0)
    parser.add_argument("--token_item_loss_scale", type=float, default=0.5)
    parser.add_argument("--page_init_head_path", type=str, default="")
    parser.add_argument("--page_init_meta_path", type=str, default="")
    parser.add_argument("--item_init_head_path", type=str, default="")
    parser.add_argument("--item_init_meta_path", type=str, default="")
    parser.add_argument("--token_init_head_path", type=str, default="")
    parser.add_argument("--token_init_meta_path", type=str, default="")
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str], log_path: Path, *, reuse_existing: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and log_path.exists():
        return
    with log_path.open("w", encoding="utf-8") as fp:
        proc = subprocess.run(
            cmd,
            cwd=str(CODE_DIR),
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def maybe_append_allocator(cmd: List[str], args: argparse.Namespace) -> None:
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_head_path", str(Path(args.allocator_head_path).resolve())])
    if str(args.allocator_meta_path).strip():
        cmd.extend(["--allocator_meta_path", str(Path(args.allocator_meta_path).resolve())])
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_blend_alpha", str(float(args.allocator_blend_alpha))])
        if int(args.allocator_keep_topk) > 0:
            cmd.extend(["--allocator_keep_topk", str(int(args.allocator_keep_topk))])


def train_page_one_pass(
    tiger,
    head: SlateCreditHead,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    head.train()
    losses: List[float] = []
    for batch in loader:
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
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train_item_one_pass(
    head,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    head.train()
    losses: List[float] = []
    for batch in loader:
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
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train_token_one_pass(
    tiger,
    head,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    head.train()
    prefix_losses: List[float] = []
    delta_losses: List[float] = []
    item_losses: List[float] = []
    total_losses: List[float] = []
    for batch in loader:
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
            float(args.token_prefix_loss_scale) * loss_prefix
            + float(args.token_delta_loss_scale) * loss_delta
            + float(args.token_item_loss_scale) * loss_item
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        prefix_losses.append(float(loss_prefix.item()))
        delta_losses.append(float(loss_delta.item()))
        item_losses.append(float(loss_item.item()))
        total_losses.append(float(loss.item()))
    return {
        "train_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "train_prefix_loss": float(np.mean(prefix_losses)) if prefix_losses else 0.0,
        "train_delta_loss": float(np.mean(delta_losses)) if delta_losses else 0.0,
        "train_item_loss": float(np.mean(item_losses)) if item_losses else 0.0,
    }


def build_eval_bundle(
    trace_path: Path,
    chain_path: Path,
    iid2sid_tok_cpu: torch.Tensor,
    sid_depth: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    page_dataset = PagePrefixDataset(
        trace_path,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        gamma=0.9,
        page_delta_field="credit",
        credit_mode=str(args.page_credit_mode),
        credit_clip=0.0,
        max_states=0,
    )
    page_loader = DataLoader(
        page_dataset,
        batch_size=min(int(args.page_batch_size), max(1, len(page_dataset))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_page_rows,
    )

    trace_map = load_trace_rows(trace_path)
    chain_groups = load_chain_groups(chain_path)
    item_dataset = ItemPrefixDataset(
        trace_map,
        chain_groups,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        max_hist_items=int(args.max_hist_items),
        item_credit_field="item_credit",
        max_pages=0,
    )
    item_loader = DataLoader(
        item_dataset,
        batch_size=min(int(args.item_batch_size), max(1, len(item_dataset))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_item_rows,
    )

    token_dataset = PrefixCriticDataset(
        chain_path,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        sid_depth=int(sid_depth),
        max_hist_items=int(args.max_hist_items),
        token_credit_field="token_credit_calibrated",
        item_credit_field="item_credit",
        target_clip=float(args.token_target_clip),
        max_records=0,
    )
    token_loader = DataLoader(
        token_dataset,
        batch_size=min(int(args.token_batch_size), max(1, len(token_dataset))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_prefix,
    )
    return {
        "page_dataset": page_dataset,
        "page_loader": page_loader,
        "item_dataset": item_dataset,
        "item_loader": item_loader,
        "token_dataset": token_dataset,
        "token_loader": token_loader,
    }


def prepare_fixed_holdout_assets(
    args: argparse.Namespace,
    py: str,
    output_root: Path,
) -> Optional[Dict[str, Path]]:
    if int(args.holdout_users) <= 0 or int(args.holdout_eval_every) <= 0:
        return None

    holdout_dir = output_root / f"fixed_holdout_{int(args.holdout_users)}users"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    trace_path = holdout_dir / "holdout_trace.jsonl"
    rollout_log = holdout_dir / "holdout_rollout.log"
    rollout_cmd = [
        py,
        str((CODE_DIR / "eval_tiger_phase2_blend_env.py").resolve()),
        "--uirm_log_path",
        str(Path(args.uirm_log_path).resolve()),
        "--slate_size",
        str(int(args.slate_size)),
        "--tiger_ckpt",
        str(Path(args.base_tiger_ckpt).resolve()),
        "--sid_mapping_path",
        str(Path(args.sid_mapping_path).resolve()),
        "--num_episodes",
        str(int(args.holdout_users)),
        "--episode_batch_size",
        str(min(128, max(1, int(args.holdout_users)))),
        "--beam_width",
        str(int(args.beam_width)),
        "--model_size",
        str(args.model_size),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.holdout_seed)),
        "--log_every",
        str(max(1, min(int(args.rollout_log_every), int(args.holdout_users)))),
        "--trace_path",
        str(trace_path),
        "--eval_log_path",
        str(rollout_log),
        "--max_hist_items",
        str(int(args.max_hist_items)),
        "--fast_base_generate",
    ]
    run_cmd(rollout_cmd, rollout_log, reuse_existing=bool(args.reuse_existing))

    chain_path = holdout_dir / "holdout_chain.jsonl"
    chain_log = holdout_dir / "build_chain.log"
    chain_cmd = [
        py,
        str((CODE_DIR / "build_tiger_slate_credit_chain.py").resolve()),
        "--trace_path",
        str(trace_path),
        "--uirm_log_path",
        str(Path(args.uirm_log_path).resolve()),
        "--sid_mapping_path",
        str(Path(args.sid_mapping_path).resolve()),
        "--device",
        "cpu",
        "--credit_mode",
        str(args.chain_credit_mode),
        "--max_hist_items",
        str(int(args.max_hist_items)),
        "--output_path",
        str(chain_path),
    ]
    maybe_append_allocator(chain_cmd, args)
    run_cmd(chain_cmd, chain_log, reuse_existing=bool(args.reuse_existing))

    return {
        "trace_path": trace_path.resolve(),
        "chain_path": chain_path.resolve(),
        "holdout_dir": holdout_dir.resolve(),
    }


def build_plots(summary: List[Dict[str, Any]], output_root: Path) -> None:
    steps = [int(row["step"]) for row in summary]
    users_per_step = int(summary[0]["users_per_step"]) if summary else 0
    x_label = f"Step ({users_per_step} rollout users)"

    def draw(metric_keys: List[Tuple[str, str]], file_name: str, title: str, y_label: str = "Loss") -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        has_points = False
        for key, label in metric_keys:
            curve = [(int(row["step"]), float(row[key])) for row in summary if key in row and row.get(key) != ""]
            if not curve:
                continue
            has_points = True
            ax.plot(
                [step for step, _value in curve],
                [value for _step, value in curve],
                marker="o",
                linewidth=2,
                label=label,
            )
        if not has_points:
            plt.close(fig)
            return
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_root / file_name, dpi=180)
        plt.close(fig)

    draw(
        [
            ("page_loss_before_mse", "Page MSE before"),
            ("page_loss_after_mse", "Page MSE after"),
            ("page_loss_before_mae", "Page MAE before"),
            ("page_loss_after_mae", "Page MAE after"),
        ],
        "page_loss_curve.png",
        "Page Q Loss vs Step",
        y_label="Loss / Error",
    )
    draw(
        [
            ("item_loss_before_mse", "Item MSE before"),
            ("item_loss_after_mse", "Item MSE after"),
            ("item_loss_before_mae", "Item MAE before"),
            ("item_loss_after_mae", "Item MAE after"),
        ],
        "item_loss_curve.png",
        "Item Q Loss vs Step",
        y_label="Loss / Error",
    )
    draw(
        [
            ("token_loss_before_delta", "Token delta before"),
            ("token_loss_after_delta", "Token delta after"),
            ("token_loss_before_prefix", "Token prefix before"),
            ("token_loss_after_prefix", "Token prefix after"),
            ("token_loss_before_item", "Token item before"),
            ("token_loss_after_item", "Token item after"),
        ],
        "token_loss_curve.png",
        "Token Q Loss vs Step",
        y_label="Loss",
    )
    holdout_users = int(summary[0].get("holdout_users", 0)) if summary else 0
    if holdout_users > 0:
        draw(
            [
                ("page_holdout_mse", "Page holdout MSE"),
                ("page_holdout_mae", "Page holdout MAE"),
            ],
            "page_holdout_curve.png",
            f"Page Q Fixed Holdout Loss vs Step ({holdout_users} users)",
            y_label="Loss / Error",
        )
        draw(
            [
                ("item_holdout_mse", "Item holdout MSE"),
                ("item_holdout_mae", "Item holdout MAE"),
            ],
            "item_holdout_curve.png",
            f"Item Q Fixed Holdout Loss vs Step ({holdout_users} users)",
            y_label="Loss / Error",
        )
        draw(
            [
                ("token_holdout_delta", "Token holdout delta"),
                ("token_holdout_prefix", "Token holdout prefix"),
                ("token_holdout_item", "Token holdout item"),
            ],
            "token_holdout_curve.png",
            f"Token Q Fixed Holdout Loss vs Step ({holdout_users} users)",
            y_label="Loss",
        )


def write_summary_files(summary: List[Dict[str, Any]], output_root: Path) -> Path:
    summary_path = output_root / "userbatch_incremental_loss_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_root / "userbatch_incremental_loss_summary.csv"
    header = [
        "step",
        "users_per_step",
        "page_states",
        "item_states",
        "token_states",
        "page_loss_before_mse",
        "page_loss_after_mse",
        "page_holdout_mse",
        "page_holdout_mae",
        "item_loss_before_mse",
        "item_loss_after_mse",
        "item_holdout_mse",
        "item_holdout_mae",
        "token_loss_before_delta",
        "token_loss_after_delta",
        "token_holdout_prefix",
        "token_holdout_delta",
        "token_holdout_item",
        "holdout_users",
        "holdout_eval_every",
        "holdout_page_states",
        "holdout_item_states",
        "holdout_token_states",
        "holdout_eval_seconds",
        "step_seconds",
    ]
    lines = [",".join(header)]
    for row in summary:
        lines.append(",".join(str(row.get(key, "")) for key in header))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_plots(summary, output_root)
    return summary_path


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(str(args.device))
    py = str(args.python_exe)

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])

    size_args = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, codebook_size = load_tiger_model(
        tiger_ckpt=str(args.base_tiger_ckpt),
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
    for p in tiger.parameters():
        p.requires_grad = False
    tiger.eval()

    if str(args.page_init_head_path).strip() and str(args.page_init_meta_path).strip():
        page_head, _page_meta = load_page_prefix_head(str(args.page_init_head_path), str(args.page_init_meta_path), device)
    else:
        page_head = SlateCreditHead(hidden_size=int(size_args["d_model"]), mlp_dim=128).to(device)
    item_head = None
    if str(args.item_init_head_path).strip() and str(args.item_init_meta_path).strip():
        item_head, _item_meta = load_item_prefix_head(str(args.item_init_head_path), str(args.item_init_meta_path), device)
    token_head = None
    if str(args.token_init_head_path).strip() and str(args.token_init_meta_path).strip():
        token_head, _token_meta = load_prefix_value_head(str(args.token_init_head_path), str(args.token_init_meta_path), device)

    page_optimizer = torch.optim.AdamW(page_head.parameters(), lr=float(args.page_lr), weight_decay=float(args.weight_decay))
    item_optimizer = (
        torch.optim.AdamW(item_head.parameters(), lr=float(args.item_lr), weight_decay=float(args.weight_decay))
        if item_head is not None
        else None
    )
    token_optimizer = (
        torch.optim.AdamW(token_head.parameters(), lr=float(args.token_lr), weight_decay=float(args.weight_decay))
        if token_head is not None
        else None
    )

    summary: List[Dict[str, Any]] = []
    rollout_policy_ckpt = str(Path(args.base_tiger_ckpt).resolve())
    holdout_assets = prepare_fixed_holdout_assets(args, py, output_root)
    holdout_bundle = (
        build_eval_bundle(
            holdout_assets["trace_path"],
            holdout_assets["chain_path"],
            iid2sid_tok_cpu,
            sid_depth,
            args,
        )
        if holdout_assets is not None
        else None
    )
    holdout_meta = {}
    if holdout_bundle is not None:
        holdout_meta = {
            "holdout_users": int(args.holdout_users),
            "holdout_eval_every": int(args.holdout_eval_every),
            "holdout_page_states": int(len(holdout_bundle["page_dataset"])),
            "holdout_item_states": int(len(holdout_bundle["item_dataset"])),
            "holdout_token_states": int(len(holdout_bundle["token_dataset"])),
            "holdout_trace_path": str(holdout_assets["trace_path"]),
            "holdout_chain_path": str(holdout_assets["chain_path"]),
        }

    for step in range(1, int(args.num_steps) + 1):
        step_start = time.time()
        step_dir = output_root / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        trace_path = step_dir / "rollout_trace.jsonl"
        rollout_log = step_dir / "rollout.log"
        rollout_cmd = [
            py,
            str((CODE_DIR / "eval_tiger_phase2_blend_env.py").resolve()),
            "--uirm_log_path",
            str(Path(args.uirm_log_path).resolve()),
            "--slate_size",
            str(int(args.slate_size)),
            "--tiger_ckpt",
            rollout_policy_ckpt,
            "--sid_mapping_path",
            str(Path(args.sid_mapping_path).resolve()),
            "--num_episodes",
            str(int(args.users_per_step)),
            "--episode_batch_size",
            str(int(args.users_per_step)),
            "--beam_width",
            str(int(args.beam_width)),
            "--model_size",
            str(args.model_size),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed) + step),
            "--log_every",
            str(int(args.rollout_log_every)),
            "--trace_path",
            str(trace_path),
            "--eval_log_path",
            str(rollout_log),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--fast_base_generate",
        ]
        run_cmd(rollout_cmd, rollout_log, reuse_existing=bool(args.reuse_existing))

        chain_path = step_dir / "rollout_chain.jsonl"
        chain_log = step_dir / "build_chain.log"
        chain_cmd = [
            py,
            str((CODE_DIR / "build_tiger_slate_credit_chain.py").resolve()),
            "--trace_path",
            str(trace_path),
            "--uirm_log_path",
            str(Path(args.uirm_log_path).resolve()),
            "--sid_mapping_path",
            str(Path(args.sid_mapping_path).resolve()),
            "--device",
            "cpu",
            "--credit_mode",
            str(args.chain_credit_mode),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--output_path",
            str(chain_path),
        ]
        maybe_append_allocator(chain_cmd, args)
        run_cmd(chain_cmd, chain_log, reuse_existing=bool(args.reuse_existing))

        step_bundle = build_eval_bundle(trace_path, chain_path, iid2sid_tok_cpu, sid_depth, args)
        page_dataset = step_bundle["page_dataset"]
        page_loader = step_bundle["page_loader"]
        page_before = evaluate_page_head(tiger, page_head, page_loader, device)
        page_train_loss = 0.0
        for _ in range(int(args.page_train_passes)):
            page_train_loss = train_page_one_pass(tiger, page_head, page_optimizer, page_loader, device)
        page_after = evaluate_page_head(tiger, page_head, page_loader, device)

        item_dataset = step_bundle["item_dataset"]
        if item_head is None:
            first_row = item_dataset[0]
            item_head = ItemPrefixValueHead(
                item_dim=int(first_row["item_features"].shape[1]),
                page_dim=int(first_row["page_features"].shape[0]),
                hidden_dim=96,
                dropout=0.10,
                stats_dim=3,
            ).to(device)
            item_optimizer = torch.optim.AdamW(item_head.parameters(), lr=float(args.item_lr), weight_decay=float(args.weight_decay))
        item_loader = step_bundle["item_loader"]
        item_before = evaluate_item_head(item_head, item_loader, device)
        item_train_loss = 0.0
        for _ in range(int(args.item_train_passes)):
            item_train_loss = train_item_one_pass(item_head, item_optimizer, item_loader, device)
        item_after = evaluate_item_head(item_head, item_loader, device)

        token_dataset = step_bundle["token_dataset"]
        if token_head is None:
            token_head = TokenPrefixValueHead(
                hidden_size=int(size_args["d_model"]),
                vocab_size=int(codebook_size) + 1,
                token_dim=32,
                mlp_dim=128,
            ).to(device)
            token_optimizer = torch.optim.AdamW(token_head.parameters(), lr=float(args.token_lr), weight_decay=float(args.weight_decay))
        token_loader = step_bundle["token_loader"]
        token_before = evaluate_prefix_head(tiger, token_head, token_loader, device, argparse.Namespace())
        token_train = {
            "train_total_loss": 0.0,
            "train_prefix_loss": 0.0,
            "train_delta_loss": 0.0,
            "train_item_loss": 0.0,
        }
        for _ in range(int(args.token_train_passes)):
            token_train = train_token_one_pass(tiger, token_head, token_optimizer, token_loader, device, args)
        token_after = evaluate_prefix_head(tiger, token_head, token_loader, device, argparse.Namespace())

        holdout_metrics: Dict[str, Any] = {}
        if holdout_bundle is not None and int(args.holdout_eval_every) > 0 and step % int(args.holdout_eval_every) == 0:
            holdout_start = time.time()
            page_holdout = evaluate_page_head(tiger, page_head, holdout_bundle["page_loader"], device)
            item_holdout = evaluate_item_head(item_head, holdout_bundle["item_loader"], device)
            token_holdout = evaluate_prefix_head(
                tiger,
                token_head,
                holdout_bundle["token_loader"],
                device,
                argparse.Namespace(),
            )
            holdout_metrics = {
                "page_holdout_mse": float(page_holdout["mse"]),
                "page_holdout_mae": float(page_holdout["mae"]),
                "item_holdout_mse": float(item_holdout["mse"]),
                "item_holdout_mae": float(item_holdout["mae"]),
                "token_holdout_prefix": float(token_holdout["prefix_loss"]),
                "token_holdout_delta": float(token_holdout["delta_loss"]),
                "token_holdout_item": float(token_holdout["item_loss"]),
                "holdout_eval_seconds": float(time.time() - holdout_start),
            }

        record = {
            "step": int(step),
            "users_per_step": int(args.users_per_step),
            **holdout_meta,
            "page_states": int(len(page_dataset)),
            "item_states": int(len(item_dataset)),
            "token_states": int(len(token_dataset)),
            "page_loss_before_mse": float(page_before["mse"]),
            "page_loss_before_mae": float(page_before["mae"]),
            "page_train_loss": float(page_train_loss),
            "page_loss_after_mse": float(page_after["mse"]),
            "page_loss_after_mae": float(page_after["mae"]),
            "item_loss_before_mse": float(item_before["mse"]),
            "item_loss_before_mae": float(item_before["mae"]),
            "item_train_loss": float(item_train_loss),
            "item_loss_after_mse": float(item_after["mse"]),
            "item_loss_after_mae": float(item_after["mae"]),
            "token_loss_before_prefix": float(token_before["prefix_loss"]),
            "token_loss_before_delta": float(token_before["delta_loss"]),
            "token_loss_before_item": float(token_before["item_loss"]),
            "token_train_total_loss": float(token_train["train_total_loss"]),
            "token_train_prefix_loss": float(token_train["train_prefix_loss"]),
            "token_train_delta_loss": float(token_train["train_delta_loss"]),
            "token_train_item_loss": float(token_train["train_item_loss"]),
            "token_loss_after_prefix": float(token_after["prefix_loss"]),
            "token_loss_after_delta": float(token_after["delta_loss"]),
            "token_loss_after_item": float(token_after["item_loss"]),
            "trace_path": str(trace_path.resolve()),
            "chain_path": str(chain_path.resolve()),
            "step_seconds": float(time.time() - step_start),
        }
        record.update(holdout_metrics)
        summary.append(record)
        summary_path = write_summary_files(summary, output_root)

    print(json.dumps({"output_root": str(output_root), "summary_path": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
