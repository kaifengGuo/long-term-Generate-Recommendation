import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_phase2_blend_common import infer_model_size_args, load_tiger_model  # noqa: E402

from tiger_hcaa.common import EpisodeDataset, collate_episodes, set_random_seed, write_json  # noqa: E402
from tiger_hcaa.models import HCAAJointCritic, load_hcaa_bundle, save_hcaa_bundle  # noqa: E402
from tiger_hcaa.train_hcaa_joint_critic import (  # noqa: E402
    build_samples_from_files,
    evaluate,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental TIGER-HCAA loss monitoring curve.")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--base_tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--users_per_step", type=int, default=64)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rollout_log_every", type=int, default=16)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--chain_credit_mode", type=str, default="return")
    parser.add_argument("--allocator_head_path", type=str, default="")
    parser.add_argument("--allocator_meta_path", type=str, default="")
    parser.add_argument("--allocator_blend_alpha", type=float, default=0.7)
    parser.add_argument("--allocator_keep_topk", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_passes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--pre_loss_scale", type=float, default=1.0)
    parser.add_argument("--post_loss_scale", type=float, default=1.0)
    parser.add_argument("--bias_loss_scale", type=float, default=0.5)
    parser.add_argument("--page_loss_scale", type=float, default=1.0)
    parser.add_argument("--item_share_loss_scale", type=float, default=0.5)
    parser.add_argument("--item_loss_scale", type=float, default=1.0)
    parser.add_argument("--token_share_loss_scale", type=float, default=0.5)
    parser.add_argument("--token_loss_scale", type=float, default=1.0)
    parser.add_argument("--page_cons_scale", type=float, default=1.0)
    parser.add_argument("--item_cons_scale", type=float, default=1.0)
    parser.add_argument("--token_cons_scale", type=float, default=1.0)
    parser.add_argument("--init_bundle_path", type=str, default="")
    parser.add_argument("--init_meta_path", type=str, default="")
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: Sequence[str], log_path: Path, *, reuse_existing: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and log_path.exists():
        return
    with log_path.open("w", encoding="utf-8") as fp:
        proc = subprocess.run(
            list(cmd),
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


def write_summary_files(summary: List[Dict[str, Any]], output_root: Path) -> Path:
    summary_jsonl = output_root / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as outfile:
        for row in summary:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(summary).to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
    build_plots(summary, output_root)
    return summary_jsonl


def build_plots(summary: List[Dict[str, Any]], output_root: Path) -> None:
    if not summary:
        return
    steps = [int(row["step"]) for row in summary]
    x_label = f"Step ({int(summary[0]['users_per_step'])} rollout users)"

    def draw(metric_keys: List[Tuple[str, str]], file_name: str, title: str, y_label: str = "Metric") -> None:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        has_points = False
        for key, label in metric_keys:
            curve = [(int(row["step"]), float(row[key])) for row in summary if key in row]
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
            ("after_loss", "total_loss"),
            ("train_loss", "train_loss"),
        ],
        "hcaa_total_loss_vs_step.png",
        "HCAA Total Loss vs Step",
        y_label="Loss",
    )
    draw(
        [
            ("after_pre_loss", "pre_value"),
            ("after_post_loss", "post_value"),
            ("after_bias_loss", "page_bias"),
            ("after_page_loss", "page_adv"),
            ("after_page_cons_loss", "page_cons"),
        ],
        "hcaa_page_metrics_vs_step.png",
        "HCAA Page Metrics vs Step",
        y_label="Loss",
    )
    draw(
        [
            ("after_item_share_loss", "item_share"),
            ("after_item_loss", "item_adv"),
            ("after_item_cons_loss", "item_cons"),
        ],
        "hcaa_item_metrics_vs_step.png",
        "HCAA Item Metrics vs Step",
        y_label="Loss",
    )
    draw(
        [
            ("after_token_share_loss", "token_share"),
            ("after_token_loss", "token_adv"),
            ("after_token_cons_loss", "token_cons"),
        ],
        "hcaa_token_metrics_vs_step.png",
        "HCAA Token Metrics vs Step",
        y_label="Loss",
    )
    draw(
        [
            ("after_page_mae", "page_mae"),
            ("after_item_mae", "item_mae"),
            ("after_token_mae", "token_mae"),
        ],
        "hcaa_mae_vs_step.png",
        "HCAA MAE vs Step",
        y_label="MAE",
    )
    draw(
        [
            ("after_page_corr", "page_corr"),
            ("after_item_corr", "item_corr"),
            ("after_token_corr", "token_corr"),
        ],
        "hcaa_corr_vs_step.png",
        "HCAA Correlation vs Step",
        y_label="Correlation",
    )


def main() -> int:
    args = parse_args()
    set_random_seed(int(args.seed))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, codebook_size_model = load_tiger_model(
        tiger_ckpt=str(Path(args.base_tiger_ckpt).resolve()),
        sid_mapping_path=str(Path(args.sid_mapping_path).resolve()),
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

    model: HCAAJointCritic | None = None
    if str(args.init_bundle_path).strip() and str(args.init_meta_path).strip():
        model, _meta = load_hcaa_bundle(str(args.init_bundle_path), str(args.init_meta_path), device)
    optimizer: torch.optim.Optimizer | None = None

    summary: List[Dict[str, Any]] = []
    py = str(args.python_exe)

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
            str(Path(args.base_tiger_ckpt).resolve()),
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
        rollout_t0 = time.time()
        run_cmd(rollout_cmd, rollout_log, reuse_existing=bool(args.reuse_existing))
        rollout_seconds = float(time.time() - rollout_t0)

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
        chain_t0 = time.time()
        run_cmd(chain_cmd, chain_log, reuse_existing=bool(args.reuse_existing))
        chain_seconds = float(time.time() - chain_t0)

        samples, data_meta, _iid2sid = build_samples_from_files(
            trace_path=str(trace_path),
            chain_path=str(chain_path),
            uirm_log_path=str(args.uirm_log_path),
            sid_mapping_path=str(args.sid_mapping_path),
            max_hist_items=int(args.max_hist_items),
            gamma=float(args.gamma),
            hazard_lambda=float(args.hazard_lambda),
            max_episodes=0,
        )
        if not samples:
            raise ValueError(f"No HCAA samples built at step {step}")
        loader = DataLoader(
            EpisodeDataset(samples),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_episodes,
        )
        eval_loader = DataLoader(
            EpisodeDataset(samples),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_episodes,
        )

        if model is None:
            model = HCAAJointCritic(
                hidden_size=int(size_cfg["d_model"]),
                page_dim=int(data_meta["page_dim"]),
                item_dim=int(data_meta["item_dim"]),
                vocab_size=int(codebook_size_model) + 1,
                mlp_dim=int(args.mlp_dim),
                token_dim=int(args.token_dim),
                dropout=float(args.dropout),
            ).to(device)
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            )

        before = evaluate(tiger, model, eval_loader, device, args)
        train_metrics: Dict[str, float] = {}
        train_t0 = time.time()
        for _ in range(int(args.train_passes)):
            train_metrics = train_one_epoch(tiger, model, optimizer, loader, device, args)
        train_seconds = float(time.time() - train_t0)
        after = evaluate(tiger, model, eval_loader, device, args)

        joint_dir = step_dir / "joint_critic"
        joint_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = joint_dir / "hcaa_joint_bundle.pt"
        meta_path = joint_dir / "hcaa_joint_meta.json"
        meta = {
            "method": "TIGER-HCAA incremental joint critic",
            "hidden_size": int(size_cfg["d_model"]),
            "page_dim": int(data_meta["page_dim"]),
            "item_dim": int(data_meta["item_dim"]),
            "vocab_size": int(codebook_size_model) + 1,
            "mlp_dim": int(args.mlp_dim),
            "token_dim": int(args.token_dim),
            "dropout": float(args.dropout),
            "step": int(step),
            "data_meta": data_meta,
        }
        save_hcaa_bundle(bundle_path, meta_path, model, meta)
        write_json(
            joint_dir / "step_metrics.json",
            {
                "step": int(step),
                "before": before,
                "train": train_metrics,
                "after": after,
                "data_meta": data_meta,
            },
        )

        record = {
            "step": int(step),
            "users_per_step": int(args.users_per_step),
            "episodes": int(len(samples)),
            "trace_path": str(trace_path.resolve()),
            "chain_path": str(chain_path.resolve()),
            "bundle_path": str(bundle_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "rollout_seconds": float(rollout_seconds),
            "chain_seconds": float(chain_seconds),
            "train_seconds": float(train_seconds),
            "step_seconds": float(time.time() - step_start),
            "train_loss": float(train_metrics.get("loss", 0.0)),
        }
        for prefix, metrics in [("before", before), ("train", train_metrics), ("after", after)]:
            for key, value in metrics.items():
                record[f"{prefix}_{key}"] = float(value)
        summary.append(record)
        summary_path = write_summary_files(summary, output_root)

    final_dir = output_root / "final_joint_critic"
    final_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        save_hcaa_bundle(
            final_dir / "hcaa_joint_bundle.pt",
            final_dir / "hcaa_joint_meta.json",
            model,
            {
                "method": "TIGER-HCAA incremental final critic",
                "hidden_size": int(size_cfg["d_model"]),
                "page_dim": int(model.page_dim),
                "item_dim": int(model.item_dim),
                "vocab_size": int(model.vocab_size),
                "mlp_dim": int(model.mlp_dim),
                "token_dim": int(model.token_dim),
                "dropout": float(model.dropout),
                "num_steps": int(args.num_steps),
            },
        )

    print(json.dumps({"output_root": str(output_root), "summary_path": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
