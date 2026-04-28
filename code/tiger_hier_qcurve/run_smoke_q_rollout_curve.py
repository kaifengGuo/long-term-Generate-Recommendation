import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke iterative rollout experiment for three-level hierarchical Q diagnostics."
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--base_tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--rollout_episodes", type=int, default=24)
    parser.add_argument("--episode_batch_size", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_every", type=int, default=12)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--chain_credit_mode", type=str, default="centered")
    parser.add_argument("--page_credit_mode", type=str, default="centered")
    parser.add_argument("--page_epochs", type=int, default=2)
    parser.add_argument("--item_epochs", type=int, default=2)
    parser.add_argument("--token_epochs", type=int, default=2)
    parser.add_argument("--actor_epochs", type=int, default=1)
    parser.add_argument("--page_valid_ratio", type=float, default=0.10)
    parser.add_argument("--item_valid_ratio", type=float, default=0.10)
    parser.add_argument("--token_valid_ratio", type=float, default=0.10)
    parser.add_argument("--actor_valid_ratio", type=float, default=0.15)
    parser.add_argument("--actor_train_scope", type=str, default="last_decoder_block")
    parser.add_argument("--allocator_head_path", type=str, default="")
    parser.add_argument("--allocator_meta_path", type=str, default="")
    parser.add_argument("--allocator_blend_alpha", type=float, default=0.7)
    parser.add_argument("--allocator_keep_topk", type=int, default=2)
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


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_history_record(metrics: Dict[str, Any]) -> Dict[str, Any]:
    best_epoch = int(metrics["best_epoch"])
    for record in metrics.get("history", []):
        if int(record.get("epoch", -1)) == best_epoch:
            return record
    raise KeyError(f"best_epoch={best_epoch} not found in history")


def maybe_append_allocator(cmd: List[str], args: argparse.Namespace) -> None:
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_head_path", str(Path(args.allocator_head_path).resolve())])
    if str(args.allocator_meta_path).strip():
        cmd.extend(["--allocator_meta_path", str(Path(args.allocator_meta_path).resolve())])
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_blend_alpha", str(float(args.allocator_blend_alpha))])
        if int(args.allocator_keep_topk) > 0:
            cmd.extend(["--allocator_keep_topk", str(int(args.allocator_keep_topk))])


def build_plots(summary: List[Dict[str, Any]], output_root: Path) -> None:
    steps = [int(row["rollout_step"]) for row in summary]

    page_mse = [float(row["page_valid_mse"]) for row in summary]
    page_mae = [float(row["page_valid_mae"]) for row in summary]
    item_mse = [float(row["item_valid_mse"]) for row in summary]
    item_mae = [float(row["item_valid_mae"]) for row in summary]
    token_prefix = [float(row["token_valid_prefix_loss"]) for row in summary]
    token_delta = [float(row["token_valid_delta_loss"]) for row in summary]
    token_item = [float(row["token_valid_item_loss"]) for row in summary]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(steps, page_mse, marker="o", label="valid_mse")
    axes[0].plot(steps, page_mae, marker="s", label="valid_mae")
    axes[0].set_title("Page Q")
    axes[0].set_xlabel("Rollout Step")
    axes[0].set_ylabel("Metric")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, item_mse, marker="o", label="valid_mse")
    axes[1].plot(steps, item_mae, marker="s", label="valid_mae")
    axes[1].set_title("Item Q")
    axes[1].set_xlabel("Rollout Step")
    axes[1].set_ylabel("Metric")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(steps, token_prefix, marker="o", label="prefix_loss")
    axes[2].plot(steps, token_delta, marker="s", label="delta_loss")
    axes[2].plot(steps, token_item, marker="^", label="item_loss")
    axes[2].set_title("Token Q")
    axes[2].set_xlabel("Rollout Step")
    axes[2].set_ylabel("Metric")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_root / "q_metrics_vs_rollout_step.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    def normalized(vals: List[float]) -> List[float]:
        base = float(vals[0]) if vals and abs(float(vals[0])) > 1e-12 else 1.0
        return [float(v) / base for v in vals]

    axes[0].plot(steps, normalized(page_mse), marker="o", label="valid_mse/base")
    axes[0].plot(steps, normalized(page_mae), marker="s", label="valid_mae/base")
    axes[0].set_title("Page Q (Normalized)")
    axes[0].set_xlabel("Rollout Step")
    axes[0].set_ylabel("Relative Metric")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, normalized(item_mse), marker="o", label="valid_mse/base")
    axes[1].plot(steps, normalized(item_mae), marker="s", label="valid_mae/base")
    axes[1].set_title("Item Q (Normalized)")
    axes[1].set_xlabel("Rollout Step")
    axes[1].set_ylabel("Relative Metric")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(steps, normalized(token_prefix), marker="o", label="prefix/base")
    axes[2].plot(steps, normalized(token_delta), marker="s", label="delta/base")
    axes[2].plot(steps, normalized(token_item), marker="^", label="item/base")
    axes[2].set_title("Token Q (Normalized)")
    axes[2].set_xlabel("Rollout Step")
    axes[2].set_ylabel("Relative Metric")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_root / "q_metrics_vs_rollout_step_normalized.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    current_actor_ckpt = str(Path(args.base_tiger_ckpt).resolve())
    base_tiger_ckpt = str(Path(args.base_tiger_ckpt).resolve())
    uirm_log_path = str(Path(args.uirm_log_path).resolve())
    sid_mapping_path = str(Path(args.sid_mapping_path).resolve())
    py = str(args.python_exe)

    summary: List[Dict[str, Any]] = []

    for step in range(1, int(args.num_iters) + 1):
        step_dir = output_root / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        trace_path = step_dir / "rollout_trace.jsonl"
        rollout_log = step_dir / "rollout.log"
        rollout_cmd = [
            py,
            str((CODE_DIR / "eval_tiger_phase2_blend_env.py").resolve()),
            "--uirm_log_path",
            uirm_log_path,
            "--slate_size",
            str(int(args.slate_size)),
            "--tiger_ckpt",
            current_actor_ckpt,
            "--sid_mapping_path",
            sid_mapping_path,
            "--num_episodes",
            str(int(args.rollout_episodes)),
            "--episode_batch_size",
            str(int(args.episode_batch_size)),
            "--beam_width",
            str(int(args.beam_width)),
            "--model_size",
            str(args.model_size),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed) + step),
            "--log_every",
            str(int(args.log_every)),
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
            uirm_log_path,
            "--sid_mapping_path",
            sid_mapping_path,
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

        page_dir = step_dir / "page_q"
        page_log = step_dir / "train_page_q.log"
        page_cmd = [
            py,
            str((CODE_DIR / "train_tiger_page_prefix_critic.py").resolve()),
            "--trace_path",
            str(trace_path),
            "--uirm_log_path",
            uirm_log_path,
            "--sid_mapping_path",
            sid_mapping_path,
            "--tiger_ckpt",
            current_actor_ckpt,
            "--model_size",
            str(args.model_size),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--credit_mode",
            str(args.page_credit_mode),
            "--valid_ratio",
            str(float(args.page_valid_ratio)),
            "--epochs",
            str(int(args.page_epochs)),
            "--save_dir",
            str(page_dir),
        ]
        run_cmd(page_cmd, page_log, reuse_existing=bool(args.reuse_existing))

        item_dir = step_dir / "item_q"
        item_log = step_dir / "train_item_q.log"
        item_cmd = [
            py,
            str((CODE_DIR / "train_tiger_item_prefix_critic.py").resolve()),
            "--trace_path",
            str(trace_path),
            "--chain_path",
            str(chain_path),
            "--uirm_log_path",
            uirm_log_path,
            "--sid_mapping_path",
            sid_mapping_path,
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--valid_ratio",
            str(float(args.item_valid_ratio)),
            "--epochs",
            str(int(args.item_epochs)),
            "--save_dir",
            str(item_dir),
        ]
        run_cmd(item_cmd, item_log, reuse_existing=bool(args.reuse_existing))

        token_dir = step_dir / "token_q"
        token_log = step_dir / "train_token_q.log"
        token_cmd = [
            py,
            str((CODE_DIR / "train_tiger_prefix_critic.py").resolve()),
            "--chain_path",
            str(chain_path),
            "--uirm_log_path",
            uirm_log_path,
            "--sid_mapping_path",
            sid_mapping_path,
            "--tiger_ckpt",
            current_actor_ckpt,
            "--device",
            str(args.device),
            "--size",
            str(args.model_size),
            "--seed",
            str(int(args.seed)),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--valid_ratio",
            str(float(args.token_valid_ratio)),
            "--epochs",
            str(int(args.token_epochs)),
            "--save_dir",
            str(token_dir),
        ]
        run_cmd(token_cmd, token_log, reuse_existing=bool(args.reuse_existing))

        hier_chain_path = step_dir / "hier_prefix_chain.jsonl"
        hier_chain_log = step_dir / "build_hier_chain.log"
        hier_chain_cmd = [
            py,
            str((CODE_DIR / "build_tiger_hier_prefix_advantage_chain.py").resolve()),
            "--trace_path",
            str(trace_path),
            "--chain_path",
            str(chain_path),
            "--uirm_log_path",
            uirm_log_path,
            "--sid_mapping_path",
            sid_mapping_path,
            "--tiger_ckpt",
            current_actor_ckpt,
            "--model_size",
            str(args.model_size),
            "--device",
            str(args.device),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--page_head_path",
            str((page_dir / "page_prefix_head.pt").resolve()),
            "--page_meta_path",
            str((page_dir / "page_prefix_meta.json").resolve()),
            "--item_head_path",
            str((item_dir / "item_prefix_head.pt").resolve()),
            "--item_meta_path",
            str((item_dir / "item_prefix_meta.json").resolve()),
            "--token_head_path",
            str((token_dir / "prefix_critic_head.pt").resolve()),
            "--token_meta_path",
            str((token_dir / "prefix_critic_meta.json").resolve()),
            "--output_path",
            str(hier_chain_path),
        ]
        run_cmd(hier_chain_cmd, hier_chain_log, reuse_existing=bool(args.reuse_existing))

        actor_dir = step_dir / "hier_actor"
        actor_log = step_dir / "train_actor.log"
        actor_cmd = [
            py,
            str((CODE_DIR / "train_tiger_phase8_hier_actor.py").resolve()),
            "--chain_path",
            str(hier_chain_path),
            "--uirm_log_path",
            uirm_log_path,
            "--tiger_ckpt",
            base_tiger_ckpt,
            "--init_tiger_ckpt",
            current_actor_ckpt,
            "--sid_mapping_path",
            sid_mapping_path,
            "--model_size",
            str(args.model_size),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--valid_ratio",
            str(float(args.actor_valid_ratio)),
            "--epochs",
            str(int(args.actor_epochs)),
            "--train_scope",
            str(args.actor_train_scope),
            "--save_dir",
            str(actor_dir),
        ]
        run_cmd(actor_cmd, actor_log, reuse_existing=bool(args.reuse_existing))

        page_metrics = read_json(page_dir / "page_prefix_metrics.json")
        item_metrics = read_json(item_dir / "item_prefix_metrics.json")
        token_metrics = read_json(token_dir / "prefix_critic_metrics.json")
        actor_metrics = read_json(actor_dir / "phase8_hier_actor_metrics.json")
        hier_meta = read_json(hier_chain_path.with_suffix(hier_chain_path.suffix + ".meta.json"))

        page_best = best_history_record(page_metrics)
        item_best = best_history_record(item_metrics)
        token_best = best_history_record(token_metrics)
        actor_best = actor_metrics["best_metrics"]

        summary.append(
            {
                "rollout_step": int(step),
                "policy_ckpt_used_for_rollout": current_actor_ckpt,
                "trace_path": str(trace_path.resolve()),
                "chain_path": str(chain_path.resolve()),
                "hier_chain_path": str(hier_chain_path.resolve()),
                "page_valid_mse": float(page_best["valid_mse"]),
                "page_valid_mae": float(page_best["valid_mae"]),
                "item_valid_mse": float(item_best["valid_mse"]),
                "item_valid_mae": float(item_best["valid_mae"]),
                "token_valid_prefix_loss": float(token_best["valid_prefix_loss"]),
                "token_valid_delta_loss": float(token_best["valid_delta_loss"]),
                "token_valid_item_loss": float(token_best["valid_item_loss"]),
                "actor_valid_loss": float(actor_best["loss"]),
                "actor_target_gain": float(actor_best["target_gain"]),
                "hier_n_pages": int(hier_meta["n_pages"]),
                "hier_n_rows": int(hier_meta["n_rows"]),
                "next_actor_ckpt": str((actor_dir / "phase8_hier_actor_tiger.pth").resolve()),
            }
        )
        current_actor_ckpt = str((actor_dir / "phase8_hier_actor_tiger.pth").resolve())

    summary_path = output_root / "rollout_q_curve_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_root / "rollout_q_curve_summary.csv"
    header = [
        "rollout_step",
        "page_valid_mse",
        "page_valid_mae",
        "item_valid_mse",
        "item_valid_mae",
        "token_valid_prefix_loss",
        "token_valid_delta_loss",
        "token_valid_item_loss",
        "actor_valid_loss",
        "actor_target_gain",
        "hier_n_pages",
        "hier_n_rows",
    ]
    lines = [",".join(header)]
    for row in summary:
        lines.append(",".join(str(row[key]) for key in header))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    build_plots(summary, output_root)
    print(json.dumps({"output_root": str(output_root), "summary_path": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
