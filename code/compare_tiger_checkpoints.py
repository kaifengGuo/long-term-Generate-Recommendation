import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> str:
    text = str(path_str).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a clean, fixed-budget comparison across multiple TIGER checkpoints."
    )
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--episode_batch_size", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps_per_episode", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Evaluation seed. Repeat this flag for multi-seed comparison.",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Checkpoint spec in the form label=path. Repeat for each model to compare.",
    )
    return parser.parse_args()


def parse_ckpt_specs(specs: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for spec in specs:
        if "=" not in str(spec):
            raise ValueError(f"Invalid --ckpt spec: {spec}. Expected label=path")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --ckpt label in spec: {spec}")
        if label in seen:
            raise ValueError(f"Duplicate checkpoint label: {label}")
        seen.add(label)
        pairs.append((label, resolve_path(path)))
    return pairs


def parse_eval_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    patterns = {
        "total_reward": r"Total Reward:\s*([0-9.\-]+)",
        "depth": r"Depth:\s*([0-9.\-]+)",
        "avg_reward": r"Average reward:\s*([0-9.\-]+)",
        "coverage": r"Coverage:\s*([0-9.\-]+)",
        "ild": r"ILD:\s*([0-9.\-]+)",
        "click": r"is_click:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "long_view": r"long_view:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "like": r"is_like:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "comment": r"is_comment:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "forward": r"is_forward:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "follow": r"is_follow:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "hate": r"is_hate:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def run_command(cmd: List[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(CODE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")
    return output


def aggregate_metric(metric_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for item in metric_list for k in item.keys()})
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [float(item[key]) for item in metric_list if key in item]
        if not vals:
            continue
        out[key] = {
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return out


def metric_delta(after: Dict[str, Dict[str, float]], before: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    out: Dict[str, float] = {}
    for key in keys:
        if key in before and key in after:
            out[key] = float(after[key]["mean"] - before[key]["mean"])
    return out


def main() -> int:
    args = parse_args()
    python = sys.executable
    seeds = args.seeds or [2026]
    ckpts = parse_ckpt_specs(args.ckpt)
    output_dir = Path(resolve_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.json"

    summary: Dict[str, object] = {
        "config": {
            "uirm_log_path": resolve_path(args.uirm_log_path),
            "sid_mapping_path": resolve_path(args.sid_mapping_path),
            "model_size": str(args.model_size),
            "device": str(args.device),
            "slate_size": int(args.slate_size),
            "episode_batch_size": int(args.episode_batch_size),
            "num_episodes": int(args.num_episodes),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "beam_width": int(args.beam_width),
            "max_hist_items": int(args.max_hist_items),
            "seeds": [int(s) for s in seeds],
        },
        "models": [],
    }

    for label, ckpt_path in ckpts:
        seed_runs: List[Dict[str, object]] = []
        print(f"[compare] evaluating {label}: {ckpt_path}", flush=True)
        for seed in seeds:
            log_path = output_dir / label / f"seed_{seed}.log"
            cmd = [
                python,
                str(CODE_DIR / "eval_tiger_env.py"),
                "--tiger_ckpt", str(ckpt_path),
                "--sid_mapping_path", str(resolve_path(args.sid_mapping_path)),
                "--uirm_log_path", str(resolve_path(args.uirm_log_path)),
                "--slate_size", str(args.slate_size),
                "--episode_batch_size", str(args.episode_batch_size),
                "--model_size", str(args.model_size),
                "--num_episodes", str(args.num_episodes),
                "--max_steps_per_episode", str(args.max_steps_per_episode),
                "--max_step_per_episode", str(args.max_steps_per_episode),
                "--beam_width", str(args.beam_width),
                "--single_response",
                "--seed", str(seed),
                "--max_hist_items", str(args.max_hist_items),
                "--device", str(args.device),
            ]
            output = run_command(cmd, log_path)
            metrics = parse_eval_metrics(output)
            seed_runs.append(
                {
                    "seed": int(seed),
                    "log_path": str(log_path.resolve()),
                    "metrics": metrics,
                }
            )
            print(
                f"[compare] {label} seed={seed} "
                f"total_reward={metrics.get('total_reward', float('nan')):.4f} "
                f"click={metrics.get('click', float('nan')):.4f} "
                f"long_view={metrics.get('long_view', float('nan')):.4f}",
                flush=True,
            )

        agg = aggregate_metric([item["metrics"] for item in seed_runs])
        summary["models"].append(
            {
                "label": label,
                "tiger_ckpt": ckpt_path,
                "seed_runs": seed_runs,
                "aggregate": agg,
            }
        )

        models = summary["models"]
        if models:
            baseline = models[0]
            baseline_agg = baseline["aggregate"]
            for idx in range(1, len(models)):
                models[idx]["delta_vs_first"] = metric_delta(models[idx]["aggregate"], baseline_agg)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    models = summary["models"]
    if models:
        baseline = models[0]
        baseline_agg = baseline["aggregate"]
        for idx in range(1, len(models)):
            models[idx]["delta_vs_first"] = metric_delta(models[idx]["aggregate"], baseline_agg)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] wrote summary to {summary_path}", flush=True)

    print("=" * 72, flush=True)
    for model in summary["models"]:
        label = model["label"]
        agg = model["aggregate"]
        total_reward = agg.get("total_reward", {}).get("mean", float("nan"))
        click = agg.get("click", {}).get("mean", float("nan"))
        long_view = agg.get("long_view", {}).get("mean", float("nan"))
        print(
            f"{label:>12s} | total_reward={total_reward:.4f} | "
            f"click={click:.4f} | long_view={long_view:.4f}",
            flush=True,
        )
        if "delta_vs_first" in model:
            delta = model["delta_vs_first"]
            print(
                f"{'':>12s} | delta_reward={delta.get('total_reward', float('nan')):+.4f} | "
                f"delta_click={delta.get('click', float('nan')):+.4f} | "
                f"delta_long_view={delta.get('long_view', float('nan')):+.4f}",
                flush=True,
            )
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
