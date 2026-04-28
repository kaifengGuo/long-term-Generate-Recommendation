import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple runs on after-eval reward.")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Each item format: label=summary_json_path")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--title", type=str, default="After Eval Reward Comparison")
    parser.add_argument(
        "--metric",
        type=str,
        choices=("learner", "rollout", "legacy"),
        default="learner",
        help="Which post-train checkpoint series to compare.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_run_arg(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Run spec must be label=path, got: {text}")
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Summary at {path} is not a list.")
    return sorted(data, key=lambda row: int(row.get("iter", 0)))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def pick_after_reward(row: Dict[str, Any], metric: str) -> float:
    if metric == "learner":
        value = row.get("after_eval_learner", {}).get("total_reward", None)
        if value is None:
            value = row.get("after_eval", {}).get("total_reward", float("nan"))
        return safe_float(value)
    if metric == "rollout":
        value = row.get("after_eval_rollout", {}).get("total_reward", None)
        if value is None:
            value = row.get("after_eval", {}).get("total_reward", float("nan"))
        return safe_float(value)
    return safe_float(row.get("after_eval", {}).get("total_reward", float("nan")))


def main() -> int:
    args = parse_args()
    run_specs = [parse_run_arg(item) for item in args.runs]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(16, 8))
    global_max_iter = 0

    for idx, (label, path) in enumerate(run_specs):
        rows = load_rows(path)
        x = [int(row.get("iter", 0)) for row in rows]
        y = [pick_after_reward(row, str(args.metric)) for row in rows]
        color = colors[idx % len(colors)]
        ax.plot(x, y, marker="o", linewidth=2.2, markersize=4.5, label=label, color=color)
        if x:
            global_max_iter = max(global_max_iter, max(x))

    for iter_idx in range(1, global_max_iter + 1):
        if iter_idx > 1:
            ax.axvline(iter_idx - 0.5, color="#bbbbbb", linewidth=1.0, alpha=0.35)
        ax.text(iter_idx, 1.01, f"iter {iter_idx}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

    ax.set_title(str(args.title))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("After total reward")
    ax.grid(True, alpha=0.2)
    if global_max_iter > 0:
        ax.set_xlim(1, global_max_iter)
        ax.set_xticks(list(range(1, global_max_iter + 1)))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(args.dpi))
    plt.close(fig)
    print(str(output_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
