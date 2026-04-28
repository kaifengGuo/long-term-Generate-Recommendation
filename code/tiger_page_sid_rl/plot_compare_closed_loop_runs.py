import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple closed-loop runs on critic train/eval loss.")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Each item format: label=summary_json_path")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--title", type=str, default="Critic Loss Comparison")
    parser.add_argument(
        "--series",
        type=str,
        choices=("train", "eval", "both"),
        default="both",
        help="Which critic loss series to render.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_run(run_path: Path) -> List[Dict[str, Any]]:
    path = run_path
    if path.is_dir():
        summary_path = path / "closed_loop_summary.json"
        if summary_path.exists():
            path = summary_path
        else:
            rows: List[Dict[str, Any]] = []
            for iter_dir in sorted(path.glob("iter_*")):
                try:
                    iter_idx = int(iter_dir.name.split("_")[-1])
                except Exception:
                    continue
                metrics_path = iter_dir / "page_qcritic" / "page_sid_qcritic_metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "iter": iter_idx,
                        "critic_metrics": metrics,
                    }
                )
            return sorted(rows, key=lambda row: int(row.get("iter", 0)))

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Summary at {path} is not a list.")
    return sorted(data, key=lambda row: int(row.get("iter", 0)))


def flatten_critic_history(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[float], List[Tuple[float, int]]]:
    xs: List[int] = []
    train_losses: List[float] = []
    valid_losses: List[float] = []
    iter_centers: List[Tuple[float, int]] = []
    cursor = 0
    for row in rows:
        iter_idx = int(row.get("iter", 0))
        history = row.get("critic_metrics", {}).get("history", [])
        start = cursor + 1
        for epoch_record in history:
            cursor += 1
            xs.append(cursor)
            train_losses.append(float(epoch_record.get("train", {}).get("loss", float("nan"))))
            valid_losses.append(float(epoch_record.get("valid", {}).get("loss", float("nan"))))
        end = cursor
        if end >= start:
            iter_centers.append(((start + end) / 2.0, iter_idx))
    return xs, train_losses, valid_losses, iter_centers


def parse_run_arg(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Run spec must be label=path, got: {text}")
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def main() -> int:
    args = parse_args()
    run_specs = [parse_run_arg(item) for item in args.runs]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(16, 8))

    top_centers: List[Tuple[float, int]] = []
    max_x = 0
    for idx, (label, path) in enumerate(run_specs):
        rows = load_run(path)
        xs, train_losses, valid_losses, iter_centers = flatten_critic_history(rows)
        if not xs:
            continue
        color = colors[idx % len(colors)]
        if args.series in ("train", "both"):
            ax.plot(
                xs,
                train_losses,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=f"{label} train critic loss",
            )
        if args.series in ("eval", "both"):
            ax.plot(
                xs,
                valid_losses,
                color=color,
                linewidth=2.0,
                linestyle="--",
                marker="o",
                markersize=3,
                alpha=0.65,
                label=f"{label} eval critic loss",
            )
        max_x = max(max_x, max(xs))
        if len(iter_centers) > len(top_centers):
            top_centers = iter_centers

    for center, iter_idx in top_centers:
        ax.axvline(center + 2.0, color="#bbbbbb", linewidth=1.0, alpha=0.35)
        ax.text(center, 1.01, f"iter {iter_idx}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

    fig.suptitle(str(args.title), y=0.985)
    ax.set_xlabel("Critic epoch index (starts at 1)")
    ax.set_ylabel("Critic loss")
    ax.grid(True, alpha=0.2)
    if max_x > 0:
        ax.set_xlim(1, max_x)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(args.dpi))
    plt.close(fig)
    print(str(output_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
