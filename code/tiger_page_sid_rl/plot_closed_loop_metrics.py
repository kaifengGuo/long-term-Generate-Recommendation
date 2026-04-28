import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_page_sid_rl.common import iter_jsonl_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot closed-loop metrics for TIGER page-SID training.")
    parser.add_argument("--summary_path", type=str, default="")
    parser.add_argument("--summary_jsonl_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_summary(summary_path: str, summary_jsonl_path: str) -> List[Dict[str, Any]]:
    if str(summary_path).strip():
        path = Path(summary_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    rows: List[Dict[str, Any]] = []
    if str(summary_jsonl_path).strip():
        path = Path(summary_jsonl_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    return rows


def get_nested(payload: Dict[str, Any], path: str, default: float = float("nan")) -> float:
    cur: Any = payload
    for key in str(path).split("."):
        if not isinstance(cur, dict) or key not in cur:
            return float(default)
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return float(default)


def get_first_nested(payload: Dict[str, Any], paths: List[str], default: float = float("nan")) -> float:
    for path in paths:
        value = get_nested(payload, path, default)
        if not np.isnan(value):
            return value
    return float(default)


def compute_group_diagnostics(group_path: Path) -> Dict[str, float]:
    if not group_path.exists():
        return {}

    behavior_source_rewards: List[float] = []
    beam_source_rewards: List[float] = []
    behavior_reward_raws: List[float] = []
    beam_reward_raws: List[float] = []
    best_source_margins: List[float] = []
    best_reward_margins: List[float] = []
    top_behavior_source = 0
    top_behavior_reward = 0
    n_groups = 0

    grouped_records: Dict[str, List[Dict[str, Any]]] = {}

    def finalize_group(records: List[Dict[str, Any]]) -> None:
        nonlocal top_behavior_source, top_behavior_reward, n_groups
        if not records:
            return
        behavior_record = next((record for record in records if bool(record.get("is_behavior"))), None)
        if behavior_record is None:
            return

        behavior_source_value = safe_float(
            behavior_record.get("reward_model_value", behavior_record.get("reward_raw", float("nan")))
        )
        behavior_reward_value = safe_float(behavior_record.get("reward_raw", float("nan")))
        best_source_record = max(
            records,
            key=lambda item: safe_float(item.get("reward_model_value", item.get("reward_raw", 0.0)), 0.0),
        )
        best_reward_record = max(records, key=lambda item: safe_float(item.get("reward_raw", 0.0), 0.0))
        top_behavior_source += int(bool(best_source_record.get("is_behavior")))
        top_behavior_reward += int(bool(best_reward_record.get("is_behavior")))
        best_source_margins.append(
            safe_float(best_source_record.get("reward_model_value", best_source_record.get("reward_raw", 0.0)), 0.0)
            - behavior_source_value
        )
        best_reward_margins.append(safe_float(best_reward_record.get("reward_raw", 0.0), 0.0) - behavior_reward_value)

        for record in records:
            source_value = safe_float(record.get("reward_model_value", record.get("reward_raw", 0.0)), 0.0)
            reward_raw = safe_float(record.get("reward_raw", 0.0), 0.0)
            if bool(record.get("is_behavior")):
                behavior_source_rewards.append(source_value)
                behavior_reward_raws.append(reward_raw)
            else:
                beam_source_rewards.append(source_value)
                beam_reward_raws.append(reward_raw)
        n_groups += 1

    for record in iter_jsonl_records(group_path):
        group_id = str(record.get("group_id", ""))
        grouped_records.setdefault(group_id, []).append(record)
    for records in grouped_records.values():
        finalize_group(records)

    beam_source_mean = float(np.mean(beam_source_rewards)) if beam_source_rewards else 0.0
    behavior_source_mean = float(np.mean(behavior_source_rewards)) if behavior_source_rewards else 0.0
    beam_reward_mean = float(np.mean(beam_reward_raws)) if beam_reward_raws else 0.0
    behavior_reward_mean = float(np.mean(behavior_reward_raws)) if behavior_reward_raws else 0.0
    top_behavior_source_frac = float(top_behavior_source / max(n_groups, 1))
    top_behavior_reward_frac = float(top_behavior_reward / max(n_groups, 1))
    return {
        "behavior_source_reward_mean": behavior_source_mean,
        "beam_source_reward_mean": beam_source_mean,
        "behavior_reward_mean": behavior_reward_mean,
        "beam_reward_mean": beam_reward_mean,
        "beam_source_minus_behavior": float(beam_source_mean - behavior_source_mean),
        "beam_reward_minus_behavior": float(beam_reward_mean - behavior_reward_mean),
        "avg_best_source_minus_behavior": float(np.mean(best_source_margins)) if best_source_margins else 0.0,
        "median_best_source_minus_behavior": float(np.median(best_source_margins)) if best_source_margins else 0.0,
        "avg_best_reward_minus_behavior": float(np.mean(best_reward_margins)) if best_reward_margins else 0.0,
        "median_best_reward_minus_behavior": float(np.median(best_reward_margins)) if best_reward_margins else 0.0,
        "pos_source_margin_frac": float(np.mean(np.asarray(best_source_margins) > 0.0)) if best_source_margins else 0.0,
        "pos_reward_margin_frac": float(np.mean(np.asarray(best_reward_margins) > 0.0)) if best_reward_margins else 0.0,
        "top_behavior_source_frac": top_behavior_source_frac,
        "top_beam_source_frac": float(1.0 - top_behavior_source_frac) if n_groups > 0 else 0.0,
        "top_behavior_reward_frac": top_behavior_reward_frac,
        "top_beam_reward_frac": float(1.0 - top_behavior_reward_frac) if n_groups > 0 else 0.0,
    }


def enrich_actor_group_summary(rows: List[Dict[str, Any]]) -> None:
    required_keys = {
        "beam_source_minus_behavior",
        "avg_best_source_minus_behavior",
        "top_behavior_source_frac",
        "top_beam_source_frac",
    }
    for row in rows:
        actor_group_summary = row.get("actor_group_summary")
        if not isinstance(actor_group_summary, dict):
            continue
        if required_keys.issubset(actor_group_summary.keys()):
            continue
        output_path = str(actor_group_summary.get("output_path", "")).strip()
        if not output_path:
            continue
        diagnostics = compute_group_diagnostics(Path(output_path))
        if diagnostics:
            actor_group_summary.update(diagnostics)


def series(rows: List[Dict[str, Any]], path: str) -> List[float]:
    return [get_nested(row, path) for row in rows]


def series_first(rows: List[Dict[str, Any]], paths: List[str]) -> List[float]:
    return [get_first_nested(row, paths) for row in rows]


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    field_specs = {
        "iter": "iter",
        "did_eval": "did_eval",
        "before_total_reward": "before_eval.total_reward",
        "rollout_total_reward": "rollout_eval.total_reward",
        "after_total_reward": "after_eval.total_reward",
        "after_rollout_total_reward": ["after_eval_rollout.total_reward", "after_eval.total_reward"],
        "after_learner_total_reward": ["after_eval_learner.total_reward", "after_eval.total_reward"],
        "delta_total_reward": "eval_delta.total_reward",
        "delta_rollout_total_reward": ["eval_delta_rollout.total_reward", "eval_delta.total_reward"],
        "delta_learner_total_reward": ["eval_delta_learner.total_reward", "eval_delta.total_reward"],
        "critic_best_valid_loss": "critic_metrics.best_valid_loss",
        "critic_valid_q_mae": "critic_metrics.history.0.valid.q_mae",
        "critic_valid_q_corr": "critic_metrics.history.0.valid.q_corr",
        "chain_page_q_abs_mean": "chain_summary.page_q_abs_mean",
        "chain_item_adv_abs_mean": "chain_summary.item_adv_abs_mean",
        "chain_sid_adv_abs_mean": "chain_summary.sid_adv_abs_mean",
        "chain_sid_item_cons_mae": "chain_summary.sid_item_cons_mae",
        "actor_loss": "actor_metrics.best_metrics.loss",
        "actor_target_gain": "actor_metrics.best_metrics.target_gain",
        "actor_approx_kl": "actor_metrics.best_metrics.approx_kl",
        "actor_clip_frac": "actor_metrics.best_metrics.clip_frac",
        "actor_signed_adv_abs": "actor_metrics.best_metrics.signed_adv_abs",
        "actor_group_beam_source_minus_behavior": "actor_group_summary.beam_source_minus_behavior",
        "actor_group_avg_best_source_minus_behavior": "actor_group_summary.avg_best_source_minus_behavior",
        "actor_group_top_behavior_source_frac": "actor_group_summary.top_behavior_source_frac",
        "actor_group_top_beam_source_frac": "actor_group_summary.top_beam_source_frac",
        "actor_group_beam_reward_minus_behavior": "actor_group_summary.beam_reward_minus_behavior",
        "actor_group_avg_best_reward_minus_behavior": "actor_group_summary.avg_best_reward_minus_behavior",
        "replay_lines_added": "replay_lines_added",
        "replay_lines_total": "replay_lines_total",
        "before_eval_time_sec": "timing.before_eval_time_sec",
        "rollout_time_sec": "timing.rollout_time_sec",
        "critic_time_sec": "timing.critic_time_sec",
        "chain_time_sec": "timing.chain_time_sec",
        "actor_time_sec": "timing.actor_time_sec",
        "after_eval_time_sec": "timing.after_eval_time_sec",
        "after_eval_rollout_time_sec": ["timing.after_eval_rollout_time_sec", "timing.after_eval_time_sec"],
        "after_eval_learner_time_sec": ["timing.after_eval_learner_time_sec", "timing.after_eval_time_sec"],
        "iter_wall_time_sec": "timing.iter_wall_time_sec",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(field_specs.keys()))
        writer.writeheader()
        for row in rows:
            item: Dict[str, Any] = {}
            for out_key, path in field_specs.items():
                if path == "iter":
                    item[out_key] = int(row.get("iter", 0))
                elif path == "did_eval":
                    item[out_key] = int(bool(row.get("did_eval", False)))
                elif path.startswith("critic_metrics.history.0"):
                    hist = row.get("critic_metrics", {}).get("history", [])
                    if hist:
                        if path.endswith("valid.q_mae"):
                            item[out_key] = float(hist[0].get("valid", {}).get("q_mae", float("nan")))
                        elif path.endswith("valid.q_corr"):
                            item[out_key] = float(hist[0].get("valid", {}).get("q_corr", float("nan")))
                        else:
                            item[out_key] = float("nan")
                    else:
                        item[out_key] = float("nan")
                elif isinstance(path, list):
                    item[out_key] = get_first_nested(row, path)
                else:
                    item[out_key] = get_nested(row, path)
            writer.writerow(item)


def make_plot(
    x: List[int],
    ys: List[List[float]],
    labels: List[str],
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.5))
    for values, label in zip(ys, labels):
        arr = np.asarray(values, dtype=np.float64)
        plt.plot(x, arr, marker="o", linewidth=1.8, markersize=4, label=label)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=int(dpi))
    plt.close()


def main() -> int:
    args = parse_args()
    rows = load_summary(str(args.summary_path), str(args.summary_jsonl_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No summary rows found to plot.")

    rows = sorted(rows, key=lambda x: int(x.get("iter", 0)))
    enrich_actor_group_summary(rows)
    x = [int(row.get("iter", 0)) for row in rows]

    save_csv(rows, output_dir / "closed_loop_summary.csv")

    make_plot(
        x,
        [
            series(rows, "before_eval.total_reward"),
            series(rows, "rollout_eval.total_reward"),
            series_first(rows, ["after_eval_learner.total_reward", "after_eval.total_reward"]),
            series_first(rows, ["after_eval_rollout.total_reward", "after_eval.total_reward"]),
            series_first(rows, ["eval_delta_learner.total_reward", "eval_delta.total_reward"]),
        ],
        [
            "before total_reward",
            "rollout total_reward",
            "after learner total_reward",
            "after rollout total_reward",
            "delta learner total_reward",
        ],
        "Environment Reward",
        "Reward",
        output_dir / "env_total_reward.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "critic_metrics.best_valid_loss"),
            series(rows, "chain_summary.page_q_abs_mean"),
        ],
        ["critic best_valid_loss", "chain page_q_abs_mean"],
        "Page Critic",
        "Value",
        output_dir / "critic_overview.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "chain_summary.item_adv_abs_mean"),
            series(rows, "chain_summary.sid_adv_abs_mean"),
            series(rows, "chain_summary.sid_item_cons_mae"),
        ],
        ["item_adv_abs_mean", "sid_adv_abs_mean", "sid_item_cons_mae"],
        "Chain Advantage",
        "Value",
        output_dir / "chain_advantage.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "actor_metrics.best_metrics.loss"),
            series(rows, "actor_metrics.best_metrics.target_gain"),
            series(rows, "actor_metrics.best_metrics.approx_kl"),
            series(rows, "actor_metrics.best_metrics.clip_frac"),
        ],
        ["actor loss", "target_gain", "approx_kl", "clip_frac"],
        "Actor Metrics",
        "Value",
        output_dir / "actor_metrics.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "actor_group_summary.beam_source_minus_behavior"),
            series(rows, "actor_group_summary.avg_best_source_minus_behavior"),
            series_first(rows, ["eval_delta_learner.total_reward", "eval_delta.total_reward"]),
        ],
        ["beam_source_minus_behavior", "avg_best_source_minus_behavior", "delta_learner_total_reward"],
        "Beam Vs Behavior Source Margin",
        "Margin",
        output_dir / "beam_behavior_source.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "actor_group_summary.top_behavior_source_frac"),
            series(rows, "actor_group_summary.top_beam_source_frac"),
            series(rows, "actor_group_summary.top_behavior_reward_frac"),
            series(rows, "actor_group_summary.top_beam_reward_frac"),
        ],
        [
            "top_behavior_source_frac",
            "top_beam_source_frac",
            "top_behavior_reward_frac",
            "top_beam_reward_frac",
        ],
        "Beam Vs Behavior Preference",
        "Fraction",
        output_dir / "beam_behavior_preference.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "replay_lines_added"),
            series(rows, "replay_lines_total"),
        ],
        ["replay_lines_added", "replay_lines_total"],
        "Replay Growth",
        "Lines",
        output_dir / "replay_growth.png",
        int(args.dpi),
    )
    make_plot(
        x,
        [
            series(rows, "timing.rollout_time_sec"),
            series(rows, "timing.critic_time_sec"),
            series(rows, "timing.chain_time_sec"),
            series(rows, "timing.actor_time_sec"),
            series(rows, "timing.iter_wall_time_sec"),
        ],
        ["rollout_time_sec", "critic_time_sec", "chain_time_sec", "actor_time_sec", "iter_wall_time_sec"],
        "Timing",
        "Seconds",
        output_dir / "timing.png",
        int(args.dpi),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
