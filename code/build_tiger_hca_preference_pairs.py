import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from tiger_phase2_blend_common import write_json
from tiger_page_sid_rl.common import iter_jsonl_records

SCORE_FIELD_CHOICES = [
    "reward_raw",
    "reward_model_value",
    "adaptive_support_pess",
    "page_q_value",
    "page_q_mean",
    "page_q_pess",
    "item_advantage",
    "item_advantage_mean",
    "item_advantage_pess",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build support-aware preference/search-distillation pairs from HCA grouped candidates.")
    parser.add_argument("--group_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default="")
    parser.add_argument(
        "--score_field",
        type=str,
        default="adaptive_support_pess",
        choices=SCORE_FIELD_CHOICES,
    )
    parser.add_argument("--safe_support_gap_max", type=float, default=0.25)
    parser.add_argument("--min_score_gap", type=float, default=0.0)
    parser.add_argument("--max_pairs_per_group", type=int, default=2)
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="mixed",
        choices=["safe_vs_behavior", "safe_vs_unsafe", "mixed", "search_distill"],
    )
    parser.add_argument("--exploit_score_field", type=str, default="page_q_mean", choices=SCORE_FIELD_CHOICES)
    parser.add_argument("--min_support_gap_delta", type=float, default=0.05)
    parser.add_argument("--min_unc_delta", type=float, default=0.0)
    parser.add_argument("--pair_score_gap_scale", type=float, default=1.0)
    parser.add_argument("--pair_raw_q_gap_scale", type=float, default=0.5)
    parser.add_argument("--pair_unc_gap_scale", type=float, default=0.25)
    parser.add_argument("--pair_support_gap_scale", type=float, default=0.5)
    return parser.parse_args()


def load_group_rows(group_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in iter_jsonl_records(group_path):
        group_id = str(payload.get("group_id", payload.get("episode_id", "na")))
        groups[group_id].append(payload)
    if not groups:
        raise ValueError(f"No usable grouped candidates in {group_path}")
    return groups


def float_field(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(record.get(str(key), default))


def score_record(record: Dict[str, Any], score_field: str) -> float:
    default = float_field(record, "reward_model_value", float_field(record, "reward_raw", 0.0))
    return float_field(record, score_field, default)


def record_sort_key(record: Dict[str, Any], score_field: str) -> tuple[float, float, float]:
    return (
        score_record(record, score_field),
        -float_field(record, "support_gap_scaled", 0.0),
        float_field(record, "support_logprob_mean", 0.0),
    )


def pair_gap_metrics(
    *,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    score_field: str,
    pair_score_gap_scale: float,
    pair_raw_q_gap_scale: float,
    pair_unc_gap_scale: float,
    pair_support_gap_scale: float,
) -> Dict[str, float]:
    score_gap = float(score_record(chosen, score_field) - score_record(rejected, score_field))
    raw_q_gap = max(
        0.0,
        float_field(rejected, "page_q_mean", float_field(rejected, "page_q_value", 0.0))
        - float_field(chosen, "page_q_mean", float_field(chosen, "page_q_value", 0.0)),
    )
    unc_gap = max(0.0, float_field(rejected, "page_q_std", 0.0) - float_field(chosen, "page_q_std", 0.0))
    support_gap_delta = max(
        0.0,
        float_field(rejected, "support_gap_scaled", 0.0) - float_field(chosen, "support_gap_scaled", 0.0),
    )
    pair_strength = (
        float(pair_score_gap_scale) * max(0.0, score_gap)
        + float(pair_raw_q_gap_scale) * raw_q_gap
        + float(pair_unc_gap_scale) * unc_gap
        + float(pair_support_gap_scale) * support_gap_delta
    )
    return {
        "teacher_score_gap": float(score_gap),
        "teacher_raw_q_gap": float(raw_q_gap),
        "teacher_unc_gap": float(unc_gap),
        "teacher_support_gap_delta": float(support_gap_delta),
        "pair_strength": float(pair_strength),
    }


def build_pair_record(
    *,
    group_id: str,
    pair_type: str,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    score_field: str,
    gap_metrics: Dict[str, float],
) -> Dict[str, Any]:
    chosen_score = score_record(chosen, score_field)
    rejected_score = score_record(rejected, score_field)
    return {
        "pair_id": f"{group_id}:{pair_type}:{int(chosen.get('candidate_item_id', -1))}:{int(rejected.get('candidate_item_id', -1))}",
        "group": str(group_id),
        "pair_type": str(pair_type),
        "score_field": str(score_field),
        "input_ids": [int(x) for x in chosen.get("input_ids", [])],
        "attention_mask": [int(x) for x in chosen.get("attention_mask", [])],
        "chosen_tokens": [int(x) for x in chosen.get("target_tokens", [])],
        "rejected_tokens": [int(x) for x in rejected.get("target_tokens", [])],
        "chosen_item_id": int(chosen.get("candidate_item_id", -1)),
        "rejected_item_id": int(rejected.get("candidate_item_id", -1)),
        "chosen_source": str(chosen.get("candidate_source", "na")),
        "rejected_source": str(rejected.get("candidate_source", "na")),
        "chosen_is_behavior": bool(chosen.get("is_behavior", False)),
        "rejected_is_behavior": bool(rejected.get("is_behavior", False)),
        "chosen_rank": int(chosen.get("candidate_rank", -1)),
        "rejected_rank": int(rejected.get("candidate_rank", -1)),
        "chosen_score": float(chosen_score),
        "rejected_score": float(rejected_score),
        "reward_gap": float(gap_metrics["pair_strength"]),
        "teacher_score_gap": float(gap_metrics["teacher_score_gap"]),
        "teacher_raw_q_gap": float(gap_metrics["teacher_raw_q_gap"]),
        "teacher_unc_gap": float(gap_metrics["teacher_unc_gap"]),
        "teacher_support_gap_delta": float(gap_metrics["teacher_support_gap_delta"]),
        "chosen_support_gap_scaled": float_field(chosen, "support_gap_scaled", 0.0),
        "rejected_support_gap_scaled": float_field(rejected, "support_gap_scaled", 0.0),
        "chosen_support_logprob_mean": float_field(chosen, "support_logprob_mean", 0.0),
        "rejected_support_logprob_mean": float_field(rejected, "support_logprob_mean", 0.0),
        "chosen_page_q_mean": float_field(chosen, "page_q_mean", float_field(chosen, "page_q_value", 0.0)),
        "rejected_page_q_mean": float_field(rejected, "page_q_mean", float_field(rejected, "page_q_value", 0.0)),
        "chosen_page_q_std": float_field(chosen, "page_q_std", 0.0),
        "rejected_page_q_std": float_field(rejected, "page_q_std", 0.0),
        "chosen_item_advantage": float_field(chosen, "item_advantage", 0.0),
        "rejected_item_advantage": float_field(rejected, "item_advantage", 0.0),
        "chosen_item_advantage_pess": float_field(
            chosen,
            "item_advantage_pess",
            float_field(chosen, "item_advantage", 0.0),
        ),
        "rejected_item_advantage_pess": float_field(
            rejected,
            "item_advantage_pess",
            float_field(rejected, "item_advantage", 0.0),
        ),
        "chosen_sid_advantage": [float(x) for x in chosen.get("sid_advantage", [])],
        "rejected_sid_advantage": [float(x) for x in rejected.get("sid_advantage", [])],
        "chosen_sid_advantage_pess": [
            float(x) for x in chosen.get("sid_advantage_pess", chosen.get("sid_advantage", []))
        ],
        "rejected_sid_advantage_pess": [
            float(x) for x in rejected.get("sid_advantage_pess", rejected.get("sid_advantage", []))
        ],
    }


def maybe_add_pair(
    *,
    out_pairs: List[Dict[str, Any]],
    seen_keys: set[tuple[int, int, str]],
    group_id: str,
    pair_type: str,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    score_field: str,
    min_score_gap: float,
    pair_score_gap_scale: float,
    pair_raw_q_gap_scale: float,
    pair_unc_gap_scale: float,
    pair_support_gap_scale: float,
) -> bool:
    if int(chosen.get("candidate_item_id", -1)) == int(rejected.get("candidate_item_id", -1)):
        return False
    chosen_tokens = [int(x) for x in chosen.get("target_tokens", [])]
    rejected_tokens = [int(x) for x in rejected.get("target_tokens", [])]
    if not chosen_tokens or not rejected_tokens or len(chosen_tokens) != len(rejected_tokens):
        return False
    reward_gap = score_record(chosen, score_field) - score_record(rejected, score_field)
    if reward_gap < float(min_score_gap):
        return False
    key = (
        int(chosen.get("candidate_item_id", -1)),
        int(rejected.get("candidate_item_id", -1)),
        str(pair_type),
    )
    if key in seen_keys:
        return False
    seen_keys.add(key)
    gap_metrics = pair_gap_metrics(
        chosen=chosen,
        rejected=rejected,
        score_field=score_field,
        pair_score_gap_scale=float(pair_score_gap_scale),
        pair_raw_q_gap_scale=float(pair_raw_q_gap_scale),
        pair_unc_gap_scale=float(pair_unc_gap_scale),
        pair_support_gap_scale=float(pair_support_gap_scale),
    )
    out_pairs.append(
        build_pair_record(
            group_id=str(group_id),
            pair_type=str(pair_type),
            chosen=chosen,
            rejected=rejected,
            score_field=str(score_field),
            gap_metrics=gap_metrics,
        )
    )
    return True


def select_best(records: Sequence[Dict[str, Any]], score_field: str) -> Dict[str, Any]:
    return max(records, key=lambda record: record_sort_key(record, score_field))


def select_search_distill_exploit(
    *,
    rows: Sequence[Dict[str, Any]],
    chosen: Dict[str, Any],
    score_field: str,
    exploit_score_field: str,
    min_support_gap_delta: float,
    min_unc_delta: float,
) -> Dict[str, Any] | None:
    chosen_support_gap = float_field(chosen, "support_gap_scaled", 0.0)
    chosen_q_std = float_field(chosen, "page_q_std", 0.0)
    exploit_rows: List[Dict[str, Any]] = []
    for row in rows:
        if int(row.get("candidate_item_id", -1)) == int(chosen.get("candidate_item_id", -1)):
            continue
        if bool(row.get("is_behavior")):
            continue
        support_gap_delta = float_field(row, "support_gap_scaled", 0.0) - chosen_support_gap
        unc_gap = float_field(row, "page_q_std", 0.0) - chosen_q_std
        exploit_score_gap = score_record(row, exploit_score_field) - score_record(chosen, exploit_score_field)
        if (
            support_gap_delta >= float(min_support_gap_delta)
            or unc_gap >= float(min_unc_delta)
            or exploit_score_gap > 0.0
        ):
            exploit_rows.append(row)
    if not exploit_rows:
        return None
    return max(
        exploit_rows,
        key=lambda record: (
            score_record(record, exploit_score_field),
            float_field(record, "support_gap_scaled", 0.0),
            float_field(record, "page_q_std", 0.0),
            -score_record(record, score_field),
        ),
    )


def main() -> int:
    args = parse_args()
    group_path = Path(args.group_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = load_group_rows(group_path)
    pair_type_counts: Counter[str] = Counter()
    reward_gaps: List[float] = []
    teacher_score_gaps: List[float] = []
    teacher_raw_q_gaps: List[float] = []
    teacher_unc_gaps: List[float] = []
    support_gap_deltas: List[float] = []
    chosen_behavior = 0
    rejected_behavior = 0
    groups_with_pairs = 0
    n_input_rows = int(sum(len(rows) for rows in groups.values()))
    n_input_groups = int(len(groups))

    with output_path.open("w", encoding="utf-8") as out:
        for group_id, rows in groups.items():
            behavior = next((row for row in rows if bool(row.get("is_behavior"))), None)
            if behavior is None or len(rows) < 2:
                continue

            safe_rows = [
                row for row in rows
                if bool(row.get("is_behavior")) or float_field(row, "support_gap_scaled", 0.0) <= float(args.safe_support_gap_max)
            ]
            if not safe_rows:
                safe_rows = [behavior]
            unsafe_rows = [
                row for row in rows
                if not bool(row.get("is_behavior")) and float_field(row, "support_gap_scaled", 0.0) > float(args.safe_support_gap_max)
            ]
            safe_rows_sorted = sorted(safe_rows, key=lambda row: record_sort_key(row, str(args.score_field)), reverse=True)
            best_safe = safe_rows_sorted[0]

            group_pairs: List[Dict[str, Any]] = []
            seen_keys: set[tuple[int, int, str]] = set()

            if str(args.pair_mode) == "search_distill":
                overall_sorted = sorted(rows, key=lambda row: record_sort_key(row, str(args.score_field)), reverse=True)
                safe_runnerup = next(
                    (
                        row
                        for row in safe_rows_sorted[1:]
                        if int(row.get("candidate_item_id", -1)) != int(best_safe.get("candidate_item_id", -1))
                    ),
                    None,
                )
                overall_runnerup = next(
                    (
                        row
                        for row in overall_sorted
                        if int(row.get("candidate_item_id", -1)) != int(best_safe.get("candidate_item_id", -1))
                    ),
                    None,
                )
                exploit_row = select_search_distill_exploit(
                    rows=rows,
                    chosen=best_safe,
                    score_field=str(args.score_field),
                    exploit_score_field=str(args.exploit_score_field),
                    min_support_gap_delta=float(args.min_support_gap_delta),
                    min_unc_delta=float(args.min_unc_delta),
                )
                candidate_specs = [
                    ("distill_vs_behavior", behavior),
                    ("distill_vs_exploit", exploit_row),
                    ("distill_vs_safe_runnerup", safe_runnerup),
                    ("distill_vs_runnerup", overall_runnerup),
                ]
                for pair_type, rejected in candidate_specs:
                    if len(group_pairs) >= int(args.max_pairs_per_group):
                        break
                    if rejected is None:
                        continue
                    maybe_add_pair(
                        out_pairs=group_pairs,
                        seen_keys=seen_keys,
                        group_id=str(group_id),
                        pair_type=str(pair_type),
                        chosen=best_safe,
                        rejected=rejected,
                        score_field=str(args.score_field),
                        min_score_gap=float(args.min_score_gap),
                        pair_score_gap_scale=float(args.pair_score_gap_scale),
                        pair_raw_q_gap_scale=float(args.pair_raw_q_gap_scale),
                        pair_unc_gap_scale=float(args.pair_unc_gap_scale),
                        pair_support_gap_scale=float(args.pair_support_gap_scale),
                    )
            if str(args.pair_mode) in {"safe_vs_behavior", "mixed"}:
                maybe_add_pair(
                    out_pairs=group_pairs,
                    seen_keys=seen_keys,
                    group_id=str(group_id),
                    pair_type="safe_vs_behavior",
                    chosen=best_safe,
                    rejected=behavior,
                    score_field=str(args.score_field),
                    min_score_gap=float(args.min_score_gap),
                    pair_score_gap_scale=float(args.pair_score_gap_scale),
                    pair_raw_q_gap_scale=float(args.pair_raw_q_gap_scale),
                    pair_unc_gap_scale=float(args.pair_unc_gap_scale),
                    pair_support_gap_scale=float(args.pair_support_gap_scale),
                )

            if len(group_pairs) < int(args.max_pairs_per_group) and str(args.pair_mode) in {"safe_vs_unsafe", "mixed"} and unsafe_rows:
                hardest_unsafe = select_best(unsafe_rows, str(args.score_field))
                maybe_add_pair(
                    out_pairs=group_pairs,
                    seen_keys=seen_keys,
                    group_id=str(group_id),
                    pair_type="safe_vs_unsafe",
                    chosen=best_safe,
                    rejected=hardest_unsafe,
                    score_field=str(args.score_field),
                    min_score_gap=float(args.min_score_gap),
                    pair_score_gap_scale=float(args.pair_score_gap_scale),
                    pair_raw_q_gap_scale=float(args.pair_raw_q_gap_scale),
                    pair_unc_gap_scale=float(args.pair_unc_gap_scale),
                    pair_support_gap_scale=float(args.pair_support_gap_scale),
                )

            if len(group_pairs) < int(args.max_pairs_per_group) and str(args.pair_mode) == "mixed" and len(safe_rows_sorted) >= 2:
                maybe_add_pair(
                    out_pairs=group_pairs,
                    seen_keys=seen_keys,
                    group_id=str(group_id),
                    pair_type="safe_vs_runnerup",
                    chosen=safe_rows_sorted[0],
                    rejected=safe_rows_sorted[1],
                    score_field=str(args.score_field),
                    min_score_gap=float(args.min_score_gap),
                    pair_score_gap_scale=float(args.pair_score_gap_scale),
                    pair_raw_q_gap_scale=float(args.pair_raw_q_gap_scale),
                    pair_unc_gap_scale=float(args.pair_unc_gap_scale),
                    pair_support_gap_scale=float(args.pair_support_gap_scale),
                )

            if not group_pairs:
                continue
            groups_with_pairs += 1
            for pair in group_pairs[: int(args.max_pairs_per_group)]:
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                pair_type_counts[str(pair["pair_type"])] += 1
                reward_gaps.append(float(pair["reward_gap"]))
                teacher_score_gaps.append(float(pair.get("teacher_score_gap", 0.0)))
                teacher_raw_q_gaps.append(float(pair.get("teacher_raw_q_gap", 0.0)))
                teacher_unc_gaps.append(float(pair.get("teacher_unc_gap", 0.0)))
                support_gap_deltas.append(float(pair["rejected_support_gap_scaled"]) - float(pair["chosen_support_gap_scaled"]))
                chosen_behavior += int(bool(pair["chosen_is_behavior"]))
                rejected_behavior += int(bool(pair["rejected_is_behavior"]))

    n_pairs = int(sum(pair_type_counts.values()))
    reward_gap_arr = np.asarray(reward_gaps, dtype=np.float32) if reward_gaps else np.zeros(1, dtype=np.float32)
    teacher_score_gap_arr = np.asarray(teacher_score_gaps, dtype=np.float32) if teacher_score_gaps else np.zeros(1, dtype=np.float32)
    teacher_raw_q_gap_arr = np.asarray(teacher_raw_q_gaps, dtype=np.float32) if teacher_raw_q_gaps else np.zeros(1, dtype=np.float32)
    teacher_unc_gap_arr = np.asarray(teacher_unc_gaps, dtype=np.float32) if teacher_unc_gaps else np.zeros(1, dtype=np.float32)
    summary = {
        "method": "TIGER-HCA Search-Distill Pair Builder" if str(args.pair_mode) == "search_distill" else "TIGER-HCA Preference Pair Builder",
        "group_path": str(group_path.resolve()),
        "output_path": str(output_path.resolve()),
        "score_field": str(args.score_field),
        "exploit_score_field": str(args.exploit_score_field),
        "safe_support_gap_max": float(args.safe_support_gap_max),
        "min_score_gap": float(args.min_score_gap),
        "min_support_gap_delta": float(args.min_support_gap_delta),
        "min_unc_delta": float(args.min_unc_delta),
        "max_pairs_per_group": int(args.max_pairs_per_group),
        "pair_mode": str(args.pair_mode),
        "pair_score_gap_scale": float(args.pair_score_gap_scale),
        "pair_raw_q_gap_scale": float(args.pair_raw_q_gap_scale),
        "pair_unc_gap_scale": float(args.pair_unc_gap_scale),
        "pair_support_gap_scale": float(args.pair_support_gap_scale),
        "n_input_rows": int(n_input_rows),
        "n_input_groups": int(n_input_groups),
        "n_pairs": int(n_pairs),
        "groups_with_pairs": int(groups_with_pairs),
        "groups_with_pairs_frac": float(groups_with_pairs / max(n_input_groups, 1)),
        "pairs_per_group_mean": float(n_pairs / max(groups_with_pairs, 1)),
        "pair_type_counts": dict(pair_type_counts),
        "reward_gap_mean": float(reward_gap_arr.mean()),
        "reward_gap_std": float(reward_gap_arr.std()),
        "reward_gap_p50": float(np.quantile(reward_gap_arr, 0.5)),
        "reward_gap_p90": float(np.quantile(reward_gap_arr, 0.9)),
        "reward_gap_p99": float(np.quantile(reward_gap_arr, 0.99)),
        "teacher_score_gap_mean": float(teacher_score_gap_arr.mean()),
        "teacher_score_gap_std": float(teacher_score_gap_arr.std()),
        "teacher_raw_q_gap_mean": float(teacher_raw_q_gap_arr.mean()),
        "teacher_unc_gap_mean": float(teacher_unc_gap_arr.mean()),
        "support_gap_delta_mean": float(np.mean(support_gap_deltas)) if support_gap_deltas else 0.0,
        "chosen_behavior_frac": float(chosen_behavior / max(n_pairs, 1)),
        "rejected_behavior_frac": float(rejected_behavior / max(n_pairs, 1)),
    }
    summary_out = Path(args.summary_out) if str(args.summary_out).strip() else output_path.with_name("hca_pref_pair_summary.json")
    write_json(summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
