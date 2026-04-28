import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dual-channel Phase7b chain with localized page-item credit."
    )
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--token_credit_field", type=str, default="token_credit_calibrated")
    parser.add_argument("--item_credit_field", type=str, default="item_credit")
    parser.add_argument("--share_field", type=str, default="selected_item_credit_shares")
    parser.add_argument("--support_field", type=str, default="base_support")
    parser.add_argument("--history_pos_ratio_field", type=str, default="history_pos_ratio")
    parser.add_argument("--history_neg_ratio_field", type=str, default="history_neg_ratio")
    parser.add_argument("--page_localize_mode", type=str, default="share", choices=["share", "uniform"])
    parser.add_argument("--page_credit_scale", type=float, default=1.0)
    return parser.parse_args()


def safe_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for x in values:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)
    return out


def get_item_share(payload: Dict[str, Any], share_field: str, mode: str) -> float:
    slate_size = max(int(payload.get("slate_size", 0)), 1)
    idx = int(payload.get("slate_item_index", 0))
    if str(mode) == "uniform":
        return 1.0 / float(slate_size)
    shares = safe_float_list(payload.get(str(share_field), []))
    if 0 <= idx < len(shares):
        share = float(shares[idx])
        if np.isfinite(share) and share >= 0.0:
            return share
    return 1.0 / float(slate_size)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0.0,
            "mean": 0.0,
            "abs_mean": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "abs_mean": float(np.abs(arr).mean()),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def main() -> int:
    args = parse_args()
    chain_path = Path(args.chain_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_pages = 0
    page_seen = set()
    page_item_credit_vals: List[float] = []
    decoder_gate_vals: List[float] = []
    veto_gate_vals: List[float] = []
    local_pos_vals: List[float] = []
    local_neg_vals: List[float] = []

    with chain_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            page_key = (int(payload.get("episode_id", -1)), int(payload.get("page_index", -1)))
            if page_key not in page_seen:
                page_seen.add(page_key)
                n_pages += 1

            share = float(get_item_share(payload, str(args.share_field), str(args.page_localize_mode)))
            slate_credit = float(payload.get("slate_credit", 0.0))
            item_credit = float(payload.get(str(args.item_credit_field), 0.0))
            support = float(payload.get(str(args.support_field), 0.0))
            history_pos_ratio = float(payload.get(str(args.history_pos_ratio_field), 0.0))
            history_neg_ratio = float(payload.get(str(args.history_neg_ratio_field), 0.0))
            token_credit = safe_float_list(payload.get(str(args.token_credit_field), []))

            page_item_credit = float(args.page_credit_scale) * slate_credit * share
            token_pos_credit = [max(float(x), 0.0) for x in token_credit]
            token_neg_credit = [max(-float(x), 0.0) for x in token_credit]
            item_pos_credit = max(item_credit, 0.0)
            item_neg_credit = max(-item_credit, 0.0)
            page_item_pos_credit = max(page_item_credit, 0.0)
            page_item_neg_credit = max(-page_item_credit, 0.0)

            decoder_gate = float(
                np.clip(
                    support * (1.0 + history_pos_ratio) * (1.0 + 0.5 * page_item_pos_credit),
                    0.0,
                    5.0,
                )
            )
            veto_gate = float(
                np.clip(
                    support * (1.0 + history_neg_ratio) * (1.0 + page_item_neg_credit),
                    0.0,
                    5.0,
                )
            )
            local_pos_mass = float(sum(token_pos_credit) + item_pos_credit + page_item_pos_credit)
            local_neg_mass = float(sum(token_neg_credit) + item_neg_credit + page_item_neg_credit)

            payload["phase7b_page_item_share"] = float(share)
            payload["phase7b_page_item_credit"] = float(page_item_credit)
            payload["phase7b_page_item_pos_credit"] = float(page_item_pos_credit)
            payload["phase7b_page_item_neg_credit"] = float(page_item_neg_credit)
            payload["phase7b_token_pos_credit"] = token_pos_credit
            payload["phase7b_token_neg_credit"] = token_neg_credit
            payload["phase7b_item_pos_credit"] = float(item_pos_credit)
            payload["phase7b_item_neg_credit"] = float(item_neg_credit)
            payload["phase7b_decoder_gate"] = float(decoder_gate)
            payload["phase7b_veto_gate"] = float(veto_gate)
            payload["phase7b_local_pos_mass"] = float(local_pos_mass)
            payload["phase7b_local_neg_mass"] = float(local_neg_mass)
            payload["phase7b_credit_version"] = "dual_channel_v1"

            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n_rows += 1
            page_item_credit_vals.append(float(page_item_credit))
            decoder_gate_vals.append(float(decoder_gate))
            veto_gate_vals.append(float(veto_gate))
            local_pos_vals.append(float(local_pos_mass))
            local_neg_vals.append(float(local_neg_mass))

    meta = {
        "method": "TIGER Phase7b Dual Channel Chain",
        "chain_path": str(chain_path.resolve()),
        "output_path": str(output_path.resolve()),
        "token_credit_field": str(args.token_credit_field),
        "item_credit_field": str(args.item_credit_field),
        "share_field": str(args.share_field),
        "page_localize_mode": str(args.page_localize_mode),
        "page_credit_scale": float(args.page_credit_scale),
        "n_rows": int(n_rows),
        "n_pages": int(n_pages),
        "page_item_credit_stats": summarize(page_item_credit_vals),
        "decoder_gate_stats": summarize(decoder_gate_vals),
        "veto_gate_stats": summarize(veto_gate_vals),
        "local_pos_mass_stats": summarize(local_pos_vals),
        "local_neg_mass_stats": summarize(local_neg_vals),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase7b-chain] saved dual-channel chain to {output_path}")
    print(f"[phase7b-chain] meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
