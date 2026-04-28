import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log
from tiger_hier_prefix_common import load_item_prefix_head, load_page_prefix_head
from tiger_phase2_blend_common import (
    build_history_tokens,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_prefix_value_head,
    load_tiger_model,
)
from tiger_slate_online_common import build_online_slate_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified page/item/token prefix-difference advantage chain for TIGER.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--page_head_path", type=str, required=True)
    parser.add_argument("--page_meta_path", type=str, required=True)
    parser.add_argument("--item_head_path", type=str, required=True)
    parser.add_argument("--item_meta_path", type=str, required=True)
    parser.add_argument("--token_head_path", type=str, required=True)
    parser.add_argument("--token_meta_path", type=str, required=True)
    parser.add_argument("--legacy_token_field", type=str, default="token_credit_calibrated")
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def load_trace_rows(trace_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "selected_item_ids" not in payload:
                continue
            key = f"{int(payload['episode_id'])}:{int(payload.get('page_index', 0))}"
            rows[key] = payload
    if not rows:
        raise ValueError(f"No usable trace rows in {trace_path}")
    return rows


def load_chain_groups(chain_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "episode_id" not in payload or "page_index" not in payload:
                continue
            key = f"{int(payload['episode_id'])}:{int(payload['page_index'])}"
            grouped.setdefault(key, []).append(payload)
    if not grouped:
        raise ValueError(f"No usable chain rows in {chain_path}")
    return grouped


def pooled_history_summary(tiger, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    enc_out = tiger.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = enc_out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


def build_history_state(
    history_items: Sequence[int],
    *,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    sid_depth: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hist_tensor = torch.tensor([int(x) for x in history_items][-int(max_hist_items):], dtype=torch.long).view(1, -1)
    input_ids, attention_mask = build_history_tokens(
        hist_tensor,
        iid2sid_tok_cpu,
        int(max_hist_items),
        int(sid_depth),
    )
    return input_ids.to(device), attention_mask.to(device)


def eval_page_prefix_value(
    tiger,
    page_head,
    history_items: Sequence[int],
    *,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    sid_depth: int,
    device: torch.device,
) -> float:
    input_ids, attention_mask = build_history_state(
        history_items,
        iid2sid_tok_cpu=iid2sid_tok_cpu,
        max_hist_items=max_hist_items,
        sid_depth=sid_depth,
        device=device,
    )
    with torch.no_grad():
        summary = pooled_history_summary(tiger, input_ids, attention_mask)
        value = page_head(summary)
    return float(value.squeeze(0).item())


def eval_item_prefix_values(
    item_head,
    item_features: np.ndarray,
    page_features: np.ndarray,
    *,
    device: torch.device,
) -> List[float]:
    total_items = int(item_features.shape[0])
    if total_items <= 0:
        return [0.0]
    batch_items = torch.tensor(np.repeat(item_features[None, :, :], total_items + 1, axis=0), dtype=torch.float32, device=device)
    batch_page = torch.tensor(np.repeat(page_features[None, :], total_items + 1, axis=0), dtype=torch.float32, device=device)
    mask = torch.zeros((total_items + 1, total_items), dtype=torch.bool, device=device)
    for prefix_len in range(1, total_items + 1):
        mask[prefix_len, :prefix_len] = True
    prefix_len = torch.arange(total_items + 1, dtype=torch.long, device=device)
    total_items_tensor = torch.full((total_items + 1,), total_items, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = item_head(
            batch_items,
            batch_page,
            mask=mask,
            prefix_len=prefix_len,
            total_items=total_items_tensor,
        )
    return [float(x) for x in pred.detach().cpu().tolist()]


def eval_token_prefix_values(
    tiger,
    token_head,
    history_items: Sequence[int],
    target_tokens: Sequence[int],
    *,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    sid_depth: int,
    device: torch.device,
) -> Tuple[List[float], List[float], List[float]]:
    target = [int(x) for x in target_tokens]
    if len(target) != int(sid_depth):
        zeros = [0.0 for _ in range(int(sid_depth))]
        return zeros, zeros, zeros
    input_ids, attention_mask = build_history_state(
        history_items,
        iid2sid_tok_cpu=iid2sid_tok_cpu,
        max_hist_items=max_hist_items,
        sid_depth=sid_depth,
        device=device,
    )
    target_tensor = torch.tensor(target, dtype=torch.long, device=device).view(1, -1)
    with torch.no_grad():
        _logits, hidden = tiger.decode_with_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids_from_targets(target_tensor),
        )
        after = token_head(hidden, target_tensor).squeeze(0)
    after_list = [float(x) for x in after.detach().cpu().tolist()]
    before_list = [0.0] + after_list[:-1]
    diff_list = [float(a - b) for a, b in zip(after_list, before_list)]
    return before_list, after_list, diff_list


def calibrate_diffs(raw: Sequence[float], target_total: float) -> List[float]:
    arr = np.asarray(list(raw), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    raw_sum = float(arr.sum())
    if abs(raw_sum) > 1e-8:
        out = arr * (float(target_total) / raw_sum)
    elif abs(float(target_total)) > 1e-8:
        out = np.ones_like(arr, dtype=np.float32) * (float(target_total) / max(float(arr.size), 1.0))
    else:
        out = np.zeros_like(arr, dtype=np.float32)
    return [float(x) for x in out.tolist()]


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace_map = load_trace_rows(Path(args.trace_path))
    chain_groups = load_chain_groups(Path(args.chain_path))
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    token_vocab_size = int(iid2sid_tok_cpu.max().item()) + 1

    size_args = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, _codebook_size = load_tiger_model(
        tiger_ckpt=str(args.tiger_ckpt),
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
    page_head, page_meta = load_page_prefix_head(str(args.page_head_path), str(args.page_meta_path), device)
    item_head, item_meta = load_item_prefix_head(str(args.item_head_path), str(args.item_meta_path), device)
    token_head, token_meta = load_prefix_value_head(str(args.token_head_path), str(args.token_meta_path), device)

    n_pages = 0
    n_rows = 0
    with output_path.open("w", encoding="utf-8") as out:
        for key in sorted(chain_groups.keys(), key=lambda x: tuple(int(v) for v in x.split(":"))):
            if key not in trace_map:
                continue
            if int(args.max_pages) > 0 and n_pages >= int(args.max_pages):
                break
            trace_row = trace_map[key]
            page_rows = sorted(chain_groups[key], key=lambda x: int(x.get("slate_item_index", 0)))
            history_items = [int(x) for x in trace_row.get("history_items", [])]
            selected_item_ids = [int(x) for x in trace_row.get("selected_item_ids", [])]
            selected_sid_tokens_list = [[int(v) for v in seq] for seq in trace_row.get("selected_sid_tokens_list", [])]
            selected_item_rewards = [float(x) for x in trace_row.get("selected_item_rewards", [])]
            slate_size = int(len(selected_item_ids))
            if slate_size <= 0:
                continue

            page_before_value = eval_page_prefix_value(
                tiger,
                page_head,
                history_items,
                iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                max_hist_items=int(args.max_hist_items),
                sid_depth=int(sid_depth),
                device=device,
            )
            page_after_value = eval_page_prefix_value(
                tiger,
                page_head,
                history_items + selected_item_ids,
                iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                max_hist_items=int(args.max_hist_items),
                sid_depth=int(sid_depth),
                device=device,
            )
            page_adv = float(page_after_value - page_before_value)

            online = build_online_slate_inputs(
                history_items=history_items,
                candidate_item_ids=selected_item_ids,
                candidate_sid_tokens_list=selected_sid_tokens_list,
                iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                max_hist_items=int(args.max_hist_items),
                token_vocab_size=int(token_vocab_size),
                base_scores=None,
            )
            item_features = np.asarray(online["item_features"], dtype=np.float32)
            page_features = np.asarray(online["page_features"], dtype=np.float32)
            item_prefix_values = eval_item_prefix_values(
                item_head,
                item_features,
                page_features,
                device=device,
            )
            item_adv_raw = [float(item_prefix_values[i + 1] - item_prefix_values[i]) for i in range(len(item_prefix_values) - 1)]
            item_adv_cal = calibrate_diffs(item_adv_raw, page_adv)
            item_raw_sum = float(sum(item_adv_raw))
            item_cal_sum = float(sum(item_adv_cal))

            page_row_by_pos = {int(row.get("slate_item_index", -1)): row for row in page_rows}
            for item_pos in range(slate_size):
                row = page_row_by_pos.get(item_pos)
                if row is None:
                    continue
                target_tokens = [int(x) for x in row.get("selected_sid_tokens", [])]
                token_context = history_items + selected_item_ids[:item_pos]
                tok_before, tok_after, tok_adv_raw = eval_token_prefix_values(
                    tiger,
                    token_head,
                    token_context,
                    target_tokens,
                    iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
                    max_hist_items=int(args.max_hist_items),
                    sid_depth=int(sid_depth),
                    device=device,
                )
                item_adv_this = float(item_adv_raw[item_pos]) if item_pos < len(item_adv_raw) else 0.0
                item_adv_this_cal = float(item_adv_cal[item_pos]) if item_pos < len(item_adv_cal) else 0.0
                tok_adv_cal = calibrate_diffs(tok_adv_raw, item_adv_this_cal)
                payload = dict(row)
                payload.update(
                    {
                        "page_prefix_value_before": float(page_before_value),
                        "page_prefix_value_after": float(page_after_value),
                        "page_adv_prefix_diff": float(page_adv),
                        "page_item_adv_sum_raw": float(item_raw_sum),
                        "page_item_adv_sum_calibrated": float(item_cal_sum),
                        "page_item_prefix_residual_raw": float(page_adv - item_raw_sum),
                        "page_item_prefix_residual_calibrated": float(page_adv - item_cal_sum),
                        "item_prefix_value_before": float(item_prefix_values[item_pos]),
                        "item_prefix_value_after": float(item_prefix_values[item_pos + 1]),
                        "item_adv_prefix_diff": float(item_adv_this),
                        "item_adv_prefix_diff_calibrated": float(item_adv_this_cal),
                        "token_prefix_value_before": [float(x) for x in tok_before],
                        "token_prefix_value_after": [float(x) for x in tok_after],
                        "token_adv_prefix_diff": [float(x) for x in tok_adv_raw],
                        "token_adv_prefix_diff_calibrated": [float(x) for x in tok_adv_cal],
                        "item_token_adv_sum_raw": float(sum(tok_adv_raw)),
                        "item_token_adv_sum_calibrated": float(sum(tok_adv_cal)),
                        "item_token_prefix_residual_raw": float(item_adv_this - sum(tok_adv_raw)),
                        "item_token_prefix_residual_calibrated": float(item_adv_this_cal - sum(tok_adv_cal)),
                        "legacy_token_credit_field": str(args.legacy_token_field),
                        "legacy_token_credit": [float(x) for x in row.get(str(args.legacy_token_field), row.get("token_credit_calibrated", []))],
                    }
                )
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n_rows += 1
            n_pages += 1

    meta = {
        "method": "TIGER Hierarchical Prefix Difference Chain",
        "trace_path": str(Path(args.trace_path).resolve()),
        "chain_path": str(Path(args.chain_path).resolve()),
        "output_path": str(output_path.resolve()),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "max_hist_items": int(args.max_hist_items),
        "page_head_path": str(Path(args.page_head_path).resolve()),
        "page_meta_path": str(Path(args.page_meta_path).resolve()),
        "item_head_path": str(Path(args.item_head_path).resolve()),
        "item_meta_path": str(Path(args.item_meta_path).resolve()),
        "token_head_path": str(Path(args.token_head_path).resolve()),
        "token_meta_path": str(Path(args.token_meta_path).resolve()),
        "page_meta": page_meta,
        "item_meta": item_meta,
        "token_meta": token_meta,
        "sid_depth": int(sid_depth),
        "n_pages": int(n_pages),
        "n_rows": int(n_rows),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hier-prefix] saved chain to {output_path}")
    print(f"[hier-prefix] meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
