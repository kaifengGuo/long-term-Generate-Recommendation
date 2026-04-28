import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tiger_phase2_blend_common import build_iid2sid_tokens, infer_model_size_args, load_tiger_model  # noqa: E402

from tiger_page_sid_rl.common import (  # noqa: E402
    build_page_samples,
    load_jsonl_rows,
    load_reader_from_uirm_log,
    pooled_history_summary,
    set_random_seed,
    write_json,
)
from tiger_page_sid_rl.models import load_page_sid_qcritic_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build item/SID advantage chain from a page critic.")
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--critic_bundle_path", type=str, required=True)
    parser.add_argument("--critic_meta_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--critic_eval_batch_size", type=int, default=256)
    parser.add_argument("--critic_pessimism_beta", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default="")
    return parser.parse_args()


def evaluate_q_variants(
    critic,
    *,
    pre_summary: torch.Tensor,
    page_features: torch.Tensor,
    user_features: torch.Tensor,
    variant_token_ids: torch.Tensor,
    eval_batch_size: int,
    device: torch.device,
    pessimism_beta: float,
) -> Dict[str, torch.Tensor]:
    q_means: List[torch.Tensor] = []
    q_stds: List[torch.Tensor] = []
    total = int(variant_token_ids.shape[0])
    for start in range(0, total, int(eval_batch_size)):
        chunk_tokens = variant_token_ids[start:start + int(eval_batch_size)].to(device)
        chunk_items = (chunk_tokens > 0).any(dim=-1)
        chunk_pre = pre_summary.expand(chunk_tokens.shape[0], -1)
        chunk_page_feat = page_features.expand(chunk_tokens.shape[0], -1)
        chunk_user_feat = user_features.expand(chunk_tokens.shape[0], -1)
        outputs = critic(
            pre_summary=chunk_pre,
            token_ids=chunk_tokens,
            item_mask=chunk_items,
            page_features=chunk_page_feat,
            user_features=chunk_user_feat,
        )
        q_means.append(outputs.get("q_mean", outputs["q_value"]).detach().cpu())
        q_stds.append(outputs.get("q_std", torch.zeros_like(outputs["q_value"])).detach().cpu())
    q_mean = torch.cat(q_means, dim=0)
    q_std = torch.cat(q_stds, dim=0)
    q_pess = q_mean - float(pessimism_beta) * q_std
    return {"q_mean": q_mean, "q_std": q_std, "q_pess": q_pess}


def build_page_variants(token_ids: torch.Tensor) -> Tuple[torch.Tensor, List[int], List[List[int]], List[int]]:
    slate_size, sid_depth = int(token_ids.shape[0]), int(token_ids.shape[1])
    variants = [token_ids.clone()]
    item_null_indices: List[int] = []
    sid_prefix_indices: List[List[int]] = [[-1 for _ in range(sid_depth)] for _ in range(slate_size)]
    valid_lengths: List[int] = [int((token_ids[item_idx] > 0).sum().item()) for item_idx in range(slate_size)]
    for item_idx in range(slate_size):
        null_variant = token_ids.clone()
        null_variant[item_idx].zero_()
        variants.append(null_variant)
        item_null_indices.append(len(variants) - 1)
        for sid_idx in range(valid_lengths[item_idx]):
            prefix_variant = token_ids.clone()
            prefix_variant[item_idx, sid_idx + 1:] = 0
            variants.append(prefix_variant)
            sid_prefix_indices[item_idx][sid_idx] = len(variants) - 1
    return torch.stack(variants, dim=0), item_null_indices, sid_prefix_indices, valid_lengths


def main() -> int:
    args = parse_args()
    set_random_seed(int(args.seed))
    device = torch.device(str(args.device))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace_rows = load_jsonl_rows(str(args.trace_path))
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), "cpu")
    sid_df = pd.read_csv(str(args.sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(args.sid_mapping_path), int(sid_depth_cfg), torch.device("cpu"))
    page_rows, page_meta = build_page_samples(
        trace_rows=trace_rows,
        iid2sid_tok_cpu=iid2sid_tok_cpu.cpu(),
        reader=reader,
        max_hist_items=int(args.max_hist_items),
        gamma=float(args.gamma),
        hazard_lambda=float(args.hazard_lambda),
        max_episodes=int(args.max_episodes),
    )
    if not page_rows:
        raise ValueError("No usable pages were built from trace for SID advantage chain.")

    size_cfg = infer_model_size_args(str(args.model_size))
    tiger, _sid_depth_model, _codebook_size_model = load_tiger_model(
        tiger_ckpt=str(args.tiger_ckpt),
        sid_mapping_path=str(args.sid_mapping_path),
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

    critic, critic_meta = load_page_sid_qcritic_bundle(
        str(args.critic_bundle_path),
        str(args.critic_meta_path),
        device,
    )
    critic.eval()

    n_rows = 0
    page_q_values: List[float] = []
    page_q_stds: List[float] = []
    page_q_pess_values: List[float] = []
    item_adv_values: List[float] = []
    item_adv_pess_values: List[float] = []
    sid_adv_values: List[float] = []
    sid_adv_pess_values: List[float] = []
    sid_cons_errors: List[float] = []
    sid_cons_pess_errors: List[float] = []
    with output_path.open("w", encoding="utf-8") as out:
        for page in page_rows:
            pre_input_ids = torch.tensor(page["pre_input_ids"], dtype=torch.long, device=device).view(1, -1)
            pre_attention_mask = torch.tensor(page["pre_attention_mask"], dtype=torch.long, device=device).view(1, -1)
            page_features = torch.tensor(page["page_features"], dtype=torch.float32, device=device).view(1, -1)
            user_features = torch.tensor(page.get("user_features", []), dtype=torch.float32, device=device).view(1, -1)
            token_ids = torch.tensor(page["token_ids"], dtype=torch.long)
            slate_size = int(token_ids.shape[0])
            sid_depth = int(token_ids.shape[1])
            if slate_size <= 0:
                continue
            with torch.no_grad():
                pre_summary = pooled_history_summary(tiger, pre_input_ids, pre_attention_mask)
                variants, item_null_indices, sid_prefix_indices, valid_lengths = build_page_variants(token_ids)
                q_outputs = evaluate_q_variants(
                    critic,
                    pre_summary=pre_summary,
                    page_features=page_features,
                    user_features=user_features,
                    variant_token_ids=variants,
                    eval_batch_size=int(args.critic_eval_batch_size),
                    device=device,
                    pessimism_beta=float(args.critic_pessimism_beta),
                )
            q_mean_values = q_outputs["q_mean"].numpy()
            q_std_values = q_outputs["q_std"].numpy()
            q_pess_values = q_outputs["q_pess"].numpy()
            full_q = float(q_mean_values[0])
            full_q_std = float(q_std_values[0])
            full_q_pess = float(q_pess_values[0])
            page_q_values.append(full_q)
            page_q_stds.append(full_q_std)
            page_q_pess_values.append(full_q_pess)
            for item_idx in range(slate_size):
                target_tokens = [int(x) for x in page["token_ids"][item_idx]]
                if not any(int(x) > 0 for x in target_tokens):
                    continue
                q_without_item = float(q_mean_values[item_null_indices[item_idx]])
                q_without_item_std = float(q_std_values[item_null_indices[item_idx]])
                q_without_item_pess = float(q_pess_values[item_null_indices[item_idx]])
                item_adv = float(full_q - q_without_item)
                item_adv_pess = float(full_q_pess - q_without_item_pess)
                sid_adv = [0.0 for _ in range(sid_depth)]
                sid_adv_pess = [0.0 for _ in range(sid_depth)]
                prefix_q_values: List[float] = []
                prefix_q_std_values: List[float] = []
                prefix_q_pess_values: List[float] = []
                prev_q = q_without_item
                prev_q_pess = q_without_item_pess
                for sid_idx in range(valid_lengths[item_idx]):
                    prefix_idx = sid_prefix_indices[item_idx][sid_idx]
                    prefix_q = float(q_mean_values[prefix_idx])
                    prefix_q_std = float(q_std_values[prefix_idx])
                    prefix_q_pess = float(q_pess_values[prefix_idx])
                    prefix_q_values.append(prefix_q)
                    prefix_q_std_values.append(prefix_q_std)
                    prefix_q_pess_values.append(prefix_q_pess)
                    sid_adv[sid_idx] = float(prefix_q - prev_q)
                    sid_adv_pess[sid_idx] = float(prefix_q_pess - prev_q_pess)
                    prev_q = prefix_q
                    prev_q_pess = prefix_q_pess
                sid_sum = float(sum(sid_adv))
                sid_pess_sum = float(sum(sid_adv_pess))
                sid_cons_errors.append(abs(item_adv - sid_sum))
                sid_cons_pess_errors.append(abs(item_adv_pess - sid_pess_sum))
                item_adv_values.append(item_adv)
                item_adv_pess_values.append(item_adv_pess)
                sid_adv_values.extend(float(x) for x in sid_adv if abs(float(x)) > 0.0)
                sid_adv_pess_values.extend(float(x) for x in sid_adv_pess if abs(float(x)) > 0.0)
                payload = {
                    "episode_id": int(page["episode_id"]),
                    "user_id": int(page["user_id"]),
                    "page_index": int(page["page_index"]),
                    "history_items": list(page["history_items"]),
                    "selected_item_ids": [int(x) for x in page["selected_item_ids"]],
                    "slate_size": int(len(page["selected_item_ids"])),
                    "slate_item_index": int(item_idx),
                    "selected_item_id": int(page["selected_item_ids"][item_idx]),
                    "selected_sid_tokens": list(target_tokens),
                    "page_q_value": float(full_q),
                    "page_q_mean": float(full_q),
                    "page_q_std": float(full_q_std),
                    "page_q_pess": float(full_q_pess),
                    "page_q_target": float(page["q_target"]),
                    "page_q_without_item": float(q_without_item),
                    "page_q_without_item_mean": float(q_without_item),
                    "page_q_without_item_std": float(q_without_item_std),
                    "page_q_without_item_pess": float(q_without_item_pess),
                    "item_advantage": float(item_adv),
                    "item_advantage_mean": float(item_adv),
                    "item_advantage_pess": float(item_adv_pess),
                    "sid_advantage": [float(x) for x in sid_adv],
                    "sid_advantage_pess": [float(x) for x in sid_adv_pess],
                    "sid_prefix_q_values": [float(x) for x in prefix_q_values],
                    "sid_prefix_q_std_values": [float(x) for x in prefix_q_std_values],
                    "sid_prefix_q_pess_values": [float(x) for x in prefix_q_pess_values],
                    "step_reward": float(page["step_reward"]),
                    "lt_reward": float(page["lt_reward"]),
                    "done": bool(page["done"]),
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n_rows += 1

    summary = {
        "method": "TIGER Page-SID Advantage Chain",
        "trace_path": str(Path(args.trace_path).resolve()),
        "critic_bundle_path": str(Path(args.critic_bundle_path).resolve()),
        "critic_meta_path": str(Path(args.critic_meta_path).resolve()),
        "output_path": str(output_path.resolve()),
        "critic_hidden_size": int(critic_meta["hidden_size"]),
        "critic_ensemble_size": int(critic_meta.get("ensemble_size", 1)),
        "critic_pessimism_beta": float(args.critic_pessimism_beta),
        "n_pages": int(len(page_rows)),
        "n_rows": int(n_rows),
        "page_q_mean": float(np.mean(page_q_values)) if page_q_values else 0.0,
        "page_q_std_mean": float(np.mean(page_q_stds)) if page_q_stds else 0.0,
        "page_q_pess_mean": float(np.mean(page_q_pess_values)) if page_q_pess_values else 0.0,
        "page_q_abs_mean": float(np.mean(np.abs(page_q_values))) if page_q_values else 0.0,
        "item_adv_abs_mean": float(np.mean(np.abs(item_adv_values))) if item_adv_values else 0.0,
        "item_adv_pess_abs_mean": float(np.mean(np.abs(item_adv_pess_values))) if item_adv_pess_values else 0.0,
        "sid_adv_abs_mean": float(np.mean(np.abs(sid_adv_values))) if sid_adv_values else 0.0,
        "sid_adv_pess_abs_mean": float(np.mean(np.abs(sid_adv_pess_values))) if sid_adv_pess_values else 0.0,
        "sid_item_cons_mae": float(np.mean(sid_cons_errors)) if sid_cons_errors else 0.0,
        "sid_item_cons_pess_mae": float(np.mean(sid_cons_pess_errors)) if sid_cons_pess_errors else 0.0,
        "sid_item_cons_max": float(np.max(sid_cons_errors)) if sid_cons_errors else 0.0,
        "page_meta": page_meta,
    }
    summary_out = Path(args.summary_out) if str(args.summary_out).strip() else output_path.with_name("page_sid_chain_summary.json")
    write_json(summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
