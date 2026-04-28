import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from build_tiger_phase3_credit_chain import load_reader_from_uirm_log
from tiger_phase2_blend_common import (
    TokenPrefixValueHead,
    build_iid2sid_tokens,
    build_history_tokens,
    decoder_input_ids_from_targets,
    load_tiger_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment phase3 TIGER credit chain with a prefix critic and hybrid token credit.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--prefix_head_path", type=str, required=True)
    parser.add_argument("--prefix_meta_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=str, default="base", choices=["mini", "small", "base", "large"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--transport_token_field", type=str, default="token_credit_calibrated")
    parser.add_argument("--item_credit_field", type=str, default="item_credit")
    parser.add_argument("--hybrid_alpha", type=float, default=0.6, help="Transport weight in hybrid token credit.")
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def apply_size_defaults(args: argparse.Namespace) -> None:
    size = str(args.size).lower()
    if size == "mini":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif size == "small":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif size == "base":
        args.num_layers = 4
        args.num_decoder_layers = 4
        args.d_model = 128
        args.d_ff = 1024
        args.num_heads = 6
        args.d_kv = 64
    elif size == "large":
        args.num_layers = 6
        args.num_decoder_layers = 6
        args.d_model = 192
        args.d_ff = 1536
        args.num_heads = 8
        args.d_kv = 24


def load_prefix_meta(meta_path: str) -> Dict[str, Any]:
    if not meta_path:
        return {}
    path = Path(meta_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_to_delta(prefix_values: np.ndarray) -> np.ndarray:
    prefix_values = np.asarray(prefix_values, dtype=np.float32).reshape(-1)
    if prefix_values.size <= 1:
        return prefix_values.copy()
    delta = prefix_values.copy()
    delta[1:] = prefix_values[1:] - prefix_values[:-1]
    return delta.astype(np.float32)


def calibrate_delta(delta: np.ndarray, item_credit: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    if delta.size == 0:
        return delta
    residual = float(item_credit) - float(delta.sum())
    delta[-1] += residual
    return delta.astype(np.float32)


def build_hybrid(transport_credit: Sequence[float], critic_credit: Sequence[float], item_credit: float, alpha: float) -> np.ndarray:
    transport_arr = np.asarray(list(transport_credit), dtype=np.float32).reshape(-1)
    critic_arr = np.asarray(list(critic_credit), dtype=np.float32).reshape(-1)
    if transport_arr.shape != critic_arr.shape:
        raise ValueError(f"shape mismatch: transport {transport_arr.shape} vs critic {critic_arr.shape}")
    hybrid = float(alpha) * transport_arr + (1.0 - float(alpha)) * critic_arr
    if hybrid.size > 0:
        hybrid[-1] += float(item_credit) - float(hybrid.sum())
    return hybrid.astype(np.float32)


def main() -> None:
    args = parse_args()
    apply_size_defaults(args)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    meta = load_prefix_meta(args.prefix_meta_path)
    token_dim = int(meta.get("token_dim", 32))
    mlp_dim = int(meta.get("mlp_dim", 128))
    max_hist_items = int(meta.get("max_hist_items", args.max_hist_items))

    reader = load_reader_from_uirm_log(args.uirm_log_path, str(device))
    tiger, sid_depth, codebook_size = load_tiger_model(
        tiger_ckpt=args.tiger_ckpt,
        sid_mapping_path=args.sid_mapping_path,
        num_layers=int(args.num_layers),
        num_decoder_layers=int(args.num_decoder_layers),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        num_heads=int(args.num_heads),
        d_kv=int(args.d_kv),
        dropout_rate=float(args.dropout_rate),
        feed_forward_proj=str(args.feed_forward_proj),
        device=device,
    )
    iid2sid_tok, _sid2iid_map = build_iid2sid_tokens(reader, args.sid_mapping_path, sid_depth, device)

    head = TokenPrefixValueHead(
        hidden_size=int(args.d_model),
        vocab_size=int(codebook_size) + 1,
        token_dim=token_dim,
        mlp_dim=mlp_dim,
    ).to(device)
    state = torch.load(args.prefix_head_path, map_location=device)
    head.load_state_dict(state)
    head.eval()

    chain_path = Path(args.chain_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with chain_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
            transport_credit = payload.get(str(args.transport_token_field), payload.get("token_credit_calibrated", payload.get("token_credit", [])))
            item_credit = float(payload.get(str(args.item_credit_field), payload.get("item_credit", 0.0)))
            history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
            if len(target_tokens) != int(sid_depth):
                continue
            if len(transport_credit) != int(sid_depth):
                continue

            hist_tensor = torch.tensor(history_items, dtype=torch.long, device=device).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok,
                int(max_hist_items),
                int(sid_depth),
            )
            target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=device).view(1, -1)
            with torch.no_grad():
                _logits, hidden = tiger.decode_with_hidden(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids_from_targets(target_tensor),
                )
                pred_prefix = head(hidden.detach(), target_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)

            critic_delta_raw = prefix_to_delta(pred_prefix)
            critic_delta_cal = calibrate_delta(critic_delta_raw, item_credit)
            hybrid_delta = build_hybrid(transport_credit, critic_delta_cal, item_credit, float(args.hybrid_alpha))

            payload["critic_prefix_value"] = [float(x) for x in pred_prefix.tolist()]
            payload["critic_item_credit_raw"] = float(pred_prefix[-1]) if pred_prefix.size > 0 else 0.0
            payload["critic_token_credit_raw"] = [float(x) for x in critic_delta_raw.tolist()]
            payload["critic_token_credit_calibrated"] = [float(x) for x in critic_delta_cal.tolist()]
            payload["hybrid_token_credit"] = [float(x) for x in hybrid_delta.tolist()]
            payload["hybrid_item_credit"] = float(np.sum(hybrid_delta))
            payload["hybrid_alpha"] = float(args.hybrid_alpha)
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            if int(args.max_records) > 0 and count >= int(args.max_records):
                break

    summary = {
        "method": "TIGER Phase4 Prefix Critic Augmentation",
        "chain_path": str(chain_path.resolve()),
        "prefix_head_path": str(Path(args.prefix_head_path).resolve()),
        "output_path": str(output_path.resolve()),
        "transport_token_field": str(args.transport_token_field),
        "item_credit_field": str(args.item_credit_field),
        "hybrid_alpha": float(args.hybrid_alpha),
        "n_records": int(count),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Augmented {count} records -> {output_path}")
    print(f"[Done] Summary: {summary_path}")


if __name__ == "__main__":
    main()
