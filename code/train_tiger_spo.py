import argparse
import json
import random
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils
from reader import *  # noqa: F401,F403

from tiger_phase2_blend_common import (
    build_history_tokens,
    build_iid2sid_tokens,
    decoder_input_ids_from_targets,
    infer_model_size_args,
    load_tiger_model,
    write_json,
)


def load_reader_from_uirm_log(uirm_log_path: str, device: str):
    with open(uirm_log_path, "r", encoding="utf-8") as infile:
        class_args = eval(infile.readline(), {"Namespace": Namespace})
        training_args = eval(infile.readline(), {"Namespace": Namespace})
    training_args.val_holdout_per_user = 0
    training_args.test_holdout_per_user = 0
    training_args.device = device
    reader_class = eval("{0}.{0}".format(class_args.reader))
    return reader_class(training_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIGER-SPO: simulator preference optimization for TIGER.")
    parser.add_argument("--pair_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Frozen reference TIGER checkpoint.")
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="", help="Actor init checkpoint. Defaults to --tiger_ckpt.")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument(
        "--train_scope",
        type=str,
        default="last_decoder_block",
        choices=["decoder_only", "last_decoder_block", "full"],
    )
    parser.add_argument("--spo_beta", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--sft_scale", type=float, default=0.10)
    parser.add_argument("--gap_scale", type=float, default=1.0)
    parser.add_argument("--gap_clip", type=float, default=2.0)
    parser.add_argument(
        "--score_normalization",
        type=str,
        default="mean_token",
        choices=["sum", "mean_token", "mean_item"],
    )
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


class TigerSPODataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_rows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], dim=0),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], dim=0),
        "chosen_tokens": torch.stack([torch.tensor(x["chosen_tokens"], dtype=torch.long) for x in batch], dim=0),
        "rejected_tokens": torch.stack([torch.tensor(x["rejected_tokens"], dtype=torch.long) for x in batch], dim=0),
        "reward_gap": torch.tensor([float(x["reward_gap"]) for x in batch], dtype=torch.float32),
        "groups": [x["group"] for x in batch],
    }


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, g in enumerate(groups):
        (valid_idx if g in valid_groups else train_idx).append(idx)
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def load_pair_rows(
    pair_path: Path,
    reader,
    sid_mapping_path: str,
    max_hist_items: int,
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    rows: List[Dict[str, Any]] = []
    slate_size = 0

    with pair_path.open("r", encoding="utf-8") as infile:
        for line_idx, line in enumerate(infile):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chosen_tokens = [[int(x) for x in seq] for seq in payload.get("chosen_sid_tokens_list", [])]
            rejected_tokens = [[int(x) for x in seq] for seq in payload.get("rejected_sid_tokens_list", [])]
            if not chosen_tokens or not rejected_tokens:
                continue
            if len(chosen_tokens) != len(rejected_tokens):
                continue
            if any(len(seq) != sid_depth for seq in chosen_tokens):
                continue
            if any(len(seq) != sid_depth for seq in rejected_tokens):
                continue
            history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(
                hist_tensor,
                iid2sid_tok_cpu,
                int(max_hist_items),
                int(sid_depth),
            )
            group = f"{payload.get('user_id', 'na')}:{payload.get('episode_progress_index', 'na')}"
            slate_size = max(slate_size, len(chosen_tokens))
            rows.append(
                {
                    "group": group,
                    "input_ids": input_ids.squeeze(0).tolist(),
                    "attention_mask": attention_mask.squeeze(0).tolist(),
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "reward_gap": float(payload.get("reward_gap", 0.0)),
                }
            )
    if not rows:
        raise ValueError(f"No usable SPO rows in {pair_path}")
    return rows, sid_depth, vocab_size, int(slate_size)


def set_train_scope(tiger, scope: str) -> int:
    for p in tiger.parameters():
        p.requires_grad = False
    name = str(scope)
    if name == "full":
        for p in tiger.parameters():
            p.requires_grad = True
    elif name == "decoder_only":
        for p in tiger.model.decoder.parameters():
            p.requires_grad = True
        for p in tiger.model.lm_head.parameters():
            p.requires_grad = True
    elif name == "last_decoder_block":
        for p in tiger.model.decoder.block[-1].parameters():
            p.requires_grad = True
        for p in tiger.model.decoder.final_layer_norm.parameters():
            p.requires_grad = True
        for p in tiger.model.lm_head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported train_scope: {scope}")
    return sum(p.numel() for p in tiger.parameters() if p.requires_grad)


def repeat_encoder_hidden(encoder_hidden: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if int(repeat_count) <= 1:
        return encoder_hidden
    return encoder_hidden.unsqueeze(1).expand(-1, int(repeat_count), -1, -1).reshape(
        encoder_hidden.shape[0] * int(repeat_count),
        encoder_hidden.shape[1],
        encoder_hidden.shape[2],
    )


def repeat_attention_mask(attention_mask: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if int(repeat_count) <= 1:
        return attention_mask
    return attention_mask.unsqueeze(1).expand(-1, int(repeat_count), -1).reshape(
        attention_mask.shape[0] * int(repeat_count),
        attention_mask.shape[1],
    )


def score_mode_divisor(token_count: int, slate_size: int, score_normalization: str) -> float:
    if str(score_normalization) == "sum":
        return 1.0
    if str(score_normalization) == "mean_item":
        return float(max(1, slate_size))
    return float(max(1, token_count))


def compute_slate_logp_scores(
    tiger,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    score_normalization: str,
    encoder_no_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, slate_size, sid_depth = target_tokens.shape
    flat_targets = target_tokens.reshape(batch_size * slate_size, sid_depth)
    decoder_input_ids = decoder_input_ids_from_targets(flat_targets)

    if bool(encoder_no_grad):
        with torch.no_grad():
            encoder_hidden = tiger.encode(input_ids, attention_mask)
    else:
        encoder_hidden = tiger.encode(input_ids, attention_mask)

    encoder_hidden = repeat_encoder_hidden(encoder_hidden, slate_size)
    attention_mask_rep = repeat_attention_mask(attention_mask, slate_size)
    logits, _ = tiger.decode_with_hidden_from_encoded(
        encoder_hidden,
        attention_mask=attention_mask_rep,
        decoder_input_ids=decoder_input_ids,
    )
    token_log_probs = F.log_softmax(logits, dim=-1)
    token_logp = token_log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
    item_logp = token_logp.sum(dim=-1).view(batch_size, slate_size)
    denom = score_mode_divisor(int(slate_size * sid_depth), int(slate_size), str(score_normalization))
    slate_scores = item_logp.sum(dim=-1) / float(denom)
    chosen_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        flat_targets.reshape(-1),
        reduction="mean",
    )
    return slate_scores, chosen_ce


def compute_spo_loss(
    actor_chosen: torch.Tensor,
    actor_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    reward_gap: torch.Tensor,
    chosen_ce: torch.Tensor,
    *,
    spo_beta: float,
    label_smoothing: float,
    sft_scale: float,
    gap_scale: float,
    gap_clip: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pref_actor = actor_chosen - actor_rejected
    pref_ref = ref_chosen - ref_rejected
    margin = float(spo_beta) * (pref_actor - pref_ref)

    pos_loss = -F.logsigmoid(margin)
    if float(label_smoothing) > 0:
        neg_loss = -F.logsigmoid(-margin)
        pos_loss = (1.0 - float(label_smoothing)) * pos_loss + float(label_smoothing) * neg_loss

    gap_weight = 1.0 + float(gap_scale) * torch.clamp(reward_gap, min=0.0, max=float(gap_clip))
    pair_loss = (gap_weight * pos_loss).mean()
    loss = pair_loss + float(sft_scale) * chosen_ce

    stats = {
        "pair_loss": float(pair_loss.item()),
        "sft_loss": float(chosen_ce.item()),
        "loss": float(loss.item()),
        "pref_actor": float(pref_actor.mean().item()),
        "pref_ref": float(pref_ref.mean().item()),
        "margin": float(margin.mean().item()),
        "pref_gain": float((pref_actor - pref_ref).mean().item()),
        "pair_acc": float((margin > 0).float().mean().item()),
        "gap_weight": float(gap_weight.mean().item()),
    }
    return loss, stats


def run_epoch(
    actor_tiger,
    ref_tiger,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    actor_tiger.train(mode=train_mode)
    if not train_mode:
        actor_tiger.eval()
    ref_tiger.eval()

    encoder_no_grad = not any(p.requires_grad for p in actor_tiger.model.encoder.parameters())
    meter: Dict[str, List[float]] = {
        "loss": [],
        "pair_loss": [],
        "sft_loss": [],
        "pref_actor": [],
        "pref_ref": [],
        "margin": [],
        "pref_gain": [],
        "pair_acc": [],
        "gap_weight": [],
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chosen_tokens = batch["chosen_tokens"].to(device)
        rejected_tokens = batch["rejected_tokens"].to(device)
        reward_gap = batch["reward_gap"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        actor_chosen, chosen_ce = compute_slate_logp_scores(
            actor_tiger,
            input_ids,
            attention_mask,
            chosen_tokens,
            score_normalization=str(args.score_normalization),
            encoder_no_grad=bool(encoder_no_grad),
        )
        actor_rejected, _ = compute_slate_logp_scores(
            actor_tiger,
            input_ids,
            attention_mask,
            rejected_tokens,
            score_normalization=str(args.score_normalization),
            encoder_no_grad=bool(encoder_no_grad),
        )
        with torch.no_grad():
            ref_chosen, _ = compute_slate_logp_scores(
                ref_tiger,
                input_ids,
                attention_mask,
                chosen_tokens,
                score_normalization=str(args.score_normalization),
                encoder_no_grad=False,
            )
            ref_rejected, _ = compute_slate_logp_scores(
                ref_tiger,
                input_ids,
                attention_mask,
                rejected_tokens,
                score_normalization=str(args.score_normalization),
                encoder_no_grad=False,
            )

        loss, stats = compute_spo_loss(
            actor_chosen,
            actor_rejected,
            ref_chosen,
            ref_rejected,
            reward_gap,
            chosen_ce,
            spo_beta=float(args.spo_beta),
            label_smoothing=float(args.label_smoothing),
            sft_scale=float(args.sft_scale),
            gap_scale=float(args.gap_scale),
            gap_clip=float(args.gap_clip),
        )

        if train_mode:
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in actor_tiger.parameters() if p.requires_grad],
                    max_norm=float(args.grad_clip_norm),
                )
            optimizer.step()

        for key in meter.keys():
            meter[key].append(float(stats[key]))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in meter.items()}


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(args.device)

    pair_path = Path(args.pair_path)
    if not pair_path.is_absolute():
        pair_path = pair_path.resolve()

    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(device))
    rows, sid_depth, vocab_size, slate_size = load_pair_rows(
        pair_path=pair_path,
        reader=reader,
        sid_mapping_path=str(args.sid_mapping_path),
        max_hist_items=int(args.max_hist_items),
        max_rows=int(args.max_rows),
    )

    groups = [str(x["group"]) for x in rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(
        Subset(TigerSPODataset(rows), train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_rows,
    )
    valid_loader = DataLoader(
        Subset(TigerSPODataset(rows), valid_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_rows,
    )

    size_args = infer_model_size_args(str(args.model_size))
    ref_tiger, sid_depth_model, codebook_size = load_tiger_model(
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
    ref_tiger.eval()
    for p in ref_tiger.parameters():
        p.requires_grad = False

    actor_init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, sid_depth_model2, _codebook_size2 = load_tiger_model(
        tiger_ckpt=actor_init_ckpt,
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
    trainable_params = set_train_scope(actor_tiger, str(args.train_scope))

    if int(sid_depth_model) != int(sid_depth) or int(sid_depth_model2) != int(sid_depth):
        raise ValueError(
            f"SID depth mismatch: pair={sid_depth}, ref_model={sid_depth_model}, actor_model={sid_depth_model2}"
        )
    if int(codebook_size + 1) != int(vocab_size):
        raise ValueError(f"Vocab mismatch: pair={vocab_size}, model={codebook_size + 1}")

    optimizer = torch.optim.AdamW(
        [p for p in actor_tiger.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_state = None
    best_valid = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(actor_tiger, ref_tiger, train_loader, device, args, optimizer=optimizer)
        valid_metrics = run_epoch(actor_tiger, ref_tiger, valid_loader, device, args, optimizer=None)
        record = {
            "epoch": int(epoch),
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(record)
        print(
            f"[tiger-spo] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['pair_acc']:.4f} "
            f"valid_gain={valid_metrics['pref_gain']:.4f}"
        )
        if best_valid is None or float(valid_metrics["loss"]) < float(best_valid):
            best_valid = float(valid_metrics["loss"])
            best_state = deepcopy(actor_tiger.state_dict())

    if best_state is None:
        raise RuntimeError("TIGER-SPO training produced no checkpoint.")

    save_dir = Path(args.save_dir) if args.save_dir else Path(actor_init_ckpt).resolve().parent / "tiger_spo"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "tiger_spo_tiger.pth"
    meta_path = save_dir / "tiger_spo_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "tiger_spo_metrics.json"
    torch.save(best_state, ckpt_path)

    meta = {
        "method": "TIGER SPO",
        "pair_path": str(pair_path.resolve()),
        "reference_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "actor_init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "vocab_size": int(vocab_size),
        "slate_size": int(slate_size),
        "train_scope": str(args.train_scope),
        "trainable_params": int(trainable_params),
        "num_rows": int(len(rows)),
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "spo_beta": float(args.spo_beta),
        "label_smoothing": float(args.label_smoothing),
        "sft_scale": float(args.sft_scale),
        "gap_scale": float(args.gap_scale),
        "gap_clip": float(args.gap_clip),
        "score_normalization": str(args.score_normalization),
        "best_valid_loss": float(best_valid),
        "history": history,
    }
    write_json(meta_path, meta)
    write_json(metrics_path, meta)
    print(f"[tiger-spo] saved checkpoint to {ckpt_path}")
    print(f"[tiger-spo] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
