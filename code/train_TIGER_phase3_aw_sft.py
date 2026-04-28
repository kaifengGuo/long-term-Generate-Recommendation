# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import T5Config, T5ForConditionalGeneration

from reader import *  # noqa: F401,F403
from tiger_phase2_blend_common import build_history_tokens, build_iid2sid_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            vocab_size=config["vocab_size"],
            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["pad_token_id"],
            feed_forward_proj=config["feed_forward_proj"],
        )
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        decoder_input_ids = torch.zeros_like(labels)
        if labels.size(1) > 1:
            decoder_input_ids[:, 1:] = labels[:, :-1]
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            use_cache=False,
        )
        return out.logits


def load_reader_from_uirm_log(uirm_log_path: str, device: str):
    with open(uirm_log_path, "r", encoding="utf-8") as infile:
        class_args = eval(infile.readline(), {"Namespace": Namespace})
        training_args = eval(infile.readline(), {"Namespace": Namespace})
    training_args.val_holdout_per_user = 0
    training_args.test_holdout_per_user = 0
    training_args.device = device
    reader_class = eval("{0}.{0}".format(class_args.reader))
    return reader_class(training_args)


class Phase3AWSFTDataset(Dataset):
    def __init__(
        self,
        *,
        chain_path: str,
        uirm_log_path: str,
        sid_mapping_path: str,
        device: str,
        max_hist_items: int,
        advantage_field: str,
        token_credit_field: str,
        max_rows: int = 0,
    ):
        super().__init__()
        reader = load_reader_from_uirm_log(str(uirm_log_path), str(device))
        sid_df = pd.read_csv(str(sid_mapping_path))
        sid_depth = len([c for c in sid_df.columns if str(c).startswith("sid")])
        iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, str(sid_mapping_path), int(sid_depth), torch.device("cpu"))

        self.samples: List[Dict[str, Any]] = []
        with open(chain_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if int(max_rows) > 0 and line_idx >= int(max_rows):
                    break
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
                target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
                advantage = float(payload.get(str(advantage_field), payload.get("item_credit", 0.0)))
                token_credit = payload.get(str(token_credit_field), None) if str(token_credit_field).strip() else None
                if len(target_tokens) != int(sid_depth) or not any(int(x) > 0 for x in target_tokens):
                    continue
                if token_credit is None or len(token_credit) != int(sid_depth):
                    token_credit = [1.0] * int(sid_depth)
                hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
                history, attention_mask = build_history_tokens(
                    hist_tensor,
                    iid2sid_tok_cpu,
                    int(max_hist_items),
                    int(sid_depth),
                )
                self.samples.append(
                    {
                        "history": history.squeeze(0),
                        "attention_mask": attention_mask.squeeze(0),
                        "target": torch.tensor(target_tokens, dtype=torch.long),
                        "advantage": torch.tensor(float(advantage), dtype=torch.float32),
                        "token_credit": torch.tensor(token_credit, dtype=torch.float32),
                        "group": str(payload["episode_id"]),
                    }
                )

        if not self.samples:
            raise ValueError(f"No usable samples in chain file: {chain_path}")
        self.sid_depth = int(sid_depth)
        self.vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
        self.sample_groups = np.asarray([s["group"] for s in self.samples], dtype=object)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_aw_sft(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "history": torch.stack([x["history"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "target": torch.stack([x["target"] for x in batch], dim=0),
        "advantage": torch.stack([x["advantage"] for x in batch], dim=0),
        "token_credit": torch.stack([x["token_credit"] for x in batch], dim=0),
        "group": [x["group"] for x in batch],
    }


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    uniq = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, g in enumerate(groups):
        if g in valid_groups:
            valid_idx.append(idx)
        else:
            train_idx.append(idx)
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return train_idx, valid_idx


def build_aw_weights(
    advantages: torch.Tensor,
    *,
    mode: str,
    adv_clip: float,
    adv_tau: float,
    min_weight: float,
    max_weight: float,
    pos_floor: float,
) -> torch.Tensor:
    adv = advantages.clamp(min=-float(adv_clip), max=float(adv_clip))
    if str(mode) == "positive_exp":
        weight = torch.exp(torch.relu(adv) / max(float(adv_tau), 1e-6))
    elif str(mode) == "positive_linear":
        weight = 1.0 + torch.relu(adv) / max(float(adv_tau), 1e-6)
    elif str(mode) == "positive_only":
        weight = torch.relu(adv)
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")
    if str(mode) == "positive_only":
        weight = torch.where(weight > 0.0, weight, torch.full_like(weight, float(pos_floor)))
    else:
        weight = torch.maximum(weight, torch.full_like(weight, float(min_weight)))
    weight = weight.clamp(min=float(min_weight), max=float(max_weight))
    return weight


def build_token_weights(
    token_advantages: torch.Tensor,
    *,
    mode: str,
    adv_clip: float,
    adv_tau: float,
    min_weight: float,
    max_weight: float,
    pos_floor: float,
    mix_ratio: float,
) -> torch.Tensor:
    mode_name = str(mode)
    if mode_name == "uniform":
        return torch.ones_like(token_advantages)

    if mode_name in {"positive_exp", "positive_linear", "positive_only"}:
        shaped = build_aw_weights(
            token_advantages,
            mode=mode_name,
            adv_clip=float(adv_clip),
            adv_tau=float(adv_tau),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
            pos_floor=float(pos_floor),
        )
        mix = float(np.clip(float(mix_ratio), 0.0, 1.0))
        if mix >= 1.0:
            return shaped
        base = torch.ones_like(shaped)
        weight = (1.0 - mix) * base + mix * shaped
        weight = weight.clamp(min=float(min_weight), max=float(max_weight))
        return weight

    adv = token_advantages.clamp(min=-float(adv_clip), max=float(adv_clip))
    tau = max(float(adv_tau), 1e-6)
    mix = float(np.clip(float(mix_ratio), 0.0, 1.0))
    base = torch.ones_like(adv)

    if mode_name == "softmax_relative":
        score = adv / tau
    elif mode_name == "positive_softmax_relative":
        score = torch.relu(adv) / tau
    else:
        raise ValueError(f"Unsupported token weight mode: {mode_name}")

    probs = torch.softmax(score, dim=1)
    # Keep average token weight near 1 while only changing the within-item allocation.
    shaped = probs * float(token_advantages.size(1))
    weight = (1.0 - mix) * base + mix * shaped
    weight = weight.clamp(min=float(min_weight), max=float(max_weight))
    return weight


def compute_aw_sft_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    token_advantages: torch.Tensor,
    *,
    weight_mode: str,
    adv_clip: float,
    adv_tau: float,
    min_weight: float,
    max_weight: float,
    pos_floor: float,
    token_weight_mode: str,
    token_adv_clip: float,
    token_adv_tau: float,
    token_min_weight: float,
    token_max_weight: float,
    token_pos_floor: float,
    token_mix_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, length, vocab = student_logits.shape
    token_ce = F.cross_entropy(
        student_logits.reshape(-1, vocab),
        labels.reshape(-1),
        reduction="none",
    ).view(bsz, length)
    token_weights = build_token_weights(
        token_advantages,
        mode=str(token_weight_mode),
        adv_clip=float(token_adv_clip),
        adv_tau=float(token_adv_tau),
        min_weight=float(token_min_weight),
        max_weight=float(token_max_weight),
        pos_floor=float(token_pos_floor),
        mix_ratio=float(token_mix_ratio),
    )
    seq_ce = (token_ce * token_weights).sum(dim=1) / (token_weights.sum(dim=1) + 1e-8)
    weights = build_aw_weights(
        advantages,
        mode=str(weight_mode),
        adv_clip=float(adv_clip),
        adv_tau=float(adv_tau),
        min_weight=float(min_weight),
        max_weight=float(max_weight),
        pos_floor=float(pos_floor),
    )
    loss = (weights * seq_ce).sum() / (weights.sum() + 1e-8)
    stats = {
        "seq_ce": float(seq_ce.mean().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_max": float(weights.max().item()),
        "token_weight_mean": float(token_weights.mean().item()),
        "token_weight_max": float(token_weights.max().item()),
    }
    return loss, stats


def compute_kl_anchor(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    s_logp = torch.log_softmax(student_logits, dim=-1)
    t_prob = torch.softmax(teacher_logits, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean")


@torch.inference_mode()
def evaluate_aw_sft(
    student: TIGER,
    teacher: Optional[TIGER],
    loader: DataLoader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    student.eval()
    if teacher is not None:
        teacher.eval()
    losses, ces, kls, weights = [], [], [], []
    for batch in loader:
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)
        advantages = batch["advantage"].to(device, non_blocking=True)
        token_advantages = batch["token_credit"].to(device, non_blocking=True)
        student_logits = student.forward_logits(input_ids, attention_mask, labels)
        loss_sft, stats = compute_aw_sft_loss(
            student_logits,
            labels,
            advantages,
            token_advantages,
            weight_mode=str(args.weight_mode),
            adv_clip=float(args.adv_clip),
            adv_tau=float(args.adv_tau),
            min_weight=float(args.min_weight),
            max_weight=float(args.max_weight),
            pos_floor=float(args.pos_floor),
            token_weight_mode=str(args.token_weight_mode),
            token_adv_clip=float(args.token_adv_clip),
            token_adv_tau=float(args.token_adv_tau),
            token_min_weight=float(args.token_min_weight),
            token_max_weight=float(args.token_max_weight),
            token_pos_floor=float(args.token_pos_floor),
            token_mix_ratio=float(args.token_mix_ratio),
        )
        loss = loss_sft
        kl_value = torch.tensor(0.0, device=device)
        if teacher is not None and float(args.kl_scale) > 0.0:
            teacher_logits = teacher.forward_logits(input_ids, attention_mask, labels)
            kl_value = compute_kl_anchor(student_logits, teacher_logits)
            loss = loss + float(args.kl_scale) * kl_value
        losses.append(float(loss.item()))
        ces.append(float(stats["seq_ce"]))
        kls.append(float(kl_value.item()))
        weights.append(float(stats["weight_mean"]))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "seq_ce": float(np.mean(ces)) if ces else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "weight_mean": float(np.mean(weights)) if weights else 0.0,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Internal TIGER Phase3 advantage-weighted SFT fine-tuning.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, default="")
    parser.add_argument("--advantage_field", type=str, default="item_credit")
    parser.add_argument("--token_credit_field", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_mode", type=str, default="positive_exp", choices=["positive_exp", "positive_linear", "positive_only"])
    parser.add_argument("--adv_clip", type=float, default=3.0)
    parser.add_argument("--adv_tau", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--max_weight", type=float, default=3.0)
    parser.add_argument("--pos_floor", type=float, default=0.05)
    parser.add_argument(
        "--token_weight_mode",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "positive_exp",
            "positive_linear",
            "positive_only",
            "softmax_relative",
            "positive_softmax_relative",
        ],
    )
    parser.add_argument("--token_adv_clip", type=float, default=1.0)
    parser.add_argument("--token_adv_tau", type=float, default=0.25)
    parser.add_argument("--token_min_weight", type=float, default=0.01)
    parser.add_argument("--token_max_weight", type=float, default=3.0)
    parser.add_argument("--token_pos_floor", type=float, default=0.01)
    parser.add_argument("--token_mix_ratio", type=float, default=1.0)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, default="")
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--val_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")
    args = parser.parse_args()

    if args.model_size == "mini":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif args.model_size == "medium":
        args.num_layers = 4
        args.num_decoder_layers = 4
        args.d_model = 128
        args.d_ff = 1024
        args.num_heads = 6
        args.d_kv = 64
    elif args.model_size == "large":
        args.num_layers = 6
        args.num_decoder_layers = 6
        args.d_model = 192
        args.d_ff = 1536
        args.num_heads = 8
        args.d_kv = 24
    return args


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logging.basicConfig(filename=args.log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = Phase3AWSFTDataset(
        chain_path=str(args.chain_path),
        uirm_log_path=str(args.uirm_log_path),
        sid_mapping_path=str(args.sid_mapping_path),
        device=str(device),
        max_hist_items=int(args.max_hist_items),
        advantage_field=str(args.advantage_field),
        token_credit_field=str(args.token_credit_field),
        max_rows=int(args.max_rows),
    )
    config = {
        "num_layers": args.num_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "d_kv": args.d_kv,
        "dropout_rate": args.dropout_rate,
        "feed_forward_proj": args.feed_forward_proj,
        "vocab_size": dataset.vocab_size,
        "pad_token_id": 0,
        "eos_token_id": 0,
    }

    student = TIGER(config).to(device)
    student.load_state_dict(torch.load(args.init_ckpt, map_location=device))
    if args.freeze_embeddings:
        for p in student.model.get_input_embeddings().parameters():
            p.requires_grad = False
    teacher = None
    teacher_ckpt = str(args.teacher_ckpt).strip() or str(args.init_ckpt)
    if float(args.kl_scale) > 0.0:
        teacher = TIGER(config).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

    train_idx, valid_idx = split_groups(dataset.sample_groups.tolist(), float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.train_num_workers), pin_memory=bool(args.pin_memory), collate_fn=collate_aw_sft)
    val_loader = DataLoader(Subset(dataset, valid_idx), batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.val_num_workers), pin_memory=bool(args.pin_memory), collate_fn=collate_aw_sft)

    optimizer = optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=float(args.lr))
    logging.info(student.n_parameters)
    print(student.n_parameters)

    best_key = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(int(args.num_epochs)):
        student.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["history"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)
            advantages = batch["advantage"].to(device, non_blocking=True)
            token_advantages = batch["token_credit"].to(device, non_blocking=True)
            student_logits = student.forward_logits(input_ids, attention_mask, labels)
            loss_sft, _stats = compute_aw_sft_loss(
                student_logits,
                labels,
                advantages,
                token_advantages,
                weight_mode=str(args.weight_mode),
                adv_clip=float(args.adv_clip),
                adv_tau=float(args.adv_tau),
                min_weight=float(args.min_weight),
                max_weight=float(args.max_weight),
                pos_floor=float(args.pos_floor),
                token_weight_mode=str(args.token_weight_mode),
                token_adv_clip=float(args.token_adv_clip),
                token_adv_tau=float(args.token_adv_tau),
                token_min_weight=float(args.token_min_weight),
                token_max_weight=float(args.token_max_weight),
                token_pos_floor=float(args.token_pos_floor),
                token_mix_ratio=float(args.token_mix_ratio),
            )
            loss = loss_sft
            if teacher is not None and float(args.kl_scale) > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher.forward_logits(input_ids, attention_mask, labels)
                loss = loss + float(args.kl_scale) * compute_kl_anchor(student_logits, teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_aw_sft(student, teacher, val_loader, device, args)
        val_metrics["epoch"] = float(epoch + 1)
        val_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(dict(val_metrics))
        logging.info(f"[Epoch {epoch+1}] {val_metrics}")
        print(f"[Epoch {epoch+1}] {val_metrics}")
        if float(val_metrics["loss"]) < best_key:
            best_key = float(val_metrics["loss"])
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    torch.save(best_state, args.save_path)
    metrics_path = args.metrics_path if str(args.metrics_path).strip() else str(Path(args.save_path).with_suffix(".metrics.json"))
    metrics = {
        "save_path": str(Path(args.save_path).resolve()),
        "init_ckpt": str(Path(args.init_ckpt).resolve()),
        "teacher_ckpt": str(Path(teacher_ckpt).resolve()),
        "chain_path": str(Path(args.chain_path).resolve()),
        "advantage_field": str(args.advantage_field),
        "token_credit_field": str(args.token_credit_field),
        "weight_mode": str(args.weight_mode),
        "adv_clip": float(args.adv_clip),
        "adv_tau": float(args.adv_tau),
        "min_weight": float(args.min_weight),
        "max_weight": float(args.max_weight),
        "pos_floor": float(args.pos_floor),
        "token_weight_mode": str(args.token_weight_mode),
        "token_adv_clip": float(args.token_adv_clip),
        "token_adv_tau": float(args.token_adv_tau),
        "token_min_weight": float(args.token_min_weight),
        "token_max_weight": float(args.token_max_weight),
        "token_pos_floor": float(args.token_pos_floor),
        "token_mix_ratio": float(args.token_mix_ratio),
        "kl_scale": float(args.kl_scale),
        "history": history,
        "n_samples": int(len(dataset)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
    }
    Path(metrics_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Phase3-AWSFT] saved checkpoint to {args.save_path}")
    print(f"[Phase3-AWSFT] saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
