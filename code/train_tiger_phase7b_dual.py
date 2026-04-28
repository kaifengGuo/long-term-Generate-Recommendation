import argparse
import json
import random
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import utils
from reader import *  # noqa: F401,F403

from tiger_phase2_blend_common import build_history_tokens, build_iid2sid_tokens, decoder_input_ids_from_targets, infer_model_size_args, load_tiger_model, write_json
from tiger_phase5_token_actor_common import TokenResidualActorHead


CODE_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = CODE_DIR / "eval_tiger_phase2_blend_env.py"


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
    parser = argparse.ArgumentParser(description="Train a dual-channel Phase7b policy.")
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--init_tiger_ckpt", type=str, default="")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--veto_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--actor_item_scale", type=float, default=0.20)
    parser.add_argument("--actor_support_scale", type=float, default=0.40)
    parser.add_argument("--actor_hist_pos_scale", type=float, default=0.40)
    parser.add_argument("--actor_page_gate_scale", type=float, default=0.25)
    parser.add_argument("--actor_kl_scale", type=float, default=0.10)
    parser.add_argument("--veto_item_scale", type=float, default=0.25)
    parser.add_argument("--veto_page_scale", type=float, default=0.80)
    parser.add_argument("--veto_support_scale", type=float, default=0.20)
    parser.add_argument("--veto_hist_neg_scale", type=float, default=0.50)
    parser.add_argument("--veto_anchor_scale", type=float, default=0.05)
    parser.add_argument("--veto_train_scale", type=float, default=0.35)
    parser.add_argument("--veto_margin_keep", type=float, default=0.90)
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--residual_clip", type=float, default=2.0)
    parser.add_argument("--min_signal_mass", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--online_eval_episodes", type=int, default=0)
    parser.add_argument("--online_eval_batch_size", type=int, default=32)
    parser.add_argument("--online_eval_device", type=str, default="")
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--phase6_joint_head_path", type=str, default="")
    parser.add_argument("--phase6_joint_meta_path", type=str, default="")
    parser.add_argument("--phase6_prefix_scale", type=float, default=0.02)
    parser.add_argument("--phase6_token_actor_scale", type=float, default=0.01)
    parser.add_argument("--eval_veto_scale", type=float, default=0.05)
    return parser.parse_args()


class Phase7bDualDataset(Dataset):
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
        "target_tokens": torch.stack([torch.tensor(x["target_tokens"], dtype=torch.long) for x in batch], dim=0),
        "token_pos_credit": torch.stack([torch.tensor(x["token_pos_credit"], dtype=torch.float32) for x in batch], dim=0),
        "token_neg_credit": torch.stack([torch.tensor(x["token_neg_credit"], dtype=torch.float32) for x in batch], dim=0),
        "item_pos_credit": torch.tensor([float(x["item_pos_credit"]) for x in batch], dtype=torch.float32),
        "item_neg_credit": torch.tensor([float(x["item_neg_credit"]) for x in batch], dtype=torch.float32),
        "page_item_pos_credit": torch.tensor([float(x["page_item_pos_credit"]) for x in batch], dtype=torch.float32),
        "page_item_neg_credit": torch.tensor([float(x["page_item_neg_credit"]) for x in batch], dtype=torch.float32),
        "decoder_gate": torch.tensor([float(x["decoder_gate"]) for x in batch], dtype=torch.float32),
        "veto_gate": torch.tensor([float(x["veto_gate"]) for x in batch], dtype=torch.float32),
        "history_pos_ratio": torch.tensor([float(x["history_pos_ratio"]) for x in batch], dtype=torch.float32),
        "history_neg_ratio": torch.tensor([float(x["history_neg_ratio"]) for x in batch], dtype=torch.float32),
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


def load_chain_rows(chain_path: Path, reader, sid_mapping_path: str, max_hist_items: int, max_rows: int, min_signal_mass: float) -> Tuple[List[Dict[str, Any]], int, int]:
    sid_df = pd.read_csv(str(sid_mapping_path))
    sid_depth_cfg = len([c for c in sid_df.columns if str(c).startswith("sid")])
    iid2sid_tok_cpu, _ = build_iid2sid_tokens(reader, sid_mapping_path, int(sid_depth_cfg), torch.device("cpu"))
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    rows: List[Dict[str, Any]] = []
    with chain_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            target_tokens = [int(x) for x in payload.get("selected_sid_tokens", [])]
            if len(target_tokens) != sid_depth:
                continue
            token_pos_credit = [float(x) for x in payload.get("phase7b_token_pos_credit", [])]
            token_neg_credit = [float(x) for x in payload.get("phase7b_token_neg_credit", [])]
            if len(token_pos_credit) != sid_depth or len(token_neg_credit) != sid_depth:
                continue
            history_items = [int(x) for x in payload.get("history_items", [])][-int(max_hist_items):]
            hist_tensor = torch.tensor(history_items, dtype=torch.long).view(1, -1)
            input_ids, attention_mask = build_history_tokens(hist_tensor, iid2sid_tok_cpu, int(max_hist_items), int(sid_depth))
            local_pos_mass = float(payload.get("phase7b_local_pos_mass", sum(token_pos_credit) + float(payload.get("phase7b_item_pos_credit", 0.0)) + float(payload.get("phase7b_page_item_pos_credit", 0.0))))
            local_neg_mass = float(payload.get("phase7b_local_neg_mass", sum(token_neg_credit) + float(payload.get("phase7b_item_neg_credit", 0.0)) + float(payload.get("phase7b_page_item_neg_credit", 0.0))))
            if max(local_pos_mass, local_neg_mass) < float(min_signal_mass):
                continue
            rows.append({
                "group": f"{int(payload['episode_id'])}",
                "input_ids": input_ids.squeeze(0).tolist(),
                "attention_mask": attention_mask.squeeze(0).tolist(),
                "target_tokens": target_tokens,
                "token_pos_credit": token_pos_credit,
                "token_neg_credit": token_neg_credit,
                "item_pos_credit": float(payload.get("phase7b_item_pos_credit", 0.0)),
                "item_neg_credit": float(payload.get("phase7b_item_neg_credit", 0.0)),
                "page_item_pos_credit": float(payload.get("phase7b_page_item_pos_credit", 0.0)),
                "page_item_neg_credit": float(payload.get("phase7b_page_item_neg_credit", 0.0)),
                "decoder_gate": float(payload.get("phase7b_decoder_gate", 0.0)),
                "veto_gate": float(payload.get("phase7b_veto_gate", 0.0)),
                "history_pos_ratio": float(payload.get("history_pos_ratio", 0.0)),
                "history_neg_ratio": float(payload.get("history_neg_ratio", 0.0)),
            })
    if not rows:
        raise ValueError(f"No usable rows in {chain_path}")
    vocab_size = int(iid2sid_tok_cpu.max().item()) + 1
    return rows, sid_depth, vocab_size


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


def score_all_residuals(head: TokenResidualActorHead, hidden: torch.Tensor) -> torch.Tensor:
    bsz, length, dim = hidden.shape
    scores = head.score_all_tokens(hidden.reshape(-1, dim))
    return scores.view(bsz, length, scores.shape[-1])


def compute_actor_loss(actor_logits: torch.Tensor, base_logits: torch.Tensor, target_tokens: torch.Tensor, token_pos_credit: torch.Tensor, item_pos_credit: torch.Tensor, page_item_pos_credit: torch.Tensor, decoder_gate: torch.Tensor, history_pos_ratio: torch.Tensor, *, actor_item_scale: float, actor_support_scale: float, actor_hist_pos_scale: float, actor_page_gate_scale: float, actor_kl_scale: float, credit_clip: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    actor_log_probs = torch.log_softmax(actor_logits, dim=-1)
    base_log_probs = torch.log_softmax(base_logits.detach(), dim=-1)
    actor_target_logp = actor_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    actor_target_prob = actor_target_logp.exp()
    base_target_prob = base_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).exp()
    if float(credit_clip) > 0.0:
        token_pos_credit = token_pos_credit.clamp(min=0.0, max=float(credit_clip))
        item_pos_credit = item_pos_credit.clamp(min=0.0, max=float(credit_clip))
        page_item_pos_credit = page_item_pos_credit.clamp(min=0.0, max=float(credit_clip))
    gate = 1.0 + float(actor_support_scale) * decoder_gate.unsqueeze(-1) + float(actor_hist_pos_scale) * history_pos_ratio.unsqueeze(-1) + float(actor_page_gate_scale) * page_item_pos_credit.unsqueeze(-1)
    pos_w = gate * (token_pos_credit + float(actor_item_scale) * item_pos_credit.unsqueeze(-1))
    pos_loss = -(pos_w * actor_target_logp).sum() / (pos_w.sum() + 1e-8)
    kl_loss = F.kl_div(actor_log_probs, base_log_probs.exp(), reduction="batchmean", log_target=False)
    loss = pos_loss + float(actor_kl_scale) * kl_loss
    return loss, {
        "actor_pos_loss": float(pos_loss.item()),
        "actor_kl_loss": float(kl_loss.item()),
        "actor_target_gain": float((actor_target_prob - base_target_prob).mean().item()),
        "actor_pos_weight": float(pos_w.mean().item()),
    }


def compute_veto_loss(actor_logits: torch.Tensor, veto_scores: torch.Tensor, target_tokens: torch.Tensor, token_neg_credit: torch.Tensor, item_neg_credit: torch.Tensor, page_item_neg_credit: torch.Tensor, veto_gate: torch.Tensor, history_neg_ratio: torch.Tensor, *, veto_item_scale: float, veto_page_scale: float, veto_support_scale: float, veto_hist_neg_scale: float, veto_anchor_scale: float, veto_train_scale: float, veto_margin_keep: float, credit_clip: float, residual_clip: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    base_log_probs = torch.log_softmax(actor_logits.detach(), dim=-1)
    if float(residual_clip) > 0.0:
        veto_scores = veto_scores.clamp(min=-float(residual_clip), max=float(residual_clip))
    veto_scores = veto_scores - veto_scores.mean(dim=-1, keepdim=True)
    combined_logits = base_log_probs - float(veto_train_scale) * veto_scores
    combined_log_probs = torch.log_softmax(combined_logits, dim=-1)
    combined_target_prob = combined_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).exp()
    base_target_prob = base_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).exp()
    if float(credit_clip) > 0.0:
        token_neg_credit = token_neg_credit.clamp(min=0.0, max=float(credit_clip))
        item_neg_credit = item_neg_credit.clamp(min=0.0, max=float(credit_clip))
        page_item_neg_credit = page_item_neg_credit.clamp(min=0.0, max=float(credit_clip))
    neg_gate = 1.0 + float(veto_support_scale) * veto_gate.unsqueeze(-1) + float(veto_hist_neg_scale) * history_neg_ratio.unsqueeze(-1)
    neg_w = neg_gate * (token_neg_credit + float(veto_item_scale) * item_neg_credit.unsqueeze(-1) + float(veto_page_scale) * page_item_neg_credit.unsqueeze(-1))
    margin = torch.relu(combined_target_prob - float(veto_margin_keep) * base_target_prob)
    veto_margin_loss = (neg_w * margin).sum() / (neg_w.sum() + 1e-8)
    veto_prob_loss = (neg_w * combined_target_prob).sum() / (neg_w.sum() + 1e-8)
    anchor_loss = veto_scores.pow(2).mean()
    loss = veto_margin_loss + veto_prob_loss + float(veto_anchor_scale) * anchor_loss
    return loss, {
        "veto_margin_loss": float(veto_margin_loss.item()),
        "veto_prob_loss": float(veto_prob_loss.item()),
        "veto_anchor_loss": float(anchor_loss.item()),
        "veto_target_drop": float((base_target_prob - combined_target_prob).mean().item()),
        "veto_neg_weight": float(neg_w.mean().item()),
    }


def forward_dual(actor_tiger, base_tiger, veto_head: TokenResidualActorHead, batch: Dict[str, torch.Tensor], device: torch.device, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_tokens = batch["target_tokens"].to(device)
    decoder_input_ids = decoder_input_ids_from_targets(target_tokens)
    with torch.no_grad():
        base_logits, _ = base_tiger.decode_with_hidden(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    actor_logits, actor_hidden = actor_tiger.decode_with_hidden(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    actor_loss, actor_stats = compute_actor_loss(actor_logits, base_logits, target_tokens, batch["token_pos_credit"].to(device), batch["item_pos_credit"].to(device), batch["page_item_pos_credit"].to(device), batch["decoder_gate"].to(device), batch["history_pos_ratio"].to(device), actor_item_scale=float(args.actor_item_scale), actor_support_scale=float(args.actor_support_scale), actor_hist_pos_scale=float(args.actor_hist_pos_scale), actor_page_gate_scale=float(args.actor_page_gate_scale), actor_kl_scale=float(args.actor_kl_scale), credit_clip=float(args.credit_clip))
    veto_scores = score_all_residuals(veto_head, actor_hidden.detach())
    veto_loss, veto_stats = compute_veto_loss(actor_logits, veto_scores, target_tokens, batch["token_neg_credit"].to(device), batch["item_neg_credit"].to(device), batch["page_item_neg_credit"].to(device), batch["veto_gate"].to(device), batch["history_neg_ratio"].to(device), veto_item_scale=float(args.veto_item_scale), veto_page_scale=float(args.veto_page_scale), veto_support_scale=float(args.veto_support_scale), veto_hist_neg_scale=float(args.veto_hist_neg_scale), veto_anchor_scale=float(args.veto_anchor_scale), veto_train_scale=float(args.veto_train_scale), veto_margin_keep=float(args.veto_margin_keep), credit_clip=float(args.credit_clip), residual_clip=float(args.residual_clip))
    loss = actor_loss + veto_loss
    stats = {"loss": float(loss.item())}
    stats.update(actor_stats)
    stats.update(veto_stats)
    return loss, stats


@torch.no_grad()
def evaluate_offline(actor_tiger, base_tiger, veto_head, loader: DataLoader, device: torch.device, args) -> Dict[str, float]:
    actor_tiger.eval()
    veto_head.eval()
    metrics: Dict[str, List[float]] = {}
    for batch in loader:
        _, stats = forward_dual(actor_tiger, base_tiger, veto_head, batch, device, args)
        for key, value in stats.items():
            metrics.setdefault(key, []).append(float(value))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def parse_eval_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    patterns = {
        "total_reward": r"Total Reward:\s*([0-9.\-]+)",
        "depth": r"Depth:\s*([0-9.\-]+)",
        "avg_reward": r"Average reward:\s*([0-9.\-]+)",
        "coverage": r"Coverage:\s*([0-9.\-]+)",
        "click": r"is_click:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "long_view": r"long_view:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def save_veto_head(head: TokenResidualActorHead, head_path: Path, meta_path: Path, *, hidden_size: int, vocab_size: int, sid_depth: int, token_dim: int, mlp_dim: int) -> None:
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, head_path)
    write_json(meta_path, {"method": "TIGER Phase7b Dual Veto Head", "hidden_size": int(hidden_size), "vocab_size": int(vocab_size), "sid_depth": int(sid_depth), "token_dim": int(token_dim), "mlp_dim": int(mlp_dim)})


def run_online_eval(actor_ckpt: Path, veto_head_path: Path, veto_meta_path: Path, args) -> Dict[str, float]:
    device = str(args.online_eval_device).strip() or str(args.device)
    cmd = [str(args.python_exe), str(EVAL_SCRIPT), "--tiger_ckpt", str(actor_ckpt), "--sid_mapping_path", str(args.sid_mapping_path), "--uirm_log_path", str(args.uirm_log_path), "--slate_size", str(args.slate_size), "--episode_batch_size", str(args.online_eval_batch_size), "--model_size", str(args.model_size), "--num_episodes", str(args.online_eval_episodes), "--max_steps_per_episode", str(args.max_steps_per_episode), "--max_step_per_episode", str(args.max_steps_per_episode), "--beam_width", str(args.beam_width), "--single_response", "--seed", str(args.seed), "--max_hist_items", str(args.max_hist_items), "--device", device, "--phase2_blend_scale", "0.0", "--phase5_token_actor_head_path", str(veto_head_path), "--phase5_token_actor_meta_path", str(veto_meta_path), "--phase5_token_actor_scale", str(-abs(float(args.eval_veto_scale)))]
    if str(args.phase6_joint_head_path).strip():
        cmd.extend(["--phase6_joint_head_path", str(args.phase6_joint_head_path), "--phase6_joint_meta_path", str(args.phase6_joint_meta_path), "--phase6_prefix_scale", str(args.phase6_prefix_scale), "--phase6_token_actor_scale", str(args.phase6_token_actor_scale)])
    proc = subprocess.run(cmd, cwd=str(CODE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
    metrics = parse_eval_metrics(proc.stdout or "")
    metrics["return_code"] = float(proc.returncode)
    return metrics


def main() -> int:
    args = parse_args()
    utils.set_random_seed(int(args.seed))
    device = torch.device(str(args.device))
    reader = load_reader_from_uirm_log(str(args.uirm_log_path), str(device))
    rows, sid_depth, vocab_size = load_chain_rows(Path(args.chain_path), reader, str(args.sid_mapping_path), int(args.max_hist_items), int(args.max_rows), float(args.min_signal_mass))

    size_cfg = infer_model_size_args(str(args.model_size))
    base_tiger, _, _ = load_tiger_model(tiger_ckpt=str(args.tiger_ckpt), sid_mapping_path=str(args.sid_mapping_path), num_layers=int(size_cfg["num_layers"]), num_decoder_layers=int(size_cfg["num_decoder_layers"]), d_model=int(size_cfg["d_model"]), d_ff=int(size_cfg["d_ff"]), num_heads=int(size_cfg["num_heads"]), d_kv=int(size_cfg["d_kv"]), dropout_rate=0.1, feed_forward_proj="relu", device=device)
    for p in base_tiger.parameters():
        p.requires_grad = False
    base_tiger.eval()

    init_ckpt = str(args.init_tiger_ckpt).strip() or str(args.tiger_ckpt)
    actor_tiger, _, _ = load_tiger_model(tiger_ckpt=str(init_ckpt), sid_mapping_path=str(args.sid_mapping_path), num_layers=int(size_cfg["num_layers"]), num_decoder_layers=int(size_cfg["num_decoder_layers"]), d_model=int(size_cfg["d_model"]), d_ff=int(size_cfg["d_ff"]), num_heads=int(size_cfg["num_heads"]), d_kv=int(size_cfg["d_kv"]), dropout_rate=0.1, feed_forward_proj="relu", device=device)
    n_trainable = set_train_scope(actor_tiger, str(args.train_scope))

    veto_head = TokenResidualActorHead(hidden_size=int(size_cfg["d_model"]), vocab_size=int(vocab_size), token_dim=int(args.token_dim), mlp_dim=int(args.mlp_dim)).to(device)

    dataset = Phase7bDualDataset(rows)
    groups = [x["group"] for x in rows]
    train_idx, valid_idx = split_groups(groups, float(args.valid_ratio), int(args.seed))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_rows)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_rows)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in actor_tiger.parameters() if p.requires_grad], "lr": float(args.lr)},
        {"params": [p for p in veto_head.parameters() if p.requires_grad], "lr": float(args.veto_lr)},
    ], weight_decay=float(args.weight_decay))

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.tiger_ckpt).resolve().parent / "phase7b_dual"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_actor_state = None
    best_veto_state = None
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    best_score = -1e18

    for epoch in range(1, int(args.epochs) + 1):
        actor_tiger.train()
        veto_head.train()
        train_losses: List[float] = []
        for batch in train_loader:
            loss, _stats = forward_dual(actor_tiger, base_tiger, veto_head, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_metrics = evaluate_offline(actor_tiger, base_tiger, veto_head, valid_loader, device, args)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0

        temp_actor_path = save_dir / f"temp_epoch_{epoch:02d}_actor.pth"
        temp_veto_path = save_dir / f"temp_epoch_{epoch:02d}_veto.pt"
        temp_veto_meta = save_dir / f"temp_epoch_{epoch:02d}_veto_meta.json"
        torch.save({k: v.detach().cpu() for k, v in actor_tiger.state_dict().items()}, temp_actor_path)
        save_veto_head(veto_head, temp_veto_path, temp_veto_meta, hidden_size=int(size_cfg["d_model"]), vocab_size=int(vocab_size), sid_depth=int(sid_depth), token_dim=int(args.token_dim), mlp_dim=int(args.mlp_dim))

        if int(args.online_eval_episodes) > 0:
            online_metrics = run_online_eval(temp_actor_path, temp_veto_path, temp_veto_meta, args)
            valid_metrics["online_total_reward"] = float(online_metrics.get("total_reward", 0.0))
            valid_metrics["online_click"] = float(online_metrics.get("click", 0.0))
            valid_metrics["online_long_view"] = float(online_metrics.get("long_view", 0.0))

        score = float(valid_metrics.get("online_total_reward", -valid_metrics["loss"]))
        history.append(dict(valid_metrics))
        if score > best_score:
            best_score = score
            best_epoch = int(epoch)
            best_actor_state = {k: v.detach().cpu() for k, v in actor_tiger.state_dict().items()}
            best_veto_state = {k: v.detach().cpu() for k, v in veto_head.state_dict().items()}
            best_metrics = dict(valid_metrics)

        print(f"[epoch {epoch}] train_loss={valid_metrics['train_loss']:.4f} valid_loss={valid_metrics['loss']:.4f} actor_gain={valid_metrics.get('actor_target_gain', 0.0):.4f} veto_drop={valid_metrics.get('veto_target_drop', 0.0):.4f} online_reward={valid_metrics.get('online_total_reward', 0.0):.4f}")

    if best_actor_state is None or best_veto_state is None:
        raise RuntimeError("Phase7b training produced no checkpoint.")

    actor_path = save_dir / "phase7b_actor_tiger.pth"
    veto_path = save_dir / "phase7b_veto_head.pt"
    veto_meta_path = save_dir / "phase7b_veto_meta.json"
    meta_path = save_dir / "phase7b_dual_meta.json"
    metrics_path = Path(args.metrics_out) if args.metrics_out else save_dir / "phase7b_dual_metrics.json"

    torch.save(best_actor_state, actor_path)
    torch.save({"model_state_dict": best_veto_state}, veto_path)
    write_json(veto_meta_path, {"method": "TIGER Phase7b Dual Veto Head", "hidden_size": int(size_cfg["d_model"]), "vocab_size": int(vocab_size), "sid_depth": int(sid_depth), "token_dim": int(args.token_dim), "mlp_dim": int(args.mlp_dim)})

    meta = {
        "method": "TIGER Phase7b Dual Channel",
        "chain_path": str(Path(args.chain_path).resolve()),
        "base_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "init_tiger_ckpt": str(Path(init_ckpt).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "sid_depth": int(sid_depth),
        "vocab_size": int(vocab_size),
        "train_scope": str(args.train_scope),
        "n_trainable": int(n_trainable),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "n_rows": int(len(rows)),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "online_eval_episodes": int(args.online_eval_episodes),
        "phase6_joint_head_path": str(Path(args.phase6_joint_head_path).resolve()) if str(args.phase6_joint_head_path).strip() else "",
    }
    write_json(meta_path, meta)
    write_json(metrics_path, {"actor_ckpt": str(actor_path.resolve()), "veto_head_path": str(veto_path.resolve()), "veto_meta_path": str(veto_meta_path.resolve()), "meta_path": str(meta_path.resolve()), "best_epoch": int(best_epoch), "best_metrics": best_metrics, "history": history})
    print(f"[phase7b] saved actor to {actor_path}")
    print(f"[phase7b] saved veto head to {veto_path}")
    print(f"[phase7b] saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
