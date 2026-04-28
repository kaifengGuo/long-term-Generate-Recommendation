import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import utils  # noqa: E402
from reader import *  # noqa: F401,F403,E402

from tiger_phase2_blend_common import build_history_tokens  # noqa: E402


def load_reader_from_uirm_log(uirm_log_path: str, device: str):
    with open(uirm_log_path, "r", encoding="utf-8") as infile:
        class_args = eval(infile.readline(), {"Namespace": Namespace})
        training_args = eval(infile.readline(), {"Namespace": Namespace})
    training_args.val_holdout_per_user = 0
    training_args.test_holdout_per_user = 0
    training_args.device = device
    reader_class = eval("{0}.{0}".format(class_args.reader))
    return reader_class(training_args)


def set_random_seed(seed: int) -> None:
    utils.set_random_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(str(x) for x in groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, g in enumerate(groups):
        (valid_idx if str(g) in valid_groups else train_idx).append(idx)
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_history_state(
    history_items: Sequence[int],
    *,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    sid_depth: int,
) -> Tuple[List[int], List[int]]:
    hist_tensor = torch.tensor([int(x) for x in history_items][-int(max_hist_items):], dtype=torch.long).view(1, -1)
    input_ids, attention_mask = build_history_tokens(
        hist_tensor,
        iid2sid_tok_cpu,
        int(max_hist_items),
        int(sid_depth),
    )
    return input_ids.squeeze(0).tolist(), attention_mask.squeeze(0).tolist()


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


def longest_prefix_ratio(hist_sid: np.ndarray, target_sid: np.ndarray, end_idx: int) -> float:
    width = int(end_idx) + 1
    if width <= 0:
        return 0.0
    match = float(np.mean(hist_sid[:width] == target_sid[:width]))
    return float(np.clip(match, 0.0, 1.0))


def compute_support_strength(history_sid: np.ndarray, target_tokens: Sequence[int]) -> float:
    target = np.asarray(list(target_tokens), dtype=np.int64).reshape(-1)
    if target.size == 0 or history_sid.size == 0:
        return 0.0
    affinity = np.zeros((history_sid.shape[0], target.shape[0]), dtype=np.float32)
    for ridx in range(history_sid.shape[0]):
        for cidx in range(target.shape[0]):
            affinity[ridx, cidx] = longest_prefix_ratio(history_sid[ridx], target, cidx)
    support_mean = float(np.mean(np.max(affinity, axis=0))) if affinity.size > 0 else 0.0
    support_full = float(np.max(affinity[:, -1])) if affinity.shape[1] > 0 else 0.0
    return float(0.5 * support_mean + 0.5 * support_full)


def weighted_response_strength(selected_response: Sequence[float], reward_weights: Sequence[float]) -> float:
    resp = np.asarray(list(selected_response), dtype=np.float32).reshape(-1)
    weights = np.asarray(list(reward_weights), dtype=np.float32).reshape(-1)
    width = min(int(resp.shape[0]), int(weights.shape[0]))
    if width <= 0:
        return 0.0
    return float(np.dot(resp[:width], weights[:width]))


def normalize_scores(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= float(eps):
        arr = np.ones_like(arr, dtype=np.float32)
        total = float(arr.sum())
    return arr / max(total, float(eps))


def build_page_features(
    *,
    page_index: int,
    max_page_index: int,
    history_len: int,
    max_hist_items: int,
    slate_size: int,
    max_slate_size: int,
    selected_item_rewards: Sequence[float],
    selected_responses: Sequence[Sequence[float]],
    step_reward: float,
    done: bool,
    response_strengths: Sequence[float],
    support_strengths: Sequence[float],
) -> Tuple[np.ndarray, List[str]]:
    rewards = np.asarray(list(selected_item_rewards), dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        rewards = np.zeros((1,), dtype=np.float32)
    resp_strengths = np.asarray(list(response_strengths), dtype=np.float32).reshape(-1)
    support_arr = np.asarray(list(support_strengths), dtype=np.float32).reshape(-1)
    if resp_strengths.size == 0:
        resp_strengths = np.zeros((1,), dtype=np.float32)
    if support_arr.size == 0:
        support_arr = np.zeros((1,), dtype=np.float32)
    resp = np.asarray(list(selected_responses), dtype=np.float32)
    resp_width = int(resp.shape[1]) if resp.ndim == 2 else 0

    feat = [
        float(min(int(page_index), int(max_page_index)) / max(int(max_page_index), 1)),
        float(min(int(history_len), int(max_hist_items)) / max(int(max_hist_items), 1)),
        float(min(int(slate_size), int(max_slate_size)) / max(int(max_slate_size), 1)),
        float(step_reward),
        float(rewards.mean()),
        float(rewards.max()),
        float(rewards.min()),
        float(np.mean(rewards > 0.0)),
        float(np.std(rewards)),
        float(resp_strengths.mean()),
        float(resp_strengths.max()),
        float(resp_strengths.min()),
        float(support_arr.mean()),
        float(support_arr.max()),
        float(support_arr.min()),
        float(1.0 if bool(done) else 0.0),
        float(resp_width),
    ]
    names = [
        "page_index_norm",
        "history_len_norm",
        "slate_size_norm",
        "step_reward",
        "item_reward_mean",
        "item_reward_max",
        "item_reward_min",
        "item_reward_pos_ratio",
        "item_reward_std",
        "response_strength_mean",
        "response_strength_max",
        "response_strength_min",
        "support_strength_mean",
        "support_strength_max",
        "support_strength_min",
        "done_flag",
        "response_width",
    ]
    return np.asarray(feat, dtype=np.float32), names


def build_item_features(
    *,
    slate_item_index: int,
    slate_size: int,
    item_reward: float,
    item_reward_centered: float,
    response_strength: float,
    response_strength_centered: float,
    support_strength: float,
    support_strength_centered: float,
    item_share: float,
    page_adv: float,
    done: bool,
) -> Tuple[np.ndarray, List[str]]:
    feat = [
        float(int(slate_item_index) / max(int(slate_size) - 1, 1)),
        float(int(slate_size)),
        float(item_reward),
        float(item_reward_centered),
        float(response_strength),
        float(response_strength_centered),
        float(support_strength),
        float(support_strength_centered),
        float(item_share),
        float(abs(float(page_adv))),
        float(1.0 if bool(done) else 0.0),
    ]
    names = [
        "slate_item_index_norm",
        "slate_size",
        "item_reward",
        "item_reward_centered",
        "response_strength",
        "response_strength_centered",
        "support_strength",
        "support_strength_centered",
        "item_share",
        "abs_page_adv",
        "done_flag",
    ]
    return np.asarray(feat, dtype=np.float32), names


def prefix_to_delta(prefix_values: torch.Tensor) -> torch.Tensor:
    if prefix_values.ndim != 2:
        raise ValueError("prefix_values must have shape [B, L]")
    first = prefix_values[:, :1]
    if prefix_values.size(1) <= 1:
        return first
    rest = prefix_values[:, 1:] - prefix_values[:, :-1]
    return torch.cat([first, rest], dim=1)


def calibrate_token_delta(delta: torch.Tensor, item_total: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 2:
        raise ValueError("delta must have shape [B, L]")
    item_total = item_total.reshape(-1)
    out = delta.clone()
    residual = item_total - out.sum(dim=-1)
    out[:, -1] = out[:, -1] + residual
    return out


def renorm_signal(x: torch.Tensor, mode: str) -> torch.Tensor:
    if str(mode) == "none":
        return x
    denom = x.abs().mean().clamp_min(1e-6)
    return x / denom


def build_sparse_mask(scores: torch.Tensor, topk: int, floor: float) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [B, L]")
    positive = scores > float(floor)
    if int(topk) <= 0:
        return torch.zeros_like(scores, dtype=torch.float32)
    if scores.shape[1] <= int(topk):
        return positive.float()
    k = min(int(topk), int(scores.shape[1]))
    idx = torch.topk(scores, k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return (mask & positive).float()


def ppo_clipped_surrogate(ratio: torch.Tensor, advantage: torch.Tensor, clip_eps: float) -> torch.Tensor:
    clipped = ratio.clamp(min=1.0 - float(clip_eps), max=1.0 + float(clip_eps))
    s1 = ratio * advantage
    s2 = clipped * advantage
    return torch.where(advantage >= 0.0, torch.minimum(s1, s2), torch.maximum(s1, s2))
