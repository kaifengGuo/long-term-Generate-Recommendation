import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_PRESET = "click_longview"
REWARD_PRESETS: Dict[str, Dict[str, float]] = {
    "click": {
        "is_click": 1.0,
    },
    "click_longview": {
        "is_click": 1.0,
        "long_view": 0.5,
    },
    "engagement": {
        "is_click": 1.0,
        "long_view": 0.5,
        "is_like": 0.2,
        "is_comment": 0.3,
        "is_forward": 0.3,
        "is_follow": 0.3,
        "is_hate": -0.5,
    },
}


def resolve_reward_weights(
    response_names: Sequence[str],
    *,
    preset: str = DEFAULT_PRESET,
    reward_weights_json: str = "",
) -> Tuple[np.ndarray, Dict[str, float]]:
    weights_map = {str(k): float(v) for k, v in REWARD_PRESETS.get(str(preset), {}).items()}
    if str(reward_weights_json).strip():
        payload = json.loads(str(reward_weights_json))
        for key, value in payload.items():
            weights_map[str(key)] = float(value)
    weights = np.asarray([float(weights_map.get(str(name), 0.0)) for name in response_names], dtype=np.float32)
    return weights, weights_map


def compute_welfare_step_reward(
    selected_responses: Sequence[Sequence[float]],
    reward_weights: Sequence[float],
) -> float:
    resp = np.asarray(list(selected_responses), dtype=np.float32)
    weights = np.asarray(list(reward_weights), dtype=np.float32).reshape(-1)
    if resp.ndim != 2 or resp.shape[0] == 0:
        return 0.0
    if resp.shape[1] != weights.shape[0]:
        width = min(int(resp.shape[1]), int(weights.shape[0]))
        if width <= 0:
            return 0.0
        resp = resp[:, :width]
        weights = weights[:width]
    per_item = (resp * weights.reshape(1, -1)).sum(axis=1)
    return float(per_item.mean())


def build_welfare_page_features(
    *,
    page_index: int,
    max_page_index: int,
    history_len: int,
    max_hist_items: int,
    slate_size: int,
    max_slate_size: int,
    selected_item_rewards: Sequence[float],
    selected_responses: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, List[str]]:
    rewards = np.asarray(list(selected_item_rewards), dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        rewards = np.zeros((1,), dtype=np.float32)
    resp = np.asarray(list(selected_responses), dtype=np.float32)
    if resp.ndim != 2:
        resp = np.zeros((max(int(slate_size), 1), 0), dtype=np.float32)

    feat: List[float] = [
        float(min(int(page_index), int(max_page_index)) / max(int(max_page_index), 1)),
        float(min(int(history_len), int(max_hist_items)) / max(int(max_hist_items), 1)),
        float(min(int(slate_size), int(max_slate_size)) / max(int(max_slate_size), 1)),
        float(rewards.mean()),
        float(rewards.max()),
        float(rewards.min()),
        float(np.mean(rewards > 0.0)),
        float(np.std(rewards)),
    ]
    names = [
        "page_index_norm",
        "history_len_norm",
        "slate_size_norm",
        "item_reward_mean",
        "item_reward_max",
        "item_reward_min",
        "item_reward_pos_ratio",
        "item_reward_std",
    ]

    if resp.size > 0:
        resp_mean = resp.mean(axis=0).astype(np.float32)
        resp_max = resp.max(axis=0).astype(np.float32)
        resp_min = resp.min(axis=0).astype(np.float32)
        for idx, value in enumerate(resp_mean.tolist()):
            feat.append(float(value))
            names.append(f"resp_mean_{idx}")
        for idx, value in enumerate(resp_max.tolist()):
            feat.append(float(value))
            names.append(f"resp_max_{idx}")
        for idx, value in enumerate(resp_min.tolist()):
            feat.append(float(value))
            names.append(f"resp_min_{idx}")
    return np.asarray(feat, dtype=np.float32), names


def aggregate_history_credit_features(history_credit: Sequence[float]) -> Dict[str, float]:
    vals = np.asarray(list(history_credit), dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return {
            "history_pos_mass": 0.0,
            "history_neg_mass": 0.0,
            "history_pos_ratio": 0.0,
            "history_neg_ratio": 0.0,
            "history_balance": 0.0,
        }
    pos = np.maximum(vals, 0.0)
    neg = np.maximum(-vals, 0.0)
    pos_mass = float(pos.sum())
    neg_mass = float(neg.sum())
    total = max(pos_mass + neg_mass, 1e-8)
    return {
        "history_pos_mass": pos_mass,
        "history_neg_mass": neg_mass,
        "history_pos_ratio": float(pos_mass / total),
        "history_neg_ratio": float(neg_mass / total),
        "history_balance": float((pos_mass - neg_mass) / total),
    }


class WelfareValueHead(nn.Module):
    def __init__(self, hidden_size: int, page_dim: int, *, mlp_dim: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.page_dim = int(page_dim)
        self.mlp_dim = int(mlp_dim)
        self.hist_norm = nn.LayerNorm(self.hidden_size)
        self.page_norm = nn.LayerNorm(self.page_dim)
        self.fc1 = nn.Linear(self.hidden_size + self.page_dim, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.out = nn.Linear(self.mlp_dim, 1)

    def forward(self, history_summary: torch.Tensor, page_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.hist_norm(history_summary), self.page_norm(page_features)], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return self.out(x).squeeze(-1)


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


def load_welfare_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[WelfareValueHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = WelfareValueHead(
        hidden_size=int(meta["hidden_size"]),
        page_dim=int(meta["page_dim"]),
        mlp_dim=int(meta.get("mlp_dim", 128)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta
