import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def allocate_item_shares_heuristic(page_credit: float, item_rewards: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(list(item_rewards), dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    centered = rewards - float(rewards.mean())
    if float(page_credit) >= 0.0:
        basis = np.maximum(centered, 0.0)
        if float(basis.sum()) <= 1e-8:
            basis = np.maximum(rewards, 0.0)
    else:
        basis = np.maximum(-centered, 0.0)
    if float(basis.sum()) <= 1e-8:
        basis = np.ones_like(rewards, dtype=np.float32)
    share = basis / max(float(basis.sum()), 1e-8)
    return share.astype(np.float32), centered.astype(np.float32)


def longest_prefix_ratio(hist_sid: np.ndarray, target_sid: np.ndarray, end_idx: int) -> float:
    width = int(end_idx) + 1
    if width <= 0:
        return 0.0
    match = float(np.mean(hist_sid[:width] == target_sid[:width]))
    return float(np.clip(match, 0.0, 1.0))


def aggregate_support(affinity: np.ndarray) -> float:
    if affinity.size == 0:
        return 0.0
    return float(np.mean(np.max(affinity, axis=0)))


def _normalize_positive(scores: Sequence[float], fallback: Sequence[float] | None = None) -> np.ndarray:
    vals = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    vals = np.maximum(vals, 0.0)
    if float(vals.sum()) <= 1e-8 and fallback is not None:
        vals = np.asarray(list(fallback), dtype=np.float32).reshape(-1)
        vals = np.maximum(vals, 0.0)
    if float(vals.sum()) <= 1e-8:
        vals = np.ones_like(vals, dtype=np.float32)
    return vals / max(float(vals.sum()), 1e-8)


def response_strengths(
    responses: Sequence[Sequence[float]],
    response_weights: Sequence[float] | None = None,
) -> np.ndarray:
    resp = np.asarray(list(responses), dtype=np.float32)
    if resp.ndim != 2:
        return np.zeros((0,), dtype=np.float32)
    if response_weights is not None:
        weights = np.asarray(list(response_weights), dtype=np.float32).reshape(-1)
        if resp.shape[1] == weights.shape[0]:
            return (resp * weights.reshape(1, -1)).sum(axis=1).astype(np.float32)
    return resp.sum(axis=1).astype(np.float32)


def prepare_history_sid(
    history_items: Sequence[int],
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
) -> Tuple[List[int], np.ndarray]:
    hist = [int(i) for i in history_items if 0 <= int(i) < int(iid2sid_tok_cpu.shape[0])]
    hist = hist[-int(max_hist_items):]
    if not hist:
        return [], np.zeros((0, int(iid2sid_tok_cpu.shape[1])), dtype=np.int64)
    hist_tensor = torch.tensor(hist, dtype=torch.long)
    hist_sid = iid2sid_tok_cpu[hist_tensor].cpu().numpy().astype(np.int64)
    valid_mask = np.any(hist_sid > 0, axis=1)
    hist_sid = hist_sid[valid_mask]
    hist = [hist[idx] for idx in range(len(hist)) if bool(valid_mask[idx])]
    return hist, hist_sid


def compute_item_support_features(history_sid: np.ndarray, target_tokens: Sequence[int]) -> Tuple[np.ndarray, float]:
    target = np.asarray(list(target_tokens), dtype=np.int64).reshape(-1)
    depth = int(target.shape[0])
    if depth <= 0:
        return np.zeros((4,), dtype=np.float32), 0.0
    if history_sid.size == 0:
        return np.zeros((2 * depth + 4,), dtype=np.float32), 0.0
    n_hist = int(history_sid.shape[0])
    affinity = np.zeros((n_hist, depth), dtype=np.float32)
    for ridx in range(n_hist):
        for cidx in range(depth):
            affinity[ridx, cidx] = longest_prefix_ratio(history_sid[ridx], target, cidx)
    best_by_block = np.max(affinity, axis=0).astype(np.float32)
    mean_by_block = np.mean(affinity, axis=0).astype(np.float32)
    overall = float(aggregate_support(affinity))
    exact_match = float(np.mean(np.all(history_sid[:, :depth] == target.reshape(1, -1), axis=1)))
    final_block = float(best_by_block[-1]) if depth > 0 else 0.0
    hist_norm = float(min(n_hist, 50) / 50.0)
    feat = np.concatenate(
        [
            best_by_block,
            mean_by_block,
            np.asarray([overall, final_block, exact_match, hist_norm], dtype=np.float32),
        ],
        axis=0,
    )
    return feat.astype(np.float32), overall


def build_bootstrap_target_shares(
    *,
    page_credit: float,
    item_rewards: Sequence[float],
    responses: Sequence[Sequence[float]],
    support_strengths: Sequence[float],
    response_weights: Sequence[float] | None = None,
    heuristic_mix: float = 0.60,
    support_mix: float = 0.25,
    response_mix: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heuristic_shares, centered_rewards = allocate_item_shares_heuristic(float(page_credit), item_rewards)
    resp_strength = response_strengths(responses, response_weights=response_weights)
    support = np.asarray(list(support_strengths), dtype=np.float32).reshape(-1)
    if support.shape[0] != heuristic_shares.shape[0]:
        support = np.zeros_like(heuristic_shares)

    if float(page_credit) >= 0.0:
        support_basis = support
        response_basis = np.maximum(resp_strength, 0.0)
    else:
        support_basis = np.maximum(1.0 - support, 0.0)
        response_basis = np.maximum(-resp_strength, 0.0)

    support_shares = _normalize_positive(support_basis, fallback=heuristic_shares)
    response_shares = _normalize_positive(response_basis, fallback=heuristic_shares)
    mix = (
        float(heuristic_mix) * heuristic_shares
        + float(support_mix) * support_shares
        + float(response_mix) * response_shares
    )
    target_shares = _normalize_positive(mix, fallback=heuristic_shares)
    return (
        target_shares.astype(np.float32),
        heuristic_shares.astype(np.float32),
        centered_rewards.astype(np.float32),
        resp_strength.astype(np.float32),
    )


def build_page_context_features(
    *,
    page_credit: float,
    slate_return_raw: float,
    step_reward: float,
    item_rewards: Sequence[float],
    history_len: int,
    page_index: int,
    max_hist_items: int,
) -> np.ndarray:
    rewards = np.asarray(list(item_rewards), dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        rewards = np.zeros((1,), dtype=np.float32)
    page_feat = np.asarray(
        [
            float(page_credit),
            float(slate_return_raw),
            float(step_reward),
            float(rewards.mean()),
            float(rewards.max()),
            float(rewards.min()),
            float(np.mean(rewards > 0.0)),
            float(np.mean(rewards != 0.0)),
            float(min(int(history_len), int(max_hist_items)) / max(int(max_hist_items), 1)),
            float(min(int(page_index), 20) / 20.0),
        ],
        dtype=np.float32,
    )
    return page_feat


def build_slate_allocator_inputs(
    *,
    history_items: Sequence[int],
    selected_item_ids: Sequence[int],
    selected_sid_tokens_list: Sequence[Sequence[int]],
    selected_responses: Sequence[Sequence[float]],
    selected_item_rewards: Sequence[float],
    response_weights: Sequence[float] | None,
    page_credit: float,
    slate_return_raw: float,
    step_reward: float,
    page_index: int,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    token_vocab_size: int,
    heuristic_mix: float = 0.60,
    support_mix: float = 0.25,
    response_mix: float = 0.15,
) -> Dict[str, Any]:
    hist_items, history_sid = prepare_history_sid(history_items, iid2sid_tok_cpu, max_hist_items)
    item_ids = [int(x) for x in selected_item_ids]
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    token_scale = float(max(int(token_vocab_size) - 1, 1))

    sid_tokens_list: List[List[int]] = []
    for idx, iid in enumerate(item_ids):
        if idx < len(selected_sid_tokens_list):
            tokens = [int(x) for x in selected_sid_tokens_list[idx]]
        elif 0 <= int(iid) < int(iid2sid_tok_cpu.shape[0]):
            tokens = [int(x) for x in iid2sid_tok_cpu[int(iid)].tolist()]
        else:
            tokens = []
        if len(tokens) != sid_depth:
            tokens = [0 for _ in range(sid_depth)]
        sid_tokens_list.append(tokens)

    responses_arr = np.asarray(list(selected_responses), dtype=np.float32)
    if responses_arr.ndim != 2:
        k_fb = len(list(response_weights)) if response_weights is not None else 7
        responses_arr = np.zeros((len(item_ids), k_fb), dtype=np.float32)
    if len(selected_item_rewards) != len(item_ids):
        rewards = np.zeros((len(item_ids),), dtype=np.float32)
    else:
        rewards = np.asarray(list(selected_item_rewards), dtype=np.float32)

    support_feats: List[np.ndarray] = []
    support_strengths: List[float] = []
    for tokens in sid_tokens_list:
        feat, strength = compute_item_support_features(history_sid, tokens)
        support_feats.append(feat)
        support_strengths.append(float(strength))
    support_strength_arr = np.asarray(support_strengths, dtype=np.float32)

    target_shares, heuristic_shares, centered_rewards, resp_strength = build_bootstrap_target_shares(
        page_credit=float(page_credit),
        item_rewards=rewards.tolist(),
        responses=responses_arr.tolist(),
        support_strengths=support_strength_arr.tolist(),
        response_weights=response_weights,
        heuristic_mix=float(heuristic_mix),
        support_mix=float(support_mix),
        response_mix=float(response_mix),
    )

    page_features = build_page_context_features(
        page_credit=float(page_credit),
        slate_return_raw=float(slate_return_raw),
        step_reward=float(step_reward),
        item_rewards=rewards.tolist(),
        history_len=len(hist_items),
        page_index=int(page_index),
        max_hist_items=int(max_hist_items),
    )

    item_features: List[np.ndarray] = []
    for idx, iid in enumerate(item_ids):
        position_norm = float(idx / max(len(item_ids) - 1, 1))
        token_feat = np.asarray(sid_tokens_list[idx], dtype=np.float32) / token_scale
        support_feat = support_feats[idx]
        item_feat = np.concatenate(
            [
                np.asarray(
                    [
                        float(rewards[idx]),
                        float(centered_rewards[idx]),
                        float(abs(centered_rewards[idx])),
                        float(resp_strength[idx]),
                        position_norm,
                        float(1.0 if rewards[idx] > float(rewards.mean()) else 0.0),
                    ],
                    dtype=np.float32,
                ),
                responses_arr[idx].astype(np.float32),
                support_feat.astype(np.float32),
                token_feat.astype(np.float32),
            ],
            axis=0,
        )
        item_features.append(item_feat.astype(np.float32))

    return {
        "history_items": hist_items,
        "history_sid": history_sid,
        "item_features": np.stack(item_features, axis=0) if item_features else np.zeros((0, 0), dtype=np.float32),
        "page_features": page_features.astype(np.float32),
        "target_shares": target_shares.astype(np.float32),
        "heuristic_shares": heuristic_shares.astype(np.float32),
        "centered_rewards": centered_rewards.astype(np.float32),
        "response_strengths": resp_strength.astype(np.float32),
        "support_strengths": support_strength_arr.astype(np.float32),
        "sid_tokens_list": sid_tokens_list,
        "item_rewards": rewards.astype(np.float32),
        "responses": responses_arr.astype(np.float32),
    }


class SlateItemAllocator(nn.Module):
    def __init__(
        self,
        *,
        item_dim: int,
        page_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.item_dim = int(item_dim)
        self.page_dim = int(page_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.item_mlp = nn.Sequential(
            nn.Linear(self.item_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.page_mlp = nn.Sequential(
            nn.Linear(self.page_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        item_features: torch.Tensor,
        page_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_hidden = self.item_mlp(item_features)
        page_hidden = self.page_mlp(page_features).unsqueeze(1).expand(-1, item_hidden.size(1), -1)
        logits = self.scorer(torch.cat([item_hidden, page_hidden], dim=-1)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits

    def predict_shares(
        self,
        item_features: torch.Tensor,
        page_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(item_features, page_features, mask=mask)
        logits = logits / max(float(temperature), 1e-6)
        return torch.softmax(logits, dim=-1)


def load_slate_allocator(
    head_path: str,
    meta_path: str,
    device: torch.device,
) -> Tuple[SlateItemAllocator, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    model = SlateItemAllocator(
        item_dim=int(meta["item_dim"]),
        page_dim=int(meta["page_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 96)),
        dropout=float(meta.get("dropout", 0.10)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, meta
