import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from tiger_slate_allocator_common import (
    compute_item_support_features,
    prepare_history_sid,
)


def build_online_slate_inputs(
    *,
    history_items: Sequence[int],
    candidate_item_ids: Sequence[int],
    candidate_sid_tokens_list: Sequence[Sequence[int]] | None,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    token_vocab_size: int,
    base_scores: Sequence[float] | None = None,
) -> Dict[str, Any]:
    hist_items, history_sid = prepare_history_sid(history_items, iid2sid_tok_cpu, max_hist_items)
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    token_scale = float(max(int(token_vocab_size) - 1, 1))
    item_ids = [int(x) for x in candidate_item_ids]
    n_items = len(item_ids)
    if base_scores is None:
        base_scores_arr = np.zeros((n_items,), dtype=np.float32)
    else:
        base_scores_arr = np.asarray(list(base_scores), dtype=np.float32).reshape(-1)
        if base_scores_arr.shape[0] != n_items:
            base_scores_arr = np.zeros((n_items,), dtype=np.float32)

    sid_tokens_list: List[List[int]] = []
    for idx, iid in enumerate(item_ids):
        if candidate_sid_tokens_list is not None and idx < len(candidate_sid_tokens_list):
            tokens = [int(x) for x in candidate_sid_tokens_list[idx]]
        elif 0 <= int(iid) < int(iid2sid_tok_cpu.shape[0]):
            tokens = [int(x) for x in iid2sid_tok_cpu[int(iid)].tolist()]
        else:
            tokens = []
        if len(tokens) != sid_depth:
            tokens = [0 for _ in range(sid_depth)]
        sid_tokens_list.append(tokens)

    support_features: List[np.ndarray] = []
    support_strengths: List[float] = []
    for tokens in sid_tokens_list:
        feat, strength = compute_item_support_features(history_sid, tokens)
        support_features.append(feat.astype(np.float32))
        support_strengths.append(float(strength))
    support_strengths_arr = np.asarray(support_strengths, dtype=np.float32)

    if base_scores_arr.size > 0 and float(np.std(base_scores_arr)) > 1e-8:
        score_norm = (base_scores_arr - float(np.mean(base_scores_arr))) / float(np.std(base_scores_arr))
    else:
        score_norm = np.zeros_like(base_scores_arr, dtype=np.float32)

    item_features: List[np.ndarray] = []
    for idx in range(n_items):
        pos_norm = float(idx / max(n_items - 1, 1))
        token_feat = np.asarray(sid_tokens_list[idx], dtype=np.float32) / token_scale
        item_feat = np.concatenate(
            [
                np.asarray(
                    [
                        float(score_norm[idx]),
                        float(base_scores_arr[idx]),
                        float(support_strengths_arr[idx]),
                        float(pos_norm),
                    ],
                    dtype=np.float32,
                ),
                support_features[idx],
                token_feat,
            ],
            axis=0,
        )
        item_features.append(item_feat.astype(np.float32))

    if support_strengths_arr.size > 0:
        page_features = np.asarray(
            [
                float(np.mean(support_strengths_arr)),
                float(np.max(support_strengths_arr)),
                float(np.min(support_strengths_arr)),
                float(np.std(support_strengths_arr)),
                float(min(len(hist_items), int(max_hist_items)) / max(int(max_hist_items), 1)),
                float(n_items / max(n_items, 1)),
                float(np.mean(score_norm)) if score_norm.size > 0 else 0.0,
                float(np.max(score_norm)) if score_norm.size > 0 else 0.0,
            ],
            dtype=np.float32,
        )
    else:
        page_features = np.zeros((8,), dtype=np.float32)

    return {
        "history_items": hist_items,
        "history_sid": history_sid,
        "sid_tokens_list": sid_tokens_list,
        "item_features": np.stack(item_features, axis=0) if item_features else np.zeros((0, 0), dtype=np.float32),
        "page_features": page_features.astype(np.float32),
        "support_strengths": support_strengths_arr.astype(np.float32),
        "base_scores": base_scores_arr.astype(np.float32),
    }


class OnlineSlateAllocatorHead(nn.Module):
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

    def forward(self, item_features: torch.Tensor, page_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        item_hidden = self.item_mlp(item_features)
        page_hidden = self.page_mlp(page_features).unsqueeze(1).expand(-1, item_hidden.size(1), -1)
        logits = self.scorer(torch.cat([item_hidden, page_hidden], dim=-1)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits

    def predict_shares(self, item_features: torch.Tensor, page_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(item_features, page_features, mask=mask)
        return torch.softmax(logits, dim=-1)


class SlateValueHead(nn.Module):
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
        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, item_features: torch.Tensor, page_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        item_hidden = self.item_mlp(item_features)
        if mask is None:
            mean_pool = item_hidden.mean(dim=1)
            max_pool = item_hidden.max(dim=1).values
        else:
            mask_f = mask.float().unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean_pool = (item_hidden * mask_f).sum(dim=1) / denom
            fill = torch.full_like(item_hidden, -1e9)
            max_pool = torch.where(mask.unsqueeze(-1), item_hidden, fill).max(dim=1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        page_hidden = self.page_mlp(page_features)
        return self.value(torch.cat([mean_pool, max_pool, page_hidden], dim=-1)).squeeze(-1)


def load_online_slate_allocator(head_path: str, meta_path: str, device: torch.device) -> Tuple[OnlineSlateAllocatorHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = OnlineSlateAllocatorHead(
        item_dim=int(meta["item_dim"]),
        page_dim=int(meta["page_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 96)),
        dropout=float(meta.get("dropout", 0.10)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta


def load_slate_value_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[SlateValueHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = SlateValueHead(
        item_dim=int(meta["item_dim"]),
        page_dim=int(meta["page_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 96)),
        dropout=float(meta.get("dropout", 0.10)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta
