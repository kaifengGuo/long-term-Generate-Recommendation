import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from tiger_phase6_joint_common import SlateCreditHead


class ItemPrefixValueHead(nn.Module):
    """Predict cumulative page-internal value from an item prefix."""

    def __init__(
        self,
        *,
        item_dim: int,
        page_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        stats_dim: int = 3,
    ):
        super().__init__()
        self.item_dim = int(item_dim)
        self.page_dim = int(page_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.stats_dim = int(stats_dim)

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
            nn.Linear(self.hidden_dim * 4 + self.stats_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        item_features: torch.Tensor,
        page_features: torch.Tensor,
        *,
        mask: torch.Tensor,
        prefix_len: torch.Tensor,
        total_items: torch.Tensor,
    ) -> torch.Tensor:
        if item_features.ndim != 3:
            raise ValueError("item_features must have shape [B, N, D]")
        if page_features.ndim != 2:
            raise ValueError("page_features must have shape [B, D]")
        if mask.ndim != 2:
            raise ValueError("mask must have shape [B, N]")

        item_hidden = self.item_mlp(item_features)
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (item_hidden * mask_f).sum(dim=1) / denom

        fill = torch.full_like(item_hidden, -1e9)
        max_pool = torch.where(mask.unsqueeze(-1), item_hidden, fill).max(dim=1).values
        has_any = mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))

        batch_size = int(item_hidden.shape[0])
        last_hidden = torch.zeros((batch_size, self.hidden_dim), dtype=item_hidden.dtype, device=item_hidden.device)
        has_item = prefix_len > 0
        if bool(has_item.any()):
            idx = (prefix_len[has_item] - 1).long().clamp(min=0, max=item_hidden.shape[1] - 1)
            last_hidden[has_item] = item_hidden[has_item, idx]

        page_hidden = self.page_mlp(page_features)
        total_safe = total_items.float().clamp_min(1.0)
        prefix_ratio = prefix_len.float() / total_safe
        stats = torch.stack(
            [
                prefix_ratio,
                has_item.float(),
                1.0 - prefix_ratio,
            ],
            dim=-1,
        )
        fused = torch.cat([mean_pool, max_pool, last_hidden, page_hidden, stats], dim=-1)
        return self.value(fused).squeeze(-1)


def load_page_prefix_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[SlateCreditHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = SlateCreditHead(
        hidden_size=int(meta["hidden_size"]),
        mlp_dim=int(meta.get("mlp_dim", 128)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta


def load_item_prefix_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[ItemPrefixValueHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = ItemPrefixValueHead(
        item_dim=int(meta["item_dim"]),
        page_dim=int(meta["page_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 96)),
        dropout=float(meta.get("dropout", 0.10)),
        stats_dim=int(meta.get("stats_dim", 3)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta
