import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from tiger_hcaa.common import masked_softmax


class SessionPotentialHead(nn.Module):
    def __init__(self, hidden_size: int, *, mlp_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.mlp_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, 1),
        )

    def forward(self, history_summary: torch.Tensor) -> torch.Tensor:
        return self.net(history_summary).squeeze(-1)


class PageBiasHead(nn.Module):
    def __init__(self, page_dim: int, *, mlp_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.page_dim = int(page_dim)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.page_dim),
            nn.Linear(self.page_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, 1),
        )

    def forward(self, page_features: torch.Tensor) -> torch.Tensor:
        return self.net(page_features).squeeze(-1)


class ItemAttentionAllocator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        page_dim: int,
        item_dim: int,
        *,
        model_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.page_dim = int(page_dim)
        self.item_dim = int(item_dim)
        self.model_dim = int(model_dim)
        self.dropout = float(dropout)

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 3 + self.page_dim),
            nn.Linear(self.hidden_size * 3 + self.page_dim, self.model_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.Tanh(),
        )
        self.item_mlp = nn.Sequential(
            nn.LayerNorm(self.item_dim),
            nn.Linear(self.item_dim, self.model_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.Tanh(),
        )
        self.query = nn.Linear(self.model_dim, self.model_dim)
        self.key = nn.Linear(self.model_dim, self.model_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.Tanh(),
            nn.Linear(self.model_dim, 1),
        )

    def forward(
        self,
        pre_summary: torch.Tensor,
        post_summary: torch.Tensor,
        page_features: torch.Tensor,
        item_features: torch.Tensor,
        item_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_summary = post_summary - pre_summary
        context_input = torch.cat([pre_summary, post_summary, delta_summary, page_features], dim=-1)
        page_context = self.context_mlp(context_input)
        item_hidden = self.item_mlp(item_features)

        query = self.query(page_context).unsqueeze(1)
        key = self.key(item_hidden)
        score = (query * key).sum(dim=-1) / math.sqrt(float(self.model_dim))
        score = score + self.gate(
            torch.cat([item_hidden, page_context.unsqueeze(1).expand_as(item_hidden)], dim=-1)
        ).squeeze(-1)
        shares = masked_softmax(score, item_mask, dim=-1)
        return shares, score, page_context, item_hidden


class TokenAttentionTokenizer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        model_dim: int = 128,
        token_dim: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.token_dim = int(token_dim)
        self.dropout = float(dropout)

        self.item_context_mlp = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.Tanh(),
        )
        self.token_emb = nn.Embedding(self.vocab_size, self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size + self.token_dim),
            nn.Linear(self.hidden_size + self.token_dim, self.model_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.Tanh(),
        )
        self.query = nn.Linear(self.model_dim, self.model_dim)
        self.key = nn.Linear(self.model_dim, self.model_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.Tanh(),
            nn.Linear(self.model_dim, 1),
        )

    def forward(
        self,
        page_context: torch.Tensor,
        item_hidden: torch.Tensor,
        token_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_context = self.item_context_mlp(
            torch.cat([page_context.unsqueeze(1).expand_as(item_hidden), item_hidden], dim=-1)
        )
        token_repr = self.token_mlp(
            torch.cat([token_hidden, self.token_emb(token_ids)], dim=-1)
        )
        query = self.query(item_context).unsqueeze(2)
        key = self.key(token_repr)
        score = (query * key).sum(dim=-1) / math.sqrt(float(self.model_dim))
        score = score + self.gate(
            torch.cat([token_repr, item_context.unsqueeze(2).expand_as(token_repr)], dim=-1)
        ).squeeze(-1)
        shares = masked_softmax(score, token_mask, dim=-1)
        return shares, score, item_context, token_repr


class HCAAJointCritic(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        page_dim: int,
        item_dim: int,
        vocab_size: int,
        mlp_dim: int = 128,
        token_dim: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.page_dim = int(page_dim)
        self.item_dim = int(item_dim)
        self.vocab_size = int(vocab_size)
        self.mlp_dim = int(mlp_dim)
        self.token_dim = int(token_dim)
        self.dropout = float(dropout)

        self.session_head = SessionPotentialHead(self.hidden_size, mlp_dim=self.mlp_dim, dropout=self.dropout)
        self.page_bias_head = PageBiasHead(self.page_dim, mlp_dim=self.mlp_dim, dropout=self.dropout)
        self.item_allocator = ItemAttentionAllocator(
            self.hidden_size,
            self.page_dim,
            self.item_dim,
            model_dim=self.mlp_dim,
            dropout=self.dropout,
        )
        self.token_tokenizer = TokenAttentionTokenizer(
            self.hidden_size,
            self.vocab_size,
            model_dim=self.mlp_dim,
            token_dim=self.token_dim,
            dropout=self.dropout,
        )

    def forward(
        self,
        *,
        pre_summary: torch.Tensor,
        post_summary: torch.Tensor,
        page_features: torch.Tensor,
        item_features: torch.Tensor,
        item_mask: torch.Tensor,
        token_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pre_value = self.session_head(pre_summary)
        post_value = self.session_head(post_summary)
        page_bias = self.page_bias_head(page_features)
        page_adv = post_value - pre_value - page_bias

        item_shares, item_logits, page_context, item_hidden = self.item_allocator(
            pre_summary,
            post_summary,
            page_features,
            item_features,
            item_mask,
        )
        item_adv = item_shares * page_adv.unsqueeze(-1)

        token_shares, token_logits, item_context, token_repr = self.token_tokenizer(
            page_context,
            item_hidden,
            token_hidden,
            token_ids,
            token_mask,
        )
        token_adv = token_shares * item_adv.unsqueeze(-1)

        return {
            "pre_value": pre_value,
            "post_value": post_value,
            "page_bias": page_bias,
            "page_adv": page_adv,
            "item_shares": item_shares,
            "item_logits": item_logits,
            "item_adv": item_adv,
            "token_shares": token_shares,
            "token_logits": token_logits,
            "token_adv": token_adv,
            "page_context": page_context,
            "item_hidden": item_hidden,
            "item_context": item_context,
            "token_repr": token_repr,
        }


def save_hcaa_bundle(
    bundle_path: str | Path,
    meta_path: str | Path,
    model: HCAAJointCritic,
    meta: Dict[str, Any],
) -> None:
    bundle_path = Path(bundle_path)
    meta_path = Path(meta_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, bundle_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_hcaa_bundle(
    bundle_path: str | Path,
    meta_path: str | Path,
    device: torch.device,
) -> Tuple[HCAAJointCritic, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    model = HCAAJointCritic(
        hidden_size=int(meta["hidden_size"]),
        page_dim=int(meta["page_dim"]),
        item_dim=int(meta["item_dim"]),
        vocab_size=int(meta["vocab_size"]),
        mlp_dim=int(meta.get("mlp_dim", 128)),
        token_dim=int(meta.get("token_dim", 32)),
        dropout=float(meta.get("dropout", 0.10)),
    )
    payload = torch.load(bundle_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, meta

