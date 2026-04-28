import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from tiger_page_sid_rl.common import masked_softmax


class PageSIDQCritic(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        page_feat_dim: int,
        item_dim: int = 128,
        model_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.page_feat_dim = int(page_feat_dim)
        self.item_dim = int(item_dim)
        self.model_dim = int(model_dim)
        self.dropout = float(dropout)

        self.token_emb = nn.Embedding(self.vocab_size, self.item_dim)
        self.page_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size + self.page_feat_dim),
            nn.Linear(self.hidden_size + self.page_feat_dim, self.model_dim),
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
        self.out = nn.Sequential(
            nn.LayerNorm(self.model_dim * 4),
            nn.Linear(self.model_dim * 4, self.model_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, 1),
        )

    def item_representations(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_mask = token_ids > 0
        token_emb = self.token_emb(token_ids)
        item_repr = (token_emb * token_mask.unsqueeze(-1).float()).sum(dim=2)
        item_mask = token_mask.any(dim=-1)
        return item_repr, item_mask

    def forward_from_item_repr(
        self,
        *,
        pre_summary: torch.Tensor,
        item_repr: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        page_context = self.page_mlp(torch.cat([pre_summary, page_features], dim=-1))
        item_hidden = self.item_mlp(item_repr)
        query = self.query(page_context).unsqueeze(1)
        key = self.key(item_hidden)
        logits = (query * key).sum(dim=-1) / math.sqrt(float(self.model_dim))
        logits = logits + self.gate(
            torch.cat([item_hidden, page_context.unsqueeze(1).expand_as(item_hidden)], dim=-1)
        ).squeeze(-1)
        shares = masked_softmax(logits, item_mask, dim=-1)
        pooled = (shares.unsqueeze(-1) * item_hidden).sum(dim=1)
        mean_item = (item_hidden * item_mask.unsqueeze(-1).float()).sum(dim=1) / item_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        max_item = item_hidden.masked_fill(~item_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_item = torch.where(torch.isfinite(max_item), max_item, torch.zeros_like(max_item))
        q_value = self.out(torch.cat([page_context, pooled, mean_item, max_item], dim=-1)).squeeze(-1)
        return {
            "q_value": q_value,
            "item_repr": item_repr,
            "item_hidden": item_hidden,
            "item_mask": item_mask,
            "item_shares": shares,
            "item_logits": logits,
            "page_context": page_context,
        }

    def forward(
        self,
        *,
        pre_summary: torch.Tensor,
        token_ids: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        item_repr, derived_mask = self.item_representations(token_ids)
        final_mask = derived_mask if item_mask is None else (item_mask & derived_mask)
        return self.forward_from_item_repr(
            pre_summary=pre_summary,
            item_repr=item_repr,
            item_mask=final_mask,
            page_features=page_features,
            user_features=user_features,
        )


class PageSIDQCriticV8(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        page_feat_dim: int,
        user_feat_dim: int = 0,
        sid_depth: int = 4,
        item_dim: int = 128,
        model_dim: int = 192,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.page_feat_dim = int(page_feat_dim)
        self.user_feat_dim = int(user_feat_dim)
        self.sid_depth = max(int(sid_depth), 1)
        self.item_dim = int(item_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.token_emb = nn.Embedding(self.vocab_size, self.item_dim)
        self.stage_emb = nn.Embedding(self.sid_depth + 1, self.item_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.item_dim),
            nn.Linear(self.item_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )
        self.item_proj = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
        )

        if self.user_feat_dim > 0:
            self.user_mlp = nn.Sequential(
                nn.LayerNorm(self.user_feat_dim),
                nn.Linear(self.user_feat_dim, self.model_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
            )
        else:
            self.user_mlp = None

        page_input_dim = self.hidden_size + self.page_feat_dim + (self.model_dim if self.user_mlp is not None else 0)
        self.page_mlp = nn.Sequential(
            nn.LayerNorm(page_input_dim),
            nn.Linear(page_input_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
        )

        self.item_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.model_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.item_attn_norms = nn.ModuleList([nn.LayerNorm(self.model_dim) for _ in range(self.num_layers)])
        self.item_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.model_dim),
                    nn.Linear(self.model_dim, self.model_dim * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.model_dim * 2, self.model_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.model_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.model_dim * 5),
            nn.Linear(self.model_dim * 5, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, 1),
        )

    def item_representations(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_mask = token_ids > 0
        stage_ids = torch.arange(int(token_ids.shape[-1]), device=token_ids.device, dtype=torch.long).view(1, 1, -1) + 1
        token_hidden = self.token_emb(token_ids) + self.stage_emb(stage_ids)
        token_hidden = self.token_proj(token_hidden)
        token_scores = self.token_gate(token_hidden).squeeze(-1)
        token_shares = masked_softmax(token_scores, token_mask, dim=-1)
        token_weighted = (token_hidden * token_shares.unsqueeze(-1)).sum(dim=2)
        token_mean = (token_hidden * token_mask.unsqueeze(-1).float()).sum(dim=2) / token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        item_repr = torch.cat([token_weighted, token_mean], dim=-1)
        item_mask = token_mask.any(dim=-1)
        return item_repr, item_mask

    def forward_from_item_repr(
        self,
        *,
        pre_summary: torch.Tensor,
        item_repr: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.user_mlp is not None and user_features is not None and int(user_features.shape[-1]) > 0:
            user_context = self.user_mlp(user_features)
            page_input = torch.cat([pre_summary, page_features, user_context], dim=-1)
        else:
            user_context = torch.zeros(
                (pre_summary.shape[0], self.model_dim),
                dtype=pre_summary.dtype,
                device=pre_summary.device,
            )
            page_input = torch.cat([pre_summary, page_features], dim=-1)

        page_context = self.page_mlp(page_input)
        item_hidden = self.item_proj(item_repr)
        safe_item_mask = item_mask.clone()
        if safe_item_mask.numel() > 0:
            empty_rows = ~safe_item_mask.any(dim=-1)
            if bool(empty_rows.any()):
                safe_item_mask = safe_item_mask.clone()
                item_hidden = item_hidden.clone()
                safe_item_mask[empty_rows, 0] = True
                item_hidden[empty_rows, 0] = 0.0

        key_padding_mask = ~safe_item_mask
        for attn_layer, attn_norm, ffn in zip(self.item_attn_layers, self.item_attn_norms, self.item_ffns):
            attn_out, _ = attn_layer(
                item_hidden,
                item_hidden,
                item_hidden,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            item_hidden = attn_norm(item_hidden + attn_out)
            item_hidden = item_hidden + ffn(item_hidden)

        page_query = page_context.unsqueeze(1)
        cross_out, cross_weights = self.cross_attn(
            page_query,
            item_hidden,
            item_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        cross_out = self.cross_norm(page_query + cross_out).squeeze(1)
        cross_weights = cross_weights.mean(dim=1).squeeze(1)
        logits = torch.log(cross_weights.clamp_min(1e-8)) + self.gate(
            torch.cat([item_hidden, page_context.unsqueeze(1).expand_as(item_hidden)], dim=-1)
        ).squeeze(-1)
        shares = masked_softmax(logits, safe_item_mask, dim=-1)
        pooled = (shares.unsqueeze(-1) * item_hidden).sum(dim=1)
        mean_item = (item_hidden * safe_item_mask.unsqueeze(-1).float()).sum(dim=1) / safe_item_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        max_item = item_hidden.masked_fill(~safe_item_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_item = torch.where(torch.isfinite(max_item), max_item, torch.zeros_like(max_item))
        q_value = self.out(torch.cat([page_context, user_context, cross_out, pooled, max_item], dim=-1)).squeeze(-1)
        return {
            "q_value": q_value,
            "item_repr": item_repr,
            "item_hidden": item_hidden,
            "item_mask": safe_item_mask,
            "item_shares": shares,
            "item_logits": logits,
            "page_context": page_context,
            "user_context": user_context,
            "cross_context": cross_out,
            "mean_item": mean_item,
        }

    def forward(
        self,
        *,
        pre_summary: torch.Tensor,
        token_ids: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        item_repr, derived_mask = self.item_representations(token_ids)
        final_mask = derived_mask if item_mask is None else (item_mask & derived_mask)
        return self.forward_from_item_repr(
            pre_summary=pre_summary,
            item_repr=item_repr,
            item_mask=final_mask,
            page_features=page_features,
            user_features=user_features,
        )


class PageSIDQCriticV9Additive(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        page_feat_dim: int,
        user_feat_dim: int = 0,
        sid_depth: int = 4,
        item_dim: int = 128,
        model_dim: int = 192,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
        max_slot_embeddings: int = 32,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.page_feat_dim = int(page_feat_dim)
        self.user_feat_dim = int(user_feat_dim)
        self.sid_depth = max(int(sid_depth), 1)
        self.item_dim = int(item_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.max_slot_embeddings = int(max_slot_embeddings)

        self.token_emb = nn.Embedding(self.vocab_size, self.item_dim)
        self.stage_emb = nn.Embedding(self.sid_depth + 1, self.item_dim)
        self.slot_emb = nn.Embedding(self.max_slot_embeddings, self.model_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.item_dim),
            nn.Linear(self.item_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )
        self.item_proj = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
        )

        if self.user_feat_dim > 0:
            self.user_mlp = nn.Sequential(
                nn.LayerNorm(self.user_feat_dim),
                nn.Linear(self.user_feat_dim, self.model_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
            )
        else:
            self.user_mlp = None

        page_input_dim = self.hidden_size + self.page_feat_dim + (self.model_dim if self.user_mlp is not None else 0)
        self.page_mlp = nn.Sequential(
            nn.LayerNorm(page_input_dim),
            nn.Linear(page_input_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
        )

        self.item_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.model_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.item_attn_norms = nn.ModuleList([nn.LayerNorm(self.model_dim) for _ in range(self.num_layers)])
        self.item_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.model_dim),
                    nn.Linear(self.model_dim, self.model_dim * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.model_dim * 2, self.model_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.model_dim)
        self.share_gate = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )
        self.base_out = nn.Sequential(
            nn.LayerNorm(self.model_dim * 5),
            nn.Linear(self.model_dim * 5, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, 1),
        )
        self.item_value_head = nn.Sequential(
            nn.LayerNorm(self.model_dim * 3),
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, 1),
        )
        self.item_value_gate = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 1),
        )

    def item_representations(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_mask = token_ids > 0
        stage_ids = torch.arange(int(token_ids.shape[-1]), device=token_ids.device, dtype=torch.long).view(1, 1, -1) + 1
        token_hidden = self.token_emb(token_ids) + self.stage_emb(stage_ids)
        token_hidden = self.token_proj(token_hidden)
        token_scores = self.token_gate(token_hidden).squeeze(-1)
        token_shares = masked_softmax(token_scores, token_mask, dim=-1)
        token_weighted = (token_hidden * token_shares.unsqueeze(-1)).sum(dim=2)
        token_mean = (token_hidden * token_mask.unsqueeze(-1).float()).sum(dim=2) / token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        item_repr = torch.cat([token_weighted, token_mean], dim=-1)
        item_mask = token_mask.any(dim=-1)
        return item_repr, item_mask

    def forward_from_item_repr(
        self,
        *,
        pre_summary: torch.Tensor,
        item_repr: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.user_mlp is not None and user_features is not None and int(user_features.shape[-1]) > 0:
            user_context = self.user_mlp(user_features)
            page_input = torch.cat([pre_summary, page_features, user_context], dim=-1)
        else:
            user_context = torch.zeros(
                (pre_summary.shape[0], self.model_dim),
                dtype=pre_summary.dtype,
                device=pre_summary.device,
            )
            page_input = torch.cat([pre_summary, page_features], dim=-1)

        page_context = self.page_mlp(page_input)
        item_hidden = self.item_proj(item_repr)

        slot_ids = torch.arange(int(item_hidden.shape[1]), device=item_hidden.device, dtype=torch.long)
        slot_ids = slot_ids.clamp(max=self.max_slot_embeddings - 1).view(1, -1)
        item_hidden = item_hidden + self.slot_emb(slot_ids)

        safe_item_mask = item_mask.clone()
        if safe_item_mask.numel() > 0:
            empty_rows = ~safe_item_mask.any(dim=-1)
            if bool(empty_rows.any()):
                safe_item_mask = safe_item_mask.clone()
                item_hidden = item_hidden.clone()
                safe_item_mask[empty_rows, 0] = True
                item_hidden[empty_rows, 0] = 0.0

        key_padding_mask = ~safe_item_mask
        for attn_layer, attn_norm, ffn in zip(self.item_attn_layers, self.item_attn_norms, self.item_ffns):
            attn_out, _ = attn_layer(
                item_hidden,
                item_hidden,
                item_hidden,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            item_hidden = attn_norm(item_hidden + attn_out)
            item_hidden = item_hidden + ffn(item_hidden)

        page_query = page_context.unsqueeze(1)
        cross_out, cross_weights = self.cross_attn(
            page_query,
            item_hidden,
            item_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        cross_out = self.cross_norm(page_query + cross_out).squeeze(1)
        cross_weights = cross_weights.mean(dim=1).squeeze(1)

        share_logits = torch.log(cross_weights.clamp_min(1e-8)) + self.share_gate(
            torch.cat([item_hidden, page_context.unsqueeze(1).expand_as(item_hidden)], dim=-1)
        ).squeeze(-1)
        shares = masked_softmax(share_logits, safe_item_mask, dim=-1)
        pooled = (shares.unsqueeze(-1) * item_hidden).sum(dim=1)
        mean_item = (item_hidden * safe_item_mask.unsqueeze(-1).float()).sum(dim=1) / safe_item_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        max_item = item_hidden.masked_fill(~safe_item_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_item = torch.where(torch.isfinite(max_item), max_item, torch.zeros_like(max_item))

        base_value = self.base_out(torch.cat([page_context, user_context, cross_out, pooled, max_item], dim=-1)).squeeze(-1)
        item_joint = torch.cat(
            [
                item_hidden,
                page_context.unsqueeze(1).expand_as(item_hidden),
                cross_out.unsqueeze(1).expand_as(item_hidden),
            ],
            dim=-1,
        )
        item_delta = self.item_value_head(item_joint).squeeze(-1)
        item_gate = torch.sigmoid(
            self.item_value_gate(
                torch.cat([item_hidden, page_context.unsqueeze(1).expand_as(item_hidden)], dim=-1)
            ).squeeze(-1)
        )
        item_contrib = item_delta * item_gate * safe_item_mask.float()
        q_value = base_value + item_contrib.sum(dim=1)
        return {
            "q_value": q_value,
            "item_repr": item_repr,
            "item_hidden": item_hidden,
            "item_mask": safe_item_mask,
            "item_shares": shares,
            "item_logits": share_logits,
            "page_context": page_context,
            "user_context": user_context,
            "cross_context": cross_out,
            "mean_item": mean_item,
            "base_value": base_value,
            "item_contrib": item_contrib,
        }

    def forward(
        self,
        *,
        pre_summary: torch.Tensor,
        token_ids: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        item_repr, derived_mask = self.item_representations(token_ids)
        final_mask = derived_mask if item_mask is None else (item_mask & derived_mask)
        return self.forward_from_item_repr(
            pre_summary=pre_summary,
            item_repr=item_repr,
            item_mask=final_mask,
            page_features=page_features,
            user_features=user_features,
        )


class PageSIDQCriticEnsemble(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        if not members:
            raise ValueError("members must be non-empty")
        self.members = nn.ModuleList(list(members))

    @property
    def ensemble_size(self) -> int:
        return int(len(self.members))

    def _merge_outputs(self, member_outputs: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        merged = dict(member_outputs[0])
        q_values = torch.stack([out["q_value"] for out in member_outputs], dim=-1)
        q_mean = q_values.mean(dim=-1)
        if int(q_values.shape[-1]) > 1:
            q_std = q_values.std(dim=-1, unbiased=False)
        else:
            q_std = torch.zeros_like(q_mean)
        merged["q_values"] = q_values
        merged["q_mean"] = q_mean
        merged["q_std"] = q_std
        merged["q_value"] = q_mean
        return merged

    def forward_from_item_repr(
        self,
        *,
        pre_summary: torch.Tensor,
        item_repr: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        member_outputs = [
            member.forward_from_item_repr(
                pre_summary=pre_summary,
                item_repr=item_repr,
                item_mask=item_mask,
                page_features=page_features,
                user_features=user_features,
            )
            for member in self.members
        ]
        return self._merge_outputs(member_outputs)

    def forward(
        self,
        *,
        pre_summary: torch.Tensor,
        token_ids: torch.Tensor,
        item_mask: torch.Tensor,
        page_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        member_outputs = [
            member(
                pre_summary=pre_summary,
                token_ids=token_ids,
                item_mask=item_mask,
                page_features=page_features,
                user_features=user_features,
            )
            for member in self.members
        ]
        return self._merge_outputs(member_outputs)


def build_page_sid_qcritic_base(
    *,
    arch: str,
    hidden_size: int,
    vocab_size: int,
    page_feat_dim: int,
    item_dim: int = 128,
    model_dim: int = 128,
    dropout: float = 0.10,
    user_feat_dim: int = 0,
    sid_depth: int = 4,
    num_heads: int = 4,
    num_layers: int = 2,
) -> nn.Module:
    common_kwargs = {
        "hidden_size": int(hidden_size),
        "vocab_size": int(vocab_size),
        "page_feat_dim": int(page_feat_dim),
        "item_dim": int(item_dim),
        "model_dim": int(model_dim),
        "dropout": float(dropout),
    }
    if str(arch).lower() == "v8":
        return PageSIDQCriticV8(
            **common_kwargs,
            user_feat_dim=int(user_feat_dim),
            sid_depth=int(sid_depth),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
        )
    if str(arch).lower() in {"v9add", "v9_add", "additive"}:
        return PageSIDQCriticV9Additive(
            **common_kwargs,
            user_feat_dim=int(user_feat_dim),
            sid_depth=int(sid_depth),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
        )
    return PageSIDQCritic(**common_kwargs)


def build_page_sid_qcritic_from_meta(meta: Dict[str, Any]) -> nn.Module:
    arch = str(meta.get("arch", "base")).lower()
    ensemble_size = max(int(meta.get("ensemble_size", 1)), 1)
    base_kwargs = {
        "arch": str(arch),
        "hidden_size": int(meta["hidden_size"]),
        "vocab_size": int(meta["vocab_size"]),
        "page_feat_dim": int(meta["page_feat_dim"]),
        "item_dim": int(meta.get("item_dim", 128)),
        "model_dim": int(meta.get("model_dim", 128)),
        "dropout": float(meta.get("dropout", 0.10)),
        "user_feat_dim": int(meta.get("user_feat_dim", 0)),
        "sid_depth": int(meta.get("sid_depth", 4)),
        "num_heads": int(meta.get("num_heads", 4)),
        "num_layers": int(meta.get("num_layers", 2)),
    }
    if int(ensemble_size) > 1:
        return PageSIDQCriticEnsemble(
            [build_page_sid_qcritic_base(**base_kwargs) for _ in range(int(ensemble_size))]
        )
    return build_page_sid_qcritic_base(**base_kwargs)


def save_page_sid_qcritic_bundle(
    bundle_path: str | Path,
    meta_path: str | Path,
    model: nn.Module,
    meta: Dict[str, Any],
) -> None:
    bundle = Path(bundle_path)
    meta_file = Path(meta_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, bundle)
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_page_sid_qcritic_bundle(
    bundle_path: str | Path,
    meta_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    model = build_page_sid_qcritic_from_meta(meta)
    payload = torch.load(bundle_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, meta
