# -*- coding: utf-8 -*-
"""
BERT4Rec baseline for the KuaiRand strict same-protocol evaluation.

Conventions match the existing SASRec baseline:
- item ids are in [1..n_items]
- PAD is 0
- MASK is n_items + 1
- input histories are left padded to max_len

At online evaluation time we append one MASK token to the right side of the
observed history and rank the current candidate pool by the hidden state at
that mask position. This keeps the action space and environment protocol
identical to SASRec/GRU4Rec/TIGER strict eval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BERT4RecConfig:
    n_items: int
    max_len: int = 50
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2


class BERT4Rec(nn.Module):
    def __init__(self, cfg: BERT4RecConfig):
        super().__init__()
        self.cfg = cfg
        self.n_items = int(cfg.n_items)
        self.max_len = int(cfg.max_len)
        self.d_model = int(cfg.d_model)
        self.pad_id = 0
        self.mask_id = self.n_items + 1

        self.item_emb = nn.Embedding(self.n_items + 2, self.d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.dropout = nn.Dropout(float(cfg.dropout))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(cfg.n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.n_layers))
        self.ln_final = nn.LayerNorm(self.d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B,L), values in [0..n_items+1].
        returns: (B,L,D).
        """
        B, L = input_ids.shape
        device = input_ids.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_ids.clamp(min=0, max=self.mask_id)) + self.pos_emb(pos)
        x = self.dropout(x)
        key_padding_mask = input_ids.eq(self.pad_id)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.ln_final(x)

    def append_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Convert a left-padded history sequence into a next-item query by
        dropping the oldest slot and appending MASK at the final position.
        """
        B, L = seq.shape
        if L != self.max_len:
            raise ValueError(f"Expected seq length {self.max_len}, got {L}")
        mask_col = torch.full((B, 1), self.mask_id, dtype=torch.long, device=seq.device)
        return torch.cat([seq[:, 1:], mask_col], dim=1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Compatibility method used by the existing sequential eval policy.
        The returned embedding is the final MASK hidden state.
        """
        query = self.append_mask(seq)
        h = self.forward(query)
        return h[:, -1, :]

    def predict_next_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """
        returns logits over [0..n_items]. PAD logit is masked. The MASK token
        itself is not exposed as an item candidate.
        """
        u = self.encode(seq)
        logits = u @ self.item_emb.weight[: self.n_items + 1].t()
        logits[:, 0] = -1e9
        return logits

    def score_candidates(self, user_emb: torch.Tensor, cand_iids: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        user_emb: (B,D), usually the MASK hidden state.
        cand_iids: (C,) or (B,C), ids in [1..n_items].
        returns: (B,C)
        """
        if not torch.is_tensor(cand_iids):
            cand_iids = torch.tensor(cand_iids, dtype=torch.long, device=user_emb.device)
        cand = cand_iids.to(user_emb.device).long().clamp(min=0, max=self.n_items)
        if cand.dim() == 1:
            cand = cand.unsqueeze(0)

        emb = self.item_emb(cand)
        if emb.size(0) == 1 and user_emb.size(0) != 1:
            emb = emb.expand(user_emb.size(0), -1, -1)
        scores = (emb * user_emb.unsqueeze(1)).sum(dim=-1)
        if (cand == 0).any():
            scores = scores.masked_fill(cand == 0, -1e9)
        return scores

    def compute_mlm_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        labels: (B,L), item ids for masked positions and -100 elsewhere.
        """
        h = self.forward(input_ids)
        logits = h @ self.item_emb.weight[: self.n_items + 1].t()
        return F.cross_entropy(logits.reshape(-1, self.n_items + 1), labels.reshape(-1), ignore_index=-100)

    def compute_loss_ce(self, seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Next-item CE compatibility helper for generic ranking utilities.
        """
        logits = self.predict_next_logits(seq)
        return F.cross_entropy(logits, target, ignore_index=0)
