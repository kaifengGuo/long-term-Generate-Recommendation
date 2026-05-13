# -*- coding: utf-8 -*-
"""
P5-style constrained item generator baseline.

This is intentionally named "P5-style" instead of "OpenP5": it implements the
core recommendation-as-generation interface with item tokens and constrained
candidate likelihoods, but it does not claim to be an official OpenP5
reproduction with natural-language prompt templates or pretrained T5 weights.

Conventions:
- item ids are in [1..n_items]
- PAD is 0
- BOS decoder token is n_items + 1
- online eval scores the current candidate pool by the generated first item
  token logit, under the same environment protocol as other baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class P5StyleRecConfig:
    n_items: int
    max_len: int = 50
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    dropout: float = 0.2


class P5StyleRec(nn.Module):
    def __init__(self, cfg: P5StyleRecConfig):
        super().__init__()
        self.cfg = cfg
        self.n_items = int(cfg.n_items)
        self.max_len = int(cfg.max_len)
        self.d_model = int(cfg.d_model)
        self.pad_id = 0
        self.bos_id = self.n_items + 1

        self.item_emb = nn.Embedding(self.n_items + 2, self.d_model, padding_idx=self.pad_id)
        self.enc_pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.dec_pos_emb = nn.Embedding(1, self.d_model)
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
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(cfg.n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.n_encoder_layers))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(cfg.n_decoder_layers))
        self.ln_final = nn.LayerNorm(self.d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.enc_pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dec_pos_emb.weight, mean=0.0, std=0.02)

    def encode_memory(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        device = seq.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq.clamp(min=0, max=self.bos_id)) + self.enc_pos_emb(pos)
        x = self.dropout(x)
        key_padding_mask = seq.eq(self.pad_id)
        # Transformer attention becomes ill-defined when every encoder token is
        # masked. Keep one harmless PAD slot visible for truly empty histories.
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, -1] = False
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def predict_next_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Score the generated first item token conditioned on the history prompt.
        """
        B, L = seq.shape
        if L != self.max_len:
            raise ValueError(f"Expected seq length {self.max_len}, got {L}")
        memory = self.encode_memory(seq)
        memory_padding_mask = seq.eq(self.pad_id)
        all_pad = memory_padding_mask.all(dim=1)
        if all_pad.any():
            memory_padding_mask = memory_padding_mask.clone()
            memory_padding_mask[all_pad, -1] = False
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=seq.device)
        dec_pos = torch.zeros((B, 1), dtype=torch.long, device=seq.device)
        tgt = self.item_emb(bos) + self.dec_pos_emb(dec_pos)
        tgt = self.dropout(tgt)
        h = self.decoder(tgt, memory, memory_key_padding_mask=memory_padding_mask)
        h = self.ln_final(h[:, 0, :])
        logits = h @ self.item_emb.weight[: self.n_items + 1].t()
        logits[:, 0] = -1e9
        return logits

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Compatibility with the existing sequential policy. We return candidate
        logits as the "user representation", and score_candidates gathers them.
        """
        return self.predict_next_logits(seq)

    def score_candidates(self, user_repr: torch.Tensor, cand_iids: Union[torch.Tensor, list]) -> torch.Tensor:
        if not torch.is_tensor(cand_iids):
            cand_iids = torch.tensor(cand_iids, dtype=torch.long, device=user_repr.device)
        cand = cand_iids.to(user_repr.device).long().clamp(min=0, max=self.n_items)
        if cand.dim() == 1:
            cand = cand.unsqueeze(0)
        if cand.size(0) == 1 and user_repr.size(0) != 1:
            cand = cand.expand(user_repr.size(0), -1)
        scores = user_repr.gather(dim=1, index=cand)
        if (cand == 0).any():
            scores = scores.masked_fill(cand == 0, -1e9)
        return scores

    def compute_loss_ce(self, seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.predict_next_logits(seq)
        return F.cross_entropy(logits, target, ignore_index=0)
