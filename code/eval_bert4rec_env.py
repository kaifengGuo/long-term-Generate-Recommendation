# -*- coding: utf-8 -*-
"""
Strict same-protocol online evaluation for BERT4Rec.

This intentionally reuses the existing SASRec env-eval loop and policy wrapper.
BERT4Rec exposes the same encode()/score_candidates() interface, where encode()
appends a final MASK token and returns the MASK hidden state. Therefore candidate
selection, click-history maintenance, reward aggregation, logging, and metrics
remain byte-for-byte aligned with the established SASRec baseline.
"""
from __future__ import annotations

import sys

import torch

import eval_sasrec_env as seq_eval
import utils
from model.bert4rec import BERT4Rec, BERT4RecConfig


def load_bert4rec_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg_dict = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None
    if cfg_dict is None:
        item_w = state["item_emb.weight"]
        pos_w = state["pos_emb.weight"]
        n_items = int(item_w.shape[0] - 2)
        d_model = int(item_w.shape[1])
        max_len = int(pos_w.shape[0])
        cfg = BERT4RecConfig(n_items=n_items, max_len=max_len, d_model=d_model, n_heads=4, n_layers=2, dropout=0.2)
    else:
        cfg = BERT4RecConfig(**cfg_dict)
    model = BERT4Rec(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def main() -> None:
    argv = list(sys.argv[1:])
    for idx, token in enumerate(argv):
        if token == "--bert4rec_ckpt":
            argv[idx] = "--sasrec_ckpt"
    sys.argv = [sys.argv[0]] + argv
    seq_eval.load_sasrec_model = load_bert4rec_model
    args = seq_eval.parse_args()
    print("[Eval] BERT4Rec strict wrapper enabled; using SASRec-compatible policy loop.")
    utils.set_random_seed(args.seed)
    seq_eval.run_eval(args)


if __name__ == "__main__":
    main()
