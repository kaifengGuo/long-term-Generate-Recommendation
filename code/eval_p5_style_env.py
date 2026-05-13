# -*- coding: utf-8 -*-
"""
Strict same-protocol online evaluation for the P5-style constrained generator.

The env loop is reused from eval_sasrec_env.py so candidate handling, click
history, reward aggregation, and metric logging stay aligned with the existing
sequential baselines.
"""
from __future__ import annotations

import sys

import torch

import eval_sasrec_env as seq_eval
import utils
from model.p5_style_rec import P5StyleRec, P5StyleRecConfig


def load_p5_style_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg_dict = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None
    if cfg_dict is None:
        item_w = state["item_emb.weight"]
        pos_w = state["enc_pos_emb.weight"]
        n_items = int(item_w.shape[0] - 2)
        d_model = int(item_w.shape[1])
        max_len = int(pos_w.shape[0])
        cfg = P5StyleRecConfig(n_items=n_items, max_len=max_len, d_model=d_model)
    else:
        cfg = P5StyleRecConfig(**cfg_dict)
    model = P5StyleRec(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def main() -> None:
    argv = list(sys.argv[1:])
    for idx, token in enumerate(argv):
        if token == "--p5_style_ckpt":
            argv[idx] = "--sasrec_ckpt"
    sys.argv = [sys.argv[0]] + argv
    seq_eval.load_sasrec_model = load_p5_style_model
    args = seq_eval.parse_args()
    print("[Eval] P5-style constrained generator wrapper enabled; not an official OpenP5 reproduction.")
    utils.set_random_seed(args.seed)
    seq_eval.run_eval(args)


if __name__ == "__main__":
    main()
