# Baseline Compliance Notes

This note records what can be claimed as a strict same-protocol baseline.

## BERT4Rec

Status: implemented as a valid strict online baseline.

Protocol:
- Train on the same KuaiRand Pure CSV as SASRec/GRU4Rec.
- Build the same global `video_id -> item_id` catalog mapping over all CSV rows.
- Use click-only user histories, matching the sequential baseline setting.
- At evaluation time, append one `[MASK]` token to the observed click history and score the current environment candidate pool.
- Return local candidate indices to `KREnvironment_WholeSession_GPU`.
- Keep `slate_size=6`, `item_correlation=0.2`, `single_response`, reward aggregation, episode count, and metric logging identical to the existing strict eval scripts.

Claim boundary:
- This is a BERT4Rec-style Cloze sequential recommender adapted to the project's online candidate-pool evaluation.
- It is not using side information, simulator rollouts, TIGER SID tokens, or future online feedback during eval.

## P5 / OpenP5

Status: implemented as a conservative `P5-style constrained item generator`; not safe to call canonical OpenP5.

Canonical requirement:
- A true P5/OpenP5 baseline should use a prompt-to-generation model and evaluate item candidates by generated item likelihood or constrained generation.
- It should train only on the same allowed recommendation logs and then act inside the same environment/candidate-pool protocol.

Risk:
- A small local seq2seq adapter over item ids can be useful diagnostically, but it should be labeled `P5-style adapter`, not `OpenP5`, unless we integrate and run the official OpenP5/P5 recipe.
- The current server has `transformers`, but no `sentencepiece`, so directly loading many T5 checkpoints may require an additional dependency or a custom tokenizer-free setup.

Recommended compliant path:
- Current code path: `model/p5_style_rec.py`, `train_p5_style_baseline.py`, `eval_p5_style_env.py`.
- Report name: `P5-style` or `P5-style constrained generator`.
- Do not report this as `OpenP5` unless we later integrate the official OpenP5/P5 prompt/template/pretraining recipe.
- If paper-grade comparison matters: integrate the official OpenP5/P5 code path, freeze the prompt templates, and document preprocessing/training/eval parity.

## SlateQ

Status: a `SlateQ-like TIGER slate-value reranker` launch script is provided, but it is not safe to claim as canonical SlateQ.

Canonical requirement:
- SlateQ decomposes slate value with item-wise contribution terms and accounts for slate interactions/choice effects.
- A fair implementation must be trained in the same simulator/log setup and evaluated by the same strict online env metrics.

Current codebase situation:
- The bundle already contains slate critic/allocator utilities (`train_tiger_slate_critic.py`, `train_tiger_slate_allocator.py`, `tiger_slate_allocator_common.py`, `tiger_slate_online_common.py`).
- These are useful for a SlateQ-like baseline, but they are not automatically a faithful reproduction of canonical SlateQ.
- Current script: `run_strict_slateq_like_eval3_20260512.sh`.
- Report name: `SlateQ-like` or `TIGER + slate-value reranker`, not plain `SlateQ`.
- The script collects TIGER base traces, trains a slate-value head from returns, and reranks TIGER candidate slates in the strict environment.

Recommended compliant path:
- Use the current script as a diagnostic baseline.
- Only call it `SlateQ` if the training objective, item-wise decomposition, user-choice/continuation assumptions, and policy improvement step match the SlateQ paper closely enough and are documented.
