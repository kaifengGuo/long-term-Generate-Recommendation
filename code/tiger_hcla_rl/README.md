# TIGER HCLA-RL

`TIGER HCLA-RL` is an isolated research track for long-term value attribution and RL updates of TIGER.

The pipeline is:

1. `build_hcla_longterm_chain.py`
2. `train_hcla_longterm_critic.py`
3. `train_hcla_actor.py`
4. `run_hcla_iterative.py`
5. `run_hcla_ablation_sweep.py`

This folder reuses a few shared loaders and utility functions from the parent codebase, but it does not modify any existing method files.

`run_hcla_iterative.py` writes `baseline_eval_metrics`, `eval_metrics`, and `eval_delta` into
`iterative_summary.json`. For small smoke runs, eval batch size is automatically clamped to the
number of requested episodes so that 5-episode sanity checks do not run with an oversized batch.
It also supports `--actor_init_ckpt`, so iter 1 can use one checkpoint as the PPO-style reference
policy and a different checkpoint as the actor warm start.

`run_hcla_ablation_sweep.py` reads a JSON config list, reuses one shared baseline evaluation, and
writes per-config actor metrics plus `delta_vs_baseline` into one sweep summary. A starter config
is provided at `tiger_hcla_rl/configs/ablation_sweep_v1.json`.
