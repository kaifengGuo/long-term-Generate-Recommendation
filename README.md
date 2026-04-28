# SAGERec

Clean code release for support-aware adaptive-trust post-training of generative recommendation policies.

## What Is Included

- `code/`: core training, evaluation, simulator, TIGER, critic, rollout, attribution, and actor-update code.
- `code/tiger_page_sid_rl/run_page_sid_closed_loop.py`: main closed-loop pipeline.
- `code/tiger_page_sid_rl/run_sagerec_adaptive_grpo.sh`: clean launch script for the current main method.
- `code/tiger_page_sid_rl/train_page_critic.py`: page/item/SID critic training.
- `code/tiger_page_sid_rl/build_sid_advantage_chain.py`: hierarchical item/SID attribution construction.
- `code/build_tiger_hca_grpo_groups.py`: group candidate construction with pessimistic support-aware value.
- `code/train_tiger_hca_grpo_actor.py`: adaptive-trust GRPO actor update.
- Baseline model/evaluator code needed for comparison and reuse.

## What Is Intentionally Excluded

- Local visualization artifacts and one-off analysis outputs.
- Local memory files, private experiment notes, and server records.
- Private connection helpers or local machine utilities.
- Cached Python files.
- Generated `output/` and `results/` directories.
- Vendored external raw datasets and external repository snapshots.

This release is code-first. Checkpoints, simulator logs, and large datasets should be provided separately if the recipient needs to reproduce a run.

## Main Method

The clean mainline is support-aware adaptive-trust GRPO:

1. Collect closed-loop rollouts from a TIGER policy.
2. Train an ensemble page critic to estimate long-term page value and uncertainty.
3. Build candidate groups with a pessimistic support-aware score.
4. Use hierarchical item/SID attribution to expose credit assignment signals.
5. Update the actor with group-relative policy optimization.
6. Adaptively tighten the trust region when support gap or critic uncertainty is high.

The launcher keeps the method focused on GRPO plus support/uncertainty-aware trust control. Pairwise DPO-style ablations are not part of this clean entrypoint.

## Baseline And Ablations

The original base TIGER policy and completed ablation summary are documented in
`docs/ablation_summary.md`. Model checkpoints and generated result directories are intentionally
kept outside git and should be supplied separately when reproducing the numbers.

## Run

Prepare checkpoints and simulator artifacts under the expected relative paths, or pass them explicitly through environment variables:

```bash
cd SAGERec_release_20260428
PYTHON_BIN=python \
TIGER_CKPT=/path/to/base_tiger.pth \
UIRM_LOG_PATH=/path/to/user_response_model.log \
bash code/tiger_page_sid_rl/run_sagerec_adaptive_grpo.sh
```

Common knobs:

```bash
NUM_ITERS=3
ROLLOUT_EPISODES=2500
EVAL_EPISODES=200
DEVICE=cuda:0
SEED=2026
```

Outputs are written to `results/` inside this release directory.
