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
- Strict baseline model/evaluator code needed for comparison and reuse.
- `code/baseline_notes/`: claim-boundary notes for newly added baselines.

## What Is Intentionally Excluded

- Local visualization artifacts and one-off analysis outputs.
- Local memory files, private experiment notes, and server records.
- Private connection helpers or local machine utilities.
- Cached Python files.
- Generated `output/` and `results/` directories.
- Generated caches, jsonl traces, checkpoints, and simulator logs.
- Vendored external raw datasets and external repository snapshots.

This release is code-first. Checkpoints, simulator logs, and large datasets should be provided separately if the recipient needs to reproduce a run.

## Main Method

The clean mainline is HCA-LCB-GRPO, a DPO-free post-training path for generative recommendation:

1. Collect closed-loop rollouts from a TIGER policy.
2. Train an ensemble page critic to estimate long-term page value and uncertainty.
3. Build candidate groups with a lower-confidence long-term score.
4. Use hierarchical item/SID attribution to expose credit assignment signals.
5. Update the actor with group-relative policy optimization.
6. Keep the policy close to the base model with clipped GRPO and conservative EMA rollout selection.

The current TIGER suite launcher exposes the main DPO-free path as `hca_lcb_grpo`. Pairwise DPO-style methods are kept as baselines/ablations, not as the clean entrypoint.

## Strict Baselines

The release now includes strict same-protocol launchers for:

- Sequential recommenders: SASRec, GRU4Rec, BERT4Rec, and P5-style constrained generation.
- RL and slate variants: HAC, DDPG, TD3, A2C, and SlateQ-like TIGER slate-value reranking.
- Post-training baselines: S-DPO, SPRec, ReRe-style GRPO, Plain DPO, and OneRec LTV-GRPO variants.

Claim boundaries are important:

- `P5-style` is not an official OpenP5 reproduction unless the official prompt/pretraining recipe is integrated.
- `SlateQ-like` is not canonical SlateQ unless the SlateQ decomposition/objective is matched and documented.
- Generated result folders are intentionally excluded; rerun the launchers or provide checkpoints/results separately.

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
