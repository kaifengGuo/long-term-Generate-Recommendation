# Base Policy And Ablation Summary

This file summarizes the original base policy and the completed ablations under the same strict evaluation protocol used during development.

## Evaluation Protocol

- Policy family: TIGER generative recommendation policy.
- Reward content: single-response evaluation.
- Slate size: 6.
- Beam width: 16.
- Eval episodes: 200 for closed-loop after-eval unless otherwise noted.
- Item correlation: 0.2.
- Phase-2 blend scale: 0.20.
- Random top-k sampling: 10.
- Main seed: 2026 for single-run ablations.

The original base TIGER checkpoint is not stored in git. Pass it through `TIGER_CKPT` when running the release launcher. In the ablation table below, the first row is the original no-post-training base policy.

## Completed Main Ablations

| Method | What Changed | Iter | Before | Rollout | After Rollout | After Learner | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TIGER base | Original base policy, no closed-loop post-training | 0 | 4.5417 | - | 4.5417 | - | Reference baseline; full strict metrics: depth 12.26, avg step reward 0.3706, coverage 0.81, ILD 0.9870, click 37.0597%, long view 28.9270% |
| GRPO-only + EMA | Remove DPO anchor; keep GRPO actor update | 1 | 4.5417 | 4.6387 | 4.6783 | 4.5133 | One-step rollout improves, learner slightly below base |
| GRPO-only + EMA | Same as above, continued | 2 | 4.6783 | 4.6205 | 4.4308 | 4.4025 | Starts drifting downward |
| GRPO-only + EMA | Same as above, continued | 3 | 4.4308 | 4.6711 | 4.2208 | 4.2017 | Multi-iteration pure GRPO is unstable |
| GRPO + DPO-seq + EMA | Add sequence-level DPO conservative anchor | 1 | 4.5417 | 4.6387 | 4.6208 | 4.6750 | DPO stabilizes learner |
| GRPO + DPO-seq + EMA | Same as above, continued | 2 | 4.6208 | 4.7693 | 4.5550 | 4.3200 | Rollout branch remains better than GRPO-only |
| GRPO + DPO-seq + EMA | Same as above, continued | 3 | 4.5550 | 4.8225 | 4.7075 | 4.4142 | Strong EMA rollout recovery; learner still weaker |
| GRPO + strong KL | Remove DPO and support gap; use stronger KL and tighter clip | 1 | 4.5417 | 4.6387 | 4.5217 | 4.2900 | KL helps but does not solve drift |
| GRPO + strong KL | Same as above, continued | 2 | 4.5217 | 4.7249 | 4.4408 | 4.2500 | Rollout improves but actor update regresses |
| GRPO + strong KL | Same as above, continued | 3 | 4.4408 | 4.6383 | 4.3250 | 4.3742 | Better than worst GRPO-only learner, still below base |
| No support gap | Keep DPO-seq, remove support-gap terms | 1 | 4.5417 | 4.6387 | 4.6600 | 4.2850 | Learner collapses despite good rollout |
| No support gap | Same as above, continued | 2 | 4.6600 | 4.7249 | 4.3542 | 4.0875 | Both rollout and learner regress |
| No support gap | Same as above, continued | 3 | 4.3542 | 4.7541 | 4.4433 | 4.2567 | Support gap is useful for stability |
| DPO-only support-aware | Remove GRPO actor loss; use support-aware preference actor only | 1 | 4.5417 | 4.6387 | 4.2817 | 4.4458 | Pairwise objective alone underperforms base |
| DPO-only support-aware | Same as above, continued | 2 | 4.2817 | 4.6653 | 4.7008 | 4.5358 | Temporary EMA recovery |
| DPO-only support-aware | Same as above, continued | 3 | 4.7008 | 4.6869 | 4.4558 | 3.8825 | Preference-only update is not stable |
| SAGERec adaptive-trust GRPO | Remove DPO; use support-aware pessimistic reward plus adaptive KL/clip | 1 | 4.5417 | 4.6387 | 4.4742 | 4.6825 | Current mainline; learner improves over base in iter1 |
| SAGERec adaptive-trust GRPO | Same as above, continued | 2 | 4.4742 | 4.6742 | 4.7367 | 4.7383 | Best checkpoint in this run; both rollout and learner improve over base |
| SAGERec adaptive-trust GRPO | Same as above, continued | 3 | 4.7367 | 4.6611 | 4.4567 | 4.3483 | Over-updates after iter2; motivates early stopping or adaptive stop criteria |

## Notes

- SAGERec adaptive-trust GRPO finished 3 iterations. The best checkpoint in this run is iter2, with `after_learner = 4.7383` and `after_rollout = 4.7367`.
- Iter3 regresses below the base policy, which suggests the closed-loop post-training objective benefits from early stopping or adaptive stop criteria.
- The table intentionally records the original base TIGER row so that improvements and regressions are visible.
- Checkpoints, simulator artifacts, and heavy generated groups are excluded from git. They should be distributed separately only when needed for reproduction.
