#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="$PROJECT_ROOT/code/tiger_page_sid_rl/run_page_sid_closed_loop.py"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date +%Y%m%d_%H%M%S)}"

TIGER_CKPT="${TIGER_CKPT:-$PROJECT_ROOT/output/KuaiRand_Pure/env/tiger_sid_krpure_mini_strong_seed2026_20260415_225310.pth}"
UIRM_LOG_PATH="${UIRM_LOG_PATH:-$PROJECT_ROOT/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log}"
SID_MAPPING_PATH="${SID_MAPPING_PATH:-$PROJECT_ROOT/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv}"

SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ITERS="${NUM_ITERS:-3}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-2500}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
SLATE_SIZE="${SLATE_SIZE:-6}"
BEAM_WIDTH="${BEAM_WIDTH:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/sagerec_adaptive_grpo_${TIMESTAMP_TAG}}"

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$TIGER_CKPT" ]]; then
  echo "Base TIGER checkpoint not found: $TIGER_CKPT" >&2
  exit 1
fi

if [[ ! -f "$UIRM_LOG_PATH" ]]; then
  echo "User response model log not found: $UIRM_LOG_PATH" >&2
  exit 1
fi

if [[ ! -f "$SID_MAPPING_PATH" ]]; then
  echo "SID mapping not found: $SID_MAPPING_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" "$RUNNER" \
  --tiger_ckpt "$TIGER_CKPT" \
  --uirm_log_path "$UIRM_LOG_PATH" \
  --sid_mapping_path "$SID_MAPPING_PATH" \
  --output_root "$OUTPUT_ROOT" \
  --model_size mini \
  --device "$DEVICE" \
  --seed "$SEED" \
  --slate_size "$SLATE_SIZE" \
  --episode_batch_size 32 \
  --max_steps_per_episode 20 \
  --beam_width "$BEAM_WIDTH" \
  --initial_temper 20 \
  --item_correlation 0.2 \
  --max_hist_items 50 \
  --num_iters "$NUM_ITERS" \
  --rollout_episodes "$ROLLOUT_EPISODES" \
  --eval_episodes "$EVAL_EPISODES" \
  --eval_every 1 \
  --replay_recent_iters 1 \
  --rollout_phase2_blend_scale 0.20 \
  --rollout_random_topk_sample 10 \
  --rollout_random_item_prob 0.0 \
  --eval_use_phase2_blend \
  --eval_phase2_blend_scale 0.20 \
  --eval_random_topk_sample 10 \
  --eval_random_item_prob 0.0 \
  --critic_batch_size 96 \
  --critic_epochs 3 \
  --critic_lr 1e-3 \
  --critic_weight_decay 1e-4 \
  --critic_valid_ratio 0.15 \
  --critic_arch v9add \
  --critic_item_dim 128 \
  --critic_model_dim 256 \
  --critic_num_heads 8 \
  --critic_num_layers 3 \
  --critic_dropout 0.1 \
  --critic_ensemble_size 5 \
  --critic_pessimism_beta 1.0 \
  --critic_eval_batch_size 512 \
  --critic_target_heuristic_mix 0.60 \
  --critic_target_support_mix 0.25 \
  --critic_target_response_mix 0.15 \
  --critic_page_loss_scale 1.0 \
  --critic_item_loss_scale 0.75 \
  --critic_prefix_loss_scale 0.50 \
  --critic_page_huber_beta 1.0 \
  --critic_item_huber_beta 0.10 \
  --critic_prefix_huber_beta 0.05 \
  --critic_rank_loss_scale 0.15 \
  --critic_monotonic_loss_scale 0.05 \
  --critic_rank_min_gap 0.05 \
  --actor_method grpo \
  --actor_update_every 1 \
  --rollout_policy_sync_mode ema \
  --rollout_policy_sync_tau 0.20 \
  --actor_batch_size 256 \
  --actor_epochs 2 \
  --actor_lr 2e-6 \
  --actor_weight_decay 1e-4 \
  --actor_train_scope decoder_only \
  --actor_token_adv_field sid_advantage_pess \
  --actor_item_adv_field item_advantage_pess \
  --actor_page_reward_field page_q_pess \
  --actor_item_adv_scale 0.10 \
  --actor_page_gate_scale 0.10 \
  --actor_page_gate_min 0.90 \
  --actor_page_gate_max 1.10 \
  --actor_page_gate_mode signed_tanh \
  --actor_positive_topk 2 \
  --actor_positive_floor 0.0 \
  --actor_negative_topk 2 \
  --actor_negative_floor 0.0 \
  --actor_credit_clip 3.0 \
  --actor_renorm_mode batch_abs \
  --actor_clip_eps 0.08 \
  --actor_kl_scale 0.10 \
  --actor_adaptive_kl_support_scale 1.00 \
  --actor_adaptive_kl_unc_scale 0.25 \
  --actor_adaptive_clip_support_scale 1.00 \
  --actor_adaptive_clip_unc_scale 0.25 \
  --actor_min_clip_eps 0.03 \
  --actor_trust_support_field support_gap_scaled \
  --actor_trust_unc_field uncertainty_ratio \
  --actor_entropy_scale 0.0 \
  --actor_sft_scale 0.0 \
  --actor_group_size 8 \
  --actor_group_beam_width 64 \
  --actor_group_num_shards 8 \
  --actor_group_reward_field adaptive_support_pess \
  --actor_group_reward_transform raw \
  --actor_group_support_penalty_scale 0.10 \
  --actor_group_support_gap_temperature 0.50 \
  --actor_group_support_gap_clip 4.0 \
  --actor_group_adaptive_beta_unc_scale 0.25 \
  --actor_group_adaptive_beta_support_scale 0.25 \
  --plot_dpi 160

