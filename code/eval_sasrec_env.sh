#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -e

ROOT_PATH="${PROJECT_ROOT}"
KUAI_ROOT="${KUAI_ROOT:-${ROOT_PATH}/../KuaiSim-main}"

SASREC_CKPT="${SASREC_CKPT:-${KUAI_ROOT}/code/checkpoints/checkpoints/sasrec/best.pt}"
UIRM_LOG_PATH="${UIRM_LOG_PATH:-${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log}"
LEAVE_MODEL_PATH="${LEAVE_MODEL_PATH:-${ROOT_PATH}/code/output/Kuairand_Pure/env/user_exit_model_v3.pt}"

python "${ROOT_PATH}/code/eval_sasrec_env.py" \
  --sasrec_ckpt "${SASREC_CKPT}" \
  --num_episodes 300 \
  --uirm_log_path "${UIRM_LOG_PATH}" \
  --slate_size 1 \
  --max_steps_per_episode 20 \
  --seed 2026 \
  --single_response \
  --use_leave_model \
  --leave_model_path "${LEAVE_MODEL_PATH}" \
  --leave_logit_temperature 1.0 \
  --leave_logit_bias -0.2 \
  --leave_min_survival_steps 2 \
  --leave_early_step_max 5 \
  --leave_early_prob_scale 0.65
