#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_PATH="${PROJECT_ROOT}"

MODEL_SIZE="${MODEL_SIZE:-mini}"
DEVICE="${DEVICE:-cuda:0}"
TRACE_PATH="${TRACE_PATH:-}"
HEAD_PATH="${HEAD_PATH:-}"
META_PATH="${META_PATH:-}"
BLEND_SCALE="${BLEND_SCALE:-0.20}"

ARGS=(
  --tiger_ckpt "${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_${MODEL_SIZE}.pth"
  --sid_mapping_path "${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
  --uirm_log_path "${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.abs.model.log"
  --slate_size 1
  --episode_batch_size 32
  --model_size "${MODEL_SIZE}"
  --num_episodes 1000
  --max_steps_per_episode 20
  --max_step_per_episode 20
  --beam_width 64
  --single_response
  --seed 2026
  --max_hist_items 50
  --device "${DEVICE}"
  --phase2_blend_scale "${BLEND_SCALE}"
)

if [[ -n "${HEAD_PATH}" ]]; then
  ARGS+=(--phase2_head_path "${HEAD_PATH}")
else
  ARGS+=(--fast_base_generate)
fi

if [[ -n "${META_PATH}" ]]; then
  ARGS+=(--phase2_meta_path "${META_PATH}")
fi

if [[ -n "${TRACE_PATH}" ]]; then
  ARGS+=(--trace_path "${TRACE_PATH}")
fi

python "${ROOT_PATH}/code/eval_tiger_phase2_blend_env.py" "${ARGS[@]}"
