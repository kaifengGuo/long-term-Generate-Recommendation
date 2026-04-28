#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_PATH="${PROJECT_ROOT}"

TRACE_PATH="${TRACE_PATH:-${ROOT_PATH}/results/phase2_blend_trace.jsonl}"
MODEL_SIZE="${MODEL_SIZE:-mini}"
DEVICE="${DEVICE:-cuda:0}"

python "${ROOT_PATH}/code/train_tiger_phase2_blend.py" \
  --trace_dir "${TRACE_PATH}" \
  --uirm_log_path "${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.abs.model.log" \
  --tiger_ckpt "${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_${MODEL_SIZE}.pth" \
  --sid_mapping_path "${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv" \
  --model_size "${MODEL_SIZE}" \
  --device "${DEVICE}" \
  --epochs 8 \
  --batch_size 128 \
  --gamma 0.9 \
  --save_dir "${ROOT_PATH}/output/KuaiRand_Pure/env/phase2_blend_${MODEL_SIZE}" \
  --metrics_out "${ROOT_PATH}/results/phase2_blend_${MODEL_SIZE}_metrics.json"
