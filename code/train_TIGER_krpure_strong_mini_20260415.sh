#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="${ROOT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date +%Y%m%d_%H%M%S)}"

DATA_PATH="${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv"
SID_PATH="${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
UIRM_LOG="${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log"

SAVE_PATH="${SAVE_PATH:-${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_mini_strong_seed2026_${TIMESTAMP_TAG}.pth}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${ROOT_PATH}/output/KuaiRand_Pure/env/log/tiger_sid_krpure_mini_strong_seed2026_${TIMESTAMP_TAG}.log}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${ROOT_PATH}/output/KuaiRand_Pure/env/log/tiger_sid_krpure_mini_strong_seed2026_eval2026_${TIMESTAMP_TAG}.log}"

mkdir -p "$(dirname "${SAVE_PATH}")" "$(dirname "${TRAIN_LOG_PATH}")" "$(dirname "${EVAL_LOG_PATH}")"

echo "[train] save_path=${SAVE_PATH}"
echo "[train] train_log=${TRAIN_LOG_PATH}"
echo "[train] eval_log=${EVAL_LOG_PATH}"

"${PYTHON_BIN}" "${ROOT_PATH}/code/train_TIGER_krpure.py" \
  --log_paths "${DATA_PATH}" \
  --sid_mapping_path "${SID_PATH}" \
  --max_hist_items 50 \
  --batch_size 256 \
  --infer_size 512 \
  --train_num_workers 4 \
  --val_num_workers 4 \
  --pin_memory \
  --num_epochs 20 \
  --early_stop 6 \
  --model_size mini \
  --beam_size 50 \
  --topk_list 5 10 20 \
  --lr 1e-4 \
  --seed 2026 \
  --device cuda \
  --save_path "${SAVE_PATH}" \
  --log_path "${TRAIN_LOG_PATH}"

"${PYTHON_BIN}" "${ROOT_PATH}/code/eval_tiger_phase2_blend_env.py" \
  --tiger_ckpt "${SAVE_PATH}" \
  --sid_mapping_path "${SID_PATH}" \
  --uirm_log_path "${UIRM_LOG}" \
  --slate_size 6 \
  --episode_batch_size 32 \
  --model_size mini \
  --num_episodes 200 \
  --max_steps_per_episode 20 \
  --max_step_per_episode 20 \
  --beam_width 16 \
  --single_response \
  --initial_temper 20 \
  --item_correlation 0.2 \
  --seed 2026 \
  --max_hist_items 50 \
  --device cuda:0 \
  --phase2_blend_scale 0.20 \
  --fast_base_generate \
  --random_topk_sample 10 \
  --log_every 50 \
  > "${EVAL_LOG_PATH}" 2>&1

echo "[done] strict eval log saved to ${EVAL_LOG_PATH}"
