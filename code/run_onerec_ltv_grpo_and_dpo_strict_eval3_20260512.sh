#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE="${SMOKE:-0}"
MODEL_SIZE="${MODEL_SIZE:-mini}"
TRAIN_SEED="${TRAIN_SEED:-2026}"
EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-111 112 113}"
IFS=' ' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_STR"

if [[ "$SMOKE" == "1" ]]; then
  EVAL_EPISODES="${EVAL_EPISODES:-20}"
  DPO_EPOCHS="${DPO_EPOCHS:-1}"
  LTV_GRPO_EPOCHS="${LTV_GRPO_EPOCHS:-1}"
  DPO_EVAL_EVERY="${DPO_EVAL_EVERY:-100000}"
  LTV_GRPO_EVAL_EVERY="${LTV_GRPO_EVAL_EVERY:-100000}"
  EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-111}"
  IFS=' ' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_STR"
else
  EVAL_EPISODES="${EVAL_EPISODES:-200}"
  DPO_EPOCHS="${DPO_EPOCHS:-5}"
  LTV_GRPO_EPOCHS="${LTV_GRPO_EPOCHS:-5}"
  DPO_EVAL_EVERY="${DPO_EVAL_EVERY:-200}"
  LTV_GRPO_EVAL_EVERY="${LTV_GRPO_EVAL_EVERY:-200}"
fi

OUT_ROOT="${OUT_ROOT:-$ROOT/results/baselines/onerec_ltv_grpo_dpo_strict_eval3_${TIMESTAMP_TAG}}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/results/baselines/posttrain_onerec_ckpts_seed${TRAIN_SEED}}"
BASE_DIR="${BASE_DIR:-$CKPT_ROOT/onerec_base_${MODEL_SIZE}_h2_e5}"
PLAIN_DPO_DIR="${PLAIN_DPO_DIR:-$CKPT_ROOT/plain_dpo_${MODEL_SIZE}_h2_e${DPO_EPOCHS}}"
LTV_GRPO_DIR="${LTV_GRPO_DIR:-$CKPT_ROOT/ltv_lcb_grpo_${MODEL_SIZE}_h2_e${LTV_GRPO_EPOCHS}_beta${LTV_BETA:-1.0}}"

LOG_CSV="code/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv"
USER_FEAT="code/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv"
SID_MAP="code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
UIRM_LOG="code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log"

LTV_BETA="${LTV_BETA:-1.0}"
LTV_GROUP_SIZE="${LTV_GROUP_SIZE:-16}"
LTV_LR="${LTV_LR:-1e-5}"
LTV_KL_COEF="${LTV_KL_COEF:-0.001}"
LTV_SFT_COEF="${LTV_SFT_COEF:-0.5}"

mkdir -p "$OUT_ROOT" "$CKPT_ROOT"

log_msg() {
  echo "[$(date '+%F %T')] $*" | tee -a "$OUT_ROOT/suite.log"
}

parse_eval_log() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()

def grab(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            for idx in range(1, (match.lastindex or 0) + 1):
                val = match.group(idx)
                if val:
                    return val
    return ""

vals = [
    grab(r"Total Reward:\s*([0-9.\-]+)"),
    grab(r"Depth:\s*([0-9.\-]+)"),
    grab([r"Average reward:\s*([0-9.\-]+)", r"Avg Step Reward:\s*([0-9.\-]+)"]),
    grab(r"Coverage:\s*([0-9.\-]+)"),
    grab(r"ILD:\s*([0-9.\-]+)"),
    grab(r"is_click:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
    grab(r"long_view:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
    grab(r"is_like:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
    grab(r"is_comment:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
    grab(r"is_forward:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
    grab(r"is_follow:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
    grab(r"is_hate:\s*\d+/\d+\s*\(([0-9.\-]+)%\)"),
]
print("\t".join(vals))
PY
}

summarize_runs() {
  "$PYTHON_BIN" - "$OUT_ROOT/per_run.tsv" "$OUT_ROOT/summary.tsv" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = list(csv.DictReader(src.open("r", encoding="utf-8"), delimiter="\t"))
metric_keys = [
    "total_reward", "depth", "avg_reward", "coverage", "ild", "click_pct",
    "long_view_pct", "like_pct", "comment_pct", "forward_pct", "follow_pct",
    "hate_pct",
]
by_policy = {}
for row in rows:
    by_policy.setdefault(row["policy"], []).append(row)

summary_rows = []
for policy, items in by_policy.items():
    out = {
        "policy": policy,
        "train_seed": items[0]["train_seed"],
        "ckpt": items[0]["ckpt"],
        "n_eval_seeds": str(len(items)),
        "eval_seeds": ",".join(x["eval_seed"] for x in items),
    }
    for key in metric_keys:
        vals = [float(x[key]) for x in items if x.get(key, "") != ""]
        mean = statistics.mean(vals) if vals else 0.0
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[f"{key}_mean"] = f"{mean:.4f}"
        out[f"{key}_std"] = f"{std:.4f}"
    summary_rows.append(out)

fields = ["policy", "train_seed", "ckpt", "n_eval_seeds", "eval_seeds"] + [
    f"{k}_{s}" for k in metric_keys for s in ("mean", "std")
]
with dst.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(summary_rows)
PY
}

best_or_last_ckpt() {
  local dir="$1"
  if [[ -f "$dir/best.pt" ]]; then
    echo "$dir/best.pt"
  elif [[ -f "$dir/last.pt" ]]; then
    echo "$dir/last.pt"
  else
    find "$dir" -maxdepth 1 -type f -name 'epoch_*.pt' | sort -V | tail -1
  fi
}

ensure_base() {
  if [[ -f "$BASE_DIR/best.pt" || -f "$BASE_DIR/last.pt" ]]; then
    log_msg "reuse OneRec base checkpoint from $BASE_DIR"
    return
  fi
  mkdir -p "$BASE_DIR"
  log_msg "train missing OneRec base: dir=$BASE_DIR"
  "$PYTHON_BIN" code/train_onerec_value.py \
    --log_paths "$LOG_CSV" \
    --sid_mapping_path "$SID_MAP" \
    --user_feat_path "$USER_FEAT" \
    --label_col is_click \
    --model_dir "$BASE_DIR" \
    --model_size "$MODEL_SIZE" \
    --max_hist_len 50 \
    --sid_depth 4 \
    --num_classes 32 \
    --num_layers 3 \
    --hist_num_layers 2 \
    --nhead 4 \
    --device cuda:0 \
    --seed "$TRAIN_SEED" \
    --epochs 5 \
    --batch_size 2048 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --num_workers 8 \
    --gamma 0.99 \
    --beam_width 50 \
    --topk_list 1,5,10,20 \
    --w_rev 0.0 \
    --w_ltv 0.0 \
    --w_cls 1.0 \
    --value_token_weights "0,0,0,1" \
    --use_user_token \
    > "$BASE_DIR/train.log" 2>&1
}

train_plain_dpo() {
  if [[ -f "$PLAIN_DPO_DIR/best.pt" ]]; then
    log_msg "reuse Plain DPO checkpoint from $PLAIN_DPO_DIR"
    return
  fi
  mkdir -p "$PLAIN_DPO_DIR"
  local init_ckpt
  init_ckpt="$(best_or_last_ckpt "$BASE_DIR")"
  log_msg "train Plain DPO baseline: init=$init_ckpt dir=$PLAIN_DPO_DIR epochs=$DPO_EPOCHS"
  "$PYTHON_BIN" code/train_s_dpo.py \
    --log_paths "$LOG_CSV" \
    --sid_mapping_path "$SID_MAP" \
    --user_feat_path "$USER_FEAT" \
    --label_col is_click \
    --gt_spec "is_click:1" \
    --gt_gate is_click \
    --hist_from_gt 0 \
    --model_size "$MODEL_SIZE" \
    --num_layers 3 \
    --hist_num_layers 2 \
    --sid_depth 4 \
    --num_classes 32 \
    --init_ckpt "$init_ckpt" \
    --num_neg 1 \
    --neg_beam 8 \
    --neg_pick top \
    --beta 0.01 \
    --dpo_coef 1.0 \
    --sft_coef 0.0 \
    --dpo_last_n 0 \
    --use_old_model 1 \
    --old_model_update_every 1000 \
    --batch_size 2048 \
    --num_workers 8 \
    --lr 5e-6 \
    --epochs "$DPO_EPOCHS" \
    --eval_beam 50 \
    --eval_every "$DPO_EVAL_EVERY" \
    --log_every 50 \
    --model_dir "$PLAIN_DPO_DIR" \
    > "$PLAIN_DPO_DIR/train.log" 2>&1
}

train_ltv_grpo() {
  if [[ -f "$LTV_GRPO_DIR/best.pt" ]]; then
    log_msg "reuse OneRec LTV-LCB GRPO checkpoint from $LTV_GRPO_DIR"
    return
  fi
  mkdir -p "$LTV_GRPO_DIR"
  local init_ckpt
  init_ckpt="$(best_or_last_ckpt "$BASE_DIR")"
  log_msg "train OneRec LTV-LCB GRPO: init=$init_ckpt dir=$LTV_GRPO_DIR epochs=$LTV_GRPO_EPOCHS beta=$LTV_BETA"
  "$PYTHON_BIN" code/train_rere_grpo.py \
    --log_paths "$LOG_CSV" \
    --sid_mapping_path "$SID_MAP" \
    --user_feat_path "$USER_FEAT" \
    --label_col is_click \
    --gt_spec "is_click:1" \
    --gt_gate is_click \
    --hist_from_gt 0 \
    --gamma 0.99 \
    --sid_depth 4 \
    --num_classes 32 \
    --model_size "$MODEL_SIZE" \
    --num_layers 3 \
    --hist_num_layers 2 \
    --batch_size 1024 \
    --num_workers 8 \
    --lr "$LTV_LR" \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --epochs "$LTV_GRPO_EPOCHS" \
    --reward_mode ltv_lcb \
    --ltv_beta "$LTV_BETA" \
    --ltv_unknown_policy global_lcb \
    --policy_coef 0.5 \
    --group_size "$LTV_GROUP_SIZE" \
    --w_rank 0.0 \
    --sft_coef "$LTV_SFT_COEF" \
    --ref_update_mode ema \
    --ref_update_tau 0.05 \
    --ref_update_every 1 \
    --kl_coef "$LTV_KL_COEF" \
    --use_old_model 0 \
    --old_model_update_every 50 \
    --skip_nohit 0 \
    --log_every 50 \
    --eval_every "$LTV_GRPO_EVAL_EVERY" \
    --eval_beam 50 \
    --topk "1,5,10,20,50" \
    --init_ckpt "$init_ckpt" \
    --model_dir "$LTV_GRPO_DIR" \
    > "$LTV_GRPO_DIR/train.log" 2>&1
}

eval_ckpt() {
  local policy="$1"
  local ckpt="$2"
  local policy_id
  policy_id="$(echo "$policy" | tr ' /+' '___')"
  for eval_seed in "${EVAL_SEEDS[@]}"; do
    local log_path="$OUT_ROOT/eval_${policy_id}_seed${eval_seed}.log"
    log_msg "eval $policy seed=$eval_seed ckpt=$ckpt episodes=$EVAL_EPISODES"
    "$PYTHON_BIN" code/eval_onerec_value_rerank.py \
      --onerec_ckpt "$ckpt" \
      --uirm_log_path "$UIRM_LOG" \
      --sid_mapping_path "$SID_MAP" \
      --slate_size 6 \
      --episode_batch_size 32 \
      --single_response \
      --initial_temper 20 \
      --item_correlation 0.2 \
      --num_episodes "$EVAL_EPISODES" \
      --max_steps_per_episode 20 \
      --max_step_per_episode 20 \
      --beam_width 64 \
      --rerank_alpha 0.0 \
      --rerank_formula add \
      --seed "$eval_seed" \
      --sid_depth 4 \
      --num_classes 32 \
      --max_hist_len 50 \
      --device cuda:0 \
      --log_every 50 \
      > "$log_path" 2>&1
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$policy" "$TRAIN_SEED" "$eval_seed" "$ckpt" "$(parse_eval_log "$log_path")" >> "$OUT_ROOT/per_run.tsv"
    summarize_runs
  done
}

printf 'policy\ttrain_seed\teval_seed\tckpt\ttotal_reward\tdepth\tavg_reward\tcoverage\tild\tclick_pct\tlong_view_pct\tlike_pct\tcomment_pct\tforward_pct\tfollow_pct\thate_pct\n' > "$OUT_ROOT/per_run.tsv"

log_msg "onerec_ltv_grpo_dpo_strict_eval3_start"
log_msg "out_root=$OUT_ROOT"
log_msg "ckpt_root=$CKPT_ROOT"
log_msg "claim_boundary=pure OneRec policy scoring only; strict env; rerank_alpha=0; no eval-time simulator/value/oracle rerank."
log_msg "method_note=Plain DPO is log-positive-vs-top-beam-negative; LTV-LCB GRPO uses SID reward mean_LTV-beta*std_LTV from training logs."
log_msg "eval_seeds=${EVAL_SEEDS[*]} eval_episodes=$EVAL_EPISODES smoke=$SMOKE"

ensure_base
BASE_CKPT="$(best_or_last_ckpt "$BASE_DIR")"
eval_ckpt "OneRec base" "$BASE_CKPT"

train_plain_dpo
eval_ckpt "Plain DPO" "$(best_or_last_ckpt "$PLAIN_DPO_DIR")"

train_ltv_grpo
eval_ckpt "OneRec LTV-LCB GRPO" "$(best_or_last_ckpt "$LTV_GRPO_DIR")"

summarize_runs
log_msg "onerec_ltv_grpo_dpo_strict_eval3_done"
cat "$OUT_ROOT/summary.tsv"
