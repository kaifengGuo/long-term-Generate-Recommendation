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
  GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
  DPO_EVAL_EVERY="${DPO_EVAL_EVERY:-100000}"
  GRPO_EVAL_EVERY="${GRPO_EVAL_EVERY:-100000}"
else
  EVAL_EPISODES="${EVAL_EPISODES:-200}"
  DPO_EPOCHS="${DPO_EPOCHS:-5}"
  GRPO_EPOCHS="${GRPO_EPOCHS:-5}"
  DPO_EVAL_EVERY="${DPO_EVAL_EVERY:-200}"
  GRPO_EVAL_EVERY="${GRPO_EVAL_EVERY:-200}"
fi

OUT_ROOT="${OUT_ROOT:-$ROOT/results/baselines/onerec_fixed_posttrain_strict_eval3_${TIMESTAMP_TAG}}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/results/baselines/posttrain_onerec_ckpts_seed${TRAIN_SEED}}"
BASE_DIR="${BASE_DIR:-$CKPT_ROOT/onerec_base_${MODEL_SIZE}_h2_e5}"

DPO_COEF="${DPO_COEF:-0.2}"
DPO_SFT_COEF="${DPO_SFT_COEF:-1.0}"
DPO_BETA="${DPO_BETA:-0.01}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_LAST_N="${DPO_LAST_N:-2}"
DPO_NEG_PICK="${DPO_NEG_PICK:-rank_range}"
DPO_NEG_LOW="${DPO_NEG_LOW:-2}"
DPO_NEG_HIGH="${DPO_NEG_HIGH:-12}"
SAFE_DPO_DIR="${SAFE_DPO_DIR:-$CKPT_ROOT/safe_dpo_sft_${MODEL_SIZE}_h2_e${DPO_EPOCHS}_dpo${DPO_COEF}_sft${DPO_SFT_COEF}_last${DPO_LAST_N}_lr${DPO_LR}}"

GRPO_REWARD_MODE="${GRPO_REWARD_MODE:-hybrid_rere_ltv_lcb}"
GRPO_LTV_BETA="${GRPO_LTV_BETA:-1.0}"
GRPO_LTV_MIX_RERE="${GRPO_LTV_MIX_RERE:-0.5}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-16}"
GRPO_POLICY_COEF="${GRPO_POLICY_COEF:-0.2}"
GRPO_SFT_COEF="${GRPO_SFT_COEF:-1.0}"
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_KL_COEF="${GRPO_KL_COEF:-0.001}"
GRPO_ADD_GT="${GRPO_ADD_GT:-1}"
GRPO_SKIP_NOHIT="${GRPO_SKIP_NOHIT:-1}"
GRPO_W_RANK="${GRPO_W_RANK:-0.5}"
HYBRID_GRPO_DIR="${HYBRID_GRPO_DIR:-$CKPT_ROOT/hybrid_ltv_grpo_${MODEL_SIZE}_h2_e${GRPO_EPOCHS}_mix${GRPO_LTV_MIX_RERE}_p${GRPO_POLICY_COEF}_sft${GRPO_SFT_COEF}_lr${GRPO_LR}}"

LOG_CSV="code/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv"
USER_FEAT="code/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv"
SID_MAP="code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
UIRM_LOG="code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log"

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

rows = list(csv.DictReader(Path(sys.argv[1]).open("r", encoding="utf-8"), delimiter="\t"))
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
        out[f"{key}_mean"] = f"{(statistics.mean(vals) if vals else 0.0):.4f}"
        out[f"{key}_std"] = f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.4f}"
    summary_rows.append(out)

fields = ["policy", "train_seed", "ckpt", "n_eval_seeds", "eval_seeds"] + [
    f"{k}_{s}" for k in metric_keys for s in ("mean", "std")
]
with Path(sys.argv[2]).open("w", encoding="utf-8", newline="") as f:
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
  log_msg "ERROR: missing OneRec base checkpoint at $BASE_DIR"
  exit 1
}

train_safe_dpo() {
  if [[ -f "$SAFE_DPO_DIR/best.pt" ]]; then
    log_msg "reuse Safe DPO+SFT checkpoint from $SAFE_DPO_DIR"
    return
  fi
  mkdir -p "$SAFE_DPO_DIR"
  local init_ckpt
  init_ckpt="$(best_or_last_ckpt "$BASE_DIR")"
  log_msg "train Safe DPO+SFT: init=$init_ckpt dir=$SAFE_DPO_DIR epochs=$DPO_EPOCHS dpo_coef=$DPO_COEF sft_coef=$DPO_SFT_COEF neg=$DPO_NEG_PICK last_n=$DPO_LAST_N"
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
    --neg_beam 16 \
    --neg_pick "$DPO_NEG_PICK" \
    --neg_rank_low "$DPO_NEG_LOW" \
    --neg_rank_high "$DPO_NEG_HIGH" \
    --beta "$DPO_BETA" \
    --dpo_coef "$DPO_COEF" \
    --sft_coef "$DPO_SFT_COEF" \
    --dpo_last_n "$DPO_LAST_N" \
    --use_old_model 1 \
    --old_model_update_every 1000 \
    --batch_size 2048 \
    --num_workers 8 \
    --lr "$DPO_LR" \
    --epochs "$DPO_EPOCHS" \
    --eval_beam 50 \
    --eval_every "$DPO_EVAL_EVERY" \
    --log_every 50 \
    --model_dir "$SAFE_DPO_DIR" \
    > "$SAFE_DPO_DIR/train.log" 2>&1
}

train_hybrid_grpo() {
  if [[ -f "$HYBRID_GRPO_DIR/best.pt" ]]; then
    log_msg "reuse Hybrid LTV-GRPO checkpoint from $HYBRID_GRPO_DIR"
    return
  fi
  mkdir -p "$HYBRID_GRPO_DIR"
  local init_ckpt
  init_ckpt="$(best_or_last_ckpt "$BASE_DIR")"
  log_msg "train Hybrid LTV-GRPO: init=$init_ckpt dir=$HYBRID_GRPO_DIR epochs=$GRPO_EPOCHS reward=$GRPO_REWARD_MODE mix=$GRPO_LTV_MIX_RERE add_gt=$GRPO_ADD_GT skip_nohit=$GRPO_SKIP_NOHIT"
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
    --lr "$GRPO_LR" \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --epochs "$GRPO_EPOCHS" \
    --reward_mode "$GRPO_REWARD_MODE" \
    --ltv_beta "$GRPO_LTV_BETA" \
    --ltv_unknown_policy global_lcb \
    --ltv_mix_rere "$GRPO_LTV_MIX_RERE" \
    --policy_coef "$GRPO_POLICY_COEF" \
    --group_size "$GRPO_GROUP_SIZE" \
    --w_rank "$GRPO_W_RANK" \
    --sft_coef "$GRPO_SFT_COEF" \
    --ref_update_mode ema \
    --ref_update_tau 0.05 \
    --ref_update_every 1 \
    --kl_coef "$GRPO_KL_COEF" \
    --use_old_model 0 \
    --old_model_update_every 50 \
    --add_gt "$GRPO_ADD_GT" \
    --skip_nohit "$GRPO_SKIP_NOHIT" \
    --log_every 50 \
    --eval_every "$GRPO_EVAL_EVERY" \
    --eval_beam 50 \
    --topk "1,5,10,20,50" \
    --init_ckpt "$init_ckpt" \
    --model_dir "$HYBRID_GRPO_DIR" \
    > "$HYBRID_GRPO_DIR/train.log" 2>&1
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

log_msg "onerec_fixed_posttrain_strict_eval3_start"
log_msg "out_root=$OUT_ROOT"
log_msg "claim_boundary=pure OneRec policy scoring only; strict env; rerank_alpha=0; no eval-time simulator/value/oracle rerank."
log_msg "fixes=Safe DPO uses SFT anchor, rank-range negatives, last-token DPO, lower LR; Hybrid GRPO uses ReRe+LTV-LCB reward, add_gt, skip_nohit, stronger SFT anchor."
log_msg "eval_seeds=${EVAL_SEEDS[*]} eval_episodes=$EVAL_EPISODES smoke=$SMOKE"

ensure_base
BASE_CKPT="$(best_or_last_ckpt "$BASE_DIR")"
eval_ckpt "OneRec base" "$BASE_CKPT"

train_safe_dpo
eval_ckpt "Safe DPO+SFT" "$(best_or_last_ckpt "$SAFE_DPO_DIR")"

train_hybrid_grpo
eval_ckpt "Hybrid LTV-GRPO" "$(best_or_last_ckpt "$HYBRID_GRPO_DIR")"

summarize_runs
log_msg "onerec_fixed_posttrain_strict_eval3_done"
cat "$OUT_ROOT/summary.tsv"
