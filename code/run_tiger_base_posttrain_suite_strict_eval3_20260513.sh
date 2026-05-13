#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/baselines/tiger_base_posttrain_strict_eval3_${TIMESTAMP_TAG}}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
SMOKE="${SMOKE:-0}"
METHODS_STR="${METHODS_STR:-dpo_style sprec_style rere_grpo hca_lcb_grpo}"
IFS=' ' read -r -a METHODS <<< "$METHODS_STR"

UIRM_LOG="${UIRM_LOG:-$ROOT/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log}"
SID_MAP="${SID_MAP:-$ROOT/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv}"
BASE_TIGER_CKPT="${TIGER_CKPT:-$ROOT/output/KuaiRand_Pure/env/tiger_sid_krpure_mini_strong_seed2026_20260415_225310.pth}"

TRAIN_SEED="${TRAIN_SEED:-2026}"
EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-111 112 113}"
IFS=' ' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_STR"

if [[ "$SMOKE" == "1" ]]; then
  ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-96}"
  INTERNAL_EVAL_EPISODES="${INTERNAL_EVAL_EPISODES:-20}"
  STRICT_EVAL_EPISODES="${STRICT_EVAL_EPISODES:-20}"
  CRITIC_EPOCHS="${CRITIC_EPOCHS:-1}"
  ACTOR_EPOCHS="${ACTOR_EPOCHS:-1}"
  ACTOR_GROUP_MAX_ROWS="${ACTOR_GROUP_MAX_ROWS:-512}"
  EVAL_SEEDS=("${EVAL_SEEDS[0]}")
else
  ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-2500}"
  INTERNAL_EVAL_EPISODES="${INTERNAL_EVAL_EPISODES:-200}"
  STRICT_EVAL_EPISODES="${STRICT_EVAL_EPISODES:-200}"
  CRITIC_EPOCHS="${CRITIC_EPOCHS:-3}"
  ACTOR_EPOCHS="${ACTOR_EPOCHS:-2}"
  ACTOR_GROUP_MAX_ROWS="${ACTOR_GROUP_MAX_ROWS:-0}"
fi

mkdir -p "$OUT_ROOT"
cd "$ROOT"

DRIVER_LOG="$OUT_ROOT/suite.log"
PER_RUN_TSV="$OUT_ROOT/per_run.tsv"
SUMMARY_TSV="$OUT_ROOT/summary.tsv"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

wait_for_pid_if_needed() {
  if [[ -n "$WAIT_FOR_PID" ]]; then
    log "waiting for existing run pid=$WAIT_FOR_PID"
    while kill -0 "$WAIT_FOR_PID" >/dev/null 2>&1; do
      sleep 60
    done
    log "waited pid finished: $WAIT_FOR_PID"
  fi
}

parse_eval_log() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()

def grab(patterns, default=""):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return default

def rate(names):
    for name in names:
        value = grab(rf"\b{re.escape(name)}:\s*\d+/\d+\s*\(([-0-9.]+)%\)")
        if value != "":
            return value
    return ""

vals = [
    grab([r"Total Reward:\s*([-0-9.]+)", r"Total reward:\s*([-0-9.]+)"]),
    grab(r"Depth:\s*([-0-9.]+)"),
    grab([r"Avg Step Reward:\s*([-0-9.]+)", r"Average reward:\s*([-0-9.]+)"]),
    grab(r"Coverage:\s*([-0-9.]+)"),
    grab(r"ILD:\s*([-0-9.]+)"),
    rate(["is_click", "click"]),
    rate(["long_view"]),
    rate(["is_like", "like"]),
    rate(["is_comment", "comment"]),
    rate(["is_forward", "forward"]),
    rate(["is_follow", "follow"]),
    rate(["is_hate", "hate"]),
]
print("\t".join(vals))
PY
}

write_summary() {
  "$PYTHON_BIN" - "$PER_RUN_TSV" "$SUMMARY_TSV" <<'PY'
import csv
import math
import sys
from collections import defaultdict

src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, "r", encoding="utf-8"), delimiter="\t"))
metrics = [
    "total_reward", "depth", "avg_reward", "coverage", "ild", "click_pct",
    "long_view_pct", "like_pct", "comment_pct", "forward_pct", "follow_pct", "hate_pct",
]

def to_float(value):
    try:
        return float(value)
    except Exception:
        return None

def mean_std(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return "", ""
    mean = sum(vals) / len(vals)
    if len(vals) <= 1:
        return f"{mean:.4f}", "0.0000"
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return f"{mean:.4f}", f"{math.sqrt(var):.4f}"

groups = defaultdict(list)
for row in rows:
    key = (row["policy"], row["variant"], row["train_seed"], row["ckpt"])
    groups[key].append(row)

fieldnames = ["policy", "variant", "train_seed", "ckpt", "n_eval_seeds", "eval_seeds"]
for metric in metrics:
    fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

with open(dst, "w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for key, group_rows in sorted(groups.items()):
        policy, variant, train_seed, ckpt = key
        out = {
            "policy": policy,
            "variant": variant,
            "train_seed": train_seed,
            "ckpt": ckpt,
            "n_eval_seeds": str(len(group_rows)),
            "eval_seeds": ",".join(row["eval_seed"] for row in group_rows),
        }
        for metric in metrics:
            mean, std = mean_std([to_float(row.get(metric, "")) for row in group_rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        writer.writerow(out)
PY
}

eval_tiger_ckpt() {
  local policy="$1"
  local variant="$2"
  local ckpt="$3"
  require_file "$ckpt"
  for eval_seed in "${EVAL_SEEDS[@]}"; do
    local log_path="$OUT_ROOT/eval_${policy}_${variant}_seed${eval_seed}.log"
    log "strict eval policy=$policy variant=$variant seed=$eval_seed ckpt=$ckpt"
    "$PYTHON_BIN" code/eval_tiger_phase2_blend_env.py \
      --tiger_ckpt "$ckpt" \
      --sid_mapping_path "$SID_MAP" \
      --uirm_log_path "$UIRM_LOG" \
      --slate_size 6 \
      --episode_batch_size 32 \
      --model_size mini \
      --num_episodes "$STRICT_EVAL_EPISODES" \
      --max_steps_per_episode 20 \
      --max_step_per_episode 20 \
      --beam_width 16 \
      --single_response \
      --initial_temper 20 \
      --item_correlation 0.2 \
      --seed "$eval_seed" \
      --max_hist_items 50 \
      --device cuda:0 \
      --phase2_blend_scale 0.20 \
      --fast_base_generate \
      --random_topk_sample 10 \
      --log_every 50 \
      > "$log_path" 2>&1
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$policy" "$variant" "$TRAIN_SEED" "$eval_seed" "$ckpt" "$(parse_eval_log "$log_path")" \
      >> "$PER_RUN_TSV"
  done
  write_summary
}

run_closed_loop_method() {
  local method="$1"
  local run_dir="$OUT_ROOT/$method"
  mkdir -p "$run_dir"
  log "run TIGER post-train method=$method out=$run_dir"

  local common_args=(
    code/tiger_page_sid_rl/run_page_sid_closed_loop.py
    --tiger_ckpt "$BASE_TIGER_CKPT"
    --uirm_log_path "$UIRM_LOG"
    --sid_mapping_path "$SID_MAP"
    --output_root "$run_dir"
    --model_size mini
    --device cuda:0
    --seed "$TRAIN_SEED"
    --slate_size 6
    --episode_batch_size 32
    --max_steps_per_episode 20
    --beam_width 16
    --initial_temper 20
    --item_correlation 0.2
    --max_hist_items 50
    --num_iters 1
    --rollout_episodes "$ROLLOUT_EPISODES"
    --eval_episodes "$INTERNAL_EVAL_EPISODES"
    --eval_every 1
    --replay_recent_iters 1
    --rollout_phase2_blend_scale 0.20
    --rollout_random_topk_sample 10
    --rollout_random_item_prob 0.0
    --eval_use_phase2_blend
    --eval_phase2_blend_scale 0.20
    --eval_random_topk_sample 10
    --eval_random_item_prob 0.0
    --critic_batch_size 96
    --critic_epochs "$CRITIC_EPOCHS"
    --critic_lr 1e-3
    --critic_weight_decay 1e-4
    --critic_valid_ratio 0.15
    --critic_arch v9add
    --critic_item_dim 128
    --critic_model_dim 256
    --critic_num_heads 8
    --critic_num_layers 3
    --critic_dropout 0.1
    --critic_ensemble_size 5
    --critic_pessimism_beta 1.0
    --critic_eval_batch_size 512
    --critic_target_heuristic_mix 0.60
    --critic_target_support_mix 0.25
    --critic_target_response_mix 0.15
    --critic_page_loss_scale 1.0
    --critic_item_loss_scale 0.75
    --critic_prefix_loss_scale 0.50
    --critic_page_huber_beta 1.0
    --critic_item_huber_beta 0.10
    --critic_prefix_huber_beta 0.05
    --critic_rank_loss_scale 0.15
    --critic_monotonic_loss_scale 0.05
    --critic_rank_min_gap 0.05
    --rollout_policy_sync_mode ema
    --rollout_policy_sync_tau 0.20
    --rollout_policy_acceptance always
    --actor_batch_size 256
    --actor_epochs "$ACTOR_EPOCHS"
    --actor_lr 2e-6
    --actor_weight_decay 1e-4
    --actor_train_scope decoder_only
    --actor_group_size 8
    --actor_group_beam_width 64
    --actor_group_num_shards 8
    --actor_group_max_rows "$ACTOR_GROUP_MAX_ROWS"
    --actor_group_advantage_mode zscore
    --actor_item_adv_scale 0.10
    --actor_page_gate_scale 0.10
    --actor_page_gate_min 0.90
    --actor_page_gate_max 1.10
    --actor_page_gate_mode signed_tanh
    --actor_positive_topk 2
    --actor_positive_floor 0.0
    --actor_negative_topk 2
    --actor_negative_floor 0.0
    --actor_credit_clip 3.0
    --actor_renorm_mode batch_abs
    --actor_clip_eps 0.10
    --actor_kl_scale 0.05
    --actor_entropy_scale 0.0
    --actor_sft_scale 0.0
    --plot_dpi 160
  )

  local method_args=()
  case "$method" in
    dpo_style)
      method_args=(
        --actor_method pref
        --actor_group_reward_field page_q_mean
        --actor_group_support_penalty_scale 0.0
        --actor_group_adaptive_beta_unc_scale 0.0
        --actor_group_adaptive_beta_support_scale 0.0
        --actor_pref_score_field page_q_mean
        --actor_pref_safe_support_gap_max 999.0
        --actor_pref_min_gap 0.0
        --actor_pref_max_pairs_per_group 2
        --actor_pref_pair_mode safe_vs_behavior
        --actor_pref_beta 1.0
        --actor_pref_sft_scale 0.05
        --actor_pref_gap_scale 1.0
        --actor_pref_gap_clip 2.0
        --actor_pref_score_normalization mean_token
        --actor_pref_attr_pair_scale 0.0
      )
      ;;
    sprec_style)
      method_args=(
        --actor_method pref
        --actor_group_reward_field adaptive_support_pess
        --actor_group_support_penalty_scale 0.10
        --actor_group_support_gap_temperature 0.50
        --actor_group_support_gap_clip 4.0
        --actor_group_adaptive_beta_unc_scale 0.25
        --actor_group_adaptive_beta_support_scale 0.25
        --actor_pref_score_field adaptive_support_pess
        --actor_pref_safe_support_gap_max 0.25
        --actor_pref_min_gap 0.02
        --actor_pref_max_pairs_per_group 4
        --actor_pref_pair_mode mixed
        --actor_pref_pair_score_gap_scale 1.0
        --actor_pref_pair_raw_q_gap_scale 0.5
        --actor_pref_pair_unc_gap_scale 0.25
        --actor_pref_pair_support_gap_scale 0.75
        --actor_pref_beta 1.0
        --actor_pref_sft_scale 0.05
        --actor_pref_gap_scale 1.0
        --actor_pref_gap_clip 4.0
        --actor_pref_score_normalization mean_token
        --actor_pref_attr_adv_mode pess
        --actor_pref_attr_pair_scale 0.25
        --actor_pref_attr_item_scale 0.10
      )
      ;;
    rere_grpo)
      method_args=(
        --actor_method grpo
        --actor_token_adv_field sid_advantage
        --actor_item_adv_field item_advantage
        --actor_page_reward_field page_q_mean
        --actor_group_reward_field page_q_mean
        --actor_group_support_penalty_scale 0.0
        --actor_group_adaptive_beta_unc_scale 0.0
        --actor_group_adaptive_beta_support_scale 0.0
        --actor_attr_weight_mode separate_relu
        --actor_attr_fallback_mode abs_max
        --actor_adv_combine_mode multiplicative
      )
      ;;
    hca_lcb_grpo)
      method_args=(
        --actor_method grpo
        --actor_credit_mode pessimistic_strong
        --actor_token_adv_field sid_advantage_pess
        --actor_item_adv_field item_advantage_pess
        --actor_page_reward_field page_q_pess
        --actor_group_reward_field page_q_pess
        --actor_group_support_penalty_scale 0.0
        --actor_group_adaptive_beta_unc_scale 0.0
        --actor_group_adaptive_beta_support_scale 0.0
        --actor_attr_weight_mode signed_combined
        --actor_attr_fallback_mode abs_max
        --actor_adv_combine_mode additive_zero_sum
        --actor_hca_residual_scale 0.50
        --actor_seq_clip_eps 0.10
        --actor_seq_kl_scale 0.02
      )
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 1
      ;;
  esac

  "$PYTHON_BIN" "${common_args[@]}" "${method_args[@]}" 2>&1 | tee "$run_dir/launch_stdout.log"

  local learner_ckpt=""
  case "$method" in
    dpo_style|sprec_style)
      learner_ckpt="$run_dir/iter_01/pref_actor/tiger_hca_pref_actor_tiger.pth"
      ;;
    rere_grpo|hca_lcb_grpo)
      learner_ckpt="$run_dir/iter_01/grpo_actor/tiger_hca_grpo_actor_tiger.pth"
      ;;
  esac
  local rollout_ckpt="$run_dir/iter_01/rollout_policy/tiger_rollout_policy_tiger.pth"

  eval_tiger_ckpt "$method" "after_learner" "$learner_ckpt"
  eval_tiger_ckpt "$method" "after_rollout_ema" "$rollout_ckpt"
}

require_file "$BASE_TIGER_CKPT"
require_file "$UIRM_LOG"
require_file "$SID_MAP"

printf 'policy\tvariant\ttrain_seed\teval_seed\tckpt\ttotal_reward\tdepth\tavg_reward\tcoverage\tild\tclick_pct\tlong_view_pct\tlike_pct\tcomment_pct\tforward_pct\tfollow_pct\thate_pct\n' > "$PER_RUN_TSV"

log "tiger_base_posttrain_strict_eval3_start"
log "out_root=$OUT_ROOT"
log "base_tiger_ckpt=$BASE_TIGER_CKPT"
log "methods=${METHODS[*]}"
log "train_seed=$TRAIN_SEED eval_seeds=${EVAL_SEEDS[*]} smoke=$SMOKE"
log "rollout_episodes=$ROLLOUT_EPISODES strict_eval_episodes=$STRICT_EVAL_EPISODES"

wait_for_pid_if_needed

eval_tiger_ckpt "TIGER_base" "base" "$BASE_TIGER_CKPT"

for method in "${METHODS[@]}"; do
  run_closed_loop_method "$method"
done

write_summary
log "done summary=$SUMMARY_TSV per_run=$PER_RUN_TSV"
