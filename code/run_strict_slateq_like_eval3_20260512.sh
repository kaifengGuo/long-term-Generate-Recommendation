#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/baselines/slateq_like_strict_eval3_${TIMESTAMP_TAG}}"
EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-111 112 113}"
IFS=' ' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_STR"

TRACE_EPISODES="${TRACE_EPISODES:-500}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-10}"
SLATE_VALUE_SCALE="${SLATE_VALUE_SCALE:-0.10}"
SLATE_RERANK_POOL="${SLATE_RERANK_POOL:-12}"
SLATE_GREEDY_CANDIDATES="${SLATE_GREEDY_CANDIDATES:-8}"

mkdir -p "$OUT_ROOT"
cd "$ROOT"

UIRM_LOG="code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log"
SID_MAP="code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
TIGER_CKPT="output/KuaiRand_Pure/env/tiger_sid_krpure_mini_fast1e.pth"
TRACE_PATH="$OUT_ROOT/tiger_base_trace_for_slateq_like.jsonl"
CRITIC_DIR="$OUT_ROOT/slate_value_head"

parse_eval_log() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
def grab(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            for idx in range(1, (m.lastindex or 0) + 1):
                val = m.group(idx)
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

printf 'policy\ttrain_seed\teval_seed\tckpt\ttotal_reward\tdepth\tavg_reward\tcoverage\tild\tclick_pct\tlong_view_pct\tlike_pct\tcomment_pct\tforward_pct\tfollow_pct\thate_pct\n' > "$OUT_ROOT/per_run.tsv"

echo "[$(date '+%F %T')] slateq_like_strict_eval3_start" | tee -a "$OUT_ROOT/suite.log"
echo "out_root=$OUT_ROOT" | tee -a "$OUT_ROOT/suite.log"
echo "eval_seeds=${EVAL_SEEDS[*]}" | tee -a "$OUT_ROOT/suite.log"
echo "claim_boundary=SlateQ-like TIGER slate-value reranker; not canonical SlateQ" | tee -a "$OUT_ROOT/suite.log"
echo "trace_episodes=$TRACE_EPISODES critic_epochs=$CRITIC_EPOCHS slate_value_scale=$SLATE_VALUE_SCALE" | tee -a "$OUT_ROOT/suite.log"

if [ ! -s "$TRACE_PATH" ]; then
  echo "[$(date '+%F %T')] collect TIGER base trace for slate-value training" | tee -a "$OUT_ROOT/suite.log"
  "$PYTHON_BIN" code/eval_tiger_phase2_blend_env.py \
    --tiger_ckpt "$TIGER_CKPT" \
    --sid_mapping_path "$SID_MAP" \
    --uirm_log_path "$UIRM_LOG" \
    --slate_size 6 \
    --episode_batch_size 32 \
    --model_size mini \
    --num_episodes "$TRACE_EPISODES" \
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
    --trace_path "$TRACE_PATH" \
    --log_every 100 \
    > "$OUT_ROOT/collect_trace.log" 2>&1
fi

if [ ! -f "$CRITIC_DIR/slate_value_head.pt" ]; then
  mkdir -p "$CRITIC_DIR"
  echo "[$(date '+%F %T')] train slate-value head" | tee -a "$OUT_ROOT/suite.log"
  "$PYTHON_BIN" code/train_tiger_slate_critic.py \
    --trace_path "$TRACE_PATH" \
    --uirm_log_path "$UIRM_LOG" \
    --sid_mapping_path "$SID_MAP" \
    --device cuda:0 \
    --seed 2026 \
    --gamma 0.9 \
    --target_field return \
    --credit_mode return \
    --max_hist_items 50 \
    --batch_size 128 \
    --epochs "$CRITIC_EPOCHS" \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --hidden_dim 96 \
    --dropout 0.10 \
    --save_dir "$CRITIC_DIR" \
    > "$OUT_ROOT/train_slate_value.log" 2>&1
fi

for eval_seed in "${EVAL_SEEDS[@]}"; do
  log_path="$OUT_ROOT/eval_SlateQLike_seed${eval_seed}.log"
  echo "[$(date '+%F %T')] eval SlateQ-like seed=$eval_seed" | tee -a "$OUT_ROOT/suite.log"
  "$PYTHON_BIN" code/eval_tiger_phase2_blend_env.py \
    --tiger_ckpt "$TIGER_CKPT" \
    --sid_mapping_path "$SID_MAP" \
    --uirm_log_path "$UIRM_LOG" \
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
    --seed "$eval_seed" \
    --max_hist_items 50 \
    --device cuda:0 \
    --phase2_blend_scale 0.20 \
    --fast_base_generate \
    --random_topk_sample 10 \
    --slate_value_head_path "$CRITIC_DIR/slate_value_head.pt" \
    --slate_value_meta_path "$CRITIC_DIR/slate_value_meta.json" \
    --slate_value_scale "$SLATE_VALUE_SCALE" \
    --slate_rerank_pool "$SLATE_RERANK_POOL" \
    --slate_greedy_candidates "$SLATE_GREEDY_CANDIDATES" \
    --log_every 50 \
    > "$log_path" 2>&1
  printf 'SlateQ-like\t2026\t%s\tslate_value_head.pt\t%s\n' "$eval_seed" "$(parse_eval_log "$log_path")" >> "$OUT_ROOT/per_run.tsv"
done

"$PYTHON_BIN" - "$OUT_ROOT/per_run.tsv" "$OUT_ROOT/summary.tsv" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = list(csv.DictReader(src.open("r", encoding="utf-8"), delimiter="\t"))
metric_keys = ["total_reward", "depth", "avg_reward", "coverage", "ild", "click_pct", "long_view_pct", "like_pct", "comment_pct", "forward_pct", "follow_pct", "hate_pct"]
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
fields = ["policy", "train_seed", "ckpt", "n_eval_seeds", "eval_seeds"] + [f"{k}_{s}" for k in metric_keys for s in ("mean", "std")]
with dst.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(summary_rows)
PY

echo "[$(date '+%F %T')] slateq_like_strict_eval3_done" | tee -a "$OUT_ROOT/suite.log"
cat "$OUT_ROOT/summary.tsv"
