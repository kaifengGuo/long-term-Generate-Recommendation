#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/baselines/bert4rec_envmap_strict_eval3_${TIMESTAMP_TAG}}"
SEQ_ROOT="${SEQ_ROOT:-$ROOT/results/baselines/seq_ckpts_envmap}"
SEQ_TRAIN_SEED="${SEQ_TRAIN_SEED:-2026}"
EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-111 112 113}"
IFS=' ' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_STR"

mkdir -p "$OUT_ROOT" "$SEQ_ROOT"
cd "$ROOT"

UIRM_LOG="code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log"
CSV_PATH="code/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv"
BERT4REC_DIR="$SEQ_ROOT/bert4rec_firstseen_seed${SEQ_TRAIN_SEED}"

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

echo "[$(date '+%F %T')] bert4rec_envmap_strict_eval3_start" | tee -a "$OUT_ROOT/suite.log"
echo "out_root=$OUT_ROOT" | tee -a "$OUT_ROOT/suite.log"
echo "seq_root=$SEQ_ROOT" | tee -a "$OUT_ROOT/suite.log"
echo "seq_train_seed=$SEQ_TRAIN_SEED" | tee -a "$OUT_ROOT/suite.log"
echo "eval_seeds=${EVAL_SEEDS[*]}" | tee -a "$OUT_ROOT/suite.log"

if [ ! -f "$BERT4REC_DIR/best.pt" ]; then
  mkdir -p "$BERT4REC_DIR"
  echo "[$(date '+%F %T')] train BERT4Rec with first_seen/env item mapping" | tee -a "$OUT_ROOT/suite.log"
  "$PYTHON_BIN" code/train_bert4rec_baseline.py \
    --csv_path "$CSV_PATH" \
    --save_dir "$BERT4REC_DIR" \
    --cache_dir "$ROOT/code/cache_bert4rec_firstseen" \
    --mapping_order first_seen \
    --device cuda:0 \
    --max_len 50 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --dropout 0.2 \
    --batch_size 2048 \
    --epochs 80 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --early_stop_patience 10 \
    --grad_clip 1.0 \
    --mask_prob 0.20 \
    --force_last_mask \
    --seed "$SEQ_TRAIN_SEED" \
    --eval_seed "$SEQ_TRAIN_SEED" \
    --num_workers 4 \
    --save_metric_k 10 \
    > "$BERT4REC_DIR/train.log" 2>&1
else
  echo "[$(date '+%F %T')] reuse BERT4Rec checkpoint $BERT4REC_DIR/best.pt" | tee -a "$OUT_ROOT/suite.log"
fi

for eval_seed in "${EVAL_SEEDS[@]}"; do
  log_path="$OUT_ROOT/eval_BERT4Rec_seed${eval_seed}.log"
  echo "[$(date '+%F %T')] eval BERT4Rec seed=$eval_seed" | tee -a "$OUT_ROOT/suite.log"
  "$PYTHON_BIN" code/eval_bert4rec_env.py \
    --bert4rec_ckpt "$BERT4REC_DIR/best.pt" \
    --uirm_log_path "$UIRM_LOG" \
    --slate_size 6 \
    --episode_batch_size 32 \
    --single_response \
    --initial_temper 20 \
    --item_correlation 0.2 \
    --num_episodes 200 \
    --max_steps_per_episode 20 \
    --max_step_per_episode 20 \
    --device cuda:0 \
    --seed "$eval_seed" \
    --max_hist_len 50 \
    --hist_mode click \
    --log_every 50 \
    > "$log_path" 2>&1
  printf 'BERT4Rec\t%s\t%s\tbest.pt\t%s\n' "$SEQ_TRAIN_SEED" "$eval_seed" "$(parse_eval_log "$log_path")" >> "$OUT_ROOT/per_run.tsv"
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

echo "[$(date '+%F %T')] bert4rec_envmap_strict_eval3_done" | tee -a "$OUT_ROOT/suite.log"
cat "$OUT_ROOT/summary.tsv"
