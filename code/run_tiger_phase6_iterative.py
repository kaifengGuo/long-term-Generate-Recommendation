import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulator-in-the-loop iterative rollout refresh for TIGER Phase6 Joint.")
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--episode_batch_size", type=int, default=32)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--num_iters", type=int, default=1)
    parser.add_argument("--rollout_episodes", type=int, default=300)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--credit_mode", type=str, default="centered", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--transport_epsilon", type=float, default=0.35)
    parser.add_argument("--transport_iter", type=int, default=16)
    parser.add_argument("--cf_topk_history", type=int, default=5)
    parser.add_argument("--cf_smooth", type=float, default=0.05)
    parser.add_argument("--allocator_head_path", type=str, default="")
    parser.add_argument("--allocator_meta_path", type=str, default="")
    parser.add_argument("--allocator_blend_alpha", type=float, default=0.7)
    parser.add_argument("--allocator_keep_topk", type=int, default=2)
    parser.add_argument("--heuristic_mix", type=float, default=0.60)
    parser.add_argument("--support_mix", type=float, default=0.25)
    parser.add_argument("--response_mix", type=float, default=0.15)
    parser.add_argument("--base_prefix_head_path", type=str, default="")
    parser.add_argument("--base_prefix_meta_path", type=str, default="")
    parser.add_argument("--base_prefix_scale", type=float, default=0.05)
    parser.add_argument("--init_joint_head_path", type=str, default="")
    parser.add_argument("--init_joint_meta_path", type=str, default="")
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--rollout_phase6_prefix_scale", type=float, default=0.05)
    parser.add_argument("--rollout_phase6_token_scale", type=float, default=0.01)
    parser.add_argument("--eval_phase6_prefix_scale", type=float, default=0.05)
    parser.add_argument("--eval_phase6_token_scale", type=float, default=0.01)
    parser.add_argument("--train_epochs", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_max_pages", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from the last completed iter in output_root/iterative_summary.json.")
    return parser.parse_args()


def run_command(cmd: List[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(CODE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")
    return output


def parse_eval_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    patterns = {
        "total_reward": r"Total Reward:\s*([0-9.\-]+)",
        "depth": r"Depth:\s*([0-9.\-]+)",
        "avg_reward": r"Average reward:\s*([0-9.\-]+)",
        "coverage": r"Coverage:\s*([0-9.\-]+)",
        "click": r"is_click:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "long_view": r"long_view:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def compute_chain_sign_stats(chain_path: Path) -> Dict[str, float]:
    pos_page = neg_page = zero_page = 0
    pos_item = neg_item = zero_item = 0
    pos_tok = neg_tok = zero_tok = 0
    page_seen = set()
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            key = (int(x["episode_id"]), int(x["page_index"]))
            if key not in page_seen:
                page_seen.add(key)
                sc = float(x.get("slate_credit", 0.0))
                if sc > 1e-8:
                    pos_page += 1
                elif sc < -1e-8:
                    neg_page += 1
                else:
                    zero_page += 1
            ic = float(x.get("item_credit", 0.0))
            if ic > 1e-8:
                pos_item += 1
            elif ic < -1e-8:
                neg_item += 1
            else:
                zero_item += 1
            for tok in x.get("token_credit_calibrated", []):
                tv = float(tok)
                if tv > 1e-8:
                    pos_tok += 1
                elif tv < -1e-8:
                    neg_tok += 1
                else:
                    zero_tok += 1
    n_pages = max(len(page_seen), 1)
    n_items = max(pos_item + neg_item + zero_item, 1)
    n_tokens = max(pos_tok + neg_tok + zero_tok, 1)
    return {
        "n_pages": float(len(page_seen)),
        "page_pos_ratio": float(pos_page / n_pages),
        "page_neg_ratio": float(neg_page / n_pages),
        "page_zero_ratio": float(zero_page / n_pages),
        "item_pos_ratio": float(pos_item / n_items),
        "item_neg_ratio": float(neg_item / n_items),
        "item_zero_ratio": float(zero_item / n_items),
        "token_pos_ratio": float(pos_tok / n_tokens),
        "token_neg_ratio": float(neg_tok / n_tokens),
        "token_zero_ratio": float(zero_tok / n_tokens),
    }


def maybe_add_prefix(cmd: List[str], head_path: str, meta_path: str, scale: float) -> None:
    if not str(head_path).strip():
        return
    cmd.extend(["--phase4_prefix_head_path", str(head_path)])
    if str(meta_path).strip():
        cmd.extend(["--phase4_prefix_meta_path", str(meta_path)])
    cmd.extend(["--phase4_prefix_scale", str(scale)])


def maybe_add_joint(cmd: List[str], head_path: str, meta_path: str, prefix_scale: float, token_scale: float) -> None:
    if not str(head_path).strip():
        return
    cmd.extend(["--phase6_joint_head_path", str(head_path)])
    if str(meta_path).strip():
        cmd.extend(["--phase6_joint_meta_path", str(meta_path)])
    cmd.extend(["--phase6_prefix_scale", str(prefix_scale), "--phase6_token_actor_scale", str(token_scale)])


def main() -> int:
    args = parse_args()
    python = sys.executable
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    current_joint_head = str(args.init_joint_head_path).strip()
    current_joint_meta = str(args.init_joint_meta_path).strip()
    summary_path = output_root / "iterative_summary.json"
    summary: List[Dict[str, object]] = []
    start_iter = 1
    if bool(args.resume) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary:
            last_entry = summary[-1]
            start_iter = int(last_entry.get("iter", 0)) + 1
            current_joint_head = str(last_entry.get("joint_head_path", current_joint_head)).strip()
            current_joint_meta = str(last_entry.get("joint_meta_path", current_joint_meta)).strip()
            print(
                f"[resume] loaded {len(summary)} completed iters from {summary_path}; "
                f"continuing at iter {start_iter}"
            )
    for iter_idx in range(int(start_iter), int(args.num_iters) + 1):
        iter_dir = output_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        trace_path = iter_dir / "rollout_trace.jsonl"
        rollout_log = iter_dir / "rollout.log"
        rollout_cmd = [
            python,
            str(CODE_DIR / "eval_tiger_phase2_blend_env.py"),
            "--tiger_ckpt", str(args.tiger_ckpt),
            "--sid_mapping_path", str(args.sid_mapping_path),
            "--uirm_log_path", str(args.uirm_log_path),
            "--slate_size", str(args.slate_size),
            "--episode_batch_size", str(args.episode_batch_size),
            "--model_size", str(args.model_size),
            "--num_episodes", str(args.rollout_episodes),
            "--max_steps_per_episode", str(args.max_steps_per_episode),
            "--max_step_per_episode", str(args.max_steps_per_episode),
            "--beam_width", str(args.beam_width),
            "--single_response",
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--device", str(args.device),
            "--phase2_blend_scale", "0.0",
            "--trace_path", str(trace_path),
        ]
        maybe_add_prefix(rollout_cmd, args.base_prefix_head_path, args.base_prefix_meta_path, float(args.base_prefix_scale))
        maybe_add_joint(
            rollout_cmd,
            current_joint_head,
            current_joint_meta,
            float(args.rollout_phase6_prefix_scale),
            float(args.rollout_phase6_token_scale),
        )
        rollout_text = run_command(rollout_cmd, rollout_log)

        chain_path = iter_dir / "rollout_chain.jsonl"
        chain_log = iter_dir / "build_chain.log"
        chain_cmd = [
            python,
            str(CODE_DIR / "build_tiger_slate_credit_chain.py"),
            "--trace_path", str(trace_path),
            "--uirm_log_path", str(args.uirm_log_path),
            "--sid_mapping_path", str(args.sid_mapping_path),
            "--device", "cpu",
            "--gamma", str(args.gamma),
            "--max_hist_items", str(args.max_hist_items),
            "--credit_mode", str(args.credit_mode),
            "--credit_clip", str(args.credit_clip),
            "--transport_epsilon", str(args.transport_epsilon),
            "--transport_iter", str(args.transport_iter),
            "--cf_topk_history", str(args.cf_topk_history),
            "--cf_smooth", str(args.cf_smooth),
            "--allocator_blend_alpha", str(args.allocator_blend_alpha),
            "--allocator_keep_topk", str(args.allocator_keep_topk),
            "--heuristic_mix", str(args.heuristic_mix),
            "--support_mix", str(args.support_mix),
            "--response_mix", str(args.response_mix),
            "--output_path", str(chain_path),
        ]
        if str(args.allocator_head_path).strip():
            chain_cmd.extend(["--allocator_head_path", str(args.allocator_head_path)])
        if str(args.allocator_meta_path).strip():
            chain_cmd.extend(["--allocator_meta_path", str(args.allocator_meta_path)])
        if str(args.allocator_head_path).strip() or str(args.allocator_meta_path).strip():
            chain_cmd.extend(["--allocator_device", "cpu"])
        run_command(chain_cmd, chain_log)
        chain_stats = compute_chain_sign_stats(chain_path)

        train_dir = iter_dir / "phase6_joint"
        train_log = iter_dir / "train_joint.log"
        train_cmd = [
            python,
            str(CODE_DIR / "train_tiger_phase6_joint.py"),
            "--chain_path", str(chain_path),
            "--uirm_log_path", str(args.uirm_log_path),
            "--tiger_ckpt", str(args.tiger_ckpt),
            "--sid_mapping_path", str(args.sid_mapping_path),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--valid_ratio", "0.15",
            "--batch_size", str(args.train_batch_size),
            "--epochs", str(args.train_epochs),
            "--lr", str(args.train_lr),
            "--weight_decay", str(args.train_weight_decay),
            "--save_dir", str(train_dir),
        ]
        if int(args.train_max_pages) > 0:
            train_cmd.extend(["--max_pages", str(args.train_max_pages)])
        if bool(args.warm_start) and str(current_joint_head).strip():
            train_cmd.extend(["--init_joint_head_path", str(current_joint_head)])
        run_command(train_cmd, train_log)

        new_joint_head = str(train_dir / "phase6_joint_heads.pt")
        new_joint_meta = str(train_dir / "phase6_joint_meta.json")
        eval_log = iter_dir / "eval.log"
        eval_cmd = [
            python,
            str(CODE_DIR / "eval_tiger_phase2_blend_env.py"),
            "--tiger_ckpt", str(args.tiger_ckpt),
            "--sid_mapping_path", str(args.sid_mapping_path),
            "--uirm_log_path", str(args.uirm_log_path),
            "--slate_size", str(args.slate_size),
            "--episode_batch_size", str(args.episode_batch_size),
            "--model_size", str(args.model_size),
            "--num_episodes", str(args.eval_episodes),
            "--max_steps_per_episode", str(args.max_steps_per_episode),
            "--max_step_per_episode", str(args.max_steps_per_episode),
            "--beam_width", str(args.beam_width),
            "--single_response",
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--device", str(args.device),
            "--phase2_blend_scale", "0.0",
        ]
        maybe_add_prefix(eval_cmd, args.base_prefix_head_path, args.base_prefix_meta_path, float(args.base_prefix_scale))
        maybe_add_joint(
            eval_cmd,
            new_joint_head,
            new_joint_meta,
            float(args.eval_phase6_prefix_scale),
            float(args.eval_phase6_token_scale),
        )
        eval_text = run_command(eval_cmd, eval_log)
        eval_metrics = parse_eval_metrics(eval_text)
        rollout_metrics = parse_eval_metrics(rollout_text)

        current_joint_head = new_joint_head
        current_joint_meta = new_joint_meta
        summary.append(
            {
                "iter": int(iter_idx),
                "trace_path": str(trace_path.resolve()),
                "chain_path": str(chain_path.resolve()),
                "joint_head_path": str(Path(new_joint_head).resolve()),
                "joint_meta_path": str(Path(new_joint_meta).resolve()),
                "rollout_metrics": rollout_metrics,
                "chain_stats": chain_stats,
                "eval_metrics": eval_metrics,
                "rollout_log": str(rollout_log.resolve()),
                "chain_log": str(chain_log.resolve()),
                "train_log": str(train_log.resolve()),
                "eval_log": str(eval_log.resolve()),
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary[-1], ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
