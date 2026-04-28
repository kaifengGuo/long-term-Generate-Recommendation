import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
ROLLOUT_EVAL_SCRIPT = PROJECT_ROOT / "eval_tiger_phase2_blend_env.py"
STANDARD_EVAL_SCRIPT = PROJECT_ROOT / "eval_tiger_env.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative runner for the isolated TIGER HCLA-RL route.")
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument(
        "--actor_init_ckpt",
        type=str,
        default="",
        help="Optional actor init checkpoint for iter 1. Defaults to --tiger_ckpt.",
    )
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--trace_path", type=str, default="", help="Optional existing trace path for iter 1.")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--episode_batch_size", type=int, default=32)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--num_iters", type=int, default=1)
    parser.add_argument("--rollout_episodes", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.35)
    parser.add_argument("--credit_mode", type=str, default="centered", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--reward_preset", type=str, default="click_longview")
    parser.add_argument("--item_share_mode", type=str, default="bootstrap", choices=["bootstrap", "heuristic"])
    parser.add_argument("--share_heuristic_mix", type=float, default=0.60)
    parser.add_argument("--share_support_mix", type=float, default=0.25)
    parser.add_argument("--share_response_mix", type=float, default=0.15)
    parser.add_argument("--chain_max_pages", type=int, default=0)
    parser.add_argument("--allocator_epochs", type=int, default=8)
    parser.add_argument("--allocator_batch_size", type=int, default=64)
    parser.add_argument("--actor_epochs", type=int, default=1)
    parser.add_argument("--critic_epochs", type=int, default=2)
    parser.add_argument("--actor_batch_size", type=int, default=32)
    parser.add_argument("--critic_batch_size", type=int, default=32)
    parser.add_argument("--actor_adv_blend_alpha", type=float, default=0.50)
    parser.add_argument("--actor_item_share_blend_alpha", type=float, default=0.40)
    parser.add_argument("--actor_item_adv_scale", type=float, default=0.15)
    parser.add_argument("--actor_page_adv_scale", type=float, default=0.15)
    parser.add_argument("--actor_ppo_loss_scale", type=float, default=1.00)
    parser.add_argument("--actor_sibling_loss_scale", type=float, default=0.35)
    parser.add_argument("--actor_sibling_adv_beta", type=float, default=0.75)
    parser.add_argument("--actor_sibling_temperature", type=float, default=1.00)
    parser.add_argument("--actor_sibling_score_clip", type=float, default=6.0)
    parser.add_argument("--actor_kl_scale", type=float, default=0.05)
    parser.add_argument("--actor_sft_scale", type=float, default=0.05)
    return parser.parse_args()


def run_command(cmd: List[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    text = proc.stdout or ""
    log_path.write_text(text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")
    return text


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
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def metric_delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    out: Dict[str, float] = {}
    for key in keys:
        if key in before and key in after:
            out[key] = float(after[key] - before[key])
    return out


def resolve_batch_size(num_episodes: int, default_batch_size: int = 32) -> int:
    total = int(num_episodes)
    if total <= 0:
        return int(default_batch_size)
    return max(1, min(int(default_batch_size), total))


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    current_tiger_ckpt = str(Path(args.tiger_ckpt).resolve())
    initial_actor_ckpt = str(Path(args.actor_init_ckpt).resolve()) if str(args.actor_init_ckpt).strip() else current_tiger_ckpt
    summary: List[Dict[str, object]] = []

    for iter_idx in range(1, int(args.num_iters) + 1):
        iter_dir = output_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        rollout_batch_size = resolve_batch_size(int(args.rollout_episodes), int(args.episode_batch_size))
        eval_batch_size = resolve_batch_size(int(args.eval_episodes), int(args.episode_batch_size))

        baseline_metrics: Dict[str, float] = {}
        if int(args.eval_episodes) > 0:
            baseline_cmd = [
                python,
                str(STANDARD_EVAL_SCRIPT),
                "--tiger_ckpt", current_tiger_ckpt,
                "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
                "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
                "--slate_size", str(args.slate_size),
                "--episode_batch_size", str(eval_batch_size),
                "--model_size", str(args.model_size),
                "--num_episodes", str(args.eval_episodes),
                "--max_steps_per_episode", str(args.max_steps_per_episode),
                "--max_step_per_episode", str(args.max_steps_per_episode),
                "--beam_width", str(args.beam_width),
                "--single_response",
                "--seed", str(args.seed),
                "--max_hist_items", str(args.max_hist_items),
                "--device", str(args.device),
            ]
            baseline_text = run_command(baseline_cmd, iter_dir / "baseline_eval.log")
            baseline_metrics = parse_eval_metrics(baseline_text)

        if iter_idx == 1 and str(args.trace_path).strip():
            trace_path = Path(args.trace_path).resolve()
            rollout_metrics: Dict[str, float] = {}
        else:
            trace_path = iter_dir / "rollout_trace.jsonl"
            rollout_cmd = [
                python,
                str(ROLLOUT_EVAL_SCRIPT),
                "--tiger_ckpt", current_tiger_ckpt,
                "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
                "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
                "--slate_size", str(args.slate_size),
                "--episode_batch_size", str(rollout_batch_size),
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
                "--fast_base_generate",
                "--trace_path", str(trace_path),
            ]
            rollout_text = run_command(rollout_cmd, iter_dir / "rollout.log")
            rollout_metrics = parse_eval_metrics(rollout_text)

        chain_path = iter_dir / "hcla_chain.jsonl"
        chain_cmd = [
            python,
            str(THIS_DIR / "build_hcla_longterm_chain.py"),
            "--trace_path", str(trace_path),
            "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
            "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
            "--device", "cpu",
            "--seed", str(args.seed),
            "--gamma", str(args.gamma),
            "--max_hist_items", str(args.max_hist_items),
            "--credit_mode", str(args.credit_mode),
            "--credit_clip", str(args.credit_clip),
            "--reward_preset", str(args.reward_preset),
            "--hazard_lambda", str(args.hazard_lambda),
            "--item_share_mode", str(args.item_share_mode),
            "--share_heuristic_mix", str(args.share_heuristic_mix),
            "--share_support_mix", str(args.share_support_mix),
            "--share_response_mix", str(args.share_response_mix),
            "--output_path", str(chain_path),
        ]
        if int(args.chain_max_pages) > 0:
            chain_cmd.extend(["--max_pages", str(args.chain_max_pages)])
        run_command(chain_cmd, iter_dir / "build_chain.log")

        critic_dir = iter_dir / "critic"
        critic_cmd = [
            python,
            str(THIS_DIR / "train_hcla_longterm_critic.py"),
            "--chain_path", str(chain_path),
            "--tiger_ckpt", current_tiger_ckpt,
            "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--batch_size", str(args.critic_batch_size),
            "--epochs", str(args.critic_epochs),
            "--save_dir", str(critic_dir),
        ]
        run_command(critic_cmd, iter_dir / "train_critic.log")

        allocator_dir = iter_dir / "item_allocator"
        allocator_cmd = [
            python,
            str(PROJECT_ROOT / "train_tiger_online_slate_allocator.py"),
            "--chain_path", str(chain_path),
            "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
            "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--batch_size", str(args.allocator_batch_size),
            "--epochs", str(args.allocator_epochs),
            "--save_dir", str(allocator_dir),
        ]
        if int(args.chain_max_pages) > 0:
            allocator_cmd.extend(["--max_pages", str(args.chain_max_pages)])
        run_command(allocator_cmd, iter_dir / "train_allocator.log")

        actor_dir = iter_dir / "actor"
        actor_init_ckpt = initial_actor_ckpt if iter_idx == 1 else current_tiger_ckpt
        actor_cmd = [
            python,
            str(THIS_DIR / "train_hcla_actor.py"),
            "--chain_path", str(chain_path),
            "--tiger_ckpt", current_tiger_ckpt,
            "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
            "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
            "--init_tiger_ckpt", actor_init_ckpt,
            "--critic_bundle_path", str((critic_dir / "critic_bundle.pt").resolve()),
            "--critic_meta_path", str((critic_dir / "critic_bundle_meta.json").resolve()),
            "--item_allocator_head_path", str((allocator_dir / "online_slate_allocator_head.pt").resolve()),
            "--item_allocator_meta_path", str((allocator_dir / "online_slate_allocator_meta.json").resolve()),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--batch_size", str(args.actor_batch_size),
            "--epochs", str(args.actor_epochs),
            "--adv_blend_alpha", str(args.actor_adv_blend_alpha),
            "--item_share_blend_alpha", str(args.actor_item_share_blend_alpha),
            "--item_adv_scale", str(args.actor_item_adv_scale),
            "--page_adv_scale", str(args.actor_page_adv_scale),
            "--ppo_loss_scale", str(args.actor_ppo_loss_scale),
            "--sibling_loss_scale", str(args.actor_sibling_loss_scale),
            "--sibling_adv_beta", str(args.actor_sibling_adv_beta),
            "--sibling_temperature", str(args.actor_sibling_temperature),
            "--sibling_score_clip", str(args.actor_sibling_score_clip),
            "--kl_scale", str(args.actor_kl_scale),
            "--sft_scale", str(args.actor_sft_scale),
            "--save_dir", str(actor_dir),
        ]
        run_command(actor_cmd, iter_dir / "train_actor.log")
        current_tiger_ckpt = str((actor_dir / "hcla_actor_tiger.pth").resolve())

        eval_metrics: Dict[str, float] = {}
        if int(args.eval_episodes) > 0:
            eval_cmd = [
                python,
                str(STANDARD_EVAL_SCRIPT),
                "--tiger_ckpt", current_tiger_ckpt,
                "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
                "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
                "--slate_size", str(args.slate_size),
                "--episode_batch_size", str(eval_batch_size),
                "--model_size", str(args.model_size),
                "--num_episodes", str(args.eval_episodes),
                "--max_steps_per_episode", str(args.max_steps_per_episode),
                "--max_step_per_episode", str(args.max_steps_per_episode),
                "--beam_width", str(args.beam_width),
                "--single_response",
                "--seed", str(args.seed),
                "--max_hist_items", str(args.max_hist_items),
                "--device", str(args.device),
            ]
            eval_text = run_command(eval_cmd, iter_dir / "eval.log")
            eval_metrics = parse_eval_metrics(eval_text)

        eval_delta = metric_delta(eval_metrics, baseline_metrics)
        entry = {
            "iter": int(iter_idx),
            "input_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()) if iter_idx == 1 else summary[-1]["actor_ckpt"],
            "actor_init_tiger_ckpt": str(Path(actor_init_ckpt).resolve()),
            "trace_path": str(trace_path),
            "chain_path": str(chain_path.resolve()),
            "critic_bundle": str((critic_dir / "critic_bundle.pt").resolve()),
            "critic_meta": str((critic_dir / "critic_bundle_meta.json").resolve()),
            "item_allocator_head": str((allocator_dir / "online_slate_allocator_head.pt").resolve()),
            "item_allocator_meta": str((allocator_dir / "online_slate_allocator_meta.json").resolve()),
            "actor_ckpt": str(Path(current_tiger_ckpt).resolve()),
            "rollout_metrics": rollout_metrics,
            "baseline_eval_metrics": baseline_metrics,
            "eval_metrics": eval_metrics,
            "eval_delta": eval_delta,
        }
        summary.append(entry)
        (output_root / "iterative_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[hcla-iter] completed {len(summary)} iteration(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
