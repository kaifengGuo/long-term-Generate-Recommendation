import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> str:
    text = str(path_str).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulator-in-the-loop iterative refresh for TIGER Phase8c sparse raw actor."
    )
    parser.add_argument("--tiger_ckpt", type=str, required=True, help="Initial TIGER checkpoint to refresh from.")
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
    parser.add_argument("--eval_episodes", type=int, default=200)
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
    parser.add_argument("--init_page_head_path", type=str, default="")
    parser.add_argument("--init_page_meta_path", type=str, default="")
    parser.add_argument("--init_item_head_path", type=str, default="")
    parser.add_argument("--init_item_meta_path", type=str, default="")
    parser.add_argument("--page_epochs", type=int, default=6)
    parser.add_argument("--item_epochs", type=int, default=8)
    parser.add_argument("--token_epochs", type=int, default=6)
    parser.add_argument("--page_batch_size", type=int, default=64)
    parser.add_argument("--item_batch_size", type=int, default=128)
    parser.add_argument("--token_batch_size", type=int, default=64)
    parser.add_argument("--page_lr", type=float, default=1e-3)
    parser.add_argument("--item_lr", type=float, default=1e-3)
    parser.add_argument("--token_lr", type=float, default=1e-3)
    parser.add_argument("--phase8c_epochs", type=int, default=1)
    parser.add_argument("--phase8c_batch_size", type=int, default=64)
    parser.add_argument("--phase8c_lr", type=float, default=2e-6)
    parser.add_argument("--phase8c_weight_decay", type=float, default=1e-4)
    parser.add_argument("--phase8c_train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument("--phase8c_item_weight_scale", type=float, default=0.05)
    parser.add_argument("--phase8c_item_neg_scale", type=float, default=0.02)
    parser.add_argument("--phase8c_neg_scale", type=float, default=0.05)
    parser.add_argument("--phase8c_kl_scale", type=float, default=1.0)
    parser.add_argument("--phase8c_credit_clip", type=float, default=2.0)
    parser.add_argument("--phase8c_page_gate_scale", type=float, default=0.10)
    parser.add_argument("--phase8c_page_gate_min", type=float, default=0.90)
    parser.add_argument("--phase8c_page_gate_max", type=float, default=1.10)
    parser.add_argument("--phase8c_positive_topk", type=int, default=1)
    parser.add_argument("--phase8c_negative_topk", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
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


def maybe_add_allocator(cmd: List[str], args: argparse.Namespace) -> None:
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_head_path", str(args.allocator_head_path)])
    if str(args.allocator_meta_path).strip():
        cmd.extend(["--allocator_meta_path", str(args.allocator_meta_path)])
    if str(args.allocator_head_path).strip() or str(args.allocator_meta_path).strip():
        cmd.extend(["--allocator_device", "cpu"])


def read_metrics_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    python = sys.executable
    output_root = Path(resolve_path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "iterative_summary.json"
    summary: List[Dict[str, object]] = []
    current_tiger_ckpt = resolve_path(args.tiger_ckpt)
    current_page_head = resolve_path(args.init_page_head_path)
    current_page_meta = resolve_path(args.init_page_meta_path)
    current_item_head = resolve_path(args.init_item_head_path)
    current_item_meta = resolve_path(args.init_item_meta_path)
    uirm_log_path = resolve_path(args.uirm_log_path)
    sid_mapping_path = resolve_path(args.sid_mapping_path)
    allocator_head_path = resolve_path(args.allocator_head_path)
    allocator_meta_path = resolve_path(args.allocator_meta_path)
    start_iter = 1

    if bool(args.resume) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary:
            last = summary[-1]
            start_iter = int(last.get("iter", 0)) + 1
            current_tiger_ckpt = resolve_path(str(last.get("phase8c_ckpt", current_tiger_ckpt)))
            current_page_head = resolve_path(str(last.get("page_head_path", current_page_head)))
            current_page_meta = resolve_path(str(last.get("page_meta_path", current_page_meta)))
            current_item_head = resolve_path(str(last.get("item_head_path", current_item_head)))
            current_item_meta = resolve_path(str(last.get("item_meta_path", current_item_meta)))
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
            "--tiger_ckpt", str(current_tiger_ckpt),
            "--sid_mapping_path", str(sid_mapping_path),
            "--uirm_log_path", str(uirm_log_path),
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
            "--fast_base_generate",
            "--trace_path", str(trace_path.resolve()),
        ]
        rollout_text = run_command(rollout_cmd, rollout_log)
        rollout_metrics = parse_eval_metrics(rollout_text)

        chain_path = iter_dir / "slate_chain.jsonl"
        chain_log = iter_dir / "build_slate_chain.log"
        chain_cmd = [
            python,
            str(CODE_DIR / "build_tiger_slate_credit_chain.py"),
            "--trace_path", str(trace_path.resolve()),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
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
        if allocator_head_path:
            chain_cmd.extend(["--allocator_head_path", allocator_head_path])
        if allocator_meta_path:
            chain_cmd.extend(["--allocator_meta_path", allocator_meta_path])
        if allocator_head_path or allocator_meta_path:
            chain_cmd.extend(["--allocator_device", "cpu"])
        run_command(chain_cmd, chain_log)

        page_dir = iter_dir / "page_prefix"
        page_log = iter_dir / "train_page_prefix.log"
        page_cmd = [
            python,
            str(CODE_DIR / "train_tiger_page_prefix_critic.py"),
            "--trace_path", str(trace_path.resolve()),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
            "--tiger_ckpt", str(current_tiger_ckpt),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--gamma", str(args.gamma),
            "--page_delta_field", "credit",
            "--credit_mode", str(args.credit_mode),
            "--credit_clip", str(args.credit_clip),
            "--valid_ratio", "0.10",
            "--batch_size", str(args.page_batch_size),
            "--epochs", str(args.page_epochs),
            "--lr", str(args.page_lr),
            "--weight_decay", "1e-4",
            "--save_dir", str(page_dir.resolve()),
        ]
        if current_page_head and current_page_meta:
            page_cmd.extend(["--init_head_path", current_page_head, "--init_meta_path", current_page_meta])
        run_command(page_cmd, page_log)
        current_page_head = str((page_dir / "page_prefix_head.pt").resolve())
        current_page_meta = str((page_dir / "page_prefix_meta.json").resolve())

        item_dir = iter_dir / "item_prefix"
        item_log = iter_dir / "train_item_prefix.log"
        item_cmd = [
            python,
            str(CODE_DIR / "train_tiger_item_prefix_critic.py"),
            "--trace_path", str(trace_path.resolve()),
            "--chain_path", str(chain_path.resolve()),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--item_credit_field", "item_credit",
            "--valid_ratio", "0.10",
            "--batch_size", str(args.item_batch_size),
            "--epochs", str(args.item_epochs),
            "--lr", str(args.item_lr),
            "--weight_decay", "1e-4",
            "--save_dir", str(item_dir.resolve()),
        ]
        if current_item_head and current_item_meta:
            item_cmd.extend(["--init_head_path", current_item_head, "--init_meta_path", current_item_meta])
        run_command(item_cmd, item_log)
        current_item_head = str((item_dir / "item_prefix_head.pt").resolve())
        current_item_meta = str((item_dir / "item_prefix_meta.json").resolve())

        token_dir = iter_dir / "token_prefix"
        token_log = iter_dir / "train_token_prefix.log"
        token_cmd = [
            python,
            str(CODE_DIR / "train_tiger_prefix_critic.py"),
            "--chain_path", str(chain_path.resolve()),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
            "--tiger_ckpt", str(current_tiger_ckpt),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--size", "mini" if str(args.model_size) == "mini" else "base",
            "--max_hist_items", str(args.max_hist_items),
            "--token_credit_field", "token_credit_calibrated",
            "--item_credit_field", "item_credit",
            "--valid_ratio", "0.10",
            "--batch_size", str(args.token_batch_size),
            "--epochs", str(args.token_epochs),
            "--lr", str(args.token_lr),
            "--weight_decay", "1e-4",
            "--save_dir", str(token_dir.resolve()),
        ]
        run_command(token_cmd, token_log)
        token_head = str((token_dir / "prefix_critic_head.pt").resolve())
        token_meta = str((token_dir / "prefix_critic_meta.json").resolve())

        hier_chain_path = iter_dir / "hier_prefix_chain.jsonl"
        hier_log = iter_dir / "build_hier_chain.log"
        hier_cmd = [
            python,
            str(CODE_DIR / "build_tiger_hier_prefix_advantage_chain.py"),
            "--trace_path", str(trace_path.resolve()),
            "--chain_path", str(chain_path.resolve()),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
            "--tiger_ckpt", str(current_tiger_ckpt),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--max_hist_items", str(args.max_hist_items),
            "--page_head_path", str(current_page_head),
            "--page_meta_path", str(current_page_meta),
            "--item_head_path", str(current_item_head),
            "--item_meta_path", str(current_item_meta),
            "--token_head_path", str(token_head),
            "--token_meta_path", str(token_meta),
            "--output_path", str(hier_chain_path.resolve()),
        ]
        run_command(hier_cmd, hier_log)

        phase8c_dir = iter_dir / "phase8c_actor"
        phase8c_log = iter_dir / "train_phase8c.log"
        phase8c_cmd = [
            python,
            str(CODE_DIR / "train_tiger_phase8c_sparse_raw_actor.py"),
            "--chain_path", str(hier_chain_path.resolve()),
            "--uirm_log_path", str(uirm_log_path),
            "--tiger_ckpt", str(current_tiger_ckpt),
            "--init_tiger_ckpt", str(current_tiger_ckpt),
            "--sid_mapping_path", str(sid_mapping_path),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--batch_size", str(args.phase8c_batch_size),
            "--epochs", str(args.phase8c_epochs),
            "--lr", str(args.phase8c_lr),
            "--weight_decay", str(args.phase8c_weight_decay),
            "--train_scope", str(args.phase8c_train_scope),
            "--item_weight_scale", str(args.phase8c_item_weight_scale),
            "--item_neg_scale", str(args.phase8c_item_neg_scale),
            "--neg_scale", str(args.phase8c_neg_scale),
            "--kl_scale", str(args.phase8c_kl_scale),
            "--credit_clip", str(args.phase8c_credit_clip),
            "--renorm_mode", "batch_abs",
            "--page_gate_scale", str(args.phase8c_page_gate_scale),
            "--page_gate_min", str(args.phase8c_page_gate_min),
            "--page_gate_max", str(args.phase8c_page_gate_max),
            "--positive_topk", str(args.phase8c_positive_topk),
            "--negative_topk", str(args.phase8c_negative_topk),
            "--save_dir", str(phase8c_dir.resolve()),
        ]
        run_command(phase8c_cmd, phase8c_log)
        current_tiger_ckpt = str((phase8c_dir / "phase8c_sparse_raw_actor_tiger.pth").resolve())

        eval_log = iter_dir / "eval.log"
        eval_cmd = [
            python,
            str(CODE_DIR / "eval_tiger_env.py"),
            "--tiger_ckpt", str(current_tiger_ckpt),
            "--sid_mapping_path", str(sid_mapping_path),
            "--uirm_log_path", str(uirm_log_path),
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
        ]
        eval_text = run_command(eval_cmd, eval_log)
        eval_metrics = parse_eval_metrics(eval_text)

        phase8c_metrics = read_metrics_json(phase8c_dir / "phase8c_sparse_raw_actor_metrics.json")
        page_metrics = read_metrics_json(page_dir / "page_prefix_metrics.json")
        item_metrics = read_metrics_json(item_dir / "item_prefix_metrics.json")
        token_metrics = read_metrics_json(token_dir / "prefix_critic_metrics.json")

        record = {
            "iter": int(iter_idx),
            "input_tiger_ckpt": resolve_path(args.tiger_ckpt) if iter_idx == int(start_iter) else summary[-1].get("phase8c_ckpt", ""),
            "rollout_trace_path": str(trace_path.resolve()),
            "slate_chain_path": str(chain_path.resolve()),
            "page_head_path": str(Path(current_page_head).resolve()),
            "page_meta_path": str(Path(current_page_meta).resolve()),
            "item_head_path": str(Path(current_item_head).resolve()),
            "item_meta_path": str(Path(current_item_meta).resolve()),
            "token_head_path": str(Path(token_head).resolve()),
            "token_meta_path": str(Path(token_meta).resolve()),
            "hier_chain_path": str(hier_chain_path.resolve()),
            "phase8c_ckpt": str(Path(current_tiger_ckpt).resolve()),
            "rollout_metrics": rollout_metrics,
            "eval_metrics": eval_metrics,
            "page_metrics": page_metrics,
            "item_metrics": item_metrics,
            "token_metrics": token_metrics,
            "phase8c_metrics": phase8c_metrics,
            "logs": {
                "rollout_log": str(rollout_log.resolve()),
                "slate_chain_log": str(chain_log.resolve()),
                "page_log": str(page_log.resolve()),
                "item_log": str(item_log.resolve()),
                "token_log": str(token_log.resolve()),
                "hier_log": str(hier_log.resolve()),
                "phase8c_log": str(phase8c_log.resolve()),
                "eval_log": str(eval_log.resolve()),
            },
        }
        summary.append(record)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
