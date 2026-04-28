import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parents[1]
METHOD_DIR = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> str:
    text = str(path_str).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop rollout -> page Q critic -> SID advantage -> actor refresh.")
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument(
        "--init_learner_tiger_ckpt",
        type=str,
        default="",
        help="Optional learner checkpoint to resume from. If empty, learner starts from --tiger_ckpt.",
    )
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
    parser.add_argument("--initial_temper", type=float, default=20.0)
    parser.add_argument("--item_correlation", type=float, default=0.0)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--num_iters", type=int, default=1)
    parser.add_argument("--rollout_episodes", type=int, default=64)
    parser.add_argument("--eval_episodes", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--replay_recent_iters", type=int, default=0, help="If >0, critic replay only keeps the most recent N rollout traces.")
    parser.add_argument("--keep_heavy_intermediates", action="store_true")
    parser.add_argument("--shared_page_credit_metric", action="store_true", help="Align rollout/eval reward protocol to the shared page_credit_loop setting: slate=1, steps=20, beam=16, blend=0.2, top-k sampling=10.")
    parser.add_argument("--rollout_phase2_blend_scale", type=float, default=0.0)
    parser.add_argument("--rollout_random_topk_sample", type=int, default=0)
    parser.add_argument("--rollout_random_item_prob", type=float, default=0.0)
    parser.add_argument("--eval_use_phase2_blend", action="store_true")
    parser.add_argument("--eval_phase2_blend_scale", type=float, default=0.0)
    parser.add_argument("--eval_random_topk_sample", type=int, default=0)
    parser.add_argument("--eval_random_item_prob", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.0)
    parser.add_argument("--critic_batch_size", type=int, default=32)
    parser.add_argument("--critic_epochs", type=int, default=3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--critic_weight_decay", type=float, default=1e-4)
    parser.add_argument("--critic_valid_ratio", type=float, default=0.15)
    parser.add_argument("--critic_item_dim", type=int, default=128)
    parser.add_argument("--critic_model_dim", type=int, default=128)
    parser.add_argument("--critic_dropout", type=float, default=0.10)
    parser.add_argument("--critic_arch", type=str, default="base", choices=["base", "v8", "v9add"])
    parser.add_argument("--critic_num_heads", type=int, default=4)
    parser.add_argument("--critic_num_layers", type=int, default=2)
    parser.add_argument("--critic_ensemble_size", type=int, default=1)
    parser.add_argument("--critic_pessimism_beta", type=float, default=0.0)
    parser.add_argument("--critic_eval_batch_size", type=int, default=256)
    parser.add_argument("--critic_target_heuristic_mix", type=float, default=0.60)
    parser.add_argument("--critic_target_support_mix", type=float, default=0.25)
    parser.add_argument("--critic_target_response_mix", type=float, default=0.15)
    parser.add_argument("--critic_page_loss_scale", type=float, default=1.0)
    parser.add_argument("--critic_item_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_prefix_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_page_huber_beta", type=float, default=1.0)
    parser.add_argument("--critic_item_huber_beta", type=float, default=1.0)
    parser.add_argument("--critic_prefix_huber_beta", type=float, default=1.0)
    parser.add_argument("--critic_rank_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_monotonic_loss_scale", type=float, default=0.0)
    parser.add_argument("--critic_rank_min_gap", type=float, default=0.05)
    parser.add_argument("--actor_batch_size", type=int, default=64)
    parser.add_argument("--actor_epochs", type=int, default=1)
    parser.add_argument("--actor_lr", type=float, default=2e-6)
    parser.add_argument("--actor_weight_decay", type=float, default=1e-4)
    parser.add_argument("--actor_update_every", type=int, default=1, help="Update actor every N iterations. 1 means update each iteration.")
    parser.add_argument(
        "--rollout_policy_sync_mode",
        type=str,
        default="hard",
        choices=["hard", "ema"],
        help="How to sync the rollout/sampling policy toward the learner policy after each actor update.",
    )
    parser.add_argument(
        "--rollout_policy_sync_tau",
        type=float,
        default=1.0,
        help="Soft update factor for rollout policy sync. 1.0 means hard copy to learner.",
    )
    parser.add_argument("--actor_method", type=str, default="hac", choices=["hac", "wsft", "grpo", "pref", "hybrid", "posterior"])
    parser.add_argument("--actor_train_scope", type=str, default="last_decoder_block", choices=["decoder_only", "last_decoder_block", "full"])
    parser.add_argument(
        "--actor_credit_mode",
        type=str,
        default="manual",
        choices=["manual", "pessimistic_strong"],
        help="Optional preset that wires actor rewards/advantages to the pessimistic critic signals.",
    )
    parser.add_argument("--actor_token_adv_field", type=str, default="sid_advantage")
    parser.add_argument("--actor_item_adv_field", type=str, default="item_advantage")
    parser.add_argument("--actor_page_adv_field", type=str, default="page_q_value")
    parser.add_argument(
        "--actor_page_reward_field",
        type=str,
        default="reward_raw",
        choices=[
            "reward_raw",
            "reward_model_value",
            "reward_margin_vs_behavior",
            "page_q_value",
            "page_q_mean",
            "page_q_pess",
            "item_advantage",
            "item_advantage_mean",
            "item_advantage_pess",
        ],
    )
    parser.add_argument("--actor_item_adv_scale", type=float, default=0.0)
    parser.add_argument("--actor_page_adv_scale", type=float, default=0.0)
    parser.add_argument("--actor_page_gate_scale", type=float, default=0.0)
    parser.add_argument("--actor_page_gate_min", type=float, default=0.85)
    parser.add_argument("--actor_page_gate_max", type=float, default=1.15)
    parser.add_argument(
        "--actor_page_gate_mode",
        type=str,
        default="abs_tanh",
        choices=["abs_tanh", "signed_tanh", "positive_tanh", "none"],
    )
    parser.add_argument("--actor_positive_topk", type=int, default=4)
    parser.add_argument("--actor_positive_floor", type=float, default=0.0)
    parser.add_argument("--actor_negative_topk", type=int, default=4)
    parser.add_argument("--actor_negative_floor", type=float, default=0.0)
    parser.add_argument("--actor_credit_clip", type=float, default=3.0)
    parser.add_argument("--actor_renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--actor_clip_eps", type=float, default=0.20)
    parser.add_argument("--actor_kl_scale", type=float, default=0.05)
    parser.add_argument("--actor_adaptive_kl_support_scale", type=float, default=0.0)
    parser.add_argument("--actor_adaptive_kl_unc_scale", type=float, default=0.0)
    parser.add_argument("--actor_adaptive_clip_support_scale", type=float, default=0.0)
    parser.add_argument("--actor_adaptive_clip_unc_scale", type=float, default=0.0)
    parser.add_argument("--actor_min_clip_eps", type=float, default=0.02)
    parser.add_argument("--actor_trust_support_field", type=str, default="support_gap_scaled")
    parser.add_argument("--actor_trust_unc_field", type=str, default="uncertainty_ratio")
    parser.add_argument("--actor_entropy_scale", type=float, default=0.0)
    parser.add_argument("--actor_sft_scale", type=float, default=0.0)
    parser.add_argument("--actor_mle_scale", type=float, default=0.10)
    parser.add_argument("--actor_neg_scale", type=float, default=0.25)
    parser.add_argument("--actor_group_size", type=int, default=4)
    parser.add_argument("--actor_group_beam_width", type=int, default=16)
    parser.add_argument("--actor_group_max_rows", type=int, default=0)
    parser.add_argument("--actor_group_num_shards", type=int, default=1)
    parser.add_argument(
        "--actor_group_reward_field",
        type=str,
        default="page_q_value",
        choices=[
            "page_q_value",
            "page_q_mean",
            "page_q_pess",
            "item_advantage",
            "item_advantage_mean",
            "item_advantage_pess",
            "adaptive_support_pess",
        ],
    )
    parser.add_argument(
        "--actor_group_reward_transform",
        type=str,
        default="raw",
        choices=["raw", "centered", "clipped_margin", "tanh_margin"],
    )
    parser.add_argument("--actor_group_margin_clip", type=float, default=0.0)
    parser.add_argument("--actor_group_margin_temperature", type=float, default=1.0)
    parser.add_argument("--actor_group_support_penalty_scale", type=float, default=0.0)
    parser.add_argument("--actor_group_support_gap_temperature", type=float, default=1.0)
    parser.add_argument("--actor_group_support_gap_clip", type=float, default=0.0)
    parser.add_argument("--actor_group_adaptive_beta_unc_scale", type=float, default=0.0)
    parser.add_argument("--actor_group_adaptive_beta_support_scale", type=float, default=0.0)
    parser.add_argument("--actor_pref_score_field", type=str, default="")
    parser.add_argument("--actor_pref_safe_support_gap_max", type=float, default=0.25)
    parser.add_argument("--actor_pref_min_gap", type=float, default=0.0)
    parser.add_argument("--actor_pref_max_pairs_per_group", type=int, default=2)
    parser.add_argument(
        "--actor_pref_pair_mode",
        type=str,
        default="mixed",
        choices=["safe_vs_behavior", "safe_vs_unsafe", "mixed", "search_distill"],
    )
    parser.add_argument(
        "--actor_pref_exploit_score_field",
        type=str,
        default="page_q_mean",
        choices=[
            "reward_raw",
            "reward_model_value",
            "adaptive_support_pess",
            "page_q_value",
            "page_q_mean",
            "page_q_pess",
            "item_advantage",
            "item_advantage_mean",
            "item_advantage_pess",
        ],
    )
    parser.add_argument("--actor_pref_min_support_gap_delta", type=float, default=0.05)
    parser.add_argument("--actor_pref_min_unc_delta", type=float, default=0.0)
    parser.add_argument("--actor_pref_pair_score_gap_scale", type=float, default=1.0)
    parser.add_argument("--actor_pref_pair_raw_q_gap_scale", type=float, default=0.5)
    parser.add_argument("--actor_pref_pair_unc_gap_scale", type=float, default=0.25)
    parser.add_argument("--actor_pref_pair_support_gap_scale", type=float, default=0.5)
    parser.add_argument("--actor_pref_beta", type=float, default=1.0)
    parser.add_argument("--actor_pref_label_smoothing", type=float, default=0.0)
    parser.add_argument("--actor_pref_sft_scale", type=float, default=0.05)
    parser.add_argument("--actor_pref_gap_scale", type=float, default=1.0)
    parser.add_argument("--actor_pref_gap_clip", type=float, default=2.0)
    parser.add_argument("--actor_pref_score_normalization", type=str, default="mean_token", choices=["sum", "mean_token"])
    parser.add_argument("--actor_pref_attr_adv_mode", type=str, default="pess", choices=["raw", "pess"])
    parser.add_argument("--actor_pref_attr_pair_scale", type=float, default=0.0)
    parser.add_argument("--actor_pref_attr_item_scale", type=float, default=0.10)
    parser.add_argument("--actor_pref_attr_credit_clip", type=float, default=3.0)
    parser.add_argument("--actor_pref_attr_renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--actor_pref_attr_topk", type=int, default=2)
    parser.add_argument("--actor_pref_attr_floor", type=float, default=0.0)
    parser.add_argument("--actor_hybrid_pref_anchor_scale", type=float, default=0.50)
    parser.add_argument(
        "--actor_posterior_score_field",
        type=str,
        default="reward_model_value",
        choices=[
            "reward_model_value",
            "reward_raw",
            "reward_margin_vs_behavior",
            "page_q_value",
            "page_q_mean",
            "page_q_pess",
            "item_advantage",
            "item_advantage_mean",
            "item_advantage_pess",
            "adaptive_support_pess",
        ],
    )
    parser.add_argument("--actor_posterior_prior_field", type=str, default="support_logprob_mean", choices=["none", "support_logprob_sum", "support_logprob_mean"])
    parser.add_argument("--actor_posterior_score_scale", type=float, default=1.0)
    parser.add_argument("--actor_posterior_prior_scale", type=float, default=0.20)
    parser.add_argument("--actor_posterior_temperature", type=float, default=1.0)
    parser.add_argument("--actor_posterior_teacher_logit_clip", type=float, default=20.0)
    parser.add_argument("--actor_posterior_teacher_safe_support_gap_max", type=float, default=-1.0)
    parser.add_argument("--actor_posterior_teacher_safe_uncertainty_ratio_max", type=float, default=0.0)
    parser.add_argument("--actor_posterior_teacher_topk", type=int, default=0)
    parser.add_argument("--actor_posterior_teacher_behavior_mix", type=float, default=0.0)
    parser.add_argument(
        "--actor_posterior_reference_kl_mode",
        type=str,
        default="safe_uniform",
        choices=["teacher", "safe_uniform", "all_uniform"],
    )
    parser.add_argument("--actor_posterior_reference_kl_scale", type=float, default=0.0)
    parser.add_argument("--actor_posterior_score_normalization", type=str, default="mean_token", choices=["sum", "mean_token"])
    parser.add_argument("--actor_posterior_attr_adv_mode", type=str, default="pess", choices=["raw", "pess"])
    parser.add_argument("--actor_posterior_attr_item_scale", type=float, default=0.10)
    parser.add_argument("--actor_posterior_attr_temperature", type=float, default=1.0)
    parser.add_argument("--actor_posterior_attr_mix", type=float, default=0.50)
    parser.add_argument("--actor_posterior_attr_credit_clip", type=float, default=3.0)
    parser.add_argument("--actor_posterior_attr_renorm_mode", type=str, default="batch_abs", choices=["none", "batch_abs"])
    parser.add_argument("--disable_auto_plot", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=160)
    args = parser.parse_args()
    if bool(args.shared_page_credit_metric):
        args.slate_size = 1
        args.max_steps_per_episode = 20
        args.beam_width = 16
        args.rollout_phase2_blend_scale = 0.20
        args.rollout_random_topk_sample = 10
        args.rollout_random_item_prob = 0.0
        args.eval_use_phase2_blend = True
        args.eval_phase2_blend_scale = 0.20
        args.eval_random_topk_sample = 10
        args.eval_random_item_prob = 0.0
    if str(args.actor_credit_mode).lower() == "pessimistic_strong":
        args.actor_token_adv_field = "sid_advantage_pess"
        args.actor_item_adv_field = "item_advantage_pess"
        args.actor_page_adv_field = "page_q_pess"
        args.actor_page_reward_field = "page_q_pess"
        args.actor_group_reward_field = "page_q_pess"
        args.actor_page_gate_mode = "signed_tanh"
        if float(args.actor_item_adv_scale) <= 0.0:
            args.actor_item_adv_scale = 0.10
        if float(args.actor_page_adv_scale) <= 0.0:
            args.actor_page_adv_scale = 0.10
        if float(args.actor_page_gate_scale) <= 0.0:
            args.actor_page_gate_scale = 0.10
    if not str(args.actor_pref_score_field).strip():
        args.actor_pref_score_field = str(args.actor_group_reward_field)
    return args


def sync_rollout_policy_checkpoint(
    *,
    rollout_ckpt: str,
    learner_ckpt: str,
    mode: str,
    tau: float,
    output_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(mode).strip().lower()
    tau = float(min(max(float(tau), 0.0), 1.0))

    if mode == "hard" or tau >= 1.0 - 1e-8:
        shutil.copy2(str(learner_ckpt), str(output_path))
        return str(output_path), {
            "mode": "hard",
            "tau": 1.0,
            "rollout_ckpt_in": str(Path(rollout_ckpt).resolve()),
            "learner_ckpt_in": str(Path(learner_ckpt).resolve()),
            "rollout_ckpt_out": str(output_path),
        }

    rollout_state = torch.load(str(rollout_ckpt), map_location="cpu")
    learner_state = torch.load(str(learner_ckpt), map_location="cpu")
    merged_state: Dict[str, Any] = {}
    ema_tensor_keys = 0

    for key, learner_value in learner_state.items():
        rollout_value = rollout_state.get(key)
        if (
            isinstance(learner_value, torch.Tensor)
            and isinstance(rollout_value, torch.Tensor)
            and learner_value.shape == rollout_value.shape
            and torch.is_floating_point(learner_value)
            and torch.is_floating_point(rollout_value)
        ):
            merged_state[key] = (
                rollout_value.detach().cpu().to(dtype=learner_value.dtype) * (1.0 - tau)
                + learner_value.detach().cpu() * tau
            )
            ema_tensor_keys += 1
        else:
            merged_state[key] = learner_value

    for key, rollout_value in rollout_state.items():
        if key not in merged_state:
            merged_state[key] = rollout_value

    torch.save(merged_state, str(output_path))
    return str(output_path), {
        "mode": "ema",
        "tau": tau,
        "ema_tensor_keys": int(ema_tensor_keys),
        "rollout_ckpt_in": str(Path(rollout_ckpt).resolve()),
        "learner_ckpt_in": str(Path(learner_ckpt).resolve()),
        "rollout_ckpt_out": str(output_path),
    }


def run_command(cmd: List[str], log_path: Path) -> Tuple[str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        # Run child scripts from the bundle root so packaged relative paths like
        # `code/dataset/...` in the rewritten UIRM log resolve correctly.
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration_sec = float(time.perf_counter() - start)
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")
    return output, duration_sec


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


def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    out: Dict[str, float] = {}
    for key in keys:
        if key in before and key in after:
            out[key] = float(after[key] - before[key])
    return out


def append_jsonl(dst: Path, src: Path) -> int:
    n_lines = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as src_fp, dst.open("a", encoding="utf-8") as dst_fp:
        for line in src_fp:
            line = line.strip()
            if not line:
                continue
            dst_fp.write(line + "\n")
            n_lines += 1
    return int(n_lines)


def rebuild_jsonl(dst: Path, src_paths: List[Path]) -> int:
    n_lines = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as dst_fp:
        for src in src_paths:
            if not src.exists():
                continue
            with src.open("r", encoding="utf-8") as src_fp:
                for line in src_fp:
                    line = line.strip()
                    if not line:
                        continue
                    dst_fp.write(line + "\n")
                    n_lines += 1
    return int(n_lines)


def write_jsonl_manifest(dst: Path, src_paths: List[Path]) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "jsonl_manifest",
        "paths": [str(path.resolve()) for path in src_paths],
        "num_paths": int(len(src_paths)),
    }
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def cleanup_paths(paths: List[Path]) -> List[str]:
    removed: List[str] = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path.resolve()))
        except Exception:
            continue
    return removed


def split_jsonl_round_robin(src: Path, dst_paths: List[Path], max_lines: int = 0) -> Tuple[int, List[int]]:
    total = 0
    counts = [0 for _ in dst_paths]
    for dst in dst_paths:
        dst.parent.mkdir(parents=True, exist_ok=True)
    handles = [dst.open("w", encoding="utf-8") for dst in dst_paths]
    try:
        with src.open("r", encoding="utf-8") as src_fp:
            for line in src_fp:
                line = line.strip()
                if not line:
                    continue
                if int(max_lines) > 0 and total >= int(max_lines):
                    break
                shard_idx = total % len(handles)
                handles[shard_idx].write(line + "\n")
                counts[shard_idx] += 1
                total += 1
    finally:
        for handle in handles:
            handle.close()
    return int(total), [int(x) for x in counts]


def run_commands_parallel(cmd_specs: List[Tuple[List[str], Path]]) -> Tuple[List[str], float]:
    if not cmd_specs:
        return [], 0.0
    start = time.perf_counter()
    procs: List[Tuple[subprocess.Popen, Path, List[str]]] = []
    for cmd, log_path in cmd_specs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        procs.append((proc, log_path, cmd))

    outputs: List[str] = []
    failures: List[Tuple[int, List[str], Path]] = []
    for proc, log_path, cmd in procs:
        output, _stderr = proc.communicate()
        text = output or ""
        log_path.write_text(text, encoding="utf-8")
        outputs.append(text)
        if int(proc.returncode) != 0:
            failures.append((int(proc.returncode), cmd, log_path))

    duration_sec = float(time.perf_counter() - start)
    if failures:
        messages = [
            f"({code}) {' '.join(cmd)}\nSee log: {log_path}"
            for code, cmd, log_path in failures
        ]
        raise RuntimeError("Parallel command failure(s):\n" + "\n\n".join(messages))
    return outputs, duration_sec


def merge_actor_group_summaries(
    *,
    shard_summaries: List[Dict[str, Any]],
    shard_summary_paths: List[Path],
    chain_path: Path,
    output_path: Path,
    summary_out: Path,
    num_shards: int,
    shard_row_counts: List[int],
) -> Dict[str, Any]:
    if not shard_summaries:
        merged = {
            "method": "TIGER-HCA-GRPO Group Builder",
            "chain_path": str(chain_path.resolve()),
            "output_path": str(output_path.resolve()),
            "n_input_rows": 0,
            "n_groups": 0,
            "n_rows": 0,
            "num_shards": int(num_shards),
            "shard_row_counts": [int(x) for x in shard_row_counts],
        }
        summary_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        return merged

    base = dict(shard_summaries[0])
    n_input_rows_total = int(sum(int(s.get("n_input_rows", 0)) for s in shard_summaries))
    n_groups_total = int(sum(int(s.get("n_groups", 0)) for s in shard_summaries))
    n_rows_total = int(sum(int(s.get("n_rows", 0)) for s in shard_summaries))
    behavior_rows_total = float(
        sum(float(s.get("behavior_ratio", 0.0)) * int(s.get("n_rows", 0)) for s in shard_summaries)
    )
    beam_rows_total = max(float(n_rows_total) - behavior_rows_total, 0.0)

    def weighted_average(key: str, weight: str) -> float:
        total_weight = 0.0
        total_value = 0.0
        for shard in shard_summaries:
            if weight == "n_rows":
                shard_weight = float(shard.get("n_rows", 0))
            elif weight == "n_groups":
                shard_weight = float(shard.get("n_groups", 0))
            elif weight == "behavior_rows":
                shard_weight = float(shard.get("behavior_ratio", 0.0)) * float(shard.get("n_rows", 0))
            elif weight == "beam_rows":
                shard_weight = max(
                    float(shard.get("n_rows", 0)) - (float(shard.get("behavior_ratio", 0.0)) * float(shard.get("n_rows", 0))),
                    0.0,
                )
            else:
                shard_weight = 0.0
            total_weight += shard_weight
            total_value += float(shard.get(key, 0.0)) * shard_weight
        if total_weight <= 0.0:
            return 0.0
        return float(total_value / total_weight)

    merged = dict(base)
    merged.update(
        {
            "chain_path": str(chain_path.resolve()),
            "output_path": str(output_path.resolve()),
            "summary_out": str(summary_out.resolve()),
            "n_input_rows": int(n_input_rows_total),
            "n_groups": int(n_groups_total),
            "n_rows": int(n_rows_total),
            "avg_group_size": float(n_rows_total / max(n_groups_total, 1)),
            "behavior_ratio": float(behavior_rows_total / max(float(n_rows_total), 1.0)),
            "reward_raw_mean": weighted_average("reward_raw_mean", "n_rows"),
            "reward_raw_std": weighted_average("reward_raw_std", "n_rows"),
            "group_adv_abs_mean": weighted_average("group_adv_abs_mean", "n_rows"),
            "item_adv_abs_mean": weighted_average("item_adv_abs_mean", "n_rows"),
            "sid_adv_abs_mean": weighted_average("sid_adv_abs_mean", "n_rows"),
            "behavior_source_reward_mean": weighted_average("behavior_source_reward_mean", "behavior_rows"),
            "beam_source_reward_mean": weighted_average("beam_source_reward_mean", "beam_rows"),
            "behavior_reward_mean": weighted_average("behavior_reward_mean", "behavior_rows"),
            "beam_reward_mean": weighted_average("beam_reward_mean", "beam_rows"),
            "avg_best_source_minus_behavior": weighted_average("avg_best_source_minus_behavior", "n_groups"),
            "median_best_source_minus_behavior": weighted_average("median_best_source_minus_behavior", "n_groups"),
            "avg_best_reward_minus_behavior": weighted_average("avg_best_reward_minus_behavior", "n_groups"),
            "median_best_reward_minus_behavior": weighted_average("median_best_reward_minus_behavior", "n_groups"),
            "pos_source_margin_frac": weighted_average("pos_source_margin_frac", "n_groups"),
            "pos_reward_margin_frac": weighted_average("pos_reward_margin_frac", "n_groups"),
            "top_behavior_source_frac": weighted_average("top_behavior_source_frac", "n_groups"),
            "top_behavior_reward_frac": weighted_average("top_behavior_reward_frac", "n_groups"),
            "num_shards": int(num_shards),
            "shard_row_counts": [int(x) for x in shard_row_counts],
            "shard_summary_paths": [str(path.resolve()) for path in shard_summary_paths],
        }
    )
    merged["beam_source_minus_behavior"] = float(merged["beam_source_reward_mean"] - merged["behavior_source_reward_mean"])
    merged["beam_reward_minus_behavior"] = float(merged["beam_reward_mean"] - merged["behavior_reward_mean"])
    merged["top_beam_source_frac"] = float(1.0 - merged["top_behavior_source_frac"]) if n_groups_total > 0 else 0.0
    merged["top_beam_reward_frac"] = float(1.0 - merged["top_behavior_reward_frac"]) if n_groups_total > 0 else 0.0
    summary_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def should_eval(iter_idx: int, eval_every: int, eval_episodes: int) -> bool:
    if int(eval_episodes) <= 0:
        return False
    if int(eval_every) <= 0:
        return False
    return int(iter_idx) == 1 or (int(iter_idx) % int(eval_every) == 0)


def should_update_actor(iter_idx: int, actor_update_every: int) -> bool:
    interval = max(int(actor_update_every), 1)
    return int(iter_idx) == 1 or (int(iter_idx) % interval == 0)


def build_policy_eval_cmd(
    *,
    python: str,
    tiger_ckpt: str,
    sid_mapping_path: str,
    uirm_log_path: str,
    slate_size: int,
    episode_batch_size: int,
    model_size: str,
    num_episodes: int,
    max_steps_per_episode: int,
    beam_width: int,
    initial_temper: float,
    item_correlation: float,
    seed: int,
    max_hist_items: int,
    device: str,
    use_phase2_blend: bool,
    phase2_blend_scale: float,
    random_topk_sample: int,
    random_item_prob: float,
    trace_path: str = "",
) -> List[str]:
    script_name = "eval_tiger_phase2_blend_env.py" if bool(use_phase2_blend) else "eval_tiger_env.py"
    cmd = [
        python,
        str(CODE_DIR / script_name),
        "--tiger_ckpt", str(tiger_ckpt),
        "--sid_mapping_path", str(sid_mapping_path),
        "--uirm_log_path", str(uirm_log_path),
        "--slate_size", str(int(slate_size)),
        "--episode_batch_size", str(int(episode_batch_size)),
        "--model_size", str(model_size),
        "--num_episodes", str(int(num_episodes)),
        "--max_steps_per_episode", str(int(max_steps_per_episode)),
        "--max_step_per_episode", str(int(max_steps_per_episode)),
        "--beam_width", str(int(beam_width)),
        "--single_response",
        "--initial_temper", str(float(initial_temper)),
        "--item_correlation", str(float(item_correlation)),
        "--seed", str(int(seed)),
        "--max_hist_items", str(int(max_hist_items)),
        "--device", str(device),
    ]
    if bool(use_phase2_blend):
        cmd.extend(
            [
                "--phase2_blend_scale", str(float(phase2_blend_scale)),
                "--fast_base_generate",
            ]
        )
        if int(random_topk_sample) > 0:
            cmd.extend(["--random_topk_sample", str(int(random_topk_sample))])
        if float(random_item_prob) != 0.0:
            cmd.extend(["--random_item_prob", str(float(random_item_prob))])
        if str(trace_path).strip():
            cmd.extend(["--trace_path", str(Path(trace_path).resolve())])
    return cmd


def refresh_plots(
    *,
    python: str,
    output_root: Path,
    summary_json_path: Path,
    summary_jsonl_path: Path,
    dpi: int,
) -> None:
    plot_log = output_root / "plot_refresh.log"
    cmd = [
        python,
        str(METHOD_DIR / "plot_closed_loop_metrics.py"),
        "--summary_path", str(summary_json_path.resolve()),
        "--summary_jsonl_path", str(summary_jsonl_path.resolve()),
        "--output_dir", str((output_root / "plots").resolve()),
        "--dpi", str(dpi),
    ]
    try:
        _output, _duration = run_command(cmd, plot_log)
    except Exception as exc:  # pragma: no cover - plotting should not kill the run
        plot_log.write_text(f"plot refresh failed: {exc}\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    python = sys.executable
    output_root = Path(resolve_path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "closed_loop_summary.json"
    summary_jsonl_path = output_root / "closed_loop_summary.jsonl"
    replay_trace_path = output_root / "replay_trace.jsonl"
    summary: List[Dict[str, object]] = []
    recent_trace_paths: List[Path] = []

    rollout_tiger_ckpt = resolve_path(args.tiger_ckpt)
    learner_tiger_ckpt = resolve_path(args.init_learner_tiger_ckpt) or rollout_tiger_ckpt
    current_critic_bundle = ""
    current_critic_meta = ""
    uirm_log_path = resolve_path(args.uirm_log_path)
    sid_mapping_path = resolve_path(args.sid_mapping_path)

    for iter_idx in range(1, int(args.num_iters) + 1):
        iter_start = time.perf_counter()
        iter_dir = output_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        do_eval = should_eval(int(iter_idx), int(args.eval_every), int(args.eval_episodes))

        before_eval_metrics: Dict[str, float] = {}
        before_eval_time_sec = 0.0
        if do_eval:
            before_eval_log = iter_dir / "before_eval.log"
            before_eval_cmd = build_policy_eval_cmd(
                python=python,
                tiger_ckpt=str(rollout_tiger_ckpt),
                sid_mapping_path=str(sid_mapping_path),
                uirm_log_path=str(uirm_log_path),
                slate_size=int(args.slate_size),
                episode_batch_size=int(args.episode_batch_size),
                model_size=str(args.model_size),
                num_episodes=int(args.eval_episodes),
                max_steps_per_episode=int(args.max_steps_per_episode),
                beam_width=int(args.beam_width),
                initial_temper=float(args.initial_temper),
                item_correlation=float(args.item_correlation),
                seed=int(args.seed),
                max_hist_items=int(args.max_hist_items),
                device=str(args.device),
                use_phase2_blend=bool(args.eval_use_phase2_blend),
                phase2_blend_scale=float(args.eval_phase2_blend_scale),
                random_topk_sample=int(args.eval_random_topk_sample),
                random_item_prob=float(args.eval_random_item_prob),
            )
            before_eval_text, before_eval_time_sec = run_command(before_eval_cmd, before_eval_log)
            before_eval_metrics = parse_eval_metrics(before_eval_text)

        trace_path = iter_dir / "rollout_trace.jsonl"
        rollout_log = iter_dir / "rollout.log"
        rollout_cmd = build_policy_eval_cmd(
            python=python,
            tiger_ckpt=str(rollout_tiger_ckpt),
            sid_mapping_path=str(sid_mapping_path),
            uirm_log_path=str(uirm_log_path),
            slate_size=int(args.slate_size),
            episode_batch_size=int(args.episode_batch_size),
            model_size=str(args.model_size),
            num_episodes=int(args.rollout_episodes),
            max_steps_per_episode=int(args.max_steps_per_episode),
            beam_width=int(args.beam_width),
            initial_temper=float(args.initial_temper),
            item_correlation=float(args.item_correlation),
            seed=int(args.seed),
            max_hist_items=int(args.max_hist_items),
            device=str(args.device),
            use_phase2_blend=True,
            phase2_blend_scale=float(args.rollout_phase2_blend_scale),
            random_topk_sample=int(args.rollout_random_topk_sample),
            random_item_prob=float(args.rollout_random_item_prob),
            trace_path=str(trace_path.resolve()),
        )
        rollout_text, rollout_time_sec = run_command(rollout_cmd, rollout_log)
        rollout_metrics = parse_eval_metrics(rollout_text)
        recent_trace_paths.append(trace_path)
        if int(args.replay_recent_iters) > 0:
            recent_trace_paths = recent_trace_paths[-int(args.replay_recent_iters):]
            replay_lines_total = rebuild_jsonl(replay_trace_path, recent_trace_paths)
            replay_lines_added = count_jsonl_lines(trace_path)
        else:
            replay_lines_added = append_jsonl(replay_trace_path, trace_path)
            replay_lines_total = count_jsonl_lines(replay_trace_path)

        critic_dir = iter_dir / "page_qcritic"
        critic_metrics_path = critic_dir / "page_sid_qcritic_metrics.json"
        critic_log = iter_dir / "train_page_critic.log"
        critic_cmd = [
            python,
            str(METHOD_DIR / "train_page_critic.py"),
            "--trace_path", str(replay_trace_path.resolve()),
            "--tiger_ckpt", str(rollout_tiger_ckpt),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--gamma", str(args.gamma),
            "--hazard_lambda", str(args.hazard_lambda),
            "--batch_size", str(args.critic_batch_size),
            "--epochs", str(args.critic_epochs),
            "--lr", str(args.critic_lr),
            "--weight_decay", str(args.critic_weight_decay),
            "--valid_ratio", str(args.critic_valid_ratio),
            "--item_dim", str(args.critic_item_dim),
            "--model_dim", str(args.critic_model_dim),
            "--dropout", str(args.critic_dropout),
            "--critic_arch", str(args.critic_arch),
            "--critic_num_heads", str(args.critic_num_heads),
            "--critic_num_layers", str(args.critic_num_layers),
            "--ensemble_size", str(args.critic_ensemble_size),
            "--critic_target_heuristic_mix", str(args.critic_target_heuristic_mix),
            "--critic_target_support_mix", str(args.critic_target_support_mix),
            "--critic_target_response_mix", str(args.critic_target_response_mix),
            "--critic_page_loss_scale", str(args.critic_page_loss_scale),
            "--critic_item_loss_scale", str(args.critic_item_loss_scale),
            "--critic_prefix_loss_scale", str(args.critic_prefix_loss_scale),
            "--critic_page_huber_beta", str(args.critic_page_huber_beta),
            "--critic_item_huber_beta", str(args.critic_item_huber_beta),
            "--critic_prefix_huber_beta", str(args.critic_prefix_huber_beta),
            "--critic_rank_loss_scale", str(args.critic_rank_loss_scale),
            "--critic_monotonic_loss_scale", str(args.critic_monotonic_loss_scale),
            "--critic_rank_min_gap", str(args.critic_rank_min_gap),
            "--save_dir", str(critic_dir.resolve()),
            "--metrics_out", str(critic_metrics_path.resolve()),
        ]
        if current_critic_bundle and current_critic_meta:
            critic_cmd.extend(["--init_bundle_path", str(current_critic_bundle)])
            critic_cmd.extend(["--init_meta_path", str(current_critic_meta)])
        _critic_text, critic_time_sec = run_command(critic_cmd, critic_log)
        critic_metrics = read_json(critic_metrics_path)
        current_critic_bundle = str((critic_dir / "page_sid_qcritic_bundle.pt").resolve())
        current_critic_meta = str((critic_dir / "page_sid_qcritic_meta.json").resolve())

        chain_path = iter_dir / "sid_advantage_chain.jsonl"
        chain_summary_path = iter_dir / "sid_advantage_chain_summary.json"
        chain_log = iter_dir / "build_sid_chain.log"
        chain_cmd = [
            python,
            str(METHOD_DIR / "build_sid_advantage_chain.py"),
            "--trace_path", str(trace_path.resolve()),
            "--critic_bundle_path", str(current_critic_bundle),
            "--critic_meta_path", str(current_critic_meta),
            "--tiger_ckpt", str(rollout_tiger_ckpt),
            "--uirm_log_path", str(uirm_log_path),
            "--sid_mapping_path", str(sid_mapping_path),
            "--model_size", str(args.model_size),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--max_hist_items", str(args.max_hist_items),
            "--gamma", str(args.gamma),
            "--hazard_lambda", str(args.hazard_lambda),
            "--critic_eval_batch_size", str(args.critic_eval_batch_size),
            "--critic_pessimism_beta", str(args.critic_pessimism_beta),
            "--output_path", str(chain_path.resolve()),
            "--summary_out", str(chain_summary_path.resolve()),
        ]
        _chain_text, chain_time_sec = run_command(chain_cmd, chain_log)
        chain_summary = read_json(chain_summary_path)

        actor_group_summary: Dict[str, Any] = {}
        actor_pair_summary: Dict[str, Any] = {}
        actor_pair_path: Path | None = None
        shard_chain_paths: List[Path] = []
        shard_output_paths: List[Path] = []
        actor_group_time_sec = 0.0
        actor_updated = should_update_actor(int(iter_idx), int(args.actor_update_every))
        actor_method = str(args.actor_method).lower()
        policy_sync_summary: Dict[str, Any] = {
            "enabled": True,
            "mode": str(args.rollout_policy_sync_mode),
            "tau": float(args.rollout_policy_sync_tau),
        }
        actor_dir_name = {
            "hac": "sid_actor",
            "wsft": "wsft_actor",
            "grpo": "grpo_actor",
            "pref": "pref_actor",
            "hybrid": "hybrid_actor",
            "posterior": "posterior_actor",
        }[actor_method]
        actor_dir = iter_dir / actor_dir_name
        actor_log = iter_dir / f"train_{actor_method}_actor.log"
        if not bool(actor_updated):
            actor_metrics = {
                "skipped": True,
                "reason": "actor_update_every",
                "actor_update_every": int(args.actor_update_every),
                "rollout_tiger_ckpt": str(rollout_tiger_ckpt),
                "learner_tiger_ckpt": str(learner_tiger_ckpt),
            }
            next_rollout_tiger_ckpt = str(rollout_tiger_ckpt)
            next_learner_tiger_ckpt = str(learner_tiger_ckpt)
            policy_sync_summary.update(
                {
                    "skipped": True,
                    "reason": "actor_update_every",
                    "rollout_ckpt_out": str(next_rollout_tiger_ckpt),
                    "learner_ckpt_out": str(next_learner_tiger_ckpt),
                }
            )
            actor_time_sec = 0.0
            actor_log.write_text(
                json.dumps(actor_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif actor_method == "hac":
            actor_metrics_path = actor_dir / "tiger_hac_actor_metrics.json"
            actor_cmd = [
                python,
                str(CODE_DIR / "train_tiger_hac_actor.py"),
                "--chain_path", str(chain_path.resolve()),
                "--uirm_log_path", str(uirm_log_path),
                "--tiger_ckpt", str(rollout_tiger_ckpt),
                "--init_tiger_ckpt", str(learner_tiger_ckpt),
                "--sid_mapping_path", str(sid_mapping_path),
                "--model_size", str(args.model_size),
                "--device", str(args.device),
                "--seed", str(args.seed),
                "--max_hist_items", str(args.max_hist_items),
                "--token_adv_field", str(args.actor_token_adv_field),
                "--item_adv_field", str(args.actor_item_adv_field),
                "--page_adv_field", str(args.actor_page_adv_field),
                "--batch_size", str(args.actor_batch_size),
                "--epochs", str(args.actor_epochs),
                "--lr", str(args.actor_lr),
                "--weight_decay", str(args.actor_weight_decay),
                "--train_scope", str(args.actor_train_scope),
                "--item_adv_scale", str(args.actor_item_adv_scale),
                "--page_adv_scale", str(args.actor_page_adv_scale),
                "--page_gate_scale", str(args.actor_page_gate_scale),
                "--page_gate_min", str(args.actor_page_gate_min),
                "--page_gate_max", str(args.actor_page_gate_max),
                "--positive_topk", str(args.actor_positive_topk),
                "--positive_floor", str(args.actor_positive_floor),
                "--negative_topk", str(args.actor_negative_topk),
                "--negative_floor", str(args.actor_negative_floor),
                "--credit_clip", str(args.actor_credit_clip),
                "--renorm_mode", str(args.actor_renorm_mode),
                "--clip_eps", str(args.actor_clip_eps),
                "--kl_scale", str(args.actor_kl_scale),
                "--entropy_scale", str(args.actor_entropy_scale),
                "--sft_scale", str(args.actor_sft_scale),
                "--save_dir", str(actor_dir.resolve()),
                "--metrics_out", str(actor_metrics_path.resolve()),
            ]
            _actor_text, actor_time_sec = run_command(actor_cmd, actor_log)
            actor_metrics = read_json(actor_metrics_path)
            next_learner_tiger_ckpt = str((actor_dir / "tiger_hac_actor_tiger.pth").resolve())
        elif actor_method == "wsft":
            actor_metrics_path = actor_dir / "tiger_hca_wsft_actor_metrics.json"
            actor_cmd = [
                python,
                str(CODE_DIR / "train_tiger_hca_wsft_actor.py"),
                "--chain_path", str(chain_path.resolve()),
                "--uirm_log_path", str(uirm_log_path),
                "--tiger_ckpt", str(rollout_tiger_ckpt),
                "--init_tiger_ckpt", str(learner_tiger_ckpt),
                "--sid_mapping_path", str(sid_mapping_path),
                "--model_size", str(args.model_size),
                "--device", str(args.device),
                "--seed", str(args.seed),
                "--max_hist_items", str(args.max_hist_items),
                "--token_adv_field", str(args.actor_token_adv_field),
                "--item_adv_field", str(args.actor_item_adv_field),
                "--page_adv_field", str(args.actor_page_adv_field),
                "--batch_size", str(args.actor_batch_size),
                "--epochs", str(args.actor_epochs),
                "--lr", str(args.actor_lr),
                "--weight_decay", str(args.actor_weight_decay),
                "--train_scope", str(args.actor_train_scope),
                "--item_adv_scale", str(args.actor_item_adv_scale),
                "--page_adv_scale", str(args.actor_page_adv_scale),
                "--page_gate_scale", str(args.actor_page_gate_scale),
                "--page_gate_min", str(args.actor_page_gate_min),
                "--page_gate_max", str(args.actor_page_gate_max),
                "--positive_topk", str(args.actor_positive_topk),
                "--positive_floor", str(args.actor_positive_floor),
                "--negative_topk", str(args.actor_negative_topk),
                "--negative_floor", str(args.actor_negative_floor),
                "--credit_clip", str(args.actor_credit_clip),
                "--renorm_mode", str(args.actor_renorm_mode),
                "--kl_scale", str(args.actor_kl_scale),
                "--entropy_scale", str(args.actor_entropy_scale),
                "--mle_scale", str(args.actor_mle_scale),
                "--neg_scale", str(args.actor_neg_scale),
                "--save_dir", str(actor_dir.resolve()),
                "--metrics_out", str(actor_metrics_path.resolve()),
            ]
            _actor_text, actor_time_sec = run_command(actor_cmd, actor_log)
            actor_metrics = read_json(actor_metrics_path)
            next_learner_tiger_ckpt = str((actor_dir / "tiger_hca_wsft_actor_tiger.pth").resolve())
        else:
            actor_group_path = iter_dir / "hca_grpo_groups.jsonl"
            actor_group_manifest_path = iter_dir / "hca_grpo_groups_manifest.json"
            actor_group_summary_path = iter_dir / "hca_grpo_group_summary.json"
            actor_group_log = iter_dir / "build_grpo_groups.log"
            num_group_shards = max(int(args.actor_group_num_shards), 1)
            if num_group_shards == 1:
                actor_group_cmd = [
                    python,
                    str(CODE_DIR / "build_tiger_hca_grpo_groups.py"),
                    "--chain_path", str(chain_path.resolve()),
                    "--critic_bundle_path", str(current_critic_bundle),
                    "--critic_meta_path", str(current_critic_meta),
                    "--tiger_ckpt", str(rollout_tiger_ckpt),
                    "--uirm_log_path", str(uirm_log_path),
                    "--sid_mapping_path", str(sid_mapping_path),
                    "--model_size", str(args.model_size),
                    "--device", str(args.device),
                    "--seed", str(args.seed),
                    "--max_hist_items", str(args.max_hist_items),
                    "--group_size", str(args.actor_group_size),
                    "--beam_width", str(args.actor_group_beam_width),
                    "--max_rows", str(args.actor_group_max_rows),
                    "--critic_eval_batch_size", str(args.critic_eval_batch_size),
                    "--critic_pessimism_beta", str(args.critic_pessimism_beta),
                    "--reward_field", str(args.actor_group_reward_field),
                    "--reward_transform", str(args.actor_group_reward_transform),
                    "--reward_margin_clip", str(args.actor_group_margin_clip),
                    "--reward_margin_temperature", str(args.actor_group_margin_temperature),
                    "--support_penalty_scale", str(args.actor_group_support_penalty_scale),
                    "--support_gap_temperature", str(args.actor_group_support_gap_temperature),
                    "--support_gap_clip", str(args.actor_group_support_gap_clip),
                    "--adaptive_beta_unc_scale", str(args.actor_group_adaptive_beta_unc_scale),
                    "--adaptive_beta_support_scale", str(args.actor_group_adaptive_beta_support_scale),
                    "--output_path", str(actor_group_path.resolve()),
                    "--summary_out", str(actor_group_summary_path.resolve()),
                ]
                _group_text, actor_group_time_sec = run_command(actor_group_cmd, actor_group_log)
                actor_group_summary = read_json(actor_group_summary_path)
            else:
                shard_chain_paths = [iter_dir / f"hca_grpo_groups_chain_shard_{shard_idx:02d}.jsonl" for shard_idx in range(num_group_shards)]
                shard_output_paths = [iter_dir / f"hca_grpo_groups_shard_{shard_idx:02d}.jsonl" for shard_idx in range(num_group_shards)]
                shard_summary_paths = [iter_dir / f"hca_grpo_group_summary_shard_{shard_idx:02d}.json" for shard_idx in range(num_group_shards)]
                shard_log_paths = [iter_dir / f"build_grpo_groups_shard_{shard_idx:02d}.log" for shard_idx in range(num_group_shards)]
                split_total_rows, shard_row_counts = split_jsonl_round_robin(
                    chain_path,
                    shard_chain_paths,
                    max_lines=int(args.actor_group_max_rows),
                )
                active_shard_indices = [idx for idx, row_count in enumerate(shard_row_counts) if int(row_count) > 0]
                parallel_cmds: List[Tuple[List[str], Path]] = []
                for shard_idx in active_shard_indices:
                    shard_cmd = [
                        python,
                        str(CODE_DIR / "build_tiger_hca_grpo_groups.py"),
                        "--chain_path", str(shard_chain_paths[shard_idx].resolve()),
                        "--critic_bundle_path", str(current_critic_bundle),
                        "--critic_meta_path", str(current_critic_meta),
                        "--tiger_ckpt", str(rollout_tiger_ckpt),
                        "--uirm_log_path", str(uirm_log_path),
                        "--sid_mapping_path", str(sid_mapping_path),
                        "--model_size", str(args.model_size),
                        "--device", str(args.device),
                        "--seed", str(args.seed),
                        "--max_hist_items", str(args.max_hist_items),
                        "--group_size", str(args.actor_group_size),
                        "--beam_width", str(args.actor_group_beam_width),
                        "--max_rows", "0",
                        "--critic_eval_batch_size", str(args.critic_eval_batch_size),
                        "--critic_pessimism_beta", str(args.critic_pessimism_beta),
                        "--reward_field", str(args.actor_group_reward_field),
                        "--reward_transform", str(args.actor_group_reward_transform),
                        "--reward_margin_clip", str(args.actor_group_margin_clip),
                        "--reward_margin_temperature", str(args.actor_group_margin_temperature),
                        "--support_penalty_scale", str(args.actor_group_support_penalty_scale),
                        "--support_gap_temperature", str(args.actor_group_support_gap_temperature),
                        "--support_gap_clip", str(args.actor_group_support_gap_clip),
                        "--adaptive_beta_unc_scale", str(args.actor_group_adaptive_beta_unc_scale),
                        "--adaptive_beta_support_scale", str(args.actor_group_adaptive_beta_support_scale),
                        "--output_path", str(shard_output_paths[shard_idx].resolve()),
                        "--summary_out", str(shard_summary_paths[shard_idx].resolve()),
                    ]
                    parallel_cmds.append((shard_cmd, shard_log_paths[shard_idx]))

                shard_outputs, actor_group_time_sec = run_commands_parallel(parallel_cmds)
                actor_group_log.write_text("\n\n".join(shard_outputs), encoding="utf-8")
                actor_group_path = actor_group_manifest_path
                write_jsonl_manifest(actor_group_path, [shard_output_paths[idx] for idx in active_shard_indices])
                actor_group_summary = merge_actor_group_summaries(
                    shard_summaries=[read_json(shard_summary_paths[idx]) for idx in active_shard_indices],
                    shard_summary_paths=[shard_summary_paths[idx] for idx in active_shard_indices],
                    chain_path=chain_path,
                    output_path=actor_group_path,
                    summary_out=actor_group_summary_path,
                    num_shards=len(active_shard_indices),
                    shard_row_counts=shard_row_counts,
                )
                actor_group_summary["split_input_rows"] = int(split_total_rows)

            if actor_method == "grpo":
                actor_metrics_path = actor_dir / "tiger_hca_grpo_actor_metrics.json"
                actor_cmd = [
                    python,
                    str(CODE_DIR / "train_tiger_hca_grpo_actor.py"),
                    "--group_path", str(actor_group_path.resolve()),
                    "--tiger_ckpt", str(rollout_tiger_ckpt),
                    "--init_tiger_ckpt", str(learner_tiger_ckpt),
                    "--sid_mapping_path", str(sid_mapping_path),
                    "--model_size", str(args.model_size),
                    "--device", str(args.device),
                    "--seed", str(args.seed),
                    "--group_adv_field", "group_advantage",
                    "--token_adv_field", str(args.actor_token_adv_field),
                    "--item_adv_field", str(args.actor_item_adv_field),
                    "--page_reward_field", str(args.actor_page_reward_field),
                    "--batch_size", str(args.actor_batch_size),
                    "--epochs", str(args.actor_epochs),
                    "--lr", str(args.actor_lr),
                    "--weight_decay", str(args.actor_weight_decay),
                    "--train_scope", str(args.actor_train_scope),
                    "--item_adv_scale", str(args.actor_item_adv_scale),
                    "--page_gate_scale", str(args.actor_page_gate_scale),
                    "--page_gate_min", str(args.actor_page_gate_min),
                    "--page_gate_max", str(args.actor_page_gate_max),
                    "--page_gate_mode", str(args.actor_page_gate_mode),
                    "--positive_topk", str(args.actor_positive_topk),
                    "--positive_floor", str(args.actor_positive_floor),
                    "--negative_topk", str(args.actor_negative_topk),
                    "--negative_floor", str(args.actor_negative_floor),
                    "--credit_clip", str(args.actor_credit_clip),
                    "--renorm_mode", str(args.actor_renorm_mode),
                    "--clip_eps", str(args.actor_clip_eps),
                    "--kl_scale", str(args.actor_kl_scale),
                    "--adaptive_kl_support_scale", str(args.actor_adaptive_kl_support_scale),
                    "--adaptive_kl_unc_scale", str(args.actor_adaptive_kl_unc_scale),
                    "--adaptive_clip_support_scale", str(args.actor_adaptive_clip_support_scale),
                    "--adaptive_clip_unc_scale", str(args.actor_adaptive_clip_unc_scale),
                    "--min_clip_eps", str(args.actor_min_clip_eps),
                    "--trust_support_field", str(args.actor_trust_support_field),
                    "--trust_unc_field", str(args.actor_trust_unc_field),
                    "--entropy_scale", str(args.actor_entropy_scale),
                    "--sft_scale", str(args.actor_sft_scale),
                    "--save_dir", str(actor_dir.resolve()),
                    "--metrics_out", str(actor_metrics_path.resolve()),
                ]
                _actor_text, actor_time_sec = run_command(actor_cmd, actor_log)
                actor_metrics = read_json(actor_metrics_path)
                next_learner_tiger_ckpt = str((actor_dir / "tiger_hca_grpo_actor_tiger.pth").resolve())
            elif actor_method == "posterior":
                actor_metrics_path = actor_dir / "tiger_hca_posterior_actor_metrics.json"
                actor_cmd = [
                    python,
                    str(CODE_DIR / "train_tiger_hca_posterior_actor.py"),
                    "--group_path", str(actor_group_path.resolve()),
                    "--tiger_ckpt", str(rollout_tiger_ckpt),
                    "--init_tiger_ckpt", str(learner_tiger_ckpt),
                    "--sid_mapping_path", str(sid_mapping_path),
                    "--model_size", str(args.model_size),
                    "--device", str(args.device),
                    "--seed", str(args.seed),
                    "--batch_size", str(args.actor_batch_size),
                    "--epochs", str(args.actor_epochs),
                    "--lr", str(args.actor_lr),
                    "--weight_decay", str(args.actor_weight_decay),
                    "--train_scope", str(args.actor_train_scope),
                    "--score_field", str(args.actor_posterior_score_field),
                    "--prior_field", str(args.actor_posterior_prior_field),
                    "--score_scale", str(args.actor_posterior_score_scale),
                    "--prior_scale", str(args.actor_posterior_prior_scale),
                    "--posterior_temperature", str(args.actor_posterior_temperature),
                    "--teacher_logit_clip", str(args.actor_posterior_teacher_logit_clip),
                    "--teacher_safe_support_gap_max", str(args.actor_posterior_teacher_safe_support_gap_max),
                    "--teacher_safe_uncertainty_ratio_max", str(args.actor_posterior_teacher_safe_uncertainty_ratio_max),
                    "--teacher_topk", str(args.actor_posterior_teacher_topk),
                    "--teacher_behavior_mix", str(args.actor_posterior_teacher_behavior_mix),
                    "--reference_kl_mode", str(args.actor_posterior_reference_kl_mode),
                    "--reference_kl_scale", str(args.actor_posterior_reference_kl_scale),
                    "--score_normalization", str(args.actor_posterior_score_normalization),
                    "--attr_adv_mode", str(args.actor_posterior_attr_adv_mode),
                    "--attr_item_scale", str(args.actor_posterior_attr_item_scale),
                    "--attr_temperature", str(args.actor_posterior_attr_temperature),
                    "--attr_mix", str(args.actor_posterior_attr_mix),
                    "--attr_credit_clip", str(args.actor_posterior_attr_credit_clip),
                    "--attr_renorm_mode", str(args.actor_posterior_attr_renorm_mode),
                    "--save_dir", str(actor_dir.resolve()),
                    "--metrics_out", str(actor_metrics_path.resolve()),
                ]
                _actor_text, actor_time_sec = run_command(actor_cmd, actor_log)
                actor_metrics = read_json(actor_metrics_path)
                next_learner_tiger_ckpt = str((actor_dir / "tiger_hca_posterior_actor_tiger.pth").resolve())
            else:
                actor_pair_path = iter_dir / "hca_pref_pairs.jsonl"
                actor_pair_summary_path = iter_dir / "hca_pref_pair_summary.json"
                actor_pair_log = iter_dir / "build_pref_pairs.log"
                actor_pair_cmd = [
                    python,
                    str(CODE_DIR / "build_tiger_hca_preference_pairs.py"),
                    "--group_path", str(actor_group_path.resolve()),
                    "--output_path", str(actor_pair_path.resolve()),
                    "--summary_out", str(actor_pair_summary_path.resolve()),
                    "--score_field", str(args.actor_pref_score_field),
                    "--safe_support_gap_max", str(args.actor_pref_safe_support_gap_max),
                    "--min_score_gap", str(args.actor_pref_min_gap),
                    "--max_pairs_per_group", str(args.actor_pref_max_pairs_per_group),
                    "--pair_mode", str(args.actor_pref_pair_mode),
                    "--exploit_score_field", str(args.actor_pref_exploit_score_field),
                    "--min_support_gap_delta", str(args.actor_pref_min_support_gap_delta),
                    "--min_unc_delta", str(args.actor_pref_min_unc_delta),
                    "--pair_score_gap_scale", str(args.actor_pref_pair_score_gap_scale),
                    "--pair_raw_q_gap_scale", str(args.actor_pref_pair_raw_q_gap_scale),
                    "--pair_unc_gap_scale", str(args.actor_pref_pair_unc_gap_scale),
                    "--pair_support_gap_scale", str(args.actor_pref_pair_support_gap_scale),
                ]
                _pair_text, actor_pair_time_sec = run_command(actor_pair_cmd, actor_pair_log)
                actor_pair_summary = read_json(actor_pair_summary_path)

                if actor_method == "hybrid":
                    actor_metrics_path = actor_dir / "tiger_hca_hybrid_actor_metrics.json"
                    actor_cmd = [
                        python,
                        str(CODE_DIR / "train_tiger_hca_hybrid_actor.py"),
                        "--group_path", str(actor_group_path.resolve()),
                        "--pair_path", str(actor_pair_path.resolve()),
                        "--tiger_ckpt", str(rollout_tiger_ckpt),
                        "--init_tiger_ckpt", str(learner_tiger_ckpt),
                        "--sid_mapping_path", str(sid_mapping_path),
                        "--model_size", str(args.model_size),
                        "--device", str(args.device),
                        "--seed", str(args.seed),
                        "--group_adv_field", "group_advantage",
                        "--token_adv_field", str(args.actor_token_adv_field),
                        "--item_adv_field", str(args.actor_item_adv_field),
                        "--page_reward_field", str(args.actor_page_reward_field),
                        "--batch_size", str(args.actor_batch_size),
                        "--epochs", str(args.actor_epochs),
                        "--lr", str(args.actor_lr),
                        "--weight_decay", str(args.actor_weight_decay),
                        "--train_scope", str(args.actor_train_scope),
                        "--item_adv_scale", str(args.actor_item_adv_scale),
                        "--page_gate_scale", str(args.actor_page_gate_scale),
                        "--page_gate_min", str(args.actor_page_gate_min),
                        "--page_gate_max", str(args.actor_page_gate_max),
                        "--page_gate_mode", str(args.actor_page_gate_mode),
                        "--positive_topk", str(args.actor_positive_topk),
                        "--positive_floor", str(args.actor_positive_floor),
                        "--negative_topk", str(args.actor_negative_topk),
                        "--negative_floor", str(args.actor_negative_floor),
                        "--credit_clip", str(args.actor_credit_clip),
                        "--renorm_mode", str(args.actor_renorm_mode),
                        "--clip_eps", str(args.actor_clip_eps),
                        "--kl_scale", str(args.actor_kl_scale),
                        "--adaptive_kl_support_scale", str(args.actor_adaptive_kl_support_scale),
                        "--adaptive_kl_unc_scale", str(args.actor_adaptive_kl_unc_scale),
                        "--adaptive_clip_support_scale", str(args.actor_adaptive_clip_support_scale),
                        "--adaptive_clip_unc_scale", str(args.actor_adaptive_clip_unc_scale),
                        "--min_clip_eps", str(args.actor_min_clip_eps),
                        "--trust_support_field", str(args.actor_trust_support_field),
                        "--trust_unc_field", str(args.actor_trust_unc_field),
                        "--entropy_scale", str(args.actor_entropy_scale),
                        "--grpo_sft_scale", str(args.actor_sft_scale),
                        "--pref_anchor_scale", str(args.actor_hybrid_pref_anchor_scale),
                        "--pref_beta", str(args.actor_pref_beta),
                        "--label_smoothing", str(args.actor_pref_label_smoothing),
                        "--pref_sft_scale", str(args.actor_pref_sft_scale),
                        "--gap_scale", str(args.actor_pref_gap_scale),
                        "--gap_clip", str(args.actor_pref_gap_clip),
                        "--score_normalization", str(args.actor_pref_score_normalization),
                        "--attr_adv_mode", str(args.actor_pref_attr_adv_mode),
                        "--attr_pair_scale", str(args.actor_pref_attr_pair_scale),
                        "--attr_item_scale", str(args.actor_pref_attr_item_scale),
                        "--attr_credit_clip", str(args.actor_pref_attr_credit_clip),
                        "--attr_renorm_mode", str(args.actor_pref_attr_renorm_mode),
                        "--attr_topk", str(args.actor_pref_attr_topk),
                        "--attr_floor", str(args.actor_pref_attr_floor),
                        "--save_dir", str(actor_dir.resolve()),
                        "--metrics_out", str(actor_metrics_path.resolve()),
                    ]
                else:
                    actor_metrics_path = actor_dir / "tiger_hca_pref_actor_metrics.json"
                    actor_cmd = [
                        python,
                        str(CODE_DIR / "train_tiger_hca_pref_actor.py"),
                        "--pair_path", str(actor_pair_path.resolve()),
                        "--tiger_ckpt", str(rollout_tiger_ckpt),
                        "--init_tiger_ckpt", str(learner_tiger_ckpt),
                        "--sid_mapping_path", str(sid_mapping_path),
                        "--model_size", str(args.model_size),
                        "--device", str(args.device),
                        "--seed", str(args.seed),
                        "--batch_size", str(args.actor_batch_size),
                        "--epochs", str(args.actor_epochs),
                        "--lr", str(args.actor_lr),
                        "--weight_decay", str(args.actor_weight_decay),
                        "--train_scope", str(args.actor_train_scope),
                        "--pref_beta", str(args.actor_pref_beta),
                        "--label_smoothing", str(args.actor_pref_label_smoothing),
                        "--sft_scale", str(args.actor_pref_sft_scale),
                        "--gap_scale", str(args.actor_pref_gap_scale),
                        "--gap_clip", str(args.actor_pref_gap_clip),
                        "--score_normalization", str(args.actor_pref_score_normalization),
                        "--attr_adv_mode", str(args.actor_pref_attr_adv_mode),
                        "--attr_pair_scale", str(args.actor_pref_attr_pair_scale),
                        "--attr_item_scale", str(args.actor_pref_attr_item_scale),
                        "--attr_credit_clip", str(args.actor_pref_attr_credit_clip),
                        "--attr_renorm_mode", str(args.actor_pref_attr_renorm_mode),
                        "--attr_topk", str(args.actor_pref_attr_topk),
                        "--attr_floor", str(args.actor_pref_attr_floor),
                        "--save_dir", str(actor_dir.resolve()),
                        "--metrics_out", str(actor_metrics_path.resolve()),
                    ]
                _actor_text, actor_time_sec = run_command(actor_cmd, actor_log)
                actor_group_time_sec += float(actor_pair_time_sec)
                actor_metrics = read_json(actor_metrics_path)
                if actor_method == "hybrid":
                    next_learner_tiger_ckpt = str((actor_dir / "tiger_hca_hybrid_actor_tiger.pth").resolve())
                else:
                    next_learner_tiger_ckpt = str((actor_dir / "tiger_hca_pref_actor_tiger.pth").resolve())

        if bool(actor_updated):
            rollout_policy_dir = iter_dir / "rollout_policy"
            rollout_policy_ckpt_path = rollout_policy_dir / "tiger_rollout_policy_tiger.pth"
            next_rollout_tiger_ckpt, sync_detail = sync_rollout_policy_checkpoint(
                rollout_ckpt=str(rollout_tiger_ckpt),
                learner_ckpt=str(next_learner_tiger_ckpt),
                mode=str(args.rollout_policy_sync_mode),
                tau=float(args.rollout_policy_sync_tau),
                output_path=rollout_policy_ckpt_path,
            )
            policy_sync_summary.update(sync_detail)

        after_eval_metrics: Dict[str, float] = {}
        after_eval_rollout_metrics: Dict[str, float] = {}
        after_eval_learner_metrics: Dict[str, float] = {}
        after_eval_time_sec = 0.0
        after_eval_rollout_time_sec = 0.0
        after_eval_learner_time_sec = 0.0
        if do_eval and bool(actor_updated):
            same_eval_ckpt = Path(str(next_rollout_tiger_ckpt)).resolve() == Path(str(next_learner_tiger_ckpt)).resolve()

            after_eval_learner_log = iter_dir / "after_eval_learner.log"
            after_eval_learner_cmd = build_policy_eval_cmd(
                python=python,
                tiger_ckpt=str(next_learner_tiger_ckpt),
                sid_mapping_path=str(sid_mapping_path),
                uirm_log_path=str(uirm_log_path),
                slate_size=int(args.slate_size),
                episode_batch_size=int(args.episode_batch_size),
                model_size=str(args.model_size),
                num_episodes=int(args.eval_episodes),
                max_steps_per_episode=int(args.max_steps_per_episode),
                beam_width=int(args.beam_width),
                initial_temper=float(args.initial_temper),
                item_correlation=float(args.item_correlation),
                seed=int(args.seed),
                max_hist_items=int(args.max_hist_items),
                device=str(args.device),
                use_phase2_blend=bool(args.eval_use_phase2_blend),
                phase2_blend_scale=float(args.eval_phase2_blend_scale),
                random_topk_sample=int(args.eval_random_topk_sample),
                random_item_prob=float(args.eval_random_item_prob),
            )
            after_eval_learner_text, after_eval_learner_time_sec = run_command(after_eval_learner_cmd, after_eval_learner_log)
            after_eval_learner_metrics = parse_eval_metrics(after_eval_learner_text)

            if same_eval_ckpt:
                after_eval_rollout_metrics = dict(after_eval_learner_metrics)
                after_eval_rollout_time_sec = float(after_eval_learner_time_sec)
                (iter_dir / "after_eval_rollout.log").write_text(after_eval_learner_text, encoding="utf-8")
                (iter_dir / "after_eval.log").write_text(after_eval_learner_text, encoding="utf-8")
            else:
                after_eval_rollout_log = iter_dir / "after_eval_rollout.log"
                after_eval_rollout_cmd = build_policy_eval_cmd(
                    python=python,
                    tiger_ckpt=str(next_rollout_tiger_ckpt),
                    sid_mapping_path=str(sid_mapping_path),
                    uirm_log_path=str(uirm_log_path),
                    slate_size=int(args.slate_size),
                    episode_batch_size=int(args.episode_batch_size),
                    model_size=str(args.model_size),
                    num_episodes=int(args.eval_episodes),
                    max_steps_per_episode=int(args.max_steps_per_episode),
                    beam_width=int(args.beam_width),
                    initial_temper=float(args.initial_temper),
                    item_correlation=float(args.item_correlation),
                    seed=int(args.seed),
                    max_hist_items=int(args.max_hist_items),
                    device=str(args.device),
                    use_phase2_blend=bool(args.eval_use_phase2_blend),
                    phase2_blend_scale=float(args.eval_phase2_blend_scale),
                    random_topk_sample=int(args.eval_random_topk_sample),
                    random_item_prob=float(args.eval_random_item_prob),
                )
                after_eval_rollout_text, after_eval_rollout_time_sec = run_command(after_eval_rollout_cmd, after_eval_rollout_log)
                after_eval_rollout_metrics = parse_eval_metrics(after_eval_rollout_text)
                (iter_dir / "after_eval.log").write_text(after_eval_rollout_text, encoding="utf-8")

            after_eval_time_sec = float(after_eval_rollout_time_sec)
            after_eval_metrics = dict(after_eval_rollout_metrics)
        elif do_eval:
            after_eval_metrics = dict(before_eval_metrics)
            after_eval_rollout_metrics = dict(before_eval_metrics)
            after_eval_learner_metrics = dict(before_eval_metrics)

        iter_wall_time_sec = float(time.perf_counter() - iter_start)
        cleanup_removed: List[str] = []
        if not bool(args.keep_heavy_intermediates):
            cleanup_targets: List[Path] = []
            cleanup_targets.extend(shard_chain_paths)
            cleanup_targets.extend(shard_output_paths)
            if chain_path.exists():
                cleanup_targets.append(chain_path)
            if actor_pair_path is not None and actor_pair_path.exists():
                cleanup_targets.append(actor_pair_path)
            cleanup_removed = cleanup_paths(cleanup_targets)
        iter_summary = {
            "iter": int(iter_idx),
            "input_tiger_ckpt": str(rollout_tiger_ckpt),
            "actor_method": str(actor_method),
            "reward_protocol": {
                "shared_page_credit_metric": bool(args.shared_page_credit_metric),
                "replay_recent_iters": int(args.replay_recent_iters),
                "slate_size": int(args.slate_size),
                "max_steps_per_episode": int(args.max_steps_per_episode),
                "beam_width": int(args.beam_width),
                "initial_temper": float(args.initial_temper),
                "item_correlation": float(args.item_correlation),
                "rollout_phase2_blend_scale": float(args.rollout_phase2_blend_scale),
                "rollout_random_topk_sample": int(args.rollout_random_topk_sample),
                "rollout_random_item_prob": float(args.rollout_random_item_prob),
                "eval_use_phase2_blend": bool(args.eval_use_phase2_blend),
                "eval_phase2_blend_scale": float(args.eval_phase2_blend_scale),
                "eval_random_topk_sample": int(args.eval_random_topk_sample),
                "eval_random_item_prob": float(args.eval_random_item_prob),
                "critic_ensemble_size": int(args.critic_ensemble_size),
                "critic_pessimism_beta": float(args.critic_pessimism_beta),
                "critic_target_heuristic_mix": float(args.critic_target_heuristic_mix),
                "critic_target_support_mix": float(args.critic_target_support_mix),
                "critic_target_response_mix": float(args.critic_target_response_mix),
                "critic_page_loss_scale": float(args.critic_page_loss_scale),
                "critic_item_loss_scale": float(args.critic_item_loss_scale),
                "critic_prefix_loss_scale": float(args.critic_prefix_loss_scale),
                "critic_page_huber_beta": float(args.critic_page_huber_beta),
                "critic_item_huber_beta": float(args.critic_item_huber_beta),
                "critic_prefix_huber_beta": float(args.critic_prefix_huber_beta),
                "critic_rank_loss_scale": float(args.critic_rank_loss_scale),
                "critic_monotonic_loss_scale": float(args.critic_monotonic_loss_scale),
                "critic_rank_min_gap": float(args.critic_rank_min_gap),
                "actor_credit_mode": str(args.actor_credit_mode),
                "actor_token_adv_field": str(args.actor_token_adv_field),
                "actor_item_adv_field": str(args.actor_item_adv_field),
                "actor_page_adv_field": str(args.actor_page_adv_field),
                "actor_page_reward_field": str(args.actor_page_reward_field),
                "actor_item_adv_scale": float(args.actor_item_adv_scale),
                "actor_page_adv_scale": float(args.actor_page_adv_scale),
                "actor_page_gate_scale": float(args.actor_page_gate_scale),
                "actor_page_gate_mode": str(args.actor_page_gate_mode),
                "actor_clip_eps": float(args.actor_clip_eps),
                "actor_kl_scale": float(args.actor_kl_scale),
                "actor_adaptive_kl_support_scale": float(args.actor_adaptive_kl_support_scale),
                "actor_adaptive_kl_unc_scale": float(args.actor_adaptive_kl_unc_scale),
                "actor_adaptive_clip_support_scale": float(args.actor_adaptive_clip_support_scale),
                "actor_adaptive_clip_unc_scale": float(args.actor_adaptive_clip_unc_scale),
                "actor_min_clip_eps": float(args.actor_min_clip_eps),
                "actor_trust_support_field": str(args.actor_trust_support_field),
                "actor_trust_unc_field": str(args.actor_trust_unc_field),
                "actor_group_reward_field": str(args.actor_group_reward_field),
                "actor_group_reward_transform": str(args.actor_group_reward_transform),
                "actor_group_margin_clip": float(args.actor_group_margin_clip),
                "actor_group_margin_temperature": float(args.actor_group_margin_temperature),
                "actor_group_support_penalty_scale": float(args.actor_group_support_penalty_scale),
                "actor_group_support_gap_temperature": float(args.actor_group_support_gap_temperature),
                "actor_group_support_gap_clip": float(args.actor_group_support_gap_clip),
                "actor_group_adaptive_beta_unc_scale": float(args.actor_group_adaptive_beta_unc_scale),
                "actor_group_adaptive_beta_support_scale": float(args.actor_group_adaptive_beta_support_scale),
                "actor_group_num_shards": int(args.actor_group_num_shards),
                "actor_pref_score_field": str(args.actor_pref_score_field),
                "actor_pref_safe_support_gap_max": float(args.actor_pref_safe_support_gap_max),
                "actor_pref_min_gap": float(args.actor_pref_min_gap),
                "actor_pref_max_pairs_per_group": int(args.actor_pref_max_pairs_per_group),
                "actor_pref_pair_mode": str(args.actor_pref_pair_mode),
                "actor_pref_exploit_score_field": str(args.actor_pref_exploit_score_field),
                "actor_pref_min_support_gap_delta": float(args.actor_pref_min_support_gap_delta),
                "actor_pref_min_unc_delta": float(args.actor_pref_min_unc_delta),
                "actor_pref_pair_score_gap_scale": float(args.actor_pref_pair_score_gap_scale),
                "actor_pref_pair_raw_q_gap_scale": float(args.actor_pref_pair_raw_q_gap_scale),
                "actor_pref_pair_unc_gap_scale": float(args.actor_pref_pair_unc_gap_scale),
                "actor_pref_pair_support_gap_scale": float(args.actor_pref_pair_support_gap_scale),
                "actor_pref_beta": float(args.actor_pref_beta),
                "actor_pref_label_smoothing": float(args.actor_pref_label_smoothing),
                "actor_pref_sft_scale": float(args.actor_pref_sft_scale),
                "actor_pref_gap_scale": float(args.actor_pref_gap_scale),
                "actor_pref_gap_clip": float(args.actor_pref_gap_clip),
                "actor_pref_score_normalization": str(args.actor_pref_score_normalization),
                "actor_pref_attr_adv_mode": str(args.actor_pref_attr_adv_mode),
                "actor_pref_attr_pair_scale": float(args.actor_pref_attr_pair_scale),
                "actor_pref_attr_item_scale": float(args.actor_pref_attr_item_scale),
                "actor_pref_attr_credit_clip": float(args.actor_pref_attr_credit_clip),
                "actor_pref_attr_renorm_mode": str(args.actor_pref_attr_renorm_mode),
                "actor_pref_attr_topk": int(args.actor_pref_attr_topk),
                "actor_pref_attr_floor": float(args.actor_pref_attr_floor),
                "actor_hybrid_pref_anchor_scale": float(args.actor_hybrid_pref_anchor_scale),
                "actor_posterior_score_field": str(args.actor_posterior_score_field),
                "actor_posterior_prior_field": str(args.actor_posterior_prior_field),
                "actor_posterior_score_scale": float(args.actor_posterior_score_scale),
                "actor_posterior_prior_scale": float(args.actor_posterior_prior_scale),
                "actor_posterior_temperature": float(args.actor_posterior_temperature),
                "actor_posterior_teacher_logit_clip": float(args.actor_posterior_teacher_logit_clip),
                "actor_posterior_teacher_safe_support_gap_max": float(args.actor_posterior_teacher_safe_support_gap_max),
                "actor_posterior_teacher_safe_uncertainty_ratio_max": float(args.actor_posterior_teacher_safe_uncertainty_ratio_max),
                "actor_posterior_teacher_topk": int(args.actor_posterior_teacher_topk),
                "actor_posterior_teacher_behavior_mix": float(args.actor_posterior_teacher_behavior_mix),
                "actor_posterior_reference_kl_mode": str(args.actor_posterior_reference_kl_mode),
                "actor_posterior_reference_kl_scale": float(args.actor_posterior_reference_kl_scale),
                "actor_posterior_score_normalization": str(args.actor_posterior_score_normalization),
                "actor_posterior_attr_adv_mode": str(args.actor_posterior_attr_adv_mode),
                "actor_posterior_attr_item_scale": float(args.actor_posterior_attr_item_scale),
                "actor_posterior_attr_temperature": float(args.actor_posterior_attr_temperature),
                "actor_posterior_attr_mix": float(args.actor_posterior_attr_mix),
                "actor_posterior_attr_credit_clip": float(args.actor_posterior_attr_credit_clip),
                "actor_posterior_attr_renorm_mode": str(args.actor_posterior_attr_renorm_mode),
            },
            "rollout_trace_path": str(trace_path.resolve()),
            "replay_trace_path": str(replay_trace_path.resolve()),
            "replay_trace_sources": [str(path.resolve()) for path in recent_trace_paths],
            "replay_lines_added": int(replay_lines_added),
            "replay_lines_total": int(replay_lines_total),
            "page_qcritic_bundle": str(current_critic_bundle),
            "page_qcritic_meta": str(current_critic_meta),
            "chain_path": str(chain_path.resolve()),
            "actor_ckpt": str(next_learner_tiger_ckpt),
            "actor_updated": bool(actor_updated),
            "did_eval": bool(do_eval),
            "before_eval": before_eval_metrics,
            "rollout_eval": rollout_metrics,
            "after_eval": after_eval_metrics,
            "after_eval_rollout": after_eval_rollout_metrics,
            "after_eval_learner": after_eval_learner_metrics,
            "eval_delta": metric_delta(after_eval_metrics, before_eval_metrics) if do_eval else {},
            "eval_delta_rollout": metric_delta(after_eval_rollout_metrics, before_eval_metrics) if do_eval else {},
            "eval_delta_learner": metric_delta(after_eval_learner_metrics, before_eval_metrics) if do_eval else {},
            "critic_metrics": critic_metrics,
            "chain_summary": chain_summary,
            "actor_group_summary": actor_group_summary,
            "actor_pair_summary": actor_pair_summary,
            "actor_metrics": actor_metrics,
            "timing": {
                "before_eval_time_sec": float(before_eval_time_sec),
                "rollout_time_sec": float(rollout_time_sec),
                "critic_time_sec": float(critic_time_sec),
                "chain_time_sec": float(chain_time_sec),
                "actor_group_time_sec": float(actor_group_time_sec),
                "actor_time_sec": float(actor_time_sec),
                "after_eval_time_sec": float(after_eval_time_sec),
                "after_eval_rollout_time_sec": float(after_eval_rollout_time_sec),
                "after_eval_learner_time_sec": float(after_eval_learner_time_sec),
                "iter_wall_time_sec": float(iter_wall_time_sec),
            },
            "rollout_policy_sync": policy_sync_summary,
            "cleanup_removed": cleanup_removed,
            "input_rollout_tiger_ckpt": str(rollout_tiger_ckpt),
            "input_learner_tiger_ckpt": str(learner_tiger_ckpt),
            "output_rollout_tiger_ckpt": str(next_rollout_tiger_ckpt),
            "output_learner_tiger_ckpt": str(next_learner_tiger_ckpt),
        }
        summary.append(iter_summary)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with summary_jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(iter_summary, ensure_ascii=False) + "\n")

        if not bool(args.disable_auto_plot):
            refresh_plots(
                python=python,
                output_root=output_root,
                summary_json_path=summary_path,
                summary_jsonl_path=summary_jsonl_path,
                dpi=int(args.plot_dpi),
            )

        rollout_tiger_ckpt = next_rollout_tiger_ckpt
        learner_tiger_ckpt = next_learner_tiger_ckpt
        print(json.dumps(iter_summary, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
