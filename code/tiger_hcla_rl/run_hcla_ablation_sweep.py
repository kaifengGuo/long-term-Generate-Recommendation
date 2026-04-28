import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
CHAIN_SCRIPT = THIS_DIR / "build_hcla_longterm_chain.py"
CRITIC_SCRIPT = THIS_DIR / "train_hcla_longterm_critic.py"
ACTOR_SCRIPT = THIS_DIR / "train_hcla_actor.py"
EVAL_SCRIPT = PROJECT_ROOT / "eval_tiger_env.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible HCLA-RL ablation sweeps from a JSON config file.")
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--configs_json", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026, help="Training seed for chain/critic/actor runs.")
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--episode_batch_size", type=int, default=32)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument(
        "--eval_seed",
        action="append",
        dest="eval_seeds",
        type=int,
        help="Evaluation seed. Repeat this flag for multi-seed sweeps.",
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.35)
    parser.add_argument("--credit_mode", type=str, default="centered", choices=["return", "centered", "zscore"])
    parser.add_argument("--credit_clip", type=float, default=3.0)
    parser.add_argument("--reward_preset", type=str, default="click_longview")
    parser.add_argument("--chain_max_pages", type=int, default=0)
    parser.add_argument("--actor_epochs", type=int, default=1)
    parser.add_argument("--critic_epochs", type=int, default=1)
    parser.add_argument("--actor_batch_size", type=int, default=32)
    parser.add_argument("--critic_batch_size", type=int, default=32)
    parser.add_argument("--skip_existing", action="store_true", help="Reuse finished config outputs if present.")
    parser.add_argument("--max_configs", type=int, default=0)
    return parser.parse_args()


def resolve_path(path_str: str, base_dir: Path | None = None) -> str:
    text = str(path_str).strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        return str(path.resolve())
    anchor = base_dir or PROJECT_ROOT
    return str((anchor / path).resolve())


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip()).strip("_") or "cfg"


def load_configs(configs_path: Path, max_configs: int) -> List[Dict[str, Any]]:
    payload = json.loads(configs_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        configs = payload.get("configs", [])
    else:
        configs = payload
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"No configs found in {configs_path}")
    out: List[Dict[str, Any]] = []
    for cfg in configs:
        if not isinstance(cfg, dict):
            raise ValueError("Each config must be a JSON object")
        name = str(cfg.get("name", "")).strip()
        if not name:
            raise ValueError(f"Config missing non-empty name: {cfg}")
        out.append(cfg)
    if int(max_configs) > 0:
        out = out[: int(max_configs)]
    return out


def encode_cli_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_cli_args(arg_map: Dict[str, Any]) -> List[str]:
    cmd: List[str] = []
    for key, value in arg_map.items():
        if value is None:
            continue
        flag = f"--{str(key)}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, encode_cli_value(value)])
    return cmd


def run_command(cmd: Sequence[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        list(cmd),
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
        "ild": r"ILD:\s*([0-9.\-]+)",
        "click": r"is_click:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "long_view": r"long_view:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "like": r"is_like:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "comment": r"is_comment:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "forward": r"is_forward:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "follow": r"is_follow:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
        "hate": r"is_hate:\s*\d+/\d+\s*\(([0-9.\-]+)%\)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def aggregate_metrics(metric_list: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for item in metric_list for k in item.keys()})
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [float(item[key]) for item in metric_list if key in item]
        if not vals:
            continue
        out[key] = {
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return out


def metric_delta(after: Dict[str, Dict[str, float]], before: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    out: Dict[str, float] = {}
    for key in keys:
        if key in before and key in after:
            out[key] = float(after[key]["mean"] - before[key]["mean"])
    return out


def resolve_eval_batch_size(num_episodes: int, default_batch_size: int) -> int:
    if int(num_episodes) <= 0:
        return max(1, int(default_batch_size))
    return max(1, min(int(default_batch_size), int(num_episodes)))


def evaluate_checkpoint(
    *,
    python: str,
    ckpt_path: str,
    label: str,
    output_dir: Path,
    args: argparse.Namespace,
    eval_seeds: Sequence[int],
) -> Dict[str, Any]:
    batch_size = resolve_eval_batch_size(int(args.eval_episodes), int(args.episode_batch_size))
    seed_runs: List[Dict[str, Any]] = []
    for seed in eval_seeds:
        log_path = output_dir / label / f"seed_{seed}.log"
        cmd = [
            python,
            str(EVAL_SCRIPT),
            "--tiger_ckpt", str(ckpt_path),
            "--sid_mapping_path", str(Path(args.sid_mapping_path).resolve()),
            "--uirm_log_path", str(Path(args.uirm_log_path).resolve()),
            "--slate_size", str(args.slate_size),
            "--episode_batch_size", str(batch_size),
            "--model_size", str(args.model_size),
            "--num_episodes", str(args.eval_episodes),
            "--max_steps_per_episode", str(args.max_steps_per_episode),
            "--max_step_per_episode", str(args.max_steps_per_episode),
            "--beam_width", str(args.beam_width),
            "--single_response",
            "--seed", str(seed),
            "--max_hist_items", str(args.max_hist_items),
            "--device", str(args.device),
        ]
        text = run_command(cmd, log_path)
        seed_runs.append(
            {
                "seed": int(seed),
                "log_path": str(log_path.resolve()),
                "metrics": parse_eval_metrics(text),
            }
        )
    return {
        "label": str(label),
        "tiger_ckpt": str(Path(ckpt_path).resolve()),
        "seed_runs": seed_runs,
        "aggregate": aggregate_metrics([x["metrics"] for x in seed_runs]),
    }


def train_config(
    *,
    python: str,
    cfg: Dict[str, Any],
    cfg_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    chain_dir = cfg_dir
    critic_dir = cfg_dir / "critic"
    actor_dir = cfg_dir / "actor"
    chain_path = chain_dir / "hcla_chain.jsonl"

    chain_args = {
        "trace_path": str(Path(args.trace_path).resolve()),
        "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "device": "cpu",
        "seed": int(args.seed),
        "gamma": float(args.gamma),
        "max_hist_items": int(args.max_hist_items),
        "credit_mode": str(args.credit_mode),
        "credit_clip": float(args.credit_clip),
        "reward_preset": str(args.reward_preset),
        "hazard_lambda": float(args.hazard_lambda),
        "output_path": str(chain_path),
    }
    if int(args.chain_max_pages) > 0:
        chain_args["max_pages"] = int(args.chain_max_pages)
    chain_args.update(cfg.get("chain_args", {}))
    run_command(
        [python, str(CHAIN_SCRIPT), *build_cli_args(chain_args)],
        cfg_dir / "build_chain.log",
    )

    critic_args = {
        "chain_path": str(chain_path),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "model_size": str(args.model_size),
        "device": str(args.device),
        "seed": int(args.seed),
        "batch_size": int(args.critic_batch_size),
        "epochs": int(args.critic_epochs),
        "save_dir": str(critic_dir),
    }
    critic_args.update(cfg.get("critic_args", {}))
    run_command(
        [python, str(CRITIC_SCRIPT), *build_cli_args(critic_args)],
        cfg_dir / "train_critic.log",
    )

    actor_args = {
        "chain_path": str(chain_path),
        "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
        "init_tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
        "critic_bundle_path": str((critic_dir / "critic_bundle.pt").resolve()),
        "critic_meta_path": str((critic_dir / "critic_bundle_meta.json").resolve()),
        "model_size": str(args.model_size),
        "device": str(args.device),
        "seed": int(args.seed),
        "batch_size": int(args.actor_batch_size),
        "epochs": int(args.actor_epochs),
        "save_dir": str(actor_dir),
    }
    actor_args.update(cfg.get("actor_args", {}))
    run_command(
        [python, str(ACTOR_SCRIPT), *build_cli_args(actor_args)],
        cfg_dir / "train_actor.log",
    )
    actor_ckpt = actor_dir / "hcla_actor_tiger.pth"
    return {
        "chain_path": str(chain_path.resolve()),
        "critic_bundle_path": str((critic_dir / "critic_bundle.pt").resolve()),
        "critic_meta_path": str((critic_dir / "critic_bundle_meta.json").resolve()),
        "actor_ckpt": str(actor_ckpt.resolve()),
    }


def main() -> int:
    args = parse_args()
    python = sys.executable
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configs_path = Path(args.configs_json).resolve()
    configs = load_configs(configs_path, int(args.max_configs))
    eval_seeds = args.eval_seeds or [2026]
    baseline_path = output_root / "baseline_eval.json"
    summary_path = output_root / "sweep_summary.json"

    summary: Dict[str, Any] = {
        "config": {
            "tiger_ckpt": str(Path(args.tiger_ckpt).resolve()),
            "uirm_log_path": str(Path(args.uirm_log_path).resolve()),
            "sid_mapping_path": str(Path(args.sid_mapping_path).resolve()),
            "trace_path": str(Path(args.trace_path).resolve()),
            "model_size": str(args.model_size),
            "device": str(args.device),
            "train_seed": int(args.seed),
            "eval_seeds": [int(x) for x in eval_seeds],
            "eval_episodes": int(args.eval_episodes),
        },
        "baseline": {},
        "results": [],
    }

    if baseline_path.exists() and bool(args.skip_existing):
        baseline_eval = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline_eval = evaluate_checkpoint(
            python=python,
            ckpt_path=str(Path(args.tiger_ckpt).resolve()),
            label="base",
            output_dir=output_root / "eval",
            args=args,
            eval_seeds=eval_seeds,
        )
        baseline_path.write_text(json.dumps(baseline_eval, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["baseline"] = baseline_eval
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for cfg in configs:
        cfg_name = str(cfg["name"])
        cfg_slug = safe_name(cfg_name)
        cfg_dir = output_root / cfg_slug
        cfg_dir.mkdir(parents=True, exist_ok=True)
        result: Dict[str, Any] = {
            "name": cfg_name,
            "slug": cfg_slug,
            "notes": str(cfg.get("notes", "")),
            "raw_config": cfg,
        }
        reuse_ckpt = str(cfg.get("actor_ckpt", "")).strip()
        if reuse_ckpt:
            actor_ckpt = resolve_path(reuse_ckpt, configs_path.parent)
            result["mode"] = "reuse_ckpt"
            result["artifacts"] = {"actor_ckpt": actor_ckpt}
        else:
            result["mode"] = "train"
            actor_ckpt_path = cfg_dir / "actor" / "hcla_actor_tiger.pth"
            if actor_ckpt_path.exists() and bool(args.skip_existing):
                result["artifacts"] = {
                    "actor_ckpt": str(actor_ckpt_path.resolve()),
                    "chain_path": str((cfg_dir / "hcla_chain.jsonl").resolve()),
                    "critic_bundle_path": str((cfg_dir / "critic" / "critic_bundle.pt").resolve()),
                    "critic_meta_path": str((cfg_dir / "critic" / "critic_bundle_meta.json").resolve()),
                }
            else:
                result["artifacts"] = train_config(
                    python=python,
                    cfg=cfg,
                    cfg_dir=cfg_dir,
                    args=args,
                )
            actor_ckpt = str(result["artifacts"]["actor_ckpt"])

        actor_eval_path = cfg_dir / "actor_eval.json"
        if actor_eval_path.exists() and bool(args.skip_existing):
            actor_eval = json.loads(actor_eval_path.read_text(encoding="utf-8"))
        else:
            actor_eval = evaluate_checkpoint(
                python=python,
                ckpt_path=actor_ckpt,
                label=cfg_slug,
                output_dir=cfg_dir / "eval",
                args=args,
                eval_seeds=eval_seeds,
            )
            actor_eval_path.write_text(json.dumps(actor_eval, ensure_ascii=False, indent=2), encoding="utf-8")
        result["actor_eval"] = actor_eval
        result["delta_vs_baseline"] = metric_delta(actor_eval["aggregate"], baseline_eval["aggregate"])
        summary["results"].append(result)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        delta = result["delta_vs_baseline"]
        print(
            f"[ablation] {cfg_name}: "
            f"delta_reward={delta.get('total_reward', float('nan')):+.4f} | "
            f"delta_click={delta.get('click', float('nan')):+.4f} | "
            f"delta_long_view={delta.get('long_view', float('nan')):+.4f}"
        )

    ranking = sorted(
        [
            {
                "name": item["name"],
                "delta_reward": float(item.get("delta_vs_baseline", {}).get("total_reward", float("-inf"))),
                "delta_click": float(item.get("delta_vs_baseline", {}).get("click", float("nan"))),
                "delta_long_view": float(item.get("delta_vs_baseline", {}).get("long_view", float("nan"))),
            }
            for item in summary["results"]
        ],
        key=lambda x: x["delta_reward"],
        reverse=True,
    )
    summary["ranking"] = ranking
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablation] wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
