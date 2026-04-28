import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


CODE_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chain and train TIGER-HCAA joint critic.")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--chain_path", type=str, default="")
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--uirm_log_path", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--chain_credit_mode", type=str, default="return")
    parser.add_argument("--allocator_head_path", type=str, default="")
    parser.add_argument("--allocator_meta_path", type=str, default="")
    parser.add_argument("--allocator_blend_alpha", type=float, default=0.7)
    parser.add_argument("--allocator_keep_topk", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hazard_lambda", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(CODE_DIR), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def maybe_append_allocator(cmd: List[str], args: argparse.Namespace) -> None:
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_head_path", str(Path(args.allocator_head_path).resolve())])
    if str(args.allocator_meta_path).strip():
        cmd.extend(["--allocator_meta_path", str(Path(args.allocator_meta_path).resolve())])
    if str(args.allocator_head_path).strip():
        cmd.extend(["--allocator_blend_alpha", str(float(args.allocator_blend_alpha))])
        if int(args.allocator_keep_topk) > 0:
            cmd.extend(["--allocator_keep_topk", str(int(args.allocator_keep_topk))])


def main() -> int:
    args = parse_args()
    py = str(args.python_exe)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    chain_path = Path(args.chain_path).resolve() if str(args.chain_path).strip() else save_dir / "rollout_chain.jsonl"

    if not chain_path.exists():
        chain_cmd = [
            py,
            str((CODE_DIR / "build_tiger_slate_credit_chain.py").resolve()),
            "--trace_path",
            str(Path(args.trace_path).resolve()),
            "--uirm_log_path",
            str(Path(args.uirm_log_path).resolve()),
            "--sid_mapping_path",
            str(Path(args.sid_mapping_path).resolve()),
            "--device",
            "cpu",
            "--credit_mode",
            str(args.chain_credit_mode),
            "--max_hist_items",
            str(int(args.max_hist_items)),
            "--output_path",
            str(chain_path),
        ]
        maybe_append_allocator(chain_cmd, args)
        run_cmd(chain_cmd)

    train_cmd = [
        py,
        str((CODE_DIR / "tiger_hcaa" / "train_hcaa_joint_critic.py").resolve()),
        "--trace_path",
        str(Path(args.trace_path).resolve()),
        "--chain_path",
        str(chain_path),
        "--tiger_ckpt",
        str(Path(args.tiger_ckpt).resolve()),
        "--uirm_log_path",
        str(Path(args.uirm_log_path).resolve()),
        "--sid_mapping_path",
        str(Path(args.sid_mapping_path).resolve()),
        "--model_size",
        str(args.model_size),
        "--device",
        str(args.device),
        "--max_hist_items",
        str(int(args.max_hist_items)),
        "--gamma",
        str(float(args.gamma)),
        "--hazard_lambda",
        str(float(args.hazard_lambda)),
        "--batch_size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(float(args.lr)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--valid_ratio",
        str(float(args.valid_ratio)),
        "--save_dir",
        str(save_dir),
    ]
    run_cmd(train_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

