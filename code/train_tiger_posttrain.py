# -*- coding: utf-8 -*-
"""Unified TIGER post-train launcher.

Method mapping:
  dpo  -> train_tiger_spo.py
  sdpo -> train_tiger_slate_spo.py
  grpo -> train_tiger_hca_grpo_actor.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


METHOD_TO_SCRIPT = {
    "dpo": "train_tiger_spo.py",
    "sdpo": "train_tiger_slate_spo.py",
    "grpo": "train_tiger_hca_grpo_actor.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified TIGER post-train launcher.")
    parser.add_argument("method", choices=sorted(METHOD_TO_SCRIPT.keys()))
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve().with_name(METHOD_TO_SCRIPT[str(args.method)])
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"[TIGER-POSTTRAIN] method={args.method} script={script_path.name}")
    print(f"[TIGER-POSTTRAIN] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
