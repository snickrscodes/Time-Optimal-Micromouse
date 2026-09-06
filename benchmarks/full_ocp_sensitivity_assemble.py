from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .full_ocp_sensitivity import assemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("core", "smoke"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.output_dir, profile=args.profile, status_path=args.status)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")
    sys.stdout.flush(); sys.stderr.flush()


if __name__ == "__main__":
    main()
