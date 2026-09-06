from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .full_ocp import assemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble isolated Level-2 component results.")
    parser.add_argument("--profile", choices=("core", "smoke"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.output_dir, profile=args.profile)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")
    sys.stdout.flush(); sys.stderr.flush()


if __name__ == "__main__":
    main()
