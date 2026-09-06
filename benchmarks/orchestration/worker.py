from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from benchmarks.config import SUITES


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal isolated benchmark-suite worker.")
    parser.add_argument("--suite", required=True, choices=tuple(SUITES))
    parser.add_argument("--profile", required=True, choices=("core", "smoke"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()

    definition = SUITES[args.suite]
    if definition.module.startswith("benchmarks.orchestration."):
        raise RuntimeError(f"suite {args.suite} has a dedicated orchestration module")
    module = importlib.import_module(definition.module)
    module.run(args.output_dir, profile=args.profile)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")
    sys.stdout.flush(); sys.stderr.flush()


if __name__ == "__main__":
    main()
