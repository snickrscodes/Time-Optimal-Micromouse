#!/usr/bin/env python3
"""Fresh-process worker for one production active-basis fixed topology."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.geometry_homotopy.serial_runtime import force_single_thread_environment
force_single_thread_environment()

import argparse
import json

from planning.active_basis_optimizer import run_active_basis_pipeline_in_process


def _marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("complete\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--completion-marker", type=Path, required=True)
    args = ap.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    run_active_basis_pipeline_in_process(request, args.work_dir)
    _marker(args.completion_marker)


if __name__ == "__main__":
    main()
