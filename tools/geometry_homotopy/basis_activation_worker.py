#!/usr/bin/env python3
"""Serial supervised selective-basis branch worker.

The parent state machine launches exactly one instance at a time and waits for
it to complete.  Process isolation is used only as a watchdog boundary: if a
numerically pathological native/HiGHS call ignores its internal time limit, the
parent can terminate this worker while retaining the already-certified reduced
incumbent.
"""
from __future__ import annotations

# Force deterministic single-thread numerical backends before NumPy/SciPy load.
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.geometry_homotopy.serial_runtime import (
    force_single_thread_environment,
    thread_environment_snapshot,
)
force_single_thread_environment()

import argparse
from dataclasses import asdict
import json
import time

import numpy as np

from tools.geometry_homotopy.basis_branch import BasisBranchPolicy, run_basis_activation_branch
from tools.geometry_homotopy.research_common import build_historical_pair, parameter_sha256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_completion_marker(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("complete\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reduced", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--policy-json", type=Path)
    ap.add_argument("--completion-marker", type=Path)
    args = ap.parse_args()

    policy = BasisBranchPolicy()
    if args.policy_json is not None:
        policy = BasisBranchPolicy(**json.loads(args.policy_json.read_text(encoding="utf-8")))

    cells, reduced, full = build_historical_pair(args.seed)
    x = np.asarray(np.load(args.reduced), dtype=float)
    if x.shape != reduced.initial_parameters.shape:
        raise ValueError(
            f"reduced checkpoint shape {x.shape} does not match seed-{args.seed} reduced basis "
            f"{reduced.initial_parameters.shape}"
        )

    started = time.perf_counter()
    progress_path = args.output.with_suffix(".events.jsonl")
    incumbent_meta_path = args.output.with_suffix(".incumbent.json")
    incumbent_x_path = args.output.with_suffix(".incumbent.npy")
    incumbent_segment_cells_path = args.output.with_suffix(".incumbent.segment_cells.npy")
    for stale in (progress_path, incumbent_meta_path, incumbent_x_path, incumbent_segment_cells_path):
        stale.unlink(missing_ok=True)

    def event_callback(event: str, payload: dict) -> None:
        _append_jsonl(progress_path, {
            "event": event,
            "worker_monotonic_seconds": time.perf_counter(),
            **payload,
        })

    def incumbent_callback(problem, candidate_x, payload: dict) -> None:
        kind = "reduced" if problem is reduced else "hybrid"
        _atomic_npy(incumbent_x_path, np.asarray(candidate_x, dtype=float))
        segment_cells_path = None
        if kind == "hybrid":
            _atomic_npy(
                incumbent_segment_cells_path,
                np.asarray(problem.corridor.segment_cells, dtype=np.int64),
            )
            segment_cells_path = str(incumbent_segment_cells_path)
        else:
            incumbent_segment_cells_path.unlink(missing_ok=True)
        _atomic_json(incumbent_meta_path, {
            "schema_version": 1,
            "seed": int(args.seed),
            "kind": kind,
            "parameters": str(incumbent_x_path),
            "segment_cells": segment_cells_path,
            "parameter_sha256": parameter_sha256(candidate_x),
            **payload,
        })

    _atomic_json(args.output, {
        "schema_version": 2,
        "status": "running",
        "seed": int(args.seed),
        "policy": asdict(policy),
        "thread_environment": thread_environment_snapshot(),
        "reduced_parameter_sha256": parameter_sha256(x),
    })

    result = run_basis_activation_branch(
        cells,
        reduced,
        full,
        x,
        policy=policy,
        event_callback=event_callback,
        incumbent_callback=incumbent_callback,
    )
    summary = result.summary_dict()
    final_x_path = args.output.with_suffix(".final.npy")
    _atomic_npy(final_x_path, result.final_parameters)

    final_kind = "reduced" if result.final_problem is reduced else "hybrid"
    segment_cells_path = None
    if final_kind == "hybrid":
        segment_cells_path = args.output.with_suffix(".segment_cells.npy")
        _atomic_npy(segment_cells_path, np.asarray(result.final_problem.corridor.segment_cells, dtype=np.int64))

    payload = {
        "schema_version": 2,
        "status": "complete",
        "seed": int(args.seed),
        "policy": asdict(policy),
        "thread_environment": thread_environment_snapshot(),
        "worker_wall_seconds": time.perf_counter() - started,
        "progress_jsonl": str(progress_path),
        "latest_incumbent": str(incumbent_meta_path) if incumbent_meta_path.exists() else None,
        "branch_converged": bool(result.converged),
        "branch_status": result.status,
        "branch_termination_reason": result.termination_reason,
        "final_kind": final_kind,
        "final_parameters": str(final_x_path),
        "final_segment_cells": None if segment_cells_path is None else str(segment_cells_path),
        "final_parameter_sha256": parameter_sha256(result.final_parameters),
        "result": summary,
    }
    _atomic_json(args.output, payload)
    # Marker is written only after result + associated arrays are fully durable
    # from Python's perspective.  Parent may then kill a teardown hang safely.
    _write_completion_marker(args.completion_marker)


if __name__ == "__main__":
    main()
