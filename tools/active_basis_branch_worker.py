#!/usr/bin/env python3
"""Generic supervised selective-basis worker for production planner routes.

This is the production-route analogue of the historical-five worker.  It is
launched synchronously by the active-basis route worker; process isolation is a
watchdog boundary only, never branch parallelism.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
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

from planning.maze_routes import OpenRoomSpan
from tools.geometry_homotopy.basis_branch import BasisBranchPolicy, run_basis_activation_branch
from tools.geometry_homotopy.goal_entry import build_goal_entry_problem
from tools.geometry_homotopy.research_common import parameter_sha256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
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


def _write_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("complete\n", encoding="utf-8")
    os.replace(tmp, path)


def _spans(rows: list[dict]) -> tuple[OpenRoomSpan, ...]:
    return tuple(
        OpenRoomSpan(
            start_index=int(row["start_index"]),
            end_index=int(row["end_index"]),
            cells=tuple(tuple(int(v) for v in c) for c in row["cells"]),
            xmin=int(row["xmin"]), xmax=int(row["xmax"]),
            ymin=int(row["ymin"]), ymax=int(row["ymax"]),
            block_index=int(row.get("block_index", -1)),
        )
        for row in rows
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--reduced", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--completion-marker", type=Path, required=True)
    args = ap.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    cells = tuple(tuple(int(v) for v in cell) for cell in request["cells"])
    spans = _spans(request.get("open_room_spans", []))
    body_length = float(request["body_length"])
    body_height = float(request["body_height"])
    clearance = float(request.get("clearance", 0.0))
    refinement = int(request.get("refinement_factor", 1))
    reduced = build_goal_entry_problem(
        cells,
        body_length=body_length,
        body_height=body_height,
        clearance=clearance,
        corridor_mode="maximal_runs",
        refinement_factor=refinement,
        open_room_spans=spans,
    )
    full = build_goal_entry_problem(
        cells,
        body_length=body_length,
        body_height=body_height,
        clearance=clearance,
        corridor_mode="overlapping_cover",
        refinement_factor=refinement,
        open_room_spans=spans,
    )
    x = np.asarray(np.load(args.reduced), dtype=float)
    if x.shape != reduced.initial_parameters.shape:
        raise ValueError(
            f"reduced checkpoint shape {x.shape} does not match reduced basis {reduced.initial_parameters.shape}"
        )
    policy = BasisBranchPolicy(**request["homotopy_policy"]["basis_branch_policy"])

    started = time.perf_counter()
    progress_path = args.output.with_suffix(".events.jsonl")
    incumbent_meta = args.output.with_suffix(".incumbent.json")
    incumbent_x = args.output.with_suffix(".incumbent.npy")
    incumbent_cells = args.output.with_suffix(".incumbent.segment_cells.npy")
    for stale in (progress_path, incumbent_meta, incumbent_x, incumbent_cells, args.completion_marker):
        stale.unlink(missing_ok=True)

    def event_callback(event: str, payload: dict) -> None:
        _append_jsonl(progress_path, {
            "event": event,
            "worker_monotonic_seconds": time.perf_counter(),
            **payload,
        })

    def incumbent_callback(problem, candidate_x, payload: dict) -> None:
        kind = "reduced" if problem is reduced else "hybrid"
        _atomic_npy(incumbent_x, np.asarray(candidate_x, dtype=float))
        segment_cells_path = None
        if kind == "hybrid":
            _atomic_npy(incumbent_cells, np.asarray(problem.corridor.segment_cells, dtype=np.int64))
            segment_cells_path = str(incumbent_cells)
        else:
            incumbent_cells.unlink(missing_ok=True)
        _atomic_json(incumbent_meta, {
            "schema_version": 1,
            "request_digest": request["request_digest"],
            "kind": kind,
            "parameters": str(incumbent_x),
            "segment_cells": segment_cells_path,
            "parameter_sha256": parameter_sha256(candidate_x),
            **payload,
        })

    _atomic_json(args.output, {
        "schema_version": 1,
        "status": "running",
        "request_digest": request["request_digest"],
        "policy": asdict(policy),
        "thread_environment": thread_environment_snapshot(),
        "reduced_parameter_sha256": parameter_sha256(x),
    })

    result = run_basis_activation_branch(
        cells, reduced, full, x,
        policy=policy,
        event_callback=event_callback,
        incumbent_callback=incumbent_callback,
    )
    final_path = args.output.with_suffix(".final.npy")
    _atomic_npy(final_path, result.final_parameters)
    final_kind = "reduced" if result.final_problem is reduced else "hybrid"
    final_segment_cells = None
    if final_kind == "hybrid":
        cell_path = args.output.with_suffix(".segment_cells.npy")
        _atomic_npy(cell_path, np.asarray(result.final_problem.corridor.segment_cells, dtype=np.int64))
        final_segment_cells = str(cell_path)

    _atomic_json(args.output, {
        "schema_version": 1,
        "status": "complete",
        "request_digest": request["request_digest"],
        "policy": asdict(policy),
        "thread_environment": thread_environment_snapshot(),
        "worker_wall_seconds": time.perf_counter() - started,
        "progress_jsonl": str(progress_path),
        "latest_incumbent": str(incumbent_meta) if incumbent_meta.exists() else None,
        "branch_converged": bool(result.converged),
        "branch_status": result.status,
        "branch_termination_reason": result.termination_reason,
        "final_kind": final_kind,
        "final_parameters": str(final_path),
        "final_segment_cells": final_segment_cells,
        "final_parameter_sha256": parameter_sha256(result.final_parameters),
        "result": result.summary_dict(),
    })
    _write_marker(args.completion_marker)


if __name__ == "__main__":
    main()
