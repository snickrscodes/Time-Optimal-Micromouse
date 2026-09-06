"""Production planner API for the qualified active-basis route optimizer.

The numerical policy is the frozen historical-five policy.  Productionization
adds only route/body/boundary-condition parameterization, durable route work
records, and a killable outer worker so the main planner never inherits solver
thread state or an unkillable native call.

The outer worker runs one route at a time.  Its selective-basis phase launches
one additional *synchronous* worker solely as a watchdog boundary, preserving
the qualification campaign's reduced-incumbent fallback semantics.  There is
no route or branch parallelism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from planning.certification import certify_parameters_on_problem
from segment.physics_profiles import get_physics_profile
from segment.physics_identity import physics_model_identity, physics_model_signature
from optimization import CorridorModel, ConstraintPool, knot_parameters_to_raw, scalar_reverse_solver
from optimization import time_model as time_model_dispatch
from planning.maze_routes import OpenRoomSpan, RouteOptimizationProblem
from tools.geometry_homotopy.active_set import CurvatureObjective
from tools.geometry_homotopy.basis_branch import BasisBranchPolicy
from tools.geometry_homotopy.goal_entry import build_goal_entry_problem
from tools.geometry_homotopy.research_common import (
    geometry_metrics,
    make_guard,
    parameter_sha256,
    run_curvature,
    run_time,
    stage_summary,
)
from tools.geometry_homotopy.serial_runtime import (
    force_single_thread_environment,
    run_serial_worker,
    thread_environment_snapshot,
)
from tools.geometry_homotopy.state_machine import HomotopyPolicy, curvature_switch_decision

Array = NDArray[np.float64]
Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ActiveBasisExecutionSettings:
    """Non-numerical production orchestration settings."""

    work_root: str | None = None
    outer_timeout_grace_seconds: float = 5.0
    require_convergence: bool = True
    reuse_complete_result: bool = True
    incomplete_basis_retries: int = 1

    def __post_init__(self) -> None:
        if self.outer_timeout_grace_seconds <= 0.0:
            raise ValueError("outer_timeout_grace_seconds must be positive")
        if self.incomplete_basis_retries < 0:
            raise ValueError("incomplete_basis_retries must be nonnegative")


class ActiveBasisNonConvergenceError(RuntimeError):
    """A route has a certified incumbent but has not established basis closure."""


@dataclass(frozen=True, slots=True)
class ActiveBasisRouteResult:
    cells: tuple[Cell, ...]
    problem: RouteOptimizationProblem
    parameters: Array
    time: float
    converged: bool
    basis_kind: str
    work_dir: str
    request_digest: str
    worker_status: str
    summary: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: Sequence[float] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "monotonic_seconds": time.perf_counter(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if os.environ.get("AME_PROGRESS_STDOUT", "").strip().lower() in {"1", "true", "yes", "on"}:
        bits = [f"[active-basis] {event}"]
        stage = payload.get("stage")
        if stage is not None:
            bits.append(f"stage={stage}")
        epoch = payload.get("epoch")
        if epoch is not None:
            bits.append(f"epoch={epoch}")
        result = payload.get("result")
        if isinstance(result, dict):
            objective = result.get("objective", result.get("current_objective"))
            if objective is not None:
                try:
                    bits.append(f"T={float(objective):.9f}s")
                except (TypeError, ValueError):
                    pass
            stop = result.get("stop_reason")
            if stop:
                bits.append(f"stop={stop}")
        if event == "route_start":
            try:
                bits.append(f"T0={float(payload.get('initial_time')):.9f}s")
            except (TypeError, ValueError):
                pass
        if event == "route_finish":
            try:
                bits.append(f"T={float(payload.get('time')):.9f}s")
            except (TypeError, ValueError):
                pass
        print(" ".join(bits), flush=True)


def _span_dict(span: OpenRoomSpan) -> dict[str, Any]:
    return {
        "start_index": span.start_index,
        "end_index": span.end_index,
        "cells": [list(c) for c in span.cells],
        "xmin": span.xmin,
        "xmax": span.xmax,
        "ymin": span.ymin,
        "ymax": span.ymax,
        "block_index": span.block_index,
    }


def _spans(rows: Sequence[dict[str, Any]]) -> tuple[OpenRoomSpan, ...]:
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


def _policy_from_dict(row: dict[str, Any]) -> HomotopyPolicy:
    payload = dict(row)
    payload["basis_branch_policy"] = BasisBranchPolicy(**payload["basis_branch_policy"])
    for name in (
        "curvature_batch_schedule", "curvature_trust_schedule", "curvature_guard_sequence",
        "reduced_time_batch_schedule", "reduced_time_trust_schedule",
    ):
        if name in payload:
            payload[name] = tuple(payload[name])
    return HomotopyPolicy(**payload)


def _scalar_time(problem: RouteOptimizationProblem, x: Sequence[float], request: dict[str, Any]) -> float:
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    return float(time_model_dispatch.evaluate_time_scalar(
        raw,
        init_w=float(request["init_w"]),
        terminal_w_max=request.get("terminal_w_max"),
        initial_k=problem.initial_state.k,
        n_scan=int(request["n_scan"]),
        envelope_scan=int(request["envelope_scan"]),
        domain_scan=int(request["domain_scan"]),
    ))


def _curvature(problem: RouteOptimizationProblem, x: Sequence[float]) -> float:
    return float(CurvatureObjective(problem.initial_state.k)(x)[0])


def _build_pair(request: dict[str, Any]) -> tuple[tuple[Cell, ...], RouteOptimizationProblem, RouteOptimizationProblem]:
    cells = tuple(tuple(int(v) for v in cell) for cell in request["cells"])
    spans = _spans(request.get("open_room_spans", ()))
    kwargs = dict(
        body_length=float(request["body_length"]),
        body_height=float(request["body_height"]),
        clearance=float(request.get("clearance", 0.0)),
        refinement_factor=int(request.get("refinement_factor", 1)),
        open_room_spans=spans,
    )
    reduced = build_goal_entry_problem(cells, corridor_mode="maximal_runs", **kwargs)
    full = build_goal_entry_problem(cells, corridor_mode="overlapping_cover", **kwargs)
    return cells, reduced, full


def _problem_from_kind(
    kind: str,
    x: Array,
    segment_cells_path: str | Path | None,
    reduced: RouteOptimizationProblem,
    full: RouteOptimizationProblem,
) -> RouteOptimizationProblem:
    if kind == "reduced":
        return replace(reduced, initial_parameters=x.copy())
    if kind != "hybrid":
        raise ValueError(f"unknown active-basis kind {kind!r}")
    if segment_cells_path is None:
        raise ValueError("hybrid result omitted segment-cell assignment")
    assignment = tuple(int(v) for v in np.load(segment_cells_path).tolist())
    corridor = CorridorModel(
        full.corridor.cells, assignment, full.corridor.body, full.corridor.clearance
    )
    return replace(full, corridor=corridor, initial_parameters=x.copy())


def _run_basis_branch(
    request: dict[str, Any], reduced: RouteOptimizationProblem, full: RouteOptimizationProblem,
    reduced_x: Array, work_dir: Path, policy: HomotopyPolicy, remaining_seconds: float,
    event_path: Path,
) -> tuple[RouteOptimizationProblem, Array, str, dict[str, Any]]:
    branch_dir = work_dir / "basis_branch_worker"
    branch_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = work_dir / "checkpoints" / "reduced_converged.npy"
    _atomic_npy(reduced_path, reduced_x)
    output = branch_dir / "result.json"
    marker = branch_dir / "complete.marker"
    stdout = branch_dir / "stdout.log"
    stderr = branch_dir / "stderr.log"
    for stale in branch_dir.glob("result.*"):
        stale.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    stdout.unlink(missing_ok=True)
    stderr.unlink(missing_ok=True)

    timeout = min(float(policy.basis_branch_worker_timeout_seconds), float(remaining_seconds))
    if timeout <= 0.0:
        raise TimeoutError("no route wall budget remains for selective-basis branch")
    script = Path(__file__).resolve().parents[1] / "tools" / "active_basis_branch_worker.py"
    command = [
        sys.executable, str(script), "--request", str(work_dir / "request.json"),
        "--reduced", str(reduced_path), "--output", str(output),
        "--completion-marker", str(marker),
    ]
    _append_event(event_path, "basis_worker_start", timeout_seconds=timeout, command=command)
    outcome = run_serial_worker(
        command, cwd=Path(__file__).resolve().parents[1],
        timeout_seconds=timeout, completion_marker=marker,
        stdout_path=stdout, stderr_path=stderr,
        env=force_single_thread_environment(os.environ.copy()),
        teardown_grace_seconds=1.0,
    )
    reduced_time = _scalar_time(reduced, reduced_x, request)
    final_problem = reduced
    final_x = reduced_x.copy()
    final_kind = "reduced"
    payload: dict[str, Any] = {
        "worker_status": outcome.status,
        "worker_returncode": outcome.returncode,
        "worker_wall_seconds": outcome.wall_seconds,
        "completion_marker_seen": outcome.completion_marker_seen,
        "forced_teardown_after_completion": outcome.forced_teardown,
        "branch_converged": False,
        "fallback_to_reduced_incumbent": outcome.status != "complete",
        "stdout": str(stdout), "stderr": str(stderr), "output": str(output),
    }

    def candidate(kind: str, parameter_path: str | Path, cells_path: str | Path | None):
        x = np.asarray(np.load(parameter_path), dtype=float)
        problem = _problem_from_kind(kind, x, cells_path, reduced, full)
        cert = certify_parameters_on_problem(
            problem, x, tolerance=policy.feasibility_tolerance,
            maximum_abs_sigma=policy.curvature_slope_limit,
        )
        if not bool(cert["certified"]):
            raise RuntimeError("basis worker candidate failed parent independent certification")
        return problem, x, cert, _scalar_time(problem, x, request)

    if outcome.status != "complete":
        incumbent_meta = output.with_suffix(".incumbent.json")
        if incumbent_meta.exists():
            try:
                meta = json.loads(incumbent_meta.read_text(encoding="utf-8"))
                p, x, cert, t = candidate(meta["kind"], meta["parameters"], meta.get("segment_cells"))
                payload["recovered_incumbent"] = meta
                payload["recovered_incumbent_time"] = t
                payload["recovered_incumbent_certification"] = cert
                if t <= reduced_time + 1.0e-10:
                    final_problem, final_x, final_kind = p, x.copy(), str(meta["kind"])
                    payload["parent_promoted_recovered_incumbent"] = True
                    payload["fallback_to_reduced_incumbent"] = False
            except Exception as exc:  # recovery must never destroy fallback
                payload["incumbent_recovery_error"] = f"{type(exc).__name__}: {exc}"
        return final_problem, final_x, final_kind, payload

    data = json.loads(output.read_text(encoding="utf-8"))
    if data.get("request_digest") != request["request_digest"] or data.get("status") != "complete":
        raise RuntimeError("basis worker returned stale or incomplete result")
    payload.update(data)
    payload["branch_converged"] = bool(data.get("branch_converged", False))
    p, x, cert, t = candidate(data["final_kind"], data["final_parameters"], data.get("final_segment_cells"))
    payload["parent_candidate_time"] = t
    payload["parent_reduced_time"] = reduced_time
    payload["parent_certification"] = cert
    if t <= reduced_time + 1.0e-10:
        final_problem, final_x, final_kind = p, x.copy(), str(data["final_kind"])
        payload["parent_promoted"] = True
        payload["fallback_to_reduced_incumbent"] = False
    else:
        payload["parent_promoted"] = False
        payload["parent_reject_reason"] = "candidate slower than certified reduced incumbent"
    return final_problem, final_x, final_kind, payload


def run_active_basis_pipeline_in_process(request: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Numerical route worker body.  Caller must establish single-thread env first."""
    started = time.perf_counter()
    profile = get_physics_profile(str(request.get("physics_profile", "legacy_grid_v1")))
    current_signature = physics_model_signature(profile)
    if request.get("physics_model_signature") != current_signature:
        raise RuntimeError("active-basis request physics-model signature does not match worker source")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = work_dir / "checkpoints"
    events = work_dir / "events.jsonl"
    events.unlink(missing_ok=True)
    policy = _policy_from_dict(request["homotopy_policy"])
    cells, reduced, full = _build_pair(request)

    # Durable basis-only resume.  A selective branch may finish with an
    # independently certified incumbent but without closure (for example after
    # a conditioning-floor continuation issue).  Re-running that route must not
    # throw away curvature + reduced transaction exhaustion.  If the same
    # request has a durable reduced-converged checkpoint, retry only the
    # selective branch and then recertify the final result.
    prior_result_path = work_dir / "result.json"
    reduced_checkpoint = work_dir / "checkpoints" / "reduced_converged.npy"
    reduced_meta_path = work_dir / "reduced_converged.json"
    if prior_result_path.exists() and reduced_checkpoint.exists() and reduced_meta_path.exists():
        try:
            prior = json.loads(prior_result_path.read_text(encoding="utf-8"))
            reduced_meta = json.loads(reduced_meta_path.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
            reduced_meta = {}
        if (
            prior.get("request_digest") == request.get("request_digest")
            and reduced_meta.get("request_digest") == request.get("request_digest")
            and bool(prior.get("reduced_converged", False))
            and not bool(prior.get("converged", False))
        ):
            x_reduced = np.asarray(np.load(reduced_checkpoint), dtype=float)
            reduced_cert = certify_parameters_on_problem(
                reduced, x_reduced, tolerance=policy.feasibility_tolerance,
                maximum_abs_sigma=policy.curvature_slope_limit,
            )
            if not bool(reduced_cert["certified"]):
                raise RuntimeError("durable reduced checkpoint failed resume certification")
            if events.exists():
                _append_event(events, "route_resume_basis", reduced_time=float(reduced_meta["time"]))
            else:
                _append_event(events, "route_resume_basis", reduced_time=float(reduced_meta["time"]))
            final_problem, final_x, basis_kind, branch = _run_basis_branch(
                request, reduced, full, x_reduced, work_dir, policy,
                float(policy.maximum_route_wall_seconds), events,
            )
            final_cert = certify_parameters_on_problem(
                final_problem, final_x, tolerance=policy.feasibility_tolerance,
                maximum_abs_sigma=policy.curvature_slope_limit,
            )
            if not bool(final_cert["certified"]):
                raise RuntimeError("resumed active-basis trajectory failed independent certification")
            final_time = _scalar_time(final_problem, final_x, request)
            final_raw = knot_parameters_to_raw(final_x, initial_k=final_problem.initial_state.k)
            physics_certification = time_model_dispatch.certify_dd_time_profile(
                final_raw, init_w=float(request["init_w"]),
                terminal_w_max=request.get("terminal_w_max"),
                initial_k=final_problem.initial_state.k, expected_time=final_time,
                n_scan=int(request["n_scan"]), envelope_scan=int(request["envelope_scan"]),
                domain_scan=int(request["domain_scan"]), profile=profile,
                compare_native=(profile.time_model == "dd_yaw_v1"),
            )
            if not bool(physics_certification.get("certified", False)):
                raise RuntimeError("resumed active-basis trajectory failed independent physics certification")
            converged = bool(branch.get("branch_converged", False))
            final_parameters = work_dir / "final.npy"
            _atomic_npy(final_parameters, final_x)
            segment_cells_path = None
            if basis_kind == "hybrid":
                segment_cells_path = work_dir / "final.segment_cells.npy"
                _atomic_npy(segment_cells_path, np.asarray(final_problem.corridor.segment_cells, dtype=np.int64))
            payload = dict(prior)
            payload.update({
                "status": "complete" if converged else "incomplete",
                "converged": converged,
                "basis_branch_converged": converged,
                "basis_kind": basis_kind,
                "reduced_time": float(reduced_meta["time"]),
                "final_time": final_time,
                "final_curvature": _curvature(final_problem, final_x),
                "final_parameters": str(final_parameters),
                "final_segment_cells": None if segment_cells_path is None else str(segment_cells_path),
                "final_parameter_sha256": parameter_sha256(final_x),
                "final_segments": final_problem.corridor.n_segments,
                "final_metrics": geometry_metrics(final_problem, final_x),
                "final_certification": final_cert,
                "final_physics_certification": physics_certification,
                "basis_branch": branch,
                "resume_basis_only": True,
                "resume_basis_wall_seconds": time.perf_counter() - started,
                "thread_environment": thread_environment_snapshot(),
            })
            _atomic_json(prior_result_path, payload)
            _append_event(events, "route_finish", **payload)
            return payload

    x = np.asarray(reduced.initial_parameters, dtype=float).copy()
    cert = certify_parameters_on_problem(
        reduced, x, tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(cert["certified"]):
        raise RuntimeError("analytic reduced initializer is not independently certified")

    initial_time = _scalar_time(reduced, x, request)
    initial_curvature = _curvature(reduced, x)
    _atomic_npy(checkpoints / "initial_reduced.npy", x)
    _append_event(events, "route_start", cells=cells, reduced_segments=reduced.corridor.n_segments,
                  full_segments=full.corridor.n_segments, initial_time=initial_time,
                  initial_curvature=initial_curvature, initial_certification=cert,
                  thread_environment=thread_environment_snapshot(), policy=asdict(policy))

    curvature_rows: list[dict[str, Any]] = []
    reduced_rows: list[dict[str, Any]] = []
    active_pool: ConstraintPool | None = None

    def remaining() -> float:
        return float(policy.maximum_route_wall_seconds - (time.perf_counter() - started))

    def ensure_budget() -> None:
        if remaining() <= 0.0:
            raise TimeoutError(f"route wall ceiling {policy.maximum_route_wall_seconds:.1f}s reached")

    q010, q005, q0025 = policy.curvature_guard_sequence
    last_result = None
    for name, q in (("curvature_q010", q010), ("curvature_q005", q005)):
        ensure_budget()
        guard = make_guard(cells, reduced, full, q)
        result = run_curvature(
            reduced, x, guard=guard, accepted_steps=policy.curvature_epoch_steps,
            batch_schedule=policy.curvature_batch_schedule,
            trust_schedule=policy.curvature_trust_schedule,
            max_filter_violation=policy.curvature_filter_violation,
            pool=active_pool, wall=min(policy.curvature_epoch_wall_seconds, remaining()),
            maximum_raw_iterations=policy.curvature_maximum_raw_iterations,
            highs_threads=policy.highs_threads,
            init_w=request["init_w"], terminal_w_max=request.get("terminal_w_max"),
            n_scan=request["n_scan"], envelope_scan=request["envelope_scan"], domain_scan=request["domain_scan"],
        )
        x = result.parameters.copy(); active_pool = result.pool.copy(); last_result = result
        row = stage_summary(result); row["guard_fraction"] = q
        curvature_rows.append(row)
        _atomic_npy(checkpoints / f"{name}.npy", x)
        _append_event(events, "stage_complete", stage=name, result=row)

    assert last_result is not None
    switch = curvature_switch_decision(last_result, policy)
    _append_event(events, "curvature_switch_decision", decision=asdict(switch))
    if switch.continue_curvature:
        ensure_budget()
        guard = make_guard(cells, reduced, full, q0025)
        result = run_curvature(
            reduced, x, guard=guard, accepted_steps=policy.curvature_epoch_steps,
            batch_schedule=policy.curvature_batch_schedule,
            trust_schedule=policy.curvature_trust_schedule,
            max_filter_violation=policy.curvature_filter_violation,
            pool=active_pool, wall=min(policy.curvature_epoch_wall_seconds, remaining()),
            maximum_raw_iterations=policy.curvature_maximum_raw_iterations,
            highs_threads=policy.highs_threads,
            init_w=request["init_w"], terminal_w_max=request.get("terminal_w_max"),
            n_scan=request["n_scan"], envelope_scan=request["envelope_scan"], domain_scan=request["domain_scan"],
        )
        x = result.parameters.copy(); active_pool = result.pool.copy()
        row = stage_summary(result); row["guard_fraction"] = q0025
        curvature_rows.append(row)
        _atomic_npy(checkpoints / "curvature_q0025.npy", x)
        _append_event(events, "stage_complete", stage="curvature_q0025", result=row)

    active_pool = None
    guard = make_guard(cells, reduced, full, policy.final_reduced_guard)
    reduced_converged = False
    for epoch in range(policy.maximum_reduced_time_epochs):
        ensure_budget()
        result = run_time(
            reduced, x, guard=guard, accepted_steps=policy.reduced_time_epoch_steps,
            batch_schedule=policy.reduced_time_batch_schedule,
            trust_schedule=policy.reduced_time_trust_schedule,
            max_filter_violation=policy.reduced_time_filter_violation,
            pool=active_pool, wall=min(policy.reduced_time_epoch_wall_seconds, remaining()),
            maximum_raw_iterations=policy.reduced_time_maximum_raw_iterations,
            pool_retention=policy.reduced_time_pool_retention,
            highs_threads=policy.highs_threads,
            init_w=request["init_w"], terminal_w_max=request.get("terminal_w_max"),
            n_scan=request["n_scan"], envelope_scan=request["envelope_scan"], domain_scan=request["domain_scan"],
        )
        x = result.parameters.copy(); active_pool = result.pool.copy()
        row = stage_summary(result); row.update(epoch=epoch, guard_fraction=policy.final_reduced_guard)
        reduced_rows.append(row)
        _atomic_npy(checkpoints / f"reduced_time_{epoch:03d}.npy", x)
        _append_event(events, "reduced_epoch_complete", epoch=epoch, result=row)
        if result.stop_reason == "no improving certified trial":
            reduced_converged = True
            break
        if result.stop_reason != "maximum accepted-step budget reached":
            raise RuntimeError(f"reduced-time stage stopped without convergence: {result.stop_reason}")
    if not reduced_converged:
        raise RuntimeError("reduced-time epoch ceiling reached without transaction exhaustion")

    reduced_time = _scalar_time(reduced, x, request)
    _atomic_npy(checkpoints / "reduced_converged.npy", x)
    _atomic_json(work_dir / "reduced_converged.json", {
        "request_digest": request["request_digest"], "time": reduced_time,
        "parameters": str(checkpoints / "reduced_converged.npy"),
        "parameter_sha256": parameter_sha256(x),
        "certification": certify_parameters_on_problem(
            reduced, x, tolerance=policy.feasibility_tolerance,
            maximum_abs_sigma=policy.curvature_slope_limit,
        ),
    })

    ensure_budget()
    final_problem, final_x, basis_kind, branch = _run_basis_branch(
        request, reduced, full, x, work_dir, policy, remaining(), events
    )
    final_cert = certify_parameters_on_problem(
        final_problem, final_x, tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(final_cert["certified"]):
        raise RuntimeError("final active-basis trajectory failed independent certification")
    final_time = _scalar_time(final_problem, final_x, request)
    final_raw = knot_parameters_to_raw(final_x, initial_k=final_problem.initial_state.k)
    physics_certification = time_model_dispatch.certify_dd_time_profile(
        final_raw,
        init_w=float(request["init_w"]),
        terminal_w_max=request.get("terminal_w_max"),
        initial_k=final_problem.initial_state.k,
        expected_time=final_time,
        n_scan=int(request["n_scan"]),
        envelope_scan=int(request["envelope_scan"]),
        domain_scan=int(request["domain_scan"]),
        profile=profile,
        compare_native=(profile.time_model == "dd_yaw_v1"),
    )
    if not bool(physics_certification.get("certified", False)):
        raise RuntimeError("final active-basis trajectory failed independent physics certification")
    converged = bool(reduced_converged and branch.get("branch_converged", False))

    final_parameters = work_dir / "final.npy"
    _atomic_npy(final_parameters, final_x)
    segment_cells_path = None
    if basis_kind == "hybrid":
        segment_cells_path = work_dir / "final.segment_cells.npy"
        _atomic_npy(segment_cells_path, np.asarray(final_problem.corridor.segment_cells, dtype=np.int64))
    payload = {
        "schema_version": 1,
        "status": "complete" if converged else "incomplete",
        "request_digest": request["request_digest"],
        "converged": converged,
        "reduced_converged": reduced_converged,
        "basis_branch_converged": bool(branch.get("branch_converged", False)),
        "basis_kind": basis_kind,
        "initial_time": initial_time,
        "reduced_time": reduced_time,
        "final_time": final_time,
        "initial_curvature": initial_curvature,
        "final_curvature": _curvature(final_problem, final_x),
        "final_parameters": str(final_parameters),
        "final_segment_cells": None if segment_cells_path is None else str(segment_cells_path),
        "final_parameter_sha256": parameter_sha256(final_x),
        "final_segments": final_problem.corridor.n_segments,
        "final_metrics": geometry_metrics(final_problem, final_x),
        "final_certification": final_cert,
        "physics_model_signature": request.get("physics_model_signature"),
        "physics_model_identity": request.get("physics_model_identity"),
        "final_physics_certification": physics_certification,
        "curvature_switch": asdict(switch),
        "curvature_stages": curvature_rows,
        "reduced_time_epochs": reduced_rows,
        "basis_branch": branch,
        "total_wall_seconds": time.perf_counter() - started,
        "thread_environment": thread_environment_snapshot(),
    }
    _atomic_json(work_dir / "result.json", payload)
    _append_event(events, "route_finish", **payload)
    return payload


def _canonical_request(
    cells: Sequence[Cell], *, open_room_spans: Sequence[OpenRoomSpan], body_length: float,
    body_height: float, clearance: float, refinement_factor: int, init_w: float,
    terminal_w_max: float | None, n_scan: int, envelope_scan: int, domain_scan: int,
    policy: HomotopyPolicy,
) -> dict[str, Any]:
    # Red Comet and historical qualification both use refinement=1.  Refined
    # active-basis support has not been separately qualified.
    if refinement_factor != 1:
        raise ValueError("active_basis_v11 currently requires geometry_refinement=1")
    bp = replace(
        policy.basis_branch_policy,
        init_w=float(init_w), terminal_w_max=terminal_w_max,
        n_scan=int(n_scan), envelope_scan=int(envelope_scan), domain_scan=int(domain_scan),
    )
    policy = replace(policy, basis_branch_policy=bp)
    request: dict[str, Any] = {
        "schema_version": 1,
        "cells": [list(map(int, c)) for c in cells],
        "open_room_spans": [_span_dict(span) for span in open_room_spans],
        "body_length": float(body_length), "body_height": float(body_height),
        "clearance": float(clearance), "refinement_factor": int(refinement_factor),
        "init_w": float(init_w), "terminal_w_max": terminal_w_max,
        "n_scan": int(n_scan), "envelope_scan": int(envelope_scan), "domain_scan": int(domain_scan),
        "homotopy_policy": asdict(policy),
        "physics_profile": os.environ.get("AME_PHYSICS_PROFILE", "legacy_grid_v1"),
        "reverse_backend": os.environ.get("AME_REVERSE_BACKEND", "native"),
    }
    profile = get_physics_profile(request["physics_profile"])
    request["physics_model_identity"] = physics_model_identity(profile)
    request["physics_model_signature"] = physics_model_signature(profile)
    blob = json.dumps(_jsonable(request), sort_keys=True, separators=(",", ":")).encode()
    request["request_digest"] = hashlib.sha256(blob).hexdigest()
    return request


def _production_policy_for_current_profile(policy: HomotopyPolicy) -> HomotopyPolicy:
    """Raise only watchdog/safety ceilings for the slower DD/yaw objective.

    Accepted-step schedules, trust policy, support thresholds and convergence
    semantics are unchanged.  These values are safety ceilings, not stopping
    criteria, so increasing them does not alter a converged numerical result.
    """
    profile = get_physics_profile(os.environ.get("AME_PHYSICS_PROFILE", "legacy_grid_v1"))
    if profile.time_model != "dd_yaw_v1":
        return policy
    bp = replace(
        policy.basis_branch_policy,
        stage_wall_seconds_initial=max(policy.basis_branch_policy.stage_wall_seconds_initial, 1800.0),
        stage_wall_seconds_relaxed=max(policy.basis_branch_policy.stage_wall_seconds_relaxed, 900.0),
    )
    return replace(
        policy,
        basis_branch_policy=bp,
        basis_branch_worker_timeout_seconds=max(policy.basis_branch_worker_timeout_seconds, 7200.0),
        curvature_epoch_wall_seconds=max(policy.curvature_epoch_wall_seconds, 900.0),
        reduced_time_epoch_wall_seconds=max(policy.reduced_time_epoch_wall_seconds, 3600.0),
        maximum_route_wall_seconds=max(policy.maximum_route_wall_seconds, 21600.0),
    )


def _canonical_boundary_scalar(value: float) -> float:
    """Collapse sub-ulp CLI/direct representations to one request identity."""
    return round(float(value), 15)


def optimize_active_basis_route(
    cells: Sequence[Cell], *, open_room_spans: Sequence[OpenRoomSpan] = (),
    body_length: float, body_height: float, init_w: float = 0.8,
    terminal_w_max: float | None = None, clearance: float = 0.0,
    refinement_factor: int = 1, n_scan: int = 96, envelope_scan: int = 48,
    domain_scan: int = 96, policy: HomotopyPolicy = HomotopyPolicy(),
    execution: ActiveBasisExecutionSettings = ActiveBasisExecutionSettings(),
) -> ActiveBasisRouteResult:
    """Optimize one fixed cell topology with the qualified active-basis V11 path."""
    # Canonicalize scalar boundary conditions before they enter the content-addressed
    # request.  The CLI specifies terminal speed while the fixed-route gate stores
    # terminal ``w=v^2`` directly; mathematically identical values such as 0.8 and
    # ``sqrt(0.8)**2 == 0.7999999999999999`` must share one cache key.
    init_w = _canonical_boundary_scalar(init_w)
    if terminal_w_max is not None:
        terminal_w_max = _canonical_boundary_scalar(terminal_w_max)
    policy = _production_policy_for_current_profile(policy)
    request = _canonical_request(
        tuple(cells), open_room_spans=tuple(open_room_spans), body_length=body_length,
        body_height=body_height, clearance=clearance, refinement_factor=refinement_factor,
        init_w=init_w, terminal_w_max=terminal_w_max, n_scan=n_scan,
        envelope_scan=envelope_scan, domain_scan=domain_scan, policy=policy,
    )
    digest = request["request_digest"]
    if execution.work_root is None:
        work_dir = Path(tempfile.mkdtemp(prefix=f"ame-active-basis-{digest[:12]}-"))
    else:
        root = Path(execution.work_root)
        root.mkdir(parents=True, exist_ok=True)
        work_dir = root / f"route_{digest[:20]}"
        work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    _atomic_json(request_path, request)
    result_path = work_dir / "result.json"
    marker = work_dir / "complete.marker"

    def load_result(worker_status: str) -> ActiveBasisRouteResult:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if data.get("request_digest") != digest:
            raise RuntimeError("stale active-basis result digest")
        if data.get("physics_model_signature") != request.get("physics_model_signature"):
            raise RuntimeError("stale active-basis result physics-model signature")
        cells2, reduced, full = _build_pair(request)
        x = np.asarray(np.load(data["final_parameters"]), dtype=float)
        problem = _problem_from_kind(data["basis_kind"], x, data.get("final_segment_cells"), reduced, full)
        cert = certify_parameters_on_problem(
            problem, x, tolerance=policy.feasibility_tolerance,
            maximum_abs_sigma=policy.curvature_slope_limit,
        )
        if not bool(cert["certified"]):
            raise RuntimeError("active-basis route result failed planner-side independent certification")
        t = _scalar_time(problem, x, request)
        if not math.isclose(t, float(data["final_time"]), rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError("active-basis worker/parent scalar-time mismatch")
        raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
        parent_physics = time_model_dispatch.certify_dd_time_profile(
            raw,
            init_w=float(request["init_w"]),
            terminal_w_max=request.get("terminal_w_max"),
            initial_k=problem.initial_state.k,
            expected_time=t,
            n_scan=int(request["n_scan"]),
            envelope_scan=int(request["envelope_scan"]),
            domain_scan=int(request["domain_scan"]),
            profile=get_physics_profile(str(request["physics_profile"])),
            compare_native=(get_physics_profile(str(request["physics_profile"])).time_model == "dd_yaw_v1"),
            include_trace=(get_physics_profile(str(request["physics_profile"])).time_model == "dd_yaw_v1"),
            trace_samples=2001,
        )
        if not bool(parent_physics.get("certified", False)):
            raise RuntimeError("active-basis result failed planner-side independent physics certification")
        data["parent_physics_certification"] = parent_physics
        _atomic_json(result_path, data)
        converged = bool(data.get("converged", False))
        if execution.require_convergence and not converged:
            raise ActiveBasisNonConvergenceError("active-basis route returned a certified but non-converged incumbent")
        return ActiveBasisRouteResult(
            tuple(cells2), problem, x.copy(), t, converged, str(data["basis_kind"]),
            str(work_dir), digest, worker_status, data,
        )

    if execution.reuse_complete_result and marker.exists() and result_path.exists():
        try:
            return load_result("reused_complete")
        except Exception:
            marker.unlink(missing_ok=True)

    script = Path(__file__).resolve().parents[1] / "tools" / "active_basis_route_worker.py"
    stdout = work_dir / "stdout.log"; stderr = work_dir / "stderr.log"
    timeout = float(policy.maximum_route_wall_seconds + execution.outer_timeout_grace_seconds)
    maximum_attempts = 1 + int(execution.incomplete_basis_retries)
    for attempt in range(maximum_attempts):
        if attempt:
            marker.unlink(missing_ok=True)
            _append_event(
                work_dir / "events.jsonl",
                "planner_retry_incomplete_basis",
                retry_index=attempt,
                maximum_retries=execution.incomplete_basis_retries,
            )
        outcome = run_serial_worker(
            [sys.executable, str(script), "--request", str(request_path), "--work-dir", str(work_dir),
             "--completion-marker", str(marker)],
            cwd=Path(__file__).resolve().parents[1], timeout_seconds=timeout,
            env=force_single_thread_environment(os.environ.copy()), completion_marker=marker,
            stdout_path=stdout, stderr_path=stderr, teardown_grace_seconds=1.0,
        )
        if outcome.status != "complete" or not result_path.exists():
            reduced_meta = work_dir / "reduced_converged.json"
            detail = ""
            if reduced_meta.exists():
                detail = " A durably certified reduced-converged fallback exists in the route work directory."
            raise TimeoutError(
                f"active-basis route worker {outcome.status} after {outcome.wall_seconds:.3f}s; "
                f"see {stderr}.{detail}"
            )
        try:
            return load_result(outcome.status if attempt == 0 else "resumed_incomplete_basis")
        except ActiveBasisNonConvergenceError:
            if attempt + 1 >= maximum_attempts:
                raise
            # The next worker invocation sees the durable reduced-converged
            # checkpoint and takes the basis-only resume path above.
            continue
    raise AssertionError("unreachable active-basis retry loop")


__all__ = [
    "ActiveBasisExecutionSettings", "ActiveBasisNonConvergenceError",
    "ActiveBasisRouteResult", "optimize_active_basis_route",
    "run_active_basis_pipeline_in_process",
]
