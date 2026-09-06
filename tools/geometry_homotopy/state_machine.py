"""Historical-five production-candidate optimizer state machine.

The state machine codifies the policy selected by the geometry-homotopy research
campaign while keeping it isolated from the production planner.  Every state
transition is certification-gated and append-only logged.  A route is called
converged only when the reduced time stage exhausts its transaction/trust
schedule and the serial selective-basis refinement also establishes its own
transaction exhaustion / structural closure.  A timeout or failed richer-basis
branch may safely fall back to the reduced incumbent, but is reported as
*incomplete*, never as converged.

This module intentionally does *not* know about Red Comet.  The five Wilson 8x8
routes (seeds 0..4) remain the architecture-development corpus.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import json
import math
import os
import sys
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from .serial_runtime import (
    THREAD_ENVIRONMENT_VARIABLES,
    force_single_thread_environment,
    run_serial_worker,
    thread_environment_snapshot,
)

# This research module itself is part of the authoritative single-thread
# baseline.  Set the common native/BLAS pools before importing NumPy/SciPy.
force_single_thread_environment()

import numpy as np

from benchmarks.common.certification import certify_parameters_on_problem
from optimization import CorridorModel, ConstraintFamily, ConstraintPool, knot_parameters_to_raw, scalar_reverse_solver

from .active_set import CurvatureObjective
from .basis_branch import BasisBranchPolicy
from .research_common import (
    build_historical_pair,
    geometry_metrics,
    lift_summary,
    make_guard,
    parameter_sha256,
    run_curvature,
    run_time,
    stage_summary,
)


class ResearchStage(str, Enum):
    INITIAL = "initial"
    CURVATURE_Q010 = "curvature_q010"
    CURVATURE_Q005 = "curvature_q005"
    CURVATURE_Q0025 = "curvature_q0025"
    REDUCED_TIME = "reduced_time"
    BASIS_BRANCH = "basis_branch"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HomotopyPolicy:
    """Frozen candidate policy from the preceding A/B research pass."""

    curvature_epoch_steps: int = 16
    curvature_batch_schedule: tuple[int, ...] = (8, 4, 2, 1)
    curvature_trust_schedule: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025)
    curvature_filter_violation: float = 1.0e-4
    curvature_maximum_raw_iterations: int = 16
    curvature_guard_sequence: tuple[float, ...] = (0.01, 0.005, 0.0025)
    curvature_switch_min_accepted: int = 4
    curvature_switch_min_relative_time_gain_per_step: float = 2.0e-4

    reduced_time_epoch_steps: int = 16
    reduced_time_batch_schedule: tuple[int, ...] = (8, 4, 2, 1)
    reduced_time_trust_schedule: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005)
    reduced_time_filter_violation: float = 1.0e-5
    reduced_time_maximum_raw_iterations: int = 16
    reduced_time_pool_retention: str = "reseed_after_promotion"
    final_reduced_guard: float = 0.0025

    # Selective-basis refinement replaces unconditional full-basis promotion.
    # The worker is launched serially and supervised only for killability; no
    # branches or routes execute concurrently.
    basis_branch_policy: BasisBranchPolicy = field(default_factory=BasisBranchPolicy)
    basis_branch_worker_timeout_seconds: float = 600.0

    # These are safety ceilings, not convergence criteria.  Hitting one makes a
    # route incomplete rather than silently labelling it converged.
    maximum_reduced_time_epochs: int = 256
    curvature_epoch_wall_seconds: float = 180.0
    reduced_time_epoch_wall_seconds: float = 240.0
    maximum_route_wall_seconds: float = 1800.0

    feasibility_tolerance: float = 2.0e-7
    curvature_slope_limit: float = 50.0
    highs_threads: int = 1

    def __post_init__(self) -> None:
        if self.curvature_guard_sequence != (0.01, 0.005, 0.0025):
            raise ValueError("candidate state machine currently expects q=.01,.005,.0025")
        if self.final_reduced_guard != self.curvature_guard_sequence[-1]:
            raise ValueError("final reduced guard must equal final guard in continuation")
        for name in ("curvature_epoch_steps", "reduced_time_epoch_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.curvature_switch_min_accepted <= 0:
            raise ValueError("curvature switch minimum accepted steps must be positive")
        if not (0.0 < self.curvature_switch_min_relative_time_gain_per_step < 1.0):
            raise ValueError("curvature switch threshold must lie in (0,1)")
        if self.highs_threads != 1:
            raise ValueError("research baseline is intentionally single-threaded: highs_threads must be 1")
        if self.basis_branch_policy.highs_threads != 1:
            raise ValueError("basis branch must remain single-threaded in the current baseline")


@dataclass(frozen=True, slots=True)
class SwitchDecision:
    accepted_steps: int
    scalar_time_before: float
    scalar_time_after: float
    absolute_time_gain: float
    relative_time_gain_per_step: float
    continue_curvature: bool


@dataclass(slots=True)
class RouteRunResult:
    seed: int
    cells: tuple[tuple[int, int], ...]
    status: str
    stage: str
    converged: bool
    initial_time: float
    final_time: float | None = None
    initial_curvature: float | None = None
    final_curvature: float | None = None
    reduced_converged: bool = False
    full_converged: bool = False  # retained for schema compatibility; active-basis branch supersedes it
    basis_branch_complete: bool = False
    basis_branch_converged: bool = False
    basis_branch_attempts: int = 0
    final_basis_kind: str | None = None
    final_segments: int | None = None
    curvature_switch: dict[str, Any] | None = None
    curvature_stages: list[dict[str, Any]] = field(default_factory=list)
    reduced_time_epochs: list[dict[str, Any]] = field(default_factory=list)
    lift: dict[str, Any] | None = None  # legacy field kept readable in older campaign JSON
    full_time_epochs: list[dict[str, Any]] = field(default_factory=list)
    basis_branch: dict[str, Any] | None = None
    total_wall_seconds: float = 0.0
    final_parameter_sha256: str | None = None
    final_certification: dict[str, Any] | None = None
    error: str | None = None


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, "monotonic_seconds": time.perf_counter(), **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save_checkpoint(directory: Path, name: str, x: Sequence[float], pool: ConstraintPool | None) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    x_path = directory / f"{name}.npy"
    np.save(x_path, np.asarray(x, dtype=float))
    pool_path: Path | None = None
    pool_size = 0
    if pool is not None:
        rows = np.asarray(
            [[e.segment, e.wall, e.corner, e.tau] for e in pool.entries()],
            dtype=float,
        )
        if rows.size == 0:
            rows = np.empty((0, 4), dtype=float)
        pool_path = directory / f"{name}.pool.npy"
        np.save(pool_path, rows)
        pool_size = int(pool.size)
    return {
        "parameters": str(x_path),
        "pool": None if pool_path is None else str(pool_path),
        "pool_size": pool_size,
        "parameter_sha256": parameter_sha256(x),
    }


def restore_pool(problem, path: str | Path) -> ConstraintPool:
    """Restore a serialized pool.  Useful for interrupted research runs."""
    rows = np.load(Path(path))
    pool = ConstraintPool(problem.corridor)
    for segment, wall, corner, tau in np.asarray(rows, dtype=float):
        family = ConstraintFamily(int(segment), int(wall), int(corner))
        pool.taus.setdefault(family, []).append(float(tau))
    for values in pool.taus.values():
        values.sort()
    return pool


def scalar_time(problem, x: Sequence[float]) -> float:
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    return float(
        scalar_reverse_solver.evaluate_time_scalar(
            raw,
            init_w=0.8,
            terminal_w_max=None,
            initial_k=problem.initial_state.k,
            n_scan=96,
            envelope_scan=48,
            domain_scan=96,
        )
    )


def curvature_value(problem, x: Sequence[float]) -> float:
    return float(CurvatureObjective(problem.initial_state.k)(x)[0])


def curvature_switch_decision(result, policy: HomotopyPolicy) -> SwitchDecision:
    before = float(result.checkpoints[0].scalar_time)
    after = float(result.scalar_time)
    accepted = int(result.accepted_steps)
    gain = max(0.0, before - after)
    relative_per_step = 0.0 if accepted <= 0 or before <= 0.0 else gain / (accepted * before)
    keep = bool(
        accepted >= policy.curvature_switch_min_accepted
        and relative_per_step >= policy.curvature_switch_min_relative_time_gain_per_step
    )
    return SwitchDecision(
        accepted_steps=accepted,
        scalar_time_before=before,
        scalar_time_after=after,
        absolute_time_gain=gain,
        relative_time_gain_per_step=relative_per_step,
        continue_curvature=keep,
    )


def _stage_payload(result, *, epoch: int | None = None, guard: float | None = None) -> dict[str, Any]:
    payload = stage_summary(result)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if guard is not None:
        payload["guard_fraction"] = float(guard)
    if result.checkpoints:
        payload["scalar_time_before"] = float(result.checkpoints[0].scalar_time)
        payload["scalar_time_gain"] = float(result.checkpoints[0].scalar_time - result.scalar_time)
    return payload



def _run_supervised_basis_branch(
    seed: int,
    cells,
    reduced,
    full,
    x: Sequence[float],
    route_dir: Path,
    checkpoint_dir: Path,
    logger: JsonlLogger,
    policy: HomotopyPolicy,
    *,
    timeout_seconds: float | None = None,
):
    """Run one basis branch synchronously behind a killable process boundary.

    Process isolation is *not* parallel scheduling.  The parent blocks here
    until this single worker either writes a durable completion marker, exits,
    or reaches its deadline.  The certified reduced solution remains the parent
    incumbent throughout.
    """
    reduced_ckpt = _save_checkpoint(checkpoint_dir, "reduced_converged", x, None)
    worker_dir = route_dir / "basis_branch_worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_output = worker_dir / "result.json"
    worker_marker = worker_dir / "complete.marker"
    worker_stdout = worker_dir / "stdout.log"
    worker_stderr = worker_dir / "stderr.log"
    worker_incumbent_meta = worker_output.with_suffix(".incumbent.json")
    worker_incumbent_x = worker_output.with_suffix(".incumbent.npy")
    worker_incumbent_segment_cells = worker_output.with_suffix(".incumbent.segment_cells.npy")
    worker_progress = worker_output.with_suffix(".events.jsonl")
    # Never let stale files from an interrupted prior branch look current.
    for stale in (
        worker_output, worker_marker, worker_stdout, worker_stderr,
        worker_incumbent_meta, worker_incumbent_x, worker_incumbent_segment_cells,
        worker_progress,
    ):
        stale.unlink(missing_ok=True)
    branch_policy_path = worker_dir / "policy.json"
    _atomic_json(branch_policy_path, asdict(policy.basis_branch_policy))
    worker_script = Path(__file__).with_name("basis_activation_worker.py")
    command = [
        sys.executable, str(worker_script),
        "--seed", str(int(seed)),
        "--reduced", str(reduced_ckpt["parameters"]),
        "--output", str(worker_output),
        "--policy-json", str(branch_policy_path),
        "--completion-marker", str(worker_marker),
    ]
    worker_env = force_single_thread_environment(os.environ.copy())
    effective_timeout = float(policy.basis_branch_worker_timeout_seconds)
    if timeout_seconds is not None:
        effective_timeout = min(effective_timeout, float(timeout_seconds))
    if not math.isfinite(effective_timeout) or effective_timeout <= 0.0:
        raise TimeoutError(f"seed {seed}: no route wall budget remains for basis branch")

    logger.emit(
        "basis_branch_worker_start",
        seed=seed,
        command=command,
        timeout_seconds=effective_timeout,
        execution_mode="serial_single_worker",
        thread_environment=thread_environment_snapshot(worker_env),
        reduced_checkpoint=reduced_ckpt,
    )
    worker_result = run_serial_worker(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=worker_env,
        timeout_seconds=effective_timeout,
        completion_marker=worker_marker,
        stdout_path=worker_stdout,
        stderr_path=worker_stderr,
        teardown_grace_seconds=1.0,
    )

    final_problem = reduced
    final_x = np.asarray(x, dtype=float).copy()
    final_basis_kind = "reduced"
    branch_payload: dict[str, Any] = {
        "worker_status": worker_result.status,
        "execution_mode": "serial_single_worker",
        "worker_timeout_seconds": effective_timeout,
        "worker_wall_seconds": worker_result.wall_seconds,
        "worker_returncode": worker_result.returncode,
        "completion_marker_seen": worker_result.completion_marker_seen,
        "forced_teardown_after_completion": worker_result.forced_teardown,
        "worker_stdout": worker_result.stdout_path,
        "worker_stderr": worker_result.stderr_path,
        "worker_output": str(worker_output),
        "worker_progress_jsonl": str(worker_progress),
        "worker_latest_incumbent": str(worker_incumbent_meta),
        "fallback_to_reduced_incumbent": worker_result.status != "complete",
        "branch_converged": False,
    }

    def load_and_certify_candidate(
        *, kind: str, parameter_path: str | Path, segment_cells_path: str | Path | None,
    ):
        candidate_x = np.asarray(np.load(parameter_path), dtype=float)
        if kind == "reduced":
            candidate_problem = reduced
        elif kind == "hybrid":
            if not segment_cells_path:
                raise RuntimeError(f"seed {seed}: hybrid candidate omitted segment cells")
            segment_cells = tuple(int(v) for v in np.load(segment_cells_path).tolist())
            corridor = CorridorModel(
                full.corridor.cells,
                segment_cells,
                full.corridor.body,
                full.corridor.clearance,
            )
            candidate_problem = replace(
                full, corridor=corridor, initial_parameters=candidate_x.copy()
            )
        else:
            raise RuntimeError(f"seed {seed}: unknown worker candidate kind {kind!r}")
        candidate_cert = certify_parameters_on_problem(
            candidate_problem, candidate_x,
            tolerance=policy.feasibility_tolerance,
            maximum_abs_sigma=policy.curvature_slope_limit,
        )
        if not bool(candidate_cert["certified"]):
            raise RuntimeError(f"seed {seed}: worker candidate failed parent independent certification")
        return candidate_problem, candidate_x, candidate_cert, scalar_time(candidate_problem, candidate_x)

    reduced_time = scalar_time(reduced, x)
    if worker_result.status != "complete":
        logger.emit(
            "basis_branch_worker_fallback",
            seed=seed,
            status=worker_result.status,
            returncode=worker_result.returncode,
            stderr=str(worker_stderr),
        )
        # Recover the latest worker-side *certified* incumbent if one had been
        # durably checkpointed before the timeout.  This improves the feasible
        # upper bound without changing the convergence verdict.
        if worker_incumbent_meta.exists():
            try:
                meta = json.loads(worker_incumbent_meta.read_text(encoding="utf-8"))
                c_problem, c_x, c_cert, c_time = load_and_certify_candidate(
                    kind=str(meta["kind"]),
                    parameter_path=meta["parameters"],
                    segment_cells_path=meta.get("segment_cells"),
                )
                branch_payload["recovered_incumbent"] = meta
                branch_payload["recovered_incumbent_time"] = c_time
                branch_payload["recovered_incumbent_certification"] = c_cert
                if c_time <= reduced_time + 1.0e-10:
                    final_problem, final_x = c_problem, c_x.copy()
                    final_basis_kind = str(meta["kind"] )
                    branch_payload["parent_promoted_recovered_incumbent"] = True
                    branch_payload["fallback_to_reduced_incumbent"] = False
            except Exception as recovery_exc:
                branch_payload["incumbent_recovery_error"] = (
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        return final_problem, final_x, final_basis_kind, branch_payload

    if not worker_output.exists():
        raise RuntimeError(f"seed {seed}: basis worker completed without result.json")
    worker_data = json.loads(worker_output.read_text(encoding="utf-8"))
    if worker_data.get("status") != "complete":
        raise RuntimeError(
            f"seed {seed}: basis worker result is incomplete: {worker_data.get('status')}"
        )
    branch_payload.update(worker_data)
    branch_payload["fallback_to_reduced_incumbent"] = False
    branch_converged = bool(worker_data.get("branch_converged", False))
    branch_payload["branch_converged"] = branch_converged

    # Even an incomplete branch may have found a faster *certified* incumbent.
    # Preserve it as an upper bound, but do not label the route converged unless
    # the worker also established branch transaction exhaustion/closure.
    candidate_kind = str(worker_data["final_kind"])
    candidate_problem, candidate_x, candidate_cert, candidate_time = load_and_certify_candidate(
        kind=candidate_kind,
        parameter_path=worker_data["final_parameters"],
        segment_cells_path=worker_data.get("final_segment_cells"),
    )
    if candidate_time <= reduced_time + 1.0e-10:
        final_problem = candidate_problem
        final_x = candidate_x.copy()
        final_basis_kind = candidate_kind
        branch_payload["parent_promoted"] = True
        branch_payload["parent_candidate_time"] = candidate_time
        branch_payload["parent_reduced_time"] = reduced_time
        branch_payload["parent_certification"] = candidate_cert
    else:
        branch_payload["parent_promoted"] = False
        branch_payload["parent_reject_reason"] = (
            "candidate slower than certified reduced incumbent"
        )
        branch_payload["parent_candidate_time"] = candidate_time
        branch_payload["parent_reduced_time"] = reduced_time

    return final_problem, final_x, final_basis_kind, branch_payload


def _record_from_json(path: Path) -> RouteRunResult:
    return RouteRunResult(**json.loads(path.read_text(encoding="utf-8")))


def resume_route_after_reduced_ceiling(
    seed: int,
    output_dir: Path,
    *,
    policy: HomotopyPolicy = HomotopyPolicy(),
) -> RouteRunResult:
    """Resume a route that stopped only because the reduced epoch safety ceiling fired.

    The last point was independently certified and the active-set policy reseeds
    after promotion, so resuming from its serialized checkpoint does not require
    hidden solver state.
    """
    resume_started = time.perf_counter()
    route_dir = Path(output_dir) / f"seed_{int(seed)}"
    state_path = route_dir / "state.json"
    checkpoint_dir = route_dir / "checkpoints"
    logger = JsonlLogger(route_dir / "events.jsonl")
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    record = _record_from_json(state_path)
    if record.converged:
        return record
    if not record.error or "reduced-time epoch ceiling" not in record.error:
        raise ValueError(f"seed {seed}: state is not a resumable reduced-ceiling stop")

    cells, reduced, full = build_historical_pair(int(seed))
    previous_epochs = len(record.reduced_time_epochs)
    if previous_epochs <= 0:
        raise ValueError("cannot resume without a serialized reduced-time epoch")
    last_index = previous_epochs - 1
    x_path = checkpoint_dir / f"reduced_time_{last_index:03d}.npy"
    pool_path = checkpoint_dir / f"reduced_time_{last_index:03d}.pool.npy"
    x = np.asarray(np.load(x_path), dtype=float)
    cert = certify_parameters_on_problem(
        reduced, x,
        tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(cert["certified"]):
        raise RuntimeError(f"seed {seed}: resume checkpoint is not certified")
    active_pool = restore_pool(reduced, pool_path) if pool_path.exists() else None
    guard = make_guard(cells, reduced, full, policy.final_reduced_guard)

    record.status = "running"
    record.stage = ResearchStage.REDUCED_TIME.value
    record.error = None
    record.reduced_converged = False
    logger.emit(
        "route_resume",
        seed=seed,
        prior_epochs=previous_epochs,
        scalar_time=scalar_time(reduced, x),
        parameter_sha256=parameter_sha256(x),
        new_epoch_ceiling=policy.maximum_reduced_time_epochs,
    )
    _atomic_json(state_path, asdict(record))

    reduced_converged = False
    for epoch in range(previous_epochs, policy.maximum_reduced_time_epochs):
        before = scalar_time(reduced, x)
        logger.emit(
            "epoch_start", seed=seed, stage=ResearchStage.REDUCED_TIME,
            epoch=epoch, scalar_time=before,
            pool_size=0 if active_pool is None else active_pool.size,
            resumed=True,
        )
        result = run_time(
            reduced, x, guard=guard,
            accepted_steps=policy.reduced_time_epoch_steps,
            batch_schedule=policy.reduced_time_batch_schedule,
            trust_schedule=policy.reduced_time_trust_schedule,
            max_filter_violation=policy.reduced_time_filter_violation,
            pool=active_pool,
            wall=policy.reduced_time_epoch_wall_seconds,
            maximum_raw_iterations=policy.reduced_time_maximum_raw_iterations,
            pool_retention=policy.reduced_time_pool_retention,
            highs_threads=policy.highs_threads,
        )
        x = result.parameters.copy()
        active_pool = result.pool.copy()
        payload = _stage_payload(result, epoch=epoch, guard=policy.final_reduced_guard)
        record.reduced_time_epochs.append(payload)
        ckpt = _save_checkpoint(checkpoint_dir, f"reduced_time_{epoch:03d}", x, active_pool)
        logger.emit(
            "epoch_complete", seed=seed, stage=ResearchStage.REDUCED_TIME,
            epoch=epoch, result=payload, checkpoint=ckpt, resumed=True,
        )
        _atomic_json(state_path, asdict(record))
        if result.stop_reason == "no improving certified trial":
            reduced_converged = True
            logger.emit(
                "stage_converged", seed=seed, stage=ResearchStage.REDUCED_TIME,
                epoch=epoch, scalar_time=result.scalar_time,
                reason=result.stop_reason, resumed=True,
            )
            break
        if result.stop_reason != "maximum accepted-step budget reached":
            record.status = "failed"
            record.stage = ResearchStage.FAILED.value
            record.error = (
                f"RuntimeError: seed {seed}: resumed reduced-time stage stopped "
                f"without convergence: {result.stop_reason}"
            )
            _atomic_json(state_path, asdict(record))
            return record

    record.reduced_converged = reduced_converged
    if not reduced_converged:
        record.status = "failed"
        record.stage = ResearchStage.FAILED.value
        record.error = (
            f"RuntimeError: seed {seed}: reduced-time epoch ceiling reached without "
            "transaction exhaustion"
        )
        record.total_wall_seconds += time.perf_counter() - resume_started
        _atomic_json(state_path, asdict(record))
        return record

    record.stage = ResearchStage.BASIS_BRANCH.value
    remaining_route_seconds = (
        policy.maximum_route_wall_seconds
        - float(record.total_wall_seconds)
        - (time.perf_counter() - resume_started)
    )
    record.basis_branch_attempts += 1
    final_problem, final_x, final_basis_kind, branch_payload = _run_supervised_basis_branch(
        int(seed), cells, reduced, full, x, route_dir, checkpoint_dir, logger, policy,
        timeout_seconds=remaining_route_seconds,
    )
    record.basis_branch = branch_payload
    record.basis_branch_complete = branch_payload.get("worker_status") == "complete"
    record.basis_branch_converged = bool(branch_payload.get("branch_converged", False))
    record.final_basis_kind = final_basis_kind
    record.final_segments = int(final_problem.corridor.n_segments)
    final_cert = certify_parameters_on_problem(
        final_problem, final_x,
        tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(final_cert["certified"]):
        record.status = "failed"
        record.stage = ResearchStage.FAILED.value
        record.error = "RuntimeError: final active-basis trajectory failed independent certification"
        _atomic_json(state_path, asdict(record))
        return record
    record.converged = bool(record.reduced_converged and record.basis_branch_converged)
    record.status = "complete" if record.converged else "incomplete"
    record.stage = ResearchStage.COMPLETE.value if record.converged else ResearchStage.INCOMPLETE.value
    if not record.converged:
        record.error = (
            "basis branch did not establish convergence; fastest independently "
            "certified incumbent retained"
        )
    record.final_time = scalar_time(final_problem, final_x)
    record.final_curvature = curvature_value(final_problem, final_x)
    record.final_parameter_sha256 = parameter_sha256(final_x)
    record.final_certification = final_cert
    record.total_wall_seconds += time.perf_counter() - resume_started
    ckpt = _save_checkpoint(checkpoint_dir, "final_active_basis", final_x, None)
    logger.emit(
        "route_complete" if record.converged else "route_incomplete",
        seed=seed, final_time=record.final_time,
        final_curvature=record.final_curvature,
        total_wall_seconds=record.total_wall_seconds,
        final_certification=final_cert, checkpoint=ckpt, resumed=True,
        basis_branch_converged=record.basis_branch_converged,
    )
    _atomic_json(state_path, asdict(record))
    return record


def resume_route_basis_branch(
    seed: int,
    output_dir: Path,
    *,
    policy: HomotopyPolicy = HomotopyPolicy(),
) -> RouteRunResult:
    """Retry only the serial selective-basis phase from a converged reduced checkpoint.

    This avoids repeating the expensive reduced continuation after a worker
    timeout, external interruption, or deliberately short research deadline.
    The retry is still strictly serial and begins from the same independently
    certified reduced incumbent.
    """
    attempt_started = time.perf_counter()
    route_dir = Path(output_dir) / f"seed_{int(seed)}"
    state_path = route_dir / "state.json"
    checkpoint_dir = route_dir / "checkpoints"
    logger = JsonlLogger(route_dir / "events.jsonl")
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    record = _record_from_json(state_path)
    if record.converged:
        return record
    if not record.reduced_converged:
        raise ValueError(f"seed {seed}: reduced basis has not converged")

    cells, reduced, full = build_historical_pair(int(seed))
    reduced_path = checkpoint_dir / "reduced_converged.npy"
    if not reduced_path.exists():
        if not record.reduced_time_epochs:
            raise FileNotFoundError("no reduced convergence checkpoint is available")
        reduced_path = checkpoint_dir / f"reduced_time_{len(record.reduced_time_epochs)-1:03d}.npy"
    x = np.asarray(np.load(reduced_path), dtype=float)
    cert = certify_parameters_on_problem(
        reduced,
        x,
        tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(cert["certified"]):
        raise RuntimeError(f"seed {seed}: reduced branch-resume checkpoint is not certified")

    record.status = "running"
    record.stage = ResearchStage.BASIS_BRANCH.value
    record.error = None
    record.basis_branch_complete = False
    record.basis_branch_converged = False
    record.basis_branch_attempts += 1
    logger.emit(
        "basis_branch_resume",
        seed=seed,
        attempt=record.basis_branch_attempts,
        reduced_time=scalar_time(reduced, x),
        reduced_parameter_sha256=parameter_sha256(x),
        worker_timeout_seconds=policy.basis_branch_worker_timeout_seconds,
        execution_mode="serial_single_worker",
    )
    _atomic_json(state_path, asdict(record))

    final_problem, final_x, final_basis_kind, branch_payload = _run_supervised_basis_branch(
        int(seed), cells, reduced, full, x, route_dir, checkpoint_dir, logger, policy,
        timeout_seconds=min(
            float(policy.maximum_route_wall_seconds),
            float(policy.basis_branch_worker_timeout_seconds),
        ),
    )
    record.basis_branch = branch_payload
    record.basis_branch_complete = branch_payload.get("worker_status") == "complete"
    record.basis_branch_converged = bool(branch_payload.get("branch_converged", False))
    record.final_basis_kind = final_basis_kind
    record.final_segments = int(final_problem.corridor.n_segments)

    final_cert = certify_parameters_on_problem(
        final_problem,
        final_x,
        tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(final_cert["certified"]):
        record.status = "failed"
        record.stage = ResearchStage.FAILED.value
        record.converged = False
        record.error = "RuntimeError: resumed basis incumbent failed independent certification"
    else:
        record.final_time = scalar_time(final_problem, final_x)
        record.final_curvature = curvature_value(final_problem, final_x)
        record.final_parameter_sha256 = parameter_sha256(final_x)
        record.final_certification = final_cert
        record.converged = bool(record.reduced_converged and record.basis_branch_converged)
        record.status = "complete" if record.converged else "incomplete"
        record.stage = ResearchStage.COMPLETE.value if record.converged else ResearchStage.INCOMPLETE.value
        if record.converged:
            record.error = None
        else:
            record.error = (
                "basis branch did not establish convergence; fastest independently "
                "certified incumbent retained"
            )
        ckpt = _save_checkpoint(checkpoint_dir, "final_active_basis", final_x, None)
        logger.emit(
            "route_complete" if record.converged else "route_incomplete",
            seed=seed,
            resumed_basis_branch=True,
            attempt=record.basis_branch_attempts,
            final_time=record.final_time,
            final_curvature=record.final_curvature,
            final_certification=final_cert,
            checkpoint=ckpt,
            basis_branch_converged=record.basis_branch_converged,
        )
    record.total_wall_seconds += time.perf_counter() - attempt_started
    _atomic_json(state_path, asdict(record))
    return record


def run_route_to_convergence(
    seed: int,
    output_dir: Path,
    *,
    policy: HomotopyPolicy = HomotopyPolicy(),
) -> RouteRunResult:
    """Run one historical route through the complete candidate state machine."""
    route_started = time.perf_counter()
    route_dir = Path(output_dir) / f"seed_{int(seed)}"
    route_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(route_dir / "events.jsonl")
    state_path = route_dir / "state.json"
    checkpoint_dir = route_dir / "checkpoints"

    cells, reduced, full = build_historical_pair(int(seed))
    x = np.asarray(reduced.initial_parameters, dtype=float).copy()
    initial_cert = certify_parameters_on_problem(
        reduced,
        x,
        tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if not bool(initial_cert["certified"]):
        raise RuntimeError(f"seed {seed}: analytic reduced initializer is not certified")

    initial_time = scalar_time(reduced, x)
    initial_curvature = curvature_value(reduced, x)
    record = RouteRunResult(
        seed=int(seed),
        cells=tuple(cells),
        status="running",
        stage=ResearchStage.INITIAL.value,
        converged=False,
        initial_time=initial_time,
        initial_curvature=initial_curvature,
    )
    logger.emit(
        "route_start",
        seed=int(seed),
        cells=cells,
        reduced_segments=reduced.corridor.n_segments,
        full_segments=full.corridor.n_segments,
        initial_time=initial_time,
        initial_curvature=initial_curvature,
        initial_certification=initial_cert,
        policy=asdict(policy),
    )
    _save_checkpoint(checkpoint_dir, "initial_reduced", x, None)
    _atomic_json(state_path, asdict(record))

    active_pool: ConstraintPool | None = None

    def ensure_route_budget() -> None:
        elapsed = time.perf_counter() - route_started
        if elapsed > policy.maximum_route_wall_seconds:
            raise TimeoutError(
                f"seed {seed}: route wall ceiling {policy.maximum_route_wall_seconds:.1f}s reached"
            )

    try:
        # Curvature continuation at q=.01 and q=.005 is unconditional.
        q010, q005, q0025 = policy.curvature_guard_sequence
        for stage, q in (
            (ResearchStage.CURVATURE_Q010, q010),
            (ResearchStage.CURVATURE_Q005, q005),
        ):
            ensure_route_budget()
            record.stage = stage.value
            guard = make_guard(cells, reduced, full, q)
            logger.emit("stage_start", seed=seed, stage=stage, guard_fraction=q)
            result = run_curvature(
                reduced,
                x,
                guard=guard,
                accepted_steps=policy.curvature_epoch_steps,
                batch_schedule=policy.curvature_batch_schedule,
                trust_schedule=policy.curvature_trust_schedule,
                max_filter_violation=policy.curvature_filter_violation,
                pool=active_pool,
                wall=policy.curvature_epoch_wall_seconds,
                maximum_raw_iterations=policy.curvature_maximum_raw_iterations,
                highs_threads=policy.highs_threads,
            )
            x = result.parameters.copy()
            active_pool = result.pool.copy()
            payload = _stage_payload(result, guard=q)
            record.curvature_stages.append(payload)
            ckpt = _save_checkpoint(checkpoint_dir, stage.value, x, active_pool)
            logger.emit("stage_complete", seed=seed, stage=stage, result=payload, checkpoint=ckpt)
            _atomic_json(state_path, asdict(record))

        # Decide whether the final curvature epoch is still buying enough route time.
        switch = curvature_switch_decision(result, policy)
        record.curvature_switch = asdict(switch)
        logger.emit("curvature_switch_decision", seed=seed, decision=asdict(switch))

        if switch.continue_curvature:
            ensure_route_budget()
            stage = ResearchStage.CURVATURE_Q0025
            record.stage = stage.value
            guard = make_guard(cells, reduced, full, q0025)
            logger.emit("stage_start", seed=seed, stage=stage, guard_fraction=q0025)
            result = run_curvature(
                reduced,
                x,
                guard=guard,
                accepted_steps=policy.curvature_epoch_steps,
                batch_schedule=policy.curvature_batch_schedule,
                trust_schedule=policy.curvature_trust_schedule,
                max_filter_violation=policy.curvature_filter_violation,
                pool=active_pool,
                wall=policy.curvature_epoch_wall_seconds,
                maximum_raw_iterations=policy.curvature_maximum_raw_iterations,
                highs_threads=policy.highs_threads,
            )
            x = result.parameters.copy()
            active_pool = result.pool.copy()
            payload = _stage_payload(result, guard=q0025)
            record.curvature_stages.append(payload)
            ckpt = _save_checkpoint(checkpoint_dir, stage.value, x, active_pool)
            logger.emit("stage_complete", seed=seed, stage=stage, result=payload, checkpoint=ckpt)
            _atomic_json(state_path, asdict(record))
        else:
            logger.emit("stage_skipped", seed=seed, stage=ResearchStage.CURVATURE_Q0025, reason="switch criterion")

        # Reduced-basis time continuation.  The current point is independently
        # certified, so stale curvature-stage finite cuts are not correctness
        # state.  Rebuild a near-active time proposal pool from this checkpoint.
        record.stage = ResearchStage.REDUCED_TIME.value
        active_pool = None
        guard = make_guard(cells, reduced, full, policy.final_reduced_guard)
        reduced_converged = False
        for epoch in range(policy.maximum_reduced_time_epochs):
            ensure_route_budget()
            before = scalar_time(reduced, x)
            logger.emit(
                "epoch_start",
                seed=seed,
                stage=ResearchStage.REDUCED_TIME,
                epoch=epoch,
                scalar_time=before,
                pool_size=0 if active_pool is None else active_pool.size,
            )
            result = run_time(
                reduced,
                x,
                guard=guard,
                accepted_steps=policy.reduced_time_epoch_steps,
                batch_schedule=policy.reduced_time_batch_schedule,
                trust_schedule=policy.reduced_time_trust_schedule,
                max_filter_violation=policy.reduced_time_filter_violation,
                pool=active_pool,
                wall=policy.reduced_time_epoch_wall_seconds,
                maximum_raw_iterations=policy.reduced_time_maximum_raw_iterations,
                pool_retention=policy.reduced_time_pool_retention,
                highs_threads=policy.highs_threads,
            )
            x = result.parameters.copy()
            active_pool = result.pool.copy()
            payload = _stage_payload(result, epoch=epoch, guard=policy.final_reduced_guard)
            record.reduced_time_epochs.append(payload)
            ckpt = _save_checkpoint(checkpoint_dir, f"reduced_time_{epoch:03d}", x, active_pool)
            logger.emit(
                "epoch_complete",
                seed=seed,
                stage=ResearchStage.REDUCED_TIME,
                epoch=epoch,
                result=payload,
                checkpoint=ckpt,
            )
            _atomic_json(state_path, asdict(record))
            if result.stop_reason == "no improving certified trial":
                reduced_converged = True
                logger.emit(
                    "stage_converged",
                    seed=seed,
                    stage=ResearchStage.REDUCED_TIME,
                    epoch=epoch,
                    scalar_time=result.scalar_time,
                    reason=result.stop_reason,
                )
                break
            if result.stop_reason != "maximum accepted-step budget reached":
                raise RuntimeError(
                    f"seed {seed}: reduced-time stage stopped without convergence: {result.stop_reason}"
                )
        record.reduced_converged = reduced_converged
        if not reduced_converged:
            raise RuntimeError(
                f"seed {seed}: reduced-time epoch ceiling reached without transaction exhaustion"
            )

        # Certification-gated selective-basis refinement.
        ensure_route_budget()
        record.stage = ResearchStage.BASIS_BRANCH.value
        remaining_route_seconds = (
            policy.maximum_route_wall_seconds - (time.perf_counter() - route_started)
        )
        record.basis_branch_attempts += 1
        final_problem, final_x, final_basis_kind, branch_payload = _run_supervised_basis_branch(
            int(seed), cells, reduced, full, x, route_dir, checkpoint_dir, logger, policy,
            timeout_seconds=remaining_route_seconds,
        )
        record.basis_branch = branch_payload
        record.basis_branch_complete = branch_payload.get("worker_status") == "complete"
        record.basis_branch_converged = bool(branch_payload.get("branch_converged", False))
        record.final_basis_kind = final_basis_kind
        record.final_segments = int(final_problem.corridor.n_segments)
        ckpt = _save_checkpoint(checkpoint_dir, "final_active_basis", final_x, None)
        logger.emit(
            "basis_branch_complete", seed=seed, result=branch_payload, checkpoint=ckpt,
            final_basis_kind=final_basis_kind, final_segments=record.final_segments,
        )
        _atomic_json(state_path, asdict(record))

        final_cert = certify_parameters_on_problem(
            final_problem,
            final_x,
            tolerance=policy.feasibility_tolerance,
            maximum_abs_sigma=policy.curvature_slope_limit,
        )
        if not bool(final_cert["certified"]):
            raise RuntimeError(f"seed {seed}: final active-basis trajectory failed independent certification")
        final_time = scalar_time(final_problem, final_x)
        final_curvature = curvature_value(final_problem, final_x)
        record.converged = bool(record.reduced_converged and record.basis_branch_converged)
        record.status = "complete" if record.converged else "incomplete"
        record.stage = ResearchStage.COMPLETE.value if record.converged else ResearchStage.INCOMPLETE.value
        if not record.converged:
            record.error = (
                "basis branch did not establish convergence; fastest independently "
                "certified incumbent retained"
            )
        record.final_time = final_time
        record.final_curvature = final_curvature
        record.final_parameter_sha256 = parameter_sha256(final_x)
        record.final_certification = final_cert
        record.total_wall_seconds = time.perf_counter() - route_started
        ckpt = _save_checkpoint(checkpoint_dir, "final_active_basis", final_x, None)
        logger.emit(
            "route_complete" if record.converged else "route_incomplete",
            seed=seed,
            final_time=final_time,
            final_curvature=final_curvature,
            total_wall_seconds=record.total_wall_seconds,
            final_certification=final_cert,
            checkpoint=ckpt,
            basis_branch_converged=record.basis_branch_converged,
        )
        _atomic_json(state_path, asdict(record))
        return record
    except Exception as exc:
        record.status = "failed"
        record.stage = ResearchStage.FAILED.value
        record.converged = False
        record.error = f"{type(exc).__name__}: {exc}"
        record.total_wall_seconds = time.perf_counter() - route_started
        logger.emit(
            "route_failed",
            seed=seed,
            error=record.error,
            total_wall_seconds=record.total_wall_seconds,
        )
        _atomic_json(state_path, asdict(record))
        return record


def run_historical_five(
    output_dir: Path,
    *,
    seeds: Iterable[int] = range(5),
    policy: HomotopyPolicy = HomotopyPolicy(),
) -> dict[str, Any]:
    """Run the selected historical routes serially for uncontaminated timings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        route_state = output_dir / f"seed_{int(seed)}" / "state.json"
        if route_state.exists():
            existing = _record_from_json(route_state)
            if existing.converged:
                result = existing
            elif existing.error and "reduced-time epoch ceiling" in existing.error:
                result = resume_route_after_reduced_ceiling(int(seed), output_dir, policy=policy)
            elif existing.reduced_converged:
                result = resume_route_basis_branch(int(seed), output_dir, policy=policy)
            else:
                # Interrupted before a certified reduced convergence checkpoint:
                # restart only that route; campaign ordering remains serial.
                import shutil
                shutil.rmtree(route_state.parent)
                result = run_route_to_convergence(int(seed), output_dir, policy=policy)
        else:
            result = run_route_to_convergence(int(seed), output_dir, policy=policy)
        rows.append(asdict(result))
        partial = {
            "schema_version": 1,
            "execution_mode": "serial_single_worker",
            "numerical_thread_environment": {name: "1" for name in THREAD_ENVIRONMENT_VARIABLES},
            "policy": asdict(policy),
            "rows": rows,
            "campaign_wall_seconds": time.perf_counter() - campaign_started,
            "complete": False,
        }
        _atomic_json(output_dir / "campaign.json", partial)
    payload = {
        "schema_version": 1,
        "execution_mode": "serial_single_worker",
        "numerical_thread_environment": {name: "1" for name in THREAD_ENVIRONMENT_VARIABLES},
        "policy": asdict(policy),
        "rows": rows,
        "campaign_wall_seconds": time.perf_counter() - campaign_started,
        "complete": all(bool(row["converged"]) for row in rows),
    }
    _atomic_json(output_dir / "campaign.json", payload)
    return payload


__all__ = [
    "HomotopyPolicy",
    "ResearchStage",
    "RouteRunResult",
    "SwitchDecision",
    "curvature_switch_decision",
    "restore_pool",
    "resume_route_after_reduced_ceiling",
    "resume_route_basis_branch",
    "run_historical_five",
    "run_route_to_convergence",
]
