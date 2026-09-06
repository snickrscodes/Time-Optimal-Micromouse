"""Supervised Sparse SQP specialist polishing for certified SLSQP routes.

This is the production integration point that was missing through V9.  SLSQP
remains the primary route optimizer.  A continuously certified SLSQP candidate
may be handed to the persistent sparse backend under explicit qualification or
shadow policy; the parent-owned SLSQP certificate is always retained.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import (
    AdaptiveBatchSettings,
    ConstraintPool,
    CurvatureSlopeConstraint,
    ExchangeSettings,
    KnotCoordinateSettings,
    LogLengthKnotMap,
    SLSQPSettings,
    SparseSQPSettings,
    SupervisorSettings,
    align_endpoint_target_to_path,
    canonicalize_phase_one_handoff,
    knot_parameters_to_raw,
    pullback_raw_gradient_to_knot_parameters,
    reverse_solver,
    run_supervised_adaptive_sparse_polish,
    scalar_reverse_solver,
    separate_rectangle_path,
    stabilize_phase_one_sparse_exchange,
)

from .maze_routes import RouteOptimizationProblem, stable_knot_bounds
from .route_optimization_policy import SparseSpecialistMode, SparseSpecialistPolicySettings
from .sparse_route_activation import (
    SparseRouteFeatures,
    count_route_turns,
    evaluate_sparse_route_eligibility,
)

Array = NDArray[np.float64]


class _TimeObjective:
    def __init__(
        self,
        initial_k: float,
        *,
        init_w: float,
        terminal_w_max: float | None,
        n_scan: int,
        envelope_scan: int,
        domain_scan: int,
    ) -> None:
        self.initial_k = float(initial_k)
        self.init_w = float(init_w)
        self.terminal_w_max = None if terminal_w_max is None else float(terminal_w_max)
        self.n_scan = int(n_scan)
        self.envelope_scan = int(envelope_scan)
        self.domain_scan = int(domain_scan)

    def __call__(self, parameters: Sequence[float]) -> tuple[float, Array]:
        raw = knot_parameters_to_raw(parameters, initial_k=self.initial_k)
        value, raw_gradient = reverse_solver.time_value_and_gradient(
            raw,
            init_w=self.init_w,
            terminal_w_max=self.terminal_w_max,
            initial_k=self.initial_k,
            n_scan=self.n_scan,
            envelope_scan=self.envelope_scan,
            domain_scan=self.domain_scan,
        )
        gradient = pullback_raw_gradient_to_knot_parameters(
            parameters,
            raw_gradient,
            initial_k=self.initial_k,
        )
        return float(value), np.asarray(gradient, dtype=float)


@dataclass(frozen=True, slots=True)
class SparseSpecialistRecord:
    schema_version: int
    mode: str
    attempted: bool
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    status: str
    baseline_time: float
    best_certified_time: float | None
    best_numerical_time: float | None
    planner_time: float
    planner_promoted: bool
    elapsed_seconds: float
    slsqp_seconds: float
    finite_pool_size: int
    canonical_state_digest: str | None
    worker_failure_reason: str | None
    worker_message: str | None
    completed_batches: int
    handoff_seconds: float = 0.0
    cuts_added: int = 0
    restoration_used: bool = False
    restoration_seconds: float = 0.0
    sparse_checkpoint_count: int = 0

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SparseSpecialistExecutionResult:
    parameters: Array
    time: float
    selected_stage: str
    record: SparseSpecialistRecord


def _append_jsonl(path: str | None, record: SparseSpecialistRecord) -> None:
    if path is None:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Telemetry must never affect planner behavior.
        return


def _certify(
    problem: RouteOptimizationProblem,
    parameters: Sequence[float],
    *,
    tolerance: float,
    slope_limit: float | None,
) -> bool:
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        return False
    from optimization import compile_geometry_path

    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    final = compile_geometry_path(raw, problem.initial_state).final_state
    endpoint = problem.terminal_violation(final)
    separation = separate_rectangle_path(x, problem.initial_state, problem.corridor)
    slope_ok = True
    if slope_limit is not None:
        slope_ok = max(abs(float(v)) for v in raw[1::2]) <= slope_limit + tolerance
    return bool(
        endpoint <= tolerance
        and separation.certified(tolerance)
        and slope_ok
    )


def run_sparse_specialist_polish(
    problem: RouteOptimizationProblem,
    parameters: Sequence[float],
    baseline_time: float,
    *,
    pool: ConstraintPool | None,
    init_w: float,
    terminal_w_max: float | None = None,
    n_scan: int,
    envelope_scan: int,
    domain_scan: int,
    feasibility_tolerance: float,
    curvature_slope_limit: float | None,
    slsqp_seconds: float,
    settings: SparseSpecialistPolicySettings,
) -> SparseSpecialistExecutionResult:
    """Optionally polish a certified SLSQP route with supervised Sparse SQP."""
    started = time.perf_counter()
    x = np.asarray(parameters, dtype=float).copy()
    baseline_time = float(baseline_time)
    if settings.mode is SparseSpecialistMode.OFF:
        record = SparseSpecialistRecord(
            1, settings.mode.value, False, False, ("specialist_disabled",),
            "disabled", baseline_time, None, None, baseline_time, False, 0.0,
            float(slsqp_seconds), 0, None, None, None, 0,
        )
        _append_jsonl(settings.telemetry_jsonl, record)
        return SparseSpecialistExecutionResult(x, baseline_time, "slsqp_certified", record)

    if not _certify(
        problem, x, tolerance=feasibility_tolerance,
        slope_limit=curvature_slope_limit,
    ):
        record = SparseSpecialistRecord(
            1, settings.mode.value, False, False, ("slsqp_handoff_not_strict",),
            "ineligible", baseline_time, None, None, baseline_time, False,
            time.perf_counter() - started, float(slsqp_seconds), 0, None, None,
            None, 0,
        )
        _append_jsonl(settings.telemetry_jsonl, record)
        return SparseSpecialistExecutionResult(x, baseline_time, "slsqp_certified", record)

    handoff_started = time.perf_counter()
    active_pool = pool.copy() if pool is not None else ConstraintPool.seeded(problem.corridor)
    coordinate_map = LogLengthKnotMap.from_knot_parameters(x)
    canonical_digest: str | None = None
    canonical_ok = False
    try:
        handoff = canonicalize_phase_one_handoff(
            x,
            active_pool,
            coordinate_map,
            endpoint_target=problem.endpoint_target,
        )
        x = handoff.physical_x.copy()
        active_pool = handoff.pool
        canonical_digest = handoff.state_digest
        canonical_ok = True
    except (ValueError, RuntimeError, FloatingPointError):
        canonical_ok = False

    features = SparseRouteFeatures(
        n_cells=len(problem.cells),
        n_variables=int(x.size),
        finite_pool_size=int(active_pool.size),
        turn_count=count_route_turns(problem.cells),
        body_length=float(problem.corridor.body.length),
        body_height=float(problem.corridor.body.height),
        phase_one_seconds=max(0.0, float(slsqp_seconds)),
        phase_one_strictly_certified=True,
        planner_fallback_certified=True,
        supervisor_healthy=True,
        canonicalization_succeeded=canonical_ok,
        canonical_diagnostics_valid=canonical_ok and canonical_digest is not None,
        estimated_sparse_seconds=float(settings.maximum_sparse_seconds),
        public_highspy_validated=settings.public_highspy_validated,
        deployment_watchdog_validated=settings.deployment_watchdog_validated,
        sufficient_shadow_evidence=settings.sufficient_shadow_evidence,
        state_sensitivity_acceptable=settings.state_sensitivity_acceptable,
        strict_handoff_certified=True,
        handoff_seconds=max(0.0, float(slsqp_seconds)),
        handoff_source="slsqp_certified",
    )
    handoff_seconds = time.perf_counter() - handoff_started
    shadow = settings.mode in {
        SparseSpecialistMode.SHADOW,
        SparseSpecialistMode.EXPERIMENTAL_PROMOTE,
    }
    decision = evaluate_sparse_route_eligibility(features, shadow_mode=shadow)
    reasons = list(decision.reasons)
    if float(slsqp_seconds) < settings.minimum_slsqp_seconds:
        reasons.append("slsqp_stage_below_specialist_cost_trigger")
    eligible = decision.eligible and not any(
        reason == "slsqp_stage_below_specialist_cost_trigger" for reason in reasons
    )
    if not eligible:
        record = SparseSpecialistRecord(
            1, settings.mode.value, False, False, tuple(reasons), "ineligible",
            baseline_time, None, None, baseline_time, False,
            time.perf_counter() - started, float(slsqp_seconds), active_pool.size,
            canonical_digest, None, None, 0,
        )
        _append_jsonl(settings.telemetry_jsonl, record)
        return SparseSpecialistExecutionResult(x, baseline_time, "slsqp_certified", record)

    sparse_settings = SparseSQPSettings(
        max_iterations=100,
        initial_trust_radius=0.08,
        maximum_trust_radius=0.20,
        maximum_filter_violation=1.0e-3,
        feasibility_tolerance=1.0e-9,
        equality_tolerance=1.0e-9,
        final_restoration_iterations=0,
        jacobian_mode="direct_sparse",
        allow_private_highs_fallback=not settings.public_highspy_validated,
    )
    exchange = stabilize_phase_one_sparse_exchange(
        ExchangeSettings(
            maximum_rounds=1,
            finite_inequality_tolerance=1.0e-8,
            finite_equality_tolerance=1.0e-8,
            require_solver_success=False,
            finite_solver="sparse_sqp",
            sparse_sqp=sparse_settings,
            final_kkt_diagnostics=False,
            slsqp=SLSQPSettings(max_iterations=120, ftol=1.0e-10),
        )
    )
    batch = AdaptiveBatchSettings(
        batch_iterations=settings.accepted_steps_per_batch,
        maximum_raw_iterations_per_batch=100,
        minimum_batches=1,
        maximum_batches=settings.maximum_batches,
        maximum_wall_seconds=settings.maximum_sparse_seconds,
        maximum_objective_evaluations=max(
            80, settings.maximum_batches * (settings.accepted_steps_per_batch + 12)
        ),
        required_stalled_certified_batches=2,
        criticality_schedule="final_only",
        continuation_violation_cap=1.0e-3,
        continuation_extended_violation_cap=1.0e-2,
        restoration_enabled=True,
        restoration_maximum_retries=2,
        restoration_accepted_steps=20,
        restoration_maximum_raw_iterations=120,
        restoration_initial_trust_radius=0.1,
        restoration_maximum_displacement=0.25,
        hessian_reset_displacement=0.1,
        objective_soft_deadline_enabled=True,
        feasibility_first_enabled=decision.use_high_work_early_certification,
    )
    aligned_target = align_endpoint_target_to_path(
        x, problem.initial_state, problem.endpoint_target
    ).aligned_target
    supervisor = SupervisorSettings(
        hard_timeout_seconds=min(45.0, settings.maximum_sparse_seconds),
        worker_startup_timeout_seconds=15.0,
        poll_interval_seconds=0.02,
        terminate_grace_seconds=0.5,
        deadline_refresh_on_checkpoint=True,
        start_method="forkserver",
    )
    objective = _TimeObjective(
        problem.initial_state.k,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_scan=domain_scan,
    )
    supervised = run_supervised_adaptive_sparse_polish(
        x,
        objective,
        problem.corridor,
        problem.initial_state,
        bounds=stable_knot_bounds(x),
        endpoint_target=aligned_target,
        additional_inequalities=problem.inequalities(
            None
            if curvature_slope_limit is None
            else CurvatureSlopeConstraint(
                curvature_slope_limit, initial_k=problem.initial_state.k
            )
        ),
        pool=active_pool,
        exchange_settings=exchange,
        batch_settings=batch,
        coordinate_map=coordinate_map,
        coordinate_settings=KnotCoordinateSettings(),
        supervisor_settings=supervisor,
    )
    elapsed = time.perf_counter() - started
    outcome = supervised.outcome
    safe = supervised.safe_result
    best_certified_time: float | None = None
    best_numerical_time: float | None = None
    safe_x: Array | None = None
    completed_batches = 0
    cuts_added = 0
    restoration_used = False
    restoration_seconds = 0.0
    sparse_checkpoint_count = 0
    if supervised.result is not None:
        completed_batches = len(supervised.result.batches)
        cuts_added = sum(int(batch_record.cuts_added) for batch_record in supervised.result.batches)
        restoration_used = any(bool(batch_record.restoration_attempted) for batch_record in supervised.result.batches)
        restoration_seconds = sum(float(batch_record.restoration_seconds) for batch_record in supervised.result.batches)
        current = supervised.result.result.current_objective
        if math.isfinite(current):
            best_numerical_time = float(current)
    if supervised.safe_checkpoint is not None:
        sparse_checkpoint_count = len(supervised.safe_checkpoint.completed_batches)
    if safe is not None:
        safe_x = np.asarray(safe.x, dtype=float).copy()
        if _certify(
            problem, safe_x, tolerance=feasibility_tolerance,
            slope_limit=curvature_slope_limit,
        ):
            raw = knot_parameters_to_raw(safe_x, initial_k=problem.initial_state.k)
            value = scalar_reverse_solver.evaluate_time_scalar(
                raw,
                init_w=init_w,
                initial_k=problem.initial_state.k,
                n_scan=n_scan,
                envelope_scan=envelope_scan,
                domain_scan=domain_scan,
            )
            if math.isfinite(value):
                best_certified_time = float(value)

    promote_allowed = settings.mode in {
        SparseSpecialistMode.AUTO,
        SparseSpecialistMode.EXPERIMENTAL_PROMOTE,
    }
    promoted = bool(
        promote_allowed
        and best_certified_time is not None
        and safe_x is not None
        and best_certified_time < baseline_time
    )
    planner_x = safe_x if promoted and safe_x is not None else x
    planner_time = best_certified_time if promoted and best_certified_time is not None else baseline_time
    status = "promoted" if promoted else (
        "shadow_completed" if settings.mode is SparseSpecialistMode.SHADOW else
        ("completed_no_improvement" if outcome.success else "worker_failed_safe_fallback")
    )
    record = SparseSpecialistRecord(
        1,
        settings.mode.value,
        True,
        True,
        tuple(reasons),
        status,
        baseline_time,
        best_certified_time,
        best_numerical_time,
        float(planner_time),
        promoted,
        elapsed,
        float(slsqp_seconds),
        active_pool.size,
        canonical_digest,
        None if outcome.success else outcome.failure_reason,
        outcome.message,
        completed_batches,
        float(handoff_seconds),
        int(cuts_added),
        bool(restoration_used),
        float(restoration_seconds),
        int(sparse_checkpoint_count),
    )
    _append_jsonl(settings.telemetry_jsonl, record)
    return SparseSpecialistExecutionResult(
        np.asarray(planner_x, dtype=float).copy(),
        float(planner_time),
        "sparse_specialist" if promoted else "slsqp_certified",
        record,
    )


__all__ = [
    "SparseSpecialistExecutionResult",
    "SparseSpecialistRecord",
    "run_sparse_specialist_polish",
]
