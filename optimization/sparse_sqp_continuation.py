"""Persistent, continuously safe adaptive sparse-SQP continuation."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .constraint_generation import (
    CombinedConstraintOracle,
    ConstraintPool,
    EndpointPoseEquality,
    KnotGeometryCache,
    RectangleConstraintOracle,
    SeparationReport,
    add_report_violations,
    separate_rectangle_path,
    stack_vector_constraints,
)
from .kkt_lsq import BoxBounds, KKTReport
from .path_optimizer import (
    ExchangeResult,
    ExchangeSettings,
    KnotCoordinateSettings,
    LogLengthKnotMap,
    run_constraint_generation,
)
from .sparse_sqp import (
    ObjectiveEvaluationTimeout,
    SparseSQPContinuationState,
    SparseSQPDiagnosticControls,
    SparseSQPResult,
)
from .sqp_supervisor_service import (
    ExternalSupervisorServiceClient,
    SupervisorServiceClientSettings,
)
from .sqp_supervisor import (
    SupervisedBatchOutcome,
    SupervisedBatchRunner,
    SupervisorSettings,
    WorkerContext,
)

Array = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class _ResolvedFeasibilityFirstPolicy:
    active: bool
    work_estimate: int
    batch_iterations: int
    restoration_sparse_retries: int


def _resolve_feasibility_first_policy(
    settings: "AdaptiveBatchSettings",
    *,
    pool_size: int,
    variable_count: int,
) -> _ResolvedFeasibilityFirstPolicy:
    work_estimate = int(pool_size * variable_count)
    active = bool(
        settings.feasibility_first_enabled
        and work_estimate >= settings.feasibility_first_work_threshold
    )
    batch_iterations = (
        min(settings.batch_iterations, settings.feasibility_first_batch_iterations)
        if active
        else settings.batch_iterations
    )
    restoration_sparse_retries = (
        0
        if active and settings.feasibility_first_direct_slsqp_restoration
        else settings.restoration_maximum_retries
    )
    return _ResolvedFeasibilityFirstPolicy(
        active, work_estimate, batch_iterations, restoration_sparse_retries
    )


@dataclass(frozen=True, slots=True)
class AdaptiveBatchSettings:
    batch_iterations: int = 5
    maximum_raw_iterations_per_batch: int = 100
    minimum_batches: int = 2
    maximum_batches: int = 16
    maximum_wall_seconds: float = 180.0
    maximum_objective_evaluations: int = 160
    relative_objective_tolerance: float = 2.0e-4
    absolute_objective_tolerance: float = 2.0e-4
    required_stalled_certified_batches: int = 2
    criticality_target: float = 1.0e-5
    criticality_schedule: Literal[
        "accepted_step", "batch", "stagnation", "final_only"
    ] = "final_only"
    intermediate_multiplier_diagnostics: bool = False
    continue_after_new_cuts: bool = True
    continue_until_first_certified: bool = True
    continuation_violation_cap: float = 1.0e-3
    continuation_extended_violation_cap: float = 1.0e-2
    continuation_extended_maximum_work: int = 150_000
    continuation_extended_enabled: bool = True
    batch_minimum_predicted_reduction: float = 1.0e-10
    restoration_enabled: bool = True
    restoration_maximum_retries: int = 2
    restoration_accepted_steps: int = 20
    restoration_maximum_raw_iterations: int = 120
    restoration_inner_final_restoration_iterations: int = 0
    restoration_initial_trust_radius: float = 1.0e-1
    restoration_trust_contraction: float = 0.5
    restoration_maximum_displacement: float = 2.5e-1
    hessian_reset_displacement: float = 1.0e-1
    restoration_slsqp_fallback: bool = True
    restoration_slsqp_max_iterations: int = 120
    restoration_slsqp_exchange_rounds: int = 4
    objective_soft_deadline_enabled: bool = True
    feasibility_first_enabled: bool = False
    feasibility_first_work_threshold: int = 150_000
    feasibility_first_batch_iterations: int = 1
    feasibility_first_direct_slsqp_restoration: bool = True

    def __post_init__(self) -> None:
        integer_positive = (
            self.batch_iterations,
            self.maximum_raw_iterations_per_batch,
            self.minimum_batches,
            self.maximum_batches,
            self.maximum_objective_evaluations,
            self.required_stalled_certified_batches,
            self.restoration_accepted_steps,
            self.restoration_maximum_raw_iterations,
            self.restoration_slsqp_max_iterations,
            self.restoration_slsqp_exchange_rounds,
            self.continuation_extended_maximum_work,
            self.feasibility_first_work_threshold,
            self.feasibility_first_batch_iterations,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("adaptive integer budgets must be positive")
        if self.restoration_maximum_retries < 0:
            raise ValueError("restoration_maximum_retries must be nonnegative")
        if (
            self.feasibility_first_enabled
            and self.feasibility_first_direct_slsqp_restoration
            and not self.restoration_slsqp_fallback
        ):
            raise ValueError(
                "feasibility-first direct restoration requires the SLSQP fallback"
            )
        if self.restoration_maximum_retries == 0 and not self.restoration_slsqp_fallback:
            raise ValueError(
                "zero sparse restoration retries requires the SLSQP fallback"
            )
        if self.maximum_batches < self.minimum_batches:
            raise ValueError("maximum_batches must be >= minimum_batches")
        if (
            self.continuation_extended_violation_cap
            < self.continuation_violation_cap
        ):
            raise ValueError(
                "extended continuation cap must be >= the base cap"
            )
        if self.restoration_inner_final_restoration_iterations < 0:
            raise ValueError(
                "restoration_inner_final_restoration_iterations must be nonnegative"
            )
        positive = (
            self.maximum_wall_seconds,
            self.continuation_violation_cap,
            self.continuation_extended_violation_cap,
            self.batch_minimum_predicted_reduction,
            self.restoration_initial_trust_radius,
            self.restoration_maximum_displacement,
            self.hessian_reset_displacement,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("adaptive positive settings must be finite")
        if not 0.0 < self.restoration_trust_contraction < 1.0:
            raise ValueError("restoration_trust_contraction must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class CutProvenance:
    batch: int
    source: Literal[
        "finite_feasible_incumbent", "current_filter_iterate", "restoration"
    ]
    cuts_added: int
    pool_size_after: int


@dataclass(frozen=True, slots=True)
class RestorationAttempt:
    retry: int
    method: Literal["sparse_sqp", "slsqp_fallback"]
    wall_seconds: float
    trust_radius: float
    start_displacement: float
    result_displacement: float
    finite_inequality: float
    equality_inf: float
    exact_upper_bound: float
    strict_certificate: bool
    solver_success: bool
    solver_message: str
    pool_size: int
    inner_final_restoration_iterations: int


@dataclass(frozen=True, slots=True)
class RestorationRecord:
    batch: int
    attempted: bool
    trigger: str
    retries: int
    accepted: bool
    maximum_displacement: float
    final_inequality: float
    final_equality: float
    final_exact_upper_bound: float
    hessian_preserved: bool
    message: str
    attempts: tuple[RestorationAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class AdaptiveBatchRecord:
    batch: int
    wall_seconds: float
    cumulative_seconds: float
    objective: float
    finite_round_seconds: float
    current_separation_seconds: float
    candidate_recheck_seconds: float
    restoration_seconds: float
    objective_seconds: float
    constraint_seconds: float
    subproblem_seconds: float
    maximum_objective_seconds: float
    current_objective: float
    current_finite_inequality: float
    current_equality_inf: float
    current_exact_worst_upper_bound: float
    continuously_certified: bool
    cuts_added: int
    cuts_from_finite: int
    cuts_from_current: int
    pool_size: int
    objective_evaluations: int
    accepted_steps: int
    first_iteration_objective: float
    first_iteration_violation: float
    first_iteration_accepted: bool
    first_iteration_step_kind: str
    first_iteration_step_norm_inf: float
    first_accepted_step_index: int | None
    first_accepted_step_objective: float | None
    objective_timeouts: int
    raw_iterations: int
    rejected_trials: int
    normalized_stationarity: float
    normalized_complementarity: float
    criticality: float
    criticality_success: bool
    multiplier_calls: int
    multiplier_seconds: float
    criticality_calls: int
    criticality_seconds: float
    objective_improvement: float
    restoration_attempted: bool
    restoration_succeeded: bool
    continuation_effective_cap: float
    continuation_extended_eligible: bool
    continuation_work_estimate: int
    continuation_decision: str
    prebatch_work_estimate: int
    effective_batch_iterations: int
    restoration_sparse_retry_budget: int
    feasibility_first_active: bool
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class AdaptivePolishResult:
    result: ExchangeResult
    best_certified_result: ExchangeResult | None
    batches: tuple[AdaptiveBatchRecord, ...]
    stop_reason: str
    total_objective_evaluations: int
    wall_seconds: float
    continuation_state: SparseSQPContinuationState | None
    cut_provenance: tuple[CutProvenance, ...]
    restorations: tuple[RestorationRecord, ...]
    final_diagnostics_complete: bool
    final_diagnostic_failure: str | None

    def checkpoint_safe_copy(self) -> "AdaptivePolishResult":
        safe_state = (
            None
            if self.continuation_state is None
            else self.continuation_state.checkpoint_safe_copy()
        )
        safe_result = _checkpoint_safe_exchange(
            self.result, state_override=safe_state, keep_rounds=True
        )
        safe_best = (
            None
            if self.best_certified_result is None
            else _checkpoint_safe_exchange(
                self.best_certified_result,
                state_override=safe_state,
                keep_rounds=True,
            )
        )
        return replace(
            self,
            result=safe_result,
            best_certified_result=safe_best,
            continuation_state=safe_state,
        )

@dataclass(frozen=True, slots=True)
class AdaptiveSafeCheckpoint:
    """Serializable recovery point containing only planner-safe output."""

    result: ExchangeResult
    continuation_state: SparseSQPContinuationState | None
    completed_batches: tuple[AdaptiveBatchRecord, ...]
    cut_provenance: tuple[CutProvenance, ...]
    restorations: tuple[RestorationRecord, ...]
    constraint_generation: int

    def checkpoint_safe_copy(self) -> "AdaptiveSafeCheckpoint":
        safe_state = (
            None
            if self.continuation_state is None
            else self.continuation_state.checkpoint_safe_copy()
        )
        safe_result = _checkpoint_safe_exchange(
            self.result, state_override=safe_state, keep_rounds=False
        )
        return replace(
            self,
            result=safe_result,
            continuation_state=safe_state,
        )


def _emit_safe_checkpoint(
    callback: Callable[[AdaptiveSafeCheckpoint], None] | None,
    checkpoint: AdaptiveSafeCheckpoint,
) -> None:
    if callback is not None:
        callback(checkpoint.checkpoint_safe_copy())

@dataclass(frozen=True, slots=True)
class SupervisedAdaptivePolishResult:
    """Parent-side outcome with an always-explicit planner-safe fallback."""

    outcome: SupervisedBatchOutcome
    result: AdaptivePolishResult | None
    safe_checkpoint: AdaptiveSafeCheckpoint | None
    safe_result: ExchangeResult | None



def _checkpoint_safe_exchange(
    result: ExchangeResult,
    *,
    state_override: SparseSQPContinuationState | None = None,
    keep_rounds: bool,
) -> ExchangeResult:
    safe_state = (
        state_override
        if state_override is not None
        else (
            None
            if result.continuation_state is None
            else result.continuation_state.checkpoint_safe_copy()
        )
    )
    if not keep_rounds:
        return replace(result, rounds=(), continuation_state=safe_state)
    safe_rounds = []
    for round_result in result.rounds:
        solver_result = round_result.finite.solver_result
        diagnostics = solver_result.diagnostics
        if isinstance(diagnostics, SparseSQPResult):
            diagnostics = replace(
                diagnostics,
                continuation_state=(
                    diagnostics.continuation_state.checkpoint_safe_copy()
                ),
            )
            solver_result = replace(
                solver_result, diagnostics=diagnostics
            )
        finite = replace(
            round_result.finite, solver_result=solver_result
        )
        safe_rounds.append(replace(round_result, finite=finite))
    return replace(
        result, rounds=tuple(safe_rounds), continuation_state=safe_state
    )


@dataclass(frozen=True, slots=True)
class _CandidateResidual:
    inequality: float
    equality: float
    report: SeparationReport

    def finite_feasible(self, settings: ExchangeSettings) -> bool:
        return (
            self.inequality <= settings.finite_inequality_tolerance
            and self.equality <= settings.finite_equality_tolerance
        )

    def certified(self, settings: ExchangeSettings) -> bool:
        return self.report.certified(settings.separation.certificate_tolerance)


def _finite_residual(
    x: Array,
    corridor: Any,
    initial_state: Any,
    pool: ConstraintPool,
    endpoint_target: Any | None,
    equalities: Any | None,
    additional_inequalities: Any | None,
) -> tuple[float, float]:
    cache = KnotGeometryCache(initial_state, 0.0)
    cache.update(x)
    endpoint = (
        EndpointPoseEquality(cache, endpoint_target)
        if endpoint_target is not None
        else None
    )
    combined_equalities = stack_vector_constraints(endpoint, equalities)
    combined = CombinedConstraintOracle(
        RectangleConstraintOracle(corridor, pool, cache),
        combined_equalities,
        additional_inequalities,
    )
    c, _, h, _ = combined.evaluate(x)
    inequality = float(max(0.0, np.max(c, initial=-math.inf)))
    equality = float(np.max(np.abs(h), initial=0.0)) if h.size else 0.0
    return inequality, equality


def _candidate_residual(
    x: Array,
    corridor: Any,
    initial_state: Any,
    pool: ConstraintPool,
    endpoint_target: Any | None,
    equalities: Any | None,
    additional_inequalities: Any | None,
    settings: ExchangeSettings,
) -> _CandidateResidual:
    inequality, equality = _finite_residual(
        x,
        corridor,
        initial_state,
        pool,
        endpoint_target,
        equalities,
        additional_inequalities,
    )
    report = separate_rectangle_path(
        x, initial_state, corridor, settings=settings.separation
    )
    return _CandidateResidual(inequality, equality, report)


def _physical_to_core(
    x: Array,
    gradient: Array,
    state: SparseSQPContinuationState,
    coordinate_map: LogLengthKnotMap | None,
) -> tuple[Array, Array]:
    if state.scaling_shift is None or state.scaling_variable is None:
        raise RuntimeError("continuation state omitted scaling metadata")
    if coordinate_map is None:
        internal = x
        internal_gradient = gradient
    else:
        internal = coordinate_map.to_optimizer(x)
        internal_gradient = coordinate_map.pullback_gradient(internal, gradient)
    core_x = (internal - state.scaling_shift) / state.scaling_variable
    core_gradient = internal_gradient * state.scaling_variable
    return core_x, core_gradient


def _rebase_after_restoration(
    state: SparseSQPContinuationState,
    x: Array,
    objective: float,
    gradient: Array,
    *,
    displacement: float,
    generation: int,
    coordinate_map: LogLengthKnotMap | None,
    hessian_reset_displacement: float,
    hessian_scale: float,
) -> tuple[SparseSQPContinuationState, bool]:
    core_x, core_gradient = _physical_to_core(x, gradient, state, coordinate_map)
    preserve = displacement <= hessian_reset_displacement
    hessian = (
        state.hessian.copy()
        if preserve
        else np.eye(core_x.size, dtype=float) * hessian_scale
    )
    rebased = replace(
        state,
        current_x=core_x.copy(),
        current_objective=float(objective),
        current_gradient=core_gradient.copy(),
        current_inequalities=np.empty(0),
        current_inequality_jacobian=np.empty((0, core_x.size)),
        current_equalities=np.empty(0),
        current_equality_jacobian=np.empty((0, core_x.size)),
        best_feasible_x=core_x.copy(),
        best_feasible_objective=float(objective),
        best_feasible_gradient=core_gradient.copy(),
        best_feasible_violation=0.0,
        certified_x=core_x.copy(),
        certified_objective=float(objective),
        certified_gradient=core_gradient.copy(),
        hessian=hessian,
        filter_entries=(),
        warm_starts={},
        constraint_generation=generation,
        diagnostic_generation=-1,
        last_event="proximity_restoration",
        native_warm_starts_retained=False,
    )
    return rebased, preserve


def _proximity_objective(target: Array):
    scale = np.maximum(1.0, np.abs(target))

    def evaluate(x: Sequence[float]) -> tuple[float, Array]:
        delta = (np.asarray(x, dtype=float) - target) / scale
        return 0.5 * float(delta @ delta), delta / scale

    return evaluate


def _mark_core_certified(
    state: SparseSQPContinuationState,
    candidate: ExchangeResult,
    batch_result: ExchangeResult,
) -> SparseSQPContinuationState:
    """Record a certified candidate only when its scaled coordinates are known."""
    if candidate.objective >= state.certified_objective:
        return state
    if (
        batch_result.current_x is not None
        and np.array_equal(candidate.x, batch_result.current_x)
        and candidate.objective == batch_result.current_objective
    ):
        return replace(
            state,
            certified_x=state.current_x.copy(),
            certified_objective=float(candidate.objective),
            certified_gradient=state.current_gradient.copy(),
        )
    if (
        state.best_feasible_x is not None
        and state.best_feasible_gradient is not None
        and np.array_equal(candidate.x, batch_result.x)
        and candidate.objective == batch_result.objective
    ):
        return replace(
            state,
            certified_x=state.best_feasible_x.copy(),
            certified_objective=float(candidate.objective),
            certified_gradient=state.best_feasible_gradient.copy(),
        )
    return state
def _run_proximity_restoration(
    target: Array,
    original_objective: Any,
    corridor: Any,
    initial_state: Any,
    *,
    bounds: Any,
    endpoint_target: Any,
    equalities: Any,
    additional_inequalities: Any,
    pool: ConstraintPool,
    exchange_settings: ExchangeSettings,
    batch_settings: AdaptiveBatchSettings,
    coordinate_map: LogLengthKnotMap | None,
    coordinate_settings: KnotCoordinateSettings,
    generation: int,
    batch: int,
    trigger: str,
    maximum_sparse_retries: int | None = None,
) -> tuple[
    ExchangeResult | None,
    ConstraintPool,
    int,
    RestorationRecord,
    Array | None,
]:
    active_pool = pool
    trust = min(
        batch_settings.restoration_initial_trust_radius,
        exchange_settings.sparse_sqp.maximum_trust_radius,
    )
    last_inequality = math.inf
    last_equality = math.inf
    last_exact = math.inf
    maximum_displacement = math.inf
    start = target.copy()
    attempts: list[RestorationAttempt] = []
    sparse_retry_budget = (
        batch_settings.restoration_maximum_retries
        if maximum_sparse_retries is None
        else int(maximum_sparse_retries)
    )
    if sparse_retry_budget < 0:
        raise ValueError("maximum_sparse_retries must be nonnegative")
    strict_solver_tolerance = min(
        exchange_settings.sparse_sqp.feasibility_tolerance,
        exchange_settings.sparse_sqp.equality_tolerance,
        1.0e-2 * exchange_settings.finite_inequality_tolerance,
        1.0e-2 * exchange_settings.finite_equality_tolerance,
    )
    for retry in range(1, sparse_retry_budget + 1):
        sparse_settings = replace(
            exchange_settings.sparse_sqp,
            max_iterations=batch_settings.restoration_maximum_raw_iterations,
            max_accepted_steps=batch_settings.restoration_accepted_steps,
            initial_trust_radius=min(
                trust, exchange_settings.sparse_sqp.maximum_trust_radius
            ),
            feasibility_tolerance=strict_solver_tolerance,
            equality_tolerance=strict_solver_tolerance,
            minimum_trust_radius=min(1.0e-9, exchange_settings.sparse_sqp.minimum_trust_radius),
            minimum_predicted_reduction=min(1.0e-14, exchange_settings.sparse_sqp.minimum_predicted_reduction),
            final_restoration_iterations=(
                batch_settings.restoration_inner_final_restoration_iterations
            ),
            diagnostics=SparseSQPDiagnosticControls.restoration(),
            objective_soft_deadline_enabled=False,
            diagnostic_only=False,
        )
        one_round = replace(
            exchange_settings,
            maximum_rounds=1,
            certified_polish_rounds=0,
            final_kkt_diagnostics=False,
            require_solver_success=False,
            sparse_sqp=sparse_settings,
        )
        attempt_started = time.perf_counter()
        start_displacement = float(np.linalg.norm(start - target, ord=np.inf))
        restored = run_constraint_generation(
            start,
            _proximity_objective(target),
            corridor,
            initial_state,
            bounds=bounds,
            endpoint_target=endpoint_target,
            equalities=equalities,
            additional_inequalities=additional_inequalities,
            pool=active_pool,
            settings=one_round,
            coordinate_map=coordinate_map,
            coordinate_settings=coordinate_settings,
            constraint_generation=generation,
            objective_identity="proximity_restoration",
        )
        active_pool = restored.pool
        generation = restored.constraint_generation
        residual = _candidate_residual(
            restored.x,
            corridor,
            initial_state,
            active_pool,
            endpoint_target,
            equalities,
            additional_inequalities,
            exchange_settings,
        )
        last_inequality = residual.inequality
        last_equality = residual.equality
        last_exact = residual.report.worst_upper_bound
        maximum_displacement = float(
            np.linalg.norm(restored.x - target, ord=np.inf)
        )
        strict = residual.finite_feasible(exchange_settings) and residual.certified(
            exchange_settings
        )
        attempts.append(
            RestorationAttempt(
                retry=retry,
                method="sparse_sqp",
                wall_seconds=time.perf_counter() - attempt_started,
                trust_radius=float(trust),
                start_displacement=start_displacement,
                result_displacement=maximum_displacement,
                finite_inequality=last_inequality,
                equality_inf=last_equality,
                exact_upper_bound=last_exact,
                strict_certificate=bool(strict),
                solver_success=bool(restored.success),
                solver_message=str(restored.message),
                pool_size=active_pool.size,
                inner_final_restoration_iterations=(
                    batch_settings.restoration_inner_final_restoration_iterations
                ),
            )
        )
        if (
            strict
            and maximum_displacement
            <= batch_settings.restoration_maximum_displacement
        ):
            value, gradient_in = original_objective(restored.x)
            original_gradient = np.asarray(gradient_in, dtype=float).copy()
            accepted = replace(
                restored,
                objective=float(value),
                success=True,
                message="continuously certified proximity restoration",
                final_separation=residual.report,
                final_finite_inequality=residual.inequality,
                final_equality_inf=residual.equality,
            )
            return (
                accepted,
                active_pool,
                generation,
                RestorationRecord(
                    batch,
                    True,
                    trigger,
                    retry,
                    True,
                    maximum_displacement,
                    last_inequality,
                    last_equality,
                    last_exact,
                    True,
                    "strict restoration certificate passed",
                    tuple(attempts),
                ),
                original_gradient,
            )
        # A contracted retry refines the previous near-feasible point while the
        # objective continues to measure displacement from the promising target.
        start = restored.x.copy()
        trust *= batch_settings.restoration_trust_contraction

    if batch_settings.restoration_slsqp_fallback:
        # Sparse restoration can stop at a few e-9 on highly dependent route
        # constraints.  A bounded SLSQP proximity solve is a last-resort primal
        # refiner only: it uses the cheap proximity objective, skips KKT
        # diagnostics, and cannot replace the safe incumbent without the same
        # independent finite and exact certificates.
        slsqp_settings = replace(
            exchange_settings.slsqp,
            max_iterations=batch_settings.restoration_slsqp_max_iterations,
            ftol=min(exchange_settings.slsqp.ftol, 1.0e-12),
        )
        fallback_settings = replace(
            exchange_settings,
            maximum_rounds=batch_settings.restoration_slsqp_exchange_rounds,
            certified_polish_rounds=0,
            finite_solver="slsqp",
            slsqp=slsqp_settings,
            final_kkt_diagnostics=False,
            require_solver_success=False,
        )
        attempt_started = time.perf_counter()
        start_displacement = float(np.linalg.norm(start - target, ord=np.inf))
        restored = run_constraint_generation(
            start,
            _proximity_objective(target),
            corridor,
            initial_state,
            bounds=bounds,
            endpoint_target=endpoint_target,
            equalities=equalities,
            additional_inequalities=additional_inequalities,
            pool=active_pool,
            settings=fallback_settings,
            coordinate_map=coordinate_map,
            coordinate_settings=coordinate_settings,
            constraint_generation=generation,
            objective_identity="proximity_restoration_slsqp_fallback",
        )
        active_pool = restored.pool
        generation = restored.constraint_generation
        residual = _candidate_residual(
            restored.x,
            corridor,
            initial_state,
            active_pool,
            endpoint_target,
            equalities,
            additional_inequalities,
            exchange_settings,
        )
        last_inequality = residual.inequality
        last_equality = residual.equality
        last_exact = residual.report.worst_upper_bound
        maximum_displacement = float(
            np.linalg.norm(restored.x - target, ord=np.inf)
        )
        strict = residual.finite_feasible(exchange_settings) and residual.certified(
            exchange_settings
        )
        attempts.append(
            RestorationAttempt(
                retry=sparse_retry_budget + 1,
                method="slsqp_fallback",
                wall_seconds=time.perf_counter() - attempt_started,
                trust_radius=float(trust),
                start_displacement=start_displacement,
                result_displacement=maximum_displacement,
                finite_inequality=last_inequality,
                equality_inf=last_equality,
                exact_upper_bound=last_exact,
                strict_certificate=bool(strict),
                solver_success=bool(restored.success),
                solver_message=str(restored.message),
                pool_size=active_pool.size,
                inner_final_restoration_iterations=0,
            )
        )
        if (
            strict
            and maximum_displacement
            <= batch_settings.restoration_maximum_displacement
        ):
            value, gradient_in = original_objective(restored.x)
            original_gradient = np.asarray(gradient_in, dtype=float).copy()
            accepted = replace(
                restored,
                objective=float(value),
                success=True,
                message="continuously certified SLSQP proximity fallback",
                final_separation=residual.report,
                final_finite_inequality=residual.inequality,
                final_equality_inf=residual.equality,
            )
            return (
                accepted,
                active_pool,
                generation,
                RestorationRecord(
                    batch, True, trigger,
                    sparse_retry_budget + 1,
                    True, maximum_displacement, last_inequality,
                    last_equality, last_exact, True,
                    "strict SLSQP fallback certificate passed",
                    tuple(attempts),
                ),
                original_gradient,
            )
    return (
        None,
        active_pool,
        generation,
        RestorationRecord(
            batch,
            True,
            trigger,
            sparse_retry_budget,
            False,
            maximum_displacement,
            last_inequality,
            last_equality,
            last_exact,
            False,
            "restoration retries exhausted after contracted-trust attempts",
            tuple(attempts),
        ),
        None,
    )


def _raw_sparse_result(exchange: ExchangeResult) -> SparseSQPResult | None:
    if not exchange.rounds:
        return None
    raw = exchange.rounds[-1].finite.solver_result.diagnostics
    return raw if isinstance(raw, SparseSQPResult) else None


def _initial_safe_result(
    x: Array,
    objective: Any,
    corridor: Any,
    initial_state: Any,
    pool: ConstraintPool,
    endpoint_target: Any,
    equalities: Any,
    additional_inequalities: Any,
    settings: ExchangeSettings,
    continuation_state: SparseSQPContinuationState | None,
    generation: int,
) -> ExchangeResult | None:
    residual = _candidate_residual(
        x,
        corridor,
        initial_state,
        pool,
        endpoint_target,
        equalities,
        additional_inequalities,
        settings,
    )
    if not residual.finite_feasible(settings) or not residual.certified(settings):
        return None
    value, _ = objective(x)
    return ExchangeResult(
        x.copy(),
        float(value),
        False,
        "initial continuously certified fallback",
        pool.copy(),
        (),
        residual.report,
        residual.inequality,
        residual.equality,
        KKTReport.not_computed(),
        0.0,
        x.copy(),
        float(value),
        continuation_state,
        generation,
    )

def run_adaptive_sparse_polish(
    x0: Sequence[float],
    objective: Any,
    corridor: Any,
    initial_state: Any,
    *,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    endpoint_target: Any | None = None,
    equalities: Any | None = None,
    additional_inequalities: Any | None = None,
    pool: ConstraintPool | None = None,
    exchange_settings: ExchangeSettings,
    batch_settings: AdaptiveBatchSettings = AdaptiveBatchSettings(),
    coordinate_map: LogLengthKnotMap | None = None,
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
    continuation_state: SparseSQPContinuationState | None = None,
    checkpoint_callback: Callable[[AdaptiveSafeCheckpoint], None] | None = None,
) -> AdaptivePolishResult:
    """Run accepted-step batches while exposing only a certified incumbent."""
    if exchange_settings.finite_solver != "sparse_sqp":
        raise ValueError("adaptive polish requires finite_solver='sparse_sqp'")

    started = time.perf_counter()
    x = np.asarray(x0, dtype=float).copy()
    active_pool = pool.copy() if pool is not None else ConstraintPool.seeded(corridor)
    generation = (
        continuation_state.constraint_generation
        if continuation_state is not None
        else 0
    )
    core_state = continuation_state
    best_certified = _initial_safe_result(
        x,
        objective,
        corridor,
        initial_state,
        active_pool,
        endpoint_target,
        equalities,
        additional_inequalities,
        exchange_settings,
        core_state,
        generation,
    )
    last_result: ExchangeResult | None = None
    records: list[AdaptiveBatchRecord] = []
    provenance: list[CutProvenance] = []
    restorations: list[RestorationRecord] = []
    total_objective_evaluations = 0
    stalled_certified = 0
    previous_certified_objective = (
        math.inf
        if best_certified is None
        else float(best_certified.objective)
    )
    stop_reason = "maximum batches reached"
    if best_certified is not None:
        _emit_safe_checkpoint(
            checkpoint_callback,
            AdaptiveSafeCheckpoint(
                best_certified,
                core_state,
                (),
                (),
                (),
                generation,
            ),
        )

    for batch in range(1, batch_settings.maximum_batches + 1):
        batch_started = time.perf_counter()
        resolved_policy = _resolve_feasibility_first_policy(
            batch_settings,
            pool_size=active_pool.size,
            variable_count=x.size,
        )
        compute_batch_criticality = (
            batch_settings.criticality_schedule == "batch"
            or (
                batch_settings.criticality_schedule == "stagnation"
                and stalled_certified > 0
            )
        )
        diagnostic_controls = (
            replace(
                SparseSQPDiagnosticControls(),
                final_criticality=compute_batch_criticality,
            )
            if batch_settings.intermediate_multiplier_diagnostics
            else SparseSQPDiagnosticControls.intermediate(
                criticality=compute_batch_criticality,
                accepted_step_criticality=(
                    batch_settings.criticality_schedule == "accepted_step"
                ),
            )
        )
        restoration_seconds = 0.0
        finite_settings = replace(
            exchange_settings.sparse_sqp,
            max_iterations=batch_settings.maximum_raw_iterations_per_batch,
            max_accepted_steps=resolved_policy.batch_iterations,
            minimum_predicted_reduction=max(
                exchange_settings.sparse_sqp.minimum_predicted_reduction,
                batch_settings.batch_minimum_predicted_reduction,
            ),
            final_restoration_iterations=0,
            diagnostics=diagnostic_controls,
            objective_soft_deadline_enabled=(
                batch_settings.objective_soft_deadline_enabled
            ),
            diagnostic_only=False,
        )
        one_round = replace(
            exchange_settings,
            maximum_rounds=1,
            certified_polish_rounds=0,
            final_kkt_diagnostics=False,
            require_solver_success=False,
            sparse_sqp=finite_settings,
        )
        finite_round_started = time.perf_counter()
        result = run_constraint_generation(
            x,
            objective,
            corridor,
            initial_state,
            bounds=bounds,
            endpoint_target=endpoint_target,
            equalities=equalities,
            additional_inequalities=additional_inequalities,
            pool=active_pool,
            settings=one_round,
            coordinate_map=coordinate_map,
            coordinate_settings=coordinate_settings,
            continuation_state=core_state,
            constraint_generation=generation,
            objective_identity="adaptive_original_objective",
        )
        last_result = result
        finite_round_seconds = time.perf_counter() - finite_round_started
        active_pool = result.pool
        generation = result.constraint_generation
        core_state = result.continuation_state
        raw = _raw_sparse_result(result)
        if raw is None or core_state is None or result.current_x is None:
            raise RuntimeError("adaptive sparse batch omitted continuation diagnostics")
        total_objective_evaluations += int(raw.objective_evaluations)

        finite_cuts = result.rounds[-1].cuts_added
        if finite_cuts:
            provenance.append(
                CutProvenance(
                    batch,
                    "finite_feasible_incumbent",
                    finite_cuts,
                    active_pool.size,
                )
            )

        current_separation_started = time.perf_counter()
        current_report = separate_rectangle_path(
            result.current_x,
            initial_state,
            corridor,
            settings=exchange_settings.separation,
        )
        current_cache = KnotGeometryCache(initial_state, 0.0)
        current_cache.update(result.current_x)
        current_cuts = add_report_violations(
            active_pool,
            current_report,
            current_cache.path.lengths,
            merge_distance=exchange_settings.merge_distance,
        )
        if current_cuts:
            generation += 1
            provenance.append(
                CutProvenance(
                    batch,
                    "current_filter_iterate",
                    current_cuts,
                    active_pool.size,
                )
            )

        current_separation_seconds = time.perf_counter() - current_separation_started
        candidate_recheck_started = time.perf_counter()
        finite_residual = _candidate_residual(
            result.x,
            corridor,
            initial_state,
            active_pool,
            endpoint_target,
            equalities,
            additional_inequalities,
            exchange_settings,
        )
        current_inequality, current_equality = _finite_residual(
            result.current_x,
            corridor,
            initial_state,
            active_pool,
            endpoint_target,
            equalities,
            additional_inequalities,
        )
        candidate_recheck_seconds = time.perf_counter() - candidate_recheck_started
        finite_certified = (
            finite_residual.finite_feasible(exchange_settings)
            and finite_residual.certified(exchange_settings)
        )
        current_certified = (
            current_inequality <= exchange_settings.finite_inequality_tolerance
            and current_equality <= exchange_settings.finite_equality_tolerance
            and current_report.certified(
                exchange_settings.separation.certificate_tolerance
            )
        )
        certified_candidates: list[ExchangeResult] = []
        if finite_certified:
            certified_candidates.append(
                replace(
                    result,
                    final_separation=finite_residual.report,
                    final_finite_inequality=finite_residual.inequality,
                    final_equality_inf=finite_residual.equality,
                )
            )
        if current_certified:
            certified_candidates.append(
                replace(
                    result,
                    x=result.current_x.copy(),
                    objective=float(result.current_objective),
                    final_separation=current_report,
                    final_finite_inequality=current_inequality,
                    final_equality_inf=current_equality,
                )
            )
        for candidate in certified_candidates:
            if best_certified is None or candidate.objective < best_certified.objective:
                best_certified = candidate

        new_cuts = finite_cuts + current_cuts
        for candidate in certified_candidates:
            core_state = _mark_core_certified(core_state, candidate, result)
        progress_threshold = max(
            batch_settings.absolute_objective_tolerance,
            batch_settings.relative_objective_tolerance
            * max(
                1.0,
                abs(result.current_objective)
                if best_certified is None
                else abs(best_certified.objective),
            ),
        )
        promising_current = (
            best_certified is None
            or result.current_objective
            < best_certified.objective - progress_threshold
        )
        current_total_violation = max(current_inequality, current_equality)
        trigger = ""
        if new_cuts and promising_current:
            trigger = "new exact cuts invalidated promising progress"
        certified_stalled_this_batch = bool(
            best_certified is not None
            and math.isfinite(previous_certified_objective)
            and previous_certified_objective - best_certified.objective
            <= progress_threshold
        )
        if not trigger and certified_stalled_this_batch and promising_current:
            trigger = "certified objective stalled for one batch"

        continuation_work_estimate = int(active_pool.size * result.current_x.size)
        continuation_extended_eligible = bool(
            batch_settings.continuation_extended_enabled
            and continuation_work_estimate
            <= batch_settings.continuation_extended_maximum_work
        )
        continuation_effective_cap = (
            batch_settings.continuation_extended_violation_cap
            if continuation_extended_eligible
            else batch_settings.continuation_violation_cap
        )
        if current_total_violation <= batch_settings.continuation_violation_cap:
            continuation_decision = "base_cap"
        elif (
            continuation_extended_eligible
            and current_total_violation <= continuation_effective_cap
        ):
            continuation_decision = "extended_cost_screen_passed"
        elif continuation_extended_eligible:
            continuation_decision = "violation_above_extended_cap"
        else:
            continuation_decision = "extended_cost_screen_rejected"
        restoration_attempted = bool(
            batch_settings.restoration_enabled
            and trigger
            and current_total_violation <= continuation_effective_cap
        )
        restoration_succeeded = False
        if restoration_attempted:
            restoration_started = time.perf_counter()
            (
                restored,
                active_pool,
                generation,
                restoration_record,
                restored_gradient,
            ) = _run_proximity_restoration(
                result.current_x,
                objective,
                corridor,
                initial_state,
                bounds=bounds,
                endpoint_target=endpoint_target,
                equalities=equalities,
                additional_inequalities=additional_inequalities,
                pool=active_pool,
                exchange_settings=exchange_settings,
                batch_settings=batch_settings,
                coordinate_map=coordinate_map,
                coordinate_settings=coordinate_settings,
                generation=generation,
                batch=batch,
                trigger=trigger,
                maximum_sparse_retries=resolved_policy.restoration_sparse_retries,
            )
            restorations.append(restoration_record)
            restoration_seconds = time.perf_counter() - restoration_started
            restoration_succeeded = restored is not None
            if restored is not None:
                assert restored_gradient is not None
                provenance.append(
                    CutProvenance(
                        batch,
                        "restoration",
                        max(0, restored.pool.size - result.pool.size),
                        active_pool.size,
                    )
                )
                last_result = restored
                if (
                    best_certified is None
                    or restored.objective < best_certified.objective
                ):
                    best_certified = restored
                displacement = float(
                    np.linalg.norm(restored.x - result.current_x, ord=np.inf)
                )
                core_state, preserved = _rebase_after_restoration(
                    core_state,
                    restored.x,
                    restored.objective,
                    restored_gradient,
                    displacement=displacement,
                    generation=generation,
                    coordinate_map=coordinate_map,
                    hessian_reset_displacement=(
                        batch_settings.hessian_reset_displacement
                    ),
                    hessian_scale=finite_settings.initial_hessian_scale,
                )
                restorations[-1] = replace(
                    restorations[-1], hessian_preserved=preserved
                )
                x = restored.x.copy()

        if not restoration_succeeded:
            if (
                current_total_violation <= continuation_effective_cap
            ):
                x = result.current_x.copy()
            else:
                x = (
                    best_certified.x.copy()
                    if best_certified is not None
                    else result.x.copy()
                )
                core_state = None

        improvement = 0.0
        if best_certified is not None:
            if math.isfinite(previous_certified_objective):
                improvement = (
                    previous_certified_objective - best_certified.objective
                )
            threshold = max(
                batch_settings.absolute_objective_tolerance,
                batch_settings.relative_objective_tolerance
                * max(1.0, abs(best_certified.objective)),
            )
            if new_cuts == 0 and improvement <= threshold:
                stalled_certified += 1
            elif improvement > threshold:
                stalled_certified = 0
            previous_certified_objective = min(
                previous_certified_objective, best_certified.objective
            )

        elapsed = time.perf_counter() - started
        current_stop: str | None = None
        if batch >= batch_settings.minimum_batches:
            if (
                best_certified is not None
                and stalled_certified
                >= batch_settings.required_stalled_certified_batches
            ):
                current_stop = "certified objective progress stalled"
            elif raw.accepted_steps == 0 and best_certified is not None:
                current_stop = "certified batch accepted no steps"
        if elapsed >= batch_settings.maximum_wall_seconds:
            current_stop = "wall-clock budget reached"
        if (
            total_objective_evaluations
            >= batch_settings.maximum_objective_evaluations
        ):
            current_stop = "objective-evaluation budget reached"

        first_iteration = raw.history[0] if raw.history else None
        first_accepted_iteration = next(
            (entry for entry in raw.history if entry.accepted), None
        )
        records.append(
            AdaptiveBatchRecord(
                batch=batch,
                wall_seconds=time.perf_counter() - batch_started,
                cumulative_seconds=elapsed,
                finite_round_seconds=finite_round_seconds,
                current_separation_seconds=current_separation_seconds,
                candidate_recheck_seconds=candidate_recheck_seconds,
                restoration_seconds=restoration_seconds,
                objective_seconds=raw.objective_seconds,
                constraint_seconds=raw.constraint_seconds,
                subproblem_seconds=raw.linear_program_seconds,
                maximum_objective_seconds=dict(raw.stage_telemetry.maxima).get("objective", 0.0),
                objective=(
                    math.inf
                    if best_certified is None
                    else float(best_certified.objective)
                ),
                current_objective=float(result.current_objective),
                current_finite_inequality=float(current_inequality),
                current_equality_inf=float(current_equality),
                current_exact_worst_upper_bound=float(
                    current_report.worst_upper_bound
                ),
                continuously_certified=best_certified is not None,
                cuts_added=new_cuts,
                cuts_from_finite=finite_cuts,
                cuts_from_current=current_cuts,
                pool_size=active_pool.size,
                objective_evaluations=raw.objective_evaluations,
                accepted_steps=raw.accepted_steps,
                first_iteration_objective=(
                    math.nan if first_iteration is None else float(first_iteration.objective)
                ),
                first_iteration_violation=(
                    math.nan if first_iteration is None else float(first_iteration.violation)
                ),
                first_iteration_accepted=(
                    False if first_iteration is None else bool(first_iteration.accepted)
                ),
                first_iteration_step_kind=(
                    "none" if first_iteration is None else first_iteration.step_kind
                ),
                first_iteration_step_norm_inf=(
                    0.0 if first_iteration is None else float(first_iteration.step_norm_inf)
                ),
                first_accepted_step_index=(
                    None
                    if first_accepted_iteration is None
                    else int(first_accepted_iteration.index)
                ),
                first_accepted_step_objective=(
                    None
                    if first_accepted_iteration is None
                    else float(first_accepted_iteration.objective)
                ),
                raw_iterations=raw.iterations,
                objective_timeouts=raw.objective_timeouts,
                rejected_trials=raw.rejected_steps,
                normalized_stationarity=raw.kkt_stationarity_scaled,
                normalized_complementarity=raw.kkt_complementarity_scaled,
                criticality=raw.kkt_criticality,
                criticality_success=raw.kkt_criticality_success,
                multiplier_calls=raw.multiplier_diagnostic_calls,
                multiplier_seconds=raw.multiplier_diagnostic_seconds,
                criticality_calls=raw.criticality_diagnostic_calls,
                criticality_seconds=raw.criticality_diagnostic_seconds,
                objective_improvement=float(improvement),
                restoration_attempted=restoration_attempted,
                restoration_succeeded=restoration_succeeded,
                continuation_effective_cap=float(continuation_effective_cap),
                continuation_extended_eligible=continuation_extended_eligible,
                continuation_work_estimate=continuation_work_estimate,
                continuation_decision=continuation_decision,
                prebatch_work_estimate=resolved_policy.work_estimate,
                effective_batch_iterations=resolved_policy.batch_iterations,
                restoration_sparse_retry_budget=resolved_policy.restoration_sparse_retries,
                feasibility_first_active=resolved_policy.active,
                stop_reason=current_stop,
            )
        )
        if best_certified is not None:
            _emit_safe_checkpoint(
                checkpoint_callback,
                AdaptiveSafeCheckpoint(
                    best_certified,
                    core_state,
                    tuple(records),
                    tuple(provenance),
                    tuple(restorations),
                    generation,
                ),
            )
        if current_stop is not None:
            stop_reason = current_stop
            break

    if last_result is None:
        raise AssertionError("adaptive polish did not execute")
    if best_certified is None:
        raise RuntimeError(
            "no continuously certified incumbent was available; refusing to "
            "expose an uncertified filter iterate"
        )

    # Diagnostic-only means no SQP step and no restoration. The final safe
    # point receives exactly one multiplier estimate and one two-radius
    # robust-criticality evaluation.
    final_controls = SparseSQPDiagnosticControls(
        multipliers_on_initial_point=False,
        multipliers_on_accepted_steps=False,
        multiplier_termination=False,
        final_multipliers=True,
        final_criticality=True,
        strict_convergence_requires_all=True,
    )
    final_sparse = replace(
        exchange_settings.sparse_sqp,
        max_iterations=1,
        max_accepted_steps=None,
        final_restoration_iterations=0,
        diagnostics=final_controls,
        diagnostic_only=True,
        # The exchange tolerances define planner-safe finite feasibility.  A
        # diagnostic-only call must not suppress KKT work merely because its
        # core defaults are tighter than the already certified route contract.
        feasibility_tolerance=max(
            exchange_settings.sparse_sqp.feasibility_tolerance,
            exchange_settings.finite_inequality_tolerance,
        ),
        equality_tolerance=max(
            exchange_settings.sparse_sqp.equality_tolerance,
            exchange_settings.finite_equality_tolerance,
        ),
        objective_soft_deadline_enabled=False,
    )
    final_exchange = replace(
        exchange_settings,
        maximum_rounds=1,
        certified_polish_rounds=0,
        final_kkt_diagnostics=False,
        require_solver_success=False,
        sparse_sqp=final_sparse,
    )
    diagnosed = run_constraint_generation(
        best_certified.x,
        objective,
        corridor,
        initial_state,
        bounds=bounds,
        endpoint_target=endpoint_target,
        equalities=equalities,
        additional_inequalities=additional_inequalities,
        pool=active_pool,
        settings=final_exchange,
        coordinate_map=coordinate_map,
        coordinate_settings=coordinate_settings,
        constraint_generation=generation,
        objective_identity="adaptive_original_objective",
    )
    final_raw = _raw_sparse_result(diagnosed)
    final_complete = bool(
        final_raw is not None and final_raw.diagnostics_complete
    )
    final_failure = None if final_raw is None else final_raw.diagnostic_failure
    diagnosed_safe = bool(
        diagnosed.final_separation.certified(
            exchange_settings.separation.certificate_tolerance
        )
        and diagnosed.final_finite_inequality
        <= exchange_settings.finite_inequality_tolerance
        and diagnosed.final_equality_inf
        <= exchange_settings.finite_equality_tolerance
    )
    if diagnosed_safe:
        best_certified = replace(
            diagnosed,
            success=final_complete,
            message=(
                "continuously certified with complete sparse-SQP diagnostics"
                if final_complete
                else "continuously certified; final diagnostics incomplete"
            ),
        )
    _emit_safe_checkpoint(
        checkpoint_callback,
        AdaptiveSafeCheckpoint(
            best_certified,
            core_state,
            tuple(records),
            tuple(provenance),
            tuple(restorations),
            generation,
        ),
    )


    return AdaptivePolishResult(
        result=best_certified,
        best_certified_result=best_certified,
        batches=tuple(records),
        stop_reason=stop_reason,
        total_objective_evaluations=total_objective_evaluations,
        wall_seconds=time.perf_counter() - started,
        continuation_state=core_state,
        cut_provenance=tuple(provenance),
        restorations=tuple(restorations),
        final_diagnostics_complete=final_complete,
        final_diagnostic_failure=final_failure,
    )


@dataclass(slots=True)
class _StageReportingObjective:
    objective: Any
    context: WorkerContext

    def __call__(self, x: Sequence[float]) -> tuple[float, Array]:
        self.context.set_stage("objective")
        value, gradient = self.objective(x)
        self.context.set_stage("sparse_sqp_solver")
        return value, np.asarray(gradient, dtype=float)

    def evaluate_with_deadline(
        self, x: Sequence[float], *, deadline: float
    ) -> tuple[float, Array]:
        self.context.set_stage("objective")
        evaluator = getattr(self.objective, "evaluate_with_deadline", None)
        if evaluator is None:
            value, gradient = self.objective(x)
        else:
            value, gradient = evaluator(x, deadline=deadline)
        if time.monotonic() > deadline:
            raise ObjectiveEvaluationTimeout(
                "objective", "objective exceeded its cooperative deadline"
            )
        self.context.set_stage("sparse_sqp_solver")
        return value, np.asarray(gradient, dtype=float)


def _adaptive_worker(payload: Any, context: WorkerContext) -> AdaptivePolishResult:
    if not isinstance(payload, dict):
        raise TypeError("adaptive worker payload must be a dictionary")
    arguments = dict(payload)
    objective = arguments.pop("objective")
    arguments["objective"] = _StageReportingObjective(objective, context)
    arguments["checkpoint_callback"] = context.emit_checkpoint
    context.set_stage("initial_continuous_certification")
    return run_adaptive_sparse_polish(**arguments)


def _valid_adaptive_result(value: Any) -> bool:
    return isinstance(value, AdaptivePolishResult)


def run_supervised_adaptive_sparse_polish(
    x0: Sequence[float],
    objective: Any,
    corridor: Any,
    initial_state: Any,
    *,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    endpoint_target: Any | None = None,
    equalities: Any | None = None,
    additional_inequalities: Any | None = None,
    pool: ConstraintPool | None = None,
    exchange_settings: ExchangeSettings,
    batch_settings: AdaptiveBatchSettings = AdaptiveBatchSettings(),
    coordinate_map: LogLengthKnotMap | None = None,
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
    continuation_state: SparseSQPContinuationState | None = None,
    supervisor_settings: SupervisorSettings = SupervisorSettings(),
    supervisor_service_settings: SupervisorServiceClientSettings | None = None,
    initial_safe_checkpoint: AdaptiveSafeCheckpoint | None = None,
) -> SupervisedAdaptivePolishResult:
    """Run adaptive polishing behind a killable, checkpointed process boundary.

    The hard deadline is refreshed only when the child emits a continuously
    certified checkpoint. Thus every individual batch and final diagnostic
    phase is bounded while a persistent child can retain native LP bases.
    """
    payload = {
        "x0": np.asarray(x0, dtype=float).copy(),
        "objective": objective,
        "corridor": corridor,
        "initial_state": initial_state,
        "bounds": bounds,
        "endpoint_target": endpoint_target,
        "equalities": equalities,
        "additional_inequalities": additional_inequalities,
        "pool": pool,
        "exchange_settings": exchange_settings,
        "batch_settings": batch_settings,
        "coordinate_map": coordinate_map,
        "coordinate_settings": coordinate_settings,
        "continuation_state": continuation_state,
    }
    runner_settings = replace(
        supervisor_settings, deadline_refresh_on_checkpoint=True
    )
    if supervisor_service_settings is None:
        with SupervisedBatchRunner(
            _adaptive_worker,
            settings=runner_settings,
            result_validator=_valid_adaptive_result,
        ) as runner:
            outcome = runner.run(
                payload, initial_checkpoint=initial_safe_checkpoint
            )
    else:
        client = ExternalSupervisorServiceClient(supervisor_service_settings)
        outcome = client.run(
            _adaptive_worker,
            payload,
            worker_settings=runner_settings,
            initial_checkpoint=initial_safe_checkpoint,
            result_validator=_valid_adaptive_result,
        )
    completed = (
        outcome.result
        if outcome.success and isinstance(outcome.result, AdaptivePolishResult)
        else None
    )
    checkpoint = (
        outcome.last_checkpoint
        if isinstance(outcome.last_checkpoint, AdaptiveSafeCheckpoint)
        else None
    )
    safe_result = (
        completed.result
        if completed is not None
        else (None if checkpoint is None else checkpoint.result)
    )
    return SupervisedAdaptivePolishResult(
        outcome,
        completed,
        checkpoint,
        safe_result,
    )


__all__ = [
    "AdaptiveBatchRecord",
    "AdaptiveBatchSettings",
    "AdaptivePolishResult",
    "AdaptiveSafeCheckpoint",
    "CutProvenance",
    "RestorationRecord",
    "SupervisedAdaptivePolishResult",
    "run_adaptive_sparse_polish",
    "run_supervised_adaptive_sparse_polish",
]
