"""Externally bounded warm-start scheduling for route optimization.

V10 promotes this scheduler into the integrated planner architecture while
retaining the V9 inline schedule as an explicit compatibility mode.  It handles
ordinary routes and topology-quotient classes, preserves parent-owned strict
certificates across stage failures, and supports analytic-first strict-Phase-I
fallback without changing the low-level SLSQP optimizer default.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import (
    BoxBounds,
    ConstraintPool,
    CurvatureSlopeConstraint,
    ExchangeSettings,
    KnotCoordinateSettings,
    LogLengthKnotMap,
    PathOptimizerSettings,
    PhaseOneSettings,
    SLSQPSettings,
    SparseSQPDiagnosticControls,
    SparseSQPSettings,
    SupervisorSettings,
    SupervisedBatchOutcome,
    SupervisedBatchRunner,
    WorkerContext,
    align_endpoint_target_to_path,
    compile_geometry_path,
    knot_parameters_to_raw,
    optimize_path,
    path_length_value_and_raw_gradient,
    project_path_to_feasibility,
    reverse_solver,
    scalar_reverse_solver,
    run_supervised_phase_one,
    separate_rectangle_path,
)

from .maze_routes import RouteOptimizationProblem, stable_knot_bounds

Array = NDArray[np.float64]


class WarmStartSchedulingMode(str, Enum):
    LEGACY_ALWAYS_BOTH = "legacy_always_both"
    BOUNDED_BEST_OF_BOTH = "bounded_best_of_both"
    LENGTH_FIRST_ESCALATION = "length_first_escalation"
    SHADOW_COMPARE = "shadow_compare"


class WarmStartInitializerMode(str, Enum):
    ANALYTIC_INITIALIZER = "analytic_initializer"
    STRICT_PHASE_ONE_INITIALIZER = "strict_phase_one_initializer"
    ANALYTIC_WITH_STRICT_PHASE_ONE_FALLBACK = "analytic_with_strict_phase_one_fallback"


class PrimaryTimeBackendMode(str, Enum):
    """Planner-level backend selection for the final time-polish stage only."""

    AUTO_QUALIFIED_FILTER_SQP = "auto_qualified_filter_sqp"
    FORCE_SLSQP = "force_slsqp"


@dataclass(frozen=True, slots=True)
class PrimaryFilterSQPPolicySettings:
    """Empirically validated operating envelope for filter-SQP primary polish.

    These limits are deployment qualification gates, not mathematical limits of
    the filter-SQP algorithm.  Broadening them requires a new route-corpus
    qualification campaign.
    """

    # Low-level/direct scheduler callers retain the historical SLSQP behavior.
    # The integrated production planner opts into AUTO explicitly.
    mode: PrimaryTimeBackendMode = PrimaryTimeBackendMode.FORCE_SLSQP
    maximum_variables: int = 22
    maximum_segments: int = 11
    maximum_internal_caps: int = 2

    def __post_init__(self) -> None:
        if self.maximum_variables <= 0:
            raise ValueError("maximum_variables must be positive")
        if self.maximum_segments <= 0:
            raise ValueError("maximum_segments must be positive")
        if self.maximum_internal_caps < 0:
            raise ValueError("maximum_internal_caps must be nonnegative")


@dataclass(frozen=True, slots=True)
class WarmStartDeadlineSettings:
    phase_one_seconds: float = 120.0
    primary_geometry_seconds: float = 120.0
    projection_seconds: float = 90.0
    alternate_geometry_seconds: float = 120.0
    exact_ranking_seconds: float = 90.0
    full_time_seconds: float = 300.0
    total_pipeline_seconds: float = 600.0
    worker_startup_seconds: float = 15.0
    terminate_grace_seconds: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class TimePolishCheckpointSettings:
    """Throttling for continuously certified SLSQP major-iterate checkpoints.

    A cheap continuous-feasibility certificate is attempted whenever the
    deterministic major-iteration cadence fires; the maximum-wall trigger is a
    secondary safeguard for unusually slow iterations.  Faster execution must
    not suppress checkpoints that can affect planner-facing recovery. Strict
    improving iterates are published directly as planner-safe
    checkpoints. Near-feasible improving iterates may also be retained as
    numerical checkpoints; they are never planner-facing and are projected and
    independently certified only after the live time-optimizer worker stops.
    """

    enabled: bool = True
    every_major_iterations: int = 1
    minimum_seconds_between_checks: float = 0.5
    maximum_seconds_between_checks: float = 5.0
    minimum_exact_time_improvement: float = 1.0e-9
    numerical_violation_cap: float = 1.0e-3
    maximum_projection_attempts: int = 3

    def __post_init__(self) -> None:
        if self.every_major_iterations <= 0:
            raise ValueError("every_major_iterations must be positive")
        if (
            not math.isfinite(self.minimum_seconds_between_checks)
            or self.minimum_seconds_between_checks < 0.0
        ):
            raise ValueError("minimum_seconds_between_checks must be finite and nonnegative")
        if (
            not math.isfinite(self.maximum_seconds_between_checks)
            or self.maximum_seconds_between_checks <= 0.0
            or self.maximum_seconds_between_checks < self.minimum_seconds_between_checks
        ):
            raise ValueError("maximum_seconds_between_checks must be finite and >= minimum")
        if (
            not math.isfinite(self.minimum_exact_time_improvement)
            or self.minimum_exact_time_improvement < 0.0
        ):
            raise ValueError("minimum_exact_time_improvement must be finite and nonnegative")
        if (
            not math.isfinite(self.numerical_violation_cap)
            or self.numerical_violation_cap <= 0.0
        ):
            raise ValueError("numerical_violation_cap must be finite and positive")
        if self.maximum_projection_attempts < 0:
            raise ValueError("maximum_projection_attempts must be nonnegative")


def _time_checkpoint_due(
    policy: TimePolishCheckpointSettings,
    *,
    since_iterations: int,
    since_seconds: float,
) -> bool:
    """Return whether a major-iterate checkpoint should be inspected.

    The iteration cadence is authoritative because retained checkpoints can
    change planner output after a supervised failure.  Wall time is only a
    secondary trigger for unusually slow iterations; faster execution must not
    suppress an iteration selected by ``every_major_iterations``.
    """
    return bool(
        since_iterations >= policy.every_major_iterations
        or since_seconds >= policy.maximum_seconds_between_checks
    )


@dataclass(frozen=True, slots=True)
class WarmStartScheduleSettings:
    mode: WarmStartSchedulingMode = WarmStartSchedulingMode.LEGACY_ALWAYS_BOTH
    initializer_mode: WarmStartInitializerMode = WarmStartInitializerMode.ANALYTIC_INITIALIZER
    deadlines: WarmStartDeadlineSettings = WarmStartDeadlineSettings()
    phase_one: PhaseOneSettings = PhaseOneSettings()
    time_checkpoint: TimePolishCheckpointSettings = TimePolishCheckpointSettings()
    primary_filter_sqp: PrimaryFilterSQPPolicySettings = PrimaryFilterSQPPolicySettings()
    ranking_time_tolerance: float = 1.0e-10
    telemetry_jsonl: str | None = None
    telemetry_test_injection: Literal["none", "raise"] = "none"
    test_injections: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.ranking_time_tolerance) or self.ranking_time_tolerance < 0.0:
            raise ValueError("ranking_time_tolerance must be finite and nonnegative")
        allowed = {"none", "hang", "crash", "error", "malformed"}
        names: set[str] = set()
        for stage, injection in self.test_injections:
            if not stage or stage in names:
                raise ValueError("test injection stage names must be unique and nonempty")
            crash_after_iteration = injection.startswith("crash_after_iteration:")
            if crash_after_iteration:
                try:
                    int(injection.partition(":")[2])
                except ValueError as error:
                    raise ValueError(f"unsupported test injection: {injection}") from error
            if injection not in allowed and not crash_after_iteration:
                raise ValueError(f"unsupported test injection: {injection}")
            names.add(stage)
        if self.telemetry_test_injection not in {"none", "raise"}:
            raise ValueError("unsupported telemetry test injection")

    def injection_for(self, stage: str) -> str:
        return dict(self.test_injections).get(stage, "none")


@dataclass(frozen=True, slots=True)
class WarmStartOptimizationConfig:
    init_w: float
    terminal_w_max: float | None
    curvature_iterations: int
    length_iterations: int
    time_iterations: int
    maximum_exchange_rounds: int
    feasibility_tolerance: float
    curvature_regularization: float
    length_curvature_regularization: float
    curvature_slope_limit: float | None
    n_scan: int
    envelope_scan: int
    domain_scan: int
    curvature_enabled: bool = True
    length_enabled: bool = True
    class_time_pilot_iterations: int = 0
    initial_parameters_override: Array | None = None
    initial_pool_override: ConstraintPool | None = None
    initial_source_label: str = "analytic_initializer"
    run_full_time: bool = True
    # Quotient-member time pilots intentionally retain the historical SLSQP
    # backend.  Only the final route time polish was qualified for filter-SQP.
    allow_primary_filter_sqp: bool = True


@dataclass(frozen=True, slots=True)
class WarmStartStageRequest:
    operation: Literal["geometry", "projection", "exact_time", "full_time"]
    stage_name: str
    problem: RouteOptimizationProblem
    parameters: Array
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None
    endpoint_target: tuple[float | None, float | None, float | None, float | None]
    additional_inequalities: Any | None
    pool: ConstraintPool | None
    optimizer_settings: PathOptimizerSettings | None
    init_w: float
    geometry_kind: Literal["curvature", "length", "time"] | None = None
    curvature_regularization: float = 0.0
    length_curvature_regularization: float = 1.0e-5
    n_scan: int = 256
    envelope_scan: int = 64
    domain_scan: int = 256
    test_injection: str = "none"
    time_checkpoint: TimePolishCheckpointSettings | None = None
    checkpoint_feasibility_tolerance: float = 1.0e-8
    checkpoint_curvature_slope_limit: float | None = None
    terminal_w_max: float | None = None


@dataclass(frozen=True, slots=True)
class TimePolishCheckpoint:
    parameters: Array
    exact_time: float
    endpoint_error: float
    corridor_upper_bound: float
    exact_violation: float
    major_iteration: int
    local_iteration: int
    exchange_round: int
    elapsed_seconds: float
    certification_seconds_total: float
    certification_checks: int
    sequence: int
    digest: str
    objective_calls: int = 0
    gradient_calls: int = 0

    def checkpoint_safe_copy(self) -> "TimePolishCheckpoint":
        return replace(self, parameters=np.asarray(self.parameters, dtype=float).copy())


@dataclass(frozen=True, slots=True)
class TimePolishNumericalCheckpoint:
    """Parent-owned accepted time-optimizer iterate that is not yet planner-safe.

    It may be projected/certified only after the time worker stops.  Publishing
    it never changes the live optimizer trajectory and it can never be returned
    directly to the planner.
    """

    parameters: Array
    objective_time: float
    endpoint_error: float
    corridor_upper_bound: float
    exact_violation: float
    major_iteration: int
    local_iteration: int
    exchange_round: int
    elapsed_seconds: float
    certification_seconds_total: float
    certification_checks: int
    sequence: int
    digest: str
    objective_calls: int = 0
    gradient_calls: int = 0

    def checkpoint_safe_copy(self) -> "TimePolishNumericalCheckpoint":
        return replace(self, parameters=np.asarray(self.parameters, dtype=float).copy())


@dataclass(frozen=True, slots=True)
class WarmStartStageResult:
    operation: str
    parameters: Array | None
    solver_success: bool | None
    message: str
    known_time: float | None
    pool: ConstraintPool | None
    checkpoint_checks: int = 0
    checkpoint_emissions: int = 0
    checkpoint_certification_seconds: float = 0.0
    backend: str | None = None
    optimizer_iterations: int = 0
    objective_calls: int = 0
    function_evaluations: int = 0
    gradient_evaluations: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    exchange_rounds: int = 0


@dataclass(frozen=True, slots=True)
class WarmStartStageRecord:
    name: str
    status: str
    elapsed_seconds: float
    failure_reason: str
    message: str
    strict_certified: bool
    solver_success: bool | None
    endpoint_error: float | None
    corridor_upper_bound: float | None
    exact_violation: float | None
    candidate_digest: str | None
    exact_time: float | None
    ranking_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class BasinComparison:
    classification: str
    maximum_spatial_separation: float | None
    rms_spatial_separation: float | None
    curvature_sign_mismatch: float | None


@dataclass(frozen=True, slots=True)
class WarmStartRunRecord:
    schema_version: int
    run_id: str
    scheduling_mode: str
    initializer_mode: str
    stage_records: tuple[WarmStartStageRecord, ...]
    primary_status: str
    alternate_trigger: str | None
    selected_candidate: str | None
    selected_candidate_digest: str | None
    fallback_candidate: str | None
    final_candidate: str | None
    planner_time: float | None
    planner_certified: bool
    phase_one_seconds: float
    primary_geometry_seconds: float
    alternate_geometry_seconds: float
    projection_seconds: float
    geometry_seconds: float
    ranking_seconds: float
    full_time_seconds: float
    total_seconds: float
    basin_comparison: BasinComparison | None
    telemetry_failed: bool
    counterfactual: dict[str, Any] | None = None
    time_checkpoint: dict[str, Any] | None = None
    time_backend: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class WarmStartExecutionResult:
    problem_index: int
    parameters: Array
    time: float
    selected_stage: str
    stages: tuple[WarmStartStageRecord, ...]
    run_record: WarmStartRunRecord
    pool: ConstraintPool | None = None


@dataclass(slots=True)
class _Candidate:
    source: str
    source_order: int
    parameters: Array
    pool: ConstraintPool | None
    generation_seconds: float
    solver_success: bool | None
    message: str
    strict: bool
    endpoint_error: float
    corridor_upper_bound: float
    exact_violation: float
    digest: str
    known_time: float | None
    exact_time: float | None = None
    ranking_seconds: float = 0.0


class _AppendOnlyTelemetry:
    def __init__(self, path: str | None, injection: str) -> None:
        self.path = None if path is None else Path(path)
        self.injection = injection
        self.failed = False

    def emit(self, value: dict[str, Any]) -> None:
        try:
            if self.injection == "raise":
                raise OSError("injected telemetry failure")
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.failed = True


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _digest(x: Sequence[float]) -> str:
    array = np.asarray(x, dtype=np.float64).copy()
    array[array == 0.0] = 0.0
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()


def _settings(iterations: int, rounds: int, config: WarmStartOptimizationConfig) -> PathOptimizerSettings:
    return PathOptimizerSettings(
        exchange=ExchangeSettings(
            maximum_rounds=rounds,
            require_solver_success=False,
            slsqp=SLSQPSettings(max_iterations=iterations, ftol=1.0e-10, display=False),
        ),
        n_scan=config.n_scan,
        envelope_scan=config.envelope_scan,
        domain_scan=config.domain_scan,
    )


def _qualified_filter_settings(
    iterations: int,
    rounds: int,
    config: WarmStartOptimizationConfig,
) -> PathOptimizerSettings:
    """Return the exact filter-SQP settings used by the qualified small-route campaign."""
    sparse = SparseSQPSettings(
        max_iterations=iterations,
        initial_trust_radius=0.08,
        maximum_trust_radius=0.16,
        operational_polish_enabled=False,
        check_initial_kkt=False,
        diagnostics=SparseSQPDiagnosticControls.intermediate(),
    )
    return PathOptimizerSettings(
        exchange=ExchangeSettings(
            maximum_rounds=rounds,
            require_solver_success=False,
            finite_solver="sparse_sqp",
            slsqp=SLSQPSettings(max_iterations=iterations, ftol=1.0e-10, display=False),
            sparse_sqp=sparse,
            final_kkt_diagnostics=False,
        ),
        n_scan=config.n_scan,
        envelope_scan=config.envelope_scan,
        domain_scan=config.domain_scan,
    )


def _active_internal_cap_count(
    problem: RouteOptimizationProblem,
    parameters: Sequence[float],
    config: WarmStartOptimizationConfig,
) -> int:
    """Count active reverse-profile cap anchors at the selected warm iterate."""
    raw = knot_parameters_to_raw(parameters, initial_k=problem.initial_state.k)
    if reverse_solver.reverse_backend() == "native":
        return reverse_solver.native_internal_cap_count(
            raw,
            init_w=config.init_w,
            terminal_w_max=config.terminal_w_max,
            forward_init_k=problem.initial_state.k,
            n_scan=config.n_scan,
            domain_scan=config.domain_scan,
        )
    build = reverse_solver.build_scalar_speed_profile(
        raw,
        init_w=config.init_w,
        terminal_w_max=config.terminal_w_max,
        forward_init_k=problem.initial_state.k,
        n_scan=config.n_scan,
        envelope_scan=config.envelope_scan,
        domain_scan=config.domain_scan,
    )
    if build.anchor_stats is None:
        raise RuntimeError("scalar profile omitted anchor statistics")
    return int(build.anchor_stats.inserted_anchors)


def _warm_start_stage_worker(request: WarmStartStageRequest, context: WorkerContext) -> WarmStartStageResult | dict[str, str]:
    injection = request.test_injection
    if injection == "hang":
        context.set_stage(request.stage_name)
        time.sleep(3600.0)
    if injection == "crash":
        os._exit(97)
    if injection == "error":
        raise RuntimeError(f"injected failure in {request.stage_name}")
    if injection == "malformed":
        return {"malformed": request.stage_name}
    context.set_stage(request.stage_name)
    problem = request.problem
    x = np.asarray(request.parameters, dtype=float)
    if request.operation in {"geometry", "full_time"}:
        if request.optimizer_settings is None:
            raise ValueError("optimizer settings required")
        checkpoint_started = time.perf_counter()
        checkpoint_checks = 0
        checkpoint_emissions = 0
        checkpoint_certification_seconds = 0.0
        checkpoint_major_iteration = 0
        checkpoint_last_checked_iteration = 0
        checkpoint_last_checked_seconds = 0.0
        checkpoint_best_time = math.inf
        checkpoint_sequence = 0
        numerical_best_time = math.inf
        numerical_sequence = 0

        def time_progress(event: dict[str, object]) -> None:
            nonlocal checkpoint_checks, checkpoint_emissions
            nonlocal checkpoint_certification_seconds, checkpoint_major_iteration
            nonlocal checkpoint_last_checked_iteration, checkpoint_last_checked_seconds
            nonlocal checkpoint_best_time, checkpoint_sequence
            nonlocal numerical_best_time, numerical_sequence
            policy = request.time_checkpoint
            if request.operation != "full_time" or policy is None or not policy.enabled:
                return
            if event.get("event") != "iteration":
                return
            if injection is not None and injection.startswith("crash_after_iteration:"):
                try:
                    crash_after = int(injection.partition(":")[2])
                except ValueError:
                    crash_after = -1
                # Test-only deterministic worker failure injection.  The check
                # occurs before this event is retained, so ``N`` accepted
                # iterates are observable and the worker exits on the next one.
                if checkpoint_major_iteration >= crash_after >= 0:
                    os._exit(97)
            parameters_in = event.get("parameters")
            if parameters_in is None:
                return
            checkpoint_major_iteration += 1
            elapsed = time.perf_counter() - checkpoint_started
            since_seconds = elapsed - checkpoint_last_checked_seconds
            since_iterations = checkpoint_major_iteration - checkpoint_last_checked_iteration
            # Checkpoint retention can affect the planner-facing fallback after
            # a supervised time-stage failure, so the major-iteration trigger
            # must be deterministic.  Wall time may force an earlier check when
            # iterations are very slow, but a faster machine must never skip an
            # iteration that the configured cadence says to inspect.
            if not _time_checkpoint_due(
                policy,
                since_iterations=since_iterations,
                since_seconds=since_seconds,
            ):
                return
            checkpoint_last_checked_iteration = checkpoint_major_iteration
            checkpoint_last_checked_seconds = elapsed
            candidate = np.asarray(parameters_in, dtype=float)
            t0 = time.perf_counter()
            strict, endpoint, corridor, violation = _certify(
                problem,
                candidate,
                request.checkpoint_feasibility_tolerance,
                request.checkpoint_curvature_slope_limit,
            )
            checkpoint_checks += 1
            if not strict:
                checkpoint_certification_seconds += time.perf_counter() - t0
                objective_value = event.get("objective_value")
                try:
                    objective_time = float(objective_value)
                except (TypeError, ValueError, OverflowError):
                    return
                if (
                    math.isfinite(objective_time)
                    and violation <= policy.numerical_violation_cap
                    and objective_time < numerical_best_time - policy.minimum_exact_time_improvement
                ):
                    numerical_best_time = objective_time
                    numerical_sequence += 1
                    local_iteration = int(event.get("iteration", 0))
                    exchange_round = int(event.get("exchange_round", 0))
                    context.set_stage("full_time_numerical_checkpoint")
                    context.emit_checkpoint(
                        TimePolishNumericalCheckpoint(
                            candidate.copy(), objective_time, endpoint, corridor, violation,
                            checkpoint_major_iteration, local_iteration, exchange_round,
                            elapsed, checkpoint_certification_seconds, checkpoint_checks,
                            numerical_sequence, _digest(candidate),
                            int(event.get("objective_calls", 0)),
                            int(event.get("gradient_calls", 0)),
                        )
                    )
                    context.set_stage(request.stage_name)
                return
            try:
                raw = knot_parameters_to_raw(
                    candidate, initial_k=problem.initial_state.k
                )
                exact_time = float(
                    scalar_reverse_solver.evaluate_time_scalar(
                        raw,
                        init_w=request.init_w,
                        terminal_w_max=request.terminal_w_max,
                        initial_k=problem.initial_state.k,
                        n_scan=request.n_scan,
                        envelope_scan=request.envelope_scan,
                        domain_scan=request.domain_scan,
                    )
                )
            except (ValueError, FloatingPointError, OverflowError):
                checkpoint_certification_seconds += time.perf_counter() - t0
                return
            checkpoint_certification_seconds += time.perf_counter() - t0
            if not math.isfinite(exact_time):
                return
            if exact_time >= checkpoint_best_time - policy.minimum_exact_time_improvement:
                return
            checkpoint_best_time = exact_time
            checkpoint_sequence += 1
            checkpoint_emissions += 1
            local_iteration = int(event.get("iteration", 0))
            exchange_round = int(event.get("exchange_round", 0))
            context.set_stage("full_time_checkpoint")
            context.emit_checkpoint(
                TimePolishCheckpoint(
                    candidate.copy(),
                    exact_time,
                    endpoint,
                    corridor,
                    violation,
                    checkpoint_major_iteration,
                    local_iteration,
                    exchange_round,
                    elapsed,
                    checkpoint_certification_seconds,
                    checkpoint_checks,
                    checkpoint_sequence,
                    _digest(candidate),
                    int(event.get("objective_calls", 0)),
                    int(event.get("gradient_calls", 0)),
                )
            )
            context.set_stage(request.stage_name)

        kwargs: dict[str, Any] = {
            "init_w": request.init_w,
            "terminal_w_max": request.terminal_w_max,
            "endpoint_target": request.endpoint_target,
            "bounds": request.bounds,
            "additional_inequalities": request.additional_inequalities,
            "pool": request.pool,
            "settings": request.optimizer_settings,
        }
        if request.operation == "full_time" or request.geometry_kind == "time":
            kwargs.update(time_weight=1.0, curvature_weight=request.curvature_regularization)
        elif request.geometry_kind == "curvature":
            kwargs.update(time_weight=0.0, curvature_weight=1.0)
        elif request.geometry_kind == "length":
            kwargs.update(
                time_weight=0.0,
                geometry_weight=1.0,
                curvature_weight=request.length_curvature_regularization,
                geometry_objective=path_length_value_and_raw_gradient,
            )
        else:
            raise ValueError("geometry kind required")
        if request.operation == "full_time":
            kwargs["progress_callback"] = time_progress
        result = optimize_path(x, problem.corridor, problem.initial_state, **kwargs)
        pool = None if result.exchange is None else result.exchange.pool
        exchange_rounds = () if result.exchange is None else result.exchange.rounds
        optimizer_iterations = sum(int(item.finite.iterations) for item in exchange_rounds)
        objective_calls = sum(int(item.finite.objective_calls) for item in exchange_rounds)
        function_evaluations = sum(
            int(item.finite.solver_result.function_evaluations) for item in exchange_rounds
        )
        gradient_evaluations = sum(
            int(item.finite.solver_result.gradient_evaluations) for item in exchange_rounds
        )
        backend = (
            None
            if not exchange_rounds
            else str(exchange_rounds[-1].finite.solver_result.backend)
        )
        accepted_steps = 0
        rejected_steps = 0
        for item in exchange_rounds:
            diagnostics = item.finite.solver_result.diagnostics
            accepted_steps += int(getattr(diagnostics, "accepted_steps", 0))
            rejected_steps += int(getattr(diagnostics, "rejected_steps", 0))
        response = WarmStartStageResult(
            request.operation,
            np.asarray(result.parameters, dtype=float).copy(),
            bool(result.success),
            str(result.message),
            float(result.time) if math.isfinite(result.time) else None,
            pool,
            checkpoint_checks,
            checkpoint_emissions,
            checkpoint_certification_seconds,
            backend,
            optimizer_iterations,
            objective_calls,
            function_evaluations,
            gradient_evaluations,
            accepted_steps,
            rejected_steps,
            len(exchange_rounds),
        )
        context.emit_checkpoint(response)
        return response
    if request.operation == "projection":
        if request.optimizer_settings is None:
            raise ValueError("optimizer settings required")
        result = project_path_to_feasibility(
            x,
            problem.corridor,
            problem.initial_state,
            bounds=request.bounds,
            endpoint_target=request.endpoint_target,
            additional_inequalities=request.additional_inequalities,
            pool=request.pool,
            settings=request.optimizer_settings.exchange,
        )
        response = WarmStartStageResult(
            request.operation,
            np.asarray(result.x, dtype=float).copy(),
            bool(result.success),
            str(result.message),
            None,
            result.pool,
        )
        context.emit_checkpoint(response)
        return response
    if request.operation == "exact_time":
        raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
        value = scalar_reverse_solver.evaluate_time_scalar(
            raw,
            init_w=request.init_w,
            terminal_w_max=request.terminal_w_max,
            initial_k=problem.initial_state.k,
            n_scan=request.n_scan,
            envelope_scan=request.envelope_scan,
            domain_scan=request.domain_scan,
        )
        if not math.isfinite(value):
            raise FloatingPointError("exact reverse time is nonfinite")
        return WarmStartStageResult("exact_time", None, None, "exact reverse-time evaluation completed", float(value), None)
    raise ValueError(request.operation)


def _valid_stage_result(value: Any) -> bool:
    if not isinstance(value, WarmStartStageResult):
        return False
    if value.parameters is not None:
        x = np.asarray(value.parameters, dtype=float)
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            return False
    return value.known_time is None or math.isfinite(value.known_time)


def run_supervised_warm_start_stage(
    request: WarmStartStageRequest,
    *,
    timeout_seconds: float,
    startup_seconds: float = 15.0,
    terminate_grace_seconds: float = 0.5,
) -> SupervisedBatchOutcome:
    """Run one stage with independent startup and hard work deadlines."""
    settings = SupervisorSettings(
        hard_timeout_seconds=max(0.01, timeout_seconds),
        worker_startup_timeout_seconds=startup_seconds,
        terminate_grace_seconds=terminate_grace_seconds,
        deadline_refresh_on_checkpoint=False,
        start_method="posix_spawn",
    )
    with SupervisedBatchRunner(
        _warm_start_stage_worker,
        settings=settings,
        result_validator=_valid_stage_result,
    ) as runner:
        return runner.run(request, hard_timeout_seconds=timeout_seconds)


def _certify(problem: RouteOptimizationProblem, x: Sequence[float], tolerance: float, slope_limit: float | None) -> tuple[bool, float, float, float]:
    array = np.asarray(x, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        return False, math.inf, math.inf, math.inf
    try:
        raw = knot_parameters_to_raw(array, initial_k=problem.initial_state.k)
        final = compile_geometry_path(raw, problem.initial_state).final_state
        endpoint = problem.terminal_violation(final)
        separation = separate_rectangle_path(array, problem.initial_state, problem.corridor)
        slope = 0.0 if slope_limit is None else max(0.0, max(abs(float(v)) for v in raw[1::2]) - slope_limit)
        violation = max(0.0, endpoint, separation.worst_upper_bound, slope)
        strict = bool(endpoint <= tolerance and separation.certified(tolerance) and slope <= tolerance)
        return strict, float(endpoint), float(separation.worst_upper_bound), float(violation)
    except (ValueError, FloatingPointError, OverflowError):
        return False, math.inf, math.inf, math.inf


def _sample(problem: RouteOptimizationProblem, x: Sequence[float], count: int = 129) -> tuple[Array, Array]:
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    path = compile_geometry_path(raw, problem.initial_state)
    lengths = np.asarray(path.lengths, dtype=float)
    cumulative = np.cumsum(lengths)
    total = float(cumulative[-1])
    points = np.empty((count,2)); curvature = np.empty(count)
    for q, station in enumerate(np.linspace(0.0,total,count)):
        i = min(int(np.searchsorted(cumulative, station, side="right")), path.n_segments-1)
        start = 0.0 if i == 0 else float(cumulative[i-1])
        frac = 0.0 if lengths[i] == 0 else min(1.0,max(0.0,(station-start)/lengths[i]))
        state = path.state_at_fraction(i,float(frac)); points[q]=(state.x,state.y); curvature[q]=state.k
    return points, curvature


def compare_basin_candidates(problem: RouteOptimizationProblem, first: Sequence[float], second: Sequence[float]) -> BasinComparison:
    a,ka=_sample(problem,first); b,kb=_sample(problem,second)
    delta=np.linalg.norm(a-b,axis=1); maximum=float(np.max(delta)); rms=float(np.sqrt(np.mean(delta*delta)))
    sa=np.where(np.abs(ka)<=1e-10,0,np.sign(ka)); sb=np.where(np.abs(kb)<=1e-10,0,np.sign(kb)); mismatch=float(np.mean(sa!=sb))
    maxk=float(np.max(np.abs(ka-kb)))
    if maximum<=1e-8 and maxk<=1e-8: label="equivalent representation"
    elif maximum<=0.05 and rms<=0.02 and mismatch<=0.08: label="same geometric basin"
    elif maximum<=0.20 and mismatch<=0.25: label="weakly distinct local shape"
    else: label="materially distinct basin"
    return BasinComparison(label,maximum,rms,mismatch)


def _remaining(started: float, stage_limit: float, total_limit: float) -> float | None:
    value=min(stage_limit,total_limit-(time.perf_counter()-started))
    return value if value>0.01 else None


def _select(
    candidates: Sequence[_Candidate],
    tolerance: float,
    *,
    require_strict: bool = True,
) -> _Candidate:
    usable=[
        c for c in candidates
        if (c.strict or not require_strict) and c.exact_time is not None
    ]
    if not usable:
        qualifier = "strict " if require_strict else ""
        raise RuntimeError(f"no {qualifier}candidate completed exact-time ranking")
    minimum=min(float(c.exact_time) for c in usable)
    tied=[c for c in usable if float(c.exact_time)<=minimum+tolerance]
    return min(tied,key=lambda c:(c.exact_violation,c.generation_seconds,c.source_order,c.digest))


def _run_stage_request(
    runner: SupervisedBatchRunner,
    request: WarmStartStageRequest,
    *,
    timeout_seconds: float,
    checkpoint_callback: Any | None = None,
) -> SupervisedBatchOutcome:
    """Run one independently bounded request in the policy-owned worker.

    The supervisor bounds child execution.  A separate parent watchdog also
    bounds startup, IPC framing, and teardown; it can close the socket and kill
    the process group even while the calling thread is blocked in framed IPC.
    """
    started=time.perf_counter()
    deadline_fired=threading.Event()
    def expire() -> None:
        deadline_fired.set()
        runner.abort_now()
    watchdog=threading.Timer(float(timeout_seconds),expire)
    watchdog.daemon=True
    watchdog.start()
    try:
        outcome=runner.run(
            request,
            hard_timeout_seconds=float(timeout_seconds),
            checkpoint_callback=checkpoint_callback,
        )
    finally:
        watchdog.cancel()
    if deadline_fired.is_set():
        return SupervisedBatchOutcome(
            False,None,runner.last_checkpoint,"hard_timeout",
            f"external stage deadline exceeded for {request.stage_name}",
            request.stage_name,None,True,time.perf_counter()-started,0,
        )
    return outcome


def create_warm_start_stage_runner(schedule: WarmStartScheduleSettings) -> SupervisedBatchRunner:
    return SupervisedBatchRunner(
        _warm_start_stage_worker,
        settings=SupervisorSettings(
            hard_timeout_seconds=schedule.deadlines.total_pipeline_seconds,
            worker_startup_timeout_seconds=schedule.deadlines.worker_startup_seconds,
            terminate_grace_seconds=schedule.deadlines.terminate_grace_seconds,
            deadline_refresh_on_checkpoint=False,
            start_method="posix_spawn",
        ),
        result_validator=_valid_stage_result,
    )


def execute_bounded_warm_start_schedule(
    problems: Sequence[RouteOptimizationProblem],
    *,
    schedule: WarmStartScheduleSettings,
    config: WarmStartOptimizationConfig,
    stage_runner: SupervisedBatchRunner | None = None,
) -> WarmStartExecutionResult:
    """Execute the bounded schedule for one route or a quotient class.

    V8 intentionally limited this API to a single route.  V10 removes that
    legacy escape hatch.  For a quotient class each member receives the same
    bounded geometry schedule and, when configured, the historical short time
    pilot.  The best continuously certified pilot is then continued through
    exactly one bounded full-time optimization without regenerating geometry.
    """
    frozen = tuple(problems)
    if not frozen:
        raise ValueError("problems must not be empty")

    def run_single(
        problem_index: int,
        settings: WarmStartScheduleSettings,
        local_config: WarmStartOptimizationConfig,
    ) -> WarmStartExecutionResult:
        # A planner may supply one long-lived supervised worker so immutable
        # segment/reverse compilation caches survive across complete topology
        # evaluations, matching the historical in-process cache locality.
        # Per-stage hard deadlines remain enforced by _run_stage_request(); a
        # timeout/crash aborts the worker and SupervisedBatchRunner restarts it
        # on the next request.  Standalone callers retain the safe one-runner-
        # per-route behavior.
        owns_runner = stage_runner is None
        runner = create_warm_start_stage_runner(settings) if owns_runner else stage_runner
        assert runner is not None
        try:
            result = _execute(frozen[problem_index], settings, local_config, runner)
        finally:
            if owns_runner:
                runner.close()
        return replace(result, problem_index=problem_index)

    def prefix_stage(prefix: str, stage: WarmStartStageRecord) -> WarmStartStageRecord:
        return replace(stage, name=f"{prefix}/{stage.name}")

    def run_policy(settings: WarmStartScheduleSettings) -> WarmStartExecutionResult:
        if len(frozen) == 1:
            return run_single(0, settings, config)

        member_results: list[WarmStartExecutionResult] = []
        pilot_iterations = int(config.class_time_pilot_iterations)
        for index in range(len(frozen)):
            member_config = replace(
                config,
                time_iterations=max(1, pilot_iterations) if pilot_iterations > 0 else config.time_iterations,
                run_full_time=pilot_iterations > 0,
                allow_primary_filter_sqp=False,
                initial_parameters_override=None,
                initial_pool_override=None,
                initial_source_label="analytic_initializer",
            )
            member_results.append(run_single(index, settings, member_config))

        winner = min(
            member_results,
            key=lambda result: (float(result.time), int(result.problem_index)),
        )
        final_schedule = replace(
            settings,
            mode=WarmStartSchedulingMode.LEGACY_ALWAYS_BOTH,
            initializer_mode=WarmStartInitializerMode.ANALYTIC_INITIALIZER,
        )
        final_config = replace(
            config,
            curvature_enabled=False,
            length_enabled=False,
            run_full_time=True,
            allow_primary_filter_sqp=True,
            initial_parameters_override=winner.parameters.copy(),
            initial_pool_override=winner.pool,
            initial_source_label=f"quotient_member_{winner.problem_index}_pilot",
        )
        final = run_single(winner.problem_index, final_schedule, final_config)

        prefixed: list[WarmStartStageRecord] = []
        for result in member_results:
            prefixed.extend(
                prefix_stage(f"member_{result.problem_index}", stage)
                for stage in result.stages
            )
        prefixed.extend(prefix_stage("class_final", stage) for stage in final.stages)

        records = tuple(prefixed)
        total_phase = sum(r.run_record.phase_one_seconds for r in member_results) + final.run_record.phase_one_seconds
        total_primary = sum(r.run_record.primary_geometry_seconds for r in member_results) + final.run_record.primary_geometry_seconds
        total_alt = sum(r.run_record.alternate_geometry_seconds for r in member_results) + final.run_record.alternate_geometry_seconds
        total_projection = sum(r.run_record.projection_seconds for r in member_results) + final.run_record.projection_seconds
        total_ranking = sum(r.run_record.ranking_seconds for r in member_results) + final.run_record.ranking_seconds
        total_full = sum(r.run_record.full_time_seconds for r in member_results) + final.run_record.full_time_seconds
        total_seconds = sum(r.run_record.total_seconds for r in member_results) + final.run_record.total_seconds
        aggregate = replace(
            final.run_record,
            scheduling_mode=settings.mode.value,
            initializer_mode=settings.initializer_mode.value,
            stage_records=records,
            primary_status="quotient_class_completed",
            selected_candidate=f"member_{winner.problem_index}/{winner.selected_stage}",
            selected_candidate_digest=winner.run_record.selected_candidate_digest,
            fallback_candidate=f"member_{winner.problem_index}/{winner.run_record.final_candidate}",
            final_candidate=f"class_final/{final.run_record.final_candidate}",
            planner_time=final.time,
            planner_certified=True,
            phase_one_seconds=total_phase,
            primary_geometry_seconds=total_primary,
            alternate_geometry_seconds=total_alt,
            projection_seconds=total_projection,
            geometry_seconds=total_primary + total_alt,
            ranking_seconds=total_ranking,
            full_time_seconds=total_full,
            total_seconds=total_seconds,
            basin_comparison=None,
            telemetry_failed=any(r.run_record.telemetry_failed for r in member_results) or final.run_record.telemetry_failed,
            counterfactual={
                "quotient_member_count": len(frozen),
                "winner_problem_index": winner.problem_index,
                "member_times": [float(r.time) for r in member_results],
                "class_time_pilot_iterations": pilot_iterations,
            },
        )
        return replace(final, stages=records, run_record=aggregate)

    if schedule.mode is WarmStartSchedulingMode.SHADOW_COMPARE:
        reference = run_policy(
            replace(schedule, mode=WarmStartSchedulingMode.BOUNDED_BEST_OF_BOTH)
        )
        counter = run_policy(
            replace(schedule, mode=WarmStartSchedulingMode.LENGTH_FIRST_ESCALATION)
        )
        cf = {
            "mode": counter.run_record.scheduling_mode,
            "planner_time": counter.time,
            "selected_stage": counter.selected_stage,
            "certified_regret_vs_reference": counter.time - reference.time,
            # Retain the V8 field name for machine-readable compatibility.
            "certified_regret_vs_legacy": counter.time - reference.time,
            "total_seconds": counter.run_record.total_seconds,
            "geometry_seconds": counter.run_record.geometry_seconds,
            "alternate_trigger": counter.run_record.alternate_trigger,
        }
        record = replace(reference.run_record, counterfactual=cf)
        shadow_telemetry = _AppendOnlyTelemetry(
            schedule.telemetry_jsonl, schedule.telemetry_test_injection
        )
        shadow_telemetry.emit({
            "event": "shadow_compare",
            "reference_run_id": reference.run_record.run_id,
            "legacy_run_id": reference.run_record.run_id,
            "counterfactual_run_id": counter.run_record.run_id,
            "reference_planner_time": reference.time,
            "legacy_planner_time": reference.time,
            "counterfactual": cf,
            "planner_output_source": "bounded_best_of_both",
        })
        record = replace(
            record,
            telemetry_failed=record.telemetry_failed or shadow_telemetry.failed,
        )
        return replace(reference, run_record=record)
    return run_policy(schedule)


def _execute(
    problem: RouteOptimizationProblem,
    schedule: WarmStartScheduleSettings,
    config: WarmStartOptimizationConfig,
    stage_runner: SupervisedBatchRunner,
) -> WarmStartExecutionResult:
    started=time.perf_counter(); run_id=uuid.uuid4().hex; telemetry=_AppendOnlyTelemetry(schedule.telemetry_jsonl,schedule.telemetry_test_injection)
    telemetry.emit({"event":"warm_start_begin","run_id":run_id,"scheduling_mode":schedule.mode.value,"initializer_mode":schedule.initializer_mode.value})
    records:list[WarmStartStageRecord]=[]; candidates:list[_Candidate]=[]; source_order=0
    phase_seconds=primary_seconds=alternate_seconds=projection_seconds=ranking_seconds=full_seconds=0.0
    primary_status="not_started"; alternate_trigger=None; d=schedule.deadlines
    bounds=stable_knot_bounds(problem.initial_parameters)
    slope=problem.inequalities(None if config.curvature_slope_limit is None else CurvatureSlopeConstraint(config.curvature_slope_limit,initial_k=problem.initial_state.k))

    def emit(record:WarmStartStageRecord)->None:
        records.append(record); telemetry.emit({"event":"stage_end","run_id":run_id,"stage":asdict(record)})

    def candidate_record(c:_Candidate)->WarmStartStageRecord:
        return WarmStartStageRecord(c.source,"strict_certified" if c.strict else "uncertified",c.generation_seconds,"none",c.message,c.strict,c.solver_success,c.endpoint_error,c.corridor_upper_bound,c.exact_violation,c.digest,c.exact_time,c.ranking_seconds)

    def aligned_target(values: Sequence[float]):
        try:
            return align_endpoint_target_to_path(
                values, problem.initial_state, problem.endpoint_target
            ).aligned_target
        except (ValueError, RuntimeError, FloatingPointError):
            # Invalid numerical trial points must be handled by the stage itself;
            # branch alignment is a handoff normalization, not a new failure mode.
            return problem.endpoint_target

    warm_tolerance = max(1.0e-4, 100.0 * config.feasibility_tolerance)

    def continuation_eligible(candidate: _Candidate) -> bool:
        """Whether a candidate may seed further bounded optimization.

        Planner exposure still requires ``candidate.strict``.  This deliberately
        preserves the V9 distinction between a near-feasible *warm start* and a
        continuously certified fallback: legacy SLSQP often crossed a useful
        basin from a candidate inside the 1e-4 warm band before the final
        projection certified it.
        """
        return bool(
            np.all(np.isfinite(candidate.parameters))
            and math.isfinite(candidate.endpoint_error)
            and math.isfinite(candidate.corridor_upper_bound)
            and candidate.endpoint_error <= warm_tolerance
            and candidate.corridor_upper_bound <= warm_tolerance
        )

    # Build initializer and parent-owned fallback when available.  The V10
    # integrated mode may begin analytically and invoke strict Phase I only
    # after bounded geometry scheduling fails to produce any strict candidate.
    def add_initializer(
        label: str,
        values: Sequence[float],
        pool0: ConstraintPool | None,
        message: str,
        *,
        solver_success: bool | None,
        generation_seconds: float,
    ) -> _Candidate:
        nonlocal source_order
        array = np.asarray(values, dtype=float).copy()
        strict, ep, co, vi = _certify(
            problem, array, config.feasibility_tolerance, config.curvature_slope_limit
        )
        source_order += 1
        candidate = _Candidate(
            label, source_order, array, pool0, generation_seconds, solver_success,
            message, strict, ep, co, vi, _digest(array), None
        )
        candidates.append(candidate)
        emit(candidate_record(candidate))
        return candidate

    def run_strict_phase_one(*, required: bool) -> _Candidate | None:
        nonlocal phase_seconds, source_order
        timeout = _remaining(started, d.phase_one_seconds, d.total_pipeline_seconds)
        if timeout is None:
            if required:
                raise TimeoutError("total deadline expired before strict Phase I")
            emit(WarmStartStageRecord(
                "strict_phase_one", "timeout", 0.0, "hard_timeout",
                "total deadline expired before strict Phase I", False, None,
                None, None, None, None, None
            ))
            return None
        telemetry.emit({
            "event": "stage_start", "run_id": run_id,
            "stage": "strict_phase_one", "timeout_seconds": timeout,
        })
        t = time.perf_counter()
        work = max(0.1, timeout - min(d.worker_startup_seconds, timeout / 2))
        phase = run_supervised_phase_one(
            problem.initial_parameters, problem.corridor, problem.initial_state,
            bounds=bounds, endpoint_target=problem.endpoint_target,
            additional_inequalities=slope, settings=schedule.phase_one,
            coordinate_map=LogLengthKnotMap.from_knot_parameters(problem.initial_parameters),
            coordinate_settings=KnotCoordinateSettings(),
            supervisor_settings=SupervisorSettings(
                hard_timeout_seconds=work,
                worker_startup_timeout_seconds=min(d.worker_startup_seconds, timeout - work),
                terminate_grace_seconds=d.terminate_grace_seconds,
                start_method="posix_spawn",
            ),
        )
        elapsed = time.perf_counter() - t
        phase_seconds += elapsed
        if phase.certified_result is None:
            outcome = phase.outcome
            emit(WarmStartStageRecord(
                "strict_phase_one",
                "timeout" if "timeout" in outcome.failure_reason else "failed",
                elapsed, outcome.failure_reason, outcome.message, False, None,
                None, None, None, None, None,
            ))
            telemetry.emit({
                "event": "strict_phase_one_unavailable", "run_id": run_id,
                "reason": outcome.failure_reason, "message": outcome.message,
                "required": required,
            })
            if required:
                raise RuntimeError(
                    f"strict Phase-I initializer unavailable: {outcome.failure_reason}: {outcome.message}"
                )
            return None
        candidate = add_initializer(
            "strict_phase_one", phase.certified_result.x, phase.certified_result.pool,
            phase.certified_result.message, solver_success=True,
            generation_seconds=elapsed,
        )
        if not candidate.strict:
            if required:
                raise RuntimeError("supervised Phase I returned non-strict result")
            return None
        return candidate

    phase_one_branch_aligned = False
    if config.initial_parameters_override is not None:
        x0 = np.asarray(config.initial_parameters_override, dtype=float).copy()
        pool = config.initial_pool_override
        add_initializer(
            config.initial_source_label, x0, pool,
            "externally supplied continuously certified warm start",
            solver_success=True, generation_seconds=0.0,
        )
    elif schedule.initializer_mode is WarmStartInitializerMode.STRICT_PHASE_ONE_INITIALIZER:
        phase_candidate = run_strict_phase_one(required=True)
        assert phase_candidate is not None
        x0 = phase_candidate.parameters.copy()
        pool = phase_candidate.pool
        phase_one_branch_aligned = True
    else:
        x0 = problem.initial_parameters.copy()
        pool = None
        add_initializer(
            "analytic_initializer", x0, pool, "analytic G2 route initializer",
            solver_success=None, generation_seconds=0.0,
        )

    def run_stage(
        name:str, operation:str, x:Array, pool0:ConstraintPool|None,
        kind:str|None, iterations:int, limit:float, *,
        align_heading_branch: bool = False,
        checkpoint_callback: Any | None = None,
        optimizer_settings_override: PathOptimizerSettings | None = None,
    )->tuple[SupervisedBatchOutcome,float]:
        timeout=_remaining(started,limit,d.total_pipeline_seconds)
        if timeout is None:
            return SupervisedBatchOutcome(False,None,None,"hard_timeout","total warm-start deadline exhausted",name,None,False,0.0,0),0.0
        telemetry.emit({"event":"stage_start","run_id":run_id,"stage":name,"timeout_seconds":timeout})
        stage_x = np.asarray(x, dtype=float).copy()
        # Preserve the historical local endpoint branch for ordinary SLSQP.
        # Periodic representative alignment is required only after strict
        # Phase I, where an equivalent 2*pi target representative may differ.
        stage_target = (
            aligned_target(stage_x) if align_heading_branch
            else problem.endpoint_target
        )
        req=WarmStartStageRequest(
            operation,name,problem,stage_x,bounds,stage_target,slope,pool0,
            (
                _settings(iterations,config.maximum_exchange_rounds,config)
                if optimizer_settings_override is None
                else optimizer_settings_override
            ),
            config.init_w,kind,config.curvature_regularization,
            config.length_curvature_regularization,config.n_scan,
            config.envelope_scan,config.domain_scan,schedule.injection_for(name),
            schedule.time_checkpoint if operation == "full_time" else None,
            config.feasibility_tolerance,config.curvature_slope_limit,
            terminal_w_max=config.terminal_w_max,
        )
        t=time.perf_counter()
        out=_run_stage_request(
            stage_runner,req,timeout_seconds=timeout,
            checkpoint_callback=checkpoint_callback,
        )
        return out,time.perf_counter()-t

    def add(name:str,out:SupervisedBatchOutcome,elapsed:float,bucket:str)->_Candidate|None:
        nonlocal source_order,primary_seconds,alternate_seconds,projection_seconds,full_seconds
        if bucket=="primary":primary_seconds+=elapsed
        elif bucket=="alternate":alternate_seconds+=elapsed
        elif bucket=="projection":projection_seconds+=elapsed
        elif bucket=="full":full_seconds+=elapsed
        if not out.success or not isinstance(out.result,WarmStartStageResult):
            emit(WarmStartStageRecord(name,"timeout" if "timeout" in out.failure_reason else "failed",elapsed,out.failure_reason,out.message,False,None,None,None,None,None,None)); return None
        r=out.result
        if r.parameters is None:
            emit(WarmStartStageRecord(name,"failed",elapsed,"malformed_result","stage omitted parameters",False,r.solver_success,None,None,None,None,r.known_time)); return None
        strict,ep,co,vi=_certify(problem,r.parameters,config.feasibility_tolerance,config.curvature_slope_limit)
        source_order+=1; c=_Candidate(name,source_order,np.asarray(r.parameters,dtype=float).copy(),r.pool,elapsed,r.solver_success,r.message,strict,ep,co,vi,_digest(r.parameters),r.known_time)
        candidates.append(c)
        emit(candidate_record(c)); return c

    def project(
        label:str,c:_Candidate|None,fallback:Array,fallback_pool:ConstraintPool|None,
        *, align_heading_branch: bool = False,
    )->tuple[_Candidate|None,bool]:
        x=fallback if c is None else c.parameters; p=fallback_pool if c is None else c.pool
        out,elapsed=run_stage(
            label,"projection",x,p,None,max(40,config.length_iterations),
            d.projection_seconds,align_heading_branch=align_heading_branch
        )
        result=add(label,out,elapsed,"projection")
        return result,(not out.success and "timeout" in out.failure_reason)

    if schedule.mode in {
        WarmStartSchedulingMode.LEGACY_ALWAYS_BOTH,
        WarmStartSchedulingMode.BOUNDED_BEST_OF_BOTH,
    }:
        if config.curvature_enabled:
            out,e=run_stage("curvature_geometry","geometry",x0,pool,"curvature",config.curvature_iterations,d.alternate_geometry_seconds,align_heading_branch=phase_one_branch_aligned); add("curvature_geometry",out,e,"alternate")
        if config.length_enabled:
            out,e=run_stage("primary_geometry","geometry",x0,pool,"length",config.length_iterations,d.primary_geometry_seconds,align_heading_branch=phase_one_branch_aligned); primary=add("primary_geometry",out,e,"primary")
            if primary is None or not primary.strict: project("primary_projection",primary,x0,pool,align_heading_branch=phase_one_branch_aligned)
        primary_status="legacy_completed"
    else:
        if config.length_enabled:
            out,e=run_stage("primary_geometry","geometry",x0,pool,"length",config.length_iterations,d.primary_geometry_seconds,align_heading_branch=phase_one_branch_aligned); primary=add("primary_geometry",out,e,"primary")
            primary_timeout=not out.success and "timeout" in out.failure_reason; projection_timeout=False
            if not primary_timeout and (primary is None or not primary.strict): primary,projection_timeout=project("primary_projection",primary,x0,pool,align_heading_branch=phase_one_branch_aligned)
            if primary is not None and primary.strict:
                primary_status="strict_certified"
            else:
                primary_status="deadline" if primary_timeout or projection_timeout else "certification_failure"; alternate_trigger=primary_status
                telemetry.emit({"event":"alternate_trigger","run_id":run_id,"reason":alternate_trigger,"primary_status":primary_status})
                if config.curvature_enabled:
                    out,e=run_stage("alternate_geometry","geometry",x0,pool,"curvature",config.curvature_iterations,d.alternate_geometry_seconds,align_heading_branch=phase_one_branch_aligned); add("alternate_geometry",out,e,"alternate")
        elif config.curvature_enabled:
            out,e=run_stage("primary_curvature_geometry","geometry",x0,pool,"curvature",config.curvature_iterations,d.primary_geometry_seconds,align_heading_branch=phase_one_branch_aligned); primary=add("primary_curvature_geometry",out,e,"primary")
            primary_timeout=not out.success and "timeout" in out.failure_reason; projection_timeout=False
            if not primary_timeout and (primary is None or not primary.strict): primary,projection_timeout=project("primary_projection",primary,x0,pool,align_heading_branch=phase_one_branch_aligned)
            primary_status=("strict_certified" if primary is not None and primary.strict else ("deadline" if primary_timeout or projection_timeout else "certification_failure"))
        else:
            primary_status="geometry_disabled"

    strict_candidates=[c for c in candidates if c.strict]
    # The new bounded-best-of-both mode preserves the safe V9 distinction
    # between planner-safe certificates and merely near-feasible internal warm
    # starts.  Length-first retains its stricter V8 escalation semantics.
    if schedule.mode is WarmStartSchedulingMode.BOUNDED_BEST_OF_BOTH:
        continuation_candidates=[c for c in candidates if continuation_eligible(c)]
    else:
        continuation_candidates=list(strict_candidates)

    # If geometry produced neither a strict fallback nor even a safe bounded
    # continuation seed, try strict Phase I now.  When a near-feasible warm
    # candidate exists, allow the supervised time solve to attempt recovery
    # first; Phase I remains available afterward if no strict result survives.
    if (
        not strict_candidates
        and not continuation_candidates
        and schedule.initializer_mode is WarmStartInitializerMode.ANALYTIC_WITH_STRICT_PHASE_ONE_FALLBACK
        and config.initial_parameters_override is None
    ):
        telemetry.emit({
            "event": "phase_one_fallback_trigger", "run_id": run_id,
            "reason": "no_warm_geometry_candidate",
        })
        phase_candidate = run_strict_phase_one(required=False)
        if phase_candidate is not None and phase_candidate.strict:
            strict_candidates.append(phase_candidate)
            continuation_candidates.append(phase_candidate)
            primary_status = f"{primary_status}_phase_one_fallback"

    if not continuation_candidates:
        telemetry.emit({"event":"warm_start_failed","run_id":run_id,"stage":"geometry_scheduling","reason":"no_continuation_candidate","primary_status":primary_status,"alternate_trigger":alternate_trigger,"stage_records":[asdict(record) for record in records]})
        raise RuntimeError("no bounded geometry or Phase-I candidate survived scheduling")

    for c in continuation_candidates:
        timeout=_remaining(started,d.exact_ranking_seconds,d.total_pipeline_seconds); name=f"{c.source}/exact_ranking"
        if timeout is None:
            emit(WarmStartStageRecord(name,"timeout",0.0,"hard_timeout","total deadline exhausted",c.strict,None,c.endpoint_error,c.corridor_upper_bound,c.exact_violation,c.digest,None)); continue
        req=WarmStartStageRequest("exact_time",name,problem,c.parameters,None,aligned_target(c.parameters),None,None,None,config.init_w,n_scan=config.n_scan,terminal_w_max=config.terminal_w_max,envelope_scan=config.envelope_scan,domain_scan=config.domain_scan,test_injection=schedule.injection_for("exact_ranking"))
        t=time.perf_counter(); out=_run_stage_request(stage_runner,req,timeout_seconds=timeout); elapsed=time.perf_counter()-t; ranking_seconds+=elapsed; c.ranking_seconds=elapsed
        if out.success and isinstance(out.result,WarmStartStageResult) and out.result.known_time is not None:
            c.exact_time=float(out.result.known_time); emit(WarmStartStageRecord(name,"completed",elapsed,"none",out.result.message,c.strict,None,c.endpoint_error,c.corridor_upper_bound,c.exact_violation,c.digest,c.exact_time,elapsed))
        else:
            emit(WarmStartStageRecord(name,"failed",elapsed,out.failure_reason,out.message,c.strict,None,c.endpoint_error,c.corridor_upper_bound,c.exact_violation,c.digest,None,elapsed))
    ranked=[c for c in continuation_candidates if c.exact_time is not None]
    if ranked:
        selected=_select(
            ranked,
            schedule.ranking_time_tolerance,
            require_strict=(schedule.mode is not WarmStartSchedulingMode.BOUNDED_BEST_OF_BOTH),
        )
    else:
        known=[candidate for candidate in continuation_candidates if candidate.known_time is not None and math.isfinite(candidate.known_time)]
        if len(known)!=1:
            telemetry.emit({"event":"warm_start_failed","run_id":run_id,"stage":"exact_ranking","reason":"no_unambiguous_ranked_candidate","continuation_candidate_count":len(continuation_candidates),"known_time_count":len(known)})
            raise RuntimeError("exact-time ranking failed without an unambiguous bounded candidate")
        selected=known[0]
        selected.exact_time=selected.known_time

    fallback = min(
        (candidate for candidate in strict_candidates if candidate.exact_time is not None),
        key=lambda candidate: float(candidate.exact_time),
        default=None,
    )
    selected_basin_fallback: _Candidate | None = selected if selected.strict else None
    selected_projection: _Candidate | None = None

    # In bounded-best-of-both mode the exact-ranked winner may intentionally be
    # a near-feasible continuation seed rather than a planner-safe certificate.
    # Do not let a later time-stage failure silently fall back to a different,
    # much slower geometry basin merely because that other basin happened to be
    # projected earlier.  Cheaply project the selected basin itself and retain
    # it as the parent-owned strict fallback when it exact-ranks better.
    if (
        not selected.strict
        and selected.exact_time is not None
        and (
            fallback is None
            or float(selected.exact_time)
            < float(fallback.exact_time) - schedule.ranking_time_tolerance
        )
    ):
        selected_projection, selected_projection_timed_out = project(
            "selected_projection",
            selected,
            selected.parameters,
            selected.pool,
            align_heading_branch=bool(
                phase_one_branch_aligned or selected.source == "strict_phase_one"
            ),
        )
        if (
            not selected_projection_timed_out
            and selected_projection is not None
            and selected_projection.strict
        ):
            timeout = _remaining(
                started, d.exact_ranking_seconds, d.total_pipeline_seconds
            )
            if timeout is not None:
                name = f"{selected_projection.source}/exact_ranking"
                req = WarmStartStageRequest(
                    "exact_time", name, problem, selected_projection.parameters, None,
                    aligned_target(selected_projection.parameters), None, None, None,
                    config.init_w, n_scan=config.n_scan, terminal_w_max=config.terminal_w_max,
                    envelope_scan=config.envelope_scan,
                    domain_scan=config.domain_scan,
                    test_injection=schedule.injection_for("exact_ranking"),
                )
                t_rank = time.perf_counter()
                ranked_out = _run_stage_request(
                    stage_runner, req, timeout_seconds=timeout
                )
                rank_elapsed = time.perf_counter() - t_rank
                ranking_seconds += rank_elapsed
                if (
                    ranked_out.success
                    and isinstance(ranked_out.result, WarmStartStageResult)
                    and ranked_out.result.known_time is not None
                ):
                    selected_projection.exact_time = float(
                        ranked_out.result.known_time
                    )
                    emit(WarmStartStageRecord(
                        name, "completed", rank_elapsed, "none",
                        ranked_out.result.message, True, None,
                        selected_projection.endpoint_error,
                        selected_projection.corridor_upper_bound,
                        selected_projection.exact_violation,
                        selected_projection.digest,
                        selected_projection.exact_time, rank_elapsed,
                    ))
                else:
                    emit(WarmStartStageRecord(
                        name, "failed", rank_elapsed, ranked_out.failure_reason,
                        ranked_out.message, True, None,
                        selected_projection.endpoint_error,
                        selected_projection.corridor_upper_bound,
                        selected_projection.exact_violation,
                        selected_projection.digest, None, rank_elapsed,
                    ))
            if (
                selected_projection.exact_time is not None
                and (
                    fallback is None
                    or float(selected_projection.exact_time)
                    < float(fallback.exact_time)
                    - schedule.ranking_time_tolerance
                )
            ):
                fallback = selected_projection
            if selected_projection.exact_time is not None:
                selected_basin_fallback = selected_projection

    # Decide the final time-polish backend only after exact ranking and after
    # the selected warm basin has its independently owned strict fallback.
    # This keeps the filter-SQP rollout fail-closed and makes the empirically
    # qualified envelope explicit in planner telemetry.
    filter_policy = schedule.primary_filter_sqp
    n_variables = int(selected.parameters.size)
    n_segments = n_variables // 2
    checkpoint_recovery_active = bool(
        schedule.time_checkpoint.enabled
        and schedule.time_checkpoint.every_major_iterations > 0
    )
    supervisor_deadline_active = bool(
        math.isfinite(d.full_time_seconds) and d.full_time_seconds > 0.0
    )
    same_basin_fallback_owned = bool(
        selected_basin_fallback is not None
        and selected_basin_fallback.strict
        and selected_basin_fallback.exact_time is not None
    )
    internal_caps: int | None = None
    eligibility_reasons: list[str] = []
    if filter_policy.mode is PrimaryTimeBackendMode.FORCE_SLSQP:
        eligibility_reasons.append("policy_forced_slsqp")
    if not config.allow_primary_filter_sqp:
        eligibility_reasons.append("not_final_time_polish")
    if n_variables > filter_policy.maximum_variables:
        eligibility_reasons.append(
            f"variables>{filter_policy.maximum_variables}"
        )
    if n_segments > filter_policy.maximum_segments:
        eligibility_reasons.append(
            f"segments>{filter_policy.maximum_segments}"
        )
    if not same_basin_fallback_owned:
        eligibility_reasons.append("selected_basin_strict_fallback_missing")
    if not checkpoint_recovery_active:
        eligibility_reasons.append("deterministic_checkpoint_recovery_inactive")
    if not supervisor_deadline_active:
        eligibility_reasons.append("supervisor_hard_deadline_inactive")
    # Avoid spending an extra authoritative scalar build on routes that are
    # already structurally outside the validated small-route envelope.
    if not any(
        reason.startswith("variables>") or reason.startswith("segments>")
        for reason in eligibility_reasons
    ) and filter_policy.mode is not PrimaryTimeBackendMode.FORCE_SLSQP:
        try:
            internal_caps = _active_internal_cap_count(problem, selected.parameters, config)
        except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
            eligibility_reasons.append(
                f"internal_cap_count_unavailable:{type(error).__name__}"
            )
        else:
            if internal_caps > filter_policy.maximum_internal_caps:
                eligibility_reasons.append(
                    f"internal_caps>{filter_policy.maximum_internal_caps}"
                )
    filter_primary_eligible = not eligibility_reasons
    selected_time_backend = "filter_sqp" if filter_primary_eligible else "slsqp"
    time_backend_summary: dict[str, Any] = {
        "selected_backend": selected_time_backend,
        "eligible": bool(filter_primary_eligible),
        "reasons": list(eligibility_reasons),
        "n_variables": n_variables,
        "n_segments": n_segments,
        "active_internal_caps": internal_caps,
        "maximum_variables": int(filter_policy.maximum_variables),
        "maximum_segments": int(filter_policy.maximum_segments),
        "maximum_internal_caps": int(filter_policy.maximum_internal_caps),
        "same_basin_fallback_owned": same_basin_fallback_owned,
        "same_basin_fallback_source": (
            None if selected_basin_fallback is None else selected_basin_fallback.source
        ),
        "deterministic_checkpoint_recovery": checkpoint_recovery_active,
        "supervisor_hard_deadline": supervisor_deadline_active,
        "fallback_action": "not_needed",
    }
    telemetry.emit({"event": "time_backend_eligibility", "run_id": run_id, **time_backend_summary})

    final=fallback
    if config.run_full_time:
        # Preserve V9 SLSQP basin semantics: geometry/exact-cut state is not a
        # valid numerical warm start for the full time exchange.  Reusing the
        # geometry pool was experimentally much slower on the 4x4 controls and
        # changed the time-SLSQP trajectory.  The continuously certified
        # geometry candidate remains parent-owned as fallback, while the time
        # exchange rebuilds its own seeded pool and exact cuts.
        full_time_branch_alignment = bool(
            phase_one_branch_aligned or selected.source == "strict_phase_one"
        )
        checkpoint_frames: list[TimePolishCheckpoint] = []
        numerical_checkpoint_frames: list[TimePolishNumericalCheckpoint] = []
        checkpoint_last_sequence = 0
        numerical_checkpoint_last_sequence = 0
        checkpoint_invalid_frames = 0

        def collect_time_checkpoint(value: Any) -> None:
            nonlocal checkpoint_last_sequence, numerical_checkpoint_last_sequence
            nonlocal checkpoint_invalid_frames
            if isinstance(value, TimePolishNumericalCheckpoint):
                try:
                    parameters = np.asarray(value.parameters, dtype=float)
                    valid = bool(
                        parameters.shape == selected.parameters.shape
                        and np.all(np.isfinite(parameters))
                        and math.isfinite(value.objective_time)
                        and value.objective_time > 0.0
                        and value.exact_violation <= schedule.time_checkpoint.numerical_violation_cap
                        and value.sequence > numerical_checkpoint_last_sequence
                        and value.digest == _digest(parameters)
                    )
                except (TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    checkpoint_invalid_frames += 1
                    telemetry.emit({
                        "event": "time_numerical_checkpoint_ignored",
                        "run_id": run_id,
                        "reason": "malformed_duplicate_or_out_of_order",
                    })
                    return
                numerical_checkpoint_last_sequence = int(value.sequence)
                numerical_checkpoint_frames.append(value.checkpoint_safe_copy())
                telemetry.emit({
                    "event": "time_numerical_checkpoint_received",
                    "run_id": run_id,
                    "sequence": int(value.sequence),
                    "major_iteration": int(value.major_iteration),
                    "exchange_round": int(value.exchange_round),
                    "objective_time": float(value.objective_time),
                    "exact_violation": float(value.exact_violation),
                    "elapsed_seconds": float(value.elapsed_seconds),
                })
                return
            if not isinstance(value, TimePolishCheckpoint):
                return
            try:
                parameters = np.asarray(value.parameters, dtype=float)
                valid = bool(
                    parameters.shape == selected.parameters.shape
                    and np.all(np.isfinite(parameters))
                    and math.isfinite(value.exact_time)
                    and value.exact_time > 0.0
                    and value.sequence > checkpoint_last_sequence
                    and value.digest == _digest(parameters)
                )
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                checkpoint_invalid_frames += 1
                telemetry.emit({
                    "event": "time_checkpoint_ignored",
                    "run_id": run_id,
                    "reason": "malformed_duplicate_or_out_of_order",
                })
                return
            checkpoint_last_sequence = int(value.sequence)
            checkpoint_frames.append(value.checkpoint_safe_copy())
            telemetry.emit({
                "event": "time_checkpoint_received",
                "run_id": run_id,
                "sequence": int(value.sequence),
                "major_iteration": int(value.major_iteration),
                "exchange_round": int(value.exchange_round),
                "exact_time": float(value.exact_time),
                "elapsed_seconds": float(value.elapsed_seconds),
            })

        primary_time_settings = (
            _qualified_filter_settings(
                config.time_iterations, config.maximum_exchange_rounds, config
            )
            if selected_time_backend == "filter_sqp"
            else None
        )
        out,e=run_stage(
            "full_time_optimization", "full_time", selected.parameters, None,
            "time", config.time_iterations, d.full_time_seconds,
            align_heading_branch=full_time_branch_alignment,
            checkpoint_callback=collect_time_checkpoint,
            optimizer_settings_override=primary_time_settings,
        )
        full_time_worker_checkpoint_checks = 0
        full_time_worker_checkpoint_emissions = 0
        full_time_worker_checkpoint_seconds = 0.0
        if isinstance(out.result, WarmStartStageResult):
            full_time_worker_checkpoint_checks = int(out.result.checkpoint_checks)
            full_time_worker_checkpoint_emissions = int(out.result.checkpoint_emissions)
            full_time_worker_checkpoint_seconds = float(
                out.result.checkpoint_certification_seconds
            )
            time_backend_summary.update({
                "worker_backend": out.result.backend,
                "optimizer_iterations": int(out.result.optimizer_iterations),
                "objective_calls": int(out.result.objective_calls),
                "function_evaluations": int(out.result.function_evaluations),
                "gradient_evaluations": int(out.result.gradient_evaluations),
                "accepted_steps": int(out.result.accepted_steps),
                "rejected_steps": int(out.result.rejected_steps),
                "exchange_rounds": int(out.result.exchange_rounds),
            })
        collect_time_checkpoint(out.last_checkpoint)
        time_candidate=add("full_time_optimization",out,e,"full")

        checkpoint_candidate: _Candidate | None = None
        checkpoint_parent_certification_seconds = 0.0
        for checkpoint in sorted(
            checkpoint_frames, key=lambda item: (item.exact_time, item.sequence)
        ):
            t_checkpoint = time.perf_counter()
            strict, ep, co, vi = _certify(
                problem, checkpoint.parameters, config.feasibility_tolerance,
                config.curvature_slope_limit,
            )
            checkpoint_parent_certification_seconds += time.perf_counter() - t_checkpoint
            if not strict:
                checkpoint_invalid_frames += 1
                telemetry.emit({
                    "event": "time_checkpoint_ignored",
                    "run_id": run_id,
                    "reason": "parent_recertification_failed",
                    "sequence": int(checkpoint.sequence),
                    "exact_time": float(checkpoint.exact_time),
                })
                continue
            source_order += 1
            checkpoint_candidate = _Candidate(
                "full_time_checkpoint", source_order,
                np.asarray(checkpoint.parameters, dtype=float).copy(), None,
                float(checkpoint.elapsed_seconds), True,
                f"certified time-optimizer checkpoint at major iteration {checkpoint.major_iteration}",
                True, ep, co, vi, checkpoint.digest, float(checkpoint.exact_time),
                float(checkpoint.exact_time), 0.0,
            )
            candidates.append(checkpoint_candidate)
            emit(candidate_record(checkpoint_candidate))
            telemetry.emit({
                "event": "time_checkpoint_parent_certified",
                "run_id": run_id,
                "sequence": int(checkpoint.sequence),
                "major_iteration": int(checkpoint.major_iteration),
                "exchange_round": int(checkpoint.exchange_round),
                "exact_time": float(checkpoint.exact_time),
                "worker_certification_seconds": float(checkpoint.certification_seconds_total),
                "worker_certification_checks": int(checkpoint.certification_checks),
                "parent_certification_seconds": checkpoint_parent_certification_seconds,
            })
            break

        if checkpoint_candidate is not None and (
            final is None or checkpoint_candidate.exact_time < float(final.exact_time)
        ):
            final = checkpoint_candidate

        numerical_projection_attempts = 0
        numerical_projection_best: _Candidate | None = None
        if (
            not out.success
            and numerical_checkpoint_frames
            and schedule.time_checkpoint.maximum_projection_attempts > 0
        ):
            # The live time-optimizer worker is already stopped/killed here.  Project
            # retained accepted iterates only now, on the restarted supervised
            # worker, so checkpoint recovery cannot perturb the SLSQP trajectory.
            for numerical_checkpoint in sorted(
                numerical_checkpoint_frames,
                key=lambda item: (item.objective_time, item.sequence),
            )[: schedule.time_checkpoint.maximum_projection_attempts]:
                numerical_projection_attempts += 1
                source_order += 1
                numerical_candidate = _Candidate(
                    f"full_time_numerical_checkpoint_{numerical_checkpoint.sequence}",
                    source_order, numerical_checkpoint.parameters.copy(), None,
                    float(numerical_checkpoint.elapsed_seconds), False,
                    f"accepted time-optimizer iterate at major iteration {numerical_checkpoint.major_iteration}",
                    False, numerical_checkpoint.endpoint_error,
                    numerical_checkpoint.corridor_upper_bound,
                    numerical_checkpoint.exact_violation, numerical_checkpoint.digest,
                    float(numerical_checkpoint.objective_time),
                    float(numerical_checkpoint.objective_time), 0.0,
                )
                projected, projection_timed_out = project(
                    f"time_checkpoint_projection_{numerical_checkpoint.sequence}",
                    numerical_candidate, numerical_candidate.parameters, None,
                    align_heading_branch=full_time_branch_alignment,
                )
                if projection_timed_out:
                    break
                if projected is None or not projected.strict:
                    continue
                timeout = _remaining(started, d.exact_ranking_seconds, d.total_pipeline_seconds)
                if timeout is None:
                    break
                name = f"{projected.source}/exact_ranking"
                req = WarmStartStageRequest(
                    "exact_time", name, problem, projected.parameters, None,
                    aligned_target(projected.parameters), None, None, None, config.init_w,
                    n_scan=config.n_scan, terminal_w_max=config.terminal_w_max, envelope_scan=config.envelope_scan,
                    domain_scan=config.domain_scan,
                    test_injection=schedule.injection_for("exact_ranking"),
                )
                t_rank = time.perf_counter()
                ranked_out = _run_stage_request(stage_runner, req, timeout_seconds=timeout)
                rank_elapsed = time.perf_counter() - t_rank
                ranking_seconds += rank_elapsed
                if (
                    ranked_out.success
                    and isinstance(ranked_out.result, WarmStartStageResult)
                    and ranked_out.result.known_time is not None
                ):
                    projected.exact_time = float(ranked_out.result.known_time)
                    emit(WarmStartStageRecord(
                        name, "completed", rank_elapsed, "none", ranked_out.result.message,
                        True, None, projected.endpoint_error, projected.corridor_upper_bound,
                        projected.exact_violation, projected.digest, projected.exact_time, rank_elapsed,
                    ))
                    if (
                        numerical_projection_best is None
                        or projected.exact_time < float(numerical_projection_best.exact_time)
                    ):
                        numerical_projection_best = projected
                else:
                    emit(WarmStartStageRecord(
                        name, "failed", rank_elapsed, ranked_out.failure_reason, ranked_out.message,
                        True, None, projected.endpoint_error, projected.corridor_upper_bound,
                        projected.exact_violation, projected.digest, None, rank_elapsed,
                    ))
            if numerical_projection_best is not None and (
                final is None or numerical_projection_best.exact_time < float(final.exact_time)
            ):
                final = numerical_projection_best

        if time_candidate is not None and not time_candidate.strict:
            time_candidate,_=project(
                "time_projection",time_candidate,selected.parameters,selected.pool,
                align_heading_branch=full_time_branch_alignment,
            )
        if time_candidate is not None and time_candidate.strict:
            time_candidate.exact_time=time_candidate.known_time
            if time_candidate.exact_time is None:
                timeout=_remaining(started,d.exact_ranking_seconds,d.total_pipeline_seconds); name=f"{time_candidate.source}/exact_ranking"
                if timeout is not None:
                    req=WarmStartStageRequest("exact_time",name,problem,time_candidate.parameters,None,aligned_target(time_candidate.parameters),None,None,None,config.init_w,n_scan=config.n_scan,terminal_w_max=config.terminal_w_max,envelope_scan=config.envelope_scan,domain_scan=config.domain_scan,test_injection=schedule.injection_for("exact_ranking"))
                    t=time.perf_counter(); ro=_run_stage_request(stage_runner,req,timeout_seconds=timeout); elapsed=time.perf_counter()-t; ranking_seconds+=elapsed
                    if ro.success and isinstance(ro.result,WarmStartStageResult) and ro.result.known_time is not None:
                        time_candidate.exact_time=float(ro.result.known_time); emit(WarmStartStageRecord(name,"completed",elapsed,"none",ro.result.message,True,None,time_candidate.endpoint_error,time_candidate.corridor_upper_bound,time_candidate.exact_violation,time_candidate.digest,time_candidate.exact_time,elapsed))
                    else: emit(WarmStartStageRecord(name,"failed",elapsed,ro.failure_reason,ro.message,True,None,time_candidate.endpoint_error,time_candidate.corridor_upper_bound,time_candidate.exact_violation,time_candidate.digest,None,elapsed))
            if time_candidate.exact_time is not None and (final is None or time_candidate.exact_time < float(final.exact_time)):
                final=time_candidate

        if selected_time_backend == "filter_sqp":
            filter_safe = bool(
                (checkpoint_candidate is not None)
                or (numerical_projection_best is not None)
                or (
                    time_candidate is not None
                    and time_candidate.strict
                    and time_candidate.exact_time is not None
                )
            )
            if not out.success:
                # A supervised worker/deadline failure is fail-closed: keep the
                # best deterministic checkpoint if one can be certified,
                # otherwise retain the independently owned selected-basin
                # projection.  Do not start a second long optimizer after an
                # infrastructure failure or exhausted deadline.
                time_backend_summary["fallback_action"] = (
                    "owned_filter_checkpoint_or_selected_basin_fallback"
                )
            elif not filter_safe:
                # Healthy filter completion but no planner-safe time-derived
                # result.  The selected-basin fallback is already strict and
                # exact-ranked, so returning it is safe and avoids serially
                # paying SLSQP after an otherwise qualified filter attempt.
                # This exceptional outcome is explicit in telemetry and is a
                # backend-regression signal, not an eligibility expansion.
                time_backend_summary["fallback_action"] = (
                    "selected_basin_fallback_after_uncertified_filter"
                )
            else:
                time_backend_summary["fallback_action"] = "not_needed"
        else:
            time_backend_summary["fallback_action"] = "slsqp_primary"

    geometry=[c for c in strict_candidates if c.source in {"curvature_geometry","alternate_geometry","primary_geometry","primary_projection"}]
    if final is None and schedule.initializer_mode is WarmStartInitializerMode.ANALYTIC_WITH_STRICT_PHASE_ONE_FALLBACK and config.initial_parameters_override is None:
        telemetry.emit({
            "event": "phase_one_fallback_trigger", "run_id": run_id,
            "reason": "no_strict_candidate_after_time_optimization",
        })
        phase_candidate = run_strict_phase_one(required=False)
        if phase_candidate is not None and phase_candidate.strict:
            timeout=_remaining(started,d.exact_ranking_seconds,d.total_pipeline_seconds)
            if timeout is not None:
                name=f"{phase_candidate.source}/exact_ranking"
                req=WarmStartStageRequest("exact_time",name,problem,phase_candidate.parameters,None,aligned_target(phase_candidate.parameters),None,None,None,config.init_w,n_scan=config.n_scan,terminal_w_max=config.terminal_w_max,envelope_scan=config.envelope_scan,domain_scan=config.domain_scan,test_injection=schedule.injection_for("exact_ranking"))
                t=time.perf_counter(); out=_run_stage_request(stage_runner,req,timeout_seconds=timeout); elapsed=time.perf_counter()-t; ranking_seconds+=elapsed
                if out.success and isinstance(out.result,WarmStartStageResult) and out.result.known_time is not None:
                    phase_candidate.exact_time=float(out.result.known_time)
                    emit(WarmStartStageRecord(name,"completed",elapsed,"none",out.result.message,True,None,phase_candidate.endpoint_error,phase_candidate.corridor_upper_bound,phase_candidate.exact_violation,phase_candidate.digest,phase_candidate.exact_time,elapsed))
            if phase_candidate.exact_time is not None:
                final=phase_candidate
                fallback=phase_candidate

    if final is None:
        telemetry.emit({"event":"warm_start_failed","run_id":run_id,"stage":"final_certification","reason":"no_strict_planner_candidate","selected_internal_candidate":selected.source})
        raise RuntimeError("bounded scheduling produced no continuously certified planner candidate")

    basin=None
    if len(geometry)>=2:
        try: basin=compare_basin_candidates(problem,geometry[0].parameters,geometry[1].parameters)
        except BaseException: basin=BasinComparison("comparison failed",None,None,None)
    total=time.perf_counter()-started
    checkpoint_summary = None
    if config.run_full_time:
        checkpoint_objective_calls = int(max(
            [0]
            + [int(item.objective_calls) for item in locals().get("checkpoint_frames", [])]
            + [int(item.objective_calls) for item in locals().get("numerical_checkpoint_frames", [])]
        ))
        checkpoint_gradient_calls = int(max(
            [0]
            + [int(item.gradient_calls) for item in locals().get("checkpoint_frames", [])]
            + [int(item.gradient_calls) for item in locals().get("numerical_checkpoint_frames", [])]
        ))
        if "function_evaluations" not in time_backend_summary and checkpoint_objective_calls:
            time_backend_summary["function_evaluations"] = checkpoint_objective_calls
            time_backend_summary["objective_calls"] = checkpoint_objective_calls
        if "gradient_evaluations" not in time_backend_summary and checkpoint_gradient_calls:
            time_backend_summary["gradient_evaluations"] = checkpoint_gradient_calls
        checkpoint_summary = {
            "time_checkpoint_frames": len(locals().get("checkpoint_frames", [])),
            "time_numerical_checkpoint_frames": len(locals().get("numerical_checkpoint_frames", [])),
            "time_checkpoint_invalid_frames": int(locals().get("checkpoint_invalid_frames", 0)),
            "time_checkpoint_projection_attempts": int(locals().get("numerical_projection_attempts", 0)),
            "time_checkpoint_projection_best": (
                None if locals().get("numerical_projection_best") is None
                else float(locals()["numerical_projection_best"].exact_time)
            ),
            "time_checkpoint_best": (
                None if locals().get("checkpoint_candidate") is None
                else float(locals()["checkpoint_candidate"].exact_time)
            ),
            "time_checkpoint_parent_certification_seconds": float(
                locals().get("checkpoint_parent_certification_seconds", 0.0)
            ),
            "time_checkpoint_worker_certification_seconds": float(max(
                [float(locals().get("full_time_worker_checkpoint_seconds", 0.0))]
                + [float(item.certification_seconds_total) for item in locals().get("checkpoint_frames", [])]
                + [float(item.certification_seconds_total) for item in locals().get("numerical_checkpoint_frames", [])]
            )),
            "time_checkpoint_worker_certification_checks": int(max(
                [int(locals().get("full_time_worker_checkpoint_checks", 0))]
                + [int(item.certification_checks) for item in locals().get("checkpoint_frames", [])]
                + [int(item.certification_checks) for item in locals().get("numerical_checkpoint_frames", [])]
            )),
            "time_checkpoint_worker_emissions": int(max(
                int(locals().get("full_time_worker_checkpoint_emissions", 0)),
                len(locals().get("checkpoint_frames", []))
                + len(locals().get("numerical_checkpoint_frames", [])),
            )),
            "time_checkpoint_objective_calls": checkpoint_objective_calls,
            "time_checkpoint_gradient_calls": checkpoint_gradient_calls,
        }
    record=WarmStartRunRecord(1,run_id,schedule.mode.value,schedule.initializer_mode.value,tuple(records),primary_status,alternate_trigger,selected.source,selected.digest,None if fallback is None else fallback.source,final.source,final.exact_time,final.strict,phase_seconds,primary_seconds,alternate_seconds,projection_seconds,primary_seconds+alternate_seconds,ranking_seconds,full_seconds,total,basin,telemetry.failed,None,checkpoint_summary,time_backend_summary)
    telemetry.emit({"event":"warm_start_complete",**record.to_json_dict()}); record=replace(record,telemetry_failed=record.telemetry_failed or telemetry.failed)
    if final.exact_time is None: raise RuntimeError("final strict candidate has no time")
    return WarmStartExecutionResult(0,final.parameters.copy(),float(final.exact_time),final.source,tuple(records),record,final.pool)


__all__=[
    "BasinComparison","PrimaryFilterSQPPolicySettings","PrimaryTimeBackendMode","TimePolishCheckpoint","TimePolishNumericalCheckpoint","TimePolishCheckpointSettings",
    "WarmStartDeadlineSettings","WarmStartExecutionResult","WarmStartInitializerMode",
    "WarmStartOptimizationConfig","WarmStartRunRecord","WarmStartScheduleSettings","WarmStartSchedulingMode",
    "WarmStartStageRecord","WarmStartStageRequest","WarmStartStageResult","compare_basin_candidates",
    "create_warm_start_stage_runner","execute_bounded_warm_start_schedule","run_supervised_warm_start_stage",
]
