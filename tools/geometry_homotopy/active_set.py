"""Experimental certification-gated active-set optimization for route geometry.

This module is deliberately isolated from the production planner.  It exploits
three facts established by the goal-entry homotopy campaign:

* the analytic ``maximal_runs`` route is already continuously feasible;
* a dense endpoint-seeded rectangle pool is unnecessary while the route is
  comfortably inside the corridor; and
* the sparse filter-SQP backend can cheaply propose a few accepted steps if we
  never expose an uncertified candidate.

The algorithm therefore treats exact continuous certification as a transaction
boundary.  Each trial starts from the last certified checkpoint, runs a *fresh*
short sparse-SQP solve, independently certifies both the finite incumbent and
current filter iterate, carries forward any exact-separation cuts, and promotes
only a strictly better certified point.  Sparse continuation/Hessian state is
intentionally discarded between transactions in this research version; this
proved materially more robust on the historical five-route corpus than carrying
an infeasible filter trajectory across batches.

Nothing here weakens the corridor model.  A sparse/near-active finite pool is
only a proposal mechanism; authoritative continuous separation remains the gate
for every accepted checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import time
from typing import Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from benchmarks.common.certification import certify_parameters_on_problem
from optimization import (
    ConstraintFamily,
    ConstraintPool,
    CurvatureSlopeConstraint,
    ExchangeSettings,
    KnotCoordinateSettings,
    LogLengthKnotMap,
    SLSQPSettings,
    SeparationSettings,
    SparseSQPSettings,
    add_report_violations,
    curvature_energy_value_and_raw_gradient,
    knot_parameters_to_raw,
    pullback_raw_gradient_to_knot_parameters,
    reverse_solver,
    run_constraint_generation,
    stack_vector_constraints,
    scalar_reverse_solver,
    separate_rectangle_path,
)
from optimization import time_model as time_model_dispatch
from planning.maze_routes import RouteOptimizationProblem, stable_knot_bounds

Array = NDArray[np.float64]
PoolStrategy = Literal["empty", "near_active", "dense"]
PoolRetention = Literal["carry", "reseed_after_promotion"]
ObjectiveKind = Literal["curvature", "time"]


class DifferentiableObjective(Protocol):
    def __call__(self, parameters: Sequence[float]) -> tuple[float, Array]: ...


@dataclass(frozen=True, slots=True)
class CurvatureObjective:
    initial_k: float

    def __call__(self, parameters: Sequence[float]) -> tuple[float, Array]:
        x = np.asarray(parameters, dtype=float)
        raw = knot_parameters_to_raw(x, initial_k=self.initial_k)
        value, raw_gradient = curvature_energy_value_and_raw_gradient(raw)
        gradient = pullback_raw_gradient_to_knot_parameters(
            x, raw_gradient, initial_k=self.initial_k
        )
        return float(value), np.asarray(gradient, dtype=float)


@dataclass(frozen=True, slots=True)
class TimeObjective:
    initial_k: float
    init_w: float = 0.8
    terminal_w_max: float | None = None
    n_scan: int = 96
    envelope_scan: int = 48
    domain_scan: int = 96

    def __call__(self, parameters: Sequence[float]) -> tuple[float, Array]:
        x = np.asarray(parameters, dtype=float)
        raw = knot_parameters_to_raw(x, initial_k=self.initial_k)
        evaluation = time_model_dispatch.time_value_and_gradient(
            raw,
            init_w=self.init_w,
            terminal_w_max=self.terminal_w_max,
            initial_k=self.initial_k,
            n_scan=self.n_scan,
            envelope_scan=self.envelope_scan,
            domain_scan=self.domain_scan,
        )
        value, raw_gradient = evaluation.value, evaluation.raw_gradient
        if raw_gradient is None:
            raise RuntimeError("time-model gradient unexpectedly unavailable")
        gradient = pullback_raw_gradient_to_knot_parameters(
            x, raw_gradient, initial_k=self.initial_k
        )
        return float(value), np.asarray(gradient, dtype=float)


@dataclass(frozen=True, slots=True)
class CertifiedActiveSetSettings:
    """Settings for fresh short-batch certified active-set optimization."""

    pool_strategy: PoolStrategy = "near_active"
    near_active_margin: float = 0.05
    # Try a moderately long proposal first, then shorten only when the candidate
    # cannot be certified.  The current best historical-corpus schedule is
    # intentionally a setting rather than a production constant.
    batch_schedule: tuple[int, ...] = (4, 2, 1)
    trust_radius_schedule: tuple[float, ...] = (0.02, 0.01, 0.005)
    maximum_accepted_steps: int = 12
    maximum_trials: int = 96
    maximum_wall_seconds: float = 60.0
    maximum_raw_iterations: int = 100
    maximum_filter_violation: float = 1.0e-4
    feasibility_tolerance: float = 2.0e-7
    curvature_slope_limit: float | None = 50.0
    absolute_improvement_tolerance: float = 1.0e-10
    relative_improvement_tolerance: float = 1.0e-10
    separation_add_tolerance: float = 1.0e-9
    initial_pool_include_midpoints: bool = False
    allow_private_highs_fallback: bool = True
    pool_retention: PoolRetention = "carry"
    use_quadratic_model: bool = True
    highs_time_limit: float = 2.0
    highs_qp_iteration_limit: int = 1000
    highs_threads: int = 1

    def __post_init__(self) -> None:
        if self.pool_strategy not in {"empty", "near_active", "dense"}:
            raise ValueError("unknown pool strategy")
        if self.pool_retention not in {"carry", "reseed_after_promotion"}:
            raise ValueError("unknown pool retention mode")
        if not math.isfinite(self.highs_time_limit) or self.highs_time_limit <= 0.0:
            raise ValueError("highs_time_limit must be finite and positive")
        if self.highs_qp_iteration_limit <= 0:
            raise ValueError("highs_qp_iteration_limit must be positive")
        if self.highs_threads <= 0:
            raise ValueError("highs_threads must be positive")
        if not self.batch_schedule or any(v <= 0 for v in self.batch_schedule):
            raise ValueError("batch_schedule must contain positive integers")
        if not self.trust_radius_schedule or any(
            not math.isfinite(v) or v <= 0.0 for v in self.trust_radius_schedule
        ):
            raise ValueError("trust_radius_schedule must contain positive finite values")
        if self.maximum_accepted_steps <= 0 or self.maximum_trials <= 0:
            raise ValueError("optimization budgets must be positive")
        if not math.isfinite(self.maximum_wall_seconds) or self.maximum_wall_seconds <= 0.0:
            raise ValueError("maximum_wall_seconds must be finite and positive")
        if self.maximum_raw_iterations <= 0:
            raise ValueError("maximum_raw_iterations must be positive")
        if not math.isfinite(self.near_active_margin) or self.near_active_margin < 0.0:
            raise ValueError("near_active_margin must be finite and nonnegative")
        if not math.isfinite(self.feasibility_tolerance) or self.feasibility_tolerance <= 0.0:
            raise ValueError("feasibility_tolerance must be finite and positive")
        if self.separation_add_tolerance > self.feasibility_tolerance:
            raise ValueError("separation_add_tolerance must not exceed feasibility_tolerance")


@dataclass(frozen=True, slots=True)
class ActiveSetTrialRecord:
    checkpoint_index: int
    trial_index: int
    requested_accepted_steps: int
    trust_radius: float
    wall_seconds: float
    pool_size_before: int
    pool_size_after: int
    cuts_added: int
    solver_accepted_steps: int
    solver_rejected_steps: int
    objective_evaluations: int
    constraint_evaluations: int
    objective_seconds: float
    constraint_seconds: float
    subproblem_seconds: float
    finite_objective: float
    current_objective: float
    finite_certified: bool
    current_certified: bool
    selected_source: str | None
    selected_objective: float | None
    selected_endpoint_error: float | None
    selected_corridor_upper_bound: float | None


@dataclass(frozen=True, slots=True)
class ActiveSetCheckpoint:
    index: int
    cumulative_accepted_steps: int
    objective: float
    scalar_time: float
    curvature_energy: float
    pool_size: int
    parameter_sha256: str


@dataclass(frozen=True, slots=True)
class CertifiedActiveSetResult:
    parameters: Array
    objective: float
    scalar_time: float
    curvature_energy: float
    pool: ConstraintPool
    certified: bool
    certification: dict[str, object]
    accepted_steps: int
    trials: tuple[ActiveSetTrialRecord, ...]
    checkpoints: tuple[ActiveSetCheckpoint, ...]
    stop_reason: str
    wall_seconds: float

    def summary_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "scalar_time": self.scalar_time,
            "curvature_energy": self.curvature_energy,
            "certified": self.certified,
            "certification": self.certification,
            "accepted_steps": self.accepted_steps,
            "trial_count": len(self.trials),
            "checkpoint_count": len(self.checkpoints),
            "pool_size": self.pool.size,
            "stop_reason": self.stop_reason,
            "wall_seconds": self.wall_seconds,
            "checkpoints": [asdict(row) for row in self.checkpoints],
            "trials": [asdict(row) for row in self.trials],
        }


def _parameter_digest(x: Sequence[float]) -> str:
    array = np.asarray(x, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _curvature_value(problem: RouteOptimizationProblem, x: Sequence[float]) -> float:
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    return float(curvature_energy_value_and_raw_gradient(raw)[0])


def _scalar_time(
    problem: RouteOptimizationProblem,
    x: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None,
    n_scan: int,
    envelope_scan: int,
    domain_scan: int,
) -> float:
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    return float(
        time_model_dispatch.evaluate_time_scalar(
            raw,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=problem.initial_state.k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
        )
    )


def _certify(
    problem: RouteOptimizationProblem,
    x: Sequence[float],
    settings: CertifiedActiveSetSettings,
    *,
    checkpoint_inequality=None,
) -> dict[str, object]:
    result = certify_parameters_on_problem(
        problem,
        x,
        tolerance=settings.feasibility_tolerance,
        maximum_abs_sigma=settings.curvature_slope_limit,
    )
    auxiliary_upper = -math.inf
    if checkpoint_inequality is not None:
        values, _jacobian = checkpoint_inequality(x)
        array = np.asarray(values, dtype=float)
        auxiliary_upper = float(np.max(array)) if array.size else -math.inf
        result["auxiliary_inequality_upper_bound"] = auxiliary_upper
        result["certified"] = bool(
            result["certified"]
            and auxiliary_upper <= settings.feasibility_tolerance
        )
    return result


def _segment_lengths(x: Sequence[float]) -> Array:
    stations = np.asarray(x, dtype=float)[0::2]
    return stations - np.concatenate(([0.0], stations[:-1]))


def _seed_wall_endpoints(
    pool: ConstraintPool, segment: int, wall: int
) -> int:
    added = 0
    for corner in range(4):
        family = ConstraintFamily(segment, wall, corner)
        values = pool.taus.setdefault(family, [])
        for tau in (0.0, 1.0):
            if tau not in values:
                values.append(tau)
                added += 1
        values.sort()
    return added


def seed_near_active_pool(
    problem: RouteOptimizationProblem,
    x: Sequence[float],
    pool: ConstraintPool,
    *,
    margin: float,
    certificate_tolerance: float,
    add_tolerance: float,
) -> int:
    """Grow ``pool`` with walls close to the current certified trajectory.

    Broad-phase wall upper bounds are sufficient for *selection*.  For walls
    that require exact corner analysis we also retain their actual local maxima.
    The heuristic pool is never itself a certificate.
    """
    report = separate_rectangle_path(
        x,
        problem.initial_state,
        problem.corridor,
        settings=SeparationSettings(
            add_tolerance=add_tolerance,
            certificate_tolerance=certificate_tolerance,
        ),
    )
    lengths = _segment_lengths(x)
    added = 0
    for wall in report.walls:
        if wall.upper_bound <= -margin:
            continue
        added += _seed_wall_endpoints(pool, wall.segment, wall.wall)
        for maximum in wall.maxima:
            added += int(
                pool.add(
                    maximum.family,
                    maximum.tau,
                    segment_length=float(lengths[maximum.family.segment]),
                    merge_distance=1.0e-8,
                )
            )
    return added


def make_initial_pool(
    problem: RouteOptimizationProblem,
    x: Sequence[float],
    settings: CertifiedActiveSetSettings,
) -> ConstraintPool:
    if settings.pool_strategy == "dense":
        return ConstraintPool.seeded(
            problem.corridor,
            include_midpoints=settings.initial_pool_include_midpoints,
        )
    pool = ConstraintPool(problem.corridor)
    if settings.pool_strategy == "near_active":
        seed_near_active_pool(
            problem,
            x,
            pool,
            margin=settings.near_active_margin,
            certificate_tolerance=settings.feasibility_tolerance,
            add_tolerance=settings.separation_add_tolerance,
        )
    return pool


def _sparse_diagnostics(exchange) -> object | None:
    if not exchange.rounds:
        return None
    return exchange.rounds[-1].finite.solver_result.diagnostics


def _objective_improved(new: float, old: float, settings: CertifiedActiveSetSettings) -> bool:
    threshold = max(
        settings.absolute_improvement_tolerance,
        settings.relative_improvement_tolerance * max(1.0, abs(old)),
    )
    return new < old - threshold


def run_certified_active_set(
    problem: RouteOptimizationProblem,
    x0: Sequence[float],
    *,
    objective: DifferentiableObjective,
    objective_kind: ObjectiveKind,
    settings: CertifiedActiveSetSettings = CertifiedActiveSetSettings(),
    pool: ConstraintPool | None = None,
    init_w: float = 0.8,
    terminal_w_max: float | None = None,
    n_scan: int = 96,
    envelope_scan: int = 48,
    domain_scan: int = 96,
    checkpoint_inequality=None,
) -> CertifiedActiveSetResult:
    """Run fresh short sparse-SQP transactions from certified checkpoints."""
    started = time.perf_counter()
    x = np.asarray(x0, dtype=float).copy()
    initial_cert = _certify(problem, x, settings, checkpoint_inequality=checkpoint_inequality)
    if not bool(initial_cert["certified"]):
        raise ValueError("active-set optimizer requires a certified starting point")
    value = float(objective(x)[0])
    active_pool = (
        make_initial_pool(problem, x, settings)
        if pool is None
        else pool.copy()
    )
    if active_pool.corridor != problem.corridor:
        raise ValueError("provided active pool belongs to another corridor")

    slope_only = (
        None
        if settings.curvature_slope_limit is None
        else CurvatureSlopeConstraint(
            settings.curvature_slope_limit,
            initial_k=problem.initial_state.k,
        )
    )
    slope_constraint = problem.inequalities(
        stack_vector_constraints(slope_only, checkpoint_inequality)
    )
    separation = SeparationSettings(
        add_tolerance=settings.separation_add_tolerance,
        certificate_tolerance=settings.feasibility_tolerance,
    )
    trials: list[ActiveSetTrialRecord] = []
    checkpoints: list[ActiveSetCheckpoint] = []
    accepted_steps = 0
    trial_index = 0

    def append_checkpoint() -> None:
        checkpoints.append(
            ActiveSetCheckpoint(
                index=len(checkpoints),
                cumulative_accepted_steps=accepted_steps,
                objective=value,
                scalar_time=_scalar_time(
                    problem,
                    x,
                    init_w=init_w,
                    terminal_w_max=terminal_w_max,
                    n_scan=n_scan,
                    envelope_scan=envelope_scan,
                    domain_scan=domain_scan,
                ),
                curvature_energy=_curvature_value(problem, x),
                pool_size=active_pool.size,
                parameter_sha256=_parameter_digest(x),
            )
        )

    append_checkpoint()
    stop_reason = "maximum accepted-step budget reached"

    while accepted_steps < settings.maximum_accepted_steps:
        elapsed = time.perf_counter() - started
        if elapsed >= settings.maximum_wall_seconds:
            stop_reason = "wall budget reached"
            break
        if trial_index >= settings.maximum_trials:
            stop_reason = "trial budget reached"
            break

        remaining = settings.maximum_accepted_steps - accepted_steps
        promoted = False
        for requested in settings.batch_schedule:
            batch_steps = min(int(requested), remaining)
            if batch_steps <= 0:
                continue
            for trust_radius in settings.trust_radius_schedule:
                if trial_index >= settings.maximum_trials:
                    break
                if time.perf_counter() - started >= settings.maximum_wall_seconds:
                    break
                trial_index += 1
                pool_before = active_pool.size
                sparse = SparseSQPSettings(
                    max_iterations=settings.maximum_raw_iterations,
                    max_accepted_steps=batch_steps,
                    initial_trust_radius=float(trust_radius),
                    maximum_trust_radius=float(trust_radius),
                    maximum_filter_violation=settings.maximum_filter_violation,
                    feasibility_tolerance=settings.feasibility_tolerance,
                    equality_tolerance=settings.feasibility_tolerance,
                    use_quadratic_model=settings.use_quadratic_model,
                    highs_time_limit=settings.highs_time_limit,
                    highs_qp_iteration_limit=settings.highs_qp_iteration_limit,
                    highs_threads=settings.highs_threads,
                    final_restoration_iterations=0,
                    jacobian_mode="direct_sparse",
                    allow_private_highs_fallback=settings.allow_private_highs_fallback,
                )
                exchange_settings = ExchangeSettings(
                    maximum_rounds=1,
                    finite_inequality_tolerance=settings.feasibility_tolerance,
                    finite_equality_tolerance=settings.feasibility_tolerance,
                    require_solver_success=False,
                    finite_solver="sparse_sqp",
                    final_kkt_diagnostics=False,
                    separation=separation,
                    sparse_sqp=sparse,
                    slsqp=SLSQPSettings(max_iterations=60, ftol=1.0e-10),
                )
                trial_started = time.perf_counter()
                exchange = run_constraint_generation(
                    x,
                    objective,
                    problem.corridor,
                    problem.initial_state,
                    bounds=stable_knot_bounds(x),
                    endpoint_target=problem.endpoint_target,
                    additional_inequalities=slope_constraint,
                    pool=active_pool,
                    settings=exchange_settings,
                    coordinate_map=LogLengthKnotMap.from_knot_parameters(x),
                    coordinate_settings=KnotCoordinateSettings(),
                )
                trial_wall = time.perf_counter() - trial_started
                active_pool = exchange.pool

                # The finite return already contributes its exact violating cuts.
                # The filter's current iterate can contain additional violations;
                # record them before rolling back to the safe checkpoint.
                if exchange.current_x is not None:
                    current_report = separate_rectangle_path(
                        exchange.current_x,
                        problem.initial_state,
                        problem.corridor,
                        settings=separation,
                    )
                    add_report_violations(
                        active_pool,
                        current_report,
                        _segment_lengths(exchange.current_x),
                        merge_distance=exchange_settings.merge_distance,
                    )

                finite_cert = _certify(problem, exchange.x, settings, checkpoint_inequality=checkpoint_inequality)
                current_cert = (
                    None
                    if exchange.current_x is None
                    else _certify(problem, exchange.current_x, settings, checkpoint_inequality=checkpoint_inequality)
                )
                candidates: list[tuple[float, Array, str, dict[str, object]]] = []
                if bool(finite_cert["certified"]) and _objective_improved(
                    float(exchange.objective), value, settings
                ):
                    candidates.append(
                        (float(exchange.objective), exchange.x.copy(), "finite", finite_cert)
                    )
                if (
                    exchange.current_x is not None
                    and current_cert is not None
                    and bool(current_cert["certified"])
                    and math.isfinite(float(exchange.current_objective))
                    and _objective_improved(
                        float(exchange.current_objective), value, settings
                    )
                ):
                    candidates.append(
                        (
                            float(exchange.current_objective),
                            exchange.current_x.copy(),
                            "current",
                            current_cert,
                        )
                    )
                selected = min(candidates, key=lambda row: row[0]) if candidates else None

                diag = _sparse_diagnostics(exchange)
                solver_accepted = int(getattr(diag, "accepted_steps", 0))
                solver_rejected = int(getattr(diag, "rejected_steps", 0))
                objective_evals = int(getattr(diag, "objective_evaluations", 0))
                constraint_evals = int(getattr(diag, "constraint_evaluations", 0))
                objective_seconds = float(getattr(diag, "objective_seconds", 0.0))
                constraint_seconds = float(getattr(diag, "constraint_seconds", 0.0))
                stage = getattr(diag, "stage_telemetry", None)
                subproblem_seconds = 0.0
                if stage is not None:
                    totals = getattr(stage, "totals", None)
                    if isinstance(totals, dict):
                        subproblem_seconds = float(
                            sum(
                                float(v)
                                for k, v in totals.items()
                                if "normal" in str(k).lower()
                                or "tangential" in str(k).lower()
                                or "subproblem" in str(k).lower()
                            )
                        )
                round_row = exchange.rounds[-1]
                selected_cert = None if selected is None else selected[3]
                trials.append(
                    ActiveSetTrialRecord(
                        checkpoint_index=len(checkpoints) - 1,
                        trial_index=trial_index,
                        requested_accepted_steps=batch_steps,
                        trust_radius=float(trust_radius),
                        wall_seconds=trial_wall,
                        pool_size_before=pool_before,
                        pool_size_after=active_pool.size,
                        cuts_added=active_pool.size - pool_before,
                        solver_accepted_steps=solver_accepted,
                        solver_rejected_steps=solver_rejected,
                        objective_evaluations=objective_evals,
                        constraint_evaluations=constraint_evals,
                        objective_seconds=objective_seconds,
                        constraint_seconds=constraint_seconds,
                        subproblem_seconds=subproblem_seconds,
                        finite_objective=float(exchange.objective),
                        current_objective=float(exchange.current_objective),
                        finite_certified=bool(finite_cert["certified"]),
                        current_certified=bool(
                            current_cert is not None and current_cert["certified"]
                        ),
                        selected_source=None if selected is None else selected[2],
                        selected_objective=None if selected is None else selected[0],
                        selected_endpoint_error=(
                            None
                            if selected_cert is None
                            else float(selected_cert["endpoint_error"])
                        ),
                        selected_corridor_upper_bound=(
                            None
                            if selected_cert is None
                            else float(selected_cert["corridor_upper_bound"])
                        ),
                    )
                )

                if selected is None:
                    continue

                value, x, _source, _selected_cert = selected
                # Count actual accepted steps when diagnostics are present.  A
                # successful returned/current candidate always follows at least
                # one accepted filter step.
                accepted_steps += max(1, solver_accepted)
                accepted_steps = min(accepted_steps, settings.maximum_accepted_steps)
                if settings.pool_strategy == "near_active":
                    if settings.pool_retention == "reseed_after_promotion":
                        # The promoted point is independently continuously certified, so
                        # stale finite cuts are not part of the correctness argument.
                        # Rebuild only the currently near-active proposal set; rejected
                        # trials at this checkpoint still share cuts until promotion.
                        active_pool = make_initial_pool(problem, x, settings)
                    else:
                        seed_near_active_pool(
                            problem,
                            x,
                            active_pool,
                            margin=settings.near_active_margin,
                            certificate_tolerance=settings.feasibility_tolerance,
                            add_tolerance=settings.separation_add_tolerance,
                        )
                append_checkpoint()
                promoted = True
                break
            if promoted:
                break
        if not promoted:
            stop_reason = "no improving certified trial"
            break

    final_cert = _certify(problem, x, settings, checkpoint_inequality=checkpoint_inequality)
    final_time = _scalar_time(
        problem,
        x,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_scan=domain_scan,
    )
    final_curvature = _curvature_value(problem, x)
    return CertifiedActiveSetResult(
        parameters=x.copy(),
        objective=float(value),
        scalar_time=final_time,
        curvature_energy=final_curvature,
        pool=active_pool.copy(),
        certified=bool(final_cert["certified"]),
        certification=final_cert,
        accepted_steps=accepted_steps,
        trials=tuple(trials),
        checkpoints=tuple(checkpoints),
        stop_reason=stop_reason,
        wall_seconds=time.perf_counter() - started,
    )


__all__ = [
    "ActiveSetCheckpoint",
    "ActiveSetTrialRecord",
    "CertifiedActiveSetResult",
    "CertifiedActiveSetSettings",
    "CurvatureObjective",
    "TimeObjective",
    "make_initial_pool",
    "run_certified_active_set",
    "seed_near_active_pool",
]
