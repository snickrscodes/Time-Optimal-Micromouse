"""Sparse filter-SQP prototype using HiGHS linear subproblems.

The implementation is deliberately backend-neutral.  It operates on an
objective exposing ``evaluate(x) -> (value, gradient)`` and a constraint oracle
exposing ``evaluate(x) -> (c, Jc, h, Jh)`` with the sign convention
``c(x) <= 0`` and ``h(x) == 0``.

This candidate uses an infinity-norm trust region and modular HiGHS
subproblems:

* a lexicographic elastic normal-step LP that minimizes the maximum
  linearized violation and then the L1 step norm;
* a convex damped-BFGS tangential QP (or optional LP model) that minimizes the
  objective model under a configurable relaxed inequality cap;
* nonlinear second-order corrections and a private final restoration phase.

Nonlinear trial points are screened with the cheap constraints before the
objective is evaluated.  A Fletcher--Leyffer style filter accepts either
objective progress or feasibility progress, while a separately retained
feasible incumbent guarantees that the caller never has to trade a valid path
for an attractive but infeasible trial.

The public ``highspy`` package is preferred. Candidate builds may explicitly
fall back to SciPy's bundled private extension; the selected backend is exposed
in every result so a release can require the public API.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class ObjectiveEvaluationTimeout(TimeoutError):
    """Cooperative objective deadline exceeded before a value was committed."""

    def __init__(self, stage: str = "objective", message: str | None = None) -> None:
        self.stage = str(stage)
        super().__init__(message or f"{self.stage} evaluation exceeded its soft deadline")


@dataclass(frozen=True, slots=True)
class SparseSQPDiagnosticControls:
    """Independent controls for expensive first-order diagnostics."""

    multipliers_on_initial_point: bool = True
    multipliers_on_accepted_steps: bool = True
    criticality_on_accepted_steps: bool = False
    multiplier_termination: bool = True
    final_multipliers: bool = True
    final_criticality: bool = True
    strict_convergence_requires_all: bool = True

    @classmethod
    def intermediate(
        cls,
        *,
        criticality: bool = False,
        accepted_step_criticality: bool = False,
    ) -> "SparseSQPDiagnosticControls":
        return cls(
            multipliers_on_initial_point=False,
            multipliers_on_accepted_steps=False,
            criticality_on_accepted_steps=accepted_step_criticality,
            multiplier_termination=False,
            final_multipliers=False,
            final_criticality=criticality,
            strict_convergence_requires_all=True,
        )

    @classmethod
    def restoration(cls) -> "SparseSQPDiagnosticControls":
        return cls.intermediate(criticality=False)


@dataclass(frozen=True, slots=True)
class SparseSQPStageTelemetry:
    """Serializable aggregate timings and call counts for one solver call."""

    totals: tuple[tuple[str, float], ...] = ()
    calls: tuple[tuple[str, int], ...] = ()
    maxima: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_dicts(
        cls,
        totals: dict[str, float],
        calls: dict[str, int],
        maxima: dict[str, float],
    ) -> "SparseSQPStageTelemetry":
        return cls(
            tuple(sorted((key, float(value)) for key, value in totals.items())),
            tuple(sorted((key, int(value)) for key, value in calls.items())),
            tuple(sorted((key, float(value)) for key, value in maxima.items())),
        )

    def total(self, stage: str) -> float:
        return dict(self.totals).get(stage, 0.0)

    def count(self, stage: str) -> int:
        return dict(self.calls).get(stage, 0)


@dataclass(frozen=True, slots=True)
class SparseSQPContinuationState:
    """Logical state required to resume a short sparse-SQP batch safely.

    Native HiGHS objects may be retained by a live worker in ``warm_starts``.
    They are deliberately removed by :meth:`checkpoint_safe_copy`; logical
    correctness never depends on serializing a private or public solver handle.
    """

    current_x: Array
    current_objective: float
    current_gradient: Array
    current_inequalities: Array
    current_inequality_jacobian: object
    current_equalities: Array
    current_equality_jacobian: object
    best_feasible_x: Array | None
    best_feasible_objective: float
    best_feasible_gradient: Array | None
    best_feasible_violation: float
    certified_x: Array | None
    certified_objective: float
    certified_gradient: Array | None
    hessian: object
    trust_radius: float
    filter_entries: tuple[tuple[float, float], ...]
    warm_starts: dict[str, object]
    successful_objective_latencies: tuple[float, ...]
    accepted_steps: int
    raw_iterations: int
    rejected_steps: int
    restoration_steps: int
    constraint_generation: int
    objective_identity: str | None
    diagnostic_generation: int
    last_event: str
    native_warm_starts_retained: bool = True
    scaling_shift: Array | None = None
    scaling_variable: Array | None = None
    scaling_inequality: Array | None = None
    scaling_equality: Array | None = None
    scaling_lower: Array | None = None
    scaling_upper: Array | None = None

    def checkpoint_safe_copy(self) -> "SparseSQPContinuationState":
        return replace(
            self,
            warm_starts={},
            native_warm_starts_retained=False,
        )

    def with_certified_incumbent(
        self,
        x: Sequence[float],
        objective: float,
        gradient: Sequence[float] | None = None,
    ) -> "SparseSQPContinuationState":
        array = np.asarray(x, dtype=float).copy()
        grad = None if gradient is None else np.asarray(gradient, dtype=float).copy()
        return replace(
            self,
            certified_x=array,
            certified_objective=float(objective),
            certified_gradient=grad,
        )


class DifferentiableObjective(Protocol):
    def evaluate(self, x: Sequence[float]) -> tuple[float, Array]: ...


class DifferentiableConstraints(Protocol):
    def evaluate(
        self, x: Sequence[float]
    ) -> tuple[Array, Array, Array, Array]: ...


@dataclass(frozen=True, slots=True)
class SparseSQPSettings:
    """Settings for the experimental sparse filter-SQP backend.

    All trust-region radii are expressed in the already-scaled solver
    coordinates.  ``maximum_filter_violation`` is intentionally larger than
    the final feasibility tolerances: it permits a short infeasible excursion
    while the best feasible incumbent is retained separately.
    """

    max_iterations: int = 30
    max_accepted_steps: int | None = None
    initial_trust_radius: float = 4.0e-2
    minimum_trust_radius: float = 1.0e-6
    maximum_trust_radius: float = 8.0e-2
    trust_contraction: float = 0.5
    trust_expansion: float = 1.5
    good_reduction_ratio: float = 0.75
    poor_reduction_ratio: float = 0.1
    acceptance_ratio: float = 1.0e-4
    maximum_filter_violation: float = 1.0e-3
    filter_tightening_factor: float = 0.1
    minimum_filter_tolerance_factor: float = 10.0
    feasibility_tolerance: float = 1.0e-8
    equality_tolerance: float = 1.0e-8
    filter_gamma_violation: float = 1.0e-3
    filter_gamma_objective: float = 1.0e-5
    minimum_predicted_reduction: float = 1.0e-10
    operational_polish_enabled: bool = True
    operational_polish_minimum_predicted_reduction: float = 3.0e-13
    operational_polish_max_iterations: int = 60
    maximum_line_search_steps: int = 9
    line_search_contraction: float = 0.5
    restoration_reduction: float = 1.0e-2
    normal_model_margin: float = 1.0e-10
    use_quadratic_model: bool = True
    fallback_to_linear_model_on_qp_failure: bool = True
    hessian_model: Literal["auto", "dense", "banded", "diagonal"] = "auto"
    hessian_bandwidth: int = 8
    dense_hessian_threshold: int = 128
    use_lagrangian_bfgs: bool = False
    multiplier_active_tolerance: float = 1.0e-5
    kkt_tolerance: float = 1.0e-8
    complementarity_tolerance: float = 1.0e-9
    kkt_check_interval: int = 1
    jacobian_mode: Literal["dense", "dense_to_csr", "direct_sparse"] = "dense_to_csr"
    warm_start_highs: bool = True
    warm_start_qp: bool = False
    allow_private_highs_fallback: bool = True
    initial_hessian_scale: float = 1.0
    automatic_hessian_scale: bool = True
    minimum_hessian_eigenvalue: float = 1.0e-4
    maximum_hessian_eigenvalue: float = 1.0e4
    bfgs_curvature_tolerance: float = 1.0e-10
    enable_second_order_correction: bool = True
    second_order_correction_before_backtracking: bool = True
    second_order_radius_fraction: float = 0.5
    objective_aware_final_restoration: bool = True
    final_restoration_iterations: int = 10
    final_restoration_trust_radius: float = 1.0e-1
    final_restoration_reduction: float = 1.0e-2
    highs_presolve: bool = True
    highs_time_limit: float = 2.0
    highs_qp_iteration_limit: int = 1000
    highs_threads: int | None = None
    require_certified_qp_optimality: bool = True
    retry_uncertified_qp_cold: bool = True
    qp_stationarity_tolerance: float = 5.0e-7
    qp_complementarity_tolerance: float = 5.0e-8
    criticality_radius: float = 1.0e-3
    criticality_secondary_radius: float = 1.0e-2
    criticality_lp_optimality_tolerance: float = 1.0e-9
    criticality_lp_retry_optimality_tolerance: float = 1.0e-8
    criticality_retry_without_presolve: bool = True
    criticality_tolerance: float = 1.0e-7
    polish_criticality_tolerance: float = 1.0e-5
    check_initial_kkt: bool = False
    diagnostic_only: bool = False
    diagnostics: SparseSQPDiagnosticControls = SparseSQPDiagnosticControls()
    objective_soft_deadline_enabled: bool = False
    objective_soft_deadline_minimum: float = 5.0
    objective_soft_deadline_p95_multiplier: float = 4.0
    objective_latency_history_size: int = 64
    display: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_accepted_steps is not None and self.max_accepted_steps <= 0:
            raise ValueError("max_accepted_steps must be positive when supplied")
        if self.maximum_line_search_steps <= 0:
            raise ValueError("maximum_line_search_steps must be positive")
        if self.final_restoration_iterations < 0:
            raise ValueError("final_restoration_iterations must be nonnegative")
        if self.hessian_model not in {"auto", "dense", "banded", "diagonal"}:
            raise ValueError("invalid hessian_model")
        if self.jacobian_mode not in {"dense", "dense_to_csr", "direct_sparse"}:
            raise ValueError("invalid jacobian_mode")
        if self.hessian_bandwidth < 0:
            raise ValueError("hessian_bandwidth must be nonnegative")
        if self.dense_hessian_threshold < 0:
            raise ValueError("dense_hessian_threshold must be nonnegative")
        if self.kkt_check_interval <= 0:
            raise ValueError("kkt_check_interval must be positive")
        if self.highs_qp_iteration_limit <= 0:
            raise ValueError("highs_qp_iteration_limit must be positive")
        if self.highs_threads is not None and self.highs_threads <= 0:
            raise ValueError("highs_threads must be positive when supplied")
        if self.operational_polish_max_iterations <= 0:
            raise ValueError("operational_polish_max_iterations must be positive")
        positive = {
            "initial_trust_radius": self.initial_trust_radius,
            "minimum_trust_radius": self.minimum_trust_radius,
            "maximum_trust_radius": self.maximum_trust_radius,
            "trust_contraction": self.trust_contraction,
            "trust_expansion": self.trust_expansion,
            "maximum_filter_violation": self.maximum_filter_violation,
            "minimum_filter_tolerance_factor": self.minimum_filter_tolerance_factor,
            "feasibility_tolerance": self.feasibility_tolerance,
            "equality_tolerance": self.equality_tolerance,
            "filter_gamma_violation": self.filter_gamma_violation,
            "filter_gamma_objective": self.filter_gamma_objective,
            "minimum_predicted_reduction": self.minimum_predicted_reduction,
            "operational_polish_minimum_predicted_reduction": (
                self.operational_polish_minimum_predicted_reduction
            ),
            "line_search_contraction": self.line_search_contraction,
            "restoration_reduction": self.restoration_reduction,
            "normal_model_margin": self.normal_model_margin,
            "initial_hessian_scale": self.initial_hessian_scale,
            "minimum_hessian_eigenvalue": self.minimum_hessian_eigenvalue,
            "maximum_hessian_eigenvalue": self.maximum_hessian_eigenvalue,
            "bfgs_curvature_tolerance": self.bfgs_curvature_tolerance,
            "multiplier_active_tolerance": self.multiplier_active_tolerance,
            "kkt_tolerance": self.kkt_tolerance,
            "complementarity_tolerance": self.complementarity_tolerance,
            "second_order_radius_fraction": self.second_order_radius_fraction,
            "final_restoration_trust_radius": self.final_restoration_trust_radius,
            "final_restoration_reduction": self.final_restoration_reduction,
            "highs_time_limit": self.highs_time_limit,
            "qp_stationarity_tolerance": self.qp_stationarity_tolerance,
            "qp_complementarity_tolerance": self.qp_complementarity_tolerance,
            "criticality_radius": self.criticality_radius,
            "criticality_secondary_radius": self.criticality_secondary_radius,
            "criticality_lp_optimality_tolerance": (
                self.criticality_lp_optimality_tolerance
            ),
            "criticality_lp_retry_optimality_tolerance": (
                self.criticality_lp_retry_optimality_tolerance
            ),
            "criticality_tolerance": self.criticality_tolerance,
            "polish_criticality_tolerance": self.polish_criticality_tolerance,
            "objective_soft_deadline_minimum": self.objective_soft_deadline_minimum,
            "objective_soft_deadline_p95_multiplier": (
                self.objective_soft_deadline_p95_multiplier
            ),
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.minimum_trust_radius <= self.initial_trust_radius:
            raise ValueError("minimum_trust_radius exceeds initial_trust_radius")
        if not self.initial_trust_radius <= self.maximum_trust_radius:
            raise ValueError("initial_trust_radius exceeds maximum_trust_radius")
        if not 0.0 < self.trust_contraction < 1.0:
            raise ValueError("trust_contraction must lie in (0, 1)")
        if not 0.0 < self.filter_tightening_factor < 1.0:
            raise ValueError("filter_tightening_factor must lie in (0, 1)")
        if self.trust_expansion <= 1.0:
            raise ValueError("trust_expansion must exceed one")
        if not 0.0 <= self.poor_reduction_ratio < self.good_reduction_ratio:
            raise ValueError("reduction-ratio thresholds are inconsistent")
        if self.good_reduction_ratio > 1.0:
            raise ValueError("good_reduction_ratio must not exceed one")
        if not 0.0 < self.acceptance_ratio < 1.0:
            raise ValueError("acceptance_ratio must lie in (0, 1)")
        if not 0.0 < self.line_search_contraction < 1.0:
            raise ValueError("line_search_contraction must lie in (0, 1)")
        if not 0.0 < self.second_order_radius_fraction <= 1.0:
            raise ValueError("second_order_radius_fraction must lie in (0, 1]")
        if self.minimum_hessian_eigenvalue > self.maximum_hessian_eigenvalue:
            raise ValueError("Hessian eigenvalue bounds are inconsistent")
        if (
            self.criticality_lp_retry_optimality_tolerance
            < self.criticality_lp_optimality_tolerance
        ):
            raise ValueError(
                "criticality LP retry tolerance must not be tighter than "
                "the primary tolerance"
            )
        if self.objective_latency_history_size <= 0:
            raise ValueError("objective_latency_history_size must be positive")
        if (
            self.operational_polish_enabled
            and self.diagnostics.final_multipliers
            and self.diagnostics.final_criticality
            and self.operational_polish_minimum_predicted_reduction
            > self.minimum_predicted_reduction
        ):
            raise ValueError(
                "operational polish reduction floor must not exceed the primary floor"
            )


@dataclass(frozen=True, slots=True)
class SparseSQPIteration:
    index: int
    objective: float
    violation: float
    trust_radius: float
    accepted: bool
    step_kind: str
    step_norm_inf: float
    line_search_alpha: float
    predicted_reduction: float
    actual_reduction: float
    reduction_ratio: float
    trial_violation: float
    normal_lp_status: int
    tangential_lp_status: int
    second_order_correction: bool
    objective_evaluated: bool


@dataclass(frozen=True, slots=True)
class SparseSQPResult:
    x: Array
    objective: float
    gradient: Array
    current_x: Array
    current_objective: float
    current_gradient: Array
    success: bool
    status: int
    message: str
    iterations: int
    objective_evaluations: int
    gradient_evaluations: int
    objective_failures: int
    constraint_evaluations: int
    linear_programs: int
    quadratic_programs: int
    normal_linear_programs: int
    tangential_linear_programs: int
    second_order_linear_programs: int
    accepted_steps: int
    rejected_steps: int
    restoration_steps: int
    screened_trials: int
    objective_seconds: float
    constraint_seconds: float
    linear_program_seconds: float
    wall_seconds: float
    final_violation: float
    current_violation: float
    best_feasible_violation: float
    trust_radius: float
    converged: bool
    usable: bool
    first_order_acceptable: bool
    kkt_stationarity: float
    kkt_stationarity_scaled: float
    kkt_complementarity: float
    kkt_complementarity_scaled: float
    kkt_criticality: float
    kkt_criticality_success: bool
    hessian_nonzeros: int
    jacobian_nonzeros: int
    warm_start_attempts: int
    warm_start_uses: int
    qp_rejections: int
    qp_cold_retries: int
    qp_lp_fallbacks: int
    highs_backend: str
    highs_public_api: bool
    history: tuple[SparseSQPIteration, ...]
    continuation_state: SparseSQPContinuationState
    stage_telemetry: SparseSQPStageTelemetry
    multiplier_diagnostic_calls: int
    multiplier_diagnostic_seconds: float
    criticality_diagnostic_calls: int
    criticality_diagnostic_seconds: float
    diagnostics_complete: bool
    diagnostic_failure: str | None
    objective_timeouts: int
    last_timeout_stage: str | None
    accepted_step_budget_reached: bool


@dataclass(slots=True)
class _EvaluationCounters:
    jacobian_mode: Literal["dense", "dense_to_csr", "direct_sparse"] = "dense_to_csr"
    soft_deadline_enabled: bool = False
    soft_deadline_minimum: float = 5.0
    soft_deadline_p95_multiplier: float = 4.0
    latency_history_size: int = 64
    objective_evaluations: int = 0
    gradient_evaluations: int = 0
    objective_failures: int = 0
    objective_timeouts: int = 0
    constraint_evaluations: int = 0
    objective_seconds: float = 0.0
    constraint_seconds: float = 0.0
    maximum_jacobian_nonzeros: int = 0
    successful_objective_latencies: list[float] = field(default_factory=list)
    last_timeout_stage: str | None = None
    stage_totals: dict[str, float] = field(default_factory=dict)
    stage_calls: dict[str, int] = field(default_factory=dict)
    stage_maxima: dict[str, float] = field(default_factory=dict)

    def record_stage(self, stage: str, elapsed: float) -> None:
        self.stage_totals[stage] = self.stage_totals.get(stage, 0.0) + elapsed
        self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
        self.stage_maxima[stage] = max(self.stage_maxima.get(stage, 0.0), elapsed)

    def objective_soft_deadline(self) -> float:
        if self.successful_objective_latencies:
            p95 = float(np.quantile(self.successful_objective_latencies, 0.95))
            return max(
                self.soft_deadline_minimum,
                self.soft_deadline_p95_multiplier * p95,

            )
        return self.soft_deadline_minimum

    def objective(
        self, objective: DifferentiableObjective, x: Array
    ) -> tuple[float, Array]:
        started = time.perf_counter()
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        try:
            if self.soft_deadline_enabled and hasattr(objective, "evaluate_with_deadline"):
                deadline = time.monotonic() + self.objective_soft_deadline()
                value, gradient_in = objective.evaluate_with_deadline(  # type: ignore[attr-defined]
                    x, deadline=deadline
                )
            else:
                value, gradient_in = objective.evaluate(x)
            value = float(value)
            gradient = np.asarray(gradient_in, dtype=float)
            if gradient.shape != x.shape:
                raise ValueError("objective gradient has wrong shape")
            if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("objective returned nonfinite data")
            elapsed = time.perf_counter() - started
            self.successful_objective_latencies.append(elapsed)
            del self.successful_objective_latencies[:-self.latency_history_size]
            self.record_stage("objective", elapsed)
            return value, gradient.copy()
        except TimeoutError as exc:
            self.objective_failures += 1
            self.objective_timeouts += 1
            self.last_timeout_stage = str(getattr(exc, "stage", "objective"))
            if isinstance(exc, ObjectiveEvaluationTimeout):
                raise
            raise ObjectiveEvaluationTimeout(self.last_timeout_stage, str(exc)) from exc
        except Exception:
            self.objective_failures += 1
            raise
        finally:
            self.objective_seconds += time.perf_counter() - started

    def constraints(
        self, constraints: DifferentiableConstraints, x: Array
    ) -> tuple[Array, object, Array, object]:
        from scipy.sparse import csr_matrix, issparse

        started = time.perf_counter()
        self.constraint_evaluations += 1
        try:
            if (
                self.jacobian_mode == "direct_sparse"
                and hasattr(constraints, "evaluate_sparse")
            ):
                c, jc, h, jh = constraints.evaluate_sparse(x)  # type: ignore[attr-defined]
            else:
                c, jc, h, jh = constraints.evaluate(x)
            c = np.asarray(c, dtype=float)
            h = np.asarray(h, dtype=float)
            if c.ndim != 1 or h.ndim != 1:
                raise ValueError("constraint values must be one-dimensional")
            if issparse(jc):
                jc = csr_matrix(jc, dtype=float)
                jc_finite = np.all(np.isfinite(jc.data))
                jc_nnz = int(jc.nnz)
            else:
                jc = np.asarray(jc, dtype=float)
                jc_finite = np.all(np.isfinite(jc))
                jc_nnz = int(np.count_nonzero(jc))
            if issparse(jh):
                jh = csr_matrix(jh, dtype=float)
                jh_finite = np.all(np.isfinite(jh.data))
                jh_nnz = int(jh.nnz)
            else:
                jh = np.asarray(jh, dtype=float)
                jh_finite = np.all(np.isfinite(jh))
                jh_nnz = int(np.count_nonzero(jh))
            if jc.shape != (c.size, x.size):
                raise ValueError("inequality Jacobian has wrong shape")
            if jh.shape != (h.size, x.size):
                raise ValueError("equality Jacobian has wrong shape")
            if not (
                np.all(np.isfinite(c))
                and jc_finite
                and np.all(np.isfinite(h))
                and jh_finite
            ):
                raise FloatingPointError("constraints returned nonfinite data")
            self.maximum_jacobian_nonzeros = max(
                self.maximum_jacobian_nonzeros, jc_nnz + jh_nnz
            )
            if self.jacobian_mode == "dense_to_csr":
                if not issparse(jc):
                    jc = csr_matrix(jc)
                if not issparse(jh):
                    jh = csr_matrix(jh)
            return c.copy(), jc.copy(), h.copy(), jh.copy()
        finally:
            elapsed = time.perf_counter() - started
            self.constraint_seconds += elapsed
            self.record_stage("constraints", elapsed)



@dataclass(slots=True)
class _SubproblemState:
    warm_starts: dict[str, object] = field(default_factory=dict)
    warm_start_attempts: int = 0
    warm_start_uses: int = 0
    qp_rejections: int = 0
    qp_cold_retries: int = 0
    qp_lp_fallbacks: int = 0
    backend_name: str = "unknown"
    backend_public: bool = False


@dataclass(slots=True)
class _Filter:
    gamma_violation: float
    gamma_objective: float
    entries: list[tuple[float, float]] = field(default_factory=list)

    def acceptable(self, violation: float, objective: float) -> bool:
        for old_violation, old_objective in self.entries:
            if not (
                violation <= (1.0 - self.gamma_violation) * old_violation
                or objective <= old_objective
                - self.gamma_objective * max(
                    old_violation, np.finfo(float).eps
                )
            ):
                return False
        return True

    def add(self, violation: float, objective: float) -> None:
        kept: list[tuple[float, float]] = []
        for old_violation, old_objective in self.entries:
            dominated = (
                old_violation >= violation
                and old_objective >= objective
                and (old_violation > violation or old_objective > objective)
            )
            if not dominated:
                kept.append((old_violation, old_objective))
        kept.append((float(violation), float(objective)))
        self.entries = kept


def _violation(c: Array, h: Array) -> float:
    inequality = float(np.max(c, initial=-math.inf)) if c.size else -math.inf
    equality = float(np.max(np.abs(h), initial=0.0)) if h.size else 0.0
    return max(0.0, inequality, equality)


def _is_feasible(c: Array, h: Array, settings: SparseSQPSettings) -> bool:
    max_c = float(np.max(c, initial=-math.inf)) if c.size else -math.inf
    max_h = float(np.max(np.abs(h), initial=0.0)) if h.size else 0.0
    return max_c <= settings.feasibility_tolerance and max_h <= settings.equality_tolerance


@dataclass(frozen=True, slots=True)
class _MultiplierEstimate:
    active_inequalities: NDArray[np.intp]
    inequality_multipliers: Array
    equality_multipliers: Array
    active_lower: NDArray[np.intp]
    lower_multipliers: Array
    active_upper: NDArray[np.intp]
    upper_multipliers: Array
    stationarity_inf: float
    stationarity_scaled: float
    complementarity_inf: float
    complementarity_scaled: float
    success: bool


def _dense_rows(matrix: object, rows: NDArray[np.intp] | None = None) -> Array:
    from scipy.sparse import issparse

    selected = matrix if rows is None else matrix[rows]
    if issparse(selected):
        return np.asarray(selected.toarray(), dtype=float)
    return np.asarray(selected, dtype=float)


def _estimate_multipliers(
    x: Array,
    gradient: Array,
    c: Array,
    jc: object,
    h: Array,
    jh: object,
    lower: Array,
    upper: Array,
    *,
    active_tolerance: float,
) -> _MultiplierEstimate:
    from .kkt_lsq import solve_mixed_least_squares

    active = np.flatnonzero(c >= -active_tolerance).astype(np.intp)
    lower_active = np.flatnonzero(
        np.isfinite(lower)
        & (x <= lower + active_tolerance * np.maximum(1.0, np.abs(lower)))
    ).astype(np.intp)
    upper_active = np.flatnonzero(
        np.isfinite(upper)
        & (x >= upper - active_tolerance * np.maximum(1.0, np.abs(upper)))
    ).astype(np.intp)
    n = x.size
    free = (
        _dense_rows(jh).T
        if h.size
        else np.empty((n, 0), dtype=float)
    )
    columns: list[Array] = []
    if active.size:
        columns.append(_dense_rows(jc, active).T)
    if lower_active.size:
        block = np.zeros((n, lower_active.size), dtype=float)
        block[lower_active, np.arange(lower_active.size)] = -1.0
        columns.append(block)
    if upper_active.size:
        block = np.zeros((n, upper_active.size), dtype=float)
        block[upper_active, np.arange(upper_active.size)] = 1.0
        columns.append(block)
    nonnegative = (
        np.column_stack(columns)
        if columns
        else np.empty((n, 0), dtype=float)
    )
    result = solve_mixed_least_squares(free, nonnegative, -gradient)
    offset = 0
    lam = result.nonnegative[offset : offset + active.size].copy()
    offset += active.size
    lower_lam = result.nonnegative[
        offset : offset + lower_active.size
    ].copy()
    offset += lower_active.size
    upper_lam = result.nonnegative[
        offset : offset + upper_active.size
    ].copy()
    complementarity = 0.0
    if active.size:
        complementarity = max(
            complementarity,
            float(np.max(np.abs(lam * c[active]), initial=0.0)),
        )
    if lower_active.size:
        complementarity = max(
            complementarity,
            float(
                np.max(
                    np.abs(lower_lam * (x[lower_active] - lower[lower_active])),
                    initial=0.0,
                )
            ),
        )
    if upper_active.size:
        complementarity = max(
            complementarity,
            float(
                np.max(
                    np.abs(upper_lam * (upper[upper_active] - x[upper_active])),
                    initial=0.0,
                )
            ),
        )
    stationarity = float(
        result.residual_norm
        if result.residual.size == 0
        else np.linalg.norm(result.residual, ord=np.inf)
    )
    equality_term = free @ result.free if free.shape[1] else np.zeros(n)
    inequality_term = nonnegative @ result.nonnegative if nonnegative.shape[1] else np.zeros(n)
    stationarity_scale = max(
        1.0,
        float(np.linalg.norm(gradient, ord=np.inf)),
        float(np.linalg.norm(equality_term, ord=np.inf)),
        float(np.linalg.norm(inequality_term, ord=np.inf)),
    )
    complementarity_scale = max(1.0, stationarity_scale)
    return _MultiplierEstimate(
        active,
        lam,
        result.free.copy(),
        lower_active,
        lower_lam,
        upper_active,
        upper_lam,
        stationarity,
        stationarity / stationarity_scale,
        complementarity,
        complementarity / complementarity_scale,
        bool(result.success),
    )


def _linearized_criticality(
    x: Array,
    gradient: Array,
    c: Array,
    jc: object,
    h: Array,
    jh: object,
    lower: Array,
    upper: Array,
    *,
    radius: float,
    settings: SparseSQPSettings,
) -> tuple[float, bool]:
    """Scale-normalized first-order criticality in solver coordinates.

    The LP minimizes the objective derivative over the complete linearized
    feasible set inside a small infinity-norm ball.  Unlike multiplier least
    squares, this measure is stable under dependent/degenerate active rows.
    Because the finite NLP is already variable- and row-scaled before this
    routine is called, the result is directly comparable across path sizes.
    """
    from scipy.sparse import csr_matrix, vstack
    from .highs_backend import solve_linear_program

    n = x.size
    step_lower = np.maximum(-radius, lower - x)
    step_upper = np.minimum(radius, upper - x)
    blocks: list[object] = []
    row_lower: list[Array] = []
    row_upper: list[Array] = []
    if c.size:
        blocks.append(csr_matrix(jc))
        row_lower.append(np.full(c.size, -math.inf, dtype=float))
        row_upper.append(-c)
    if h.size:
        blocks.append(csr_matrix(jh))
        row_lower.append(-h)
        row_upper.append(-h)
    matrix = (
        vstack(blocks, format="csr")
        if blocks
        else csr_matrix((0, n), dtype=float)
    )
    lo = np.concatenate(row_lower) if row_lower else np.empty(0)
    hi = np.concatenate(row_upper) if row_upper else np.empty(0)
    attempts = [(True, settings.criticality_lp_optimality_tolerance)]
    if settings.criticality_retry_without_presolve:
        attempts.append((False, settings.criticality_lp_optimality_tolerance))
        if (
            settings.criticality_lp_retry_optimality_tolerance
            > settings.criticality_lp_optimality_tolerance
        ):
            attempts.append(
                (False, settings.criticality_lp_retry_optimality_tolerance)
            )

    result = None
    for presolve, optimality_tolerance in attempts:
        try:
            candidate = solve_linear_program(
                gradient, matrix, lo, hi, step_lower, step_upper,
                feasibility_tolerance=max(
                    1.0e-10, settings.feasibility_tolerance
                ),
                optimality_tolerance=optimality_tolerance,
                time_limit=settings.highs_time_limit,
                presolve=presolve,
                display=False,
                allow_private_fallback=settings.allow_private_highs_fallback,
                threads=settings.highs_threads,
            )
        except Exception:
            continue
        if (
            candidate.success
            and bool(getattr(candidate, "optimality_certified", False))
            and math.isfinite(candidate.objective)
        ):
            result = candidate
            break
    if result is None:
        return math.inf, False
    normalization = radius * max(1.0, float(np.linalg.norm(gradient, ord=1)))
    return max(0.0, -float(result.objective)) / normalization, True



def _robust_linearized_criticality(
    x: Array, gradient: Array, c: Array, jc: object, h: Array, jh: object,
    lower: Array, upper: Array, *, settings: SparseSQPSettings,
) -> tuple[float, bool]:
    """Criticality checked at two fixed solver-coordinate radii.

    The larger radius protects against false stationarity caused by finite
    feasibility tolerances at an extremely local radius; the smaller radius
    retains local first-order resolution.  Taking the maximum is conservative
    for stopping: both neighborhoods must show little feasible descent.
    """
    values: list[float] = []
    for radius in (settings.criticality_radius, settings.criticality_secondary_radius):
        value, success = _linearized_criticality(
            x, gradient, c, jc, h, jh, lower, upper, radius=radius, settings=settings
        )
        if not success:
            return math.inf, False
        values.append(value)
    return max(values, default=math.inf), True

def _lagrangian_gradient_change(
    old_gradient: Array,
    new_gradient: Array,
    old_jc: object,
    new_jc: object,
    old_jh: object,
    new_jh: object,
    estimate: _MultiplierEstimate,
) -> Array:
    change = new_gradient - old_gradient
    if estimate.active_inequalities.size:
        old_rows = _dense_rows(old_jc, estimate.active_inequalities)
        new_rows = _dense_rows(new_jc, estimate.active_inequalities)
        change = change + (new_rows - old_rows).T @ estimate.inequality_multipliers
    if estimate.equality_multipliers.size:
        old_eq = _dense_rows(old_jh)
        new_eq = _dense_rows(new_jh)
        change = change + (new_eq - old_eq).T @ estimate.equality_multipliers
    return np.asarray(change, dtype=float)


def _step_bounds(
    x: Array,
    lower: Array,
    upper: Array,
    trust_radius: float,
) -> tuple[Array, Array]:
    step_lower = np.maximum(-trust_radius, lower - x)
    step_upper = np.minimum(trust_radius, upper - x)
    return step_lower, step_upper


def _linprog(
    objective: Array,
    *,
    a_ub: object | None,
    b_ub: Array | None,
    a_eq: object | None,
    b_eq: Array | None,
    bounds: list[tuple[float | None, float | None]],
    settings: SparseSQPSettings,
    workspace: _SubproblemState,
    warm_key: str,
):
    from scipy.sparse import csr_matrix, vstack
    from .highs_backend import solve_linear_program

    n = objective.size
    blocks: list[object] = []
    row_lower: list[Array] = []
    row_upper: list[Array] = []
    if a_ub is not None:
        aub = csr_matrix(a_ub, dtype=float)
        bub = np.asarray(b_ub, dtype=float)
        blocks.append(aub)
        row_lower.append(np.full(aub.shape[0], -math.inf, dtype=float))
        row_upper.append(bub)
    if a_eq is not None:
        aeq = csr_matrix(a_eq, dtype=float)
        beq = np.asarray(b_eq, dtype=float)
        blocks.append(aeq)
        row_lower.append(beq)
        row_upper.append(beq)
    matrix = (
        vstack(blocks, format="csr")
        if blocks
        else csr_matrix((0, n), dtype=float)
    )
    row_lo = np.concatenate(row_lower) if row_lower else np.empty(0)
    row_hi = np.concatenate(row_upper) if row_upper else np.empty(0)
    col_lo = np.array(
        [-math.inf if lo is None else float(lo) for lo, _ in bounds],
        dtype=float,
    )
    col_hi = np.array(
        [math.inf if hi is None else float(hi) for _, hi in bounds],
        dtype=float,
    )
    warm = workspace.warm_starts.get(warm_key) if settings.warm_start_highs else None
    if warm is not None:
        workspace.warm_start_attempts += 1
    raw = solve_linear_program(
        objective,
        matrix,
        row_lo,
        row_hi,
        col_lo,
        col_hi,
        warm_start=warm,
        warm_start_basis=settings.warm_start_highs,
        feasibility_tolerance=max(
            1.0e-10, min(1.0e-7, settings.feasibility_tolerance)
        ),
        optimality_tolerance=1.0e-9,
        time_limit=settings.highs_time_limit,
        presolve=settings.highs_presolve and warm is None,
        display=settings.display,
        allow_private_fallback=settings.allow_private_highs_fallback,
        threads=settings.highs_threads,
    )
    workspace.backend_name = raw.backend.name
    workspace.backend_public = raw.backend.public_api
    if raw.warm_start is not None:
        workspace.warm_starts[warm_key] = raw.warm_start
    if raw.warm_start_used:
        workspace.warm_start_uses += 1
    return SimpleNamespace(
        x=raw.x,
        success=raw.success,
        status=raw.status,
        message=raw.message,
        fun=raw.objective,
        nit=raw.iterations,
        row_dual=raw.row_dual,
        col_dual=raw.column_dual,
        warm_start_used=raw.warm_start_used,
        highs_subproblems=1,
    )


def _normal_step(
    c: Array,
    jc: Array,
    h: Array,
    jh: Array,
    step_lower: Array,
    step_upper: Array,
    settings: SparseSQPSettings,
    workspace: _SubproblemState,
    *,
    warm_prefix: str = "normal",
):
    """Lexicographic maximum-violation and minimum-L1 normal step.

    A single elastic LP has a large degenerate face whenever zero linearized
    violation is attainable.  Selecting an arbitrary point on that face can
    send nonlinear restoration to a trust-region corner.  We therefore solve
    two small HiGHS problems: first minimize ``eta``; then hold ``eta`` at its
    certified optimum (plus a tiny numerical margin) and minimize ``||d||_1``.
    """
    from scipy.sparse import csr_matrix, hstack, identity, vstack

    n = step_lower.size
    first_blocks: list[object] = []
    first_rhs: list[Array] = []
    if c.size:
        first_blocks.append(
            hstack((csr_matrix(jc), -np.ones((c.size, 1))), format="csr")
        )
        first_rhs.append(-c)
    if h.size:
        ones = -np.ones((h.size, 1))
        first_blocks.append(
            hstack((csr_matrix(jh), ones), format="csr")
        )
        first_rhs.append(-h)
        first_blocks.append(
            hstack((csr_matrix(-jh), ones), format="csr")
        )
        first_rhs.append(h)
    first_a = vstack(first_blocks, format="csr") if first_blocks else None
    first_b = np.concatenate(first_rhs) if first_rhs else None
    first_objective = np.zeros(n + 1, dtype=float)
    first_objective[n] = 1.0
    first_bounds = [
        (float(lo), float(hi))
        for lo, hi in zip(step_lower, step_upper, strict=True)
    ]
    first_bounds.append((0.0, None))
    primary = _linprog(
        first_objective,
        a_ub=first_a,
        b_ub=first_b,
        a_eq=None,
        b_eq=None,
        bounds=first_bounds,
        settings=settings,
        workspace=workspace,
        warm_key=f"{warm_prefix}_primary",
    )
    if not bool(getattr(primary, "success", False)):
        primary.highs_subproblems = 1
        return primary

    eta_star = max(0.0, float(np.asarray(primary.x, dtype=float)[n]))
    eta_limit = eta_star + settings.normal_model_margin

    # Secondary variables [d, t] with t >= |d|.
    second_blocks: list[object] = []
    second_rhs: list[Array] = []
    zero_t_c = csr_matrix((c.size, n)) if c.size else None
    if c.size:
        second_blocks.append(
            hstack((csr_matrix(jc), zero_t_c), format="csr")
        )
        second_rhs.append(-c + eta_limit)
    if h.size:
        zero_t_h = csr_matrix((h.size, n))
        second_blocks.append(
            hstack((csr_matrix(jh), zero_t_h), format="csr")
        )
        second_rhs.append(-h + eta_limit)
        second_blocks.append(
            hstack((csr_matrix(-jh), zero_t_h), format="csr")
        )
        second_rhs.append(h + eta_limit)
    eye = identity(n, format="csr")
    second_blocks.append(hstack((eye, -eye), format="csr"))
    second_rhs.append(np.zeros(n, dtype=float))
    second_blocks.append(hstack((-eye, -eye), format="csr"))
    second_rhs.append(np.zeros(n, dtype=float))
    second_a = vstack(second_blocks, format="csr")
    second_b = np.concatenate(second_rhs)
    second_objective = np.concatenate(
        (np.zeros(n, dtype=float), np.ones(n, dtype=float) / max(1, n))
    )
    second_bounds = [
        (float(lo), float(hi))
        for lo, hi in zip(step_lower, step_upper, strict=True)
    ]
    second_bounds.extend((0.0, None) for _ in range(n))
    secondary = _linprog(
        second_objective,
        a_ub=second_a,
        b_ub=second_b,
        a_eq=None,
        b_eq=None,
        bounds=second_bounds,
        settings=settings,
        workspace=workspace,
        warm_key=f"{warm_prefix}_secondary",
    )
    if not bool(getattr(secondary, "success", False)):
        primary.highs_subproblems = 2
        return primary
    d = np.asarray(secondary.x, dtype=float)[:n]
    secondary.x = np.concatenate((d, np.array([eta_star], dtype=float)))
    secondary.highs_subproblems = 2
    secondary.primary_status = int(getattr(primary, "status", -1))
    return secondary


def _tangential_step(
    gradient: Array,
    hessian: object,
    c: Array,
    jc: Array,
    h: Array,
    jh: Array,
    step_lower: Array,
    step_upper: Array,
    allowed_inequality_violation: float,
    settings: SparseSQPSettings,
    workspace: _SubproblemState,
    *,
    warm_key: str = "tangential",
):
    """Solve the tangential LP or convex BFGS-QP model."""
    from scipy.sparse import csr_matrix, vstack

    if settings.use_quadratic_model:
        from .highs_backend import solve_convex_quadratic_program

        blocks: list[object] = []
        row_lower: list[Array] = []
        row_upper: list[Array] = []
        if c.size:
            blocks.append(csr_matrix(jc))
            row_lower.append(np.full(c.size, -math.inf, dtype=float))
            row_upper.append(
                np.full(c.size, allowed_inequality_violation, dtype=float) - c
            )
        if h.size:
            blocks.append(csr_matrix(jh))
            row_lower.append(-h)
            row_upper.append(-h)
        matrix = (
            vstack(blocks, format="csr")
            if blocks
            else csr_matrix((0, gradient.size), dtype=float)
        )
        lower_rows = np.concatenate(row_lower) if row_lower else np.empty(0)
        upper_rows = np.concatenate(row_upper) if row_upper else np.empty(0)
        qp_warm_key = f"{warm_key}_qp"
        warm = (
            workspace.warm_starts.get(qp_warm_key)
            if settings.warm_start_highs and settings.warm_start_qp
            else None
        )
        if warm is not None:
            workspace.warm_start_attempts += 1
        def solve_qp_once(warm_start, *, presolve: bool):
            try:
                return solve_convex_quadratic_program(
                    gradient,
                    _hessian_sparse_matrix(hessian),
                    matrix,
                    lower_rows,
                    upper_rows,
                    step_lower,
                    step_upper,
                    warm_start=warm_start,
                    warm_start_solution=(
                        settings.warm_start_highs and settings.warm_start_qp
                        and warm_start is not None
                    ),
                    feasibility_tolerance=max(
                        1.0e-10, min(1.0e-7, settings.feasibility_tolerance)
                    ),
                    optimality_tolerance=1.0e-8,
                    time_limit=settings.highs_time_limit,
                    presolve=presolve,
                    display=settings.display,
                    allow_private_fallback=settings.allow_private_highs_fallback,
                    qp_iteration_limit=settings.highs_qp_iteration_limit,
                    threads=settings.highs_threads,
                )
            except (ImportError, AttributeError):
                return None

        def qp_is_acceptable(candidate) -> tuple[bool, str]:
            if candidate is None or not bool(candidate.success):
                return False, "solver status/primal check"
            step = np.asarray(candidate.x, dtype=float)
            model_value = float(
                gradient @ step + 0.5 * _hessian_quadratic(hessian, step)
            )
            zero_feasible = (
                (not c.size or float(np.max(c, initial=-math.inf))
                 <= allowed_inequality_violation)
                and (not h.size or float(np.max(np.abs(h), initial=0.0))
                     <= settings.equality_tolerance)
            )
            model_guard = max(
                settings.minimum_predicted_reduction,
                100.0 * np.finfo(float).eps
                * max(1.0, abs(float(candidate.objective))),
            )
            if not math.isfinite(model_value):
                return False, "nonfinite model value"
            if zero_feasible and model_value > model_guard:
                return False, "positive model value despite feasible zero step"
            if settings.require_certified_qp_optimality:
                certified = bool(getattr(candidate, "optimality_certified", True))
                stat = float(
                    getattr(candidate, "checked_stationarity_scaled", 0.0)
                )
                comp = float(
                    getattr(candidate, "checked_complementarity_scaled", 0.0)
                )
                if not certified:
                    return False, "independent QP KKT certificate failed"
                if stat > settings.qp_stationarity_tolerance:
                    return False, f"QP stationarity {stat:.3e}"
                if comp > settings.qp_complementarity_tolerance:
                    return False, f"QP complementarity {comp:.3e}"
            return True, "accepted"

        qp = solve_qp_once(
            warm, presolve=settings.highs_presolve and warm is None
        )
        if qp is not None:
            workspace.backend_name = qp.backend.name
            workspace.backend_public = qp.backend.public_api
            if settings.warm_start_qp and qp.warm_start is not None:
                workspace.warm_starts[qp_warm_key] = qp.warm_start
            if qp.warm_start_used:
                workspace.warm_start_uses += 1
        qp_acceptable, rejection_reason = qp_is_acceptable(qp)
        if not qp_acceptable and qp is not None:
            workspace.qp_rejections += 1

        # A warm-started active-set QP can occasionally inherit an unusable
        # working set after the linearized constraints change.  Retry the same
        # convex model cold before discarding curvature information entirely.
        if (
            not qp_acceptable
            and settings.retry_uncertified_qp_cold
            and warm is not None
        ):
            workspace.qp_cold_retries += 1
            cold = solve_qp_once(None, presolve=settings.highs_presolve)
            cold_acceptable, cold_reason = qp_is_acceptable(cold)
            if cold is not None:
                workspace.backend_name = cold.backend.name
                workspace.backend_public = cold.backend.public_api
                if settings.warm_start_qp and cold.warm_start is not None:
                    workspace.warm_starts[qp_warm_key] = cold.warm_start
            if cold_acceptable:
                qp = cold
                qp_acceptable = True
                rejection_reason = "warm QP rejected; cold retry accepted"
            else:
                rejection_reason += f"; cold retry: {cold_reason}"

        if qp is not None and (
            qp_acceptable or not settings.fallback_to_linear_model_on_qp_failure
        ):
            return SimpleNamespace(
                x=qp.x,
                success=qp_acceptable,
                status=qp.status,
                message=(
                    qp.message
                    if qp_acceptable
                    else qp.message + f"; rejected QP: {rejection_reason}"
                ),
                fun=qp.objective,
                qp_iterations=qp.iterations,
                row_dual=qp.row_dual,
                col_dual=qp.column_dual,
                is_qp=True,
                qp_attempted=True,
                warm_start_used=qp.warm_start_used,
            )
        workspace.qp_lp_fallbacks += 1
        # A linear objective model is a robust fallback when the experimental
        # HiGHS QP active-set solver times out or returns an inaccurate step.
        # It preserves all sparse constraints and trust-region bounds.

    a_ub = csr_matrix(jc) if c.size else None
    b_ub = (
        np.full(c.size, allowed_inequality_violation, dtype=float) - c
        if c.size
        else None
    )
    a_eq = csr_matrix(jh) if h.size else None
    b_eq = -h if h.size else None
    lp_bounds = [
        (float(lo), float(hi)) for lo, hi in zip(step_lower, step_upper, strict=True)
    ]
    result = _linprog(
        gradient,
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=a_eq,
        b_eq=b_eq,
        bounds=lp_bounds,
        settings=settings,
        workspace=workspace,
        warm_key=f"{warm_key}_lp",
    )
    result.is_qp = False
    result.qp_attempted = bool(settings.use_quadratic_model)
    return result


def _initial_hessian(
    gradient: Array,
    trust_radius: float,
    settings: SparseSQPSettings,
):
    from .quasi_newton import BandedHessian

    scale = settings.initial_hessian_scale
    if settings.automatic_hessian_scale:
        scale = max(
            scale,
            float(np.linalg.norm(gradient, ord=np.inf))
            / max(trust_radius, settings.minimum_trust_radius),
        )
    scale = min(
        settings.maximum_hessian_eigenvalue,
        max(settings.minimum_hessian_eigenvalue, scale),
    )
    model = settings.hessian_model
    if model == "auto":
        model = (
            "dense"
            if gradient.size <= settings.dense_hessian_threshold
            else "diagonal"
        )
    if model == "dense":
        return np.eye(gradient.size, dtype=float) * scale
    bandwidth = 0 if model == "diagonal" else settings.hessian_bandwidth
    return BandedHessian.scaled_identity(gradient.size, scale, bandwidth)


def _hessian_matvec(hessian: object, vector: Array) -> Array:
    if hasattr(hessian, "matvec"):
        return np.asarray(hessian.matvec(vector), dtype=float)
    return np.asarray(hessian @ vector, dtype=float)


def _hessian_quadratic(hessian: object, vector: Array) -> float:
    if hasattr(hessian, "quadratic"):
        return float(hessian.quadratic(vector))
    return float(vector @ (hessian @ vector))


def _hessian_sparse_matrix(hessian: object):
    from scipy.sparse import csc_matrix

    if hasattr(hessian, "to_csc"):
        return hessian.to_csc()
    return csc_matrix(np.asarray(hessian, dtype=float))


def _hessian_nonzeros(hessian: object) -> int:
    if hasattr(hessian, "nnz"):
        return int(hessian.nnz)
    return int(np.count_nonzero(np.asarray(hessian)))


def _regularize_dense_hessian(hessian: Array, settings: SparseSQPSettings) -> Array:
    """Keep a dense BFGS model positive definite without an eigensolve.

    Powell-damped BFGS preserves positive definiteness in exact arithmetic.
    We therefore use Cholesky as the fast certification path and add a
    geometrically increasing diagonal shift only when roundoff or clipping
    has damaged definiteness.  A cheap infinity-norm rescaling prevents
    excessive QP conditioning without changing eigenvectors.
    """
    symmetric = 0.5 * (hessian + hessian.T)
    norm_inf = float(np.linalg.norm(symmetric, ord=np.inf))
    if not math.isfinite(norm_inf):
        raise FloatingPointError("nonfinite dense Hessian")
    if norm_inf > settings.maximum_hessian_eigenvalue:
        symmetric *= settings.maximum_hessian_eigenvalue / norm_inf

    shift = 0.0
    eye = np.eye(symmetric.shape[0], dtype=float)
    for _ in range(8):
        candidate = symmetric if shift == 0.0 else symmetric + shift * eye
        try:
            np.linalg.cholesky(candidate)
            return candidate
        except np.linalg.LinAlgError:
            shift = (
                settings.minimum_hessian_eigenvalue
                if shift == 0.0
                else 10.0 * shift
            )
    # This is a defensive fallback, not the normal update path.
    diagonal_floor = max(
        settings.minimum_hessian_eigenvalue,
        -float(np.min(np.diag(symmetric))) + settings.minimum_hessian_eigenvalue,
    )
    candidate = symmetric + diagonal_floor * eye
    np.linalg.cholesky(candidate)
    return candidate


def _damped_bfgs_update(
    hessian: object,
    step: Array,
    gradient_change: Array,
    settings: SparseSQPSettings,
):
    if hasattr(hessian, "damped_bfgs_update"):
        hessian.damped_bfgs_update(
            step,
            gradient_change,
            curvature_tolerance=settings.bfgs_curvature_tolerance,
            minimum_eigenvalue=settings.minimum_hessian_eigenvalue,
            maximum_eigenvalue=settings.maximum_hessian_eigenvalue,
        )
        return hessian
    step_norm = float(np.linalg.norm(step))
    if step_norm <= settings.minimum_trust_radius:
        return hessian
    bs = hessian @ step
    sbs = float(step @ bs)
    sy = float(step @ gradient_change)
    tolerance = settings.bfgs_curvature_tolerance * max(1.0, step_norm**2)
    if not math.isfinite(sbs) or sbs <= tolerance:
        return _initial_hessian(gradient_change, step_norm, settings)
    y = gradient_change.copy()
    if sy < 0.2 * sbs:
        denominator = sbs - sy
        if denominator <= tolerance:
            return hessian
        theta = 0.8 * sbs / denominator
        y = theta * y + (1.0 - theta) * bs
        sy = float(step @ y)
    if not math.isfinite(sy) or sy <= tolerance:
        return hessian
    updated = hessian - np.outer(bs, bs) / sbs + np.outer(y, y) / sy
    return _regularize_dense_hessian(updated, settings)


def _predicted_reduction(
    gradient: Array,
    hessian: object,
    step: Array,
    *,
    quadratic_model: bool = True,
) -> float:
    value = float(gradient @ step)
    if quadratic_model:
        value += 0.5 * _hessian_quadratic(hessian, step)
    return -value


def _status_code(raw: object | None) -> int:
    return -1 if raw is None else int(getattr(raw, "status", -1))


def _solve_sparse_filter_sqp_once(
    x0: Sequence[float],
    objective: DifferentiableObjective,
    constraints: DifferentiableConstraints,
    *,
    lower_bounds: Sequence[float] | None = None,
    upper_bounds: Sequence[float] | None = None,
    settings: SparseSQPSettings = SparseSQPSettings(),
    continuation_state: SparseSQPContinuationState | None = None,
    constraint_generation: int | None = None,
    objective_identity: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SparseSQPResult:
    """Solve one finite nonlinear problem with sparse LP subproblems.

    The returned ``x`` is the best feasible incumbent whenever one was found.
    ``current_x`` records the final filter iterate, which can be mildly
    infeasible.  This separation is intentional and is central to safe use in
    an exchange algorithm.
    """

    started = time.perf_counter()
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    x = x.copy()
    n = x.size
    lower = (
        np.full(n, -math.inf, dtype=float)
        if lower_bounds is None
        else np.broadcast_to(np.asarray(lower_bounds, dtype=float), (n,)).copy()
    )
    upper = (
        np.full(n, math.inf, dtype=float)
        if upper_bounds is None
        else np.broadcast_to(np.asarray(upper_bounds, dtype=float), (n,)).copy()
    )
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError("invalid variable bounds")
    active_generation = (
        int(constraint_generation)
        if constraint_generation is not None
        else (continuation_state.constraint_generation if continuation_state else 0)
    )
    generation_changed = bool(
        continuation_state is not None
        and active_generation != continuation_state.constraint_generation
    )
    if continuation_state is not None:
        if continuation_state.current_x.shape != x.shape:
            raise ValueError("continuation state dimension does not match x0")
        if (
            continuation_state.objective_identity is not None
            and objective_identity is not None
            and continuation_state.objective_identity != objective_identity
        ):
            raise ValueError("continuation state belongs to a different objective")
        x = continuation_state.current_x.copy()

    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("initial or resumed iterate violates variable bounds")

    counters = _EvaluationCounters(
        jacobian_mode=settings.jacobian_mode,
        soft_deadline_enabled=settings.objective_soft_deadline_enabled,
        soft_deadline_minimum=settings.objective_soft_deadline_minimum,
        soft_deadline_p95_multiplier=settings.objective_soft_deadline_p95_multiplier,
        latency_history_size=settings.objective_latency_history_size,
    )
    if continuation_state is None:
        value, gradient = counters.objective(objective, x)
    else:
        value = float(continuation_state.current_objective)
        gradient = continuation_state.current_gradient.copy()
        counters.successful_objective_latencies.extend(
            continuation_state.successful_objective_latencies
        )
    c, jc, h, jh = counters.constraints(constraints, x)
    violation = _violation(c, h)
    filter_set = _Filter(
        settings.filter_gamma_violation, settings.filter_gamma_objective
    )
    if continuation_state is not None and not generation_changed:
        filter_set.entries = list(continuation_state.filter_entries)
    if not filter_set.entries or generation_changed:
        filter_set.add(violation, value)
    filter_violation_cap = settings.maximum_filter_violation
    minimum_filter_cap = max(
        settings.minimum_filter_tolerance_factor
        * max(settings.feasibility_tolerance, settings.equality_tolerance),
        settings.normal_model_margin,
    )

    best_feasible_x: Array | None = None
    best_feasible_value = math.inf
    best_feasible_gradient: Array | None = None
    best_feasible_violation = math.inf
    if _is_feasible(c, h, settings):
        best_feasible_x = x.copy()
        best_feasible_value = value
        best_feasible_gradient = gradient.copy()
        best_feasible_violation = violation
    if (
        continuation_state is not None
        and continuation_state.best_feasible_x is not None
        and not np.array_equal(continuation_state.best_feasible_x, x)
    ):
        prior_c, _, prior_h, _ = counters.constraints(
            constraints, continuation_state.best_feasible_x
        )
        prior_violation = _violation(prior_c, prior_h)
        if (
            _is_feasible(prior_c, prior_h, settings)
            and continuation_state.best_feasible_objective < best_feasible_value
        ):
            best_feasible_x = continuation_state.best_feasible_x.copy()
            best_feasible_value = float(continuation_state.best_feasible_objective)
            best_feasible_gradient = continuation_state.best_feasible_gradient.copy()
            best_feasible_violation = prior_violation


    multiplier_diagnostic_calls = 0
    multiplier_diagnostic_seconds = 0.0
    criticality_diagnostic_calls = 0
    criticality_diagnostic_seconds = 0.0
    diagnostic_failure: str | None = None

    def timed_multiplier_estimate(
        xx: Array, gg: Array, cc: Array, jcc: object, hh: Array, jhh: object
    ) -> _MultiplierEstimate | None:
        nonlocal multiplier_diagnostic_calls, multiplier_diagnostic_seconds
        nonlocal diagnostic_failure
        diagnostic_started = time.perf_counter()
        multiplier_diagnostic_calls += 1
        try:
            return _estimate_multipliers(
                xx, gg, cc, jcc, hh, jhh, lower, upper,
                active_tolerance=settings.multiplier_active_tolerance,
            )
        except Exception as exc:
            diagnostic_failure = f"multiplier estimation failed: {exc}"
            return None
        finally:
            elapsed = time.perf_counter() - diagnostic_started
            multiplier_diagnostic_seconds += elapsed
            counters.record_stage("multiplier_nnls", elapsed)

    def timed_criticality(
        xx: Array, gg: Array, cc: Array, jcc: object, hh: Array, jhh: object
    ) -> tuple[float, bool]:
        nonlocal criticality_diagnostic_calls, criticality_diagnostic_seconds
        nonlocal diagnostic_failure
        diagnostic_started = time.perf_counter()
        criticality_diagnostic_calls += 1
        try:
            value_out, success_out = _robust_linearized_criticality(
                xx, gg, cc, jcc, hh, jhh, lower, upper, settings=settings,
            )
            if not success_out:
                diagnostic_failure = "robust criticality LP failed"
            return value_out, success_out
        finally:
            elapsed = time.perf_counter() - diagnostic_started
            criticality_diagnostic_seconds += elapsed
            counters.record_stage("robust_criticality_lp", elapsed)

    best_kkt_stationarity = math.inf
    best_kkt_stationarity_scaled = math.inf
    best_kkt_complementarity = math.inf
    best_kkt_complementarity_scaled = math.inf
    estimate0: _MultiplierEstimate | None = None
    if (
        best_feasible_x is not None
        and (
            settings.diagnostics.multipliers_on_initial_point
            or settings.check_initial_kkt
        )
    ):
        estimate0 = timed_multiplier_estimate(x, gradient, c, jc, h, jh)
    if estimate0 is not None:
        best_kkt_stationarity = estimate0.stationarity_inf
        best_kkt_stationarity_scaled = estimate0.stationarity_scaled
        best_kkt_complementarity = estimate0.complementarity_inf
        best_kkt_complementarity_scaled = estimate0.complementarity_scaled

    least_violation_x = x.copy()
    least_violation_value = value
    least_violation_gradient = gradient.copy()
    least_violation = violation

    if continuation_state is None:
        trust_radius = settings.initial_trust_radius
        hessian = _initial_hessian(gradient, trust_radius, settings)
        workspace = _SubproblemState()
    else:
        trust_radius = min(
            settings.maximum_trust_radius,
            max(settings.minimum_trust_radius, continuation_state.trust_radius),
        )
        hessian = continuation_state.hessian.copy()
        workspace = _SubproblemState(
            warm_starts=(
                {} if generation_changed else dict(continuation_state.warm_starts)
            )
        )
    # Diagnostic-only solves may never build an ordinary SQP subproblem.  Resolve
    # the selected backend here so final criticality telemetry still identifies
    # the implementation that the criticality LPs execute.
    from .highs_backend import backend_info

    selected_backend = backend_info(
        allow_private_fallback=settings.allow_private_highs_fallback
    )
    workspace.backend_name = selected_backend.name
    workspace.backend_public = selected_backend.public_api
    history: list[SparseSQPIteration] = []
    linear_programs = 0
    quadratic_programs = 0
    normal_linear_programs = 0
    tangential_linear_programs = 0
    second_order_linear_programs = 0
    linear_program_seconds = 0.0
    accepted_steps = 0
    rejected_steps = 0
    restoration_steps = 0
    screened_trials = 0
    message = "maximum iterations reached"
    status = 1
    initial_converged = False
    accepted_step_budget_reached = False
    if settings.check_initial_kkt and best_feasible_x is not None:
        initial_multiplier_ok = bool(
            estimate0 is not None
            and estimate0.success
            and estimate0.stationarity_scaled <= settings.kkt_tolerance
            and estimate0.complementarity_scaled <= settings.complementarity_tolerance
        )
        initial_criticality, initial_criticality_success = timed_criticality(
            x, gradient, c, jc, h, jh
        )
        initial_criticality_ok = bool(
            initial_criticality_success
            and initial_criticality <= settings.criticality_tolerance
        )
        initial_converged = (
            initial_multiplier_ok and initial_criticality_ok
            if settings.diagnostics.strict_convergence_requires_all
            else initial_multiplier_ok or initial_criticality_ok
        )
        if initial_converged:
            message = "initial point satisfies finite-NLP KKT conditions"
            status = 0

    iteration_budget = (
        0
        if initial_converged or settings.diagnostic_only
        else settings.max_iterations
    )
    for iteration in range(1, iteration_budget + 1):
        if trust_radius < settings.minimum_trust_radius:
            message = "trust region fell below minimum radius"
            status = 2
            break

        step_lower, step_upper = _step_bounds(
            x, lower, upper, trust_radius
        )
        if np.any(step_lower > step_upper):
            message = "empty trust-region/box intersection"
            status = 3
            break

        normal_raw = None
        tangential_raw = None
        normal_status = -1
        tangential_status = -1
        step_kind = "tangential"
        correction_used = False

        # A normal LP is useful whenever the current point is materially
        # infeasible.  At a feasible point the tangential LP is substantially
        # cheaper and the relaxed cap provides globalization room.
        normal_step: Array | None = None
        predicted_normal_violation = 0.0
        if violation > settings.feasibility_tolerance:
            lp_started = time.perf_counter()
            normal_raw = _normal_step(
                c, jc, h, jh, step_lower, step_upper, settings, workspace
            )
            linear_program_seconds += time.perf_counter() - lp_started
            normal_subproblems = int(getattr(normal_raw, "highs_subproblems", 1))
            linear_programs += normal_subproblems
            normal_linear_programs += normal_subproblems
            normal_status = _status_code(normal_raw)
            if bool(getattr(normal_raw, "success", False)):
                normal_vector = np.asarray(normal_raw.x, dtype=float)
                normal_step = normal_vector[:n]
                predicted_normal_violation = max(
                    0.0, float(normal_vector[n])
                )

        allowed_violation = filter_violation_cap
        if normal_step is not None and violation > filter_violation_cap:
            allowed_violation = min(
                violation,
                max(
                    settings.feasibility_tolerance,
                    predicted_normal_violation + settings.normal_model_margin,
                ),
            )

        lp_started = time.perf_counter()
        tangential_raw = _tangential_step(
            gradient,
            hessian,
            c,
            jc,
            h,
            jh,
            step_lower,
            step_upper,
            allowed_violation,
            settings,
            workspace,
            warm_key="tangential",
        )
        linear_program_seconds += time.perf_counter() - lp_started
        if bool(getattr(tangential_raw, "qp_attempted", False)):
            quadratic_programs += 1
        if not bool(getattr(tangential_raw, "is_qp", False)):
            linear_programs += 1
            tangential_linear_programs += 1
        tangential_status = _status_code(tangential_raw)
        step_uses_quadratic_model = bool(
            getattr(tangential_raw, "is_qp", False)
        )

        if bool(getattr(tangential_raw, "success", False)):
            step = np.asarray(tangential_raw.x, dtype=float)
        elif normal_step is not None:
            step = normal_step
            step_kind = "restoration"
            step_uses_quadratic_model = False
        else:
            rejected_steps += 1
            history.append(
                SparseSQPIteration(
                    iteration,
                    value,
                    violation,
                    trust_radius,
                    False,
                    "lp_failure",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    math.nan,
                    violation,
                    normal_status,
                    tangential_status,
                    False,
                    False,
                )
            )
            trust_radius *= settings.trust_contraction
            continue

        step_norm = float(np.linalg.norm(step, ord=np.inf))
        predicted_reduction = _predicted_reduction(
            gradient,
            hessian,
            step,
            quadratic_model=step_uses_quadratic_model,
        )
        if step_kind == "tangential" and (
            not math.isfinite(predicted_reduction)
            or predicted_reduction <= settings.minimum_predicted_reduction
        ):
            if normal_step is not None and float(
                np.linalg.norm(normal_step, ord=np.inf)
            ) > settings.minimum_trust_radius:
                step = normal_step
                step_kind = "restoration"
                step_uses_quadratic_model = False
                step_norm = float(np.linalg.norm(step, ord=np.inf))
                predicted_reduction = 0.0
            else:
                # A relaxed filter band solves a nearby NLP. Before declaring
                # stationarity, tighten that band and restart from the retained
                # feasible incumbent. This prevents the final projection from
                # setting the attainable accuracy (the original candidate's
                # worst convex objective error came from exactly this effect).
                if (
                    best_feasible_x is not None
                    and filter_violation_cap
                    > minimum_filter_cap * (1.0 + 8.0 * np.finfo(float).eps)
                ):
                    filter_violation_cap = max(
                        minimum_filter_cap,
                        filter_violation_cap * settings.filter_tightening_factor,
                    )
                    x = best_feasible_x.copy()
                    value = float(best_feasible_value)
                    assert best_feasible_gradient is not None
                    gradient = best_feasible_gradient.copy()
                    c, jc, h, jh = counters.constraints(constraints, x)
                    violation = _violation(c, h)
                    filter_set = _Filter(
                        settings.filter_gamma_violation,
                        settings.filter_gamma_objective,
                    )
                    filter_set.add(violation, value)
                    trust_radius = max(
                        settings.minimum_trust_radius,
                        min(trust_radius, settings.initial_trust_radius),
                    )
                    history.append(
                        SparseSQPIteration(
                            iteration, value, violation, trust_radius, False,
                            "filter_tighten", 0.0, 0.0, predicted_reduction,
                            0.0, math.nan, violation, normal_status,
                            tangential_status, False, False,
                        )
                    )
                    continue
                message = "linearized objective is stationary in the trust region"
                status = 0 if best_feasible_x is not None else 4
                history.append(
                    SparseSQPIteration(
                        iteration,
                        value,
                        violation,
                        trust_radius,
                        False,
                        "stationary",
                        step_norm,
                        0.0,
                        predicted_reduction,
                        0.0,
                        math.nan,
                        violation,
                        normal_status,
                        tangential_status,
                        False,
                        False,
                    )
                )
                break

        accepted = False
        accepted_alpha = 0.0
        actual_reduction = 0.0
        reduction_ratio = math.nan
        trial_violation = violation
        objective_evaluated = False
        trial_bundle: tuple[
            Array, float, Array, Array, Array, Array, Array, float
        ] | None = None
        soc_attempted = False

        alpha = 1.0
        for _ in range(settings.maximum_line_search_steps):
            trial_x = np.clip(x + alpha * step, lower, upper)
            trial_c, trial_jc, trial_h, trial_jh = counters.constraints(
                constraints, trial_x
            )
            trial_violation = _violation(trial_c, trial_h)

            feasibility_progress = trial_violation <= (
                1.0 - settings.restoration_reduction
            ) * violation
            screen_limit = max(
                filter_violation_cap,
                (1.0 - settings.restoration_reduction) * violation,
            )
            if trial_violation > screen_limit and not feasibility_progress:
                screened_trials += 1
                # A Maratos-style second-order correction is most useful
                # before backtracking: the QP step may be excellent in the
                # tangent model while nonlinear constraint curvature rejects
                # its full length.  Correcting the full step first avoids
                # needlessly shrinking both the accepted step and trust
                # region.  Only cheap constraints and HiGHS LPs are touched
                # until the corrected point passes the geometric screen.
                if (
                    alpha == 1.0
                    and settings.enable_second_order_correction
                    and settings.second_order_correction_before_backtracking
                    and step_kind == "tangential"
                    and step_norm > settings.minimum_trust_radius
                ):
                    soc_attempted = True
                    correction_radius = (
                        settings.second_order_radius_fraction * trust_radius
                    )
                    correction_lower, correction_upper = _step_bounds(
                        trial_x, lower, upper, correction_radius
                    )
                    lp_started = time.perf_counter()
                    correction_raw = _normal_step(
                        trial_c, trial_jc, trial_h, trial_jh,
                        correction_lower, correction_upper, settings, workspace,
                        warm_prefix="soc"
                    )
                    linear_program_seconds += time.perf_counter() - lp_started
                    correction_subproblems = int(
                        getattr(correction_raw, "highs_subproblems", 1)
                    )
                    linear_programs += correction_subproblems
                    second_order_linear_programs += correction_subproblems
                    if bool(getattr(correction_raw, "success", False)):
                        correction = np.asarray(
                            correction_raw.x, dtype=float
                        )[:n]
                        corrected_x = np.clip(
                            trial_x + correction, lower, upper
                        )
                        corr_c, corr_jc, corr_h, corr_jh = (
                            counters.constraints(constraints, corrected_x)
                        )
                        corr_violation = _violation(corr_c, corr_h)
                        corr_progress = corr_violation <= (
                            1.0 - settings.restoration_reduction
                        ) * violation
                        if corr_violation <= screen_limit or corr_progress:
                            try:
                                corr_value, corr_gradient = counters.objective(
                                    objective, corrected_x
                                )
                            except Exception:
                                pass
                            else:
                                objective_evaluated = True
                                actual_reduction = value - corr_value
                                corrected_step = corrected_x - x
                                scaled_prediction = max(
                                    _predicted_reduction(
                                        gradient,
                                        hessian,
                                        corrected_step,
                                        quadratic_model=step_uses_quadratic_model,
                                    ),
                                    settings.minimum_predicted_reduction,
                                )
                                reduction_ratio = (
                                    actual_reduction / scaled_prediction
                                )
                                objective_progress = (
                                    corr_violation
                                    <= filter_violation_cap
                                    and actual_reduction
                                    >= settings.acceptance_ratio
                                    * scaled_prediction
                                )
                                if filter_set.acceptable(
                                    corr_violation, corr_value
                                ) and (corr_progress or objective_progress):
                                    trial_bundle = (
                                        corrected_x, corr_value, corr_gradient,
                                        corr_c, corr_jc, corr_h, corr_jh,
                                        corr_violation,
                                    )
                                    accepted = True
                                    accepted_alpha = 1.0
                                    trial_violation = corr_violation
                                    correction_used = True
                                    break
                alpha *= settings.line_search_contraction
                continue

            try:
                trial_value, trial_gradient = counters.objective(
                    objective, trial_x
                )
            except Exception:
                alpha *= settings.line_search_contraction
                continue
            objective_evaluated = True
            actual_reduction = value - trial_value
            scaled_step = alpha * step
            scaled_prediction = max(
                _predicted_reduction(
                    gradient,
                    hessian,
                    scaled_step,
                    quadratic_model=step_uses_quadratic_model,
                ),
                settings.minimum_predicted_reduction,
            )
            reduction_ratio = actual_reduction / scaled_prediction
            filter_ok = filter_set.acceptable(trial_violation, trial_value)
            objective_progress = (
                trial_violation <= filter_violation_cap
                and actual_reduction
                >= settings.acceptance_ratio * scaled_prediction
            )
            if filter_ok and (feasibility_progress or objective_progress):
                trial_bundle = (
                    trial_x,
                    trial_value,
                    trial_gradient,
                    trial_c,
                    trial_jc,
                    trial_h,
                    trial_jh,
                    trial_violation,
                )
                accepted = True
                accepted_alpha = alpha
                break
            alpha *= settings.line_search_contraction

        # One second-order correction is attempted only after the nominal step
        # failed.  It uses a fresh nonlinear linearization and remains inside a
        # smaller trust region around the rejected full-step point.
        if (
            not accepted
            and settings.enable_second_order_correction
            and not soc_attempted
            and step_kind == "tangential"
            and step_norm > settings.minimum_trust_radius
        ):
            base_x = np.clip(x + step, lower, upper)
            base_c, base_jc, base_h, base_jh = counters.constraints(
                constraints, base_x
            )
            correction_radius = settings.second_order_radius_fraction * trust_radius
            correction_lower, correction_upper = _step_bounds(
                base_x, lower, upper, correction_radius
            )
            lp_started = time.perf_counter()
            correction_raw = _normal_step(
                base_c,
                base_jc,
                base_h,
                base_jh,
                correction_lower,
                correction_upper,
                settings,
                workspace,
                warm_prefix="soc_post",
            )
            linear_program_seconds += time.perf_counter() - lp_started
            correction_subproblems = int(getattr(correction_raw, "highs_subproblems", 1))
            linear_programs += correction_subproblems
            second_order_linear_programs += correction_subproblems
            if bool(getattr(correction_raw, "success", False)):
                correction = np.asarray(correction_raw.x, dtype=float)[:n]
                corrected_x = np.clip(base_x + correction, lower, upper)
                corr_c, corr_jc, corr_h, corr_jh = counters.constraints(
                    constraints, corrected_x
                )
                corr_violation = _violation(corr_c, corr_h)
                screen_limit = max(
                    filter_violation_cap,
                    (1.0 - settings.restoration_reduction) * violation,
                )
                if corr_violation <= screen_limit:
                    try:
                        corr_value, corr_gradient = counters.objective(
                            objective, corrected_x
                        )
                    except Exception:
                        pass
                    else:
                        objective_evaluated = True
                        actual_reduction = value - corr_value
                        corrected_step = corrected_x - x
                        scaled_prediction = max(
                            _predicted_reduction(
                                gradient,
                                hessian,
                                corrected_step,
                                quadratic_model=step_uses_quadratic_model,
                            ),
                            settings.minimum_predicted_reduction,
                        )
                        reduction_ratio = actual_reduction / scaled_prediction
                        feasibility_progress = corr_violation <= (
                            1.0 - settings.restoration_reduction
                        ) * violation
                        objective_progress = (
                            corr_violation <= filter_violation_cap
                            and actual_reduction
                            >= settings.acceptance_ratio * scaled_prediction
                        )
                        if filter_set.acceptable(corr_violation, corr_value) and (
                            feasibility_progress or objective_progress
                        ):
                            trial_bundle = (
                                corrected_x,
                                corr_value,
                                corr_gradient,
                                corr_c,
                                corr_jc,
                                corr_h,
                                corr_jh,
                                corr_violation,
                            )
                            accepted = True
                            accepted_alpha = 1.0
                            trial_violation = corr_violation
                            correction_used = True

        old_value = value
        old_x = x.copy()
        old_gradient = gradient.copy()
        old_jc = jc.copy()
        old_jh = jh.copy()
        terminate_after_iteration = False
        if accepted and trial_bundle is not None:
            (
                x,
                value,
                gradient,
                c,
                jc,
                h,
                jh,
                violation,
            ) = trial_bundle
            # Trust-region decisions and diagnostics must use the step that
            # was actually accepted, including any second-order correction or
            # line-search contraction, rather than the original QP proposal.
            step_norm = float(np.linalg.norm(x - old_x, ord=np.inf))
            filter_set.add(violation, value)
            multiplier_estimate: _MultiplierEstimate | None = None
            if settings.use_lagrangian_bfgs or (
                settings.diagnostics.multipliers_on_accepted_steps
                and _is_feasible(c, h, settings)
            ):
                multiplier_estimate = timed_multiplier_estimate(
                    x, gradient, c, jc, h, jh
                )
            if settings.use_quadratic_model:
                gradient_change = gradient - old_gradient
                if (
                    settings.use_lagrangian_bfgs
                    and multiplier_estimate is not None
                    and multiplier_estimate.success
                ):
                    gradient_change = _lagrangian_gradient_change(
                        old_gradient, gradient, old_jc, jc, old_jh, jh,
                        multiplier_estimate,
                    )
                hessian = _damped_bfgs_update(
                    hessian, x - old_x, gradient_change, settings
                )
            if (
                settings.diagnostics.criticality_on_accepted_steps
                and _is_feasible(c, h, settings)
            ):
                timed_criticality(x, gradient, c, jc, h, jh)
            accepted_steps += 1
            if step_kind == "restoration":
                restoration_steps += 1

            # Progress telemetry is observational only and is tied to accepted
            # optimizer progress, never elapsed wall time.  This lets the
            # supervised planner retain backend-neutral recovery checkpoints
            # without perturbing filter/trust-region decisions.
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "iteration",
                        "iteration": int(iteration),
                        "parameters": x.copy(),
                        "objective_value": float(value),
                        "violation": float(violation),
                        "trust_radius": float(trust_radius),
                        "step_kind": str(step_kind),
                        "step_norm_inf": float(step_norm),
                        "line_search_alpha": float(accepted_alpha),
                        "reduction_ratio": float(reduction_ratio),
                        "accepted_steps": int(accepted_steps),
                        "rejected_steps": int(rejected_steps),
                        "objective_calls": int(counters.objective_evaluations),
                        "gradient_calls": int(counters.gradient_evaluations),
                    }
                )

            if violation < least_violation or (
                violation <= least_violation and value < least_violation_value
            ):
                least_violation_x = x.copy()
                least_violation_value = value
                least_violation_gradient = gradient.copy()
                least_violation = violation

            if _is_feasible(c, h, settings):
                if value < best_feasible_value:
                    best_feasible_x = x.copy()
                    best_feasible_value = value
                    best_feasible_gradient = gradient.copy()
                    best_feasible_violation = violation
                    if multiplier_estimate is not None:
                        best_kkt_stationarity = multiplier_estimate.stationarity_inf
                        best_kkt_stationarity_scaled = (
                            multiplier_estimate.stationarity_scaled
                        )
                        best_kkt_complementarity = (
                            multiplier_estimate.complementarity_inf
                        )
                        best_kkt_complementarity_scaled = (
                            multiplier_estimate.complementarity_scaled
                        )
                if (
                    settings.diagnostics.multiplier_termination
                    and multiplier_estimate is not None
                    and multiplier_estimate.success
                    and multiplier_estimate.stationarity_scaled
                    <= settings.kkt_tolerance
                    and multiplier_estimate.complementarity_scaled
                    <= settings.complementarity_tolerance
                ):
                    message = "finite-NLP KKT conditions satisfied"
                    status = 0
                    terminate_after_iteration = True

            if (
                accepted_alpha >= 0.999
                and math.isfinite(reduction_ratio)
                and reduction_ratio >= settings.good_reduction_ratio
                and step_norm >= 0.8 * trust_radius
            ):
                trust_radius = min(
                    settings.maximum_trust_radius,
                    trust_radius * settings.trust_expansion,
                )
            elif (
                not math.isfinite(reduction_ratio)
                or reduction_ratio < settings.poor_reduction_ratio
                or accepted_alpha < 0.5
            ):
                trust_radius = max(
                    settings.minimum_trust_radius,
                    trust_radius * settings.trust_contraction,
                )
        else:
            rejected_steps += 1
            trust_radius *= settings.trust_contraction

        history.append(
            SparseSQPIteration(
                iteration,
                value if accepted else old_value,
                violation,
                trust_radius,
                accepted,
                step_kind,
                step_norm,
                accepted_alpha,
                predicted_reduction,
                actual_reduction,
                reduction_ratio,
                trial_violation,
                normal_status,
                tangential_status,
                correction_used,
                objective_evaluated,
            )
        )
        if settings.display:
            print(
                f"sparse-sqp {iteration:3d}: f={value:.12g} "
                f"theta={violation:.3e} delta={trust_radius:.3e} "
                f"accepted={accepted} kind={step_kind}"
            )
        if terminate_after_iteration:
            break
        if (
            settings.max_accepted_steps is not None
            and accepted_steps >= settings.max_accepted_steps
        ):
            accepted_step_budget_reached = True
            message = "accepted-step batch budget reached"
            status = 1
            break

    else:
        iteration = settings.max_iterations


    # Final restoration is performed on a private copy of the filter
    # iterate.  When possible, use an objective-aware convex QP constrained by
    # the strict linearized feasible set.  A pure normal LP is retained as a
    # guaranteed fallback.  This is materially less destructive than an L1
    # projection when the filter iterate has made useful progress along the
    # boundary of its allowed violation band.
    if (
        violation > max(settings.feasibility_tolerance, settings.equality_tolerance)
        and settings.final_restoration_iterations > 0
    ):
        restore_x = x.copy()
        restore_c = c.copy()
        restore_jc = jc.copy()
        restore_h = h.copy()
        restore_jh = jh.copy()
        restore_violation = violation
        restore_radius = min(
            settings.final_restoration_trust_radius,
            settings.maximum_trust_radius,
        )
        for _ in range(settings.final_restoration_iterations):
            if _is_feasible(restore_c, restore_h, settings):
                break
            rlo, rhi = _step_bounds(restore_x, lower, upper, restore_radius)
            candidates: list[object] = []
            if settings.objective_aware_final_restoration:
                qp_started = time.perf_counter()
                objective_restore = _tangential_step(
                    gradient,
                    hessian,
                    restore_c,
                    restore_jc,
                    restore_h,
                    restore_jh,
                    rlo,
                    rhi,
                    0.0,
                    settings,
                    workspace,
                    warm_key="final_restore",
                )
                linear_program_seconds += time.perf_counter() - qp_started
                if bool(getattr(objective_restore, "qp_attempted", False)):
                    quadratic_programs += 1
                if not bool(getattr(objective_restore, "is_qp", False)):
                    linear_programs += 1
                    tangential_linear_programs += 1
                if bool(getattr(objective_restore, "success", False)):
                    candidates.append(objective_restore)

            # Always construct the lexicographic minimum-norm normal step as a
            # fallback.  A strict QP can be linearly feasible yet fail to
            # reduce the nonlinear residual because of constraint curvature.
            lp_started = time.perf_counter()
            normal_restore = _normal_step(
                restore_c,
                restore_jc,
                restore_h,
                restore_jh,
                rlo,
                rhi,
                settings,
                workspace,
                warm_prefix="final_normal",
            )
            linear_program_seconds += time.perf_counter() - lp_started
            restore_subproblems = int(
                getattr(normal_restore, "highs_subproblems", 1)
            )
            linear_programs += restore_subproblems
            normal_linear_programs += restore_subproblems
            if bool(getattr(normal_restore, "success", False)):
                candidates.append(normal_restore)

            restoration_candidates: list[
                tuple[bool, float, float, Array, Array, Array, Array, Array]
            ] = []
            for restore_raw in candidates:
                restore_step = np.asarray(restore_raw.x, dtype=float)[:n]
                restore_alpha = 1.0
                for _line in range(settings.maximum_line_search_steps):
                    candidate_x = np.clip(
                        restore_x + restore_alpha * restore_step, lower, upper
                    )
                    candidate_c, candidate_jc, candidate_h, candidate_jh = (
                        counters.constraints(constraints, candidate_x)
                    )
                    candidate_violation = _violation(candidate_c, candidate_h)
                    candidate_feasible = _is_feasible(
                        candidate_c, candidate_h, settings
                    )
                    if candidate_violation <= (
                        1.0 - settings.final_restoration_reduction
                    ) * restore_violation or candidate_feasible:
                        displacement = candidate_x - restore_x
                        model_change = float(
                            gradient @ displacement
                            + 0.5 * _hessian_quadratic(hessian, displacement)
                        )
                        restoration_candidates.append(
                            (
                                candidate_feasible,
                                candidate_violation,
                                model_change,
                                candidate_x,
                                candidate_c,
                                candidate_jc,
                                candidate_h,
                                candidate_jh,
                            )
                        )
                        break
                    restore_alpha *= settings.line_search_contraction

            restored = bool(restoration_candidates)
            if restored:
                # Feasibility dominates.  Among feasible corrections retain
                # the one with the best objective model; otherwise choose the
                # strongest nonlinear residual reduction.  This prevents a
                # gently improving objective-aware QP from starving the much
                # faster normal/Newton correction.
                feasible_candidates = [
                    item for item in restoration_candidates if item[0]
                ]
                if feasible_candidates:
                    chosen = min(feasible_candidates, key=lambda item: item[2])
                else:
                    chosen = min(
                        restoration_candidates,
                        key=lambda item: (item[1], item[2]),
                    )
                (
                    _, restore_violation, _, restore_x, restore_c,
                    restore_jc, restore_h, restore_jh,
                ) = chosen
                restoration_steps += 1
            if not restored:
                restore_radius *= settings.trust_contraction
                if restore_radius < settings.minimum_trust_radius:
                    break
        if _is_feasible(restore_c, restore_h, settings):
            try:
                restore_value, restore_gradient = counters.objective(
                    objective, restore_x
                )
            except Exception:
                pass
            else:
                if restore_value < best_feasible_value:
                    best_feasible_x = restore_x.copy()
                    best_feasible_value = restore_value
                    best_feasible_gradient = restore_gradient.copy()
                    best_feasible_violation = restore_violation

    if best_feasible_x is not None:
        returned_x = best_feasible_x
        returned_value = best_feasible_value
        assert best_feasible_gradient is not None
        returned_gradient = best_feasible_gradient
        returned_violation = best_feasible_violation
        if status == 1 and accepted_steps > 0:
            message = "iteration limit reached; returning best feasible incumbent"
    else:
        returned_x = least_violation_x
        returned_value = least_violation_value
        returned_gradient = least_violation_gradient
        returned_violation = least_violation
        if status == 1:
            message = "iteration limit reached without a feasible incumbent"

    usable = best_feasible_x is not None
    returned_estimate: _MultiplierEstimate | None = None
    best_kkt_criticality = math.inf
    best_kkt_criticality_success = False
    if usable:
        returned_c, returned_jc, returned_h, returned_jh = counters.constraints(
            constraints, returned_x
        )
        if settings.diagnostics.final_multipliers:
            returned_estimate = timed_multiplier_estimate(
                returned_x,
                returned_gradient,
                returned_c,
                returned_jc,
                returned_h,
                returned_jh,
            )
        if returned_estimate is not None:
            best_kkt_stationarity = returned_estimate.stationarity_inf
            best_kkt_stationarity_scaled = returned_estimate.stationarity_scaled
            best_kkt_complementarity = returned_estimate.complementarity_inf
            best_kkt_complementarity_scaled = (
                returned_estimate.complementarity_scaled
            )
        if settings.diagnostics.final_criticality:
            best_kkt_criticality, best_kkt_criticality_success = timed_criticality(
                returned_x,
                returned_gradient,
                returned_c,
                returned_jc,
                returned_h,
                returned_jh,
            )
    else:
        best_kkt_stationarity = math.inf
        best_kkt_stationarity_scaled = math.inf
        best_kkt_complementarity = math.inf
        best_kkt_complementarity_scaled = math.inf
    multiplier_converged = bool(
        usable
        and settings.diagnostics.final_multipliers
        and returned_estimate is not None
        and returned_estimate.success
        and best_kkt_stationarity_scaled <= settings.kkt_tolerance
        and best_kkt_complementarity_scaled <= settings.complementarity_tolerance
    )
    criticality_converged = bool(
        usable
        and settings.diagnostics.final_criticality
        and best_kkt_criticality_success
        and best_kkt_criticality <= settings.criticality_tolerance
    )
    diagnostics_complete = bool(
        usable
        and settings.diagnostics.final_multipliers
        and settings.diagnostics.final_criticality
        and returned_estimate is not None
        and best_kkt_criticality_success
    )
    converged = bool(
        diagnostics_complete
        and (
            multiplier_converged and criticality_converged
            if settings.diagnostics.strict_convergence_requires_all
            else multiplier_converged or criticality_converged
        )
    )
    first_order_acceptable = bool(
        converged
        or (
            usable
            and best_kkt_criticality_success
            and best_kkt_criticality <= settings.polish_criticality_tolerance
        )
    )
    if converged:
        status = 0
        message = "finite-NLP KKT conditions satisfied"
    elif usable:
        if status == 1:
            message = (
                "iteration budget reached; returning a primal-feasible "
                "incumbent before KKT convergence"
            )
        elif status == 2:
            message = (
                "minimum trust radius reached; returning a primal-feasible "
                "incumbent before KKT convergence"
            )
        elif status == 0:
            status = 5
            message = (
                "primal-feasible incumbent returned; requested KKT tolerance "
                "was not met"
            )
    if accepted_step_budget_reached and usable and not converged:
        message = (
            "accepted-step batch budget reached; returning a primal-feasible "
            "incumbent without claiming KKT convergence"
        )

    previous_accepted = continuation_state.accepted_steps if continuation_state else 0
    previous_raw = continuation_state.raw_iterations if continuation_state else 0
    previous_rejected = continuation_state.rejected_steps if continuation_state else 0
    previous_restoration = (
        continuation_state.restoration_steps if continuation_state else 0
    )
    prior_certified_x = (
        None
        if continuation_state is None or continuation_state.certified_x is None
        else continuation_state.certified_x.copy()
    )
    prior_certified_gradient = (
        None
        if continuation_state is None
        or continuation_state.certified_gradient is None
        else continuation_state.certified_gradient.copy()
    )
    counters.record_stage("subproblem_solve", linear_program_seconds)
    continuation_out = SparseSQPContinuationState(
        current_x=x.copy(),
        current_objective=float(value),
        current_gradient=gradient.copy(),
        current_inequalities=c.copy(),
        current_inequality_jacobian=jc.copy(),
        current_equalities=h.copy(),
        current_equality_jacobian=jh.copy(),
        best_feasible_x=(
            None if best_feasible_x is None else best_feasible_x.copy()
        ),
        best_feasible_objective=float(best_feasible_value),
        best_feasible_gradient=(
            None if best_feasible_gradient is None else best_feasible_gradient.copy()
        ),
        best_feasible_violation=float(best_feasible_violation),
        certified_x=prior_certified_x,
        certified_objective=(
            math.inf
            if continuation_state is None
            else float(continuation_state.certified_objective)
        ),
        certified_gradient=prior_certified_gradient,
        hessian=hessian.copy(),
        trust_radius=float(trust_radius),
        filter_entries=tuple(filter_set.entries),
        warm_starts=dict(workspace.warm_starts),
        successful_objective_latencies=tuple(
            counters.successful_objective_latencies[-settings.objective_latency_history_size:]
        ),
        accepted_steps=previous_accepted + accepted_steps,
        raw_iterations=previous_raw + len(history),
        rejected_steps=previous_rejected + rejected_steps,
        restoration_steps=previous_restoration + restoration_steps,
        constraint_generation=active_generation,
        objective_identity=(
            objective_identity
            if objective_identity is not None
            else (continuation_state.objective_identity if continuation_state else None)
        ),
        diagnostic_generation=active_generation if diagnostics_complete else -1,
        last_event=(
            f"objective_timeout:{counters.last_timeout_stage}"
            if counters.objective_timeouts
            else "accepted_step_budget" if accepted_step_budget_reached
            else "constraint_generation_changed" if generation_changed
            else "batch_complete"
        ),
        native_warm_starts_retained=bool(workspace.warm_starts),
        scaling_shift=None if continuation_state is None else continuation_state.scaling_shift,
        scaling_variable=None if continuation_state is None else continuation_state.scaling_variable,
        scaling_inequality=None if continuation_state is None else continuation_state.scaling_inequality,
        scaling_equality=None if continuation_state is None else continuation_state.scaling_equality,
        scaling_lower=None if continuation_state is None else continuation_state.scaling_lower,
        scaling_upper=None if continuation_state is None else continuation_state.scaling_upper,
    )
    success = usable
    return SparseSQPResult(
        returned_x.copy(),
        float(returned_value),
        returned_gradient.copy(),
        x.copy(),
        float(value),
        gradient.copy(),
        success,
        status,
        message,
        len(history),
        counters.objective_evaluations,
        counters.gradient_evaluations,
        counters.objective_failures,
        counters.constraint_evaluations,
        linear_programs,
        quadratic_programs,
        normal_linear_programs,
        tangential_linear_programs,
        second_order_linear_programs,
        accepted_steps,
        rejected_steps,
        restoration_steps,
        screened_trials,
        counters.objective_seconds,
        counters.constraint_seconds,
        linear_program_seconds,
        time.perf_counter() - started,
        float(returned_violation),
        float(violation),
        float(best_feasible_violation),
        float(trust_radius),
        converged,
        usable,
        first_order_acceptable,
        float(best_kkt_stationarity),
        float(best_kkt_stationarity_scaled),
        float(best_kkt_complementarity),
        float(best_kkt_complementarity_scaled),
        float(best_kkt_criticality),
        bool(best_kkt_criticality_success),
        _hessian_nonzeros(hessian),
        counters.maximum_jacobian_nonzeros,
        workspace.warm_start_attempts,
        workspace.warm_start_uses,
        workspace.qp_rejections,
        workspace.qp_cold_retries,
        workspace.qp_lp_fallbacks,
        workspace.backend_name,
        workspace.backend_public,
        tuple(history),
        continuation_out,
        SparseSQPStageTelemetry.from_dicts(
            counters.stage_totals, counters.stage_calls, counters.stage_maxima
        ),
        multiplier_diagnostic_calls,
        multiplier_diagnostic_seconds,
        criticality_diagnostic_calls,
        criticality_diagnostic_seconds,
        diagnostics_complete,
        diagnostic_failure,
        counters.objective_timeouts,
        counters.last_timeout_stage,
        accepted_step_budget_reached,
    )


__all__ = [
    "SparseSQPIteration",
    "SparseSQPResult",
    "ObjectiveEvaluationTimeout",
    "SparseSQPContinuationState",
    "SparseSQPDiagnosticControls",
    "SparseSQPStageTelemetry",
    "SparseSQPSettings",
    "solve_sparse_filter_sqp",
]


def _merge_stage_telemetry(
    primary: SparseSQPStageTelemetry,
    polish: SparseSQPStageTelemetry,
) -> SparseSQPStageTelemetry:
    totals = dict(primary.totals)
    calls = dict(primary.calls)
    maxima = dict(primary.maxima)
    for key, value in polish.totals:
        totals[key] = totals.get(key, 0.0) + float(value)
    for key, value in polish.calls:
        calls[key] = calls.get(key, 0) + int(value)
    for key, value in polish.maxima:
        maxima[key] = max(maxima.get(key, 0.0), float(value))
    return SparseSQPStageTelemetry.from_dicts(totals, calls, maxima)


def _merge_operational_polish_results(
    primary: SparseSQPResult,
    polish: SparseSQPResult,
    *,
    use_polish_point: bool,
) -> SparseSQPResult:
    chosen = polish if use_polish_point else primary
    history_offset = len(primary.history)
    polish_history = tuple(
        replace(item, index=item.index + history_offset) for item in polish.history
    )
    return replace(
        chosen,
        message=(
            f"{chosen.message}; operational polish "
            f"{'accepted' if use_polish_point else 'retained primary incumbent'}"
        ),
        iterations=primary.iterations + polish.iterations,
        objective_evaluations=(
            primary.objective_evaluations + polish.objective_evaluations
        ),
        gradient_evaluations=(
            primary.gradient_evaluations + polish.gradient_evaluations
        ),
        objective_failures=primary.objective_failures + polish.objective_failures,
        constraint_evaluations=(
            primary.constraint_evaluations + polish.constraint_evaluations
        ),
        linear_programs=primary.linear_programs + polish.linear_programs,
        quadratic_programs=primary.quadratic_programs + polish.quadratic_programs,
        normal_linear_programs=(
            primary.normal_linear_programs + polish.normal_linear_programs
        ),
        tangential_linear_programs=(
            primary.tangential_linear_programs + polish.tangential_linear_programs
        ),
        second_order_linear_programs=(
            primary.second_order_linear_programs
            + polish.second_order_linear_programs
        ),
        accepted_steps=primary.accepted_steps + polish.accepted_steps,
        rejected_steps=primary.rejected_steps + polish.rejected_steps,
        restoration_steps=primary.restoration_steps + polish.restoration_steps,
        screened_trials=primary.screened_trials + polish.screened_trials,
        objective_seconds=primary.objective_seconds + polish.objective_seconds,
        constraint_seconds=primary.constraint_seconds + polish.constraint_seconds,
        linear_program_seconds=(
            primary.linear_program_seconds + polish.linear_program_seconds
        ),
        wall_seconds=primary.wall_seconds + polish.wall_seconds,
        warm_start_attempts=(
            primary.warm_start_attempts + polish.warm_start_attempts
        ),
        warm_start_uses=primary.warm_start_uses + polish.warm_start_uses,
        qp_rejections=primary.qp_rejections + polish.qp_rejections,
        qp_cold_retries=primary.qp_cold_retries + polish.qp_cold_retries,
        qp_lp_fallbacks=primary.qp_lp_fallbacks + polish.qp_lp_fallbacks,
        history=primary.history + polish_history,
        stage_telemetry=_merge_stage_telemetry(
            primary.stage_telemetry, polish.stage_telemetry
        ),
        multiplier_diagnostic_calls=(
            primary.multiplier_diagnostic_calls
            + polish.multiplier_diagnostic_calls
        ),
        multiplier_diagnostic_seconds=(
            primary.multiplier_diagnostic_seconds
            + polish.multiplier_diagnostic_seconds
        ),
        criticality_diagnostic_calls=(
            primary.criticality_diagnostic_calls
            + polish.criticality_diagnostic_calls
        ),
        criticality_diagnostic_seconds=(
            primary.criticality_diagnostic_seconds
            + polish.criticality_diagnostic_seconds
        ),
        objective_timeouts=primary.objective_timeouts + polish.objective_timeouts,
        last_timeout_stage=polish.last_timeout_stage or primary.last_timeout_stage,
    )


def solve_sparse_filter_sqp(
    x0: Sequence[float],
    objective: DifferentiableObjective,
    constraints: DifferentiableConstraints,
    *,
    lower_bounds: Sequence[float] | None = None,
    upper_bounds: Sequence[float] | None = None,
    settings: SparseSQPSettings = SparseSQPSettings(),
    continuation_state: SparseSQPContinuationState | None = None,
    constraint_generation: int | None = None,
    objective_identity: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SparseSQPResult:
    """Solve a finite NLP, with a targeted operational polish when needed.

    The ordinary production policy retains the historically better
    ``1e-10`` model-reduction floor.  Only an unrestricted, fully diagnosed,
    primal-usable solve that misses the operational criticality gate is
    restarted from its safe feasible incumbent with the tighter polish floor.
    Accepted-step continuation batches and restoration calls never trigger
    this second stage.
    """

    primary = _solve_sparse_filter_sqp_once(
        x0,
        objective,
        constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        settings=settings,
        continuation_state=continuation_state,
        constraint_generation=constraint_generation,
        objective_identity=objective_identity,
        progress_callback=progress_callback,
    )
    should_polish = bool(
        settings.operational_polish_enabled
        and settings.max_accepted_steps is None
        and not settings.diagnostic_only
        and primary.usable
        and primary.diagnostics_complete
        and not primary.first_order_acceptable
        and settings.operational_polish_minimum_predicted_reduction
        < settings.minimum_predicted_reduction
        and primary.objective_timeouts == 0
    )
    if not should_polish:
        return primary

    polish_settings = replace(
        settings,
        max_iterations=settings.operational_polish_max_iterations,
        minimum_predicted_reduction=(
            settings.operational_polish_minimum_predicted_reduction
        ),
        operational_polish_enabled=False,
    )
    polish_start = (
        primary.current_x
        if primary.current_violation
        <= max(settings.maximum_filter_violation, settings.feasibility_tolerance)
        else primary.x
    )
    polish = _solve_sparse_filter_sqp_once(
        polish_start,
        objective,
        constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        settings=polish_settings,
        continuation_state=None,
        constraint_generation=constraint_generation,
        objective_identity=objective_identity,
        progress_callback=progress_callback,
    )
    use_polish = bool(
        polish.usable
        and (
            polish.first_order_acceptable
            or (
                polish.kkt_criticality_success
                and (
                    not primary.kkt_criticality_success
                    or polish.kkt_criticality < primary.kkt_criticality
                )
            )
        )
    )
    return _merge_operational_polish_results(
        primary, polish, use_polish_point=use_polish
    )
