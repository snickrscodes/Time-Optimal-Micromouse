"""Finite constrained optimization and weighted path-objective orchestration.

This module owns the optimizer-facing layer:

* scaled SciPy SLSQP finite solves (SciPy is imported lazily),
* exact constraint-generation exchange rounds,
* elastic Phase-I corridor restoration, and
* the final weighted path optimizer.

The geometry/corridor model and exact separation oracle live in
:mod:`.constraint_generation`.  Public path parameters use the flat knot basis
``[s1, k1, s2, k2, ..., sn, kn]``.  By default SLSQP internally uses
log segment lengths, preserving strict knot ordering without local station boxes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import reverse_solver, scalar_reverse_solver
from .constraint_generation import (
    CombinedConstraintOracle,
    ConstraintPool,
    CorridorModel,
    EndpointPoseEquality,
    KnotGeometryCache,
    RectangleConstraintOracle,
    ScalarObjective,
    SeparationReport,
    SeparationSettings,
    VectorConstraint,
    add_report_violations,
    separate_compiled_path,
    stack_vector_constraints,
)
from .geometry_gradients import (
    GeometryPath,
    GeometryState,
    compile_geometry_path,
    knot_parameters_to_raw,
    pullback_raw_gradient_to_knot_parameters,
)
from .kkt_lsq import BoxBounds, KKTReport, estimate_kkt_residual
from .knot_parameterization import KnotCoordinateSettings, LogLengthKnotMap
from .phase_one_handoff import (
    align_endpoint_target_to_path,
    canonicalize_phase_one_handoff,
)
from .sparse_sqp import (
    ObjectiveEvaluationTimeout,
    SparseSQPContinuationState,
    SparseSQPResult,
    SparseSQPSettings,
    solve_sparse_filter_sqp,
)
from .total_gradients import (
    EndpointObjective,
    GeometryObjective,
    curvature_energy_value_and_raw_gradient,
)

Array: TypeAlias = NDArray[np.float64]
@dataclass(slots=True)
class CachedObjective:
    """Two-level optimizer point cache.

    ``fun`` may evaluate only the scalar objective.  ``jac`` upgrades the same
    point to a differentiable result when needed.  Objectives that do not
    expose specialized ``fun``/``jac`` methods retain the historical combined
    ``(value, gradient)`` behavior.
    """

    objective: ScalarObjective
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    seconds: float = 0.0
    fun_calls: int = 0
    jac_calls: int = 0
    scalar_evaluations: int = 0
    differentiable_evaluations: int = 0
    scalar_cache_hits: int = 0
    differentiable_cache_hits: int = 0
    scalar_to_jac_upgrades: int = 0
    jac_without_scalar: int = 0
    _x: Array | None = field(default=None, init=False, repr=False)
    _value: float | None = field(default=None, init=False, repr=False)
    _gradient: Array | None = field(default=None, init=False, repr=False)

    def _same_point(self, array: Array) -> bool:
        return self._x is not None and np.array_equal(array, self._x)

    def _validate_value(self, value: object) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise FloatingPointError("objective returned nonfinite value")
        return result

    def _validate_gradient(self, gradient: object, array: Array) -> Array:
        result = np.asarray(gradient, dtype=float)
        if result.shape != array.shape:
            raise ValueError("objective gradient has wrong shape")
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("objective returned nonfinite gradient")
        return result

    def _store_full(self, array: Array, value: object, gradient: object) -> tuple[float, Array]:
        checked_value = self._validate_value(value)
        checked_gradient = self._validate_gradient(gradient, array)
        self._x = array.copy()
        self._value = checked_value
        self._gradient = checked_gradient.copy()
        return checked_value, checked_gradient

    def _full_objective(self, array: Array) -> tuple[float, Array]:
        evaluator = getattr(self.objective, "evaluate", None)
        if evaluator is not None:
            result = evaluator(array)
            if hasattr(result, "value") and hasattr(result, "gradient"):
                return self._store_full(array, result.value, result.gradient)
        value, gradient = self.objective(array)
        return self._store_full(array, value, gradient)

    def evaluate(self, x: Sequence[float]) -> tuple[float, Array]:
        array = np.asarray(x, dtype=float)
        if self._same_point(array) and self._value is not None and self._gradient is not None:
            self.cache_hits += 1
            self.differentiable_cache_hits += 1
            return self._value, self._gradient
        started = time.perf_counter()
        self.calls += 1
        self.differentiable_evaluations += 1
        upgrading = self._same_point(array) and self._value is not None and self._gradient is None
        if upgrading:
            self.scalar_to_jac_upgrades += 1
        else:
            self.jac_without_scalar += 1
        try:
            return self._full_objective(array)
        except Exception:
            self.failures += 1
            raise
        finally:
            self.seconds += time.perf_counter() - started

    def evaluate_with_deadline(
        self, x: Sequence[float], *, deadline: float
    ) -> tuple[float, Array]:
        array = np.asarray(x, dtype=float)
        if self._same_point(array) and self._value is not None and self._gradient is not None:
            self.cache_hits += 1
            self.differentiable_cache_hits += 1
            return self._value, self._gradient
        started = time.perf_counter()
        self.calls += 1
        self.differentiable_evaluations += 1
        try:
            evaluator = getattr(self.objective, "evaluate_with_deadline", None)
            if evaluator is None:
                value, gradient = self._full_objective(array)
            else:
                value, gradient = evaluator(array, deadline=deadline)
                value, gradient = self._store_full(array, value, gradient)
            if time.monotonic() > deadline:
                raise ObjectiveEvaluationTimeout(
                    "objective", "objective exceeded its cooperative deadline"
                )
            return value, gradient
        except Exception:
            self.failures += 1
            raise
        finally:
            self.seconds += time.perf_counter() - started

    def fun(self, x: Sequence[float]) -> float:
        array = np.asarray(x, dtype=float)
        self.fun_calls += 1
        if self._same_point(array) and self._value is not None:
            self.cache_hits += 1
            self.scalar_cache_hits += 1
            return self._value
        scalar = getattr(self.objective, "fun", None)
        if scalar is None:
            return self.evaluate(array)[0]
        started = time.perf_counter()
        self.calls += 1
        self.scalar_evaluations += 1
        try:
            value = self._validate_value(scalar(array))
        except Exception:
            self.failures += 1
            raise
        finally:
            self.seconds += time.perf_counter() - started
        self._x = array.copy()
        self._value = value
        self._gradient = None
        return value

    def jac(self, x: Sequence[float]) -> Array:
        array = np.asarray(x, dtype=float)
        self.jac_calls += 1
        if self._same_point(array) and self._gradient is not None:
            self.cache_hits += 1
            self.differentiable_cache_hits += 1
            return self._gradient
        return self.evaluate(array)[1]

    def call_statistics(self) -> dict[str, int]:
        return {
            "fun_calls": self.fun_calls,
            "jac_calls": self.jac_calls,
            "scalar_evaluations": self.scalar_evaluations,
            "differentiable_evaluations": self.differentiable_evaluations,
            "scalar_cache_hits": self.scalar_cache_hits,
            "differentiable_cache_hits": self.differentiable_cache_hits,
            "scalar_to_jac_upgrades": self.scalar_to_jac_upgrades,
            "jac_without_scalar": self.jac_without_scalar,
        }


# =============================================================================
# SLSQP finite solve and exchange loop
# =============================================================================


@dataclass(frozen=True, slots=True)
class ScalingSettings:
    """Fixed affine variable scaling and frozen row scaling for SLSQP.

    ``variable_scales`` supplies explicit scales in the active optimizer
    coordinates (log lengths by default, cumulative stations in legacy mode).
    Automatic variable scaling is opt-in because generic bound-width heuristics
    can badly condition the interleaved knot-position/curvature basis. Constraint divisors are
    frozen at the finite solve's initial point, so scaled callbacks retain exact
    Jacobians.
    """

    enabled: bool = True
    variable_scales: tuple[float, ...] | None = None
    automatic_variable_scaling: bool = False
    minimum_variable_scale: float = 1.0e-3
    scale_constraints: bool = True
    minimum_constraint_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_variable_scale) or self.minimum_variable_scale <= 0.0:
            raise ValueError("minimum_variable_scale must be finite and positive")
        if not math.isfinite(self.minimum_constraint_scale) or self.minimum_constraint_scale <= 0.0:
            raise ValueError("minimum_constraint_scale must be finite and positive")
        if self.variable_scales is not None and not all(
            math.isfinite(v) and v > 0.0 for v in self.variable_scales
        ):
            raise ValueError("variable_scales must contain finite positive values")


@dataclass(frozen=True, slots=True)
class SLSQPSettings:
    max_iterations: int = 500
    ftol: float = 1.0e-14
    display: bool = False
    scaling: ScalingSettings = ScalingSettings()

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(self.ftol) or self.ftol <= 0.0:
            raise ValueError("ftol must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExchangeSettings:
    maximum_rounds: int = 12
    finite_inequality_tolerance: float = 1.0e-9
    finite_equality_tolerance: float = 1.0e-9
    kkt_tolerance: float = 2.0e-7
    active_constraint_tolerance: float = 1.0e-7
    merge_distance: float = 1.0e-8
    require_solver_success: bool = True
    separation: SeparationSettings = SeparationSettings()
    finite_solver: Literal["slsqp", "sparse_sqp"] = "slsqp"
    slsqp: SLSQPSettings = SLSQPSettings()
    sparse_sqp: SparseSQPSettings = SparseSQPSettings()
    certified_polish_rounds: int = 0
    final_kkt_diagnostics: bool = True

    def __post_init__(self) -> None:
        if self.maximum_rounds <= 0:
            raise ValueError("maximum_rounds must be positive")
        if self.certified_polish_rounds < 0:
            raise ValueError("certified_polish_rounds must be nonnegative")
        if self.finite_solver not in {"slsqp", "sparse_sqp"}:
            raise ValueError("finite_solver must be 'slsqp' or 'sparse_sqp'")
        values = (
            self.finite_inequality_tolerance,
            self.finite_equality_tolerance,
            self.kkt_tolerance,
            self.active_constraint_tolerance,
            self.merge_distance,
        )
        if not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("exchange tolerances must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SolveScaling:
    shift: Array
    variable_scale: Array
    inequality_scale: Array
    equality_scale: Array
    scaled_bounds: BoxBounds | None

    def to_scaled(self, x: Sequence[float]) -> Array:
        return (np.asarray(x, dtype=float) - self.shift) / self.variable_scale

    def to_physical(self, z: Sequence[float]) -> Array:
        return self.shift + self.variable_scale * np.asarray(z, dtype=float)


@dataclass(frozen=True, slots=True)
class NonlinearSolveResult:
    """Backend-neutral result returned by a nonlinear solver adapter."""

    x: Array
    objective: float
    gradient: Array | None
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    backend: str
    scaled_x: Array | None = None
    scaled_gradient: Array | None = None
    diagnostics: object | None = None


@dataclass(frozen=True, slots=True)
class FiniteSolveResult:
    x: Array
    objective: float
    success: bool
    message: str
    iterations: int
    max_inequality: float
    equality_inf: float
    kkt: KKTReport
    objective_calls: int
    objective_failures: int
    objective_seconds: float
    constraint_calls: int
    constraint_seconds: float
    optimizer_seconds: float
    solver_result: NonlinearSolveResult
    scaling: SolveScaling | None = None
    objective_call_statistics: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class ExchangeRoundResult:
    round_index: int
    pool_size_before: int
    pool_size_after: int
    cuts_added: int
    worst_continuous_upper_bound: float
    worst_exact_violation: float
    finite: FiniteSolveResult


@dataclass(frozen=True, slots=True)
class ExchangeResult:
    x: Array
    objective: float
    success: bool
    message: str
    pool: ConstraintPool
    rounds: tuple[ExchangeRoundResult, ...]
    final_separation: SeparationReport
    final_finite_inequality: float
    final_equality_inf: float
    final_kkt: KKTReport
    wall_seconds: float
    current_x: Array | None = None
    current_objective: float = math.inf
    continuation_state: SparseSQPContinuationState | None = None
    constraint_generation: int = 0


def _normalize_bounds(
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None,
    width: int,
) -> BoxBounds | None:
    if bounds is None:
        return None
    if isinstance(bounds, BoxBounds):
        lower = np.broadcast_to(bounds.lower, (width,)).astype(float, copy=True)
        upper = np.broadcast_to(bounds.upper, (width,)).astype(float, copy=True)
    elif hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        # Runtime compatibility with legacy/scientific bounds containers,
        # without importing or exposing their concrete type.
        lower = np.broadcast_to(
            np.asarray(getattr(bounds, "lb"), dtype=float), (width,)
        ).copy()
        upper = np.broadcast_to(
            np.asarray(getattr(bounds, "ub"), dtype=float), (width,)
        ).copy()
    else:
        if len(bounds) != width:
            raise ValueError("bounds length does not match variable count")
        lower = np.array(
            [-math.inf if lo is None else lo for lo, _ in bounds], dtype=float
        )
        upper = np.array(
            [math.inf if hi is None else hi for _, hi in bounds], dtype=float
        )
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("bounds must not contain NaNs")
    return BoxBounds(lower, upper)



def _prepare_log_length_bounds(
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None,
    width: int,
    settings: KnotCoordinateSettings,
) -> tuple[BoxBounds | None, BoxBounds]:
    """Split public knot bounds into hard station rows and internal boxes."""
    physical = _normalize_bounds(bounds, width)
    lower = np.full(width, -math.inf, dtype=float)
    upper = np.full(width, math.inf, dtype=float)
    guard = settings.maximum_abs_log_length
    lower[0::2] = max(-guard, math.log(settings.minimum_length_ratio))
    upper_log = guard
    if settings.maximum_length_ratio is not None:
        upper_log = min(upper_log, math.log(settings.maximum_length_ratio))
    upper[0::2] = upper_log
    if physical is not None:
        lower[1::2] = np.asarray(physical.lower, dtype=float)[1::2]
        upper[1::2] = np.asarray(physical.upper, dtype=float)[1::2]
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    return physical, BoxBounds(lower, upper)


def _station_bound_values_and_jacobian(
    x: Array,
    bounds: BoxBounds | None,
) -> tuple[Array, Array]:
    """Return finite cumulative-station bound rows using the ``c <= 0`` sign."""
    if bounds is None:
        return np.empty(0, dtype=float), np.empty((0, x.size), dtype=float)
    lower = np.asarray(bounds.lower, dtype=float)
    upper = np.asarray(bounds.upper, dtype=float)
    values: list[float] = []
    rows: list[Array] = []
    for index in range(0, x.size, 2):
        if math.isfinite(float(lower[index])):
            row = np.zeros(x.size, dtype=float)
            row[index] = -1.0
            values.append(float(lower[index] - x[index]))
            rows.append(row)
        if math.isfinite(float(upper[index])):
            row = np.zeros(x.size, dtype=float)
            row[index] = 1.0
            values.append(float(x[index] - upper[index]))
            rows.append(row)
    if not values:
        return np.empty(0, dtype=float), np.empty((0, x.size), dtype=float)
    return np.asarray(values, dtype=float), np.vstack(rows)


@dataclass(slots=True)
class _LogLengthObjective:
    physical: ScalarObjective
    mapping: LogLengthKnotMap
    _u: Array | None = field(default=None, init=False, repr=False)
    _x: Array | None = field(default=None, init=False, repr=False)
    _value: float | None = field(default=None, init=False, repr=False)
    _physical_gradient: Array | None = field(default=None, init=False, repr=False)
    _internal_gradient: Array | None = field(default=None, init=False, repr=False)

    def evaluate(self, variables: Sequence[float]) -> tuple[float, Array]:
        u = np.asarray(variables, dtype=float)
        same = self._u is not None and np.array_equal(u, self._u)
        if same and self._value is not None and self._internal_gradient is not None:
            return self._value, self._internal_gradient
        x = self._x if same and self._x is not None else self.mapping.to_physical(u)
        value, gradient_in = self.physical(x)
        value = float(value)
        physical_gradient = np.asarray(gradient_in, dtype=float)
        if physical_gradient.shape != x.shape:
            raise ValueError("physical objective gradient has wrong shape")
        internal_gradient = self.mapping.pullback_gradient(u, physical_gradient)
        if not math.isfinite(value) or not np.all(np.isfinite(internal_gradient)):
            raise FloatingPointError("transformed objective returned nonfinite data")
        self._u = u.copy()
        self._x = x
        self._value = value
        self._physical_gradient = physical_gradient.copy()
        self._internal_gradient = internal_gradient
        return value, internal_gradient

    def fun(self, variables: Sequence[float]) -> float:
        """Scalar-only transformed objective, preserving physical specialization."""
        u = np.asarray(variables, dtype=float)
        if self._u is not None and np.array_equal(u, self._u) and self._value is not None:
            return self._value
        x = self.mapping.to_physical(u)
        scalar = getattr(self.physical, "fun", None)
        if scalar is None:
            return self.evaluate(u)[0]
        value = float(scalar(x))
        if not math.isfinite(value):
            raise FloatingPointError("transformed scalar objective returned nonfinite value")
        self._u = u.copy()
        self._x = x
        self._value = value
        self._physical_gradient = None
        self._internal_gradient = None
        return value

    def jac(self, variables: Sequence[float]) -> Array:
        """Upgrade a scalar transformed point to a gradient without rebuilding topology."""
        u = np.asarray(variables, dtype=float)
        if self._u is not None and np.array_equal(u, self._u) and self._internal_gradient is not None:
            return self._internal_gradient
        x = (
            self._x
            if self._u is not None and np.array_equal(u, self._u) and self._x is not None
            else self.mapping.to_physical(u)
        )
        jacobian = getattr(self.physical, "jac", None)
        if jacobian is None:
            return self.evaluate(u)[1]
        physical_gradient = np.asarray(jacobian(x), dtype=float)
        if physical_gradient.shape != x.shape:
            raise ValueError("physical objective gradient has wrong shape")
        internal_gradient = self.mapping.pullback_gradient(u, physical_gradient)
        if not np.all(np.isfinite(internal_gradient)):
            raise FloatingPointError("transformed objective returned nonfinite gradient")
        # The physical objective's scalar and differentiable paths enforce their
        # own value identity.  Preserve a scalar value already cached here; if
        # jac arrives first, recover the value from the physical full cache.
        if self._value is None or self._u is None or not np.array_equal(u, self._u):
            evaluator = getattr(self.physical, "evaluate", None)
            if evaluator is not None:
                result = evaluator(x)
                value = float(result.value) if hasattr(result, "value") else float(self.physical(x)[0])
            else:
                value = float(self.physical(x)[0])
            self._value = value
        self._u = u.copy()
        self._x = np.asarray(x, dtype=float).copy()
        self._physical_gradient = physical_gradient.copy()
        self._internal_gradient = internal_gradient
        return internal_gradient

    def evaluate_with_deadline(
        self, variables: Sequence[float], *, deadline: float
    ) -> tuple[float, Array]:
        u = np.asarray(variables, dtype=float)
        if self._u is not None and np.array_equal(u, self._u):
            assert self._value is not None and self._internal_gradient is not None
            return self._value, self._internal_gradient
        x = self.mapping.to_physical(u)
        evaluator = getattr(self.physical, "evaluate_with_deadline", None)
        if evaluator is None:
            value, gradient_in = self.physical(x)
        else:
            value, gradient_in = evaluator(x, deadline=deadline)
        if time.monotonic() > deadline:
            raise ObjectiveEvaluationTimeout(
                "objective", "transformed objective exceeded its deadline"
            )
        value = float(value)
        physical_gradient = np.asarray(gradient_in, dtype=float)
        if physical_gradient.shape != x.shape:
            raise ValueError("physical objective gradient has wrong shape")
        internal_gradient = self.mapping.pullback_gradient(u, physical_gradient)
        if not math.isfinite(value) or not np.all(np.isfinite(internal_gradient)):
            raise FloatingPointError("transformed objective returned nonfinite data")
        self._u = u.copy()
        self._x = x
        self._value = value
        self._physical_gradient = physical_gradient.copy()
        self._internal_gradient = internal_gradient
        return value, internal_gradient

    def __call__(self, variables: Sequence[float]) -> tuple[float, Array]:
        return self.evaluate(variables)

    def physical_evaluation(self, variables: Sequence[float]) -> tuple[Array, float, Array]:
        u = np.asarray(variables, dtype=float)
        self.evaluate(u)
        assert self._x is not None
        assert self._value is not None
        assert self._physical_gradient is not None
        return self._x.copy(), self._value, self._physical_gradient.copy()


@dataclass(slots=True)
class _LogLengthConstraints:
    original: CombinedConstraintOracle
    mapping: LogLengthKnotMap
    physical_bounds: BoxBounds | None
    _u: Array | None = field(default=None, init=False, repr=False)
    _result: tuple[Array, Array, Array, Array] | None = field(
        default=None, init=False, repr=False
    )
    _sparse_result: tuple[Array, object, Array, object] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def rectangle(self) -> RectangleConstraintOracle:
        return self.original.rectangle

    def evaluate(self, variables: Sequence[float]) -> tuple[Array, Array, Array, Array]:
        u = np.asarray(variables, dtype=float)
        if self._u is not None and np.array_equal(u, self._u):
            if self._result is None:
                assert self._sparse_result is not None
                c, jc, h, jh = self._sparse_result
                self._result = (c, jc.toarray(), h, jh.toarray())
            return self._result
        x = self.mapping.to_physical(u)
        c, jc_physical, h, jh_physical = self.original.evaluate(x)
        jc = self.mapping.pullback_jacobian(u, jc_physical)
        jh = self.mapping.pullback_jacobian(u, jh_physical)
        station_c, station_j_physical = _station_bound_values_and_jacobian(
            x, self.physical_bounds
        )
        if station_c.size:
            station_j = self.mapping.pullback_jacobian(u, station_j_physical)
            c = np.concatenate((c, station_c))
            jc = np.vstack((jc, station_j))
        if not all(np.all(np.isfinite(v)) for v in (c, jc, h, jh)):
            raise FloatingPointError("transformed constraints returned nonfinite data")
        self._u = u.copy()
        self._result = (c, jc, h, jh)
        self._sparse_result = None
        return self._result

    def evaluate_sparse(self, variables: Sequence[float]):
        from scipy.sparse import csr_matrix, vstack

        u = np.asarray(variables, dtype=float)
        if self._u is not None and np.array_equal(u, self._u):
            if self._sparse_result is not None:
                return self._sparse_result
            assert self._result is not None
            c, jc, h, jh = self._result
            self._sparse_result = (c, csr_matrix(jc), h, csr_matrix(jh))
            return self._sparse_result
        x = self.mapping.to_physical(u)
        if hasattr(self.original, "evaluate_sparse"):
            c, jc_physical, h, jh_physical = self.original.evaluate_sparse(x)
        else:
            c, jc_physical, h, jh_physical = self.original.evaluate(x)
        jc = self.mapping.pullback_sparse_jacobian(u, jc_physical)
        jh = self.mapping.pullback_sparse_jacobian(u, jh_physical)
        station_c, station_j_physical = _station_bound_values_and_jacobian(
            x, self.physical_bounds
        )
        if station_c.size:
            station_j = self.mapping.pullback_sparse_jacobian(
                u, station_j_physical
            )
            c = np.concatenate((c, station_c))
            jc = vstack((jc, station_j), format="csr")
        if not all(
            np.all(np.isfinite(v))
            for v in (c, jc.data, h, jh.data)
        ):
            raise FloatingPointError(
                "transformed constraints returned nonfinite data"
            )
        self._u = u.copy()
        self._result = None
        self._sparse_result = (c, jc, h, jh)
        return self._sparse_result

    def inequality_values(self, variables: Sequence[float]) -> Array:
        return self.evaluate(variables)[0]

    def inequality_jacobian(self, variables: Sequence[float]) -> Array:
        return self.evaluate(variables)[1]

    def equality_values(self, variables: Sequence[float]) -> Array:
        return self.evaluate(variables)[2]

    def equality_jacobian(self, variables: Sequence[float]) -> Array:
        return self.evaluate(variables)[3]


@dataclass(slots=True)
class _LogLengthPhaseConstraints:
    rectangle: RectangleConstraintOracle
    equalities: VectorConstraint | None
    additional_inequalities: VectorConstraint | None
    mapping: LogLengthKnotMap
    physical_bounds: BoxBounds | None
    _q: Array | None = field(default=None, init=False, repr=False)
    _result: tuple[Array, Array, Array, Array] | None = field(
        default=None, init=False, repr=False
    )

    def evaluate(self, augmented: Sequence[float]) -> tuple[Array, Array, Array, Array]:
        q = np.asarray(augmented, dtype=float)
        n = self.mapping.width
        if q.shape != (n + 1,):
            raise ValueError("transformed phase-I vector has wrong shape")
        if self._q is not None and np.array_equal(q, self._q):
            assert self._result is not None
            return self._result
        u = q[:n]
        rho = float(q[n])
        x = self.mapping.to_physical(u)
        corridor_c, corridor_j_physical = self.rectangle.evaluate(x)
        corridor_j = self.mapping.pullback_jacobian(u, corridor_j_physical)
        c_corridor = corridor_c - rho
        jc_corridor = np.empty((corridor_c.size, n + 1), dtype=float)
        jc_corridor[:, :n] = corridor_j
        jc_corridor[:, n] = -1.0

        station_c, station_j_physical = _station_bound_values_and_jacobian(
            x, self.physical_bounds
        )
        if station_c.size:
            station_j = self.mapping.pullback_jacobian(u, station_j_physical)
            jc_station = np.zeros((station_c.size, n + 1), dtype=float)
            jc_station[:, :n] = station_j
            c_out = np.concatenate((c_corridor, station_c))
            jc_out = np.vstack((jc_corridor, jc_station))
        else:
            c_out = c_corridor
            jc_out = jc_corridor

        if self.additional_inequalities is not None:
            extra_c_in, extra_j_physical = self.additional_inequalities(x)
            extra_c = np.asarray(extra_c_in, dtype=float)
            extra_j_internal = self.mapping.pullback_jacobian(
                u, np.asarray(extra_j_physical, dtype=float)
            )
            if extra_j_internal.shape != (extra_c.size, n):
                raise ValueError(
                    "transformed phase-I additional-inequality Jacobian has wrong shape"
                )
            if extra_c.size:
                extra_j = np.zeros((extra_c.size, n + 1), dtype=float)
                extra_j[:, :n] = extra_j_internal
                c_out = np.concatenate((c_out, extra_c))
                jc_out = np.vstack((jc_out, extra_j))

        if self.equalities is None:
            h_out = np.empty(0, dtype=float)
            jh_out = np.empty((0, n + 1), dtype=float)
        else:
            h_in, jh_physical = self.equalities(x)
            h_out = np.asarray(h_in, dtype=float)
            jh_internal = self.mapping.pullback_jacobian(u, jh_physical)
            jh_out = np.zeros((h_out.size, n + 1), dtype=float)
            jh_out[:, :n] = jh_internal
        if not all(np.all(np.isfinite(v)) for v in (c_out, jc_out, h_out, jh_out)):
            raise FloatingPointError("transformed phase-I constraints returned nonfinite data")
        self._q = q.copy()
        self._result = (c_out, jc_out, h_out, jh_out)
        return self._result

    def inequality_values(self, augmented: Sequence[float]) -> Array:
        return self.evaluate(augmented)[0]

    def inequality_jacobian(self, augmented: Sequence[float]) -> Array:
        return self.evaluate(augmented)[1]

    def equality_values(self, augmented: Sequence[float]) -> Array:
        return self.evaluate(augmented)[2]

    def equality_jacobian(self, augmented: Sequence[float]) -> Array:
        return self.evaluate(augmented)[3]


def _finite_result_to_physical(
    finite: FiniteSolveResult,
    objective: _LogLengthObjective,
) -> FiniteSolveResult:
    x, value, physical_gradient = objective.physical_evaluation(finite.x)
    solver_result = replace(
        finite.solver_result,
        x=x.copy(),
        objective=value,
        gradient=physical_gradient.copy(),
    )
    # The retained affine scaling belongs to internal log-length coordinates;
    # exposing it beside physical cumulative stations would be misleading.
    return replace(
        finite,
        x=x,
        objective=value,
        solver_result=solver_result,
        scaling=None,
    )


def _build_variable_scaling(
    x0: Array,
    bounds: BoxBounds | None,
    settings: ScalingSettings,
) -> tuple[Array, BoxBounds | None]:
    if not settings.enabled:
        scale = np.ones_like(x0)
    elif settings.variable_scales is not None:
        scale = np.asarray(settings.variable_scales, dtype=float)
        if scale.shape != x0.shape:
            raise ValueError("variable_scales has wrong length")
    elif not settings.automatic_variable_scaling:
        scale = np.ones_like(x0)
    else:
        scale = np.maximum(np.abs(x0), settings.minimum_variable_scale)
        if bounds is not None:
            lb = np.asarray(bounds.lower, dtype=float)
            ub = np.asarray(bounds.upper, dtype=float)
            finite_both = np.isfinite(lb) & np.isfinite(ub)
            # Because the affine shift is x0, a two-sided variable should be
            # scaled by its available movement, not by its absolute coordinate.
            scale[finite_both] = np.maximum(
                0.5 * (ub[finite_both] - lb[finite_both]),
                settings.minimum_variable_scale,
            )
            finite_lo = np.isfinite(lb) & ~np.isfinite(ub)
            finite_hi = ~np.isfinite(lb) & np.isfinite(ub)
            scale[finite_lo] = np.maximum.reduce(
                (
                    np.abs(x0[finite_lo]),
                    np.abs(x0[finite_lo] - lb[finite_lo]),
                    np.full(np.count_nonzero(finite_lo), settings.minimum_variable_scale),
                )
            )
            scale[finite_hi] = np.maximum.reduce(
                (
                    np.abs(x0[finite_hi]),
                    np.abs(ub[finite_hi] - x0[finite_hi]),
                    np.full(np.count_nonzero(finite_hi), settings.minimum_variable_scale),
                )
            )
        scale = np.maximum(scale, settings.minimum_variable_scale)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("computed variable scales must be finite and positive")
    if bounds is None:
        return scale, None
    return scale, BoxBounds(
        (np.asarray(bounds.lower, dtype=float) - x0) / scale,
        (np.asarray(bounds.upper, dtype=float) - x0) / scale,
    )


def _row_scales(
    values: Array,
    jacobian: Array,
    variable_scale: Array,
    settings: ScalingSettings,
) -> Array:
    if values.size == 0:
        return np.empty(0, dtype=float)
    if not settings.enabled or not settings.scale_constraints:
        return np.ones(values.size, dtype=float)
    scaled_jacobian = jacobian * variable_scale[np.newaxis, :]
    row_norm = np.max(np.abs(scaled_jacobian), axis=1, initial=0.0)
    return np.maximum.reduce(
        (
            np.full(values.size, settings.minimum_constraint_scale),
            np.abs(values),
            row_norm,
        )
    )


@dataclass(slots=True)
class _ScaledObjective:
    cached: CachedObjective
    scaling: SolveScaling

    def evaluate(self, z: Sequence[float]) -> tuple[float, Array]:
        value, gradient = self.cached.evaluate(self.scaling.to_physical(z))
        return value, gradient * self.scaling.variable_scale

    def fun(self, z: Sequence[float]) -> float:
        return self.cached.fun(self.scaling.to_physical(z))

    def jac(self, z: Sequence[float]) -> Array:
        return self.cached.jac(self.scaling.to_physical(z)) * self.scaling.variable_scale

    def evaluate_with_deadline(
        self, z: Sequence[float], *, deadline: float
    ) -> tuple[float, Array]:
        value, gradient = self.cached.evaluate_with_deadline(
            self.scaling.to_physical(z), deadline=deadline
        )
        return value, gradient * self.scaling.variable_scale


@dataclass(slots=True)
class _ScaledConstraints:
    original: CombinedConstraintOracle
    scaling: SolveScaling
    _z: Array | None = field(default=None, init=False, repr=False)
    _result: tuple[Array, Array, Array, Array] | None = field(default=None, init=False, repr=False)
    _sparse_result: tuple[Array, object, Array, object] | None = field(
        default=None, init=False, repr=False
    )

    def evaluate(self, z: Sequence[float]) -> tuple[Array, Array, Array, Array]:
        array = np.asarray(z, dtype=float)
        if self._z is not None and np.array_equal(array, self._z):
            if self._result is None:
                assert self._sparse_result is not None
                c, jc, h, jh = self._sparse_result
                self._result = (c, jc.toarray(), h, jh.toarray())
            return self._result
        c, jc, h, jh = self.original.evaluate(self.scaling.to_physical(array))
        c_scaled = c / self.scaling.inequality_scale
        jc_scaled = (
            jc * self.scaling.variable_scale[np.newaxis, :]
        ) / self.scaling.inequality_scale[:, np.newaxis]
        h_scaled = h / self.scaling.equality_scale
        jh_scaled = (
            jh * self.scaling.variable_scale[np.newaxis, :]
        ) / self.scaling.equality_scale[:, np.newaxis]
        self._z = array.copy()
        self._result = (c_scaled, jc_scaled, h_scaled, jh_scaled)
        self._sparse_result = None
        return self._result

    def evaluate_sparse(self, z: Sequence[float]):
        from scipy.sparse import csr_matrix

        array = np.asarray(z, dtype=float)
        if self._z is not None and np.array_equal(array, self._z):
            if self._sparse_result is not None:
                return self._sparse_result
            assert self._result is not None
            c, jc, h, jh = self._result
            self._sparse_result = (c, csr_matrix(jc), h, csr_matrix(jh))
            return self._sparse_result
        physical = self.scaling.to_physical(array)
        if hasattr(self.original, "evaluate_sparse"):
            c, jc, h, jh = self.original.evaluate_sparse(physical)
        else:
            c, jc, h, jh = self.original.evaluate(physical)
            jc, jh = csr_matrix(jc), csr_matrix(jh)
        c_scaled = c / self.scaling.inequality_scale
        jc_scaled = jc.multiply(self.scaling.variable_scale[np.newaxis, :])
        jc_scaled = jc_scaled.multiply(
            (1.0 / self.scaling.inequality_scale)[:, np.newaxis]
        ).tocsr()
        h_scaled = h / self.scaling.equality_scale
        jh_scaled = jh.multiply(self.scaling.variable_scale[np.newaxis, :])
        jh_scaled = jh_scaled.multiply(
            (1.0 / self.scaling.equality_scale)[:, np.newaxis]
        ).tocsr()
        self._z = array.copy()
        self._result = None
        self._sparse_result = (c_scaled, jc_scaled, h_scaled, jh_scaled)
        return self._sparse_result

    def inequality_values(self, z: Sequence[float]) -> Array:
        return self.evaluate(z)[0]

    def inequality_jacobian(self, z: Sequence[float]) -> Array:
        return self.evaluate(z)[1]

    def equality_values(self, z: Sequence[float]) -> Array:
        return self.evaluate(z)[2]

    def equality_jacobian(self, z: Sequence[float]) -> Array:
        return self.evaluate(z)[3]


def _estimate_scaled_kkt(
    x: Array,
    gradient: Array,
    c: Array,
    jc: Array,
    h: Array,
    jh: Array,
    scaling: SolveScaling,
    *,
    active_tolerance: float,
) -> KKTReport:
    z = scaling.to_scaled(x)
    inequality_active = c >= -active_tolerance
    lower_active: Array | None = None
    upper_active: Array | None = None
    if scaling.scaled_bounds is not None:
        scaled_lb = np.asarray(scaling.scaled_bounds.lower, dtype=float)
        scaled_ub = np.asarray(scaling.scaled_bounds.upper, dtype=float)
        physical_lb = scaling.shift + scaling.variable_scale * scaled_lb
        physical_ub = scaling.shift + scaling.variable_scale * scaled_ub
        lower_active = np.isfinite(physical_lb) & (
            x
            <= physical_lb
            + active_tolerance * np.maximum(1.0, np.abs(physical_lb))
        )
        upper_active = np.isfinite(physical_ub) & (
            x
            >= physical_ub
            - active_tolerance * np.maximum(1.0, np.abs(physical_ub))
        )
    return estimate_kkt_residual(
        z,
        gradient * scaling.variable_scale,
        c / scaling.inequality_scale,
        (jc * scaling.variable_scale[np.newaxis, :])
        / scaling.inequality_scale[:, np.newaxis],
        h / scaling.equality_scale,
        (jh * scaling.variable_scale[np.newaxis, :])
        / scaling.equality_scale[:, np.newaxis],
        scaling.scaled_bounds,
        active_tolerance=active_tolerance,
        active_inequality_mask=inequality_active,
        active_lower_bound_mask=lower_active,
        active_upper_bound_mask=upper_active,
    )


def _complete_kkt_diagnostic(
    finite: FiniteSolveResult,
    objective: ScalarObjective,
    constraints: CombinedConstraintOracle,
    *,
    active_tolerance: float,
) -> FiniteSolveResult:
    started = time.perf_counter()
    value, gradient_in = objective(finite.x)
    diagnostic_seconds = time.perf_counter() - started
    gradient = np.asarray(gradient_in, dtype=float)
    if gradient.shape != finite.x.shape or not np.all(np.isfinite(gradient)):
        raise ValueError("objective gradient has wrong shape or nonfinite values")
    if not math.isfinite(float(value)):
        raise FloatingPointError("objective returned a nonfinite value")
    c, jc, h, jh = constraints.evaluate(finite.x)
    if finite.scaling is None:
        raise RuntimeError("finite solve did not retain scaling metadata")
    kkt = _estimate_scaled_kkt(
        finite.x,
        gradient,
        c,
        jc,
        h,
        jh,
        finite.scaling,
        active_tolerance=active_tolerance,
    )
    return replace(
        finite,
        kkt=kkt,
        objective_calls=finite.objective_calls + 1,
        objective_seconds=finite.objective_seconds + diagnostic_seconds,
    )


def _solve_scaled_slsqp_backend(
    x0: Array,
    objective: _ScaledObjective,
    constraints: _ScaledConstraints,
    bounds: BoxBounds | None,
    *,
    has_equalities: bool,
    settings: SLSQPSettings,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[NonlinearSolveResult, float]:
    """Run SciPy SLSQP and immediately convert its result to neutral types."""
    from scipy.optimize import minimize

    backend_constraints: list[dict[str, object]] = [
        {
            "type": "ineq",
            "fun": lambda z: -constraints.inequality_values(z),
            "jac": lambda z: -constraints.inequality_jacobian(z),
        }
    ]
    if has_equalities:
        backend_constraints.append(
            {
                "type": "eq",
                "fun": constraints.equality_values,
                "jac": constraints.equality_jacobian,
            }
        )
    backend_bounds = (
        None
        if bounds is None
        else list(zip(bounds.lower.tolist(), bounds.upper.tolist(), strict=True))
    )

    iteration = 0

    def callback(z: Array) -> None:
        nonlocal iteration
        iteration += 1
        if progress_callback is not None:
            physical = objective.scaling.to_physical(z)
            event: dict[str, object] = {
                "event": "iteration",
                "iteration": iteration,
                "parameters": physical.copy(),
            }
            # SLSQP evaluates the accepted major iterate before invoking its
            # callback.  Reuse that already-computed objective when the cache
            # confirms an exact coordinate match; never trigger another
            # differentiable time evaluation just to publish progress.
            cached = objective.cached
            if (
                cached._x is not None
                and cached._value is not None
                and np.array_equal(physical, cached._x)
                and math.isfinite(float(cached._value))
            ):
                event["objective_value"] = float(cached._value)
            # Progress telemetry is also the only parent-owned work-count
            # evidence if the supervised worker exits before returning its
            # final SciPy result.  These are actual expensive scalar builds and
            # differentiable evaluations, not merely Python callback entries.
            event["objective_calls"] = int(cached.scalar_evaluations)
            event["gradient_calls"] = int(cached.differentiable_evaluations)
            progress_callback(event)

    started = time.perf_counter()
    raw = minimize(
        objective.fun,
        x0,
        method="SLSQP",
        jac=objective.jac,
        bounds=backend_bounds,
        constraints=backend_constraints,
        callback=callback if progress_callback is not None else None,
        options={
            "maxiter": settings.max_iterations,
            "ftol": settings.ftol,
            "disp": settings.display,
        },
    )
    elapsed = time.perf_counter() - started
    raw_gradient = getattr(raw, "jac", None)
    gradient = (
        None
        if raw_gradient is None
        else np.asarray(raw_gradient, dtype=float).copy()
    )
    result = NonlinearSolveResult(
        np.asarray(raw.x, dtype=float).copy(),
        float(getattr(raw, "fun", math.nan)),
        gradient,
        bool(raw.success),
        int(getattr(raw, "status", 0)),
        str(raw.message),
        int(getattr(raw, "nit", 0)),
        int(getattr(raw, "nfev", 0)),
        int(getattr(raw, "njev", 0)),
        "scipy-slsqp",
    )
    return result, elapsed



def _solve_scaled_sparse_sqp_backend(
    x0: Array,
    objective: _ScaledObjective,
    constraints: _ScaledConstraints,
    bounds: BoxBounds | None,
    *,
    settings: SparseSQPSettings,
    continuation_state: SparseSQPContinuationState | None = None,
    constraint_generation: int | None = None,
    objective_identity: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[NonlinearSolveResult, float]:
    """Run the experimental HiGHS filter-SQP backend."""
    started = time.perf_counter()
    raw = solve_sparse_filter_sqp(
        x0,
        objective,
        constraints,
        lower_bounds=None if bounds is None else bounds.lower,
        upper_bounds=None if bounds is None else bounds.upper,
        settings=settings,
        continuation_state=continuation_state,
        constraint_generation=constraint_generation,
        objective_identity=objective_identity,
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - started
    result = NonlinearSolveResult(
        raw.x.copy(),
        float(raw.objective),
        raw.gradient.copy(),
        bool(raw.success),
        int(raw.status),
        str(raw.message),
        int(raw.iterations),
        int(raw.objective_evaluations),
        int(raw.gradient_evaluations),
        "highs-filter-sqp",
        diagnostics=raw,
    )
    return result, elapsed


def solve_finite_sparse_sqp(
    x0: Sequence[float],
    objective: ScalarObjective,
    constraints: CombinedConstraintOracle,
    *,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    settings: SparseSQPSettings = SparseSQPSettings(),
    scaling_settings: ScalingSettings = ScalingSettings(),
    active_tolerance: float = 1.0e-7,
    compute_kkt: bool = True,
    continuation_state: SparseSQPContinuationState | None = None,
    constraint_generation: int | None = None,
    objective_identity: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> FiniteSolveResult:
    """Solve one generated finite NLP with the sparse filter-SQP prototype."""
    x0_array = np.asarray(x0, dtype=float)
    if x0_array.ndim != 1 or not np.all(np.isfinite(x0_array)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    normalized_bounds = _normalize_bounds(bounds, x0_array.size)
    reusable_scaling = bool(
        continuation_state is not None
        and continuation_state.scaling_shift is not None
        and continuation_state.scaling_variable is not None
        and continuation_state.scaling_lower is not None
        and continuation_state.scaling_upper is not None
    )
    if reusable_scaling:
        assert continuation_state is not None
        shift = continuation_state.scaling_shift.copy()
        variable_scale = continuation_state.scaling_variable.copy()
        scaled_bounds = BoxBounds(
            continuation_state.scaling_lower.copy(),
            continuation_state.scaling_upper.copy(),
        )
        z0 = continuation_state.current_x.copy()
        scaling_probe = shift + variable_scale * z0
        c0, jc0, h0, jh0 = constraints.evaluate(scaling_probe)
        same_generation = (
            constraint_generation is None
            or constraint_generation == continuation_state.constraint_generation
        )
        if (
            same_generation
            and continuation_state.scaling_inequality is not None
            and continuation_state.scaling_inequality.size == c0.size
        ):
            inequality_scale = continuation_state.scaling_inequality.copy()
        else:
            inequality_scale = _row_scales(
                c0, jc0, variable_scale, scaling_settings
            )
        if (
            same_generation
            and continuation_state.scaling_equality is not None
            and continuation_state.scaling_equality.size == h0.size
        ):
            equality_scale = continuation_state.scaling_equality.copy()
        else:
            equality_scale = _row_scales(
                h0, jh0, variable_scale, scaling_settings
            )
        scaling = SolveScaling(
            shift,
            variable_scale,
            inequality_scale,
            equality_scale,
            scaled_bounds,
        )
    else:
        variable_scale, scaled_bounds = _build_variable_scaling(
            x0_array, normalized_bounds, scaling_settings
        )
        c0, jc0, h0, jh0 = constraints.evaluate(x0_array)
        inequality_scale = _row_scales(c0, jc0, variable_scale, scaling_settings)
        equality_scale = _row_scales(h0, jh0, variable_scale, scaling_settings)
        scaling = SolveScaling(
            x0_array.copy(),
            variable_scale.copy(),
            inequality_scale,
            equality_scale,
            scaled_bounds,
        )
        z0 = np.zeros_like(x0_array)
    cached_objective = CachedObjective(objective)
    scaled_objective = _ScaledObjective(cached_objective, scaling)
    scaled_constraints = _ScaledConstraints(constraints, scaling)

    def scaled_progress(event: dict[str, object]) -> None:
        if progress_callback is None:
            return
        forwarded = dict(event)
        parameters = forwarded.get("parameters")
        if parameters is not None:
            forwarded["parameters"] = scaling.to_physical(
                np.asarray(parameters, dtype=float)
            )
        progress_callback(forwarded)

    scaled_solver_result, optimizer_seconds = _solve_scaled_sparse_sqp_backend(
        z0,
        scaled_objective,
        scaled_constraints,
        scaled_bounds,
        settings=settings,
        continuation_state=continuation_state,
        constraint_generation=constraint_generation,
        objective_identity=objective_identity,
        progress_callback=(scaled_progress if progress_callback is not None else None),
    )
    raw_diagnostics = scaled_solver_result.diagnostics
    if not isinstance(raw_diagnostics, SparseSQPResult):
        raise RuntimeError("sparse SQP backend did not return diagnostics")
    state_with_scaling = replace(
        raw_diagnostics.continuation_state,
        scaling_shift=scaling.shift.copy(),
        scaling_variable=scaling.variable_scale.copy(),
        scaling_inequality=scaling.inequality_scale.copy(),
        scaling_equality=scaling.equality_scale.copy(),
        scaling_lower=(
            np.full(x0_array.size, -math.inf)
            if scaling.scaled_bounds is None
            else np.asarray(scaling.scaled_bounds.lower, dtype=float).copy()
        ),
        scaling_upper=(
            np.full(x0_array.size, math.inf)
            if scaling.scaled_bounds is None
            else np.asarray(scaling.scaled_bounds.upper, dtype=float).copy()
        ),
    )
    raw_diagnostics = replace(
        raw_diagnostics, continuation_state=state_with_scaling
    )
    scaled_solver_result = replace(
        scaled_solver_result, diagnostics=raw_diagnostics
    )
    z = scaled_solver_result.x
    x = scaling.to_physical(z)
    value = float(scaled_solver_result.objective)
    if scaled_solver_result.gradient is None:
        raise RuntimeError("sparse SQP result did not retain its objective gradient")
    gradient = (
        np.asarray(scaled_solver_result.gradient, dtype=float)
        / scaling.variable_scale
    )
    c, jc, h, jh = constraints.evaluate(x)
    max_c = float(max(0.0, np.max(c, initial=-math.inf)))
    eq_inf = float(np.linalg.norm(h, ord=np.inf)) if h.size else 0.0
    kkt = (
        _estimate_scaled_kkt(
            x,
            gradient,
            c,
            jc,
            h,
            jh,
            scaling,
            active_tolerance=active_tolerance,
        )
        if compute_kkt
        else KKTReport.not_computed()
    )

    scaled_gradient = scaled_solver_result.gradient
    physical_gradient = (
        None
        if scaled_gradient is None
        else scaled_gradient / scaling.variable_scale
    )
    solver_result = replace(
        scaled_solver_result,
        x=x.copy(),
        objective=value,
        gradient=None if physical_gradient is None else physical_gradient.copy(),
        scaled_x=z.copy(),
        scaled_gradient=(
            None if scaled_gradient is None else scaled_gradient.copy()
        ),
    )

    return FiniteSolveResult(
        x,
        value,
        solver_result.success,
        solver_result.message,
        solver_result.iterations,
        max_c,
        eq_inf,
        kkt,
        cached_objective.calls,
        cached_objective.failures,
        cached_objective.seconds,
        constraints.rectangle.calls,
        constraints.rectangle.seconds,
        optimizer_seconds,
        solver_result,
        scaling,
        cached_objective.call_statistics(),
    )


def solve_finite_slsqp(
    x0: Sequence[float],
    objective: ScalarObjective,
    constraints: CombinedConstraintOracle,
    *,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    settings: SLSQPSettings = SLSQPSettings(),
    active_tolerance: float = 1.0e-7,
    compute_kkt: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> FiniteSolveResult:
    x0_array = np.asarray(x0, dtype=float)
    if x0_array.ndim != 1 or not np.all(np.isfinite(x0_array)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    normalized_bounds = _normalize_bounds(bounds, x0_array.size)
    variable_scale, scaled_bounds = _build_variable_scaling(
        x0_array, normalized_bounds, settings.scaling
    )

    c0, jc0, h0, jh0 = constraints.evaluate(x0_array)
    inequality_scale = _row_scales(c0, jc0, variable_scale, settings.scaling)
    equality_scale = _row_scales(h0, jh0, variable_scale, settings.scaling)
    scaling = SolveScaling(
        x0_array.copy(),
        variable_scale.copy(),
        inequality_scale,
        equality_scale,
        scaled_bounds,
    )
    cached_objective = CachedObjective(objective)
    scaled_objective = _ScaledObjective(cached_objective, scaling)
    scaled_constraints = _ScaledConstraints(constraints, scaling)
    z0 = np.zeros_like(x0_array)

    scaled_solver_result, optimizer_seconds = _solve_scaled_slsqp_backend(
        z0,
        scaled_objective,
        scaled_constraints,
        scaled_bounds,
        has_equalities=bool(h0.size),
        settings=settings,
        progress_callback=progress_callback,
    )
    z = scaled_solver_result.x
    x = scaling.to_physical(z)
    value, gradient = cached_objective.evaluate(x)
    c, jc, h, jh = constraints.evaluate(x)
    max_c = float(max(0.0, np.max(c, initial=-math.inf)))
    eq_inf = float(np.linalg.norm(h, ord=np.inf)) if h.size else 0.0
    kkt = (
        _estimate_scaled_kkt(
            x,
            gradient,
            c,
            jc,
            h,
            jh,
            scaling,
            active_tolerance=active_tolerance,
        )
        if compute_kkt
        else KKTReport.not_computed()
    )

    scaled_gradient = scaled_solver_result.gradient
    physical_gradient = (
        None
        if scaled_gradient is None
        else scaled_gradient / scaling.variable_scale
    )
    solver_result = replace(
        scaled_solver_result,
        x=x.copy(),
        objective=value,
        gradient=None if physical_gradient is None else physical_gradient.copy(),
        scaled_x=z.copy(),
        scaled_gradient=(
            None if scaled_gradient is None else scaled_gradient.copy()
        ),
    )

    return FiniteSolveResult(
        x,
        value,
        solver_result.success,
        solver_result.message,
        solver_result.iterations,
        max_c,
        eq_inf,
        kkt,
        cached_objective.calls,
        cached_objective.failures,
        cached_objective.seconds,
        constraints.rectangle.calls,
        constraints.rectangle.seconds,
        optimizer_seconds,
        solver_result,
        scaling,
        cached_objective.call_statistics(),
    )




def stabilize_phase_one_sparse_exchange(
    settings: ExchangeSettings,
) -> ExchangeSettings:
    """Return route-handoff settings with deterministic identity row scaling.

    Frozen constraint scales computed at a Phase-I certificate are not physical
    state.  Route-2 controls show that perturbations near ``1e-10`` alter those
    divisors enough to select different downstream sparse trajectories.  The
    route-oriented sparse handoff therefore uses identity row scaling while
    retaining variable scaling and all strict physical tolerances.

    Generic finite-NLP campaigns are intentionally unaffected unless their
    caller opts into this handoff policy.
    """
    if settings.finite_solver != "sparse_sqp":
        raise ValueError("stable Phase-I handoff requires finite_solver='sparse_sqp'")
    scaling = replace(settings.slsqp.scaling, scale_constraints=False)
    return replace(settings, slsqp=replace(settings.slsqp, scaling=scaling))


def solve_finite_nlp(
    x0: Sequence[float],
    objective: ScalarObjective,
    constraints: CombinedConstraintOracle,
    *,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None,
    settings: ExchangeSettings,
    active_tolerance: float,
    compute_kkt: bool,
    continuation_state: SparseSQPContinuationState | None = None,
    constraint_generation: int | None = None,
    objective_identity: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> FiniteSolveResult:
    """Dispatch one generated finite problem to the selected backend."""
    if settings.finite_solver == "slsqp":
        if continuation_state is not None:
            raise ValueError("SLSQP does not consume sparse continuation state")
        return solve_finite_slsqp(
            x0,
            objective,
            constraints,
            bounds=bounds,
            settings=settings.slsqp,
            active_tolerance=active_tolerance,
            compute_kkt=compute_kkt,
            progress_callback=progress_callback,
        )
    return solve_finite_sparse_sqp(
        x0,
        objective,
        constraints,
        bounds=bounds,
        settings=settings.sparse_sqp,
        scaling_settings=settings.slsqp.scaling,
        active_tolerance=active_tolerance,
        compute_kkt=compute_kkt,
        continuation_state=continuation_state,
        constraint_generation=constraint_generation,
        objective_identity=objective_identity,
        progress_callback=progress_callback,
    )

def run_constraint_generation(
    x0: Sequence[float],
    objective: ScalarObjective,
    corridor: CorridorModel,
    initial_state: GeometryState | Sequence[float],
    *,
    initial_s: float = 0.0,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    equalities: VectorConstraint | None = None,
    endpoint_target: tuple[float | None, float | None, float | None, float | None] | None = None,
    additional_inequalities: VectorConstraint | None = None,
    pool: ConstraintPool | None = None,
    settings: ExchangeSettings = ExchangeSettings(),
    coordinate_map: LogLengthKnotMap | None = None,
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
    continuation_state: SparseSQPContinuationState | None = None,
    constraint_generation: int = 0,
    objective_identity: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> ExchangeResult:
    """Solve a rectangular corridor problem by finite NLP / exact exchange.

    The objective and supplied constraints always use the public cumulative-
    station knot basis.  When ``coordinate_map`` is supplied, each finite NLP
    is solved in positive log-length coordinates and mapped back before exact
    continuous separation.
    """
    started = time.perf_counter()
    x = np.asarray(x0, dtype=float).copy()
    if coordinate_map is not None:
        if coordinate_map.width != x.size:
            raise ValueError("coordinate map width does not match x0")
        # Validate the current path against the fixed reference map.
        coordinate_map.to_optimizer(x)
        physical_bounds, internal_bounds = _prepare_log_length_bounds(
            bounds, x.size, coordinate_settings
        )
    else:
        physical_bounds = None
        internal_bounds = None

    cache = KnotGeometryCache(initial_state, initial_s)
    cache.update(x)
    endpoint_equality = (
        EndpointPoseEquality(cache, endpoint_target)
        if endpoint_target is not None
        else None
    )
    combined_equalities = stack_vector_constraints(endpoint_equality, equalities)
    if cache.path.n_segments != corridor.n_segments:
        raise ValueError("corridor assignment length does not match initial path")
    active_pool = pool.copy() if pool is not None else ConstraintPool.seeded(corridor)
    if active_pool.corridor != corridor:
        raise ValueError("provided pool belongs to a different corridor")

    rounds: list[ExchangeRoundResult] = []
    message = "maximum exchange rounds reached"
    success = False
    final_finite: FiniteSolveResult | None = None
    final_report: SeparationReport | None = None
    certified_polish_rounds_used = 0
    active_continuation = continuation_state
    active_generation = int(constraint_generation)
    last_current_x: Array | None = None
    last_current_objective = math.inf

    for round_index in range(settings.maximum_rounds):
        before = active_pool.size

        def finite_progress(event: dict[str, object]) -> None:
            if progress_callback is None:
                return
            forwarded = dict(event)
            forwarded["exchange_round"] = round_index
            parameters = forwarded.get("parameters")
            if parameters is not None and coordinate_map is not None:
                forwarded["parameters"] = coordinate_map.to_physical(
                    np.asarray(parameters, dtype=float)
                )
            progress_callback(forwarded)
        rectangle = RectangleConstraintOracle(corridor, active_pool, cache)
        combined = CombinedConstraintOracle(
            rectangle, combined_equalities, additional_inequalities
        )

        finite_internal: FiniteSolveResult | None = None
        transformed_objective: _LogLengthObjective | None = None
        transformed_constraints: _LogLengthConstraints | None = None
        if coordinate_map is None:
            finite = solve_finite_nlp(
                x,
                objective,
                combined,
                bounds=bounds,
                settings=settings,
                active_tolerance=settings.active_constraint_tolerance,
                compute_kkt=False,
                continuation_state=active_continuation,
                constraint_generation=active_generation,
                objective_identity=objective_identity,
                progress_callback=finite_progress,
            )
            x = finite.x
        else:
            transformed_objective = _LogLengthObjective(objective, coordinate_map)
            transformed_constraints = _LogLengthConstraints(
                combined, coordinate_map, physical_bounds
            )
            u0 = coordinate_map.to_optimizer(x)
            finite_internal = solve_finite_nlp(
                u0,
                transformed_objective,
                transformed_constraints,  # type: ignore[arg-type]
                bounds=internal_bounds,
                settings=settings,
                active_tolerance=settings.active_constraint_tolerance,
                compute_kkt=False,
                continuation_state=active_continuation,
                constraint_generation=active_generation,
                objective_identity=objective_identity,
                progress_callback=finite_progress,
            )
            finite = _finite_result_to_physical(
                finite_internal, transformed_objective
            )
            x = finite.x

        diagnostic_finite = finite_internal if finite_internal is not None else finite
        raw_solver = diagnostic_finite.solver_result.diagnostics
        if isinstance(raw_solver, SparseSQPResult):
            active_continuation = raw_solver.continuation_state
            state_scaling = active_continuation
            if (
                state_scaling.scaling_shift is None
                or state_scaling.scaling_variable is None
            ):
                raise RuntimeError("continuation checkpoint omitted scaling metadata")
            current_internal = (
                state_scaling.scaling_shift
                + state_scaling.scaling_variable * state_scaling.current_x
            )
            last_current_x = (
                current_internal.copy()
                if coordinate_map is None
                else coordinate_map.to_physical(current_internal)
            )
            last_current_objective = float(state_scaling.current_objective)
        else:
            active_continuation = None
            last_current_x = finite.x.copy()
            last_current_objective = float(finite.objective)

        cache.update(x)
        separation_started = time.perf_counter()
        report = separate_compiled_path(
            cache.path,
            corridor,
            settings=settings.separation,
        )
        separation_seconds = time.perf_counter() - separation_started
        added = add_report_violations(
            active_pool,
            report,
            cache.path.lengths,
            merge_distance=settings.merge_distance,
        )
        after = active_pool.size
        if added:
            active_generation += 1

        primal_ok = (
            finite.max_inequality <= settings.finite_inequality_tolerance
            and finite.equality_inf <= settings.finite_equality_tolerance
        )
        solver_ok = finite.success or not settings.require_solver_success
        certified = report.certified(settings.separation.certificate_tolerance)
        # KKT estimation solves an auxiliary bounded least-squares problem.  It
        # is useful only for a primal-feasible, continuously certified final
        # candidate, so all cut-generating rounds skip it.
        if (
            settings.final_kkt_diagnostics
            and added == 0
            and certified
            and primal_ok
            and solver_ok
        ):
            if coordinate_map is None:
                finite = _complete_kkt_diagnostic(
                    finite,
                    objective,
                    combined,
                    active_tolerance=settings.active_constraint_tolerance,
                )
            else:
                assert finite_internal is not None
                assert transformed_objective is not None
                assert transformed_constraints is not None
                finite_internal = _complete_kkt_diagnostic(
                    finite_internal,
                    transformed_objective,
                    transformed_constraints,  # type: ignore[arg-type]
                    active_tolerance=settings.active_constraint_tolerance,
                )
                finite = _finite_result_to_physical(
                    finite_internal, transformed_objective
                )

        rounds.append(
            ExchangeRoundResult(
                round_index,
                before,
                after,
                added,
                report.worst_upper_bound,
                report.worst_exact_violation,
                finite,
            )
        )
        final_finite = finite
        final_report = report

        if added == 0:
            kkt_ok = (
                finite.kkt.computed
                and finite.kkt.stationarity_inf <= settings.kkt_tolerance
            )
            if certified and primal_ok and solver_ok and kkt_ok:
                success = True
                message = "continuously certified"
            elif finite.max_inequality > settings.finite_inequality_tolerance:
                message = "finite NLP left an existing generated constraint violated"
            elif finite.equality_inf > settings.finite_equality_tolerance:
                message = "finite NLP left an equality constraint violated"
            elif not certified:
                message = "continuous violation could not be added (deduplication/tolerance conflict)"
            elif not solver_ok:
                message = f"finite solver did not report success: {finite.message}"
            else:
                message = "continuous geometry certified but the scaled KKT tolerance was not met"
                if (
                    settings.finite_solver == "sparse_sqp"
                    and certified_polish_rounds_used
                    < settings.certified_polish_rounds
                    and round_index + 1 < settings.maximum_rounds
                ):
                    certified_polish_rounds_used += 1
                    continue
            break

    if final_finite is None or final_report is None:
        raise AssertionError("exchange loop did not execute")
    return ExchangeResult(
        x,
        final_finite.objective,
        success,
        message,
        active_pool,
        tuple(rounds),
        final_report,
        final_finite.max_inequality,
        final_finite.equality_inf,
        final_finite.kkt,
        time.perf_counter() - started,
        None if last_current_x is None else last_current_x.copy(),
        float(last_current_objective),
        active_continuation,
        active_generation,
    )


# =============================================================================
# Elastic Phase-I geometry feasibility
# =============================================================================


@dataclass(frozen=True, slots=True)
class PhaseOneSettings:
    maximum_rounds: int = 12
    regularization: float = 1.0e-6
    slack_tolerance: float = 1.0e-9
    finite_inequality_tolerance: float = 1.0e-9
    finite_equality_tolerance: float = 1.0e-9
    merge_distance: float = 1.0e-8
    require_solver_success: bool = True
    adaptive_recenter_enabled: bool = False
    adaptive_recenter_work_threshold: int = 150_000
    adaptive_recenter_cut_aware: bool = False
    adaptive_recenter_cut_threshold: int = 4
    adaptive_recenter_maximum_rounds: int = 2
    maximum_wall_seconds: float | None = None
    certification_retry_enabled: bool = True
    certification_retry_maximum_attempts: int = 1
    certification_retry_violation_cap: float = 1.5e-4
    separation: SeparationSettings = SeparationSettings()
    slsqp: SLSQPSettings = SLSQPSettings(max_iterations=500, ftol=1.0e-14)

    def __post_init__(self) -> None:
        if self.maximum_rounds <= 0:
            raise ValueError("maximum_rounds must be positive")
        if self.adaptive_recenter_work_threshold <= 0:
            raise ValueError("adaptive_recenter_work_threshold must be positive")
        if self.adaptive_recenter_cut_threshold < 0:
            raise ValueError("adaptive_recenter_cut_threshold must be nonnegative")
        if self.adaptive_recenter_maximum_rounds <= 0:
            raise ValueError("adaptive_recenter_maximum_rounds must be positive")
        if self.certification_retry_maximum_attempts < 0:
            raise ValueError(
                "certification_retry_maximum_attempts must be nonnegative"
            )
        if (
            not math.isfinite(self.certification_retry_violation_cap)
            or self.certification_retry_violation_cap < 0.0
        ):
            raise ValueError(
                "certification_retry_violation_cap must be finite and nonnegative"
            )
        if self.maximum_wall_seconds is not None and (
            not math.isfinite(self.maximum_wall_seconds)
            or self.maximum_wall_seconds <= 0.0
        ):
            raise ValueError(
                "maximum_wall_seconds must be finite and positive when supplied"
            )
        values = (
            self.regularization,
            self.slack_tolerance,
            self.finite_inequality_tolerance,
            self.finite_equality_tolerance,
            self.merge_distance,
        )
        if not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("phase-I tolerances and regularization must be finite and nonnegative")


def _phase_one_adaptive_recenter_active(
    settings: PhaseOneSettings,
    *,
    pool_size: int,
    n_variables: int,
) -> bool:
    return bool(
        settings.adaptive_recenter_enabled
        and pool_size * n_variables >= settings.adaptive_recenter_work_threshold
    )


def _phase_one_recenter_decision(
    settings: PhaseOneSettings,
    *,
    pool_size: int,
    n_variables: int,
    round_index: int,
    rounds_since_reference: int,
    previous_cuts_added: int | None,
) -> tuple[bool, str]:
    if round_index <= 0:
        return False, "initial_reference"
    if not _phase_one_adaptive_recenter_active(
        settings, pool_size=pool_size, n_variables=n_variables
    ):
        return False, "inactive"
    if not settings.adaptive_recenter_cut_aware:
        return True, "every_round"
    if rounds_since_reference >= settings.adaptive_recenter_maximum_rounds:
        return True, "maximum_reference_rounds"
    if (
        rounds_since_reference >= 1
        and previous_cuts_added is not None
        and previous_cuts_added <= settings.adaptive_recenter_cut_threshold
    ):
        return True, "low_cut_activity"
    return False, "retain_reference"


@dataclass(frozen=True, slots=True)
class PhaseOneResult:
    x: Array
    slack: float
    success: bool
    message: str
    pool: ConstraintPool
    exchange_rounds: int
    final_separation: SeparationReport
    certification_retries: int = 0
    final_finite_inequality: float = math.inf
    final_equality_inf: float = math.inf
    final_solver_success: bool = False
    final_solver_message: str = "not run"


@dataclass(slots=True)
class _PhaseConstraintOracle:
    """Cached augmented ``(x, rho)`` callbacks for one Phase-I finite solve."""

    rectangle: RectangleConstraintOracle
    equalities: VectorConstraint | None
    additional_inequalities: VectorConstraint | None
    n_variables: int
    _y: Array | None = field(default=None, init=False, repr=False)
    _result: tuple[Array, Array, Array, Array] | None = field(default=None, init=False, repr=False)

    def evaluate(self, y: Sequence[float]) -> tuple[Array, Array, Array, Array]:
        array = np.asarray(y, dtype=float)
        if array.shape != (self.n_variables + 1,):
            raise ValueError("phase-I vector has wrong shape")
        if self._y is not None and np.array_equal(array, self._y):
            assert self._result is not None
            return self._result

        c, jc = self.rectangle.evaluate(array[: self.n_variables])
        c_out = c - array[self.n_variables]
        jc_out = np.empty((c.size, self.n_variables + 1), dtype=float)
        jc_out[:, : self.n_variables] = jc
        jc_out[:, self.n_variables] = -1.0

        if self.additional_inequalities is not None:
            extra_in, extra_j_in = self.additional_inequalities(
                array[: self.n_variables]
            )
            extra_c = np.asarray(extra_in, dtype=float)
            extra_j = np.asarray(extra_j_in, dtype=float)
            if extra_j.shape != (extra_c.size, self.n_variables):
                raise ValueError(
                    "phase-I additional-inequality Jacobian has wrong shape"
                )
            if extra_c.size:
                extra_augmented = np.zeros(
                    (extra_c.size, self.n_variables + 1), dtype=float
                )
                extra_augmented[:, : self.n_variables] = extra_j
                c_out = np.concatenate((c_out, extra_c))
                jc_out = np.vstack((jc_out, extra_augmented))

        if self.equalities is None:
            h_out = np.empty(0, dtype=float)
            jh_out = np.empty((0, self.n_variables + 1), dtype=float)
        else:
            h_in, jh_in = self.equalities(array[: self.n_variables])
            h_out = np.asarray(h_in, dtype=float)
            jh = np.asarray(jh_in, dtype=float)
            if jh.shape != (h_out.size, self.n_variables):
                raise ValueError("phase-I equality Jacobian has wrong shape")
            jh_out = np.zeros((h_out.size, self.n_variables + 1), dtype=float)
            jh_out[:, : self.n_variables] = jh

        if not all(np.all(np.isfinite(v)) for v in (c_out, jc_out, h_out, jh_out)):
            raise FloatingPointError("phase-I callbacks returned nonfinite data")
        self._y = array.copy()
        self._result = (c_out, jc_out, h_out, jh_out)
        return self._result

    def inequality_values(self, y: Sequence[float]) -> Array:
        return self.evaluate(y)[0]

    def inequality_jacobian(self, y: Sequence[float]) -> Array:
        return self.evaluate(y)[1]

    def equality_values(self, y: Sequence[float]) -> Array:
        return self.evaluate(y)[2]

    def equality_jacobian(self, y: Sequence[float]) -> Array:
        return self.evaluate(y)[3]


def run_phase_one(
    x0: Sequence[float],
    corridor: CorridorModel,
    initial_state: GeometryState | Sequence[float],
    *,
    initial_s: float = 0.0,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    equalities: VectorConstraint | None = None,
    endpoint_target: tuple[float | None, float | None, float | None, float | None] | None = None,
    additional_inequalities: VectorConstraint | None = None,
    pool: ConstraintPool | None = None,
    settings: PhaseOneSettings = PhaseOneSettings(),
    coordinate_map: LogLengthKnotMap | None = None,
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    round_callback: Callable[[dict[str, object]], None] | None = None,
    checkpoint_callback: Callable[[PhaseOneResult], None] | None = None,
) -> PhaseOneResult:
    """Find a continuously corridor-feasible path with a global elastic slack."""
    x_reference = np.asarray(x0, dtype=float).copy()
    x = x_reference.copy()
    if coordinate_map is not None:
        if coordinate_map.width != x.size:
            raise ValueError("coordinate map width does not match x0")
        u_reference = coordinate_map.to_optimizer(x_reference)
        physical_bounds, internal_bounds = _prepare_log_length_bounds(
            bounds, x.size, coordinate_settings
        )
    else:
        u_reference = None
        physical_bounds = None
        internal_bounds = None

    cache = KnotGeometryCache(initial_state, initial_s)
    cache.update(x)
    endpoint_equality = (
        EndpointPoseEquality(cache, endpoint_target, periodic_heading=True)
        if endpoint_target is not None
        else None
    )
    combined_equalities = stack_vector_constraints(endpoint_equality, equalities)
    active_pool = pool.copy() if pool is not None else ConstraintPool.seeded(corridor)
    if active_pool.corridor != corridor:
        raise ValueError("provided pool belongs to a different corridor")
    rho = 0.0
    final_report: SeparationReport | None = None
    message = "maximum phase-I exchange rounds reached"
    completed_rounds = 0
    started = time.perf_counter()
    rounds_since_reference = 0
    previous_cuts_added: int | None = None
    certification_retries = 0
    final_finite_inequality = math.inf
    final_equality_inf = math.inf
    final_solver_success = False
    final_solver_message = "not run"

    for round_index in range(settings.maximum_rounds):
        if (
            round_index > 0
            and settings.maximum_wall_seconds is not None
            and time.perf_counter() - started >= settings.maximum_wall_seconds
        ):
            message = "phase-I cooperative wall-clock budget exhausted"
            break
        completed_rounds = round_index + 1
        structural_work = active_pool.size * x.size
        recentered_reference, recenter_reason = _phase_one_recenter_decision(
            settings,
            pool_size=active_pool.size,
            n_variables=x.size,
            round_index=round_index,
            rounds_since_reference=rounds_since_reference,
            previous_cuts_added=previous_cuts_added,
        )
        reference_rounds_before = rounds_since_reference
        if recentered_reference:
            x_reference = x.copy()
            rounds_since_reference = 0
            if coordinate_map is not None:
                u_reference = coordinate_map.to_optimizer(x_reference)
        rectangle = RectangleConstraintOracle(corridor, active_pool, cache)
        n = x.size

        # Make the warm start feasible for the current finite corridor cut pool,
        # including cuts introduced after the previous Phase-I solve.
        finite_values, _ = rectangle.evaluate(x)
        rho = max(rho, float(max(0.0, np.max(finite_values, initial=-math.inf))))

        if coordinate_map is None:
            def objective(y: Sequence[float]) -> tuple[float, Array]:
                array = np.asarray(y, dtype=float)
                dx = array[:n] - x_reference
                value = array[n] + 0.5 * settings.regularization * float(np.dot(dx, dx))
                grad = np.empty(n + 1, dtype=float)
                grad[:n] = settings.regularization * dx
                grad[n] = 1.0
                return value, grad

            phase_constraints = _PhaseConstraintOracle(
                rectangle, combined_equalities, additional_inequalities, n
            )
            y0 = np.concatenate((x, [rho]))
            normalized = _normalize_bounds(bounds, n)
            if normalized is None:
                phase_bounds: BoxBounds | None = BoxBounds(
                    np.concatenate((np.full(n, -math.inf), [0.0])),
                    np.full(n + 1, math.inf),
                )
            else:
                phase_bounds = BoxBounds(
                    np.concatenate((np.asarray(normalized.lower, dtype=float), [0.0])),
                    np.concatenate((np.asarray(normalized.upper, dtype=float), [math.inf])),
                )
        else:
            assert u_reference is not None and internal_bounds is not None
            u0 = coordinate_map.to_optimizer(x)

            def objective(y: Sequence[float]) -> tuple[float, Array]:
                array = np.asarray(y, dtype=float)
                du = array[:n] - u_reference
                value = array[n] + 0.5 * settings.regularization * float(np.dot(du, du))
                grad = np.empty(n + 1, dtype=float)
                grad[:n] = settings.regularization * du
                grad[n] = 1.0
                return value, grad

            phase_constraints = _LogLengthPhaseConstraints(
                rectangle,
                combined_equalities,
                additional_inequalities,
                coordinate_map,
                physical_bounds,
            )
            y0 = np.concatenate((u0, [rho]))
            phase_bounds = BoxBounds(
                np.concatenate((np.asarray(internal_bounds.lower, dtype=float), [0.0])),
                np.concatenate((np.asarray(internal_bounds.upper, dtype=float), [math.inf])),
            )

        phase_slsqp = settings.slsqp
        explicit_scales = phase_slsqp.scaling.variable_scales
        if explicit_scales is not None:
            if len(explicit_scales) == n:
                slack_scale = max(
                    1.0,
                    rho,
                    settings.slack_tolerance,
                    settings.separation.certificate_tolerance,
                )
                phase_scaling = replace(
                    phase_slsqp.scaling,
                    variable_scales=tuple(explicit_scales) + (slack_scale,),
                )
                phase_slsqp = replace(phase_slsqp, scaling=phase_scaling)
            elif len(explicit_scales) != n + 1:
                raise ValueError(
                    "phase-I variable_scales must have length n or n+1"
                )

        def phase_progress(event: dict[str, object]) -> None:
            if progress_callback is not None:
                progress_callback({"round": round_index + 1, **event})

        finite_started = time.perf_counter()
        finite = solve_finite_slsqp(
            y0,
            objective,
            phase_constraints,  # type: ignore[arg-type]
            bounds=phase_bounds,
            settings=phase_slsqp,
            compute_kkt=False,
            progress_callback=(phase_progress if progress_callback is not None else None),
        )
        finite_seconds = time.perf_counter() - finite_started
        if coordinate_map is None:
            x = finite.x[:n]
        else:
            x = coordinate_map.to_physical(finite.x[:n])
        rho = max(0.0, float(finite.x[n]))
        cache.update(x)
        separation_started = time.perf_counter()
        report = separate_compiled_path(
            cache.path,
            corridor,
            settings=settings.separation,
        )
        separation_seconds = time.perf_counter() - separation_started
        # In the elastic problem a cut is missing only when its violation
        # exceeds the optimized global slack by more than the cut tolerance.
        elastic_violations = tuple(
            m
            for m in report.violating_maxima
            if m.violation > rho + settings.separation.add_tolerance
        )
        elastic_report = SeparationReport(
            report.walls,
            elastic_violations,
            report.worst_upper_bound,
            report.worst_exact_violation,
            report.pruned_walls,
            report.exact_corner_families,
        )
        added = add_report_violations(
            active_pool,
            elastic_report,
            cache.path.lengths,
            merge_distance=settings.merge_distance,
        )
        finite_ok = finite.max_inequality <= settings.finite_inequality_tolerance
        equality_ok = finite.equality_inf <= settings.finite_equality_tolerance
        solver_ok = finite.success or not settings.require_solver_success
        certified = report.certified(settings.separation.certificate_tolerance)
        retry_violation_metric = max(
            0.0,
            rho,
            finite.max_inequality,
            finite.equality_inf,
            report.worst_upper_bound,
        )
        certification_retry_planned = bool(
            added == 0
            and settings.certification_retry_enabled
            and certification_retries
            < settings.certification_retry_maximum_attempts
            and round_index + 1 < settings.maximum_rounds
            and retry_violation_metric
            <= settings.certification_retry_violation_cap
            and not (
                rho <= settings.slack_tolerance
                and certified
                and finite_ok
                and equality_ok
                and solver_ok
            )
        )
        if round_callback is not None:
            round_callback(
                {
                    "round": round_index + 1,
                    "structural_work": structural_work,
                    "reference_recentered": recentered_reference,
                    "recenter_reason": recenter_reason,
                    "reference_rounds_before": reference_rounds_before,
                    "previous_cuts_added": previous_cuts_added,
                    "finite_seconds": finite_seconds,
                    "optimizer_seconds": finite.optimizer_seconds,
                    "objective_calls": finite.objective_calls,
                    "objective_seconds": finite.objective_seconds,
                    "constraint_calls": finite.constraint_calls,
                    "constraint_seconds": finite.constraint_seconds,
                    "iterations": finite.iterations,
                    "solver_success": finite.success,
                    "solver_message": finite.message,
                    "slack": rho,
                    "finite_inequality": finite.max_inequality,
                    "equality_inf": finite.equality_inf,
                    "separation_seconds": separation_seconds,
                    "worst_upper_bound": report.worst_upper_bound,
                    "cuts_added": added,
                    "pool_size": active_pool.size,
                    "certification_retry_planned": certification_retry_planned,
                    "certification_retry_index": certification_retries + 1,
                    "retry_violation_metric": retry_violation_metric,
                }
            )
        rounds_since_reference += 1
        previous_cuts_added = int(added)
        final_report = report
        final_finite_inequality = finite.max_inequality
        final_equality_inf = finite.equality_inf
        final_solver_success = finite.success
        final_solver_message = finite.message
        round_strictly_certified = bool(
            added == 0
            and rho <= settings.slack_tolerance
            and certified
            and finite_ok
            and equality_ok
            and solver_ok
        )
        if checkpoint_callback is not None:
            checkpoint_callback(
                PhaseOneResult(
                    x.copy(),
                    rho,
                    round_strictly_certified,
                    (
                        "continuously feasible"
                        if round_strictly_certified
                        else "phase-I round diagnostic checkpoint"
                    ),
                    active_pool.copy(),
                    round_index + 1,
                    report,
                    certification_retries,
                    finite.max_inequality,
                    finite.equality_inf,
                    finite.success,
                    finite.message,
                )
            )
        if added == 0:
            if (
                rho <= settings.slack_tolerance
                and certified
                and finite_ok
                and equality_ok
                and solver_ok
            ):
                return PhaseOneResult(
                    x,
                    rho,
                    True,
                    "continuously feasible",
                    active_pool,
                    round_index + 1,
                    report,
                    certification_retries,
                    finite.max_inequality,
                    finite.equality_inf,
                    finite.success,
                    finite.message,
                )
            if certification_retry_planned:
                certification_retries += 1
                # The current point is the least-violation checkpoint produced by
                # the bounded finite solve.  Recenter the proximity objective on
                # it before one additional strict-certification attempt.  This
                # preserves the feasible set and every authoritative tolerance;
                # only the local reference and SLSQP warm start change.
                x_reference = x.copy()
                rounds_since_reference = 0
                if coordinate_map is not None:
                    u_reference = coordinate_map.to_optimizer(x_reference)
                message = "phase-I certification retry from least-violation point"
                continue
            if not finite_ok:
                message = "phase-I finite NLP left an elastic or hard bound violated"
            elif not equality_ok:
                message = "phase-I finite NLP left an equality constraint violated"
            elif not solver_ok:
                message = f"phase-I solver did not report success: {finite.message}"
            elif not certified and rho <= settings.slack_tolerance:
                message = "continuous violation could not be added (deduplication/tolerance conflict)"
            else:
                message = "elastic problem converged but positive corridor slack remains"
            break

    if final_report is None:
        raise AssertionError("phase-I loop did not execute")
    return PhaseOneResult(
        x,
        rho,
        False,
        message,
        active_pool,
        completed_rounds,
        final_report,
        certification_retries,
        final_finite_inequality,
        final_equality_inf,
        final_solver_success,
        final_solver_message,
    )

# =============================================================================
# Final weighted path optimizer
# =============================================================================


def _checked_objective_weight(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class PathOptimizerSettings:
    """Settings for the final weighted path-optimization workflow.

    ``phase_one`` is optional.  When supplied, the optimizer first restores
    continuous corridor feasibility with the elastic Phase-I exchange before
    minimizing the requested weighted objective.  ``coordinates`` controls the
    internal knot-position representation; production defaults to positive
    log segment lengths while public inputs and outputs remain cumulative
    stations.
    """

    exchange: ExchangeSettings = ExchangeSettings()
    phase_one: PhaseOneSettings | None = None
    coordinates: KnotCoordinateSettings = KnotCoordinateSettings()
    n_scan: int = 256
    envelope_scan: int = 64
    domain_margin: float = reverse_solver.FRICTION_DOMAIN_MARGIN
    domain_scan: int = reverse_solver.FRICTION_DOMAIN_SCAN
    # Production GRIP scalar passes search the earliest switch/domain stop in
    # one traversal.  Disable only for differential regression/benchmarking.
    fused_grip_discovery: bool = True

    def __post_init__(self) -> None:
        if self.n_scan < 2:
            raise ValueError("n_scan must be at least two")
        if self.envelope_scan < 2:
            raise ValueError("envelope_scan must be at least two")
        if self.domain_scan < 2:
            raise ValueError("domain_scan must be at least two")
        if not math.isfinite(self.domain_margin) or self.domain_margin < 0.0:
            raise ValueError("domain_margin must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PathOptimizationResult:
    """Result of the final constrained weighted path optimization.

    ``parameters`` are the optimized flat knot parameters
    ``[s1, k1, ..., sn, kn]``.  ``time`` is always the exact reverse-solver
    travel time of the returned path, even when ``time_weight == 0``.

    ``objective`` is the weighted scalar minimized by SLSQP.  ``gradient`` and
    ``raw_gradient`` are its final knot- and raw-basis gradients.
    """

    time: float
    parameters: Array
    raw_parameters: tuple[float, ...]
    final_state: GeometryState
    gradient: Array
    raw_gradient: tuple[float, ...]
    success: bool
    message: str
    exchange: ExchangeResult | None
    phase_one: PhaseOneResult | None
    objective: float
    geometry_value: float
    curvature_value: float

    @property
    def knot_parameters(self) -> Array:
        """Alias for :attr:`parameters`."""
        return self.parameters


@dataclass(frozen=True, slots=True)
class _PathScalarEvaluation:
    value: float
    raw_parameters: tuple[float, ...]
    time_value: float
    geometry_value: float
    curvature_value: float
    geometry_path: GeometryPath | None
    speed_build: reverse_solver.ProdSpeedProfileBuild | reverse_solver.NativeProdSpeedProfileBuild | None


@dataclass(frozen=True, slots=True)
class _PathObjectiveEvaluation:
    value: float
    gradient: Array
    raw_parameters: tuple[float, ...]
    raw_gradient: tuple[float, ...]
    time_value: float
    geometry_value: float
    curvature_value: float
    geometry_path: GeometryPath | None


@dataclass(slots=True)
class _WeightedPathObjective:
    initial_state: GeometryState
    initial_s: float
    init_w: float | None
    terminal_w_max: float | None
    settings: PathOptimizerSettings
    time_weight: float
    geometry_weight: float
    curvature_weight: float
    endpoint_objective: EndpointObjective | None
    geometry_objective: GeometryObjective | None
    profiler: reverse_solver.PhaseProfiler | None
    _scalar_x: Array | None = field(default=None, init=False, repr=False)
    _scalar_result: _PathScalarEvaluation | None = field(default=None, init=False, repr=False)
    _x: Array | None = field(default=None, init=False, repr=False)
    _result: _PathObjectiveEvaluation | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.time_weight = _checked_objective_weight(
            self.time_weight, "time_weight"
        )
        self.geometry_weight = _checked_objective_weight(
            self.geometry_weight, "geometry_weight"
        )
        self.curvature_weight = _checked_objective_weight(
            self.curvature_weight, "curvature_weight"
        )
        if self.endpoint_objective is not None and self.geometry_objective is not None:
            raise ValueError(
                "endpoint_objective and geometry_objective are mutually exclusive"
            )

    def _release_native_scalar_build(self) -> None:
        scalar = self._scalar_result
        if scalar is not None and isinstance(
            scalar.speed_build, reverse_solver.NativeProdSpeedProfileBuild
        ):
            scalar.speed_build.close()

    def close(self) -> None:
        """Release any retained native scalar topology deterministically."""
        self._release_native_scalar_build()
        self._scalar_result = None
        self._scalar_x = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def evaluate_scalar(self, knot_params: Sequence[float]) -> _PathScalarEvaluation:
        x = np.asarray(knot_params, dtype=float)
        if self._scalar_x is not None and np.array_equal(x, self._scalar_x):
            assert self._scalar_result is not None
            return self._scalar_result
        if self._x is not None and np.array_equal(x, self._x):
            assert self._result is not None
            self._release_native_scalar_build()
            result = _PathScalarEvaluation(
                self._result.value,
                self._result.raw_parameters,
                self._result.time_value,
                self._result.geometry_value,
                self._result.curvature_value,
                self._result.geometry_path,
                None,
            )
            self._scalar_x = x.copy()
            self._scalar_result = result
            return result

        # A new optimizer point supersedes the previous scalar topology.
        # Native builds own segment/crossing caches, so release them now rather
        # than waiting for Python garbage collection.
        self._release_native_scalar_build()
        self._scalar_result = None
        self._scalar_x = None

        raw = knot_parameters_to_raw(
            x,
            initial_k=self.initial_state.k,
            initial_s=self.initial_s,
        )

        speed_build: reverse_solver.ProdSpeedProfileBuild | reverse_solver.NativeProdSpeedProfileBuild | None = None
        if self.time_weight > 0.0:
            speed_build = reverse_solver.build_scalar_speed_profile(
                raw,
                init_w=self.init_w,
                terminal_w_max=self.terminal_w_max,
                forward_init_k=self.initial_state.k,
                n_scan=self.settings.n_scan,
                envelope_scan=self.settings.envelope_scan,
                domain_margin=self.settings.domain_margin,
                domain_scan=self.settings.domain_scan,
                profiler=self.profiler,
                fused_grip_discovery=self.settings.fused_grip_discovery,
            )
            if isinstance(speed_build, reverse_solver.NativeProdSpeedProfileBuild):
                time_value = speed_build.scalar_time_value(profiler=self.profiler)
            else:
                time_value, _diagnostics = scalar_reverse_solver.scalar_time_from_build(
                    speed_build, profiler=self.profiler
                )
            time_value = float(time_value)
        else:
            time_value = 0.0

        geometry_path: GeometryPath | None = None
        if self.geometry_weight > 0.0:
            geometry_path = compile_geometry_path(raw, self.initial_state)
            if self.geometry_objective is not None:
                geometry_value, _ignored_gradient = self.geometry_objective(geometry_path)
                geometry_value = float(geometry_value)
            elif self.endpoint_objective is not None:
                geometry_value, _ignored_seed = self.endpoint_objective(
                    geometry_path.final_state
                )
                geometry_value = float(geometry_value)
            else:
                geometry_value = 0.0
            if not math.isfinite(geometry_value):
                raise ValueError("geometry objective returned nonfinite value")
        else:
            geometry_value = 0.0

        if self.curvature_weight > 0.0:
            curvature_value, _ignored_gradient = curvature_energy_value_and_raw_gradient(
                raw, initial_k=self.initial_state.k
            )
            curvature_value = float(curvature_value)
        else:
            curvature_value = 0.0

        value = math.fma(
            self.time_weight,
            time_value,
            math.fma(
                self.geometry_weight,
                geometry_value,
                self.curvature_weight * curvature_value,
            ),
        )
        result = _PathScalarEvaluation(
            float(value),
            tuple(float(v) for v in raw),
            time_value,
            geometry_value,
            curvature_value,
            geometry_path,
            speed_build,
        )
        self._scalar_x = x.copy()
        self._scalar_result = result
        return result

    def evaluate(self, knot_params: Sequence[float]) -> _PathObjectiveEvaluation:
        x = np.asarray(knot_params, dtype=float)
        if self._x is not None and np.array_equal(x, self._x):
            assert self._result is not None
            return self._result

        scalar = (
            self._scalar_result
            if self._scalar_x is not None and np.array_equal(x, self._scalar_x)
            else None
        )
        if scalar is None:
            raw = knot_parameters_to_raw(
                x,
                initial_k=self.initial_state.k,
                initial_s=self.initial_s,
            )
        else:
            raw = list(scalar.raw_parameters)
        zero_raw = [0.0] * len(raw)

        if self.time_weight > 0.0:
            if scalar is not None and scalar.speed_build is not None:
                if isinstance(
                    scalar.speed_build, reverse_solver.NativeProdSpeedProfileBuild
                ):
                    if (
                        int(self.settings.domain_scan) != scalar.speed_build.domain_scan
                        or float(self.settings.domain_margin) != scalar.speed_build.domain_margin
                    ):
                        raise ValueError(
                            "native build promotion options must match scalar-build domain settings"
                        )
                    time_value, raw_time_gradient_in = (
                        scalar.speed_build.time_value_and_gradient_native(
                            profiler=self.profiler
                        )
                    )
                else:
                    time_value, raw_time_gradient_in = (
                        reverse_solver.time_value_and_gradient_from_build(
                            scalar.speed_build,
                            domain_margin=self.settings.domain_margin,
                            domain_scan=self.settings.domain_scan,
                            profiler=self.profiler,
                        )
                    )
                if not math.isclose(
                    float(time_value), scalar.time_value, rel_tol=5.0e-13, abs_tol=5.0e-13
                ):
                    raise AssertionError(
                        "scalar and differentiable objective values disagree at one optimizer point: "
                        f"scalar={scalar.time_value:.17g}, diff={float(time_value):.17g}"
                    )
                if isinstance(
                    scalar.speed_build, reverse_solver.NativeProdSpeedProfileBuild
                ):
                    scalar.speed_build.close()
                    scalar = _PathScalarEvaluation(
                        scalar.value, scalar.raw_parameters, scalar.time_value,
                        scalar.geometry_value, scalar.curvature_value,
                        scalar.geometry_path, None,
                    )
                    self._scalar_result = scalar
            else:
                time_value, raw_time_gradient_in = reverse_solver.time_value_and_gradient(
                    raw,
                    init_w=self.init_w,
                    terminal_w_max=self.terminal_w_max,
                    initial_k=self.initial_state.k,
                    n_scan=self.settings.n_scan,
                    envelope_scan=self.settings.envelope_scan,
                    domain_margin=self.settings.domain_margin,
                    domain_scan=self.settings.domain_scan,
                    profiler=self.profiler,
                    fused_grip_discovery=self.settings.fused_grip_discovery,
                )
            time_value = float(time_value)
            raw_time_gradient = [float(v) for v in raw_time_gradient_in]
        else:
            time_value = 0.0
            raw_time_gradient = zero_raw.copy()

        geometry_path: GeometryPath | None = scalar.geometry_path if scalar is not None else None
        if self.geometry_weight > 0.0:
            if geometry_path is None:
                geometry_path = compile_geometry_path(raw, self.initial_state)
            if self.geometry_objective is not None:
                geometry_value, raw_geometry_gradient_in = self.geometry_objective(
                    geometry_path
                )
                geometry_value = float(geometry_value)
                raw_geometry_gradient = [float(v) for v in raw_geometry_gradient_in]
                if len(raw_geometry_gradient) != len(raw):
                    raise ValueError(
                        "geometry objective returned a raw gradient with the wrong length"
                    )
                if not math.isfinite(geometry_value) or not all(
                    math.isfinite(v) for v in raw_geometry_gradient
                ):
                    raise ValueError("geometry objective returned nonfinite data")
            elif self.endpoint_objective is not None:
                geometry_value, endpoint_seed = self.endpoint_objective(
                    geometry_path.final_state
                )
                geometry_value = float(geometry_value)
                if len(endpoint_seed) != 4:
                    raise ValueError(
                        "endpoint objective must return a seed of length four"
                    )
                endpoint_seed4 = tuple(map(float, endpoint_seed))
                if not math.isfinite(geometry_value) or not all(
                    math.isfinite(v) for v in endpoint_seed4
                ):
                    raise ValueError("endpoint objective returned nonfinite data")
                raw_geometry_gradient = geometry_path.endpoint_vjp(endpoint_seed4)  # type: ignore[arg-type]
            else:
                geometry_value = 0.0
                raw_geometry_gradient = zero_raw.copy()
        else:
            geometry_value = 0.0
            raw_geometry_gradient = zero_raw.copy()

        if self.curvature_weight > 0.0:
            curvature_value, raw_curvature_gradient = curvature_energy_value_and_raw_gradient(
                raw, initial_k=self.initial_state.k
            )
        else:
            curvature_value = 0.0
            raw_curvature_gradient = zero_raw.copy()

        raw_gradient = [
            math.fma(
                self.time_weight,
                gt,
                math.fma(
                    self.geometry_weight,
                    gg,
                    self.curvature_weight * gc,
                ),
            )
            for gt, gg, gc in zip(
                raw_time_gradient, raw_geometry_gradient, raw_curvature_gradient
            )
        ]
        gradient = np.asarray(
            pullback_raw_gradient_to_knot_parameters(
                x, raw_gradient,
                initial_k=self.initial_state.k,
                initial_s=self.initial_s,
            ),
            dtype=float,
        )
        value = math.fma(
            self.time_weight,
            time_value,
            math.fma(
                self.geometry_weight,
                geometry_value,
                self.curvature_weight * curvature_value,
            ),
        )
        result = _PathObjectiveEvaluation(
            float(value), gradient, tuple(float(v) for v in raw),
            tuple(float(v) for v in raw_gradient), time_value, geometry_value,
            curvature_value, geometry_path,
        )
        self._x = x.copy()
        self._result = result
        return result

    def fun(self, knot_params: Sequence[float]) -> float:
        return self.evaluate_scalar(knot_params).value

    def jac(self, knot_params: Sequence[float]) -> Array:
        return self.evaluate(knot_params).gradient

    def __call__(self, knot_params: Sequence[float]) -> tuple[float, Array]:
        result = self.evaluate(knot_params)
        return result.value, result.gradient



def _checked_initial_state(
    initial_state: GeometryState | Sequence[float],
) -> GeometryState:
    if len(initial_state) != 4:
        raise ValueError("initial_state must be (x0, y0, theta0, k0)")
    state = GeometryState(*map(float, initial_state))
    if not all(math.isfinite(value) for value in state):
        raise ValueError("initial_state must be finite")
    return state


def _phase_one_failure_result(
    phase: PhaseOneResult,
    state0: GeometryState,
    initial_s: float,
) -> PathOptimizationResult:
    raw = knot_parameters_to_raw(
        phase.x,
        initial_k=state0.k,
        initial_s=initial_s,
    )
    path = compile_geometry_path(raw, state0)
    return PathOptimizationResult(
        math.nan,
        np.asarray(phase.x, dtype=float).copy(),
        tuple(raw),
        path.final_state,
        np.full(len(phase.x), math.nan, dtype=float),
        tuple(math.nan for _ in raw),
        False,
        f"phase I failed: {phase.message}",
        None,
        phase,
        math.nan,
        math.nan,
        math.nan,
    )


@dataclass(frozen=True, slots=True)
class _ReferenceKnotObjective:
    """Dimensionless least-squares distance to one knot vector."""

    reference: Array
    scale: Array

    def __call__(self, variables: Sequence[float]) -> tuple[float, Array]:
        x = np.asarray(variables, dtype=float)
        if x.shape != self.reference.shape:
            raise ValueError("reference-knot objective received the wrong width")
        delta = (x - self.reference) / self.scale
        return 0.5 * float(np.dot(delta, delta)), delta / self.scale


def project_path_to_feasibility(
    x0: Sequence[float],
    corridor: CorridorModel,
    initial_state: GeometryState | Sequence[float],
    *,
    initial_s: float = 0.0,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    equalities: VectorConstraint | None = None,
    endpoint_target: tuple[
        float | None, float | None, float | None, float | None
    ] | None = None,
    additional_inequalities: VectorConstraint | None = None,
    pool: ConstraintPool | None = None,
    settings: ExchangeSettings = ExchangeSettings(),
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
) -> ExchangeResult:
    """Project a nearly feasible path onto all exact geometric constraints.

    SLSQP can terminate with an excellent objective and residuals around
    ``1e-6``--``1e-8`` after a difficult time solve.  Discarding that path in
    favor of a much slower warm start is unnecessarily destructive.  This
    routine performs a cheap final exchange solve with a dimensionless
    least-squares distance to the candidate, preserving its geometry while
    restoring endpoint, corridor, bound, and optional inequality feasibility.

    The projection uses the same fixed positive log-length reference as the
    candidate, so it cannot recreate zero or unordered knot intervals.
    """
    state0 = _checked_initial_state(initial_state)
    initial_s = float(initial_s)
    if not math.isfinite(initial_s):
        raise ValueError("initial_s must be finite")
    reference = np.asarray(x0, dtype=float)
    if reference.ndim != 1 or reference.size == 0 or reference.size % 2:
        raise ValueError("x0 must be a nonempty [s1,k1,...] vector")
    if not np.all(np.isfinite(reference)):
        raise ValueError("x0 must be finite")
    reference = reference.copy()

    station_scale = max(1.0, abs(float(reference[-2] - initial_s)))
    curvature_scale = max(1.0, float(np.max(np.abs(reference[1::2]))))
    scale = np.empty_like(reference)
    scale[0::2] = station_scale
    scale[1::2] = curvature_scale
    objective = _ReferenceKnotObjective(reference, scale)
    coordinate_map = (
        None
        if coordinate_settings.mode == "stations"
        else LogLengthKnotMap.from_knot_parameters(
            reference, initial_s=initial_s
        )
    )
    return run_constraint_generation(
        reference,
        objective,
        corridor,
        state0,
        initial_s=initial_s,
        bounds=bounds,
        equalities=equalities,
        endpoint_target=endpoint_target,
        additional_inequalities=additional_inequalities,
        pool=pool,
        settings=settings,
        coordinate_map=coordinate_map,
        coordinate_settings=coordinate_settings,
    )


def optimize_path(
    x0: Sequence[float],
    corridor: CorridorModel,
    initial_state: GeometryState | Sequence[float],
    *,
    init_w: float | None = None,
    terminal_w_max: float | None = None,
    initial_s: float = 0.0,
    time_weight: float = 1.0,
    geometry_weight: float = 0.0,
    curvature_weight: float = 0.0,
    endpoint_objective: EndpointObjective | None = None,
    geometry_objective: GeometryObjective | None = None,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    equalities: VectorConstraint | None = None,
    endpoint_target: tuple[
        float | None, float | None, float | None, float | None
    ] | None = None,
    additional_inequalities: VectorConstraint | None = None,
    pool: ConstraintPool | None = None,
    settings: PathOptimizerSettings = PathOptimizerSettings(),
    profiler: reverse_solver.PhaseProfiler | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> PathOptimizationResult:
    """Optimize a corridor-constrained clothoid path.

    The minimized scalar is

    ``time_weight*T + geometry_weight*G + curvature_weight*integral(k^2 ds)``.

    The three weights must be finite and nonnegative and need not sum to one.
    Setting a weight to zero disables that objective channel.  The primary
    outputs remain ``result.time`` and ``result.parameters``; travel time is
    evaluated once at the final path even when it is not part of the objective.

    ``endpoint_target`` defines hard final-pose equalities.  It is independent
    of the optional soft ``endpoint_objective`` geometry term.
    """
    state0 = _checked_initial_state(initial_state)
    initial_s = float(initial_s)
    if not math.isfinite(initial_s):
        raise ValueError("initial_s must be finite")
    time_weight = _checked_objective_weight(time_weight, "time_weight")
    geometry_weight = _checked_objective_weight(
        geometry_weight, "geometry_weight"
    )
    curvature_weight = _checked_objective_weight(
        curvature_weight, "curvature_weight"
    )
    if endpoint_objective is not None and geometry_objective is not None:
        raise ValueError(
            "endpoint_objective and geometry_objective are mutually exclusive"
        )

    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or x.size == 0 or x.size % 2:
        raise ValueError("x0 must be a nonempty flat [s1,k1,...,sn,kn] vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must be finite")
    x = x.copy()
    coordinate_map = (
        None
        if settings.coordinates.mode == "stations"
        else LogLengthKnotMap.from_knot_parameters(x, initial_s=initial_s)
    )

    phase_result: PhaseOneResult | None = None
    active_pool = pool
    if settings.phase_one is not None:
        phase_result = run_phase_one(
            x,
            corridor,
            state0,
            initial_s=initial_s,
            bounds=bounds,
            equalities=equalities,
            endpoint_target=endpoint_target,
            additional_inequalities=additional_inequalities,
            pool=active_pool,
            settings=settings.phase_one,
            coordinate_map=coordinate_map,
            coordinate_settings=settings.coordinates,
        )
        x = phase_result.x.copy()
        active_pool = phase_result.pool
        if not phase_result.success:
            return _phase_one_failure_result(phase_result, state0, initial_s)

    objective = _WeightedPathObjective(
        state0,
        initial_s,
        init_w,
        terminal_w_max,
        settings,
        time_weight,
        geometry_weight,
        curvature_weight,
        endpoint_objective,
        geometry_objective,
        profiler,
    )
    sparse_endpoint_target = endpoint_target
    exchange_settings = settings.exchange
    if phase_result is not None and settings.exchange.finite_solver == "sparse_sqp":
        # Lossless handoff canonicalization is deliberately performed before
        # constructing any sparse finite state.  It normalizes signed zero,
        # exact duplicate/order differences in the cut pool, and validates the
        # fixed coordinate map without rounding a certified path or cut.
        try:
            handoff = canonicalize_phase_one_handoff(
                x,
                active_pool,
                coordinate_map,
                endpoint_target=endpoint_target,
            )
            x = handoff.physical_x.copy()
            active_pool = handoff.pool
            if endpoint_target is not None:
                sparse_endpoint_target = align_endpoint_target_to_path(
                    x,
                    state0,
                    handoff.endpoint_target,
                    initial_s=initial_s,
                ).aligned_target
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            # Canonicalization is a qualification step, never a reason to lose
            # the parent-owned strict Phase-I certificate.  Return that safe
            # path without constructing or mutating sparse continuation state.
            evaluation = objective.evaluate(x)
            fallback_path = evaluation.geometry_path
            if fallback_path is None:
                fallback_path = compile_geometry_path(
                    evaluation.raw_parameters, state0
                )
            if time_weight > 0.0:
                fallback_time = evaluation.time_value
            else:
                fallback_time = scalar_reverse_solver.evaluate_time_scalar(
                    evaluation.raw_parameters,
                    init_w=init_w,
                    terminal_w_max=terminal_w_max,
                    initial_k=state0.k,
                    n_scan=settings.n_scan,
                    envelope_scan=settings.envelope_scan,
                    domain_margin=settings.domain_margin,
                    domain_scan=settings.domain_scan,
                    profiler=profiler,
                )
            result = PathOptimizationResult(
                float(fallback_time),
                x.copy(),
                evaluation.raw_parameters,
                fallback_path.final_state,
                evaluation.gradient.copy(),
                evaluation.raw_gradient,
                True,
                f"strict Phase-I fallback after handoff canonicalization failure: {exc}",
                None,
                phase_result,
                evaluation.value,
                evaluation.geometry_value,
                evaluation.curvature_value,
            )
            objective.close()
            return result
        exchange_settings = stabilize_phase_one_sparse_exchange(settings.exchange)

    exchange = run_constraint_generation(
        x,
        objective,
        corridor,
        state0,
        initial_s=initial_s,
        bounds=bounds,
        equalities=equalities,
        endpoint_target=sparse_endpoint_target,
        additional_inequalities=additional_inequalities,
        pool=active_pool,
        settings=exchange_settings,
        coordinate_map=coordinate_map,
        coordinate_settings=settings.coordinates,
        progress_callback=progress_callback,
    )
    evaluation = objective.evaluate(exchange.x)
    parameters = np.asarray(exchange.x, dtype=float).copy()
    gradient = np.asarray(evaluation.gradient, dtype=float).copy()

    final_path = evaluation.geometry_path
    if final_path is None:
        final_path = compile_geometry_path(evaluation.raw_parameters, state0)

    if time_weight > 0.0:
        final_time = evaluation.time_value
    else:
        # Time remains a primary return value even when excluded from the
        # scalarized optimization objective.
        final_time = scalar_reverse_solver.evaluate_time_scalar(
            evaluation.raw_parameters,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=state0.k,
            n_scan=settings.n_scan,
            envelope_scan=settings.envelope_scan,
            domain_margin=settings.domain_margin,
            domain_scan=settings.domain_scan,
            profiler=profiler,
        )

    result = PathOptimizationResult(
        float(final_time),
        parameters,
        evaluation.raw_parameters,
        final_path.final_state,
        gradient,
        evaluation.raw_gradient,
        exchange.success,
        exchange.message,
        exchange,
        phase_result,
        evaluation.value,
        evaluation.geometry_value,
        evaluation.curvature_value,
    )
    objective.close()
    return result


__all__ = [
    "CachedObjective",
    "ExchangeResult",
    "ExchangeRoundResult",
    "ExchangeSettings",
    "FiniteSolveResult",
    "NonlinearSolveResult",
    "PathOptimizationResult",
    "PathOptimizerSettings",
    "PhaseOneResult",
    "PhaseOneSettings",
    "SLSQPSettings",
    "ScalingSettings",
    "SolveScaling",
    "optimize_path",
    "project_path_to_feasibility",
    "run_constraint_generation",
    "run_phase_one",
    "solve_finite_slsqp",
]
