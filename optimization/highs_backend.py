"""Minimal HiGHS LP/QP backend with optional public-package support.

The experimental sparse SQP uses one stable interface for both the standalone
``highspy`` package and SciPy's bundled HiGHS extension.  The public package is
preferred.  SciPy's private wrapper is retained only as an explicitly reported
candidate-build fallback so the algorithm can be validated before a packaging
decision is made.

Every returned solution is independently checked against row and variable
bounds.  Warm starts are opaque backend objects and are accepted only when the
new model has the same dimensions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HighsBackendInfo:
    name: str
    public_api: bool
    version: str


@dataclass(slots=True)
class HighsWarmStart:
    n_columns: int
    n_rows: int
    basis: Any | None = None
    solution: Any | None = None


@dataclass(frozen=True, slots=True)
class HighsSolveResult:
    x: Array
    row_value: Array
    column_dual: Array
    row_dual: Array
    objective: float
    success: bool
    status: int
    message: str
    iterations: int
    primal_infeasibility: float
    dual_infeasibility: float
    wall_seconds: float
    warm_start: HighsWarmStart | None
    warm_start_used: bool
    backend: HighsBackendInfo
    checked_stationarity: float
    checked_stationarity_scaled: float
    checked_complementarity: float
    checked_complementarity_scaled: float
    checked_dual_sign_violation: float
    optimality_certified: bool


def _load_backend(*, allow_private_fallback: bool = True):
    try:
        import highspy as api  # type: ignore[import-not-found]

        solver_type = api.Highs
        version = getattr(api, "__version__", "unknown")
        if version == "unknown":
            try:
                version = str(solver_type().version())
            except Exception:
                pass
        return api, solver_type, HighsBackendInfo("highspy", True, str(version))
    except ImportError:
        if not allow_private_fallback:
            raise ImportError(
                "the public highspy package is required but is not installed"
            ) from None
        from scipy.optimize._highspy import _core as api  # type: ignore[attr-defined]

        solver_type = api._Highs
        try:
            probe = solver_type()
            version = str(probe.version())
        except Exception:
            version = "scipy-bundled"
        return api, solver_type, HighsBackendInfo(
            "scipy.optimize._highspy", False, version
        )


def backend_info(*, allow_private_fallback: bool = True) -> HighsBackendInfo:
    return _load_backend(allow_private_fallback=allow_private_fallback)[2]


def _as_csc(matrix: object, n_columns: int, zero_tolerance: float):
    from scipy.sparse import csc_matrix

    a = csc_matrix(matrix, dtype=float)
    if a.shape[1] != n_columns:
        raise ValueError("constraint matrix has the wrong width")
    if zero_tolerance > 0.0 and a.nnz:
        a.data[np.abs(a.data) <= zero_tolerance] = 0.0
        a.eliminate_zeros()
    if not np.all(np.isfinite(a.data)):
        raise ValueError("constraint matrix contains nonfinite entries")
    return a


def _checked_bounds(values: Sequence[float], size: int, name: str) -> Array:
    array = np.broadcast_to(np.asarray(values, dtype=float), (size,)).copy()
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} contains NaN")
    return array


def _build_lp(
    api: Any,
    gradient: Array,
    matrix: object,
    row_lower: Sequence[float],
    row_upper: Sequence[float],
    variable_lower: Sequence[float],
    variable_upper: Sequence[float],
    *,
    zero_tolerance: float,
):
    n = int(gradient.size)
    a = _as_csc(matrix, n, zero_tolerance)
    m = int(a.shape[0])
    row_lo = _checked_bounds(row_lower, m, "row_lower")
    row_hi = _checked_bounds(row_upper, m, "row_upper")
    col_lo = _checked_bounds(variable_lower, n, "variable_lower")
    col_hi = _checked_bounds(variable_upper, n, "variable_upper")
    if np.any(row_lo > row_hi) or np.any(col_lo > col_hi):
        raise ValueError("invalid HiGHS bounds")

    lp = api.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = gradient
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = row_lo
    lp.row_upper_ = row_hi
    lp.sense_ = api.ObjSense.kMinimize
    lp.a_matrix_.format_ = api.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = n
    lp.a_matrix_.num_row_ = m
    lp.a_matrix_.start_ = np.asarray(a.indptr, dtype=np.int32)
    lp.a_matrix_.index_ = np.asarray(a.indices, dtype=np.int32)
    lp.a_matrix_.value_ = np.asarray(a.data, dtype=float)
    return lp, a, row_lo, row_hi, col_lo, col_hi


def _configure_solver(
    solver: Any,
    *,
    feasibility_tolerance: float,
    optimality_tolerance: float,
    time_limit: float,
    presolve: bool,
    display: bool,
    qp_iteration_limit: int | None = None,
    threads: int | None = None,
) -> None:
    solver.setOptionValue("output_flag", bool(display))
    solver.setOptionValue("presolve", "on" if presolve else "off")
    solver.setOptionValue(
        "primal_feasibility_tolerance", float(feasibility_tolerance)
    )
    solver.setOptionValue(
        "dual_feasibility_tolerance", float(optimality_tolerance)
    )
    if math.isfinite(time_limit):
        solver.setOptionValue("time_limit", float(time_limit))
    if qp_iteration_limit is not None:
        solver.setOptionValue("qp_iteration_limit", int(qp_iteration_limit))
    if threads is not None:
        if int(threads) <= 0:
            raise ValueError("HiGHS threads must be positive when supplied")
        solver.setOptionValue("threads", int(threads))


def _apply_warm_start(
    solver: Any,
    warm_start: HighsWarmStart | None,
    n_columns: int,
    n_rows: int,
    *,
    use_basis: bool,
    use_solution: bool,
) -> bool:
    if (
        warm_start is None
        or warm_start.n_columns != n_columns
        or warm_start.n_rows != n_rows
    ):
        return False
    used = False
    if use_basis and warm_start.basis is not None:
        try:
            status = solver.setBasis(warm_start.basis)
            used = "kError" not in str(status)
        except Exception:
            pass
    if use_solution and warm_start.solution is not None:
        try:
            status = solver.setSolution(warm_start.solution)
            used = used or "kError" not in str(status)
        except Exception:
            pass
    return used


def _finish_result(
    api: Any,
    solver: Any,
    a: object,
    row_lo: Array,
    row_hi: Array,
    col_lo: Array,
    col_hi: Array,
    *,
    feasibility_tolerance: float,
    started: float,
    warm_start_used: bool,
    backend: HighsBackendInfo,
) -> HighsSolveResult:
    model_status = solver.getModelStatus()
    message = str(solver.modelStatusToString(model_status))
    solution = solver.getSolution()
    info = solver.getInfo()
    n = int(col_lo.size)
    m = int(row_lo.size)
    value_valid = bool(getattr(solution, "value_valid", False))
    dual_valid = bool(getattr(solution, "dual_valid", False))
    x = (
        np.asarray(solution.col_value, dtype=float).copy()
        if value_valid
        else np.zeros(n, dtype=float)
    )
    row_value = (
        np.asarray(solution.row_value, dtype=float).copy()
        if value_valid
        else np.zeros(m, dtype=float)
    )
    if value_valid and (row_value.size != m or not np.all(np.isfinite(row_value))):
        row_value = np.asarray(a @ x, dtype=float)
    col_dual = (
        np.asarray(solution.col_dual, dtype=float).copy()
        if dual_valid
        else np.zeros(n, dtype=float)
    )
    row_dual = (
        np.asarray(solution.row_dual, dtype=float).copy()
        if dual_valid
        else np.zeros(m, dtype=float)
    )

    if value_valid and np.all(np.isfinite(x)):
        if row_value.size != m:
            row_value = np.asarray(a @ x, dtype=float)
        checked_primal = max(
            float(np.max(row_lo - row_value, initial=-math.inf)),
            float(np.max(row_value - row_hi, initial=-math.inf)),
            float(np.max(col_lo - x, initial=-math.inf)),
            float(np.max(x - col_hi, initial=-math.inf)),
            0.0,
        )
    else:
        checked_primal = math.inf
    reported_primal = (
        float(info.max_primal_infeasibility)
        if bool(getattr(info, "valid", False))
        else math.inf
    )
    primal = max(checked_primal, reported_primal)
    dual = (
        float(info.max_dual_infeasibility)
        if bool(getattr(info, "valid", False))
        else math.inf
    )
    optimal = str(model_status).endswith("kOptimal")
    success = (
        optimal
        and value_valid
        and np.all(np.isfinite(x))
        and primal <= max(10.0 * feasibility_tolerance, 1.0e-9)
    )
    if not success and optimal:
        message += f"; rejected primal infeasibility {primal:.3e}"
    try:
        basis = solver.getBasis()
        if not bool(getattr(basis, "valid", False)):
            basis = None
    except Exception:
        basis = None
    try:
        retained_solution = solution if value_valid else None
    except Exception:
        retained_solution = None
    warm = HighsWarmStart(n, m, basis=basis, solution=retained_solution)
    iterations = int(
        getattr(info, "qp_iteration_count", 0)
        + getattr(info, "simplex_iteration_count", 0)
        + getattr(info, "ipm_iteration_count", 0)
    )
    objective = (
        float(info.objective_function_value)
        if bool(getattr(info, "valid", False))
        else math.nan
    )
    return HighsSolveResult(
        x=x,
        row_value=row_value,
        column_dual=col_dual,
        row_dual=row_dual,
        objective=objective,
        success=success,
        status=int(model_status),
        message=message,
        iterations=iterations,
        primal_infeasibility=primal,
        dual_infeasibility=dual,
        wall_seconds=time.perf_counter() - started,
        warm_start=warm,
        warm_start_used=warm_start_used,
        backend=backend,
        checked_stationarity=math.inf,
        checked_stationarity_scaled=math.inf,
        checked_complementarity=math.inf,
        checked_complementarity_scaled=math.inf,
        checked_dual_sign_violation=math.inf,
        optimality_certified=False,
    )


def _certify_optimality(
    result: HighsSolveResult,
    gradient_at_x: Array,
    matrix: object,
    row_lower: Array,
    row_upper: Array,
    variable_lower: Array,
    variable_upper: Array,
    *,
    optimality_tolerance: float,
    complementarity_tolerance: float | None = None,
) -> HighsSolveResult:
    """Independently check primal-dual KKT residuals for an LP/QP result.

    HiGHS uses the stationarity convention

        grad f(x) - A.T @ row_dual - column_dual = 0.

    Positive row/column duals correspond to active lower bounds; negative
    duals correspond to active upper bounds.  The normalized residuals make
    the certificate insensitive to objective and multiplier scaling.
    """
    from scipy.sparse import csc_matrix

    if not result.success or not np.all(np.isfinite(gradient_at_x)):
        return result
    a = csc_matrix(matrix, dtype=float)
    row_term = np.asarray(a.T @ result.row_dual, dtype=float).reshape(-1)
    col_term = np.asarray(result.column_dual, dtype=float)
    residual = np.asarray(gradient_at_x, dtype=float) - row_term - col_term
    stationarity = float(np.linalg.norm(residual, ord=np.inf))
    stationarity_scale = max(
        1.0,
        float(np.linalg.norm(gradient_at_x, ord=np.inf)),
        float(np.linalg.norm(row_term, ord=np.inf)),
        float(np.linalg.norm(col_term, ord=np.inf)),
    )
    stationarity_scaled = stationarity / stationarity_scale

    dual_sign = 0.0
    complementarity = 0.0
    row_value = result.row_value
    for value, lo, hi, dual in zip(
        row_value, row_lower, row_upper, result.row_dual, strict=True
    ):
        if dual > 0.0:
            if not math.isfinite(lo):
                dual_sign = max(dual_sign, float(dual))
            else:
                complementarity = max(
                    complementarity, abs(float(dual) * float(value - lo))
                )
        elif dual < 0.0:
            if not math.isfinite(hi):
                dual_sign = max(dual_sign, float(-dual))
            else:
                complementarity = max(
                    complementarity, abs(float(-dual) * float(hi - value))
                )
    for value, lo, hi, dual in zip(
        result.x, variable_lower, variable_upper, result.column_dual, strict=True
    ):
        if dual > 0.0:
            if not math.isfinite(lo):
                dual_sign = max(dual_sign, float(dual))
            else:
                complementarity = max(
                    complementarity, abs(float(dual) * float(value - lo))
                )
        elif dual < 0.0:
            if not math.isfinite(hi):
                dual_sign = max(dual_sign, float(-dual))
            else:
                complementarity = max(
                    complementarity, abs(float(-dual) * float(hi - value))
                )
    complementarity_scale = max(
        1.0,
        abs(float(result.objective)) if math.isfinite(result.objective) else 1.0,
        stationarity_scale,
    )
    complementarity_scaled = complementarity / complementarity_scale
    stat_limit = max(5.0e-7, 50.0 * optimality_tolerance)
    comp_limit = (
        max(5.0e-8, 50.0 * optimality_tolerance)
        if complementarity_tolerance is None
        else complementarity_tolerance
    )
    sign_limit = max(5.0e-8, 50.0 * optimality_tolerance)
    certified = bool(
        result.success
        and stationarity_scaled <= stat_limit
        and complementarity_scaled <= comp_limit
        and dual_sign <= sign_limit
    )
    message = result.message
    if result.success and not certified:
        message += (
            "; independent KKT check failed "
            f"(stationarity={stationarity_scaled:.3e}, "
            f"complementarity={complementarity_scaled:.3e}, "
            f"dual-sign={dual_sign:.3e})"
        )
    return replace(
        result,
        message=message,
        checked_stationarity=stationarity,
        checked_stationarity_scaled=stationarity_scaled,
        checked_complementarity=complementarity,
        checked_complementarity_scaled=complementarity_scaled,
        checked_dual_sign_violation=dual_sign,
        optimality_certified=certified,
    )


def solve_linear_program(
    gradient: Sequence[float],
    constraint_matrix: object,
    row_lower: Sequence[float],
    row_upper: Sequence[float],
    variable_lower: Sequence[float],
    variable_upper: Sequence[float],
    *,
    warm_start: HighsWarmStart | None = None,
    warm_start_basis: bool = True,
    warm_start_solution: bool = False,
    feasibility_tolerance: float = 1.0e-8,
    optimality_tolerance: float = 1.0e-8,
    time_limit: float = math.inf,
    zero_tolerance: float = 1.0e-14,
    presolve: bool = True,
    display: bool = False,
    allow_private_fallback: bool = True,
    threads: int | None = None,
) -> HighsSolveResult:
    started = time.perf_counter()
    g = np.asarray(gradient, dtype=float)
    if g.ndim != 1 or not np.all(np.isfinite(g)):
        raise ValueError("gradient must be a finite vector")
    api, solver_type, info = _load_backend(
        allow_private_fallback=allow_private_fallback
    )
    lp, a, row_lo, row_hi, col_lo, col_hi = _build_lp(
        api,
        g,
        constraint_matrix,
        row_lower,
        row_upper,
        variable_lower,
        variable_upper,
        zero_tolerance=zero_tolerance,
    )
    solver = solver_type()
    _configure_solver(
        solver,
        feasibility_tolerance=feasibility_tolerance,
        optimality_tolerance=optimality_tolerance,
        time_limit=time_limit,
        presolve=presolve,
        display=display,
        qp_iteration_limit=None,
        threads=threads,
    )
    pass_status = solver.passModel(lp)
    if "kError" in str(pass_status):
        raise RuntimeError("HiGHS rejected the LP model")
    used = _apply_warm_start(
        solver,
        warm_start,
        g.size,
        a.shape[0],
        use_basis=warm_start_basis,
        use_solution=warm_start_solution,
    )
    solver.run()
    result = _finish_result(
        api,
        solver,
        a,
        row_lo,
        row_hi,
        col_lo,
        col_hi,
        feasibility_tolerance=feasibility_tolerance,
        started=started,
        warm_start_used=used,
        backend=info,
    )
    return _certify_optimality(
        result, g, a, row_lo, row_hi, col_lo, col_hi,
        optimality_tolerance=optimality_tolerance,
    )


def solve_convex_quadratic_program(
    gradient: Sequence[float],
    hessian: object,
    constraint_matrix: object,
    row_lower: Sequence[float],
    row_upper: Sequence[float],
    variable_lower: Sequence[float],
    variable_upper: Sequence[float],
    *,
    warm_start: HighsWarmStart | None = None,
    warm_start_solution: bool = True,
    feasibility_tolerance: float = 1.0e-8,
    optimality_tolerance: float = 1.0e-8,
    time_limit: float = math.inf,
    zero_tolerance: float = 1.0e-14,
    presolve: bool = True,
    display: bool = False,
    allow_private_fallback: bool = True,
    qp_iteration_limit: int | None = None,
    threads: int | None = None,
) -> HighsSolveResult:
    from scipy.sparse import csc_matrix, tril

    started = time.perf_counter()
    g = np.asarray(gradient, dtype=float)
    if g.ndim != 1 or not np.all(np.isfinite(g)):
        raise ValueError("gradient must be a finite vector")
    api, solver_type, info = _load_backend(
        allow_private_fallback=allow_private_fallback
    )
    lp, a, row_lo, row_hi, col_lo, col_hi = _build_lp(
        api,
        g,
        constraint_matrix,
        row_lower,
        row_upper,
        variable_lower,
        variable_upper,
        zero_tolerance=zero_tolerance,
    )
    h = csc_matrix(hessian, dtype=float)
    if h.shape != (g.size, g.size):
        raise ValueError("hessian has the wrong shape")
    lower_h = csc_matrix(tril(h, format="csc"))
    if zero_tolerance > 0.0 and lower_h.nnz:
        lower_h.data[np.abs(lower_h.data) <= zero_tolerance] = 0.0
        lower_h.eliminate_zeros()
    if not np.all(np.isfinite(lower_h.data)):
        raise ValueError("hessian contains nonfinite entries")
    hessian_model = api.HighsHessian()
    hessian_model.dim_ = int(g.size)
    hessian_model.format_ = api.HessianFormat.kTriangular
    hessian_model.start_ = np.asarray(lower_h.indptr, dtype=np.int32)
    hessian_model.index_ = np.asarray(lower_h.indices, dtype=np.int32)
    hessian_model.value_ = np.asarray(lower_h.data, dtype=float)
    model = api.HighsModel()
    model.lp_ = lp
    model.hessian_ = hessian_model

    solver = solver_type()
    _configure_solver(
        solver,
        feasibility_tolerance=feasibility_tolerance,
        optimality_tolerance=optimality_tolerance,
        time_limit=time_limit,
        presolve=presolve,
        display=display,
        qp_iteration_limit=qp_iteration_limit,
        threads=threads,
    )
    pass_status = solver.passModel(model)
    if "kError" in str(pass_status):
        raise RuntimeError("HiGHS rejected the QP model")
    used = _apply_warm_start(
        solver,
        warm_start,
        g.size,
        a.shape[0],
        # HiGHS' active-set QP solver can reuse the retained basis as well
        # as the primal/dual solution. Backends that do not accept a QP basis
        # simply reject it and continue from the solution start.
        use_basis=True,
        use_solution=warm_start_solution,
    )
    solver.run()
    result = _finish_result(
        api,
        solver,
        a,
        row_lo,
        row_hi,
        col_lo,
        col_hi,
        feasibility_tolerance=feasibility_tolerance,
        started=started,
        warm_start_used=used,
        backend=info,
    )
    # HiGHS' QP info object has occasionally carried a stale or NaN internal
    # objective despite an optimal model status. The primal model value is
    # inexpensive and is the only value exposed to the SQP globalization.
    model_objective = float(g @ result.x + 0.5 * result.x @ (h @ result.x))
    if not math.isfinite(model_objective):
        return replace(
            result,
            success=False,
            objective=math.nan,
            message=result.message + "; nonfinite recomputed QP objective",
        )
    result = replace(result, objective=model_objective)
    gradient_at_x = g + np.asarray(h @ result.x, dtype=float).reshape(-1)
    return _certify_optimality(
        result, gradient_at_x, a, row_lo, row_hi, col_lo, col_hi,
        optimality_tolerance=optimality_tolerance,
    )


__all__ = [
    "HighsBackendInfo",
    "HighsSolveResult",
    "HighsWarmStart",
    "backend_info",
    "solve_convex_quadratic_program",
    "solve_linear_program",
]
