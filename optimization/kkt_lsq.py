"""NumPy-only least-squares utilities for KKT stationarity diagnostics.

The KKT multiplier estimate has the specialized form

    minimize ||F y + N z - b||_2,  subject to z >= 0,

where ``y`` contains unrestricted equality multipliers and ``z`` contains
nonnegative inequality/bound multipliers.  The implementation eliminates the
free variables with a rank-revealing SVD and solves the projected NNLS problem
with a deterministic Lawson-Hanson active-set method.

No SciPy types or algorithms are used here.  The implementation is intended to
map directly to a C++ QR/SVD plus active-set implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BoxBounds:
    """Backend-neutral componentwise bounds.

    ``lower`` and ``upper`` are one-dimensional arrays with matching shapes.
    Infinite endpoints are allowed; NaNs are not.
    """

    lower: Array
    upper: Array

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("bound arrays must be one-dimensional with matching shapes")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("bounds must not contain NaNs")
        if np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @property
    def size(self) -> int:
        return int(self.lower.size)


@dataclass(frozen=True, slots=True)
class MixedLeastSquaresResult:
    """Result of a mixed free/nonnegative linear least-squares solve."""

    free: Array
    nonnegative: Array
    residual: Array
    residual_norm: float
    success: bool
    iterations: int
    passive_variables: int
    free_rank: int
    message: str


@dataclass(frozen=True, slots=True)
class KKTReport:
    stationarity_inf: float
    active_inequalities: int
    active_lower_bounds: int
    active_upper_bounds: int
    least_squares_success: bool
    computed: bool = True

    @classmethod
    def not_computed(cls) -> "KKTReport":
        return cls(math.inf, 0, 0, 0, False, False)


def _as_matrix(value: Sequence[Sequence[float]] | Array, rows: int, name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows}, n_columns)")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} must contain only finite values")
    return array


def _stable_column_norms(matrix: Array) -> Array:
    """Euclidean column norms without avoidable overflow or underflow."""
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=float)
    maxima = np.max(np.abs(matrix), axis=0, initial=0.0)
    norms = np.zeros_like(maxima)
    nonzero = maxima > 0.0
    if np.any(nonzero):
        scaled = matrix[:, nonzero] / maxima[nonzero]
        norms[nonzero] = maxima[nonzero] * np.sqrt(
            np.sum(scaled * scaled, axis=0)
        )
    return norms


def _default_rcond(rows: int, columns: int) -> float:
    return float(np.finfo(float).eps * max(1, rows, columns))


def _least_squares_passive(matrix: Array, rhs: Array, rcond: float) -> Array:
    """Solve a passive least-squares subproblem with a verified QR fast path.

    Well-conditioned tall systems use reduced QR, which is materially faster
    than repeatedly invoking an SVD as the active set grows.  Rank-deficient,
    underdetermined, or numerically suspect systems fall back to ``lstsq``.
    The QR candidate is accepted only after a scaled normal-equation residual
    check, preserving the numerical behavior needed by the active-set logic.
    """
    rows, columns = matrix.shape
    if columns == 0:
        return np.empty(0, dtype=float)
    if columns <= rows:
        q, r = np.linalg.qr(matrix, mode="reduced")
        diagonal = np.abs(np.diag(r))
        largest = float(np.max(diagonal, initial=0.0))
        rank_threshold = max(rcond, math.sqrt(np.finfo(float).eps)) * largest
        if largest > 0.0 and np.all(diagonal > rank_threshold):
            try:
                solution = np.linalg.solve(r, q.T @ rhs)
            except np.linalg.LinAlgError:
                pass
            else:
                residual = matrix @ solution - rhs
                normal_residual = matrix.T @ residual
                matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
                scale = max(
                    1.0,
                    float(np.linalg.norm(rhs)),
                    matrix_norm * float(np.linalg.norm(solution)),
                )
                tolerance = (
                    128.0
                    * np.finfo(float).eps
                    * max(1, rows, columns)
                    * max(1.0, matrix_norm)
                    * scale
                )
                if (
                    np.all(np.isfinite(solution))
                    and float(np.max(np.abs(normal_residual), initial=0.0))
                    <= tolerance
                ):
                    return solution
    return np.linalg.lstsq(matrix, rhs, rcond=rcond)[0]


def _nnls_active_set(
    matrix: Array,
    rhs: Array,
    *,
    rcond: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_iterations: int | None,
) -> tuple[Array, bool, int, int, str]:
    """Solve ``min ||A x-b||`` with ``x>=0`` using a stable active set.

    Columns are scaled to unit 2-norm before the active-set iterations.  Every
    passive subproblem is solved by ``numpy.linalg.lstsq`` (SVD-backed on the
    supported NumPy/LAPACK builds), avoiding normal equations and their squared
    condition number.
    """
    rows, columns = matrix.shape
    if columns == 0:
        return np.empty(0, dtype=float), True, 0, 0, "no nonnegative variables"
    if rows == 0:
        return np.zeros(columns, dtype=float), True, 0, 0, "empty projected problem"

    norms = _stable_column_norms(matrix)
    usable = norms > 0.0
    if not np.any(usable):
        return np.zeros(columns, dtype=float), True, 0, 0, "all projected columns are zero"

    reduced_indices = np.flatnonzero(usable)
    scaled_matrix = np.asarray(
        matrix[:, reduced_indices] / norms[reduced_indices],
        dtype=float,
        order="F",
    )
    reduced_columns = int(reduced_indices.size)
    coefficients = np.zeros(reduced_columns, dtype=float)
    passive = np.zeros(reduced_columns, dtype=bool)

    rhs_norm = float(np.linalg.norm(rhs))
    dual_tolerance = max(
        absolute_tolerance,
        relative_tolerance * max(1.0, rhs_norm),
    )
    # Positivity decisions need a tighter threshold than the final dual test;
    # using the same threshold can incorrectly discard small but legitimate
    # coefficients after column normalization.
    positivity_floor = max(
        np.finfo(float).tiny,
        8.0 * np.finfo(float).eps * max(1.0, rhs_norm),
    )
    verification_tolerance = max(
        16.0 * dual_tolerance,
        128.0 * np.finfo(float).eps * max(1.0, rhs_norm),
    )
    if maximum_iterations is None:
        maximum_iterations = max(50, 10 * reduced_columns)
    elif maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")

    residual = rhs.copy()
    dual = scaled_matrix.T @ residual
    linear_solves = 0
    blocked = np.zeros(reduced_columns, dtype=bool)

    while linear_solves < maximum_iterations:
        eligible = (~passive) & (~blocked) & (dual > dual_tolerance)
        if not np.any(eligible):
            if np.any((~passive) & (dual > dual_tolerance)):
                # Every currently improving variable was rejected without a
                # passive-set change.  This is a numerically degenerate fixed
                # point rather than convergence.
                message = "active set stalled on degenerate entering variables"
                success = False
            else:
                message = "optimality conditions satisfied"
                success = True
            break

        entering_candidates = np.flatnonzero(eligible)
        entering = int(
            entering_candidates[np.argmax(dual[entering_candidates])]
        )
        passive_before = passive.copy()
        passive[entering] = True

        while linear_solves < maximum_iterations:
            passive_indices = np.flatnonzero(passive)
            trial = np.zeros_like(coefficients)
            if passive_indices.size:
                trial_values = _least_squares_passive(
                    scaled_matrix[:, passive_indices], rhs, rcond
                )
                linear_solves += 1
                trial[passive_indices] = trial_values

            nonpositive = passive & (trial <= positivity_floor)
            if not np.any(nonpositive):
                coefficients = trial
                blocked.fill(False)
                break

            moving = passive & (coefficients > positivity_floor) & (trial <= positivity_floor)
            if np.any(moving):
                alpha_values = coefficients[moving] / (
                    coefficients[moving] - trial[moving]
                )
                alpha = float(np.min(alpha_values))
                alpha = min(1.0, max(0.0, alpha))
                coefficients += alpha * (trial - coefficients)
            else:
                # The newly entered variable has a nonpositive passive solve,
                # so no feasible movement exists from the current point.
                coefficients = np.maximum(coefficients, 0.0)

            leaving = passive & (coefficients <= positivity_floor)
            if not np.any(leaving):
                # Guard against a zero-length step caused by rounding.
                candidate = np.flatnonzero(nonpositive)
                leaving[int(candidate[np.argmin(trial[candidate])])] = True
            coefficients[leaving] = 0.0
            passive[leaving] = False

        residual = rhs - scaled_matrix @ coefficients
        dual = scaled_matrix.T @ residual
        if np.array_equal(passive, passive_before):
            # Prevent immediate re-entry of a variable that could not produce a
            # feasible passive solution.  Any later passive-set change clears
            # this deterministic anti-cycling guard.
            blocked[entering] = True
        else:
            blocked.fill(False)
    else:
        success = False
        message = "maximum active-set iterations reached"

    # A final passive solve removes accumulated interpolation error.  If the
    # minimum-norm passive solution is slightly negative, clean it once and
    # resolve; material negativity is left for the optimality verification.
    if np.any(passive):
        passive_indices = np.flatnonzero(passive)
        refined = _least_squares_passive(
            scaled_matrix[:, passive_indices], rhs, rcond
        )
        linear_solves += 1
        tiny_negative = (refined < 0.0) & (refined >= -verification_tolerance)
        refined[tiny_negative] = 0.0
        coefficients.fill(0.0)
        coefficients[passive_indices] = refined
        dropped = passive & (coefficients <= positivity_floor)
        if np.any(dropped):
            passive[dropped] = False
            coefficients[dropped] = 0.0
            passive_indices = np.flatnonzero(passive)
            if passive_indices.size:
                refined = _least_squares_passive(
                    scaled_matrix[:, passive_indices], rhs, rcond
                )
                linear_solves += 1
                coefficients.fill(0.0)
                coefficients[passive_indices] = refined

    residual = rhs - scaled_matrix @ coefficients
    dual = scaled_matrix.T @ residual
    primal_violation = float(max(0.0, -np.min(coefficients, initial=0.0)))
    passive_stationarity = (
        float(np.max(np.abs(dual[passive]), initial=0.0))
        if np.any(passive)
        else 0.0
    )
    inactive_dual_violation = (
        float(max(0.0, np.max(dual[~passive], initial=-math.inf)))
        if np.any(~passive)
        else 0.0
    )
    verified = (
        np.all(np.isfinite(coefficients))
        and primal_violation <= verification_tolerance
        and passive_stationarity <= verification_tolerance
        and inactive_dual_violation <= verification_tolerance
    )
    success = bool(success and verified)
    if not verified:
        message = (
            "active-set result failed KKT verification "
            f"(primal={primal_violation:.3e}, "
            f"passive_dual={passive_stationarity:.3e}, "
            f"inactive_dual={inactive_dual_violation:.3e})"
        )

    nonnegative = np.zeros(columns, dtype=float)
    nonnegative[reduced_indices] = coefficients / norms[reduced_indices]
    # Remove negative signed zero and harmless roundoff before exposing the
    # result to KKT diagnostics.
    nonnegative[nonnegative < 0.0] = 0.0
    return nonnegative, success, linear_solves, int(np.count_nonzero(passive)), message


def solve_mixed_least_squares(
    free_matrix: Sequence[Sequence[float]] | Array,
    nonnegative_matrix: Sequence[Sequence[float]] | Array,
    rhs: Sequence[float] | Array,
    *,
    rcond: float | None = None,
    relative_tolerance: float | None = None,
    absolute_tolerance: float = 0.0,
    maximum_iterations: int | None = None,
) -> MixedLeastSquaresResult:
    """Solve a least-squares problem with free and nonnegative variables.

    Parameters
    ----------
    free_matrix:
        Matrix multiplying unrestricted variables.
    nonnegative_matrix:
        Matrix multiplying variables constrained to be nonnegative.
    rhs:
        Right-hand side in ``F y + N z ~= rhs``.

    The free-variable subspace is eliminated with a rank-revealing SVD.  This
    avoids mixing free variables into the NNLS active set and handles dependent
    equality Jacobian rows without normal equations.
    """
    rhs_array = np.asarray(rhs, dtype=float)
    if rhs_array.ndim != 1 or not np.all(np.isfinite(rhs_array)):
        raise ValueError("rhs must be a finite one-dimensional vector")
    rows = int(rhs_array.size)
    free = _as_matrix(free_matrix, rows, "free_matrix")
    nonnegative = _as_matrix(nonnegative_matrix, rows, "nonnegative_matrix")

    if rcond is None:
        rcond = _default_rcond(rows, max(free.shape[1], nonnegative.shape[1]))
    rcond = float(rcond)
    if not math.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and nonnegative")
    if relative_tolerance is None:
        relative_tolerance = 256.0 * np.finfo(float).eps * max(
            1, rows, free.shape[1] + nonnegative.shape[1]
        )
    relative_tolerance = float(relative_tolerance)
    absolute_tolerance = float(absolute_tolerance)
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
    ):
        raise ValueError("least-squares tolerances must be finite and nonnegative")

    free_columns = int(free.shape[1])
    if free_columns:
        # full_matrices=True supplies an explicit orthogonal-complement basis.
        # Projecting with U_perp.T avoids the cancellation in N-U(U.T N).
        u, singular_values, vh = np.linalg.svd(free, full_matrices=True)
        if singular_values.size:
            singular_threshold = rcond * float(singular_values[0])
            free_rank = int(np.count_nonzero(singular_values > singular_threshold))
        else:
            free_rank = 0
        complement = u[:, free_rank:]
        projected_rhs = complement.T @ rhs_array
        projected_nonnegative = complement.T @ nonnegative
    else:
        singular_values = np.empty(0, dtype=float)
        vh = np.empty((0, 0), dtype=float)
        u = np.empty((rows, 0), dtype=float)
        free_rank = 0
        projected_rhs = rhs_array
        projected_nonnegative = nonnegative

    z, nnls_success, iterations, passive_variables, message = _nnls_active_set(
        projected_nonnegative,
        projected_rhs,
        rcond=rcond,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_iterations=maximum_iterations,
    )

    rhs_after_nonnegative = rhs_array - nonnegative @ z
    if free_columns:
        # Re-solve in the original coordinates.  This costs one additional SVD
        # but avoids loss of accuracy from explicitly applying inverse singular
        # values when the free matrix has strongly nonuniform column scales.
        free_solution = np.linalg.lstsq(
            free, rhs_after_nonnegative, rcond=rcond
        )[0]
    else:
        free_solution = np.empty(0, dtype=float)

    residual = free @ free_solution + nonnegative @ z - rhs_array
    residual_norm = float(np.linalg.norm(residual))
    if free_columns and free_rank:
        # Iterative refinement is especially effective when the equality
        # Jacobian has strongly nonuniform singular values.  Only residual-
        # reducing corrections are accepted, so a poorly resolved correction
        # cannot degrade the multiplier estimate.
        for _ in range(3):
            correction = np.linalg.lstsq(
                free, -residual, rcond=rcond
            )[0]
            candidate_free = free_solution + correction
            candidate_residual = (
                free @ candidate_free + nonnegative @ z - rhs_array
            )
            candidate_norm = float(np.linalg.norm(candidate_residual))
            if not candidate_norm < residual_norm:
                break
            free_solution = candidate_free
            residual = candidate_residual
            residual_norm = candidate_norm

    # Verify the original mixed problem, not only the projected NNLS system.
    scale = max(1.0, float(np.linalg.norm(rhs_array)), residual_norm)
    verification = max(
        absolute_tolerance,
        32.0 * relative_tolerance * scale,
        math.sqrt(np.finfo(float).eps)
        * max(1, rows, free.shape[1] + nonnegative.shape[1])
        * scale,
    )
    free_norms = _stable_column_norms(free)
    nonnegative_norms = _stable_column_norms(nonnegative)
    free_nonzero = free_norms > 0.0
    nonnegative_nonzero = nonnegative_norms > 0.0
    free_scaled_gradient = np.zeros(free_columns, dtype=float)
    if np.any(free_nonzero):
        free_scaled_gradient[free_nonzero] = (
            free[:, free_nonzero].T @ residual
        ) / free_norms[free_nonzero]
    free_stationarity = float(
        np.max(np.abs(free_scaled_gradient), initial=0.0)
    )
    nonnegative_scaled_gradient = np.zeros(nonnegative.shape[1], dtype=float)
    if np.any(nonnegative_nonzero):
        nonnegative_scaled_gradient[nonnegative_nonzero] = (
            nonnegative[:, nonnegative_nonzero].T @ residual
        ) / nonnegative_norms[nonnegative_nonzero]
    # Activity is judged in normalized-column coordinates: z_j*||N_j|| is
    # the coefficient seen by the active-set solver and is invariant to a
    # harmless rescaling of a constraint Jacobian column.
    positive = z * nonnegative_norms > verification
    positive_stationarity = (
        float(np.max(np.abs(nonnegative_scaled_gradient[positive]), initial=0.0))
        if np.any(positive)
        else 0.0
    )
    zero_dual_violation = (
        float(max(0.0, -np.min(nonnegative_scaled_gradient[~positive], initial=math.inf)))
        if np.any(~positive)
        else 0.0
    )
    original_verified = (
        np.all(np.isfinite(free_solution))
        and np.all(np.isfinite(z))
        and np.all(z >= 0.0)
        and free_stationarity <= verification
        and positive_stationarity <= verification
        and zero_dual_violation <= verification
    )
    success = bool(nnls_success and original_verified)
    if not original_verified:
        message = (
            "mixed least-squares result failed original-space verification "
            f"(free={free_stationarity:.3e}, "
            f"positive={positive_stationarity:.3e}, "
            f"zero_dual={zero_dual_violation:.3e})"
        )

    return MixedLeastSquaresResult(
        np.asarray(free_solution, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(residual, dtype=float),
        residual_norm,
        success,
        iterations,
        passive_variables,
        free_rank,
        message,
    )


def _normalize_bounds_for_kkt(bounds: BoxBounds | None, width: int) -> BoxBounds | None:
    if bounds is None:
        return None
    if bounds.size != width:
        raise ValueError("bounds length does not match variable count")
    return bounds


def estimate_kkt_residual(
    x: Sequence[float],
    objective_gradient: Sequence[float],
    inequalities: Sequence[float],
    inequality_jacobian: Sequence[Sequence[float]],
    equalities: Sequence[float],
    equality_jacobian: Sequence[Sequence[float]],
    bounds: BoxBounds | None,
    *,
    active_tolerance: float = 1.0e-7,
    active_inequality_mask: Sequence[bool] | None = None,
    active_lower_bound_mask: Sequence[bool] | None = None,
    active_upper_bound_mask: Sequence[bool] | None = None,
) -> KKTReport:
    """Estimate first-order stationarity using mixed multiplier least squares."""
    x_array = np.asarray(x, dtype=float)
    gradient = np.asarray(objective_gradient, dtype=float)
    inequalities_array = np.asarray(inequalities, dtype=float)
    inequality_jacobian_array = np.asarray(inequality_jacobian, dtype=float)
    equalities_array = np.asarray(equalities, dtype=float)
    equality_jacobian_array = np.asarray(equality_jacobian, dtype=float)

    if x_array.ndim != 1 or gradient.shape != x_array.shape:
        raise ValueError("x and objective_gradient must be matching vectors")
    width = int(x_array.size)
    if inequality_jacobian_array.shape != (inequalities_array.size, width):
        raise ValueError("inequality Jacobian has wrong shape")
    if equality_jacobian_array.shape != (equalities_array.size, width):
        raise ValueError("equality Jacobian has wrong shape")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            x_array,
            gradient,
            inequalities_array,
            inequality_jacobian_array,
            equalities_array,
            equality_jacobian_array,
        )
    ):
        raise FloatingPointError("KKT inputs must contain only finite values")
    if not math.isfinite(active_tolerance) or active_tolerance < 0.0:
        raise ValueError("active_tolerance must be finite and nonnegative")

    if active_inequality_mask is None:
        active = np.flatnonzero(inequalities_array >= -active_tolerance)
    else:
        mask = np.asarray(active_inequality_mask, dtype=bool)
        if mask.shape != inequalities_array.shape:
            raise ValueError("active_inequality_mask has wrong shape")
        active = np.flatnonzero(mask)

    active_lower = np.empty(0, dtype=np.intp)
    active_upper = np.empty(0, dtype=np.intp)
    normalized_bounds = _normalize_bounds_for_kkt(bounds, width)
    if normalized_bounds is not None:
        lower = normalized_bounds.lower
        upper = normalized_bounds.upper
        finite_lower = np.flatnonzero(np.isfinite(lower))
        finite_upper = np.flatnonzero(np.isfinite(upper))
        if active_lower_bound_mask is None:
            active_lower = finite_lower[
                x_array[finite_lower]
                <= lower[finite_lower]
                + active_tolerance * np.maximum(1.0, np.abs(lower[finite_lower]))
            ]
        else:
            lower_mask = np.asarray(active_lower_bound_mask, dtype=bool)
            if lower_mask.shape != x_array.shape:
                raise ValueError("active_lower_bound_mask has wrong shape")
            active_lower = finite_lower[lower_mask[finite_lower]]
        if active_upper_bound_mask is None:
            active_upper = finite_upper[
                x_array[finite_upper]
                >= upper[finite_upper]
                - active_tolerance * np.maximum(1.0, np.abs(upper[finite_upper]))
            ]
        else:
            upper_mask = np.asarray(active_upper_bound_mask, dtype=bool)
            if upper_mask.shape != x_array.shape:
                raise ValueError("active_upper_bound_mask has wrong shape")
            active_upper = finite_upper[upper_mask[finite_upper]]

    nonnegative_columns = int(active.size + active_lower.size + active_upper.size)
    if nonnegative_columns:
        nonnegative_matrix = np.zeros((width, nonnegative_columns), dtype=float)
        column = 0
        if active.size:
            next_column = column + active.size
            nonnegative_matrix[:, column:next_column] = inequality_jacobian_array[active].T
            column = next_column
        if active_lower.size:
            next_column = column + active_lower.size
            nonnegative_matrix[active_lower, np.arange(column, next_column)] = -1.0
            column = next_column
        if active_upper.size:
            next_column = column + active_upper.size
            nonnegative_matrix[active_upper, np.arange(column, next_column)] = 1.0
    else:
        nonnegative_matrix = np.empty((width, 0), dtype=float)

    free_matrix = (
        equality_jacobian_array.T
        if equalities_array.size
        else np.empty((width, 0), dtype=float)
    )
    if free_matrix.shape[1] == 0 and nonnegative_matrix.shape[1] == 0:
        return KKTReport(
            float(np.linalg.norm(gradient, ord=np.inf)),
            0,
            0,
            0,
            True,
        )

    result = solve_mixed_least_squares(
        free_matrix,
        nonnegative_matrix,
        -gradient,
    )
    stationarity = float(np.linalg.norm(result.residual, ord=np.inf))
    return KKTReport(
        stationarity,
        int(active.size),
        int(active_lower.size),
        int(active_upper.size),
        bool(result.success),
    )


__all__ = [
    "BoxBounds",
    "KKTReport",
    "MixedLeastSquaresResult",
    "estimate_kkt_residual",
    "solve_mixed_least_squares",
]
