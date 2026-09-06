"""Combined time, geometry, and curvature gradients in a flat knot basis.

Knot layout
-----------
``knot_params = [s1, k1, s2, k2, ..., sn, kn]``

The fixed initial knot is supplied by ``initial_state``:
``s0 = initial_s`` and ``k0 = initial_state.k``.  The corresponding raw speed
and geometry parameters are flattened as
``[L0, sigma0, L1, sigma1, ...]``.

The scalar objective is

``time_weight * time + geometry_weight * geometry + curvature_weight * integral(k^2 ds)``.

All three weights must be finite and nonnegative.  A zero weight disables that
objective channel.  The three raw gradients are accumulated before one pullback
to the knot basis.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Sequence, TypeAlias

from . import reverse_solver
from .geometry_gradients import (
    GeometryPath,
    GeometryState,
    compile_geometry_path,
    knot_parameters_to_raw,
    pullback_raw_gradient_to_knot_parameters,
)

Float4: TypeAlias = tuple[float, float, float, float]
EndpointObjective: TypeAlias = Callable[[GeometryState], tuple[float, Sequence[float]]]
GeometryObjective: TypeAlias = Callable[[GeometryPath], tuple[float, Sequence[float]]]


class TotalGradientResult(NamedTuple):
    """Result of one combined knot-basis objective evaluation.

    ``raw_gradient`` and ``gradient`` are the weighted total gradients.
    Individual unweighted channel values and raw gradients are also returned.
    A disabled channel has value zero and an all-zero gradient.
    """

    value: float
    gradient: list[float]
    time_value: float
    geometry_value: float
    final_state: GeometryState
    raw_params: list[float]
    raw_gradient: list[float]
    raw_time_gradient: list[float]
    raw_geometry_gradient: list[float]
    curvature_value: float
    raw_curvature_gradient: list[float]


def _checked_state(initial_state: GeometryState | Sequence[float]) -> GeometryState:
    if len(initial_state) != 4:
        raise ValueError("initial_state must be (x0, y0, theta0, k0)")
    state = GeometryState(*map(float, initial_state))
    if not all(math.isfinite(value) for value in state):
        raise ValueError("initial_state must be finite")
    return state


def _checked_weight(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _zero_endpoint_objective(_state: GeometryState) -> tuple[float, Float4]:
    return 0.0, (0.0, 0.0, 0.0, 0.0)



def path_length_value_and_raw_gradient(
    path: GeometryPath,
) -> tuple[float, list[float]]:
    """Return total arclength and its raw ``[L, sigma, ...]`` gradient."""
    value = math.fsum(float(length) for length in path.lengths)
    gradient = [
        component
        for _ in path.lengths
        for component in (1.0, 0.0)
    ]
    return value, gradient

def curvature_energy_value_and_raw_gradient(
    raw_params: Sequence[float],
    *,
    initial_k: float = 0.0,
) -> tuple[float, list[float]]:
    """Evaluate ``integral(k(s)^2 ds)`` in the raw ``[L, sigma]`` basis.

    Curvature on segment ``i`` is

    ``k(s) = k_i + sigma_i*s``, ``0 <= s <= L_i``,

    with ``k_{i+1} = k_i + sigma_i*L_i``.  The value and gradient are computed
    in two linear passes.  ``initial_k`` is fixed and therefore has no returned
    derivative.
    """
    size = len(raw_params)
    if size == 0 or size % 2:
        raise ValueError("raw_params must have layout [L0, sigma0, ..., Ln, sigman]")

    initial_k = float(initial_k)
    if not math.isfinite(initial_k):
        raise ValueError("initial_k must be finite")

    n_segments = size // 2
    raw = [float(value) for value in raw_params]
    if not all(math.isfinite(value) for value in raw):
        raise ValueError("raw parameters must be finite")

    curvatures = [0.0] * (n_segments + 1)
    curvatures[0] = initial_k
    terms = [0.0] * n_segments

    for i in range(n_segments):
        length = raw[2 * i]
        sigma = raw[2 * i + 1]
        if length <= 0.0:
            raise ValueError("segment lengths must be strictly positive")
        k0 = curvatures[i]
        k1 = math.fma(sigma, length, k0)
        if not math.isfinite(k1):
            raise FloatingPointError("curvature recurrence produced a nonfinite value")
        curvatures[i + 1] = k1
        shape = math.fma(k0, k0, math.fma(k0, k1, k1 * k1))
        terms[i] = (length / 3.0) * shape

    gradient = [0.0] * size
    adjoint_k1 = 0.0
    for i in range(n_segments - 1, -1, -1):
        length = raw[2 * i]
        sigma = raw[2 * i + 1]
        k0 = curvatures[i]
        k1 = curvatures[i + 1]

        # Direct segment derivatives with k0 held fixed, followed by the
        # recurrence contribution k1 = k0 + sigma*length.
        direct_sigma = (length * length / 3.0) * math.fma(2.0, k1, k0)
        gradient[2 * i] = math.fma(adjoint_k1, sigma, k1 * k1)
        gradient[2 * i + 1] = math.fma(adjoint_k1, length, direct_sigma)

        direct_k0 = length * (k0 + k1)
        adjoint_k1 += direct_k0

    value = math.fsum(terms)
    if not math.isfinite(value) or not all(math.isfinite(v) for v in gradient):
        raise FloatingPointError("curvature-energy evaluation produced nonfinite data")
    return value, gradient


def curvature_energy_value_and_gradient(
    knot_params: Sequence[float],
    *,
    initial_s: float = 0.0,
    initial_k: float = 0.0,
) -> tuple[float, list[float]]:
    """Evaluate curvature energy and its exact flat knot-basis gradient."""
    initial_s = float(initial_s)
    initial_k = float(initial_k)
    if not math.isfinite(initial_s) or not math.isfinite(initial_k):
        raise ValueError("initial_s and initial_k must be finite")
    raw = knot_parameters_to_raw(
        knot_params,
        initial_k=initial_k,
        initial_s=initial_s,
    )
    value, raw_gradient = curvature_energy_value_and_raw_gradient(
        raw,
        initial_k=initial_k,
    )
    gradient = pullback_raw_gradient_to_knot_parameters(
        knot_params,
        raw_gradient,
        initial_k=initial_k,
        initial_s=initial_s,
    )
    return value, gradient


def linear_endpoint_objective(seed: Sequence[float]) -> EndpointObjective:
    """Create ``g(state)=seed dot (x,y,theta,k)`` and its exact seed."""
    if len(seed) != 4:
        raise ValueError("endpoint seed must have length four")
    ax, ay, atheta, ak = map(float, seed)
    if not all(math.isfinite(value) for value in (ax, ay, atheta, ak)):
        raise ValueError("endpoint seed must be finite")

    def objective(state: GeometryState) -> tuple[float, Float4]:
        value = math.fma(
            ax,
            state.x,
            math.fma(ay, state.y, math.fma(atheta, state.theta, ak * state.k)),
        )
        return value, (ax, ay, atheta, ak)

    return objective


def quadratic_endpoint_target(
    target: Sequence[float],
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> EndpointObjective:
    """Create a weighted half-squared final-state target objective."""
    if len(target) != 4 or len(weights) != 4:
        raise ValueError("target and weights must each have length four")
    tx, ty, tt, tk = map(float, target)
    wx, wy, wt, wk = map(float, weights)
    if not all(math.isfinite(value) for value in (tx, ty, tt, tk, wx, wy, wt, wk)):
        raise ValueError("target and weights must be finite")
    if min(wx, wy, wt, wk) < 0.0:
        raise ValueError("endpoint weights must be nonnegative")

    def objective(state: GeometryState) -> tuple[float, Float4]:
        dx = state.x - tx
        dy = state.y - ty
        dtheta = state.theta - tt
        dk = state.k - tk
        seed = (wx * dx, wy * dy, wt * dtheta, wk * dk)
        value = 0.5 * math.fsum(
            (wx * dx * dx, wy * dy * dy, wt * dtheta * dtheta, wk * dk * dk)
        )
        return value, seed

    return objective


def total_knot_value_and_gradient(
    knot_params: Sequence[float],
    initial_state: GeometryState | Sequence[float],
    *,
    initial_s: float = 0.0,
    init_w: float | None = None,
    time_weight: float = 1.0,
    geometry_weight: float = 1.0,
    curvature_weight: float = 0.0,
    endpoint_objective: EndpointObjective | None = None,
    geometry_objective: GeometryObjective | None = None,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = reverse_solver.FRICTION_DOMAIN_MARGIN,
    domain_scan: int = reverse_solver.FRICTION_DOMAIN_SCAN,
    profiler: reverse_solver.PhaseProfiler | None = None,
) -> TotalGradientResult:
    """Evaluate the weighted objective and exact flat knot-basis gradient.

    The objective is

    ``time_weight*T + geometry_weight*G + curvature_weight*E_k``

    where ``E_k = integral(k(s)^2 ds)``.  Weights are independent finite
    nonnegative coefficients; they are not required to sum to one.  A zero
    weight disables that channel and its callback/evaluator is not invoked.

    ``endpoint_objective`` returns ``(value, dvalue/d(x,y,theta,k))`` at the
    final state.  For a more general path-dependent term, ``geometry_objective``
    may instead return ``(value, flat_raw_gradient)`` from the compiled geometry
    tape.  The two callbacks are mutually exclusive.  All active gradients are
    accumulated in the common raw basis and pulled back once.
    """
    state0 = _checked_state(initial_state)
    initial_s = float(initial_s)
    if not math.isfinite(initial_s):
        raise ValueError("initial_s must be finite")
    time_weight = _checked_weight(time_weight, "time_weight")
    geometry_weight = _checked_weight(geometry_weight, "geometry_weight")
    curvature_weight = _checked_weight(curvature_weight, "curvature_weight")

    if endpoint_objective is not None and geometry_objective is not None:
        raise ValueError("endpoint_objective and geometry_objective are mutually exclusive")

    raw = knot_parameters_to_raw(
        knot_params,
        initial_k=state0.k,
        initial_s=initial_s,
    )
    zero_raw = [0.0] * len(raw)

    if time_weight > 0.0:
        time_value, raw_time_gradient_in = reverse_solver.time_value_and_gradient(
            raw,
            init_w=init_w,
            initial_k=state0.k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_margin=domain_margin,
            domain_scan=domain_scan,
            profiler=profiler,
        )
        time_value = float(time_value)
        raw_time_gradient = [float(v) for v in raw_time_gradient_in]
    else:
        time_value = 0.0
        raw_time_gradient = zero_raw.copy()

    # Geometry is compiled once because final_state is part of the public result
    # even when the scalar geometry channel is disabled.
    geometry_path: GeometryPath = compile_geometry_path(raw, state0)
    final_state = geometry_path.final_state
    if geometry_weight > 0.0:
        if geometry_objective is not None:
            geometry_value, raw_geometry_gradient_in = geometry_objective(geometry_path)
            geometry_value = float(geometry_value)
            raw_geometry_gradient = [float(v) for v in raw_geometry_gradient_in]
            if len(raw_geometry_gradient) != len(raw):
                raise ValueError("geometry objective returned a raw gradient with the wrong length")
            if not math.isfinite(geometry_value) or not all(
                math.isfinite(v) for v in raw_geometry_gradient
            ):
                raise ValueError("geometry objective returned nonfinite data")
        else:
            objective = endpoint_objective or _zero_endpoint_objective
            geometry_value, endpoint_seed = objective(final_state)
            geometry_value = float(geometry_value)
            if len(endpoint_seed) != 4:
                raise ValueError("endpoint objective must return a seed of length four")
            endpoint_seed4 = tuple(map(float, endpoint_seed))
            if not math.isfinite(geometry_value) or not all(
                math.isfinite(v) for v in endpoint_seed4
            ):
                raise ValueError("endpoint objective returned nonfinite data")
            raw_geometry_gradient = geometry_path.endpoint_vjp(endpoint_seed4)  # type: ignore[arg-type]
    else:
        geometry_value = 0.0
        raw_geometry_gradient = zero_raw.copy()

    if curvature_weight > 0.0:
        curvature_value, raw_curvature_gradient = curvature_energy_value_and_raw_gradient(
            raw,
            initial_k=state0.k,
        )
    else:
        curvature_value = 0.0
        raw_curvature_gradient = zero_raw.copy()

    raw_gradient = [
        math.fma(
            time_weight,
            gt,
            math.fma(geometry_weight, gg, curvature_weight * gc),
        )
        for gt, gg, gc in zip(
            raw_time_gradient,
            raw_geometry_gradient,
            raw_curvature_gradient,
        )
    ]
    gradient = pullback_raw_gradient_to_knot_parameters(
        knot_params,
        raw_gradient,
        initial_k=state0.k,
        initial_s=initial_s,
    )
    value = math.fma(
        time_weight,
        time_value,
        math.fma(geometry_weight, geometry_value, curvature_weight * curvature_value),
    )

    return TotalGradientResult(
        value,
        gradient,
        time_value,
        geometry_value,
        final_state,
        raw,
        raw_gradient,
        raw_time_gradient,
        raw_geometry_gradient,
        curvature_value,
        raw_curvature_gradient,
    )


__all__ = [
    "EndpointObjective",
    "GeometryObjective",
    "TotalGradientResult",
    "curvature_energy_value_and_gradient",
    "curvature_energy_value_and_raw_gradient",
    "linear_endpoint_objective",
    "path_length_value_and_raw_gradient",
    "quadratic_endpoint_target",
    "total_knot_value_and_gradient",
]
