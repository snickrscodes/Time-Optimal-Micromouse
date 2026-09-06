"""Authoritative continuous route certification shared by planner and benchmarks."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from optimization import compile_geometry_path, knot_parameters_to_raw, separate_rectangle_path


def certify_parameters(
    problem: Any,
    parameters: Sequence[float],
    *,
    tolerance: float,
    maximum_abs_sigma: float | None = None,
):
    raw = knot_parameters_to_raw(parameters, initial_k=problem.initial_state.k)
    geometry = compile_geometry_path(raw, problem.initial_state)
    final = geometry.final_state
    endpoint_error = problem.terminal_violation(final)
    separation = separate_rectangle_path(
        parameters, problem.initial_state, problem.corridor
    )
    slope_feasible = (
        True
        if maximum_abs_sigma is None
        else max(abs(float(value)) for value in raw[1::2])
        <= maximum_abs_sigma + tolerance
    )
    feasible = (
        endpoint_error <= tolerance
        and separation.certified(tolerance)
        and slope_feasible
        and np.all(np.isfinite(parameters))
    )
    return (
        bool(feasible),
        final,
        float(endpoint_error),
        float(separation.worst_upper_bound),
    )


def certify_parameters_on_problem(
    problem: Any,
    parameters: Sequence[float],
    *,
    tolerance: float,
    maximum_abs_sigma: float | None,
) -> dict[str, Any]:
    feasible, final, endpoint_error, corridor_upper = certify_parameters(
        problem,
        parameters,
        tolerance=tolerance,
        maximum_abs_sigma=maximum_abs_sigma,
    )
    return {
        "certified": bool(feasible),
        "endpoint_error": float(endpoint_error),
        "corridor_upper_bound": float(corridor_upper),
        "final_state": [float(v) for v in final],
    }


__all__ = ["certify_parameters", "certify_parameters_on_problem"]
