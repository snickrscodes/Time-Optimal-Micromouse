from __future__ import annotations

import numpy as np

from benchmarks.common.certification import certify_parameters_on_problem
from optimization import compile_geometry_path, knot_parameters_to_raw
from tools.geometry_homotopy.basis_activation import (
    SegmentLengthFloorConstraint,
    activate_newly_supported_children,
    analyze_transition_support,
    build_selective_basis,
)
from tools.geometry_homotopy.homotopy import turn_indices
from tools.geometry_homotopy.research_common import build_historical_pair


def _final_state(problem, x):
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    return compile_geometry_path(raw, problem.initial_state).final_state


def test_support_analysis_and_selective_basis_preserve_curve() -> None:
    cells, reduced, full = build_historical_pair(0)
    x = reduced.initial_parameters.copy()
    support = analyze_transition_support(cells, reduced, full, x)
    assert len(support) == 2 * len(turn_indices(cells))
    assert all(row.support_length >= 0.0 for row in support)

    selected = build_selective_basis(
        cells,
        reduced,
        full,
        x,
        minimum_active_child_length=1.0e-4,
        support_safety_fraction=0.8,
        activation_granularity="turn",
    )
    before = _final_state(reduced, x)
    after = _final_state(selected.problem, selected.parameters)
    np.testing.assert_allclose(
        [before.x, before.y, before.theta, before.k],
        [after.x, after.y, after.theta, after.k],
        rtol=0.0,
        atol=2.0e-12,
    )
    cert = certify_parameters_on_problem(
        selected.problem,
        selected.parameters,
        tolerance=2.0e-7,
        maximum_abs_sigma=50.0,
    )
    assert cert["certified"]


def test_length_floor_constraint_jacobian() -> None:
    constraint = SegmentLengthFloorConstraint((1, 3), 0.05)
    x = np.array([0.4, 0.0, 0.9, 0.1, 1.2, 0.2, 1.7, 0.0], dtype=float)
    values, jac = constraint(x)
    direction = np.array([0.2, 0.0, -0.1, 0.0, 0.3, 0.0, -0.2, 0.0])
    eps = 1.0e-7
    fd = (constraint(x + eps * direction)[0] - constraint(x - eps * direction)[0]) / (2.0 * eps)
    np.testing.assert_allclose(jac @ direction, fd, rtol=0.0, atol=2.0e-9)
    assert values.shape == (2,)


def test_dynamic_activation_does_not_force_unsupported_children() -> None:
    cells, reduced, full = build_historical_pair(3)
    x = reduced.initial_parameters.copy()
    selected = build_selective_basis(
        cells,
        reduced,
        full,
        x,
        minimum_active_child_length=0.1,
        support_safety_fraction=0.8,
        activation_granularity="turn",
    )
    refreshed = activate_newly_supported_children(
        selected,
        selected.parameters,
        minimum_active_child_length=0.1,
        support_safety_fraction=0.8,
    )
    assert refreshed.problem.corridor.n_segments == selected.problem.corridor.n_segments
    np.testing.assert_array_equal(refreshed.parameters, selected.parameters)
