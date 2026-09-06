"""Reduced-to-full exact route-geometry prolongation utilities.

A ``maximal_runs`` turn uses two linear-curvature segments.  The full
``overlapping_cover`` representation uses four segments so the transition from
run cells to the explicit turn cell can occur inside a two-dimensional overlap.
Given a conditioning guard fraction ``q``, each reduced turn half is subdivided
exactly at ``1-q`` (incoming) or ``q`` (outgoing).  Linear curvature and station
interpolation preserve the continuous clothoid path exactly.

The returned full-basis point is *not* assumed feasible merely because the guard
was enforced.  Every promotion must still pass the ordinary authoritative
continuous corridor certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from planning.maze_routes import RouteOptimizationProblem

Array = NDArray[np.float64]
Cell = tuple[int, int]


def turn_indices(cells: Sequence[Cell]) -> tuple[int, ...]:
    directions = [
        (b[0] - a[0], b[1] - a[1]) for a, b in zip(cells[:-1], cells[1:])
    ]
    return tuple(
        i for i in range(1, len(cells) - 1) if directions[i - 1] != directions[i]
    )


def coarse_segment_route_indices(
    cells: Sequence[Cell], *, goal_entry: bool = True
) -> tuple[int, ...]:
    result: list[int] = [0]
    directions = [
        (b[0] - a[0], b[1] - a[1]) for a, b in zip(cells[:-1], cells[1:])
    ]
    for i in range(1, len(cells) - 1):
        if directions[i - 1] == directions[i]:
            result.append(i)
        else:
            result.extend((i, i))
    result.append(len(cells) - 1)
    if goal_entry:
        result.pop()
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ProlongationResult:
    parameters: Array
    split_fractions: tuple[float | None, ...]
    minimum_child_length: float
    maximum_child_length_ratio: float


def prolong_reduced_to_full(
    cells: Sequence[Cell],
    reduced_problem: RouteOptimizationProblem,
    full_problem: RouteOptimizationProblem,
    reduced_parameters: Sequence[float],
    *,
    guard_fraction: float,
) -> ProlongationResult:
    """Exactly prolong one goal-entry ``maximal_runs`` curve to full cover basis."""
    q = float(guard_fraction)
    if not math.isfinite(q) or not 0.0 < q < 0.5:
        raise ValueError("guard_fraction must lie strictly between 0 and 0.5")
    cells = tuple(cells)
    if reduced_problem.cells != cells or full_problem.cells != cells:
        raise ValueError("prolongation problems do not match topology")
    x = np.asarray(reduced_parameters, dtype=float)
    if x.ndim != 1 or x.size != 2 * reduced_problem.corridor.n_segments:
        raise ValueError("reduced parameter dimension mismatch")

    route_indices = coarse_segment_route_indices(cells, goal_entry=True)
    if len(route_indices) != reduced_problem.corridor.n_segments:
        raise ValueError("reduced problem is not the expected goal-entry maximal-runs basis")
    turns = set(turn_indices(cells))
    by_route: dict[int, list[int]] = {}
    for segment, route_index in enumerate(route_indices):
        by_route.setdefault(route_index, []).append(segment)
    for route_index in turns:
        if len(by_route.get(route_index, ())) != 2:
            raise ValueError("every turn must own exactly two reduced segments")

    stations = x[0::2]
    curvatures = x[1::2]
    previous_stations = np.concatenate(([0.0], stations[:-1]))
    previous_curvatures = np.concatenate(
        ([reduced_problem.initial_state.k], curvatures[:-1])
    )

    out: list[float] = []
    split_fractions: list[float | None] = []
    child_lengths: list[float] = []
    child_parent_ratios: list[float] = []
    full_station = 0.0
    turn_owned_position: dict[int, int] = {k: 0 for k in turns}

    for segment, route_index in enumerate(route_indices):
        s0 = float(previous_stations[segment])
        s1 = float(stations[segment])
        k0 = float(previous_curvatures[segment])
        k1 = float(curvatures[segment])
        length = s1 - s0
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("reduced path contains nonpositive segment length")
        if route_index not in turns:
            full_station += length
            out.extend((full_station, k1))
            split_fractions.append(None)
            child_lengths.append(length)
            child_parent_ratios.append(1.0)
            continue

        position = turn_owned_position[route_index]
        turn_owned_position[route_index] = position + 1
        if position == 0:  # incoming half: reserve q at the end for turn cell
            split = 1.0 - q
        elif position == 1:  # outgoing half: reserve q at the start for turn cell
            split = q
        else:  # defensive
            raise ValueError("unexpected third reduced segment for one turn")
        ks = math.fma(split, k1 - k0, k0)
        first_length = split * length
        second_length = (1.0 - split) * length
        full_station += first_length
        out.extend((full_station, ks))
        full_station += second_length
        out.extend((full_station, k1))
        split_fractions.append(split)
        child_lengths.extend((first_length, second_length))
        child_parent_ratios.extend((split, 1.0 - split))

    array = np.asarray(out, dtype=float)
    if array.size != 2 * full_problem.corridor.n_segments:
        raise ValueError(
            f"prolongation produced {array.size // 2} segments; full problem expects "
            f"{full_problem.corridor.n_segments}"
        )
    return ProlongationResult(
        parameters=array,
        split_fractions=tuple(split_fractions),
        minimum_child_length=float(min(child_lengths)),
        maximum_child_length_ratio=float(max(child_parent_ratios)),
    )


__all__ = [
    "ProlongationResult",
    "coarse_segment_route_indices",
    "prolong_reduced_to_full",
    "turn_indices",
]
