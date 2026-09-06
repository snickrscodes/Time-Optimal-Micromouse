"""Conditioning constraints for reduced→full route homotopy.

The reduced ``maximal_runs`` basis is intentionally permissive: each Euler-turn
half can move almost completely out of the physical turn cell while remaining
valid in its large incoming/outgoing run rectangle.  That flexibility is useful
for optimization, but a reduced checkpoint that only *touches* the turn-cell
overlap prolongs into the full ``overlapping_cover`` basis with nearly collapsed
children.

``TransitionOverlapConstraint`` prevents that homotopy degeneration without
weakening the physical corridor.  For every turn it requires a point near the
end of the incoming half and a point near the start of the outgoing half to fit
inside the actual turn cell.  A guard fraction ``q`` therefore reserves at least
roughly ``q`` of each half-turn for a well-conditioned exact subdivision.

This is a research/conditioning constraint, not a physical requirement.  Final
full-basis trajectories are still accepted only by the ordinary authoritative
continuous corridor certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import KnotGeometryCache
from planning.maze_routes import RouteOptimizationProblem

Array = NDArray[np.float64]
Cell = tuple[int, int]


def _turn_indices(cells: Sequence[Cell]) -> tuple[int, ...]:
    directions = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(cells[:-1], cells[1:])
    ]
    return tuple(
        i
        for i in range(1, len(cells) - 1)
        if directions[i - 1] != directions[i]
    )


def _coarse_segment_route_indices(cells: Sequence[Cell]) -> tuple[int, ...]:
    result: list[int] = [0]
    directions = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(cells[:-1], cells[1:])
    ]
    for i in range(1, len(cells) - 1):
        if directions[i - 1] == directions[i]:
            result.append(i)
        else:
            result.extend((i, i))
    result.append(len(cells) - 1)
    return tuple(result)


def _turn_cell_by_route_index(full_problem: RouteOptimizationProblem):
    mapping = {}
    for index, cell in enumerate(full_problem.corridor.cells):
        name = cell.name
        if not name.startswith("route_turn_"):
            continue
        fields = name.split("_")
        if len(fields) < 5:
            continue
        mapping[int(fields[2])] = cell
    return mapping


@dataclass(frozen=True, slots=True)
class _GuardPoint:
    route_index: int
    segment: int
    tau: float
    turn_cell: object


@dataclass(slots=True)
class TransitionOverlapConstraint:
    """Reserve finite turn-cell overlap in a reduced maximal-runs geometry."""

    cells: tuple[Cell, ...]
    coarse_problem: RouteOptimizationProblem
    full_problem: RouteOptimizationProblem
    guard_fraction: float = 0.05
    additional_margin: float = 0.0
    _cache: KnotGeometryCache = field(init=False, repr=False)
    _guards: tuple[_GuardPoint, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        q = float(self.guard_fraction)
        if not math.isfinite(q) or not 0.0 < q < 0.5:
            raise ValueError("guard_fraction must lie strictly between 0 and 0.5")
        if not math.isfinite(self.additional_margin) or self.additional_margin < 0.0:
            raise ValueError("additional_margin must be finite and nonnegative")
        if self.coarse_problem.cells != self.cells or self.full_problem.cells != self.cells:
            raise ValueError("transition guard problems do not match the supplied topology")
        route_indices = _coarse_segment_route_indices(self.cells)
        # Goal-entry problems drop the historical final half-cell segment.
        if len(route_indices) == self.coarse_problem.corridor.n_segments + 1:
            route_indices = route_indices[:-1]
        if len(route_indices) != self.coarse_problem.corridor.n_segments:
            raise ValueError("coarse problem is not the expected maximal-runs basis")
        turn_indices = _turn_indices(self.cells)
        turn_cells = _turn_cell_by_route_index(self.full_problem)
        if set(turn_indices) != set(turn_cells):
            raise ValueError("full problem does not expose the expected turn cells")
        by_route: dict[int, list[int]] = {}
        for segment, route_index in enumerate(route_indices):
            by_route.setdefault(route_index, []).append(segment)
        guards: list[_GuardPoint] = []
        for route_index in turn_indices:
            owned = by_route.get(route_index, [])
            if len(owned) != 2:
                raise ValueError("every reduced turn must own exactly two segments")
            incoming, outgoing = owned
            guards.append(_GuardPoint(route_index, incoming, 1.0 - q, turn_cells[route_index]))
            guards.append(_GuardPoint(route_index, outgoing, q, turn_cells[route_index]))
        self._guards = tuple(guards)
        self._cache = KnotGeometryCache(self.coarse_problem.initial_state)

    @property
    def guard_points(self) -> int:
        return len(self._guards)

    @property
    def row_count(self) -> int:
        if not self._guards:
            return 0
        # Current route corridor cells are rectangles with four walls and the
        # body footprint has four corners.  Compute from the actual objects to
        # keep the implementation honest if either changes.
        return sum(
            len(guard.turn_cell.walls) * len(self.coarse_problem.corridor.body.corners)
            for guard in self._guards
        )

    def __call__(self, knot_params: Sequence[float]) -> tuple[Array, Array]:
        x = np.asarray(knot_params, dtype=float)
        path = self._cache.update(x)
        if path.n_segments != self.coarse_problem.corridor.n_segments:
            raise ValueError("transition guard path dimension mismatch")
        corners = self.coarse_problem.corridor.body.corners
        clearance = self.coarse_problem.corridor.clearance + self.additional_margin
        values: list[float] = []
        rows: list[Array] = []
        for guard in self._guards:
            pose = self._cache.point_pose_jacobian(guard.segment, guard.tau)
            theta = pose.state.theta
            c = math.cos(theta)
            s = math.sin(theta)
            for wall in guard.turn_cell.walls:
                nx, ny = wall.normal
                effective_offset = wall.offset - clearance
                for corner in corners:
                    corner_x = pose.state.x + corner.u * c - corner.v * s
                    corner_y = pose.state.y + corner.u * s + corner.v * c
                    values.append(nx * corner_x + ny * corner_y - effective_offset)
                    dcorner_dtheta_x = -corner.u * s - corner.v * c
                    dcorner_dtheta_y = corner.u * c - corner.v * s
                    theta_seed = nx * dcorner_dtheta_x + ny * dcorner_dtheta_y
                    rows.append(nx * pose.jx + ny * pose.jy + theta_seed * pose.jtheta)
        if not values:
            return np.empty(0, dtype=float), np.empty((0, x.size), dtype=float)
        value_array = np.asarray(values, dtype=float)
        jacobian = np.vstack(rows)
        if not np.all(np.isfinite(value_array)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("transition overlap constraint produced nonfinite data")
        return value_array, jacobian

    def maximum_violation(self, knot_params: Sequence[float]) -> float:
        values, _ = self(knot_params)
        return float(np.max(values)) if values.size else -math.inf


__all__ = ["TransitionOverlapConstraint"]
