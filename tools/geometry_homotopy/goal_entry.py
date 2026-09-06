"""Experimental goal-entry route problems.

The production builder currently appends the historical half-cell segment from
an entry edge to the goal-cell centre, while :class:`RouteOptimizationProblem`
already constrains the terminal state to that entry edge.  This helper trims the
redundant terminal segment without changing any other corridor or endpoint
semantics.  It is kept out of the production planner until the homotopy research
campaign is complete.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal, Sequence

from optimization import CorridorModel
from planning.maze_routes import (
    DEFAULT_BODY_HEIGHT,
    DEFAULT_BODY_LENGTH,
    OpenRoomSpan,
    RouteOptimizationProblem,
    build_route_optimization_problem,
)

Cell = tuple[int, int]


def as_goal_entry_problem(
    problem: RouteOptimizationProblem,
    *,
    terminal_segments: int = 1,
) -> RouteOptimizationProblem:
    """Return ``problem`` with its redundant terminal centre segment removed."""
    n = int(terminal_segments)
    if n <= 0 or n >= problem.corridor.n_segments:
        raise ValueError("terminal_segments must remove a proper nonempty suffix")
    parameters = problem.initial_parameters[: -2 * n].copy()
    corridor = CorridorModel(
        problem.corridor.cells,
        problem.corridor.segment_cells[:-n],
        problem.corridor.body,
        problem.corridor.clearance,
    )
    return replace(problem, initial_parameters=parameters, corridor=corridor)


def build_goal_entry_problem(
    cells: Sequence[Cell],
    *,
    body_length: float = DEFAULT_BODY_LENGTH,
    body_height: float = DEFAULT_BODY_HEIGHT,
    body_width: float | None = None,
    clearance: float = 0.0,
    corridor_mode: Literal["overlapping_cover", "maximal_runs", "per_cell"] = "overlapping_cover",
    refinement_factor: int = 1,
    open_room_spans: Sequence[OpenRoomSpan] = (),
) -> RouteOptimizationProblem:
    """Build a route problem whose geometry terminates on the goal-entry edge."""
    problem = build_route_optimization_problem(
        cells,
        body_length=body_length,
        body_height=body_height,
        body_width=body_width,
        clearance=clearance,
        corridor_mode=corridor_mode,
        refinement_factor=refinement_factor,
        open_room_spans=open_room_spans,
    )
    # The historical terminal half-cell is itself split by refinement_factor.
    return as_goal_entry_problem(problem, terminal_segments=int(refinement_factor))


__all__ = ["as_goal_entry_problem", "build_goal_entry_problem"]
