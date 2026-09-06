"""Development-quality geometry-detail renderer built from reusable layers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from optimization import knot_parameters_to_raw

from .maze import draw_maze_walls
from .sampling import sample_geometry_parameters
from .trajectory import draw_body_footprint, draw_corridor, draw_geometry_trace, draw_topology


def draw_geometry_detail(
    ax: Any,
    *,
    maze: Any,
    problem: Any,
    parameters: Sequence[float],
    cells: Sequence[tuple[int, int]],
    samples_per_unit: float = 120.0,
    minimum_samples_per_segment: int = 16,
    footprint_count: int = 8,
    show_topology: bool = False,
    show_knots: bool = True,
    show_corridor: bool = True,
) -> Any:
    """Draw one optimized geometry with its real corridor/body footprint."""

    draw_maze_walls(ax, maze, linewidths=1.0)
    if show_corridor:
        draw_corridor(ax, problem.corridor)
    if show_topology:
        draw_topology(ax, cells)
    trace = sample_geometry_parameters(
        parameters,
        problem.initial_state,
        samples_per_unit=samples_per_unit,
        minimum_samples_per_segment=minimum_samples_per_segment,
    )
    draw_geometry_trace(ax, trace, linewidth=2.0)
    if show_knots:
        # Evaluate exact segment-boundary stations, independent of display
        # sampling density.
        from .sampling import sample_geometry_stations
        raw_knots = sample_geometry_stations(
            knot_parameters_to_raw(parameters, initial_k=problem.initial_state.k),
            problem.initial_state,
            trace.knot_s,
        )
        ax.scatter(raw_knots.x, raw_knots.y, s=12, zorder=6)
    if footprint_count > 0:
        indices = np.linspace(0, len(trace.s) - 1, footprint_count).round().astype(int)
        for index in np.unique(indices):
            draw_body_footprint(ax, trace.state(int(index)), problem.corridor.body, alpha=0.65)
    return trace
