"""Reusable side-by-side optimized-topology comparison renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .maze import draw_goal_region, draw_maze_walls, draw_start
from .sampling import sample_geometry_parameters
from .trajectory import draw_geometry_trace, draw_topology


def draw_solution(
    ax: Any,
    maze: Any,
    route: Any,
    *,
    title: str,
    scenario: Any | None = None,
    samples_per_unit: float = 100.0,
    minimum_samples_per_segment: int = 12,
    show_topology: bool = True,
) -> None:
    draw_maze_walls(ax, maze)
    if show_topology:
        draw_topology(ax, route.cells, label="topology centerline")
    trace = sample_geometry_parameters(
        route.parameters,
        route.initial_state,
        samples_per_unit=samples_per_unit,
        minimum_samples_per_segment=minimum_samples_per_segment,
    )
    draw_geometry_trace(ax, trace, label="optimized clothoid path")
    draw_start(ax, route.cells[0], label="start")
    draw_goal_region(
        ax,
        scenario=scenario,
        goal_cell=None if scenario is not None else route.cells[-1],
    )
    ax.set_title(
        f"{title}\nT = {route.time:.6f} s · {len(route.cells)} cells · {route.selected_stage}",
        fontsize=10,
    )


def render_comparison(
    maze: Any,
    seed_route: Any,
    best_route: Any,
    *,
    search_result: Any,
    maze_seed: int,
    extra_openings: int,
    output: Path,
    dpi: int = 180,
    show: bool = False,
    show_topology: bool = True,
    samples_per_unit: float = 100.0,
    minimum_samples_per_segment: int = 12,
    seed_is_quotient_class: bool = False,
    scenario: Any | None = None,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharex=True, sharey=True)
    draw_solution(
        axes[0], maze, seed_route,
        title=("A* seed quotient class after optimization" if seed_is_quotient_class else "Naive A* topology after optimization"),
        scenario=scenario,
        samples_per_unit=samples_per_unit,
        minimum_samples_per_segment=minimum_samples_per_segment,
        show_topology=show_topology,
    )
    final_label = "Final best path" if search_result.exhausted else "Best path found (search capped)"
    draw_solution(
        axes[1], maze, best_route,
        title=final_label,
        scenario=scenario,
        samples_per_unit=samples_per_unit,
        minimum_samples_per_segment=minimum_samples_per_segment,
        show_topology=show_topology,
    )
    improvement = 100.0 * (seed_route.time - best_route.time) / seed_route.time
    figure.suptitle(
        f"Time-optimal maze planner · seed={maze_seed} · openings={extra_openings}\n"
        f"improvement={improvement:.3f}% · expanded={search_result.expanded} · "
        f"complete={search_result.complete_paths_evaluated} · "
        f"bound-pruned={search_result.pruned_by_bound} · "
        f"reach-pruned={search_result.pruned_by_reachability} · "
        f"exhausted={search_result.exhausted}",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
