"""Presentation-oriented A* versus kinodynamic topology comparison.

The renderer consumes previously generated result JSON and never launches
planning or optimization.  It supports both ``main.py`` result metadata and
Benchmark-1 ``topology_search.json`` records.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

from mazegen import Maze, MazeGenerator
from planning.maze_io import load_maze_scenario
from visualization.maze import draw_goal_region, draw_maze_walls, draw_start
from visualization.records import RouteSnapshotView, route_view_from_record
from visualization.sampling import sample_geometry_parameters
from visualization.trajectory import draw_geometry_trace, draw_topology


@dataclass(frozen=True, slots=True)
class TopologyCaseStudy:
    maze: Maze
    scenario: Any | None
    astar: RouteSnapshotView
    best: RouteSnapshotView
    astar_record: Mapping[str, Any]
    best_record: Mapping[str, Any]
    topology_changed: bool
    improvement_percent: float
    search_statistics: Mapping[str, Any]
    title: str
    scale_m_per_grid_unit: float | None = None


def _add_openings(maze: Maze, count: int, seed: int) -> Maze:
    connections = maze.connection_dict()
    candidates = []
    for y in range(maze.height):
        for x in range(maze.width):
            cell = (x, y)
            for neighbor in ((x + 1, y), (x, y + 1)):
                if neighbor[0] >= maze.width or neighbor[1] >= maze.height:
                    continue
                if neighbor not in connections[cell]:
                    candidates.append((cell, neighbor))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for first, second in candidates[:count]:
        connections[first].add(second)
        connections[second].add(first)
    return Maze.from_connections(maze.width, maze.height, connections, maze.goal)


def _resolve_path(candidate: str | None, *, result_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    if candidate is None:
        raise ValueError("result does not identify a maze file; pass --maze")
    value = Path(candidate)
    probes = [value]
    if not value.is_absolute():
        probes.extend((result_path.parent / value, Path.cwd() / value))
    for probe in probes:
        if probe.exists():
            return probe
    raise FileNotFoundError(f"maze file from result is not available: {candidate!r}; pass --maze")


def load_case_study(
    result_path: str | Path,
    *,
    maze_override: str | Path | None = None,
    case_name: str | None = None,
) -> TopologyCaseStudy:
    result_path = Path(result_path)
    document = json.loads(result_path.read_text())
    override = None if maze_override is None else Path(maze_override)

    # main.py metadata format.
    if "seed_route" in document and "best_route" in document and "search" in document:
        configuration = document.get("configuration", {})
        maze_path = _resolve_path(configuration.get("maze_file"), result_path=result_path, override=override)
        scenario = load_maze_scenario(maze_path)
        astar_record = document["seed_route"]
        best_record = document["best_route"]
        astar = route_view_from_record(astar_record)
        best = route_view_from_record(best_record)
        changed = tuple(astar.cells) != tuple(best.cells)
        improvement = 100.0 * (astar.time - best.time) / astar.time
        return TopologyCaseStudy(
            maze=scenario.maze,
            scenario=scenario,
            astar=astar,
            best=best,
            astar_record=astar_record,
            best_record=best_record,
            topology_changed=changed,
            improvement_percent=improvement,
            search_statistics=document["search"],
            title=scenario.name,
            scale_m_per_grid_unit=scenario.scale.cell_pitch_m,
        )

    # Benchmark 1 aggregate format.
    if "cases" in document:
        cases = document["cases"]
        if case_name is None:
            case = max(cases, key=lambda row: row["comparison"]["percentage_time_improvement"])
        else:
            case = next(row for row in cases if row["case"]["name"] == case_name)
        spec = case["case"]
        maze = MazeGenerator(spec["width"], spec["height"], spec["seed"]).generate()
        maze = _add_openings(maze, spec["extra_openings"], spec["opening_seed"])
        astar_record = dict(case["astar"]["route"])
        best_record = dict(case["branch_and_bound"]["route"])
        astar_record.setdefault("certification", case["astar"].get("certification"))
        best_record.setdefault("certification", case["branch_and_bound"].get("certification"))
        return TopologyCaseStudy(
            maze=maze,
            scenario=None,
            astar=route_view_from_record(astar_record),
            best=route_view_from_record(best_record),
            astar_record=astar_record,
            best_record=best_record,
            topology_changed=bool(case["comparison"]["topology_changed"]),
            improvement_percent=float(case["comparison"]["percentage_time_improvement"]),
            search_statistics=case["branch_and_bound"]["search_statistics"],
            title=str(spec["name"]),
            scale_m_per_grid_unit=None,
        )
    raise ValueError("unrecognized A*/B&B result schema")


def _route_length(route: RouteSnapshotView) -> float:
    return float(math.fsum(float(v) for v in route.raw_parameters[0::2]))


def _format_length(value: float, scale: float | None) -> str:
    if scale is None:
        return f"{value:.2f} grid units"
    return f"{value * scale:.2f} m"


def _certified(record: Mapping[str, Any]) -> bool | None:
    cert = record.get("certification")
    if isinstance(cert, Mapping):
        return bool(cert.get("certified"))
    # main.py route snapshots already retain stage-level independent geometry
    # certification.  Use the selected stage as a fallback when a dedicated
    # top-level certificate was not serialized by an older result file.
    selected = str(record.get("selected_stage", ""))
    stages = record.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            name = str(stage.get("name", ""))
            if name == selected or (selected.startswith(name + "+") and stage.get("feasible") is not None):
                return bool(stage.get("feasible"))
    return None


def _draw_panel(
    ax: Any,
    case: TopologyCaseStudy,
    route: RouteSnapshotView,
    record: Mapping[str, Any],
    *,
    heading: str,
    panel_label: str,
    show_topology: bool,
) -> None:
    draw_maze_walls(ax, case.maze, colors="#151515", linewidths=1.35)
    if show_topology:
        draw_topology(ax, route.cells, color="#8f8f8f", linewidth=1.0, alpha=0.55, label="discrete topology")
    trace = sample_geometry_parameters(
        route.parameters,
        route.initial_state,
        samples_per_unit=90.0,
        minimum_samples_per_segment=10,
    )
    draw_geometry_trace(ax, trace, color="#0077a8", linewidth=2.0, label="optimized trajectory")
    draw_start(ax, route.cells[0], color="#111111", s=30, label="start")
    draw_goal_region(
        ax,
        scenario=case.scenario,
        goal_cell=None if case.scenario is not None else route.cells[-1],
        facecolor="#4c956c",
        edgecolor="#4c956c",
        alpha=0.12,
    )

    topological = max(0, len(route.cells) - 1)
    optimized = _route_length(route)
    cert = _certified(record)
    cert_text = " · certified" if cert is True else "" if cert is None else " · NOT CERTIFIED"
    ax.set_title(f"{panel_label}  {heading}", loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax.text(
        0.02, 0.02,
        f"Topology length  {_format_length(float(topological), case.scale_m_per_grid_unit)}\n"
        f"Clothoid length   {_format_length(optimized, case.scale_m_per_grid_unit)}\n"
        f"Traversal time    {route.time:.4f} s{cert_text}",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.94},
        zorder=20,
    )


def render_astar_vs_bnb_case_study(
    case: TopologyCaseStudy,
    output: str | Path,
    *,
    show_topology: bool = True,
    dpi: int = 220,
    show: bool = False,
) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.55), sharex=True, sharey=True)
    _draw_panel(
        axes[0], case, case.astar, case.astar_record,
        heading="Shortest-distance A* topology", panel_label="A", show_topology=show_topology,
    )
    _draw_panel(
        axes[1], case, case.best, case.best_record,
        heading="Kinodynamic branch-and-bound", panel_label="B", show_topology=show_topology,
    )

    changed = "different topology" if case.topology_changed else "same topology"
    both_certified = _certified(case.astar_record) is True and _certified(case.best_record) is True
    quality = "lower certified traversal time" if both_certified else "lower traversal time"
    figure.suptitle(
        f"{case.title}\n"
        f"same continuous optimizer · {case.improvement_percent:.2f}% {quality} · {changed}",
        fontsize=13.0,
        fontweight="semibold",
        y=0.99,
    )
    stats = case.search_statistics
    complete = stats.get("complete_paths_evaluated", stats.get("complete_route_optimizations_including_seed", "?"))
    expanded = stats.get("expanded", "?")
    pruned = stats.get("pruned_by_bound", "?")
    reach = stats.get("pruned_by_reachability", "?")
    figure.text(
        0.5, 0.018,
        f"B&B diagnostics: expanded {expanded} · complete routes optimized {complete} · "
        f"bound pruned {pruned} · reachability pruned {reach}",
        ha="center", va="bottom", fontsize=8.6, color="0.30",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04), fontsize=8.5)
    figure.tight_layout(rect=(0.015, 0.075, 0.985, 0.92), w_pad=1.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
