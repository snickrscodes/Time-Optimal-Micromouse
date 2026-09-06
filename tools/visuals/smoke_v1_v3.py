"""Generate non-polished development artifacts exercising visualization V1-V3.

This script deliberately consumes checked-in benchmark JSON instead of running
any optimizer.  It is a regression/demo harness, not a final README renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from mazegen import Maze, MazeGenerator
from planning import build_route_optimization_problem
from visualization.comparison import draw_solution
from visualization.detail import draw_geometry_detail
from visualization.records import route_view_from_record
from visualization.speed import build_speed_profile_trace
from visualization.speed_plot import draw_speed_profile


def add_openings(maze: Maze, count: int, seed: int) -> Maze:
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


def choose_case(document, name: str | None):
    cases = document["cases"]
    if name is not None:
        return next(case for case in cases if case["case"]["name"] == name)
    return max(cases, key=lambda case: case["comparison"]["percentage_time_improvement"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("benchmark_results/reference/topology_search.json"))
    parser.add_argument("--case")
    parser.add_argument("--output-dir", type=Path, default=Path("visualization_smoke"))
    args = parser.parse_args()

    document = json.loads(args.result.read_text())
    case = choose_case(document, args.case)
    spec = case["case"]
    maze = MazeGenerator(spec["width"], spec["height"], spec["seed"]).generate()
    maze = add_openings(maze, spec["extra_openings"], spec["opening_seed"])
    astar = route_view_from_record(case["astar"]["route"])
    best = route_view_from_record(case["branch_and_bound"]["route"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # V1: route/maze layer comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    draw_solution(axes[0], maze, astar, title="Optimized A* topology")
    draw_solution(axes[1], maze, best, title="Optimized B&B topology")
    fig.tight_layout()
    fig.savefig(args.output_dir / "v1_comparison.svg", bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    # V2: body/corridor geometry layers on the B&B route.
    cfg = best.config
    problem = build_route_optimization_problem(
        best.cells,
        body_length=float(cfg.get("body_length", 5.0 / 9.0)),
        body_height=float(cfg.get("body_height", 4.0 / 9.0)),
        corridor_mode=str(cfg.get("corridor_mode", "overlapping_cover")),
        refinement_factor=int(cfg.get("geometry_refinement", 1)),
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_geometry_detail(
        ax,
        maze=maze,
        problem=problem,
        parameters=best.parameters,
        cells=best.cells,
        footprint_count=9,
        show_topology=True,
    )
    ax.set_title("V2 development geometry layers")
    fig.tight_layout()
    fig.savefig(args.output_dir / "v2_geometry_detail.svg", bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    # V3: exact scalar topology + presentation time trace.
    init_w = float(cfg["init_w"])
    trace = build_speed_profile_trace(
        best.raw_parameters,
        init_w=init_w,
        terminal_w_max=init_w,
        initial_k=best.initial_state.k,
        n_scan=int(cfg.get("n_scan", 256)),
        envelope_scan=int(cfg.get("envelope_scan", 64)),
        domain_scan=int(cfg.get("domain_scan", 64)),
        time_tolerance=1e-5,
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    draw_speed_profile(axes[0], axes[1], trace)
    axes[0].set_title(
        f"V3 development speed trace · T={trace.exact_total_time:.6f}s · "
        f"quadrature error={trace.integration_error:.2e}s"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "v3_speed_profile.svg", bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    summary = {
        "case": spec["name"],
        "comparison_improvement_percent": case["comparison"]["percentage_time_improvement"],
        "speed_exact_time": trace.exact_total_time,
        "speed_recorded_time": best.time,
        "speed_quadrature_error": trace.integration_error,
        "speed_modes": sorted(set(trace.mode)),
        "mode_switches": [event.__dict__ if hasattr(event, "__dict__") else {
            "station": event.station, "kind": event.kind, "from_mode": event.from_mode,
            "to_mode": event.to_mode, "from_pass": event.from_pass, "to_pass": event.to_pass,
        } for event in trace.events],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
