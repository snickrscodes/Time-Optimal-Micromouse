"""Generate development-only smoke artifacts for visualization phases V4-V7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from mazegen import Maze, MazeGenerator, compress_to_graph
from main import add_random_openings
from optimization import RectangleBody
from planning import SearchSettings, branch_and_bound_junction_paths
from planning.time_bounds import TimeBoundResult
from visualization.animation import build_playback_trace, render_trajectory_gif
from visualization.architecture import draw_system_architecture
from tools.visuals.ocp_comparison import render as render_ocp_comparison
from visualization.records import route_view_from_record
from visualization.search import draw_search_tree
from visualization.speed import build_speed_profile_trace


def choose_topology_case(document):
    return max(document["cases"], key=lambda case: case["comparison"]["percentage_time_improvement"])


class PrefixLengthLowerBound:
    name = "smoke_prefix_length"

    def evaluate(self, request):
        value = float(max(0, len(request.cell_path) - 1))
        return TimeBoundResult(self.name, value, value, value, 0.0, 0.0, 0, True, ())


def turn_count(cells):
    directions = []
    for a, b in zip(cells[:-1], cells[1:]):
        directions.append((b[0] - a[0], b[1] - a[1]))
    return sum(x != y for x, y in zip(directions[:-1], directions[1:]))


def build_search_trace():
    maze = MazeGenerator(3, 3, 19).generate()
    maze = add_random_openings(maze, 2, seed=19 ^ 0x5EED5EED)
    start = (0, 0)
    graph = compress_to_graph(maze, start, maze.goal)
    events = []

    def evaluator(_junctions, cells):
        # Smoke-only deterministic objective.  PrefixLengthLowerBound remains
        # admissible because the terminal value is path length + nonnegative
        # turn penalty.  The production B&B implementation itself is real.
        return float(len(cells) - 1) + 0.35 * turn_count(cells)

    result = branch_and_bound_junction_paths(
        maze, graph, start, maze.goal,
        lower_bound=PrefixLengthLowerBound(),
        complete_path_time=evaluator,
        init_w=0.8,
        settings=SearchSettings(
            maximum_node_visits=1,
            maximum_expansions=200,
            use_complete_bound_refinement=False,
        ),
        search_trace_observer=events.append,
    )
    return maze, graph, result, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-result", type=Path, default=Path("benchmark_results/reference/topology_search.json"))
    parser.add_argument("--ocp-result", type=Path, default=Path("benchmark_results/reference/full_ocp_resolution_control.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualization_smoke_v4_v7"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topology_doc = json.loads(args.topology_result.read_text())
    case = choose_topology_case(topology_doc)
    spec = case["case"]
    maze = MazeGenerator(spec["width"], spec["height"], spec["seed"]).generate()
    maze = add_random_openings(maze, spec["extra_openings"], seed=spec["opening_seed"])
    route = route_view_from_record(case["branch_and_bound"]["route"])
    cfg = route.config
    speed = build_speed_profile_trace(
        route.raw_parameters,
        init_w=float(cfg["init_w"]),
        terminal_w_max=float(cfg.get("goal_max_speed", cfg["init_w"] ** 0.5)) ** 2 if "goal_max_speed" in cfg else float(cfg["init_w"]),
        initial_k=route.initial_state.k,
        n_scan=int(cfg.get("n_scan", 256)),
        envelope_scan=int(cfg.get("envelope_scan", 64)),
        domain_scan=int(cfg.get("domain_scan", 64)),
        time_tolerance=1e-5,
    )
    playback = build_playback_trace(route.raw_parameters, route.initial_state, speed, fps=16.0)
    body = RectangleBody.centered(float(cfg.get("body_length", 5/9)), float(cfg.get("body_height", 4/9)))
    gif = render_trajectory_gif(
        args.output_dir / "v4_hero_animation.gif",
        maze=maze, playback=playback, body=body, cells=route.cells, fps=16.0, dpi=80,
        title="V4 development real-time playback",
    )

    ocp_doc = json.loads(args.ocp_result.read_text())
    ocp_path = args.output_dir / "v5_ocp_comparison.svg"
    render_ocp_comparison(args.ocp_result, ocp_path)
    fine = ocp_doc["fine_resolution_summary"]

    _maze, _graph, search_result, events = build_search_trace()
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_search_tree(ax, events)
    fig.tight_layout()
    search_path = args.output_dir / "v6_search_tree.svg"
    fig.savefig(search_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_system_architecture(ax)
    fig.tight_layout()
    architecture_path = args.output_dir / "v7_architecture.svg"
    fig.savefig(architecture_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    generated_events = sum(e.kind == "generated" for e in events)
    expanded_events = sum(e.kind == "expanded" for e in events)
    summary = {
        "v4": {
            "case": spec["name"], "frames": len(playback.time),
            "duration_seconds": playback.duration, "gif_bytes": gif.stat().st_size,
        },
        "v5": {
            "case": ocp_doc["case"],
            "ocp_intervals": fine["ocp_intervals"],
            "structured_segments": fine["structured_segments"],
            "structured_time": fine["structured_time_seconds"],
            "ocp_time": fine["ocp_objective_time_seconds"],
            "hybrid_on_ocp_geometry_time": fine["hybrid_replay_time_seconds"],
        },
        "v6": {
            "events": len(events), "generated_events": generated_events,
            "result_generated": search_result.generated,
            "expanded_events": expanded_events, "result_expanded": search_result.expanded,
            "best_time": search_result.best_time,
        },
        "v7": {"artifact": architecture_path.name},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
