"""Cheap fixed-maze topology/search preflight for the Red Comet case study.

This tool performs no continuous trajectory optimization.  It intentionally
answers the questions that should be resolved *before* launching an expensive
16x16 production B&B campaign:

* how large is the compressed/relevant graph?
* how many simple start-to-goal junction paths actually exist under the
  production no-revisit policy?
* how long/turn-heavy are those topologies?
* what do the production lower bounds say about each complete topology?
* how many alternatives could remain competitive for a given seed incumbent?

The output is deterministic JSON plus a small Markdown summary and an optional
contact sheet of every simple topology.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

from mazegen import compress_to_graph
from planning.branch_and_bound import (
    CompletePathEvaluation, SearchSettings, _block_cut_relevant_subgraph,
    branch_and_bound_junction_paths,
)
from planning.graph_blocks import undirected_edges
from planning.maze_io import load_maze_scenario
from planning.maze_routes import (
    DEFAULT_BODY_HEIGHT,
    DEFAULT_BODY_LENGTH,
    astar_junction_path,
    expand_junction_path,
)
from planning.time_bounds import PortalMotorTimeLowerBound, TimeBoundRequest
from segment.constants import PHYSICS_PROFILE, PHYSICS_PROFILE_NAME
from planning.topology_quotient import OpenRoomTopologyQuotient


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _turn_metrics(cells: tuple[tuple[int, int], ...]) -> tuple[int, tuple[int, ...]]:
    directions = [
        (second[0] - first[0], second[1] - first[1])
        for first, second in zip(cells[:-1], cells[1:])
    ]
    if not directions:
        return 0, ()
    runs: list[int] = []
    current = directions[0]
    length = 1
    for direction in directions[1:]:
        if direction == current:
            length += 1
        else:
            runs.append(length)
            current = direction
            length = 1
    runs.append(length)
    return max(0, len(runs) - 1), tuple(runs)


def _enumerate_paths(graph: Any, relevant: Any, source: int, target: int, maximum: int) -> tuple[list[tuple[int, ...]], bool]:
    paths: list[tuple[int, ...]] = []
    stack: list[tuple[int, tuple[int, ...], int]] = [(source, (source,), 1 << source)]
    truncated = False
    while stack:
        node, path, mask = stack.pop()
        if node == target:
            paths.append(path)
            if len(paths) >= maximum:
                truncated = bool(stack)
                break
            continue
        # Reverse sorted push -> deterministic ascending pop order.
        for edge in sorted(relevant.adjacency[node], key=lambda item: item.to, reverse=True):
            bit = 1 << edge.to
            if mask & bit:
                continue
            stack.append((edge.to, path + (edge.to,), mask | bit))
    return paths, truncated


def _path_distance(graph: Any, path: tuple[int, ...]) -> int:
    total = 0
    for first, second in zip(path[:-1], path[1:]):
        edge = next(item for item in graph.adj[first] if item.to == second)
        total += int(edge.length)
    return total


def _render_gallery(path: Path, *, scenario: Any, records: list[dict[str, Any]], columns: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.maze import draw_goal_region, draw_maze_walls, draw_start
    from visualization.trajectory import draw_topology

    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        ax.axis("on")
        draw_maze_walls(ax, scenario.maze)
        draw_topology(ax, tuple(tuple(cell) for cell in record["cell_path"]), label=None)
        draw_start(ax, scenario.start, label=None)
        draw_goal_region(ax, scenario=scenario)
        tag = "A*" if record["is_astar"] else f"P{record['id']}"
        ax.set_title(
            f"{tag} · {record['grid_steps']} steps · {record['turns']} turns\n"
            f"complete LB {record['complete_lower_bound_seconds']:.2f} s",
            fontsize=9,
        )
    figure.suptitle("Red Comet maze · every simple start→goal topology under no-revisit policy", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(figure)


def run_preflight(
    maze_path: Path,
    *,
    output_dir: Path,
    maximum_paths: int = 1_000_000,
    init_w: float = 0.8,
    terminal_w_max: float = 0.8,
    body_length: float | None = None,
    body_height: float | None = None,
    render_gallery: bool = True,
    use_topology_quotient: bool = True,
) -> dict[str, Any]:
    scenario = load_maze_scenario(maze_path)
    if body_length is None:
        body_length = PHYSICS_PROFILE.body_length_grid or DEFAULT_BODY_LENGTH
    if body_height is None:
        body_height = PHYSICS_PROFILE.body_width_grid or DEFAULT_BODY_HEIGHT
    maze = scenario.maze
    graph = compress_to_graph(maze, scenario.start, scenario.canonical_goal)
    source = graph.index[scenario.start]
    target = graph.index[scenario.canonical_goal]
    relevant = _block_cut_relevant_subgraph(graph, source, target)
    astar = tuple(astar_junction_path(graph, scenario.start, scenario.canonical_goal))

    started = time.perf_counter()
    junction_paths, truncated = _enumerate_paths(graph, relevant, source, target, maximum_paths)
    enumeration_seconds = time.perf_counter() - started

    quotient = OpenRoomTopologyQuotient(
        maze, graph, body_length=body_length, body_height=body_height
    )
    lower_bound = PortalMotorTimeLowerBound(
        maximum_iterations=100,
        cache_warm_starts=True,
        maze=maze,
        body_length=body_length,
        body_height=body_height,
        use_inscribed_disk_gates=True,
        use_two_sided_time=True,
        use_dual_certificate=False,
        use_complete_cover_bound=True,
        projection_directions=12,
    )
    start_point = (scenario.start[0] + 0.5, scenario.start[1] + 0.5)
    goal_point = (scenario.canonical_goal[0] + 0.5, scenario.canonical_goal[1] + 0.5)

    records: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for path_id, junction_path in enumerate(junction_paths):
        cells = tuple(expand_junction_path(maze, graph, junction_path))
        grid_steps = _path_distance(graph, junction_path)
        turns, straight_runs = _turn_metrics(cells)
        signature = quotient.signature(junction_path) if quotient.active else junction_path
        signatures.add(repr(signature))
        request = TimeBoundRequest(
            cells,
            start_point,
            goal_point,
            init_w,
            terminal_w_max=terminal_w_max,
            goal_radius=math.sqrt(0.5),
        )
        t0 = time.perf_counter()
        base = lower_bound.evaluate(request)
        base_seconds = time.perf_counter() - t0
        # The complete-cover routine uses incumbent gap only as an evaluation
        # trigger.  A synthetic incumbent one second above the base requests the
        # stronger certified lower bound without pretending it is a real upper
        # bound or affecting any planner result.
        t0 = time.perf_counter()
        complete = lower_bound.refine_complete(
            request, base.time_lower_bound + 1.0, base
        )
        complete_seconds = time.perf_counter() - t0
        records.append({
            "id": path_id,
            "is_astar": junction_path == astar,
            "junction_path": list(junction_path),
            "cell_path": [list(cell) for cell in cells],
            "junction_count": len(junction_path),
            "cell_count": len(cells),
            "grid_steps": grid_steps,
            "topological_length_m": grid_steps * scenario.scale.cell_pitch_m,
            "turns": turns,
            "straight_run_lengths": list(straight_runs),
            "maximum_straight_run_cells": max(straight_runs, default=0),
            "base_lower_bound_seconds": float(base.time_lower_bound),
            "complete_lower_bound_seconds": float(complete.time_lower_bound),
            "base_bound_name": base.name,
            "complete_bound_name": complete.name,
            "base_bound_wall_seconds": base_seconds,
            "complete_refinement_wall_seconds": complete_seconds,
            "quotient_signature": repr(signature),
        })

    records.sort(key=lambda row: (row["grid_steps"], row["turns"], row["id"]))
    for index, row in enumerate(records):
        row["rank_by_distance"] = index + 1

    astar_record = next(row for row in records if row["is_astar"])
    alternatives = [row for row in records if not row["is_astar"]]
    minimum_alternative_lb = min((row["complete_lower_bound_seconds"] for row in alternatives), default=math.inf)
    longer_family = [row for row in alternatives if row["grid_steps"] >= 121]
    minimum_long_lb = min((row["complete_lower_bound_seconds"] for row in longer_family), default=math.inf)

    # Useful incumbent thresholds.  This is a leaf-level competition count,
    # not a prediction that prefix B&B necessarily visits every such leaf.
    thresholds = sorted(set([
        astar_record["complete_lower_bound_seconds"],
        minimum_alternative_lb,
        minimum_long_lb,
        20.0, 22.0, 24.0, 26.0,
    ]))
    competition = []
    for threshold in thresholds:
        if not math.isfinite(threshold):
            continue
        competition.append({
            "incumbent_seconds": threshold,
            "complete_topologies_with_lb_below_incumbent": sum(
                row["complete_lower_bound_seconds"] < threshold - 1e-12
                for row in records
            ),
        })

    # Conditional dry B&B: use the real prefix/leaf bound hierarchy and the
    # production search implementation, but replace every continuous leaf NLP
    # with a fixed hypothetical incumbent.  This measures discrete search work
    # *conditional* on the seed optimizer eventually returning that time.
    conditional_search = []
    dry_thresholds = sorted(set((3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 18.0, 20.0, 22.0, 24.0, 26.0, minimum_alternative_lb, minimum_long_lb)))
    for threshold in dry_thresholds:
        if not math.isfinite(threshold):
            continue
        dry_bound = PortalMotorTimeLowerBound(
            maximum_iterations=100, cache_warm_starts=True, maze=maze,
            body_length=body_length, body_height=body_height,
            use_inscribed_disk_gates=True, use_two_sided_time=True,
            use_dual_certificate=False, use_complete_cover_bound=True,
            projection_directions=12,
        )
        def dry_evaluator(nodes, cells, *, _threshold=float(threshold)):
            return CompletePathEvaluation(_threshold, tuple(nodes), tuple(cells))
        t0 = time.perf_counter()
        dry = branch_and_bound_junction_paths(
            maze, graph, scenario.start, scenario.canonical_goal,
            lower_bound=dry_bound, complete_path_time=dry_evaluator,
            init_w=init_w, terminal_w_max=terminal_w_max,
            settings=SearchSettings(maximum_node_visits=1, maximum_expansions=20_000),
            seed_junction_path=astar,
            topology_quotient=(quotient if use_topology_quotient and quotient.active else None),
        )
        conditional_search.append({
            "hypothetical_seed_incumbent_seconds": float(threshold),
            "expanded": dry.expanded,
            "generated": dry.generated,
            "pruned_by_bound": dry.pruned_by_bound,
            "pruned_by_reachability": dry.pruned_by_reachability,
            "pruned_by_complete_bound": dry.pruned_by_complete_bound,
            "complete_alternative_evaluations": dry.complete_paths_evaluated,
            "exhausted": dry.exhausted,
            "wall_seconds": time.perf_counter() - t0,
        })

    edge_count = len(undirected_edges(graph))
    result: dict[str, Any] = {
        "schema": "red-comet-preflight-v1",
        "maze": {
            "name": scenario.name,
            "path": str(maze_path),
            "sha256": scenario.source_sha256,
            "width": maze.width,
            "height": maze.height,
            "cell_pitch_m": scenario.scale.cell_pitch_m,
            "start_source": list(scenario.to_source_cell(scenario.start)),
            "canonical_goal_source": list(scenario.to_source_cell(scenario.canonical_goal)),
            "goal_cells_source": [list(scenario.to_source_cell(cell)) for cell in scenario.goal_region.cells],
        },
        "graph": {
            "junctions": len(graph.nodes),
            "undirected_edges": edge_count,
            "cycle_rank": max(0, edge_count - len(graph.nodes) + 1),
            "relevant_junctions": len(relevant.relevant_vertices),
            "relevant_edges": len(relevant.relevant_edges),
            "relevant_cycle_rank": relevant.cycle_rank,
            "block_cut_removed_nodes": relevant.removed_nodes,
            "block_cut_removed_edges": relevant.removed_edges,
            "topology_quotient_rooms": len(quotient.rooms),
        },
        "enumeration": {
            "maximum_paths_safeguard": maximum_paths,
            "truncated": truncated,
            "simple_paths": len(records),
            "quotient_classes": len(signatures),
            "seconds": enumeration_seconds,
            "distance_histogram": dict(sorted(Counter(row["grid_steps"] for row in records).items())),
        },
        "astar": astar_record,
        "search_preflight": {
            "minimum_non_astar_complete_lb_seconds": minimum_alternative_lb,
            "minimum_121plus_step_complete_lb_seconds": minimum_long_lb,
            "interpretation": {
                "if_astar_certified_time_below_min_non_astar_lb": "all alternatives are leaf-bound dominated before continuous optimization",
                "if_astar_certified_time_below_min_121plus_lb": "all 121+ step long-route families are leaf-bound dominated; only shorter alternatives can remain competitive",
            },
            "incumbent_thresholds": competition,
            "conditional_dry_bnb": conditional_search,
            "bound_statistics": asdict(lower_bound.statistics),
        },
        "paths": records,
        "physics_profile": {
            "name": PHYSICS_PROFILE_NAME,
            "description": PHYSICS_PROFILE.description,
            "cell_pitch_m": PHYSICS_PROFILE.cell_pitch_m,
            "v_max_grid_cells_per_second": PHYSICS_PROFILE.v_max,
            "v_max_m_per_second": PHYSICS_PROFILE.v_max_mps,
            "a_max_m_per_second2": PHYSICS_PROFILE.a_max_mps2,
            "a_brake_m_per_second2": PHYSICS_PROFILE.a_brake_mps2,
            "mu_g_m_per_second2": PHYSICS_PROFILE.mu_g_mps2,
            "body_length_grid": body_length,
            "body_width_grid": body_height,
        },
        "search_configuration": {
            "topology_quotient_used_in_conditional_dry_bnb": bool(use_topology_quotient and quotient.active),
        },
        "model_scale_note": {
            "v_max_grid_cells_per_second": PHYSICS_PROFILE.v_max,
            "v_max_m_per_second": PHYSICS_PROFILE.v_max * scenario.scale.cell_pitch_m,
            "warning": (
                "The selected named physics profile controls planner dynamics in this fresh process. "
                "Red Comet calibration remains a model mapping rather than an instrumented race-day identification."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "preflight.json", result)
    if render_gallery:
        _render_gallery(output_dir / "topology_gallery.svg", scenario=scenario, records=records)

    md = [
        f"# Red Comet search preflight — {scenario.name}",
        "",
        "This analysis performs **no continuous trajectory optimization**.",
        "",
        "## Graph/search size",
        "",
        f"- Junction graph: **{len(graph.nodes)} nodes / {edge_count} edges**, cycle rank **{max(0, edge_count-len(graph.nodes)+1)}**.",
        f"- Block-cut relevant subgraph: **{len(relevant.relevant_vertices)} nodes / {len(relevant.relevant_edges)} edges**, cycle rank **{relevant.cycle_rank}**.",
        f"- Exact simple-path enumeration under the production no-revisit policy: **{len(records)} paths** in {enumeration_seconds:.4f} s; truncated={truncated}.",
        f"- Conservative open-room quotient detects **{len(quotient.rooms)} room(s)** and leaves **{len(signatures)} distinct classes** among these paths.",
        "",
        "## A* and lower-bound thresholds",
        "",
        f"- A* topology: **{astar_record['grid_steps']} grid steps = {astar_record['topological_length_m']:.2f} m**, {astar_record['turns']} direction changes.",
        f"- A* complete lower bound: **{astar_record['complete_lower_bound_seconds']:.3f} s**.",
        f"- Smallest complete lower bound among *other* topologies: **{minimum_alternative_lb:.3f} s**.",
        f"- Smallest complete lower bound among the 121+ step long-route family: **{minimum_long_lb:.3f} s**.",
        "",
        "Therefore a certified A* incumbent below the last threshold would allow the production complete-leaf bound to reject every 121+ step long-route topology without launching its continuous NLP.  This is a leaf-level statement; prefix pruning can be stronger or weaker depending on when the bound tightens.",
        "",
        "### Conditional dry B&B",
        "",
        "The following runs use the real production prefix/leaf bounds and search implementation, but replace every expensive continuous leaf optimization with a fixed hypothetical seed incumbent.  They therefore estimate discrete search cost conditional on the A* optimizer producing that time; they are not trajectory results.",
        "",
        "| hypothetical A* incumbent (s) | expanded | generated | complete alternatives reached | bound pruned | reachability pruned |",
        "|---:|---:|---:|---:|---:|---:|",
        *[f"| {row['hypothetical_seed_incumbent_seconds']:.3f} | {row['expanded']} | {row['generated']} | {row['complete_alternative_evaluations']} | {row['pruned_by_bound']} | {row['pruned_by_reachability']} |" for row in conditional_search],
        "",
        "## Candidate topologies",
        "",
        "| rank | A* | steps | meters | turns | max straight run | complete LB (s) |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        md.append(
            f"| {row['rank_by_distance']} | {'yes' if row['is_astar'] else ''} | {row['grid_steps']} | {row['topological_length_m']:.2f} | {row['turns']} | {row['maximum_straight_run_cells']} | {row['complete_lower_bound_seconds']:.3f} |"
        )
    md += [
        "",
        "## Physical profile",
        "",
        f"- Selected profile: **`{PHYSICS_PROFILE_NAME}`**.",
        f"- Vehicle footprint used for bounds: **{body_length:.6f} x {body_height:.6f} cells**.",
        f"- `V_MAX={PHYSICS_PROFILE.v_max:.6f}` cells/s = **{PHYSICS_PROFILE.v_max*scenario.scale.cell_pitch_m:.3f} m/s** at this maze pitch.",
        f"- Conditional dry B&B topology quotient enabled: **{bool(use_topology_quotient and quotient.active)}**.",
        "",
        "The Red Comet profile is an explicit calibrated case-study model, not a claim of instrumented race-day parameter identification.",
        "",
    ]
    (output_dir / "PREFLIGHT.md").write_text("\n".join(md))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", type=Path, default=Path("examples/mazes/historical/red_comet_reference_maze.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/red_comet_preflight"))
    parser.add_argument("--maximum-paths", type=int, default=1_000_000)
    parser.add_argument("--no-gallery", action="store_true")
    parser.add_argument("--body-length", type=float)
    parser.add_argument("--body-width", type=float)
    parser.add_argument("--no-topology-quotient", action="store_true", help="match active_basis_v11 final search by disabling quotienting in dry B&B")
    args = parser.parse_args()
    result = run_preflight(
        args.maze,
        output_dir=args.output_dir,
        maximum_paths=args.maximum_paths,
        body_length=args.body_length, body_height=args.body_width,
        render_gallery=not args.no_gallery,
        use_topology_quotient=not args.no_topology_quotient,
    )
    print(json.dumps({
        "simple_paths": result["enumeration"]["simple_paths"],
        "relevant_cycle_rank": result["graph"]["relevant_cycle_rank"],
        "astar_steps": result["astar"]["grid_steps"],
        "astar_complete_lb_seconds": result["astar"]["complete_lower_bound_seconds"],
        "minimum_non_astar_complete_lb_seconds": result["search_preflight"]["minimum_non_astar_complete_lb_seconds"],
        "minimum_long_route_complete_lb_seconds": result["search_preflight"]["minimum_121plus_step_complete_lb_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
