from __future__ import annotations

from . import SCHEMA_VERSION

import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from planning import (
    CompletePathEvaluation,
    PortalMotorTimeLowerBound,
    SearchSettings,
    astar_junction_path,
    branch_and_bound_junction_paths,
    expand_junction_path,
)

from .common.certification import certification_passed, certify_optimized_route as independent_certify
from .common.environment import environment_metadata
from .common.io import write_json
from .common.routes import RouteEvaluator, build_case, optimized_path_length, route_to_record, topology_length_cells
from .config import CORE_OPTIMIZATION, CORE_TOPOLOGY_CASES, SMOKE_OPTIMIZATION, SMOKE_TOPOLOGY_CASES, MazeCase, OptimizationConfig


def _bound(case_maze, config: OptimizationConfig) -> PortalMotorTimeLowerBound:
    return PortalMotorTimeLowerBound(
        maximum_iterations=100,
        cache_warm_starts=True,
        maze=case_maze,
        body_length=config.body_length,
        body_height=config.body_height,
        use_inscribed_disk_gates=True,
        use_two_sided_time=True,
        use_dual_certificate=False,
        use_complete_cover_bound=True,
        complete_refinement_gap=2.0,
        projection_directions=12,
    )


def run_case(case: MazeCase, config: OptimizationConfig) -> dict[str, Any]:
    start = (0, 0)
    maze, graph = build_case(case, start=start)
    astar_nodes = tuple(astar_junction_path(graph, start, maze.goal))
    astar_cells = tuple(expand_junction_path(maze, graph, astar_nodes))

    with RouteEvaluator(config) as evaluator:
        astar_route = evaluator.optimize(astar_cells)
        astar_cert = independent_certify(astar_route, config)
        if not astar_cert["certified"]:
            raise RuntimeError(f"A* seed route failed independent certification: {case.name}")

        evaluations_before_search = len(evaluator.cache)
        new_optimizations = 0

        def evaluate_complete(nodes: tuple[int, ...], cells: tuple[tuple[int, int], ...]):
            nonlocal new_optimizations
            key = tuple(cells)
            before = key in evaluator.cache
            route = evaluator.optimize(cells)
            cert = independent_certify(route, config)
            if not cert["certified"]:
                raise RuntimeError(f"B&B candidate failed independent certification: {case.name}, {nodes}")
            if not before:
                new_optimizations += 1
            return CompletePathEvaluation(route.time, nodes, tuple(route.cells))

        started = time.perf_counter()
        search = branch_and_bound_junction_paths(
            maze,
            graph,
            start,
            maze.goal,
            lower_bound=_bound(maze, config),
            complete_path_time=evaluate_complete,
            init_w=config.init_w,
            settings=SearchSettings(
                maximum_node_visits=1,
                maximum_expansions=20_000,
                use_block_cut_pruning=True,
                use_no_revisit_reachability=True,
                use_complete_bound_refinement=True,
            ),
            seed_junction_path=astar_nodes,
            topology_quotient=None,
        )
        search_wall = time.perf_counter() - started
        if not search.exhausted:
            raise RuntimeError(f"B&B did not exhaust search for {case.name}")

        best_cells = tuple(search.best_cell_path)
        best_route = evaluator.cache.get(best_cells)
        if best_route is None:
            best_route = evaluator.optimize(best_cells)
        best_cert = independent_certify(best_route, config)
        if not best_cert["certified"]:
            raise RuntimeError(f"selected B&B route failed certification: {case.name}")

        astar_centerline = topology_length_cells(astar_cells)
        best_centerline = topology_length_cells(best_cells)
        improvement = astar_route.time - best_route.time
        percent = 100.0 * improvement / astar_route.time if astar_route.time else 0.0

        return {
            "case": case.to_dict(),
            "maze": {
                "goal": maze.goal,
                "cells": maze.width * maze.height,
                "junctions": len(graph.nodes),
            },
            "astar": {
                "junction_path": astar_nodes,
                "cell_path": astar_cells,
                "topology_centerline_length": astar_centerline,
                "optimized_clothoid_length": optimized_path_length(astar_route),
                "route": route_to_record(astar_route, config),
                "certification": astar_cert,
            },
            "branch_and_bound": {
                "selected_junction_path": search.best_junction_path,
                "selected_cell_path": search.best_cell_path,
                "topology_centerline_length": best_centerline,
                "optimized_clothoid_length": optimized_path_length(best_route),
                "route": route_to_record(best_route, config),
                "certification": best_cert,
                "search_wall_seconds": search_wall,
                "complete_route_optimizations_including_seed": 1 + new_optimizations,
                "new_complete_route_optimizations_during_search": new_optimizations,
                "search_statistics": asdict(search),
            },
            "comparison": {
                "topology_changed": tuple(astar_cells) != tuple(best_cells),
                "selected_topology_geometrically_longer": best_centerline > astar_centerline + 1e-12,
                "absolute_time_improvement_seconds": improvement,
                "percentage_time_improvement": percent,
            },
        }


def aggregate(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    certified = [
        row for row in cases
        if certification_passed(row["astar"])
        and certification_passed(row["branch_and_bound"])
    ]
    improvements = [row["comparison"]["percentage_time_improvement"] for row in certified]
    abs_improvements = [row["comparison"]["absolute_time_improvement_seconds"] for row in certified]
    return {
        "cases_total": len(cases),
        "cases_in_quality_aggregate": len(certified),
        "topology_changes": sum(bool(row["comparison"]["topology_changed"]) for row in certified),
        "longer_but_faster_cases": sum(
            bool(row["comparison"]["selected_topology_geometrically_longer"])
            and row["comparison"]["absolute_time_improvement_seconds"] > 0.0
            for row in certified
        ),
        "time_improvement_percent": {
            "mean": statistics.fmean(improvements) if improvements else None,
            "median": statistics.median(improvements) if improvements else None,
            "max": max(improvements) if improvements else None,
            "min": min(improvements) if improvements else None,
        },
        "absolute_time_improvement_seconds": {
            "mean": statistics.fmean(abs_improvements) if abs_improvements else None,
            "median": statistics.median(abs_improvements) if abs_improvements else None,
            "max": max(abs_improvements) if abs_improvements else None,
        },
        "total_complete_route_optimizations": sum(
            int(row["branch_and_bound"]["complete_route_optimizations_including_seed"])
            for row in certified
        ),
        "total_search_wall_seconds": sum(float(row["branch_and_bound"]["search_wall_seconds"]) for row in certified),
    }


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    cases = CORE_TOPOLOGY_CASES if profile == "core" else SMOKE_TOPOLOGY_CASES
    config = CORE_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    rows = [run_case(case, config) for case in cases]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "topology_search",
        "profile": profile,
        "environment": environment_metadata(config),
        "settings": {
            "topology_quotient": False,
            "note": "Quotienting is disabled so every graph path is an auditable topology; both A* and B&B use the same integrated route optimizer.",
        },
        "cases": rows,
        "aggregate": aggregate(rows),
    }
    write_json(output_dir / "topology_search.json", result)
    return result
