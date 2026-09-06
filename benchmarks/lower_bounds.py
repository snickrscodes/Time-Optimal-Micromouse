from __future__ import annotations

from . import SCHEMA_VERSION

import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from planning import (
    CompletePathEvaluation,
    PortalMotorTimeLowerBound,
    SearchSettings,
    TimeBoundRequest,
    astar_junction_path,
    branch_and_bound_junction_paths,
    expand_junction_path,
    PlannerOptimizationPolicy,
    WarmStartDeadlineSettings,
    SparseSpecialistPolicySettings,
    SparseSpecialistMode,
)

from .common.certification import certify_optimized_route as independent_certify
from .common.environment import environment_metadata
from .common.io import write_json
from .common.routes import RouteEvaluator, build_case, route_to_record
from .config import (
    BOUND_CASES,
    CORE_OPTIMIZATION,
    BOUND_OPTIMIZATION,
    MAX_EXHAUSTIVE_PATHS,
    SMOKE_BOUND_CASES,
    SMOKE_OPTIMIZATION,
    MazeCase,
    OptimizationConfig,
)

ADMISSIBILITY_TOLERANCE = 2.0e-9


def check_admissibility(lower_bound: float, best_completion_time: float, tolerance: float = ADMISSIBILITY_TOLERANCE) -> float:
    residual = float(lower_bound) - float(best_completion_time)
    if residual > tolerance:
        raise AssertionError(
            f"lower-bound admissibility violation: LB={lower_bound:.17g}, "
            f"completion={best_completion_time:.17g}, residual={residual:.3e}, tol={tolerance:.3e}"
        )
    return residual


def enumerate_simple_junction_paths(graph, source: int, target: int, *, max_paths: int) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    stack: list[tuple[int, tuple[int, ...], int]] = [(source, (source,), 1 << source)]
    while stack:
        node, path, mask = stack.pop()
        if node == target:
            paths.append(path)
            if len(paths) > max_paths:
                raise RuntimeError(f"simple path count exceeded safeguard {max_paths}")
            continue
        edges = sorted(graph.adj[node], key=lambda edge: edge.to, reverse=True)
        for edge in edges:
            if (mask >> edge.to) & 1:
                continue
            stack.append((edge.to, (*path, edge.to), mask | (1 << edge.to)))
    return paths


def _make_bound(maze, config: OptimizationConfig, level: str) -> PortalMotorTimeLowerBound:
    if level not in {"basic", "two_sided", "projection12", "complete_cover"}:
        raise ValueError(level)
    return PortalMotorTimeLowerBound(
        maximum_iterations=100,
        cache_warm_starts=True,
        maze=maze,
        body_length=config.body_length,
        body_height=config.body_height,
        use_inscribed_disk_gates=True,
        use_two_sided_time=level != "basic",
        use_dual_certificate=False,
        use_complete_cover_bound=level == "complete_cover",
        complete_refinement_gap=1.0e6,
        projection_directions=12 if level in {"projection12", "complete_cover"} else 0,
    )


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def run_case(case: MazeCase, config: OptimizationConfig, *, max_paths: int = MAX_EXHAUSTIVE_PATHS) -> dict[str, Any]:
    start = (0, 0)
    maze, graph = build_case(case, start=start)
    source = graph.index[start]
    target = graph.index[maze.goal]
    paths = enumerate_simple_junction_paths(graph, source, target, max_paths=max_paths)
    astar_nodes = tuple(astar_junction_path(graph, start, maze.goal))

    exhaustive: dict[tuple[int, ...], dict[str, Any]] = {}
    base_policy = PlannerOptimizationPolicy.integrated_v10()
    benchmark_policy = replace(
        base_policy,
        warm_start=replace(
            base_policy.warm_start,
            deadlines=WarmStartDeadlineSettings(
                phase_one_seconds=6.0,
                primary_geometry_seconds=6.0,
                projection_seconds=6.0,
                alternate_geometry_seconds=6.0,
                exact_ranking_seconds=5.0,
                full_time_seconds=8.0,
                total_pipeline_seconds=24.0,
                worker_startup_seconds=5.0,
                terminate_grace_seconds=0.2,
            ),
        ),
        sparse_specialist=replace(base_policy.sparse_specialist, mode=SparseSpecialistMode.OFF),
    )
    with RouteEvaluator(config, policy=benchmark_policy) as evaluator:
        exhaustive_started = time.perf_counter()
        for nodes in paths:
            cells = tuple(expand_junction_path(maze, graph, nodes))
            route = evaluator.optimize(cells)
            certification = independent_certify(route, config)
            if not certification["certified"]:
                raise RuntimeError(f"exhaustive route failed independent certification: {case.name}, {nodes}")
            exhaustive[nodes] = {
                "nodes": nodes,
                "cells": cells,
                "time": float(route.time),
                "certification": certification,
                "route": route_to_record(route, config),
            }
        exhaustive_wall = time.perf_counter() - exhaustive_started

        best_nodes = min(paths, key=lambda p: exhaustive[p]["time"])
        best_time = float(exhaustive[best_nodes]["time"])

        prefix_to_best: dict[tuple[int, ...], float] = {}
        for nodes in paths:
            t = float(exhaustive[nodes]["time"])
            for end in range(1, len(nodes) + 1):
                prefix = nodes[:end]
                old = prefix_to_best.get(prefix)
                if old is None or t < old:
                    prefix_to_best[prefix] = t

        bound_levels: dict[str, Any] = {}
        start_point = (start[0] + 0.5, start[1] + 0.5)
        goal_point = (maze.goal[0] + 0.5, maze.goal[1] + 0.5)
        for level in ("basic", "two_sided", "projection12", "complete_cover"):
            bound = _make_bound(maze, config, level)
            ratios: list[float] = []
            residuals: list[float] = []
            prefix_rows: list[dict[str, Any]] = []
            started = time.perf_counter()
            for prefix, completion_time in sorted(prefix_to_best.items()):
                cells = tuple(expand_junction_path(maze, graph, prefix))
                request = TimeBoundRequest(
                    cells, start_point, goal_point, config.init_w,
                    goal_radius=math.sqrt(0.5),
                )
                result = bound.evaluate(request)
                if level == "complete_cover" and prefix[-1] == target:
                    result = bound.refine_complete(request, completion_time + 1.0, result)
                residual = check_admissibility(result.time_lower_bound, completion_time)
                residuals.append(residual)
                if completion_time > 0.0:
                    ratios.append(result.time_lower_bound / completion_time)
                prefix_rows.append({
                    "junction_prefix": prefix,
                    "cell_prefix": cells,
                    "best_completion_time": completion_time,
                    "lower_bound": float(result.time_lower_bound),
                    "bound_name": result.name,
                    "ratio": result.time_lower_bound / completion_time if completion_time > 0.0 else None,
                    "admissibility_residual": residual,
                    "solve_seconds": float(result.solve_seconds),
                })
            prefix_wall = time.perf_counter() - started

            search_bound = _make_bound(maze, config, level)
            def cached_complete(nodes: tuple[int, ...], cells: tuple[tuple[int, int], ...]):
                record = exhaustive[tuple(nodes)]
                return CompletePathEvaluation(record["time"], tuple(nodes), tuple(cells))

            search_started = time.perf_counter()
            search = branch_and_bound_junction_paths(
                maze,
                graph,
                start,
                maze.goal,
                lower_bound=search_bound,
                complete_path_time=cached_complete,
                init_w=config.init_w,
                settings=SearchSettings(
                    maximum_node_visits=1,
                    maximum_expansions=20_000,
                    use_block_cut_pruning=True,
                    use_no_revisit_reachability=True,
                    use_complete_bound_refinement=(level == "complete_cover"),
                ),
                seed_junction_path=astar_nodes,
                topology_quotient=None,
            )
            search_wall = time.perf_counter() - search_started
            if not search.exhausted:
                raise RuntimeError(f"B&B ablation did not exhaust for {case.name}/{level}")
            match_error = abs(float(search.best_time) - best_time)
            if match_error > 2.0e-8:
                raise AssertionError(
                    f"B&B/exhaustive mismatch for {case.name}/{level}: {search.best_time} vs {best_time}"
                )
            complete_optimizations = 1 + int(search.complete_paths_evaluated)
            avoided_fraction = 1.0 - complete_optimizations / len(paths)
            bound_levels[level] = {
                "prefix_count": len(prefix_rows),
                "tightness_ratio": _stats(ratios),
                "admissibility_residual": _stats(residuals),
                "maximum_admissibility_residual": max(residuals) if residuals else None,
                "bound_evaluation_wall_seconds": prefix_wall,
                "bound_statistics": asdict(bound.statistics),
                "prefixes": prefix_rows,
                "search": {
                    "best_time": float(search.best_time),
                    "match_error": match_error,
                    "complete_route_optimizations_including_seed": complete_optimizations,
                    "complete_route_optimizations_avoided_fraction": avoided_fraction,
                    "wall_seconds_with_cached_complete_routes": search_wall,
                    "statistics": asdict(search),
                    "bound_statistics": asdict(search_bound.statistics),
                },
            }

        production = bound_levels["complete_cover"]
        return {
            "case": case.to_dict(),
            "maze": {"goal": maze.goal, "junctions": len(graph.nodes), "cells": maze.width * maze.height},
            "simple_path_count": len(paths),
            "exhaustive_wall_seconds": exhaustive_wall,
            "exhaustive_best_time": best_time,
            "exhaustive_best_junction_path": best_nodes,
            "exhaustive_routes": [exhaustive[nodes] for nodes in paths],
            "production_bound": {
                "bb_best_time": production["search"]["best_time"],
                "match_error": production["search"]["match_error"],
                "complete_route_optimizations_including_seed": production["search"]["complete_route_optimizations_including_seed"],
                "complete_route_optimizations_avoided_fraction": production["search"]["complete_route_optimizations_avoided_fraction"],
                "prefix_tightness_ratio": production["tightness_ratio"],
                "maximum_admissibility_residual": production["maximum_admissibility_residual"],
            },
            "ablation": bound_levels,
        }


def aggregate(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    avoided = [row["production_bound"]["complete_route_optimizations_avoided_fraction"] for row in cases]
    residuals = [row["production_bound"]["maximum_admissibility_residual"] for row in cases]
    ratios = [row["production_bound"]["prefix_tightness_ratio"]["median"] for row in cases]
    return {
        "cases": len(cases),
        "bb_matches_exhaustive": sum(row["production_bound"]["match_error"] <= 2.0e-8 for row in cases),
        "total_simple_paths": sum(row["simple_path_count"] for row in cases),
        "total_prefixes_checked": sum(row["ablation"]["complete_cover"]["prefix_count"] for row in cases),
        "admissibility_violations": sum(
            1
            for row in cases
            for prefix in row["ablation"]["complete_cover"]["prefixes"]
            if prefix["admissibility_residual"] > ADMISSIBILITY_TOLERANCE
        ),
        "complete_route_optimizations_avoided_fraction": _stats(avoided),
        "median_prefix_tightness_ratio_across_cases": statistics.median(ratios) if ratios else None,
        "maximum_observed_admissibility_residual": max(residuals) if residuals else None,
        "admissibility_tolerance": ADMISSIBILITY_TOLERANCE,
    }


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    cases = BOUND_CASES if profile == "core" else SMOKE_BOUND_CASES
    config = BOUND_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    rows = [run_case(case, config) for case in cases]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "lower_bounds",
        "profile": profile,
        "environment": environment_metadata(config),
        "settings": {
            "maximum_simple_paths": MAX_EXHAUSTIVE_PATHS,
            "admissibility_tolerance": ADMISSIBILITY_TOLERANCE,
            "topology_policy": "simple junction paths, maximum_node_visits=1, topology quotient disabled",
        },
        "cases": rows,
        "aggregate": aggregate(rows),
    }
    write_json(output_dir / "lower_bounds.json", result)
    return result
