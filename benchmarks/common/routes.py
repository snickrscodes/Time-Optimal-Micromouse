from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from mazegen import MazeGenerator, compress_to_graph
from main import OptimizedRoute, add_random_openings, optimize_complete_route_class
from optimization import knot_parameters_to_raw
from planning import PlannerOptimizationPolicy, create_warm_start_stage_runner

from .io import json_safe
from visualization.records import route_snapshot


def build_case(case: Any, *, start: tuple[int, int] = (0, 0)):
    maze = MazeGenerator(case.width, case.height, case.seed).generate()
    maze = add_random_openings(maze, case.extra_openings, seed=case.resolved_opening_seed())
    graph = compress_to_graph(maze, start, maze.goal)
    return maze, graph


def topology_length_cells(cells: Sequence[tuple[int, int]]) -> float:
    return float(max(0, len(cells) - 1))


def optimized_path_length(route: OptimizedRoute) -> float:
    raw = knot_parameters_to_raw(route.parameters, initial_k=route.initial_state.k)
    return float(math.fsum(float(v) for v in raw[0::2]))


def route_to_record(route: OptimizedRoute, config: Any) -> dict[str, Any]:
    """Benchmark compatibility wrapper around the neutral route snapshot."""
    return route_snapshot(route, config=config.to_dict())



class RouteEvaluator:
    """Production complete-route optimizer with deterministic benchmark settings."""

    def __init__(self, config: Any, *, policy: PlannerOptimizationPolicy | None = None):
        self.config = config
        self.policy = policy or PlannerOptimizationPolicy.integrated_v10()
        self.runner = None
        self.cache: dict[tuple[tuple[int, int], ...], OptimizedRoute] = {}
        self.optimization_seconds: dict[tuple[tuple[int, int], ...], float] = {}

    def __enter__(self):
        self.runner = create_warm_start_stage_runner(self.policy.warm_start)
        self.runner.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.runner is not None:
            self.runner.close()
        self.runner = None

    def optimize(
        self,
        cells: Sequence[tuple[int, int]],
        *,
        curvature_warm_start: bool = True,
        length_warm_start: bool = True,
    ) -> OptimizedRoute:
        key = tuple(cells)
        if curvature_warm_start and length_warm_start and key in self.cache:
            return self.cache[key]
        c = self.config
        started = time.perf_counter()
        route = optimize_complete_route_class(
            ((key, ()),),
            init_w=c.init_w,
            optimization_mode="time",
            curvature_warm_start=curvature_warm_start,
            length_warm_start=length_warm_start,
            curvature_iterations=c.curvature_iterations,
            length_iterations=c.length_iterations,
            time_iterations=c.time_iterations,
            class_time_pilot_iterations=c.class_time_pilot_iterations,
            maximum_exchange_rounds=c.maximum_exchange_rounds,
            feasibility_tolerance=c.feasibility_tolerance,
            curvature_regularization=c.curvature_regularization,
            length_curvature_regularization=c.length_curvature_regularization,
            corridor_mode=c.corridor_mode,
            body_length=c.body_length,
            body_height=c.body_height,
            geometry_refinement=c.geometry_refinement,
            curvature_slope_limit=c.curvature_slope_limit,
            n_scan=c.n_scan,
            envelope_scan=c.envelope_scan,
            domain_scan=c.domain_scan,
            optimization_policy=self.policy,
            warm_start_stage_runner=self.runner,
        )
        elapsed = time.perf_counter() - started
        if curvature_warm_start and length_warm_start:
            self.cache[key] = route
            self.optimization_seconds[key] = elapsed
        return route
