"""Shared helpers for the historical-five homotopy optimizer research campaign."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from typing import Sequence

import numpy as np

from benchmarks.common.certification import certify_parameters_on_problem
from mazegen import MazeGenerator
from optimization import ConstraintPool, knot_parameters_to_raw
from planning.maze_routes import shortest_cell_path

from .active_set import (
    CertifiedActiveSetResult,
    CertifiedActiveSetSettings,
    CurvatureObjective,
    TimeObjective,
    run_certified_active_set,
)
from .goal_entry import build_goal_entry_problem
from .homotopy import prolong_reduced_to_full
from .transition_guard import TransitionOverlapConstraint


def historical_cells(seed: int) -> tuple[tuple[int, int], ...]:
    maze = MazeGenerator(8, 8, int(seed)).generate()
    return tuple(shortest_cell_path(maze, (0, 0), maze.goal))


def build_historical_pair(seed: int):
    cells = historical_cells(seed)
    reduced = build_goal_entry_problem(cells, corridor_mode="maximal_runs")
    full = build_goal_entry_problem(cells, corridor_mode="overlapping_cover")
    return cells, reduced, full


def parameter_sha256(x: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


def geometry_metrics(problem, x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    stations = a[0::2]
    prev = np.concatenate(([0.0], stations[:-1]))
    lengths = stations - prev
    raw = knot_parameters_to_raw(a, initial_k=problem.initial_state.k)
    return {
        "segments": int(len(stations)),
        "variables": int(a.size),
        "total_length": float(stations[-1]),
        "minimum_segment_length": float(np.min(lengths)),
        "maximum_segment_length": float(np.max(lengths)),
        "maximum_abs_sigma": float(np.max(np.abs(raw[1::2]))),
    }


def make_guard(cells, reduced, full, q: float):
    return TransitionOverlapConstraint(
        tuple(cells), reduced, full, guard_fraction=float(q)
    )


def run_curvature(
    reduced,
    x0,
    *,
    guard,
    accepted_steps: int,
    batch_schedule=(4, 2, 1),
    trust_schedule=(0.02, 0.01, 0.005),
    max_filter_violation=1.0e-4,
    pool=None,
    wall=120.0,
    maximum_raw_iterations=100,
    pool_retention="carry",
    use_quadratic_model=True,
    highs_time_limit=2.0,
    highs_qp_iteration_limit=1000,
    highs_threads=1,
    init_w=0.8,
    terminal_w_max=None,
    n_scan=96,
    envelope_scan=48,
    domain_scan=96,
) -> CertifiedActiveSetResult:
    settings = CertifiedActiveSetSettings(
        pool_strategy="near_active",
        batch_schedule=tuple(batch_schedule),
        trust_radius_schedule=tuple(trust_schedule),
        maximum_accepted_steps=int(accepted_steps),
        maximum_trials=max(96, 12 * int(accepted_steps)),
        maximum_wall_seconds=float(wall),
        maximum_filter_violation=float(max_filter_violation),
        maximum_raw_iterations=int(maximum_raw_iterations),
        pool_retention=str(pool_retention),
        use_quadratic_model=bool(use_quadratic_model),
        highs_time_limit=float(highs_time_limit),
        highs_qp_iteration_limit=int(highs_qp_iteration_limit),
        highs_threads=int(highs_threads),
    )
    return run_certified_active_set(
        reduced,
        x0,
        objective=CurvatureObjective(reduced.initial_state.k),
        objective_kind="curvature",
        settings=settings,
        pool=pool,
        init_w=float(init_w),
        terminal_w_max=terminal_w_max,
        n_scan=int(n_scan),
        envelope_scan=int(envelope_scan),
        domain_scan=int(domain_scan),
        checkpoint_inequality=guard,
    )


def run_time(
    problem,
    x0,
    *,
    guard=None,
    accepted_steps: int,
    batch_schedule=(8, 4, 2, 1),
    trust_schedule=(0.04, 0.02, 0.01),
    max_filter_violation=1.0e-5,
    pool=None,
    wall=180.0,
    maximum_raw_iterations=100,
    pool_retention="carry",
    use_quadratic_model=True,
    highs_time_limit=2.0,
    highs_qp_iteration_limit=1000,
    highs_threads=1,
    init_w=0.8,
    terminal_w_max=None,
    n_scan=96,
    envelope_scan=48,
    domain_scan=96,
) -> CertifiedActiveSetResult:
    settings = CertifiedActiveSetSettings(
        pool_strategy="near_active",
        batch_schedule=tuple(batch_schedule),
        trust_radius_schedule=tuple(trust_schedule),
        maximum_accepted_steps=int(accepted_steps),
        maximum_trials=max(128, 12 * int(accepted_steps)),
        maximum_wall_seconds=float(wall),
        maximum_filter_violation=float(max_filter_violation),
        maximum_raw_iterations=int(maximum_raw_iterations),
        pool_retention=str(pool_retention),
        use_quadratic_model=bool(use_quadratic_model),
        highs_time_limit=float(highs_time_limit),
        highs_qp_iteration_limit=int(highs_qp_iteration_limit),
        highs_threads=int(highs_threads),
    )
    return run_certified_active_set(
        problem,
        x0,
        objective=TimeObjective(
            problem.initial_state.k,
            init_w=float(init_w),
            terminal_w_max=terminal_w_max,
            n_scan=int(n_scan),
            envelope_scan=int(envelope_scan),
            domain_scan=int(domain_scan),
        ),
        objective_kind="time",
        settings=settings,
        pool=pool,
        init_w=float(init_w),
        terminal_w_max=terminal_w_max,
        n_scan=int(n_scan),
        envelope_scan=int(envelope_scan),
        domain_scan=int(domain_scan),
        checkpoint_inequality=guard,
    )


def stage_summary(result: CertifiedActiveSetResult) -> dict[str, object]:
    return {
        "objective": float(result.objective),
        "scalar_time": float(result.scalar_time),
        "curvature_energy": float(result.curvature_energy),
        "certified": bool(result.certified),
        "accepted_steps": int(result.accepted_steps),
        "trial_count": len(result.trials),
        "pool_size": int(result.pool.size),
        "stop_reason": result.stop_reason,
        "wall_seconds": float(result.wall_seconds),
        "parameter_sha256": parameter_sha256(result.parameters),
        "checkpoints": [asdict(c) for c in result.checkpoints],
        "trials": [asdict(t) for t in result.trials],
    }


def lift_summary(cells, reduced, full, x, q: float) -> tuple[object, dict[str, object]]:
    lift = prolong_reduced_to_full(
        cells, reduced, full, x, guard_fraction=float(q)
    )
    cert = certify_parameters_on_problem(
        full, lift.parameters, tolerance=2.0e-7, maximum_abs_sigma=50.0
    )
    return lift, {
        "guard_fraction": float(q),
        "minimum_child_length": float(lift.minimum_child_length),
        "maximum_child_length_ratio": float(lift.maximum_child_length_ratio),
        "certification": cert,
        "metrics": geometry_metrics(full, lift.parameters),
        "parameter_sha256": parameter_sha256(lift.parameters),
    }
