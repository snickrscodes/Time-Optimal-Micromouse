"""Visualize the naive A* seed and final branch-and-bound path side by side.

The script is intentionally a thin application layer over the production
planner.  It:

1. generates a random maze and optionally adds extra legal openings;
2. finds a shortest-distance seed topology with naive A*;
3. optimizes that complete route using the existing constrained path optimizer;
4. runs the hierarchical certified path-indexed branch-and-bound search;
5. densely samples both optimized clothoid geometries; and
6. writes a matched Matplotlib comparison figure and JSON metadata.

Examples
--------
A small exhaustive visualization with full time optimization::

    python examples/visualize_planner.py --seed 7 --width 4 --height 4 \\
        --extra-openings 3 --output planner_comparison.png

A faster geometry-only preview::

    python examples/visualize_planner.py --optimization-mode curvature \\
        --width 5 --height 5 --extra-openings 4 --show

A raw Wilson maze has only one simple route under the current generator.  Set
``--extra-openings`` above zero when you want a meaningful topology search.
The planner itself never assumes the maze is perfect or acyclic.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mazegen import Cell, Maze, MazeGenerator, compress_to_graph
from optimization import (
    CurvatureSlopeConstraint,
    ExchangeSettings,
    GeometryState,
    PathOptimizerSettings,
    SLSQPSettings,
    SupervisedBatchRunner,
    compile_geometry_path,
    knot_parameters_to_raw,
    optimize_path,
    path_length_value_and_raw_gradient,
    project_path_to_feasibility,
    reverse_solver,
    scalar_reverse_solver,
    separate_rectangle_path,
)
from planning import (
    DEFAULT_BODY_HEIGHT,
    DEFAULT_BODY_LENGTH,
    CompletePathEvaluation,
    OpenRoomSpan,
    OpenRoomTopologyQuotient,
    PortalMotorTimeLowerBound,
    SearchSettings,
    astar_junction_path,
    branch_and_bound_junction_paths,
    build_route_optimization_problem,
    expand_junction_path,
    stable_knot_bounds,
    PlannerArchitectureMode,
    PlannerOptimizationPolicy,
    PrimaryFilterSQPPolicySettings,
    PrimaryTimeBackendMode,
    SparseSpecialistMode,
    SparseSpecialistPolicySettings,
    SparseSpecialistRecord,
    WarmStartDeadlineSettings,
    WarmStartInitializerMode,
    WarmStartOptimizationConfig,
    WarmStartRunRecord,
    WarmStartScheduleSettings,
    WarmStartSchedulingMode,
    create_warm_start_stage_runner,
    execute_bounded_warm_start_schedule,
    run_sparse_specialist_polish,
    load_maze_scenario,
)
from planning.active_basis_optimizer import ActiveBasisExecutionSettings, optimize_active_basis_route
from visualization.comparison import draw_solution as _draw_solution, render_comparison as _render_comparison
from visualization.maze import draw_maze_walls
from visualization.records import route_snapshot
from visualization.sampling import sample_geometry_parameters
from visualization.trajectory import topology_centerline

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StageRecord:
    name: str
    time: float
    feasible: bool
    solver_success: bool | None
    message: str
    endpoint_error: float
    corridor_upper_bound: float
    total_length: float
    minimum_segment_length: float
    maximum_abs_sigma: float
    elapsed_seconds: float
    # Diagnostic-only optimizer telemetry. These fields do not participate in
    # route selection or certification and default to None for scheduler stages
    # whose worker result intentionally exposes only high-level status.
    optimizer_iterations: int | None = None
    objective_calls: int | None = None
    gradient_calls: int | None = None
    constraint_calls: int | None = None
    exchange_rounds: int | None = None
    kkt_stationarity_inf: float | None = None
    optimizer_backend: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizedRoute:
    cells: tuple[Cell, ...]
    parameters: Array
    initial_state: GeometryState
    time: float
    selected_stage: str
    stages: tuple[StageRecord, ...]
    warm_start_record: WarmStartRunRecord | None = None
    sparse_specialist_record: SparseSpecialistRecord | None = None
    architecture: str = "legacy_v9"
    active_basis_record: dict[str, Any] | None = None


def add_random_openings(maze: Maze, count: int, *, seed: int) -> Maze:
    """Add legal grid edges without relying on Wilson/tree properties."""
    if count <= 0:
        return maze
    connections = maze.connection_dict()
    candidates: list[tuple[Cell, Cell]] = []
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
    return Maze.from_connections(
        maze.width,
        maze.height,
        connections,
        maze.goal,
    )


def optimizer_settings(
    maximum_iterations: int,
    maximum_exchange_rounds: int,
    *,
    n_scan: int,
    envelope_scan: int,
    domain_scan: int,
) -> PathOptimizerSettings:
    return PathOptimizerSettings(
        exchange=ExchangeSettings(
            maximum_rounds=maximum_exchange_rounds,
            require_solver_success=False,
            slsqp=SLSQPSettings(
                max_iterations=maximum_iterations,
                ftol=1.0e-10,
                display=False,
            ),
        ),
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_scan=domain_scan,
    )


def wrapped_angle_error(value: float, target: float) -> float:
    return abs(math.atan2(math.sin(value - target), math.cos(value - target)))


def certify_parameters(
    problem,
    parameters: Sequence[float],
    *,
    tolerance: float,
    maximum_abs_sigma: float | None = None,
) -> tuple[bool, GeometryState, float, float]:
    raw = knot_parameters_to_raw(
        parameters,
        initial_k=problem.initial_state.k,
    )
    geometry = compile_geometry_path(raw, problem.initial_state)
    final = geometry.final_state
    endpoint_error = problem.terminal_violation(final)
    separation = separate_rectangle_path(
        parameters,
        problem.initial_state,
        problem.corridor,
    )
    slope_feasible = (
        True
        if maximum_abs_sigma is None
        else max(abs(float(value)) for value in raw[1::2])
        <= maximum_abs_sigma + tolerance
    )
    feasible = (
        endpoint_error <= tolerance
        and separation.certified(tolerance)
        and slope_feasible
        and np.all(np.isfinite(parameters))
    )
    return (
        bool(feasible),
        final,
        float(endpoint_error),
        float(separation.worst_upper_bound),
    )


def scalar_time(
    problem,
    parameters: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None = None,
    n_scan: int,
    envelope_scan: int,
    domain_scan: int,
) -> float:
    raw = knot_parameters_to_raw(
        parameters,
        initial_k=problem.initial_state.k,
    )
    value = scalar_reverse_solver.evaluate_time_scalar(
        raw,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        initial_k=problem.initial_state.k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_scan=domain_scan,
    )
    return float(value)


def optimize_complete_route_class(
    route_variants: Sequence[tuple[Sequence[Cell], Sequence[OpenRoomSpan]]],
    *,
    init_w: float,
    terminal_w_max: float | None = None,
    optimization_mode: str,
    curvature_warm_start: bool,
    length_warm_start: bool = True,
    curvature_iterations: int = 100,
    length_iterations: int = 60,
    time_iterations: int,
    class_time_pilot_iterations: int = 8,
    maximum_exchange_rounds: int,
    feasibility_tolerance: float,
    curvature_regularization: float,
    length_curvature_regularization: float = 1.0e-5,
    corridor_mode: str = "overlapping_cover",
    body_length: float = DEFAULT_BODY_LENGTH,
    body_height: float = DEFAULT_BODY_HEIGHT,
    geometry_refinement: int = 1,
    curvature_slope_limit: float | None = 50.0,
    n_scan: int,
    envelope_scan: int,
    domain_scan: int,
    warm_start_schedule: WarmStartScheduleSettings | None = None,
    optimization_policy: PlannerOptimizationPolicy | None = None,
    warm_start_stage_runner: SupervisedBatchRunner | None = None,
) -> OptimizedRoute:
    """Optimize one continuous quotient class from all member initializers.

    Every graph member is first explored by the inexpensive geometry stages in
    the same wall-aware union corridor.  For multi-member classes, a short time
    pilot is launched from each member basin; one full time solve then continues
    from the best pilot.  All feasible pilot and warm candidates remain eligible
    fallbacks.  This avoids representative-path bias while reducing the dominant
    reverse-profile iteration budget for graph paths that encode the same room.
    """
    frozen_variants = tuple(
        (tuple(cells), tuple(spans)) for cells, spans in route_variants
    )
    if not frozen_variants:
        raise ValueError("route_variants must not be empty")
    if class_time_pilot_iterations < 0:
        raise ValueError("class_time_pilot_iterations must be nonnegative")
    if terminal_w_max is None:
        terminal_w_max = init_w
    terminal_w_max = float(terminal_w_max)
    if not math.isfinite(terminal_w_max) or terminal_w_max <= 0.0:
        raise ValueError("terminal_w_max must be finite and positive")
    problems = tuple(
        build_route_optimization_problem(
            cells,
            body_length=body_length,
            body_height=body_height,
            corridor_mode=corridor_mode,
            refinement_factor=geometry_refinement,
            open_room_spans=spans,
        )
        for cells, spans in frozen_variants
    )
    reference = problems[0]
    for problem in problems[1:]:
        if problem.initial_state != reference.initial_state:
            raise ValueError("quotient initializers have different initial states")
        if problem.endpoint_target != reference.endpoint_target:
            raise ValueError("quotient initializers have different endpoint targets")
        if problem.terminal_cell != reference.terminal_cell:
            raise ValueError("quotient initializers have different terminal cells")

    policy = optimization_policy or PlannerOptimizationPolicy.integrated_v10()
    effective_schedule = warm_start_schedule
    if (
        effective_schedule is None
        and optimization_mode == "time"
        and policy.architecture is PlannerArchitectureMode.INTEGRATED_V10
    ):
        effective_schedule = policy.warm_start

    if policy.architecture is PlannerArchitectureMode.ACTIVE_BASIS_V11:
        if optimization_mode != "time":
            raise ValueError("active_basis_v11 supports time optimization only")
        if len(frozen_variants) != 1:
            raise ValueError(
                "active_basis_v11 currently requires one explicit topology; "
                "disable topology quotienting for production V11 runs"
            )
        selected_cells, selected_spans = frozen_variants[0]
        # The qualification campaign used the authoritative reverse-profile
        # resolution 96/48/96.  V11 deliberately freezes that resolution rather
        # than inheriting lower preview settings from the legacy CLI.
        active = optimize_active_basis_route(
            selected_cells,
            open_room_spans=selected_spans,
            body_length=body_length,
            body_height=body_height,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            refinement_factor=geometry_refinement,
            n_scan=96,
            envelope_scan=48,
            domain_scan=96,
            execution=ActiveBasisExecutionSettings(
                work_root=policy.active_basis.work_root,
                require_convergence=policy.active_basis.require_convergence,
            ),
        )
        return OptimizedRoute(
            active.cells, active.parameters, active.problem.initial_state, active.time,
            "active_basis_v11", (), None, None, policy.architecture.value, active.summary,
        )

    if effective_schedule is not None:
        if optimization_mode != "time":
            raise ValueError("bounded warm-start scheduling currently supports time optimization only")
        scheduled = execute_bounded_warm_start_schedule(
            problems,
            schedule=effective_schedule,
            config=WarmStartOptimizationConfig(
                init_w=init_w,
                terminal_w_max=terminal_w_max,
                curvature_iterations=curvature_iterations,
                length_iterations=length_iterations,
                time_iterations=time_iterations,
                maximum_exchange_rounds=maximum_exchange_rounds,
                feasibility_tolerance=feasibility_tolerance,
                curvature_regularization=curvature_regularization,
                length_curvature_regularization=length_curvature_regularization,
                curvature_slope_limit=curvature_slope_limit,
                n_scan=n_scan,
                envelope_scan=envelope_scan,
                domain_scan=domain_scan,
                curvature_enabled=curvature_warm_start,
                length_enabled=length_warm_start,
                class_time_pilot_iterations=class_time_pilot_iterations,
            ),
            stage_runner=warm_start_stage_runner,
        )
        selected_problem = problems[scheduled.problem_index]
        translated = tuple(
            StageRecord(
                name=stage.name,
                time=math.inf if stage.exact_time is None else stage.exact_time,
                feasible=stage.strict_certified,
                solver_success=stage.solver_success,
                message=f"{stage.status}: {stage.message}",
                endpoint_error=math.inf if stage.endpoint_error is None else stage.endpoint_error,
                corridor_upper_bound=math.inf if stage.corridor_upper_bound is None else stage.corridor_upper_bound,
                total_length=math.nan,
                minimum_segment_length=math.nan,
                maximum_abs_sigma=math.nan,
                elapsed_seconds=stage.elapsed_seconds,
            )
            for stage in scheduled.stages
        )
        specialist = run_sparse_specialist_polish(
            selected_problem,
            scheduled.parameters,
            scheduled.time,
            pool=scheduled.pool,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
            feasibility_tolerance=feasibility_tolerance,
            curvature_slope_limit=curvature_slope_limit,
            slsqp_seconds=scheduled.run_record.full_time_seconds,
            settings=policy.sparse_specialist,
        )
        parameters = specialist.parameters
        route_time = specialist.time
        selected_stage = (
            f"{scheduled.selected_stage}+sparse_specialist"
            if specialist.record.planner_promoted
            else scheduled.selected_stage
        )
        return OptimizedRoute(
            selected_problem.cells, parameters, selected_problem.initial_state,
            route_time, selected_stage, translated, scheduled.run_record,
            specialist.record, policy.architecture.value,
        )

    base_bounds = tuple(stable_knot_bounds(problem.initial_parameters) for problem in problems)
    slope_constraints = tuple(
        problem.inequalities(
            None
            if curvature_slope_limit is None
            else CurvatureSlopeConstraint(
                curvature_slope_limit,
                initial_k=problem.initial_state.k,
            )
        )
        for problem in problems
    )
    stages: list[StageRecord] = []
    candidates: list[tuple[float, str, Array, int]] = []
    warm_candidates: list[tuple[float, str, Array, int]] = []
    warm_candidates_by_problem: list[list[tuple[float, str, Array, int]]] = [
        [] for _ in problems
    ]
    warm_tolerance = max(1.0e-4, 100.0 * feasibility_tolerance)
    multiple = len(problems) > 1

    def stage_name(index: int, name: str) -> str:
        return f"member_{index}/{name}" if multiple else name

    def optimizer_telemetry(source: object | None) -> dict[str, object | None]:
        exchange = getattr(source, "exchange", source) if source is not None else None
        rounds = getattr(exchange, "rounds", ()) if exchange is not None else ()
        if not rounds:
            return {
                "optimizer_iterations": None,
                "objective_calls": None,
                "gradient_calls": None,
                "constraint_calls": None,
                "exchange_rounds": 0 if exchange is not None else None,
                "kkt_stationarity_inf": None,
                "optimizer_backend": None,
            }
        finite = [round_result.finite for round_result in rounds]
        final_kkt = getattr(exchange, "final_kkt", None)
        return {
            "optimizer_iterations": sum(item.iterations for item in finite),
            "objective_calls": sum(item.objective_calls for item in finite),
            "gradient_calls": sum(item.solver_result.gradient_evaluations for item in finite),
            "constraint_calls": sum(item.constraint_calls for item in finite),
            "exchange_rounds": len(rounds),
            "kkt_stationarity_inf": (
                None if final_kkt is None else float(final_kkt.stationarity_inf)
            ),
            "optimizer_backend": str(finite[-1].solver_result.backend),
        }

    def record_candidate(
        problem_index: int,
        name: str,
        parameters: Sequence[float],
        *,
        solver_success: bool | None,
        message: str,
        elapsed: float,
        known_time: float | None = None,
        diagnostics_source: object | None = None,
    ) -> bool:
        problem = problems[problem_index]
        label = stage_name(problem_index, name)
        array = np.asarray(parameters, dtype=float).copy()
        feasible, _final, endpoint_error, corridor_upper = certify_parameters(
            problem,
            array,
            tolerance=feasibility_tolerance,
            maximum_abs_sigma=curvature_slope_limit,
        )
        try:
            value = (
                float(known_time)
                if known_time is not None and math.isfinite(known_time)
                else scalar_time(
                    problem,
                    array,
                    init_w=init_w,
                    terminal_w_max=terminal_w_max,
                    n_scan=n_scan,
                    envelope_scan=envelope_scan,
                    domain_scan=domain_scan,
                )
            )
        except Exception as error:
            value = math.inf
            feasible = False
            message = f"{message}; time evaluation failed: {error}"
        if math.isfinite(value):
            if feasible:
                candidates.append((value, label, array, problem_index))
            if (
                endpoint_error <= warm_tolerance
                and corridor_upper <= warm_tolerance
                and np.all(np.isfinite(array))
            ):
                item = (value, label, array, problem_index)
                warm_candidates.append(item)
                warm_candidates_by_problem[problem_index].append(item)
        raw = knot_parameters_to_raw(
            array,
            initial_k=problem.initial_state.k,
        )
        lengths = raw[0::2]
        sigmas = raw[1::2]
        stages.append(
            StageRecord(
                name=label,
                time=value,
                feasible=feasible,
                solver_success=solver_success,
                message=message,
                endpoint_error=endpoint_error,
                corridor_upper_bound=corridor_upper,
                total_length=float(math.fsum(lengths)),
                minimum_segment_length=float(min(lengths)),
                maximum_abs_sigma=float(max(abs(v) for v in sigmas)),
                elapsed_seconds=elapsed,
                **optimizer_telemetry(diagnostics_source),
            )
        )
        return feasible

    for problem_index, problem in enumerate(problems):
        bounds = base_bounds[problem_index]
        slope_constraint = slope_constraints[problem_index]
        started = time.perf_counter()
        record_candidate(
            problem_index,
            "exact_initializer",
            problem.initial_parameters,
            solver_success=None,
            message="analytic G2 route initializer",
            elapsed=time.perf_counter() - started,
        )

        if curvature_warm_start and optimization_mode in {"curvature", "time"}:
            started = time.perf_counter()
            curvature_result = optimize_path(
                problem.initial_parameters,
                problem.corridor,
                problem.initial_state,
                init_w=init_w,
                time_weight=0.0,
                curvature_weight=1.0,
                endpoint_target=problem.endpoint_target,
                bounds=bounds,
                additional_inequalities=slope_constraint,
                settings=optimizer_settings(
                    curvature_iterations,
                    maximum_exchange_rounds,
                    n_scan=n_scan,
                    envelope_scan=envelope_scan,
                    domain_scan=domain_scan,
                ),
            )
            record_candidate(
                problem_index,
                "curvature_optimized",
                curvature_result.parameters,
                solver_success=curvature_result.success,
                message=curvature_result.message,
                elapsed=time.perf_counter() - started,
                known_time=curvature_result.time,
                diagnostics_source=curvature_result,
            )

        if length_warm_start and optimization_mode == "time":
            started = time.perf_counter()
            length_result = optimize_path(
                problem.initial_parameters,
                problem.corridor,
                problem.initial_state,
                init_w=init_w,
                time_weight=0.0,
                geometry_weight=1.0,
                curvature_weight=length_curvature_regularization,
                geometry_objective=path_length_value_and_raw_gradient,
                endpoint_target=problem.endpoint_target,
                bounds=bounds,
                additional_inequalities=slope_constraint,
                settings=optimizer_settings(
                    length_iterations,
                    maximum_exchange_rounds,
                    n_scan=n_scan,
                    envelope_scan=envelope_scan,
                    domain_scan=domain_scan,
                ),
            )
            length_feasible = record_candidate(
                problem_index,
                "length_optimized",
                length_result.parameters,
                solver_success=length_result.success,
                message=length_result.message,
                elapsed=time.perf_counter() - started,
                known_time=length_result.time,
                diagnostics_source=length_result,
            )
            if not length_feasible:
                started = time.perf_counter()
                length_projection = project_path_to_feasibility(
                    length_result.parameters,
                    problem.corridor,
                    problem.initial_state,
                    bounds=bounds,
                    endpoint_target=problem.endpoint_target,
                    additional_inequalities=slope_constraint,
                    pool=(
                        None
                        if length_result.exchange is None
                        else length_result.exchange.pool
                    ),
                    settings=optimizer_settings(
                        max(40, length_iterations),
                        maximum_exchange_rounds,
                        n_scan=n_scan,
                        envelope_scan=envelope_scan,
                        domain_scan=domain_scan,
                    ).exchange,
                )
                record_candidate(
                    problem_index,
                    "length_feasibility_polish",
                    length_projection.x,
                    solver_success=length_projection.success,
                    message=length_projection.message,
                    elapsed=time.perf_counter() - started,
                    diagnostics_source=length_projection,
                )

    if optimization_mode == "time":
        selected_warm: tuple[float, str, Array, int] | None = None
        use_class_pilots = (
            len(problems) > 1 and class_time_pilot_iterations > 0
        )
        if use_class_pilots:
            pilot_outputs: list[tuple[float, str, Array, int]] = []
            pilot_rounds = min(2, maximum_exchange_rounds)
            for problem_index, problem in enumerate(problems):
                local_warms = warm_candidates_by_problem[problem_index]
                if local_warms:
                    _value, _name, pilot_warm, _index = min(
                        local_warms, key=lambda item: item[0]
                    )
                else:
                    pilot_warm = problem.initial_parameters.copy()

                candidate_start = len(candidates)
                warm_start = len(warm_candidates)
                started = time.perf_counter()
                pilot_result = optimize_path(
                    pilot_warm,
                    problem.corridor,
                    problem.initial_state,
                    init_w=init_w,
                    terminal_w_max=terminal_w_max,
                    time_weight=1.0,
                    curvature_weight=curvature_regularization,
                    endpoint_target=problem.endpoint_target,
                    bounds=base_bounds[problem_index],
                    additional_inequalities=slope_constraints[problem_index],
                    settings=optimizer_settings(
                        class_time_pilot_iterations,
                        pilot_rounds,
                        n_scan=n_scan,
                        envelope_scan=envelope_scan,
                        domain_scan=domain_scan,
                    ),
                )
                pilot_feasible = record_candidate(
                    problem_index,
                    "time_pilot",
                    pilot_result.parameters,
                    solver_success=pilot_result.success,
                    message=pilot_result.message,
                    elapsed=time.perf_counter() - started,
                    known_time=pilot_result.time,
                    diagnostics_source=pilot_result,
                )
                if not pilot_feasible:
                    started = time.perf_counter()
                    pilot_projection = project_path_to_feasibility(
                        pilot_result.parameters,
                        problem.corridor,
                        problem.initial_state,
                        bounds=base_bounds[problem_index],
                        endpoint_target=problem.endpoint_target,
                        additional_inequalities=slope_constraints[problem_index],
                        pool=(
                            None
                            if pilot_result.exchange is None
                            else pilot_result.exchange.pool
                        ),
                        settings=optimizer_settings(
                            max(20, class_time_pilot_iterations),
                            pilot_rounds,
                            n_scan=n_scan,
                            envelope_scan=envelope_scan,
                            domain_scan=domain_scan,
                        ).exchange,
                    )
                    record_candidate(
                        problem_index,
                        "time_pilot_feasibility_polish",
                        pilot_projection.x,
                        solver_success=pilot_projection.success,
                        message=pilot_projection.message,
                        elapsed=time.perf_counter() - started,
                        diagnostics_source=pilot_projection,
                    )

                local_feasible = candidates[candidate_start:]
                if local_feasible:
                    pilot_outputs.append(
                        min(local_feasible, key=lambda item: item[0])
                    )
                    continue
                local_near = warm_candidates[warm_start:]
                if local_near:
                    pilot_outputs.append(min(local_near, key=lambda item: item[0]))
                elif local_warms:
                    pilot_outputs.append(min(local_warms, key=lambda item: item[0]))

            if pilot_outputs:
                selected_warm = min(pilot_outputs, key=lambda item: item[0])

        if selected_warm is None:
            if warm_candidates:
                selected_warm = min(warm_candidates, key=lambda item: item[0])
            else:
                selected_warm = (
                    math.inf,
                    "fallback_initializer",
                    problems[0].initial_parameters.copy(),
                    0,
                )

        _warm_time, _warm_name, warm, problem_index = selected_warm
        problem = problems[problem_index]
        started = time.perf_counter()
        time_result = optimize_path(
            warm,
            problem.corridor,
            problem.initial_state,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            time_weight=1.0,
            curvature_weight=curvature_regularization,
            endpoint_target=problem.endpoint_target,
            bounds=base_bounds[problem_index],
            additional_inequalities=slope_constraints[problem_index],
            settings=optimizer_settings(
                time_iterations,
                maximum_exchange_rounds,
                n_scan=n_scan,
                envelope_scan=envelope_scan,
                domain_scan=domain_scan,
            ),
        )
        time_feasible = record_candidate(
            problem_index,
            "time_optimized",
            time_result.parameters,
            solver_success=time_result.success,
            message=time_result.message,
            elapsed=time.perf_counter() - started,
            known_time=time_result.time,
            diagnostics_source=time_result,
        )
        if not time_feasible:
            started = time.perf_counter()
            time_projection = project_path_to_feasibility(
                time_result.parameters,
                problem.corridor,
                problem.initial_state,
                bounds=base_bounds[problem_index],
                endpoint_target=problem.endpoint_target,
                additional_inequalities=slope_constraints[problem_index],
                pool=None if time_result.exchange is None else time_result.exchange.pool,
                settings=optimizer_settings(
                    max(60, time_iterations),
                    maximum_exchange_rounds,
                    n_scan=n_scan,
                    envelope_scan=envelope_scan,
                    domain_scan=domain_scan,
                ).exchange,
            )
            record_candidate(
                problem_index,
                "time_feasibility_polish",
                time_projection.x,
                solver_success=time_projection.success,
                message=time_projection.message,
                elapsed=time.perf_counter() - started,
                diagnostics_source=time_projection,
            )

    if not candidates:
        details = "; ".join(
            f"{stage.name}: feasible={stage.feasible}, {stage.message}"
            for stage in stages
        )
        raise RuntimeError(f"no independently feasible route candidate: {details}")

    best_time, selected_stage, best_parameters, best_problem_index = min(
        candidates, key=lambda item: item[0]
    )
    best_problem = problems[best_problem_index]
    return OptimizedRoute(
        best_problem.cells,
        best_parameters,
        best_problem.initial_state,
        best_time,
        selected_stage,
        tuple(stages),
        architecture=policy.architecture.value,
    )


def optimize_complete_route(
    cells: Sequence[Cell],
    *,
    open_room_spans: Sequence[OpenRoomSpan] = (),
    **kwargs: Any,
) -> OptimizedRoute:
    """Backward-compatible one-member wrapper around class optimization."""
    return optimize_complete_route_class(((tuple(cells), tuple(open_room_spans)),), **kwargs)


def sample_geometry(
    route: OptimizedRoute,
    *,
    samples_per_unit: float,
    minimum_samples_per_segment: int,
) -> Array:
    """Backward-compatible XY view over the shared visualization trace."""
    return sample_geometry_parameters(
        route.parameters,
        route.initial_state,
        samples_per_unit=samples_per_unit,
        minimum_samples_per_segment=minimum_samples_per_segment,
    ).xy


def draw_maze(ax: Any, maze: Maze) -> None:
    """Backward-compatible wrapper around :mod:`visualization.maze`."""
    draw_maze_walls(ax, maze)


def draw_solution(*args: Any, **kwargs: Any) -> None:
    """Compatibility wrapper for the reusable route renderer."""
    _draw_solution(*args, **kwargs)


def render_comparison(*args: Any, **kwargs: Any) -> None:
    """Compatibility wrapper for the reusable comparison renderer."""
    _render_comparison(*args, **kwargs)


def parse_cell(text: str) -> Cell:
    try:
        x_text, y_text = text.split(",", maxsplit=1)
        return int(x_text), int(y_text)
    except Exception as error:
        raise argparse.ArgumentTypeError("cell must have form x,y") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze-file", type=Path, help="load an ame-maze-v1 JSON maze instead of generating one")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--seed", type=int, default=337080536)
    parser.add_argument("--start", type=parse_cell, default=(0, 0))
    parser.add_argument("--extra-openings", type=int, default=3)
    parser.add_argument("--opening-seed", type=int)
    parser.add_argument("--init-w", type=float, default=0.8)
    parser.add_argument("--goal-max-speed", type=float, help="maximum terminal speed on entering the goal cell; defaults to sqrt(init-w)")
    parser.add_argument(
        "--optimization-mode",
        choices=("none", "curvature", "time"),
        default="time",
        help="'time' is the production comparison; other modes are faster previews.",
    )
    parser.add_argument(
        "--planner-architecture",
        choices=tuple(mode.value for mode in PlannerArchitectureMode),
        default=PlannerArchitectureMode.INTEGRATED_V10.value,
        help="route-optimization architecture; V10 integrated mode activates bounded scheduling",
    )
    parser.add_argument(
        "--active-basis-work-dir",
        type=Path,
        help="persistent per-topology work/checkpoint root for active_basis_v11",
    )
    parser.add_argument(
        "--warm-start-mode",
        choices=tuple(mode.value for mode in WarmStartSchedulingMode),
        default=WarmStartSchedulingMode.BOUNDED_BEST_OF_BOTH.value,
        help="bounded warm-start scheduling policy used by the integrated architecture; V10 defaults to best-of-both for certified quality parity",
    )
    parser.add_argument(
        "--warm-start-initializer",
        choices=tuple(mode.value for mode in WarmStartInitializerMode),
        default=WarmStartInitializerMode.ANALYTIC_WITH_STRICT_PHASE_ONE_FALLBACK.value,
        help="initializer policy; V10 defaults to analytic-first with strict Phase-I fallback",
    )
    parser.add_argument(
        "--warm-start-telemetry",
        type=Path,
        help="append-only JSONL telemetry for bounded warm-start scheduling",
    )
    parser.add_argument(
        "--primary-time-backend",
        choices=tuple(mode.value for mode in PrimaryTimeBackendMode),
        default=PrimaryTimeBackendMode.AUTO_QUALIFIED_FILTER_SQP.value,
        help=(
            "final time-polish backend policy; auto uses filter-SQP only inside "
            "the validated <=22-variable/<=11-segment/<=2-cap envelope"
        ),
    )
    parser.add_argument(
        "--sparse-specialist",
        choices=tuple(mode.value for mode in SparseSpecialistMode),
        default=SparseSpecialistMode.AUTO.value,
        help=(
            "post-SLSQP specialist policy; auto obeys production qualification, "
            "shadow evaluates without changing planner output"
        ),
    )
    parser.add_argument("--sparse-min-slsqp-seconds", type=float, default=20.0)
    parser.add_argument("--sparse-max-seconds", type=float, default=120.0)
    parser.add_argument("--sparse-batches", type=int, default=5)
    parser.add_argument("--sparse-accepted-steps", type=int, default=5)
    parser.add_argument(
        "--sparse-telemetry", type=Path,
        help="append-only JSONL telemetry for the supervised sparse specialist",
    )
    parser.add_argument(
        "--curvature-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--length-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--curvature-iterations", type=int, default=100)
    parser.add_argument("--length-iterations", type=int, default=120)
    parser.add_argument("--time-iterations", type=int, default=60)
    parser.add_argument("--exchange-rounds", type=int, default=5)
    parser.add_argument("--curvature-regularization", type=float, default=0.0)
    parser.add_argument(
        "--length-curvature-regularization",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--body-length",
        type=float,
        default=DEFAULT_BODY_LENGTH,
        help="vehicle length in grid-cell units (default: 5/9)",
    )
    parser.add_argument(
        "--body-height",
        "--body-width",
        dest="body_height",
        type=float,
        default=DEFAULT_BODY_HEIGHT,
        help="vehicle lateral height in grid-cell units (default: 4/9)",
    )
    parser.add_argument(
        "--corridor-mode",
        choices=("overlapping_cover", "maximal_runs", "per_cell"),
        default="overlapping_cover",
    )
    parser.add_argument(
        "--geometry-refinement",
        type=int,
        default=1,
        help="exact subdivisions per analytic initializer segment",
    )
    parser.add_argument(
        "--curvature-slope-limit",
        type=float,
        default=50.0,
        help="maximum absolute dk/ds; pass 0 to disable",
    )
    parser.add_argument("--feasibility-tolerance", type=float, default=2.0e-7)
    parser.add_argument("--n-scan", type=int, default=48)
    parser.add_argument("--envelope-scan", type=int, default=24)
    parser.add_argument("--domain-scan", type=int, default=48)
    parser.add_argument("--bound-iterations", type=int, default=100)
    parser.add_argument(
        "--bound-disk-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="contract portals by the vehicle's centered inscribed disk",
    )
    parser.add_argument(
        "--bound-two-sided-speed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce the relaxed terminal speed as well as the initial speed",
    )
    parser.add_argument(
        "--bound-dual-certificate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the additional portal SOCP dual certificate",
    )
    parser.add_argument(
        "--bound-dual-gap-trigger",
        type=float,
        default=1.0e-6,
        help="relative affine-certificate gap that triggers the portal SOCP dual",
    )
    parser.add_argument(
        "--bound-projection-directions",
        type=int,
        choices=(0, 12),
        default=12,
        help="number of fixed reversal-projection axes; 0 disables, 12 is production",
    )
    parser.add_argument(
        "--bound-complete-cover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refine near-incumbent leaves with the disk-eroded corridor cover",
    )
    parser.add_argument(
        "--bound-complete-gap",
        type=float,
        default=2.0,
        help="maximum seconds below incumbent that triggers the complete-cover bound",
    )
    parser.add_argument(
        "--block-cut-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reachability-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--topology-quotient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge graph paths that differ only inside certified open 2x2 rooms",
    )
    parser.add_argument(
        "--topology-quotient-max-variants",
        type=int,
        default=16,
        help="maximum member initializers explored by one quotient class",
    )
    parser.add_argument(
        "--topology-quotient-pilot-iterations",
        type=int,
        default=8,
        help=(
            "short time-optimization pilot per quotient initializer before "
            "the single full class polish"
        ),
    )
    parser.add_argument("--maximum-node-visits", type=int, default=1)
    parser.add_argument("--maximum-expansions", type=int, default=20_000)
    parser.add_argument("--allow-partial-search", action="store_true")
    parser.add_argument("--samples-per-unit", type=float, default=100.0)
    parser.add_argument("--minimum-samples-per-segment", type=int, default=12)
    parser.add_argument(
        "--show-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, default=Path("planner_comparison.png"))
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--search-trace-output", type=Path,
        help="optional JSON trace of diagnostic B&B events for postprocessing/visualization",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.goal_max_speed is None:
        args.goal_max_speed = math.sqrt(args.init_w)
    if not math.isfinite(args.goal_max_speed) or not 0.0 < args.goal_max_speed <= 4.0:
        parser.error("goal-max-speed must lie in (0, 4]")
    terminal_w_max = args.goal_max_speed * args.goal_max_speed

    if args.maze_file is None:
        if args.width <= 0 or args.height <= 0:
            parser.error("maze dimensions must be positive")
        if not (0 <= args.start[0] < args.width and 0 <= args.start[1] < args.height):
            parser.error("start lies outside the maze")
        if args.extra_openings < 0:
            parser.error("extra-openings must be nonnegative")
    if args.geometry_refinement < 1:
        parser.error("geometry-refinement must be positive")
    if not math.isfinite(args.body_length) or args.body_length <= 0.0:
        parser.error("body-length must be finite and positive")
    if not math.isfinite(args.body_height) or args.body_height <= 0.0:
        parser.error("body-height must be finite and positive")
    if args.length_curvature_regularization < 0.0:
        parser.error("length-curvature-regularization must be nonnegative")
    if args.curvature_slope_limit < 0.0:
        parser.error("curvature-slope-limit must be nonnegative")
    if not math.isfinite(args.bound_complete_gap) or args.bound_complete_gap < 0.0:
        parser.error("bound-complete-gap must be finite and nonnegative")
    if not math.isfinite(args.bound_dual_gap_trigger) or args.bound_dual_gap_trigger < 0.0:
        parser.error("bound-dual-gap-trigger must be finite and nonnegative")
    if args.topology_quotient_max_variants <= 0:
        parser.error("topology-quotient-max-variants must be positive")
    if args.topology_quotient_pilot_iterations < 0:
        parser.error("topology-quotient-pilot-iterations must be nonnegative")
    if args.topology_quotient and args.maximum_node_visits != 1:
        parser.error("topology quotienting requires maximum-node-visits=1")
    if args.topology_quotient and args.corridor_mode != "overlapping_cover":
        parser.error("topology quotienting requires corridor-mode=overlapping_cover")

    if args.planner_architecture == PlannerArchitectureMode.LEGACY_V9.value:
        optimization_policy = PlannerOptimizationPolicy.legacy_v9()
    elif args.planner_architecture == PlannerArchitectureMode.ACTIVE_BASIS_V11.value:
        optimization_policy = PlannerOptimizationPolicy.active_basis_v11(
            work_root=(None if args.active_basis_work_dir is None else str(args.active_basis_work_dir)),
            require_convergence=True,
        )
    else:
        optimization_policy = PlannerOptimizationPolicy(
            architecture=PlannerArchitectureMode.INTEGRATED_V10,
            warm_start=WarmStartScheduleSettings(
                mode=WarmStartSchedulingMode(args.warm_start_mode),
                initializer_mode=WarmStartInitializerMode(args.warm_start_initializer),
                deadlines=WarmStartDeadlineSettings(),
                primary_filter_sqp=PrimaryFilterSQPPolicySettings(
                    mode=PrimaryTimeBackendMode(args.primary_time_backend)
                ),
                telemetry_jsonl=(
                    None if args.warm_start_telemetry is None
                    else str(args.warm_start_telemetry)
                ),
            ),
            sparse_specialist=SparseSpecialistPolicySettings(
                mode=SparseSpecialistMode(args.sparse_specialist),
                minimum_slsqp_seconds=args.sparse_min_slsqp_seconds,
                maximum_sparse_seconds=args.sparse_max_seconds,
                maximum_batches=args.sparse_batches,
                accepted_steps_per_batch=args.sparse_accepted_steps,
                telemetry_jsonl=(
                    None if args.sparse_telemetry is None else str(args.sparse_telemetry)
                ),
            ),
        )

    if (
        optimization_policy.architecture is PlannerArchitectureMode.ACTIVE_BASIS_V11
        and args.topology_quotient
    ):
        parser.error("active_basis_v11 currently requires --no-topology-quotient")

    scenario = None
    if args.maze_file is not None:
        scenario = load_maze_scenario(args.maze_file)
        maze = scenario.maze
        planning_start = scenario.start
        goal = scenario.canonical_goal
        maze_seed = None
        opening_seed = None
    else:
        maze_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
        opening_seed = (
            args.opening_seed
            if args.opening_seed is not None
            else maze_seed ^ 0x5EED5EED
        )
        maze = MazeGenerator(args.width, args.height, maze_seed).generate()
        maze = add_random_openings(maze, args.extra_openings, seed=opening_seed)
        planning_start = args.start
        goal = maze.goal
    graph = compress_to_graph(maze, planning_start, goal)
    seed_nodes = tuple(astar_junction_path(graph, planning_start, goal))
    seed_cells = tuple(expand_junction_path(maze, graph, seed_nodes))

    topology_quotient = None
    if args.topology_quotient:
        quotient_candidate = OpenRoomTopologyQuotient(
            maze,
            graph,
            body_length=args.body_length,
            body_height=args.body_height,
            maximum_class_variants=args.topology_quotient_max_variants,
        )
        if quotient_candidate.active:
            topology_quotient = quotient_candidate
    warm_start_stage_runner: SupervisedBatchRunner | None = None
    if (
        args.optimization_mode == "time"
        and optimization_policy.architecture is PlannerArchitectureMode.INTEGRATED_V10
    ):
        warm_start_stage_runner = create_warm_start_stage_runner(
            optimization_policy.warm_start
        )
        warm_start_stage_runner.start()

    class_cache: dict[object, tuple[OptimizedRoute, CompletePathEvaluation]] = {}
    selected_route_cache: dict[tuple[Cell, ...], OptimizedRoute] = {}
    evaluation_counter = 0

    def evaluate_complete_impl(
        nodes: tuple[int, ...],
        cells: tuple[Cell, ...],
        *,
        announce: bool,
    ) -> CompletePathEvaluation:
        nonlocal evaluation_counter
        if topology_quotient is None:
            class_key: object = nodes
            variants = ((nodes, cells, ()),)
            room_count = 0
        else:
            route_class = topology_quotient.route_class(nodes)
            class_key = route_class.key
            variants = tuple(
                (variant.junction_path, variant.cell_path, variant.open_room_spans)
                for variant in route_class.variants
            )
            room_count = len(route_class.quotient_room_indices)
        cached = class_cache.get(class_key)
        if cached is not None:
            return cached[1]

        if announce:
            evaluation_counter += 1
            if not args.quiet:
                if len(variants) == 1:
                    description = f"topology with {len(variants[0][1])} cells"
                else:
                    description = (
                        f"quotient class with {len(variants)} initializers, "
                        f"{room_count} open room(s)"
                    )
                print(
                    f"[{evaluation_counter}] optimizing complete {description}...",
                    flush=True,
                )
        started = time.perf_counter()
        solution = optimize_complete_route_class(
            tuple((variant_cells, spans) for _nodes, variant_cells, spans in variants),
            init_w=args.init_w,
            terminal_w_max=terminal_w_max,
            optimization_mode=args.optimization_mode,
            curvature_warm_start=args.curvature_warm_start,
            length_warm_start=args.length_warm_start,
            curvature_iterations=args.curvature_iterations,
            length_iterations=args.length_iterations,
            time_iterations=args.time_iterations,
            class_time_pilot_iterations=args.topology_quotient_pilot_iterations,
            maximum_exchange_rounds=args.exchange_rounds,
            feasibility_tolerance=args.feasibility_tolerance,
            curvature_regularization=args.curvature_regularization,
            length_curvature_regularization=args.length_curvature_regularization,
            corridor_mode=args.corridor_mode,
            body_length=args.body_length,
            body_height=args.body_height,
            geometry_refinement=args.geometry_refinement,
            curvature_slope_limit=(
                None if args.curvature_slope_limit == 0.0
                else args.curvature_slope_limit
            ),
            n_scan=args.n_scan,
            envelope_scan=args.envelope_scan,
            domain_scan=args.domain_scan,
            optimization_policy=optimization_policy,
            warm_start_stage_runner=warm_start_stage_runner,
        )
        selected_nodes = nodes
        for variant_nodes, variant_cells, _spans in variants:
            if tuple(solution.cells) == tuple(variant_cells):
                selected_nodes = variant_nodes
                break
        evaluation = CompletePathEvaluation(
            solution.time,
            tuple(selected_nodes),
            tuple(solution.cells),
        )
        class_cache[class_key] = (solution, evaluation)
        selected_route_cache[evaluation.cell_path] = solution
        if announce and not args.quiet:
            print(
                f"    T={solution.time:.6f}s via {solution.selected_stage} "
                f"({time.perf_counter() - started:.2f}s)",
                flush=True,
            )
        return evaluation

    def evaluate_complete(
        nodes: tuple[int, ...], cells: tuple[Cell, ...]
    ) -> CompletePathEvaluation:
        return evaluate_complete_impl(nodes, cells, announce=True)

    search_trace_events = []
    try:
        seed_evaluation = evaluate_complete_impl(seed_nodes, seed_cells, announce=False)
        seed_route = selected_route_cache[seed_evaluation.cell_path]

        def report_complete_candidate(_nodes, cells, bound, incumbent):
            if args.quiet:
                return
            print(
                f"    leaf LB={bound.time_lower_bound:.6f}s "
                f"({bound.name}), incumbent={incumbent:.6f}s, "
                f"gap={incumbent - bound.time_lower_bound:.6f}s, "
                f"cells={len(cells)}",
                flush=True,
            )

        search_started = time.perf_counter()
        search = branch_and_bound_junction_paths(
            maze,
            graph,
            planning_start,
            goal,
            lower_bound=PortalMotorTimeLowerBound(
                maximum_iterations=args.bound_iterations,
                cache_warm_starts=True,
                maze=maze,
                body_length=args.body_length,
                body_height=args.body_height,
                use_inscribed_disk_gates=args.bound_disk_gates,
                use_two_sided_time=args.bound_two_sided_speed,
                use_dual_certificate=args.bound_dual_certificate,
                dual_gap_trigger=args.bound_dual_gap_trigger,
                use_complete_cover_bound=args.bound_complete_cover,
                complete_refinement_gap=args.bound_complete_gap,
                projection_directions=args.bound_projection_directions,
            ),
            complete_path_time=evaluate_complete,
            init_w=args.init_w,
            terminal_w_max=terminal_w_max,
            settings=SearchSettings(
                maximum_node_visits=args.maximum_node_visits,
                maximum_expansions=args.maximum_expansions,
                use_block_cut_pruning=args.block_cut_pruning,
                use_no_revisit_reachability=args.reachability_pruning,
                use_complete_bound_refinement=args.bound_complete_cover,
            ),
            seed_junction_path=seed_nodes,
            complete_candidate_observer=report_complete_candidate,
            topology_quotient=topology_quotient,
            search_trace_observer=(
                search_trace_events.append if args.search_trace_output is not None else None
            ),
        )
        search_seconds = time.perf_counter() - search_started
    finally:
        if warm_start_stage_runner is not None:
            warm_start_stage_runner.close()
    if not search.exhausted and not args.allow_partial_search:
        raise RuntimeError(
            "branch-and-bound hit maximum-expansions before exhaustion; "
            "raise the cap or pass --allow-partial-search"
        )

    best_key = tuple(search.best_cell_path)
    best_route = selected_route_cache.get(best_key)
    if best_route is None:
        evaluation = evaluate_complete_impl(
            tuple(search.best_junction_path), best_key, announce=False
        )
        best_route = selected_route_cache[evaluation.cell_path]

    render_comparison(
        maze,
        seed_route,
        best_route,
        search_result=search,
        maze_seed=maze_seed,
        extra_openings=args.extra_openings,
        output=args.output,
        dpi=args.dpi,
        show=args.show,
        show_topology=args.show_topology,
        samples_per_unit=args.samples_per_unit,
        minimum_samples_per_segment=args.minimum_samples_per_segment,
        seed_is_quotient_class=(
            topology_quotient is not None
            and len(topology_quotient.route_class(seed_nodes).variants) > 1
        ),
        scenario=scenario,
    )

    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    configuration = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    from segment.constants import PHYSICS_PROFILE, PHYSICS_PROFILE_NAME
    from segment.physics_identity import physics_model_identity, physics_model_signature
    metadata = {
        "configuration": configuration,
        "physics_model_signature": physics_model_signature(PHYSICS_PROFILE),
        "physics_model_identity": physics_model_identity(PHYSICS_PROFILE),
        "physics_profile": {
            "name": PHYSICS_PROFILE_NAME,
            "description": PHYSICS_PROFILE.description,
            "cell_pitch_m": PHYSICS_PROFILE.cell_pitch_m,
            "mu_g": PHYSICS_PROFILE.mu_g,
            "a_brake": PHYSICS_PROFILE.a_brake,
            "a_max": PHYSICS_PROFILE.a_max,
            "v_max": PHYSICS_PROFILE.v_max,
            "b_emf": PHYSICS_PROFILE.b_emf,
            "time_model": PHYSICS_PROFILE.time_model,
            "dd_effective_track_m": PHYSICS_PROFILE.dd_effective_track_m,
            "dd_yaw_inertia_scale": PHYSICS_PROFILE.dd_yaw_inertia_scale,
        },
        "maze_seed": maze_seed,
        "opening_seed": opening_seed,
        "goal": goal,
        "maze_scenario": (
            None if scenario is None else {
                "name": scenario.name,
                "source_path": scenario.source_path,
                "source_sha256": scenario.source_sha256,
                "coordinate_origin": scenario.coordinate_origin,
                "source_coordinates": {
                    "start": scenario.to_source_cell(scenario.start),
                    "goal_cells": [
                        scenario.to_source_cell(cell)
                        for cell in scenario.goal_region.cells
                    ],
                    "goal_entrances": [
                        {
                            "outside": scenario.to_source_cell(entry.outside_cell),
                            "inside": scenario.to_source_cell(entry.inside_cell),
                        }
                        for entry in scenario.goal_region.entrances
                    ],
                    "canonical_goal": scenario.to_source_cell(scenario.canonical_goal),
                },
                "normalized_internal_coordinates": {
                    "start": scenario.start,
                    "goal_cells": scenario.goal_region.cells,
                    "goal_entrances": [
                        {"outside": entry.outside_cell, "inside": entry.inside_cell}
                        for entry in scenario.goal_region.entrances
                    ],
                    "canonical_goal": scenario.canonical_goal,
                },
                "start_heading": scenario.start_heading,
                "scale": {
                    "unit_mm": scenario.scale.unit_mm,
                    "wall_thickness_units": scenario.scale.wall_thickness_units,
                    "cell_pitch_units": scenario.scale.cell_pitch_units,
                    "cell_pitch_m": scenario.scale.cell_pitch_m,
                },
                "metadata": dict(scenario.metadata),
            }
        ),
        "junctions": len(graph.nodes),
        "search_seconds": search_seconds,
        "search": asdict(search),
        "seed_route": route_snapshot(seed_route, config=configuration),
        "best_route": route_snapshot(best_route, config=configuration),
        "relative_improvement": (
            seed_route.time - best_route.time
        ) / seed_route.time,
        "complete_route_cache_size": len(class_cache),
        "topology_quotient": (
            None
            if topology_quotient is None
            else {
                "rooms": len(topology_quotient.rooms),
                "statistics": asdict(topology_quotient.statistics),
            }
        ),
        "figure": str(args.output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=list) + "\n")
    if args.search_trace_output is not None:
        args.search_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.search_trace_output.write_text(
            json.dumps([asdict(event) for event in search_trace_events], indent=2, default=list) + "\n"
        )

    print(f"maze seed: {maze_seed}")
    print(f"seed time: {seed_route.time:.9f} s")
    print(f"best time: {best_route.time:.9f} s")
    print(f"search exhausted: {search.exhausted}")
    print(f"figure: {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
