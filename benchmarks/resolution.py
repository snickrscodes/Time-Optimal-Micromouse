from __future__ import annotations

from . import SCHEMA_VERSION

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optimization import (
    CorridorModel,
    ExchangeSettings,
    PathOptimizerSettings,
    SLSQPSettings,
    compile_geometry_path,
    curvature_energy_value_and_gradient,
    knot_parameters_to_raw,
    optimize_path,
    separate_rectangle_path,
)
from planning import build_route_optimization_problem

from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.status import CERTIFICATION_FAILURE, SUCCESS, classify_exception
from .config import CORE_OPTIMIZATION, SMOKE_OPTIMIZATION, OptimizationConfig


@dataclass(frozen=True, slots=True)
class RefinementIdentity:
    maximum_position_error: float
    maximum_heading_error: float
    maximum_curvature_error: float
    curvature_energy_error: float


def _state_at_station(path: Any, station: float) -> Any:
    station = min(max(float(station), 0.0), float(sum(path.lengths)))
    offset = 0.0
    for index, length in enumerate(path.lengths):
        next_offset = offset + length
        if station <= next_offset or index + 1 == path.n_segments:
            fraction = 0.0 if length == 0.0 else (station - offset) / length
            return path.state_at_fraction(index, min(1.0, max(0.0, fraction)))
        offset = next_offset
    raise AssertionError("station lookup fell through")


def _wrapped_angle_error(value: float, target: float) -> float:
    return abs(math.atan2(math.sin(value - target), math.cos(value - target)))


def exact_refine_parameters(
    parameters, *, initial_s: float, initial_k: float, factor: int = 2
) -> np.ndarray:
    """Split every linear-curvature segment into exact child segments."""
    if factor < 1:
        raise ValueError("factor must be positive")
    source = np.asarray(parameters, dtype=float)
    if source.ndim != 1 or source.size == 0 or source.size % 2:
        raise ValueError("parameters must be flat [s1,k1,...]")

    refined: list[float] = []
    s0 = float(initial_s)
    k0 = float(initial_k)
    for s1, k1 in source.reshape(-1, 2):
        s1 = float(s1)
        k1 = float(k1)
        for child in range(1, factor + 1):
            fraction = child / factor
            refined.extend((
                math.fma(fraction, s1 - s0, s0),
                math.fma(fraction, k1 - k0, k0),
            ))
        s0 = s1
        k0 = k1
    return np.asarray(refined, dtype=float)


def refine_corridor(corridor: Any, *, factor: int, CorridorModel: Any) -> Any:
    assignments = tuple(
        cell_index
        for cell_index in corridor.segment_cells
        for _ in range(factor)
    )
    return CorridorModel(corridor.cells, assignments, corridor.body, corridor.clearance)


def compare_exact_paths(
    old_parameters,
    new_parameters,
    *,
    initial_state: Any,
    compile_geometry_path: Any,
    knot_parameters_to_raw: Any,
    curvature_energy_value_and_gradient: Any,
    samples: int = 2001,
) -> RefinementIdentity:
    old_raw = knot_parameters_to_raw(old_parameters, initial_k=initial_state.k)
    new_raw = knot_parameters_to_raw(new_parameters, initial_k=initial_state.k)
    old_path = compile_geometry_path(old_raw, initial_state)
    new_path = compile_geometry_path(new_raw, initial_state)
    total = float(sum(old_path.lengths))
    if not math.isclose(total, float(sum(new_path.lengths)), rel_tol=0.0, abs_tol=2.0e-13):
        raise AssertionError("exact refinement changed total station length")

    maximum_position = 0.0
    maximum_heading = 0.0
    maximum_curvature = 0.0
    for station in np.linspace(0.0, total, samples):
        first = _state_at_station(old_path, float(station))
        second = _state_at_station(new_path, float(station))
        maximum_position = max(
            maximum_position, math.hypot(first.x - second.x, first.y - second.y)
        )
        maximum_heading = max(
            maximum_heading, _wrapped_angle_error(first.theta, second.theta)
        )
        maximum_curvature = max(maximum_curvature, abs(first.k - second.k))

    old_energy, _ = curvature_energy_value_and_gradient(
        old_parameters, initial_k=initial_state.k
    )
    new_energy, _ = curvature_energy_value_and_gradient(
        new_parameters, initial_k=initial_state.k
    )
    return RefinementIdentity(
        maximum_position,
        maximum_heading,
        maximum_curvature,
        abs(float(old_energy) - float(new_energy)),
    )


def _certify(parameters, problem, corridor, tolerance: float) -> tuple[bool, float, float]:
    raw = knot_parameters_to_raw(parameters, initial_k=problem.initial_state.k)
    path = compile_geometry_path(raw, problem.initial_state)
    final = path.final_state
    endpoint_error = problem.terminal_violation(final)
    separation = separate_rectangle_path(parameters, problem.initial_state, corridor)
    feasible = endpoint_error <= tolerance and separation.certified(tolerance) and np.all(np.isfinite(parameters))
    return bool(feasible), float(endpoint_error), float(separation.worst_upper_bound)


def _optimize(parameters, problem, corridor, config: OptimizationConfig, *, iterations: int):
    settings = PathOptimizerSettings(
        exchange=ExchangeSettings(
            maximum_rounds=config.maximum_exchange_rounds,
            require_solver_success=False,
            slsqp=SLSQPSettings(max_iterations=iterations, ftol=1.0e-10, display=False),
        ),
        n_scan=config.n_scan,
        envelope_scan=config.envelope_scan,
        domain_scan=config.domain_scan,
    )
    array = np.asarray(parameters, dtype=float)
    bounds = [
        (float(value), float(value)) if index % 2 == 0 else (-10.0, 10.0)
        for index, value in enumerate(array)
    ]
    started = time.perf_counter()
    result = optimize_path(
        array,
        corridor,
        problem.initial_state,
        init_w=config.init_w,
        time_weight=1.0,
        curvature_weight=0.0,
        endpoint_target=problem.endpoint_target,
        additional_inequalities=problem.inequalities(),
        bounds=bounds,
        settings=settings,
    )
    wall = time.perf_counter() - started
    cert, endpoint_error, corridor_upper = _certify(result.parameters, problem, corridor, config.feasibility_tolerance)
    energy, _ = curvature_energy_value_and_gradient(result.parameters, initial_k=problem.initial_state.k)
    exchange = result.exchange
    return result, {
        "segments": len(result.parameters)//2,
        "time": float(result.time),
        "curvature_energy": float(energy),
        "wall_seconds": wall,
        "certified": cert,
        "endpoint_error": endpoint_error,
        "corridor_upper_bound": corridor_upper,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "execution_status": SUCCESS if cert else CERTIFICATION_FAILURE,
        "error": None,
        "exchange_rounds": 0 if exchange is None else len(exchange.rounds),
        "iterations": 0 if exchange is None else sum(r.finite.iterations for r in exchange.rounds),
    }


def _attempt_optimize(parameters, problem, corridor, config: OptimizationConfig, *, iterations: int):
    array = np.asarray(parameters, dtype=float)
    started = time.perf_counter()
    try:
        return _optimize(array, problem, corridor, config, iterations=iterations)
    except Exception as exc:
        return None, {
            "segments": int(len(array) // 2),
            "time": None,
            "curvature_energy": None,
            "wall_seconds": float(time.perf_counter() - started),
            "certified": False,
            "endpoint_error": None,
            "corridor_upper_bound": None,
            "solver_success": False,
            "solver_message": None,
            "execution_status": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "exchange_rounds": 0,
            "iterations": 0,
        }


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    topology_path = output_dir / "topology_search.json"
    if not topology_path.exists():
        raise FileNotFoundError("resolution benchmark requires topology_search.json")
    topology = read_json(topology_path)
    config = CORE_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    source_cases = [c for c in topology["cases"] if c["branch_and_bound"]["certification"]["certified"]]
    source_cases = source_cases[: (2 if profile == "core" else 1)]
    iterations = 16 if profile == "core" else 4
    rows = []
    for case in source_cases:
        route = case["branch_and_bound"]["route"]
        cells = tuple(tuple(c) for c in route["cells"])
        problem = build_route_optimization_problem(
            cells,
            body_length=config.body_length,
            body_height=config.body_height,
            corridor_mode=config.corridor_mode,
            refinement_factor=config.geometry_refinement,
        )
        base = np.asarray(route["parameters"], dtype=float)
        refined = exact_refine_parameters(base, initial_s=0.0, initial_k=problem.initial_state.k, factor=2)
        refined_corridor = refine_corridor(problem.corridor, factor=2, CorridorModel=CorridorModel)
        identity = compare_exact_paths(
            base,
            refined,
            initial_state=problem.initial_state,
            compile_geometry_path=compile_geometry_path,
            knot_parameters_to_raw=knot_parameters_to_raw,
            curvature_energy_value_and_gradient=curvature_energy_value_and_gradient,
            samples=1001 if profile == "core" else 101,
        )
        n_result, n_record = _attempt_optimize(base, problem, problem.corridor, config, iterations=iterations)
        two_result, two_record = _attempt_optimize(refined, problem, refined_corridor, config, iterations=iterations)
        comparable = n_record["time"] is not None and two_record["time"] is not None
        rows.append({
            "name": case["case"]["name"],
            "initial_prolongation_identity": {
                "maximum_position_error": identity.maximum_position_error,
                "maximum_heading_error": identity.maximum_heading_error,
                "maximum_curvature_error": identity.maximum_curvature_error,
                "curvature_energy_error": identity.curvature_energy_error,
            },
            "N": n_record,
            "2N": two_record,
            "relative_time_difference_2N_minus_N": ((two_record["time"] - n_record["time"]) / n_record["time"]) if comparable and n_record["time"] else None,
            "absolute_time_difference_2N_minus_N": (two_record["time"] - n_record["time"]) if comparable else None,
            "curvature_energy_difference_2N_minus_N": (two_record["curvature_energy"] - n_record["curvature_energy"]) if comparable else None,
            "runtime_ratio_2N_over_N": two_record["wall_seconds"] / n_record["wall_seconds"] if comparable and n_record["wall_seconds"] else None,
            "N_parameters": None if n_result is None else n_result.parameters.tolist(),
            "2N_parameters": None if two_result is None else two_result.parameters.tolist(),
        })

    certified = [r for r in rows if r["N"]["certified"] and r["2N"]["certified"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "resolution",
        "profile": profile,
        "environment": environment_metadata(config),
        "settings": {
            "fixed_stations": True,
            "optimization_iterations_per_level": iterations,
            "refinement_factor": 2,
            "note": "2N is an exact geometric prolongation of the same starting curve; station coordinates are fixed so the benchmark isolates curvature-resolution sensitivity.",
        },
        "rows": rows,
        "aggregate": {
            "cases": len(rows),
            "certified_pairs": len(certified),
            "median_absolute_relative_time_difference": statistics.median([abs(r["relative_time_difference_2N_minus_N"]) for r in certified]) if certified else None,
            "maximum_absolute_relative_time_difference": max([abs(r["relative_time_difference_2N_minus_N"]) for r in certified], default=None),
            "median_runtime_ratio_2N_over_N": statistics.median([r["runtime_ratio_2N_over_N"] for r in certified]) if certified else None,
            "maximum_initial_position_prolongation_error": max([r["initial_prolongation_identity"]["maximum_position_error"] for r in rows], default=None),
        },
    }
    write_json(output_dir / "resolution.json", result)
    return result
