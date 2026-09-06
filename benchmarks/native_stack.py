from __future__ import annotations

from . import SCHEMA_VERSION

import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from optimization import reverse_solver, scalar_reverse_solver
from planning import PlannerOptimizationPolicy

from .common.certification import certify_optimized_route as independent_certify
from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.routes import RouteEvaluator, route_to_record
from .common.runtime import reverse_backend
from .common.timing import median_timed
from .config import CORE_OPTIMIZATION, SMOKE_OPTIMIZATION, OptimizationConfig


def _median_timed(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    return median_timed(fn, repeats=repeats, warmup=1)


def _errors(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
    absolute = np.abs(aa - bb)
    relative = absolute / np.maximum(1.0e-10, np.maximum(np.abs(aa), np.abs(bb)))
    return float(np.max(absolute, initial=0.0)), float(np.max(relative, initial=0.0))


def _trajectories(topology: dict[str, Any], count: int) -> list[dict[str, Any]]:
    out = []
    for case in topology["cases"]:
        if not case["branch_and_bound"]["certification"]["certified"]:
            continue
        route = case["branch_and_bound"]["route"]
        out.append({"name": case["case"]["name"], **route})
        if len(out) >= count:
            break
    return out


def _route_optimizer_stats(route) -> dict[str, Any]:
    stages = list(route.stages)
    selected = next((stage for stage in stages if stage.name == route.selected_stage), None)
    return {
        "iterations": sum(stage.optimizer_iterations or 0 for stage in stages),
        "objective_calls": sum(stage.objective_calls or 0 for stage in stages),
        "gradient_calls": sum(stage.gradient_calls or 0 for stage in stages),
        "constraint_calls": sum(stage.constraint_calls or 0 for stage in stages),
        "exchange_rounds": sum(stage.exchange_rounds or 0 for stage in stages),
        "selected_stage_kkt_stationarity_inf": (
            None if selected is None else selected.kkt_stationarity_inf
        ),
        "optimizer_backends": sorted({
            stage.optimizer_backend for stage in stages if stage.optimizer_backend
        }),
    }


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    topology_path = output_dir / "topology_search.json"
    if not topology_path.exists():
        raise FileNotFoundError("native benchmark requires topology_search.json")
    topology = read_json(topology_path)
    config: OptimizationConfig = CORE_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    trajectories = _trajectories(topology, 4 if profile == "core" else 1)
    repeats = 7 if profile == "core" else 2

    micro_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        raw = [float(v) for v in trajectory["raw_parameters"]]
        initial_k = float(trajectory.get("initial_k", 0.0))
        kwargs = dict(init_w=config.init_w, initial_k=initial_k, n_scan=config.n_scan,
                      envelope_scan=config.envelope_scan, domain_scan=config.domain_scan)
        backend_data: dict[str, Any] = {}
        for backend in ("python", "native"):
            with reverse_backend(backend):
                scalar_seconds, scalar_value = _median_timed(
                    lambda: scalar_reverse_solver.evaluate_time_scalar(raw, **kwargs), repeats
                )
                grad_seconds, (grad_value, gradient) = _median_timed(
                    lambda: reverse_solver.time_value_and_gradient(raw, **kwargs), repeats
                )
                profiler = reverse_solver.PhaseProfiler()
                diagnostics = scalar_reverse_solver.evaluate_time_scalar_result(raw, profiler=profiler, **kwargs)
                profiler_grad = reverse_solver.PhaseProfiler()
                reverse_solver.time_value_and_gradient(raw, profiler=profiler_grad, **kwargs)
                backend_data[backend] = {
                    "scalar_value": float(scalar_value),
                    "gradient_time_value": float(grad_value),
                    "gradient": [float(v) for v in gradient],
                    "scalar_median_seconds": scalar_seconds,
                    "time_gradient_median_seconds": grad_seconds,
                    "scalar_diagnostics": {
                        "topology_seconds": diagnostics.topology_seconds,
                        "integration_seconds": diagnostics.integration_seconds,
                        "envelope_intervals": diagnostics.envelope_intervals,
                        "integrand_evaluations": diagnostics.integrand_evaluations,
                        "scalar_passes": diagnostics.scalar_passes,
                        "scalar_segments": diagnostics.scalar_segments,
                    },
                    "scalar_phase_profiler": profiler.snapshot(),
                    "gradient_phase_profiler": profiler_grad.snapshot(),
                }
        max_abs, max_rel = _errors(backend_data["native"]["gradient"], backend_data["python"]["gradient"])
        micro_rows.append({
            "name": trajectory["name"],
            "dimension": len(raw),
            "raw_parameters": raw,
            "python": backend_data["python"],
            "native": backend_data["native"],
            "time_value_absolute_error": abs(backend_data["native"]["gradient_time_value"] - backend_data["python"]["gradient_time_value"]),
            "maximum_absolute_gradient_error": max_abs,
            "maximum_relative_gradient_error": max_rel,
            "scalar_speedup": backend_data["python"]["scalar_median_seconds"] / backend_data["native"]["scalar_median_seconds"],
            "time_gradient_speedup": backend_data["python"]["time_gradient_median_seconds"] / backend_data["native"]["time_gradient_median_seconds"],
        })

    # Complete optimizer A/B. Use the repository's legacy-v9 production route
    # orchestration here because it runs the reverse backend in-process; the
    # integrated-v10 warm-start supervisor uses worker processes whose backend
    # selection is intentionally isolated, which would make a Python/native A/B
    # ambiguous. All geometry optimization and independent certification settings
    # are otherwise identical between the two backends.
    end_to_end_rows: list[dict[str, Any]] = []
    policy = PlannerOptimizationPolicy.legacy_v9()
    for trajectory in trajectories[: (2 if profile == "core" else 1)]:
        cells = tuple(tuple(cell) for cell in trajectory["cells"])
        results: dict[str, Any] = {}
        for backend in ("python", "native"):
            with reverse_backend(backend):
                started = time.perf_counter()
                with RouteEvaluator(config, policy=policy) as evaluator:
                    optimized = evaluator.optimize(cells)
                wall = time.perf_counter() - started
            certification = independent_certify(optimized, config)
            results[backend] = {
                "final_time": float(optimized.time),
                "parameters": optimized.parameters.tolist(),
                "success": bool(certification["certified"]),
                "message": next(
                    (stage.message for stage in optimized.stages if stage.name == optimized.selected_stage),
                    optimized.selected_stage,
                ),
                "selected_stage": optimized.selected_stage,
                "architecture": optimized.architecture,
                "wall_seconds": wall,
                "certified": bool(certification["certified"]),
                "endpoint_error": certification["endpoint_error"],
                "corridor_upper_bound": certification["corridor_upper_bound"],
                **_route_optimizer_stats(optimized),
            }
        p = np.asarray(results["python"]["parameters"], dtype=float)
        n = np.asarray(results["native"]["parameters"], dtype=float)
        end_to_end_rows.append({
            "name": trajectory["name"],
            "policy": "legacy_v9",
            "python": results["python"],
            "native": results["native"],
            "final_time_absolute_difference": abs(results["python"]["final_time"] - results["native"]["final_time"]),
            "maximum_parameter_absolute_difference": float(np.max(np.abs(p - n), initial=0.0)),
            "wall_speedup_python_over_native": results["python"]["wall_seconds"] / results["native"]["wall_seconds"],
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "native_stack",
        "profile": profile,
        "environment": environment_metadata(config),
        "microbenchmarks": micro_rows,
        "end_to_end_optimizations": end_to_end_rows,
        "aggregate": {
            "micro_cases": len(micro_rows),
            "median_scalar_speedup": statistics.median([r["scalar_speedup"] for r in micro_rows]),
            "median_time_gradient_speedup": statistics.median([r["time_gradient_speedup"] for r in micro_rows]),
            "maximum_time_value_absolute_error": max(r["time_value_absolute_error"] for r in micro_rows),
            "maximum_absolute_gradient_error": max(r["maximum_absolute_gradient_error"] for r in micro_rows),
            "end_to_end_cases": len(end_to_end_rows),
            "median_end_to_end_speedup": statistics.median([r["wall_speedup_python_over_native"] for r in end_to_end_rows]),
            "maximum_end_to_end_time_difference": max(r["final_time_absolute_difference"] for r in end_to_end_rows),
            "all_end_to_end_certified": all(r["python"]["certified"] and r["native"]["certified"] for r in end_to_end_rows),
        },
    }
    write_json(output_dir / "native_stack.json", result)
    return result
