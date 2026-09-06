from __future__ import annotations

from . import SCHEMA_VERSION

import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from main import scalar_time
from planning import build_route_optimization_problem

from .common.certification import certify_optimized_route as independent_certify
from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.routes import RouteEvaluator, route_to_record
from .common.status import OPTIMIZER_FAILURE, SUCCESS
from .config import CORE_OPTIMIZATION, SMOKE_OPTIMIZATION, OptimizationConfig

VARIANTS = (
    ("direct_time", False, False),
    ("length_then_time", False, True),
    ("curvature_then_time", True, False),
    ("production_warm_start", True, True),
)


def _representatives(topology: dict[str, Any], count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in topology["cases"]:
        route = case["branch_and_bound"]["route"]
        cert = case["branch_and_bound"]["certification"]
        if cert["certified"]:
            rows.append({"name": case["case"]["name"], "cells": tuple(tuple(c) for c in route["cells"])})
        if len(rows) >= count:
            break
    return rows


def _time_backend_metrics(record) -> dict[str, Any]:
    backend = {} if record is None or record.time_backend is None else dict(record.time_backend)
    checkpoint = {} if record is None or record.time_checkpoint is None else dict(record.time_checkpoint)
    return {
        "slsqp_major_iterations": backend.get("optimizer_iterations"),
        "objective_calls": backend.get("objective_calls", checkpoint.get("time_checkpoint_objective_calls")),
        "gradient_calls": backend.get("gradient_evaluations", checkpoint.get("time_checkpoint_gradient_calls")),
        "constraint_calls": backend.get("constraint_calls"),
        "exchange_rounds": backend.get("exchange_rounds"),
        "optimizer_backend": backend.get("worker_backend") or backend.get("selected_backend"),
        "kkt_stationarity_residual": backend.get("kkt_stationarity_inf"),
    }


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    topology_path = output_dir / "topology_search.json"
    if not topology_path.exists():
        raise FileNotFoundError("warm-start benchmark requires topology_search.json")
    topology = read_json(topology_path)
    config: OptimizationConfig = CORE_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    representatives = _representatives(topology, 2 if profile == "core" else 1)
    rows: list[dict[str, Any]] = []

    with RouteEvaluator(config) as evaluator:
        for representative in representatives:
            cells = representative["cells"]
            problem = build_route_optimization_problem(
                cells,
                body_length=config.body_length,
                body_height=config.body_height,
                corridor_mode=config.corridor_mode,
                refinement_factor=config.geometry_refinement,
            )
            try:
                initializer_time = scalar_time(
                    problem,
                    problem.initial_parameters,
                    init_w=config.init_w,
                    n_scan=config.n_scan,
                    envelope_scan=config.envelope_scan,
                    domain_scan=config.domain_scan,
                )
            except Exception:
                initializer_time = math.inf

            for name, curvature, length in VARIANTS:
                started = time.perf_counter()
                try:
                    route = evaluator.optimize(
                        cells,
                        curvature_warm_start=curvature,
                        length_warm_start=length,
                    )
                except Exception as exc:
                    wall = time.perf_counter() - started
                    rows.append({
                        "route_name": representative["name"],
                        "variant": name,
                        "curvature_warm_start": curvature,
                        "length_warm_start": length,
                        "execution_status": OPTIMIZER_FAILURE,
                        "error": f"{type(exc).__name__}: {exc}",
                        "certification": {
                            "certified": False,
                            "endpoint_error": None,
                            "corridor_upper_bound": None,
                            "final_state": None,
                        },
                        "warm_start_generation_seconds": None,
                        "time_optimization_seconds": None,
                        "total_wall_seconds": wall,
                        "initializer_travel_time": initializer_time,
                        "stage_records": [],
                        "final_travel_time": None,
                        "final_improvement_from_initializer": None,
                        "endpoint_error": None,
                        "worst_continuous_corridor_upper_bound": None,
                        "selected_stage": None,
                        "slsqp_major_iterations": None,
                        "objective_calls": None,
                        "gradient_calls": None,
                        "constraint_calls": None,
                        "exchange_rounds": None,
                        "optimizer_backend": None,
                        "kkt_stationarity_residual": None,
                        "route": None,
                    })
                    continue

                wall = time.perf_counter() - started
                cert = independent_certify(route, config)
                run_record = route.warm_start_record
                warm_seconds = None
                time_seconds = None
                if run_record is not None:
                    time_seconds = float(run_record.full_time_seconds)
                    warm_seconds = max(0.0, float(run_record.total_seconds) - time_seconds)
                metrics = _time_backend_metrics(run_record)
                final_improvement = (
                    initializer_time - route.time if math.isfinite(initializer_time) else None
                )
                rows.append({
                    "route_name": representative["name"],
                    "variant": name,
                    "curvature_warm_start": curvature,
                    "length_warm_start": length,
                    "execution_status": SUCCESS if cert["certified"] else "certification_failure",
                    "error": None,
                    "certification": cert,
                    "warm_start_generation_seconds": warm_seconds,
                    "time_optimization_seconds": time_seconds,
                    "total_wall_seconds": wall,
                    "initializer_travel_time": initializer_time,
                    "stage_records": [] if run_record is None else [stage.__dict__ if hasattr(stage, "__dict__") else {
                        field: getattr(stage, field) for field in stage.__dataclass_fields__
                    } for stage in run_record.stage_records],
                    "final_travel_time": float(route.time),
                    "final_improvement_from_initializer": final_improvement,
                    "endpoint_error": cert["endpoint_error"],
                    "worst_continuous_corridor_upper_bound": cert["corridor_upper_bound"],
                    "selected_stage": route.selected_stage,
                    **metrics,
                    "route": route_to_record(route, config),
                })

    aggregate: dict[str, Any] = {"routes": len(representatives), "variants": {}}
    for name, _curv, _length in VARIANTS:
        subset = [row for row in rows if row["variant"] == name]
        certified = [row for row in subset if row["certification"]["certified"]]
        aggregate["variants"][name] = {
            "certified": len(certified),
            "attempted": len(subset),
            "certification_rate": len(certified) / len(subset) if subset else None,
            "median_final_time": statistics.median([r["final_travel_time"] for r in certified]) if certified else None,
            "median_total_wall_seconds": statistics.median([r["total_wall_seconds"] for r in subset]) if subset else None,
            "median_final_improvement_from_initializer": statistics.median([
                r["final_improvement_from_initializer"] for r in certified if r["final_improvement_from_initializer"] is not None
            ]) if certified else None,
        }

    paired = []
    for representative in representatives:
        route_rows = [row for row in rows if row["route_name"] == representative["name"]]
        direct = next((row for row in route_rows if row["variant"] == "direct_time"), None)
        production = next((row for row in route_rows if row["variant"] == "production_warm_start"), None)
        if (direct is None or production is None or
                not direct["certification"]["certified"] or
                not production["certification"]["certified"]):
            continue
        improvement_pct = 100.0 * (direct["final_travel_time"] - production["final_travel_time"]) / direct["final_travel_time"]
        paired.append({
            "route_name": representative["name"],
            "final_time_improvement_percent": improvement_pct,
            "wall_speedup_direct_over_production": direct["total_wall_seconds"] / production["total_wall_seconds"],
            "production_lower_final_time": production["final_travel_time"] < direct["final_travel_time"],
            "production_lower_wall_time": production["total_wall_seconds"] < direct["total_wall_seconds"],
        })
    aggregate["production_vs_direct"] = {
        "certified_pairs": len(paired),
        "production_lower_final_time_cases": sum(r["production_lower_final_time"] for r in paired),
        "production_lower_wall_time_cases": sum(r["production_lower_wall_time"] for r in paired),
        "median_final_time_improvement_percent": statistics.median([r["final_time_improvement_percent"] for r in paired]) if paired else None,
        "min_final_time_improvement_percent": min([r["final_time_improvement_percent"] for r in paired], default=None),
        "max_final_time_improvement_percent": max([r["final_time_improvement_percent"] for r in paired], default=None),
        "median_wall_speedup_direct_over_production": statistics.median([r["wall_speedup_direct_over_production"] for r in paired]) if paired else None,
        "pairs": paired,
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "warm_start",
        "profile": profile,
        "environment": environment_metadata(config),
        "settings": {
            "same_final_time_iteration_budget": config.time_iterations,
            "same_exchange_round_budget": config.maximum_exchange_rounds,
            "planner_architecture": "integrated_v10",
            "note": "Unavailable low-level counters are emitted as null rather than reconstructed from benchmark code.",
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    write_json(output_dir / "warm_start.json", result)
    return result
