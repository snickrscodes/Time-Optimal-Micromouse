from __future__ import annotations

from . import SCHEMA_VERSION

import math
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from optimization import reverse_solver, scalar_reverse_solver

from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.runtime import reverse_backend
from .common.timing import median_timed
from .config import CORE_OPTIMIZATION, SMOKE_OPTIMIZATION, OptimizationConfig


def _relative_error(actual: float, reference: float, floor: float = 1.0e-10) -> float:
    return abs(actual - reference) / max(floor, abs(reference), abs(actual))


def _gradient_errors(actual: Sequence[float], reference: Sequence[float]) -> tuple[float, float]:
    a = np.asarray(actual, dtype=float)
    b = np.asarray(reference, dtype=float)
    absolute = np.abs(a - b)
    relative = absolute / np.maximum(1.0e-10, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(absolute, initial=0.0)), float(np.max(relative, initial=0.0))


def median_runtime(fn: Callable[[], Any], *, repeats: int) -> tuple[float, Any]:
    return median_timed(fn, repeats=repeats, warmup=1)


def five_point_derivative(fn: Callable[[np.ndarray], float], x: Sequence[float], index: int, h: float) -> float:
    base = np.asarray(x, dtype=float)
    p2 = base.copy(); p2[index] += 2.0 * h
    p1 = base.copy(); p1[index] += h
    m1 = base.copy(); m1[index] -= h
    m2 = base.copy(); m2[index] -= 2.0 * h
    return float((-fn(p2) + 8.0 * fn(p1) - 8.0 * fn(m1) + fn(m2)) / (12.0 * h))


def scale_aware_step(value: float) -> float:
    # Truncation O(h^4) balanced with binary64 roundoff gives eps^(1/5).
    return float(np.finfo(float).eps ** 0.2 * max(1.0, abs(float(value))))


def _trajectory_records(topology: dict[str, Any], *, max_cases: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for case in topology["cases"]:
        for label in ("branch_and_bound", "astar"):
            record = case[label]["route"]
            cert = case[label]["certification"]
            if not cert["certified"]:
                continue
            raw = tuple(float(v) for v in record["raw_parameters"])
            if raw in seen:
                continue
            seen.add(raw)
            selected.append({"name": f"{case['case']['name']}/{label}", **record})
            if len(selected) >= max_cases:
                return selected
    return selected


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    topology_path = output_dir / "topology_search.json"
    if not topology_path.exists():
        raise FileNotFoundError("gradient benchmark requires topology_search.json; run topology_search first")
    topology = read_json(topology_path)
    config: OptimizationConfig = CORE_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    trajectories = _trajectory_records(topology, max_cases=5 if profile == "core" else 1)
    if not trajectories:
        raise RuntimeError("no certified trajectories available for gradient benchmark")

    repeats = 7 if profile == "core" else 2
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        raw = [float(v) for v in trajectory["raw_parameters"]]
        initial_k = float(trajectory.get("initial_k", 0.0))
        kwargs = dict(
            init_w=config.init_w,
            initial_k=initial_k,
            n_scan=config.n_scan,
            envelope_scan=config.envelope_scan,
            domain_scan=config.domain_scan,
        )
        with reverse_backend("python"):
            py_seconds, (py_value, py_grad) = median_runtime(
                lambda: reverse_solver.time_value_and_gradient(raw, **kwargs), repeats=repeats
            )
        with reverse_backend("native"):
            native_seconds, (native_value, native_grad) = median_runtime(
                lambda: reverse_solver.time_value_and_gradient(raw, **kwargs), repeats=repeats
            )
        max_abs, max_rel = _gradient_errors(native_grad, py_grad)
        rows.append({
            "name": trajectory["name"],
            "raw_parameters": raw,
            "initial_k": initial_k,
            "gradient_dimension": len(raw),
            "python_time_value": py_value,
            "native_time_value": native_value,
            "scalar_time_absolute_difference": abs(native_value - py_value),
            "maximum_absolute_gradient_difference": max_abs,
            "maximum_relative_gradient_difference": max_rel,
            "python_median_seconds": py_seconds,
            "native_median_seconds": native_seconds,
            "speedup_python_over_native": py_seconds / native_seconds if native_seconds > 0.0 else None,
        })

    finite_rows: list[dict[str, Any]] = []
    smooth_trajectories = trajectories[: (3 if profile == "core" else 1)]
    for trajectory in smooth_trajectories:
        raw = np.asarray(trajectory["raw_parameters"], dtype=float)
        initial_k = float(trajectory.get("initial_k", 0.0))
        kwargs = dict(
            init_w=config.init_w,
            initial_k=initial_k,
            n_scan=config.n_scan,
            envelope_scan=config.envelope_scan,
            domain_scan=config.domain_scan,
        )
        with reverse_backend("native"):
            value, gradient = reverse_solver.time_value_and_gradient(raw, **kwargs)
            def scalar_fn(candidate: np.ndarray) -> float:
                return float(scalar_reverse_solver.evaluate_time_scalar(candidate, **kwargs))

            for index, analytical in enumerate(gradient):
                h = scale_aware_step(raw[index])
                # Length coordinates must remain strictly positive at all stencil points.
                if index % 2 == 0 and raw[index] - 2.0 * h <= 0.0:
                    finite_rows.append({
                        "name": trajectory["name"], "index": index, "checked": False,
                        "reason": "length perturbation would become nonpositive", "h": h,
                    })
                    continue
                try:
                    d1 = five_point_derivative(scalar_fn, raw, index, h)
                    d2 = five_point_derivative(scalar_fn, raw, index, 0.5 * h)
                except Exception as exc:
                    finite_rows.append({
                        "name": trajectory["name"], "index": index, "checked": False,
                        "reason": f"physical evaluation failed under perturbation: {type(exc).__name__}: {exc}", "h": h,
                    })
                    continue
                # A large disagreement between two nearby fifth-order stencils is a deterministic
                # indicator of nonsmooth event switching / roundoff instability, not a pass.
                stability_scale = max(1.0e-7, 5.0e-4 * max(1.0, abs(d1), abs(d2)))
                if abs(d1 - d2) > stability_scale:
                    finite_rows.append({
                        "name": trajectory["name"], "index": index, "checked": False,
                        "reason": "finite-difference derivative unstable across h and h/2",
                        "h": h, "d_h": d1, "d_half_h": d2,
                    })
                    continue
                abs_error = abs(float(analytical) - d2)
                rel_error = _relative_error(float(analytical), d2)
                finite_rows.append({
                    "name": trajectory["name"], "index": index, "checked": True,
                    "h": h, "analytical": float(analytical), "finite_difference": d2,
                    "finite_difference_at_h": d1,
                    "absolute_error": abs_error, "relative_error": rel_error,
                })

    checked = [row for row in finite_rows if row.get("checked")]
    abs_errors = [row["absolute_error"] for row in checked]
    rel_errors = [row["relative_error"] for row in checked]
    def percentile(values: Sequence[float], q: float) -> float | None:
        return float(np.percentile(values, q)) if values else None

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "gradients",
        "profile": profile,
        "environment": environment_metadata(config),
        "implementation_crosscheck": rows,
        "finite_difference_checks": finite_rows,
        "aggregate": {
            "implementation_cases": len(rows),
            "median_time_value_absolute_difference": statistics.median([r["scalar_time_absolute_difference"] for r in rows]),
            "maximum_time_value_absolute_difference": max(r["scalar_time_absolute_difference"] for r in rows),
            "median_maximum_absolute_gradient_difference": statistics.median([r["maximum_absolute_gradient_difference"] for r in rows]),
            "maximum_absolute_gradient_difference": max(r["maximum_absolute_gradient_difference"] for r in rows),
            "maximum_relative_gradient_difference": max(r["maximum_relative_gradient_difference"] for r in rows),
            "median_native_speedup": statistics.median([r["speedup_python_over_native"] for r in rows]),
            "finite_difference_coordinates_total": len(finite_rows),
            "finite_difference_coordinates_checked": len(checked),
            "finite_difference_coordinates_excluded": len(finite_rows) - len(checked),
            "finite_difference_absolute_error": {
                "median": statistics.median(abs_errors) if abs_errors else None,
                "p95": percentile(abs_errors, 95.0),
                "max": max(abs_errors) if abs_errors else None,
            },
            "finite_difference_relative_error": {
                "median": statistics.median(rel_errors) if rel_errors else None,
                "p95": percentile(rel_errors, 95.0),
                "max": max(rel_errors) if rel_errors else None,
            },
        },
    }
    write_json(output_dir / "gradients.json", result)
    return result
