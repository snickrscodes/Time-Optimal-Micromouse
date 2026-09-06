from __future__ import annotations

from . import SCHEMA_VERSION

import bisect
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from optimization import reverse_solver, scalar_reverse_solver
from segment.constants import A_BRAKE, A_MAX, B_EMF, MU_G, V_MAX

from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.runtime import reverse_backend
from .common.status import CERTIFICATION_FAILURE, OPTIMIZER_FAILURE, SUCCESS
from .common.environment import casadi_ipopt_runtime_metadata
from .config import (
    CORE_OPTIMIZATION,
    DIRECT_TRANSCRIPTION_CASE_NAMES,
    DIRECT_TRANSCRIPTION_MESHES,
    DIRECT_TRANSCRIPTION_WARM_MESHES,
    SMOKE_DIRECT_TRANSCRIPTION_CASE_NAMES,
    SMOKE_DIRECT_TRANSCRIPTION_MESHES,
    SMOKE_OPTIMIZATION,
    OptimizationConfig,
)


@dataclass(frozen=True, slots=True)
class Geometry1D:
    lengths: np.ndarray
    sigmas: np.ndarray
    stations: np.ndarray
    knot_curvatures: np.ndarray

    @property
    def total_length(self) -> float:
        return float(self.stations[-1])

    def curvature(self, station: float) -> float:
        s = min(max(float(station), 0.0), self.total_length)
        if s >= self.total_length:
            return float(self.knot_curvatures[-1])
        i = bisect.bisect_right(self.stations, s) - 1
        i = max(0, min(i, len(self.lengths) - 1))
        return float(
            self.knot_curvatures[i]
            + self.sigmas[i] * (s - self.stations[i])
        )


def geometry_from_raw(raw_parameters: Sequence[float], initial_k: float) -> Geometry1D:
    raw = np.asarray(raw_parameters, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or raw.size % 2:
        raise ValueError("raw parameters must be a non-empty flat [L, sigma, ...] vector")
    lengths = raw[0::2].copy()
    sigmas = raw[1::2].copy()
    if not np.all(np.isfinite(raw)) or np.any(lengths <= 0.0):
        raise ValueError("raw geometry contains non-finite values or non-positive lengths")
    stations = np.concatenate(([0.0], np.cumsum(lengths)))
    knot_curvatures = np.empty(len(lengths) + 1, dtype=float)
    knot_curvatures[0] = float(initial_k)
    for i, (length, sigma) in enumerate(zip(lengths, sigmas)):
        knot_curvatures[i + 1] = knot_curvatures[i] + length * sigma
    return Geometry1D(lengths, sigmas, stations, knot_curvatures)


def transcription_stations(geometry: Geometry1D, base_intervals: int) -> np.ndarray:
    if base_intervals < 2:
        raise ValueError("base_intervals must be at least 2")
    uniform = np.linspace(0.0, geometry.total_length, int(base_intervals) + 1)
    # Exact clothoid knots are included so every transcription interval lies
    # within one fixed linear-curvature segment. The requested mesh size still
    # controls the maximum interval length; report the actual decision-node count.
    nodes = np.unique(np.concatenate((uniform, geometry.stations)))
    if nodes[0] != 0.0 or nodes[-1] != geometry.total_length:
        raise AssertionError("transcription station construction lost an endpoint")
    return nodes


def piecewise_linear_time(stations: Sequence[float], w: Sequence[float]) -> float:
    s = np.asarray(stations, dtype=float)
    ww = np.asarray(w, dtype=float)
    if s.shape != ww.shape or s.ndim != 1 or len(s) < 2:
        raise ValueError("stations and w must be equal-length 1D arrays")
    if np.any(np.diff(s) <= 0.0) or np.any(ww <= 0.0):
        raise ValueError("stations must increase and squared speed must be positive")
    ds = np.diff(s)
    # Exact integral of 1/sqrt(w) when w is linear on each interval.
    return float(np.sum(2.0 * ds / (np.sqrt(ww[:-1]) + np.sqrt(ww[1:]))))


def certify_piecewise_linear_profile(
    geometry: Geometry1D,
    stations: Sequence[float],
    w: Sequence[float],
    *,
    init_w: float,
    tolerance: float,
) -> dict[str, Any]:
    """Continuously certify the interpolated transcription profile.

    Every station interval lies in one clothoid segment, so both w(s) and k(s)
    are linear and q(s)=w(s)k(s) is quadratic. The friction-circle maximum on
    an interval is therefore attained at an endpoint or the single interior
    stationary point of q. Motor, brake, speed, positivity, and endpoint limits
    are likewise checked continuously rather than only on the NLP collocation
    points.
    """
    s = np.asarray(stations, dtype=float)
    ww = np.asarray(w, dtype=float)
    if s.shape != ww.shape or len(s) < 2:
        raise ValueError("invalid profile arrays")

    max_motor = -math.inf
    max_brake = -math.inf
    max_friction = -math.inf
    max_speed = -math.inf
    max_positivity = -math.inf
    worst_friction_station = 0.0
    minimum_w = float(np.min(ww))
    maximum_w = float(np.max(ww))

    for i in range(len(s) - 1):
        s0 = float(s[i]); s1 = float(s[i + 1]); ds = s1 - s0
        if ds <= 0.0:
            raise ValueError("stations must increase strictly")
        w0 = float(ww[i]); w1 = float(ww[i + 1])
        slope_w = (w1 - w0) / ds
        acceleration = 0.5 * slope_w

        # Since the motor RHS decreases monotonically with speed, its worst
        # continuous residual occurs at the maximum endpoint w of a linear cell.
        local_w_max = max(w0, w1)
        max_motor = max(
            max_motor,
            acceleration + B_EMF * math.sqrt(max(local_w_max, 0.0)) - A_MAX,
        )
        max_brake = max(max_brake, -A_BRAKE - acceleration)
        max_speed = max(max_speed, local_w_max - V_MAX * V_MAX)
        max_positivity = max(max_positivity, -min(w0, w1))

        k0 = geometry.curvature(s0)
        k1 = geometry.curvature(s1)
        slope_k = (k1 - k0) / ds
        candidates = [0.0, ds]
        denominator = 2.0 * slope_w * slope_k
        numerator = -(slope_w * k0 + slope_k * w0)
        scale = max(1.0, abs(slope_w * k0), abs(slope_k * w0))
        if abs(denominator) > 64.0 * math.ulp(scale):
            root = numerator / denominator
            if 0.0 < root < ds:
                candidates.append(root)
        for local_s in candidates:
            w_here = w0 + slope_w * local_s
            k_here = k0 + slope_k * local_s
            residual = acceleration * acceleration + (w_here * k_here) ** 2 - MU_G * MU_G
            if residual > max_friction:
                max_friction = residual
                worst_friction_station = s0 + local_s

    start_error = abs(float(ww[0]) - float(init_w))
    end_cap_residual = float(ww[-1]) - float(init_w)
    maximum_residual = max(
        max_motor,
        max_brake,
        max_friction,
        max_speed,
        max_positivity,
        start_error - tolerance,
        end_cap_residual - tolerance,
    )
    certified = (
        np.all(np.isfinite(ww))
        and minimum_w > 0.0
        and max_motor <= tolerance
        and max_brake <= tolerance
        and max_friction <= tolerance
        and max_speed <= tolerance
        and start_error <= tolerance
        and end_cap_residual <= tolerance
    )
    return {
        "certified": bool(certified),
        "tolerance": float(tolerance),
        "minimum_w": minimum_w,
        "maximum_w": maximum_w,
        "maximum_motor_residual": float(max_motor),
        "maximum_brake_residual": float(max_brake),
        "maximum_friction_circle_residual": float(max_friction),
        "worst_friction_station": float(worst_friction_station),
        "maximum_speed_residual": float(max_speed),
        "start_w_error": float(start_error),
        "terminal_w_cap_residual": float(end_cap_residual),
        "maximum_residual": float(maximum_residual),
        "independent_piecewise_linear_time": piecewise_linear_time(s, ww),
    }


def _production_profile_samples(
    raw: Sequence[float],
    initial_k: float,
    stations: Sequence[float],
    config: OptimizationConfig,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    with reverse_backend("python"):
        build = reverse_solver.build_scalar_speed_profile(
            raw,
            init_w=config.init_w,
            forward_init_k=initial_k,
            n_scan=config.n_scan,
            envelope_scan=config.envelope_scan,
            domain_scan=config.domain_scan,
        )
    elapsed = time.perf_counter() - started

    nodes = np.asarray(stations, dtype=float)
    values = np.empty_like(nodes)
    envelope_index = 0
    eps = 2.0e-11 * max(1.0, float(nodes[-1]))
    for j, station in enumerate(nodes):
        while (
            envelope_index + 1 < len(build.envelope)
            and station > build.envelope[envelope_index].abs1 + eps
        ):
            envelope_index += 1
        ep = build.envelope[envelope_index]
        if not (ep.abs0 - eps <= station <= ep.abs1 + eps):
            # At an exact envelope boundary either adjacent winner is valid;
            # search locally rather than depending on floating-point ordering.
            match = next(
                (
                    candidate for candidate in build.envelope
                    if candidate.abs0 - eps <= station <= candidate.abs1 + eps
                ),
                None,
            )
            if match is None:
                raise FloatingPointError(f"production envelope does not cover s={station:.17g}")
            ep = match
        record = build.scalar_passes[ep.pass_index].segments[ep.source_index]
        values[j] = reverse_solver.segment_w_at_abs(record, float(station))
    return values, float(elapsed)


def _casadi_ipopt_metadata() -> dict[str, Any]:
    return casadi_ipopt_runtime_metadata()


def _build_ipopt_problem(
    geometry: Geometry1D,
    stations: np.ndarray,
    *,
    init_w: float,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    import casadi as ca

    started = time.perf_counter()
    n = len(stations)
    w = ca.MX.sym("w", n)
    objective = 0
    constraints = []
    lower_g: list[float] = []
    upper_g: list[float] = []

    for i in range(n - 1):
        s0 = float(stations[i]); s1 = float(stations[i + 1]); ds = s1 - s0
        wi = w[i]; wj = w[i + 1]
        acceleration = (wj - wi) / (2.0 * ds)
        objective += 2.0 * ds / (ca.sqrt(wi) + ca.sqrt(wj))

        constraints.append(acceleration)
        lower_g.append(-A_BRAKE)
        upper_g.append(float("inf"))

        # Motor power is monotone in w, so checking both endpoints bounds the
        # full linear-w interval.  For friction, k(s) and w(s) are linear on
        # every transcription interval (exact clothoid knots are stations), so
        # q(s)=w(s)k(s) is quadratic.  Bounding its three quadratic Bernstein
        # coefficients is a conservative continuous interval guarantee, while
        # the independent certificate below still recomputes the exact interior
        # extremum rather than trusting the NLP constraints.
        for wa in (wi, wj):
            constraints.append(acceleration + B_EMF * ca.sqrt(wa))
            lower_g.append(-float("inf"))
            upper_g.append(A_MAX)

        k0 = geometry.curvature(s0)
        k1 = geometry.curvature(s1)
        q0 = wi * k0
        q2 = wj * k1
        qmid = 0.5 * (wi + wj) * (0.5 * (k0 + k1))
        q1 = 2.0 * qmid - 0.5 * (q0 + q2)
        for qb in (q0, q1, q2):
            constraints.append(acceleration * acceleration + qb * qb)
            lower_g.append(-float("inf"))
            upper_g.append(MU_G * MU_G)

    nlp = {"x": w, "f": objective, "g": ca.vertcat(*constraints)}
    options = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1.0e-9,
        "ipopt.acceptable_tol": 1.0e-8,
        "ipopt.constr_viol_tol": 1.0e-10,
        "ipopt.acceptable_constr_viol_tol": 1.0e-9,
    }
    solver = ca.nlpsol("direct_transcription", "ipopt", nlp, options)
    build_seconds = time.perf_counter() - started

    lower_x = np.full(n, 1.0e-9, dtype=float)
    upper_x = np.full(n, V_MAX * V_MAX, dtype=float)
    lower_x[0] = float(init_w)
    upper_x[0] = float(init_w)
    upper_x[-1] = min(float(init_w), float(V_MAX * V_MAX))
    return (
        solver,
        lower_x,
        upper_x,
        np.asarray(lower_g, dtype=float),
        np.asarray(upper_g, dtype=float),
        float(build_seconds),
    )


def solve_direct_transcription(
    geometry: Geometry1D,
    *,
    base_intervals: int,
    init_w: float,
    initial_guess: Sequence[float] | None,
    certification_tolerance: float,
) -> dict[str, Any]:
    stations = transcription_stations(geometry, base_intervals)
    if initial_guess is None:
        x0 = np.full(len(stations), float(init_w), dtype=float)
        initialization = "constant_endpoint_speed"
    else:
        x0 = np.asarray(initial_guess, dtype=float)
        if x0.shape != stations.shape:
            raise ValueError("initial_guess shape does not match transcription stations")
        initialization = "production_profile_sampled"

    solver, lbx, ubx, lbg, ubg, build_seconds = _build_ipopt_problem(
        geometry, stations, init_w=init_w
    )
    started = time.perf_counter()
    solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    solve_seconds = time.perf_counter() - started
    stats = solver.stats()
    w = np.asarray(solution["x"], dtype=float).reshape(-1)
    objective = float(solution["f"])
    certification = certify_piecewise_linear_profile(
        geometry,
        stations,
        w,
        init_w=init_w,
        tolerance=certification_tolerance,
    )
    objective_crosscheck_error = abs(
        objective - certification["independent_piecewise_linear_time"]
    )
    return {
        "base_uniform_intervals": int(base_intervals),
        "decision_nodes": int(len(stations)),
        "maximum_station_spacing": float(np.max(np.diff(stations))),
        "initialization": initialization,
        "solver_success": bool(stats.get("success", False)),
        "solver_return_status": stats.get("return_status"),
        "execution_status": (
            SUCCESS if bool(stats.get("success", False)) and certification["certified"]
            else CERTIFICATION_FAILURE if bool(stats.get("success", False))
            else OPTIMIZER_FAILURE
        ),
        "ipopt_iterations": int(stats.get("iter_count", -1)),
        "nlp_build_seconds": float(build_seconds),
        "ipopt_solve_seconds": float(solve_seconds),
        "build_plus_solve_seconds": float(build_seconds + solve_seconds),
        "objective_time": objective,
        "objective_crosscheck_absolute_error": float(objective_crosscheck_error),
        "certification": certification,
        "stations": stations.tolist(),
        "squared_speed": w.tolist(),
    }


def _median_production_time(
    raw: Sequence[float],
    initial_k: float,
    config: OptimizationConfig,
    repeats: int,
) -> tuple[float, float]:
    kwargs = dict(
        init_w=config.init_w,
        initial_k=initial_k,
        n_scan=config.n_scan,
        envelope_scan=config.envelope_scan,
        domain_scan=config.domain_scan,
    )
    with reverse_backend("native"):
        value = scalar_reverse_solver.evaluate_time_scalar(raw, **kwargs)  # warm-up
        timings: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            value = scalar_reverse_solver.evaluate_time_scalar(raw, **kwargs)
            timings.append(time.perf_counter() - started)
    return float(value), float(statistics.median(timings))


def _trajectories(topology: dict[str, Any], *, profile: str) -> list[dict[str, Any]]:
    requested = DIRECT_TRANSCRIPTION_CASE_NAMES if profile == "core" else SMOKE_DIRECT_TRANSCRIPTION_CASE_NAMES
    by_name = {case["case"]["name"]: case for case in topology["cases"]}
    out: list[dict[str, Any]] = []
    for name in requested:
        case = by_name.get(name)
        if case is None:
            raise RuntimeError(f"predeclared direct-transcription case missing from topology results: {name}")
        cert = case["branch_and_bound"]["certification"]
        if not cert["certified"]:
            raise RuntimeError(f"predeclared fixed geometry is not certified: {name}")
        route = case["branch_and_bound"]["route"]
        out.append({"name": name, **route})
    return out


def _safe_median(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(vals) if vals else None


def run(output_dir: Path, *, profile: str = "core") -> dict[str, Any]:
    topology_path = output_dir / "topology_search.json"
    if not topology_path.exists():
        raise FileNotFoundError("direct-transcription benchmark requires topology_search.json")
    topology = read_json(topology_path)
    config: OptimizationConfig = CORE_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION
    trajectories = _trajectories(topology, profile=profile)
    if not trajectories:
        raise RuntimeError("no certified fixed geometries available for direct transcription")

    mesh_sizes = DIRECT_TRANSCRIPTION_MESHES if profile == "core" else SMOKE_DIRECT_TRANSCRIPTION_MESHES
    repeats = 7 if profile == "core" else 2
    rows: list[dict[str, Any]] = []

    for trajectory in trajectories:
        raw = [float(v) for v in trajectory["raw_parameters"]]
        initial_k = float(trajectory.get("initial_k", 0.0))
        geometry = geometry_from_raw(raw, initial_k)
        production_time, production_median_seconds = _median_production_time(
            raw, initial_k, config, repeats
        )

        for mesh in mesh_sizes:
            stations = transcription_stations(geometry, mesh)
            initializations: list[tuple[str, np.ndarray | None, float]] = [("cold", None, 0.0)]
            # The production-sampled start is a basin cross-check, not the main
            # generic baseline. Keep it through 512 intervals; 1024-node warm
            # solves can spend disproportionate time following the sampled
            # hybrid kinks and add little evidence once cold refinement is shown.
            if mesh in DIRECT_TRANSCRIPTION_WARM_MESHES or profile != "core":
                production_samples, warm_start_generation_seconds = _production_profile_samples(
                    raw, initial_k, stations, config
                )
                initializations.append((
                    "production_sampled", production_samples, warm_start_generation_seconds
                ))
            for initialization, initial_guess, warm_generation_seconds in initializations:
                result = solve_direct_transcription(
                    geometry,
                    base_intervals=mesh,
                    init_w=config.init_w,
                    initial_guess=initial_guess,
                    certification_tolerance=config.feasibility_tolerance,
                )
                result["initialization_label"] = initialization
                result["warm_start_generation_seconds"] = float(warm_generation_seconds)
                result["production_time"] = production_time
                result["production_scalar_median_seconds"] = production_median_seconds
                result["absolute_time_difference"] = abs(result["objective_time"] - production_time)
                result["relative_time_difference"] = (
                    (result["objective_time"] - production_time) / production_time
                )
                result["relative_time_difference_percent"] = 100.0 * result["relative_time_difference"]
                result["speedup_production_over_ipopt_solve"] = (
                    result["ipopt_solve_seconds"] / production_median_seconds
                    if production_median_seconds > 0.0 else None
                )
                result["speedup_production_over_build_plus_solve"] = (
                    result["build_plus_solve_seconds"] / production_median_seconds
                    if production_median_seconds > 0.0 else None
                )
                rows.append({
                    "name": trajectory["name"],
                    "segments": len(raw) // 2,
                    "geometry_length": geometry.total_length,
                    "initial_k": initial_k,
                    **result,
                })

    certified_rows = [row for row in rows if row["certification"]["certified"]]
    cold_rows = [row for row in rows if row["initialization_label"] == "cold"]
    warm_rows = [row for row in rows if row["initialization_label"] == "production_sampled"]
    finest_mesh = max(mesh_sizes)
    finest_cold = [
        row for row in cold_rows
        if row["base_uniform_intervals"] == finest_mesh and row["certification"]["certified"]
    ]
    matched_pairs = []
    for cold in cold_rows:
        warm = next(
            (
                candidate for candidate in warm_rows
                if candidate["name"] == cold["name"]
                and candidate["base_uniform_intervals"] == cold["base_uniform_intervals"]
            ),
            None,
        )
        if warm is None:
            continue
        matched_pairs.append({
            "name": cold["name"],
            "base_uniform_intervals": cold["base_uniform_intervals"],
            "objective_absolute_difference": abs(cold["objective_time"] - warm["objective_time"]),
            "cold_iterations": cold["ipopt_iterations"],
            "warm_iterations": warm["ipopt_iterations"],
            "cold_solve_seconds": cold["ipopt_solve_seconds"],
            "warm_solve_seconds": warm["ipopt_solve_seconds"],
        })

    aggregate = {
        "cases": len(trajectories),
        "mesh_sizes": list(mesh_sizes),
        "rows": len(rows),
        "certified_rows": len(certified_rows),
        "certification_rate": len(certified_rows) / len(rows) if rows else None,
        "finest_mesh": finest_mesh,
        "finest_mesh_certified_cold_cases": len(finest_cold),
        "finest_mesh_cold_relative_time_difference_percent": {
            "median": _safe_median(abs(row["relative_time_difference_percent"]) for row in finest_cold),
            "max": max((abs(row["relative_time_difference_percent"]) for row in finest_cold), default=None),
        },
        "finest_mesh_cold_ipopt_solve_seconds": {
            "median": _safe_median(row["ipopt_solve_seconds"] for row in finest_cold),
            "max": max((row["ipopt_solve_seconds"] for row in finest_cold), default=None),
        },
        "finest_mesh_cold_production_scalar_seconds": {
            "median": _safe_median(row["production_scalar_median_seconds"] for row in finest_cold),
            "max": max((row["production_scalar_median_seconds"] for row in finest_cold), default=None),
        },
        "finest_mesh_cold_runtime_ratio_ipopt_over_production": {
            "median": _safe_median(row["speedup_production_over_ipopt_solve"] for row in finest_cold),
            "min": min((row["speedup_production_over_ipopt_solve"] for row in finest_cold), default=None),
            "max": max((row["speedup_production_over_ipopt_solve"] for row in finest_cold), default=None),
        },
        "maximum_cold_warm_objective_difference": max(
            (pair["objective_absolute_difference"] for pair in matched_pairs), default=None
        ),
        "matched_initialization_pairs": len(matched_pairs),
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "direct_transcription",
        "profile": profile,
        "environment": {
            **environment_metadata(config),
            **_casadi_ipopt_metadata(),
        },
        "methodology": {
            "state": "w = v^2 at arc-length stations",
            "control_representation": "piecewise-constant longitudinal acceleration implied by piecewise-linear w",
            "objective": "exact integral of 1/sqrt(w) for linear w on each station interval",
            "mesh": "uniform base grid augmented with every exact clothoid knot",
            "friction_interval_constraint": "quadratic Bernstein-envelope bound on q(s)=w(s)k(s); exact continuous extrema rechecked independently",
            "constraints": [
                "a >= -A_BRAKE",
                "a <= A_MAX - B_EMF*sqrt(w)",
                "a^2 + (w*k)^2 <= MU_G^2",
                "0 < w <= V_MAX^2",
                "w(0) = init_w; w(L) <= terminal_w_max (benchmark default terminal_w_max=init_w)",
            ],
            "cold_initialization": "constant w = init_w",
            "matched_initialization": "production hybrid profile sampled on the same decision nodes at the 256-interval core mesh; generation time excluded from IPOPT solve timing and reported separately",
            "continuous_certificate": "exact interval check for piecewise-linear w and piecewise-linear k, including analytic interior extrema of w*k",
            "quality_aggregate_policy": "only independently certified direct-transcription rows contribute to certified solution-quality aggregates",
        },
        "rows": rows,
        "initialization_crosscheck": matched_pairs,
        "aggregate": aggregate,
    }
    write_json(output_dir / "direct_transcription.json", result)
    return result
