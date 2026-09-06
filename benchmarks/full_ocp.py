from __future__ import annotations

from . import SCHEMA_VERSION

import bisect
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from optimization import CorridorModel, compile_geometry_path, knot_parameters_to_raw, scalar_reverse_solver, separate_rectangle_path
from segment.constants import A_BRAKE, A_MAX, B_EMF, MU_G, V_MAX
from planning import build_route_optimization_problem

from .common.certification import certify_parameters_on_problem, certified_only
from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.runtime import reverse_backend
from .common.status import CERTIFICATION_FAILURE, OPTIMIZER_FAILURE, SUCCESS
from .config import (
    FULL_OCP_CASE_NAME,
    FULL_OCP_OPTIMIZATION,
    FULL_OCP_MESHES,
    FULL_OCP_STRUCTURED_REFINEMENTS,
    SMOKE_FULL_OCP_CASE_NAME,
    SMOKE_FULL_OCP_MESHES,
    SMOKE_OPTIMIZATION,
    OptimizationConfig,
)
from .direct_transcription import (
    _casadi_ipopt_metadata,
    _production_profile_samples,
    certify_piecewise_linear_profile,
    geometry_from_raw,
)


# 8-point Gauss-Legendre quadrature on [-1, 1].  On each OCP interval
# theta(s) is quadratic, so this gives a high-accuracy generic integration of
# x'=cos(theta), y'=sin(theta) without importing the production Fresnel kernel.
_GL8_X = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
     0.1834346424956498,
     0.5255324099163290,
     0.7966664774136267,
     0.9602898564975363,
)
_GL8_W = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)
_WALL_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0)
_MAX_WALL_EXCHANGE_ROUNDS = 6
_DYNAMICS_SAMPLES = (0.0, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class Phase:
    cell_index: int
    source_segment_start: int
    source_segment_end: int  # exclusive
    initial_length: float


@dataclass(frozen=True, slots=True)
class MeshLayout:
    phases: tuple[Phase, ...]
    intervals_per_phase: tuple[int, ...]
    phase_for_interval: tuple[int, ...]

    @property
    def intervals(self) -> int:
        return len(self.phase_for_interval)


def _phases(problem) -> tuple[Phase, ...]:
    raw = knot_parameters_to_raw(problem.initial_parameters, initial_k=problem.initial_state.k)
    lengths = np.asarray(raw[0::2], dtype=float)
    assignments = tuple(problem.corridor.segment_cells)
    if len(lengths) != len(assignments):
        raise AssertionError("initializer/corridor segment count mismatch")
    out: list[Phase] = []
    begin = 0
    while begin < len(assignments):
        cell = assignments[begin]
        end = begin + 1
        while end < len(assignments) and assignments[end] == cell:
            end += 1
        out.append(Phase(cell, begin, end, float(np.sum(lengths[begin:end]))))
        begin = end
    return tuple(out)


def _allocate_intervals(phases: Sequence[Phase], total: int) -> tuple[int, ...]:
    if total < 2 * len(phases):
        raise ValueError("mesh needs at least two intervals per corridor phase")
    lengths = np.asarray([p.initial_length for p in phases], dtype=float)
    remaining = total - 2 * len(phases)
    ideal = remaining * lengths / float(np.sum(lengths))
    extra = np.floor(ideal).astype(int)
    left = remaining - int(np.sum(extra))
    order = np.argsort(-(ideal - extra), kind="stable")
    for idx in order[:left]:
        extra[idx] += 1
    counts = tuple(int(v + 2) for v in extra)
    if sum(counts) != total:
        raise AssertionError("interval allocation failed")
    return counts


def build_mesh_layout(problem, base_intervals: int) -> MeshLayout:
    phases = _phases(problem)
    counts = _allocate_intervals(phases, int(base_intervals))
    mapping: list[int] = []
    for p, count in enumerate(counts):
        mapping.extend([p] * count)
    return MeshLayout(phases, counts, tuple(mapping))


def _phase_lengths_from_raw(raw: Sequence[float], phases: Sequence[Phase]) -> np.ndarray:
    lengths = np.asarray(raw[0::2], dtype=float)
    return np.asarray([
        float(np.sum(lengths[p.source_segment_start:p.source_segment_end])) for p in phases
    ], dtype=float)


def _mesh_stations(layout: MeshLayout, phase_lengths: Sequence[float]) -> np.ndarray:
    phase_lengths = np.asarray(phase_lengths, dtype=float)
    stations = [0.0]
    s = 0.0
    for length, count in zip(phase_lengths, layout.intervals_per_phase, strict=True):
        h = float(length) / count
        for _ in range(count):
            s += h
            stations.append(s)
    return np.asarray(stations, dtype=float)


def _sample_geometry(raw: Sequence[float], initial_state, stations: Sequence[float]) -> np.ndarray:
    path = compile_geometry_path(raw, initial_state)
    ends = np.cumsum(np.asarray(raw[0::2], dtype=float))
    starts = np.concatenate(([0.0], ends[:-1]))
    values = np.empty((4, len(stations)), dtype=float)
    for j, station_in in enumerate(stations):
        station = min(max(float(station_in), 0.0), float(ends[-1]))
        if station >= ends[-1]:
            state = path.final_state
        else:
            i = bisect.bisect_right(ends.tolist(), station)
            i = max(0, min(i, path.n_segments - 1))
            length = float(ends[i] - starts[i])
            tau = (station - starts[i]) / length
            state = path.state_at_fraction(i, tau)
        values[:, j] = state
    return values


def _initial_guess(problem, layout: MeshLayout, *, route_record: dict[str, Any] | None,
                   config: OptimizationConfig) -> dict[str, np.ndarray]:
    if route_record is None:
        parameters = np.asarray(problem.initial_parameters, dtype=float)
        raw = np.asarray(knot_parameters_to_raw(parameters, initial_k=problem.initial_state.k), dtype=float)
        phase_lengths = np.asarray([p.initial_length for p in layout.phases], dtype=float)
        stations = _mesh_stations(layout, phase_lengths)
        states4 = _sample_geometry(raw, problem.initial_state, stations)
        w = np.full(len(stations), config.init_w, dtype=float)
        label = "analytic_initializer_constant_speed"
    else:
        raw = np.asarray(route_record["raw_parameters"], dtype=float)
        phase_lengths = _phase_lengths_from_raw(raw, layout.phases)
        stations = _mesh_stations(layout, phase_lengths)
        states4 = _sample_geometry(raw, problem.initial_state, stations)
        w, _ = _production_profile_samples(raw, float(route_record.get("initial_k", 0.0)), stations, config)
        label = "production_solution_sampled"

    phase_base = np.asarray([p.initial_length for p in layout.phases], dtype=float)
    z = np.log(phase_lengths / phase_base)
    sigma = np.diff(states4[3]) / np.diff(stations)
    acceleration = np.diff(w) / (2.0 * np.diff(stations))
    states = np.vstack((states4, w))
    return {
        "label": np.asarray([label], dtype=object),
        "z": z,
        "states": states,
        "sigma": sigma,
        "acceleration": acceleration,
        "stations": stations,
    }


def _gauss_displacement(ca, theta0, k0, sigma, h):
    half = 0.5 * h
    dx = 0
    dy = 0
    for q, weight in zip(_GL8_X, _GL8_W, strict=True):
        t = half * (q + 1.0)
        theta = theta0 + k0 * t + 0.5 * sigma * t * t
        dx += weight * ca.cos(theta)
        dy += weight * ca.sin(theta)
    return half * dx, half * dy


def _local_pose(ca, x0, y0, theta0, k0, sigma, h, alpha: float):
    d = float(alpha) * h
    if alpha == 0.0:
        return x0, y0, theta0, k0
    dx, dy = _gauss_displacement(ca, theta0, k0, sigma, d)
    theta = theta0 + k0 * d + 0.5 * sigma * d * d
    k = k0 + sigma * d
    return x0 + dx, y0 + dy, theta, k


def _wall_constraints(opti, ca, problem, cell_index: int, x, y, theta):
    cell = problem.corridor.cells[cell_index]
    body = problem.corridor.body
    clearance = float(problem.corridor.clearance)
    c = ca.cos(theta); s = ca.sin(theta)
    for corner in body.corners:
        cx = x + corner.u * c - corner.v * s
        cy = y + corner.u * s + corner.v * c
        for wall in cell.walls:
            nx, ny = wall.normal
            opti.subject_to(nx * cx + ny * cy <= wall.offset - clearance)


def _wall_constraint_cut(opti, ca, problem, cell_index: int, corner_index: int, wall_index: int, x, y, theta):
    cell = problem.corridor.cells[cell_index]
    body = problem.corridor.body
    clearance = float(problem.corridor.clearance)
    corner = body.corners[int(corner_index)]
    wall = cell.walls[int(wall_index)]
    c = ca.cos(theta); ss = ca.sin(theta)
    cx = x + corner.u * c - corner.v * ss
    cy = y + corner.u * ss + corner.v * c
    nx, ny = wall.normal
    opti.subject_to(nx * cx + ny * cy <= wall.offset - clearance)


def _build_problem(problem, layout: MeshLayout, *, config: OptimizationConfig, wall_cuts: Sequence[tuple[int, int, int, float]] = ()):

    import casadi as ca

    started = time.perf_counter()
    opti = ca.Opti()
    n = layout.intervals
    pcount = len(layout.phases)
    z = opti.variable(pcount)
    X = opti.variable(5, n + 1)  # x,y,theta,k,w
    sigma = opti.variable(n)
    accel = opti.variable(n)

    # Broad numerical safeguards.  The production optimizer itself leaves
    # station coordinates unbounded; these log-length bounds span >400x and are
    # recorded so we can verify that the solution is nowhere near them.
    opti.subject_to(opti.bounded(-16.0, z, 3.0))
    opti.subject_to(opti.bounded(-10.0, X[3, :], 10.0))
    opti.subject_to(opti.bounded(1.0e-8, X[4, :], V_MAX * V_MAX))
    if config.curvature_slope_limit is not None:
        opti.subject_to(opti.bounded(-config.curvature_slope_limit, sigma, config.curvature_slope_limit))
    opti.subject_to(accel >= -A_BRAKE)

    start = problem.initial_state
    gx, gy = problem.terminal_cell
    target = problem.endpoint_target
    opti.subject_to(X[0, 0] == start.x)
    opti.subject_to(X[1, 0] == start.y)
    opti.subject_to(X[2, 0] == start.theta)
    opti.subject_to(X[3, 0] == start.k)
    opti.subject_to(X[4, 0] == config.init_w)
    # Terminate on the same finite goal-entry portal as production.  Exactly
    # one of x/y is fixed by the route's final cell transition; the other may
    # slide along the shared edge.  Heading/curvature are free and speed is
    # capped rather than fixed.
    opti.subject_to(opti.bounded(float(gx), X[0, -1], float(gx + 1)))
    opti.subject_to(opti.bounded(float(gy), X[1, -1], float(gy + 1)))
    if target[0] is not None:
        opti.subject_to(X[0, -1] == float(target[0]))
    if target[1] is not None:
        opti.subject_to(X[1, -1] == float(target[1]))
    opti.subject_to(X[4, -1] <= config.init_w)

    phase_lengths = [layout.phases[p].initial_length * ca.exp(z[p]) for p in range(pcount)]
    objective = 0
    cuts_by_interval: dict[int, list[tuple[int, int, float]]] = {}
    for cut_interval, wall_index, corner_index, tau in wall_cuts:
        cuts_by_interval.setdefault(int(cut_interval), []).append(
            (int(wall_index), int(corner_index), float(tau))
        )
    interval = 0
    for phase_index, count in enumerate(layout.intervals_per_phase):
        h = phase_lengths[phase_index] / count
        cell_index = layout.phases[phase_index].cell_index
        for _local in range(count):
            xi, yi, thi, ki, wi = [X[r, interval] for r in range(5)]
            xj, yj, thj, kj, wj = [X[r, interval + 1] for r in range(5)]
            sj = sigma[interval]
            aj = accel[interval]

            dx, dy = _gauss_displacement(ca, thi, ki, sj, h)
            opti.subject_to(xj == xi + dx)
            opti.subject_to(yj == yi + dy)
            opti.subject_to(thj == thi + ki * h + 0.5 * sj * h * h)
            opti.subject_to(kj == ki + sj * h)
            opti.subject_to(wj == wi + 2.0 * aj * h)

            objective += 2.0 * h / (ca.sqrt(wi) + ca.sqrt(wj))

            # Motor RHS is monotone in w, so endpoint checks are continuous
            # over an interval with linear w.  For the friction circle, q=w*k
            # is quadratic in normalized station alpha.  Bounding all three
            # quadratic Bernstein coefficients bounds q everywhere because the
            # Bernstein basis is nonnegative and sums to one.
            opti.subject_to(aj + B_EMF * ca.sqrt(wi) <= A_MAX)
            opti.subject_to(aj + B_EMF * ca.sqrt(wj) <= A_MAX)
            q0 = wi * ki
            q2 = wj * kj
            wmid = 0.5 * (wi + wj)
            kmid = 0.5 * (ki + kj)
            qmid = wmid * kmid
            q1 = 2.0 * qmid - 0.5 * (q0 + q2)
            for qb in (q0, q1, q2):
                opti.subject_to(aj * aj + qb * qb <= MU_G * MU_G)

            for alpha in _WALL_SAMPLES:
                xp, yp, thp, _kp = _local_pose(ca, xi, yi, thi, ki, sj, h, alpha)
                _wall_constraints(opti, ca, problem, cell_index, xp, yp, thp)
            for wall_index, corner_index, alpha in cuts_by_interval.get(interval, ()):
                xp, yp, thp, _kp = _local_pose(ca, xi, yi, thi, ki, sj, h, alpha)
                _wall_constraint_cut(
                    opti, ca, problem, cell_index, corner_index, wall_index, xp, yp, thp
                )
            interval += 1

    opti.minimize(objective)
    opts = {
        "print_time": False,
        "expand": True,
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 2000,
            "tol": 1.0e-8,
            "acceptable_tol": 1.0e-7,
            "constr_viol_tol": 1.0e-9,
            "acceptable_constr_viol_tol": 1.0e-8,
            "mu_strategy": "adaptive",
        },
    }
    opti.solver("ipopt", opts)
    return opti, z, X, sigma, accel, objective, phase_lengths, float(time.perf_counter() - started)


def _expanded_certificate(problem, layout: MeshLayout, phase_lengths: np.ndarray,
                          curvatures: np.ndarray, squared_speed: np.ndarray,
                          *, config: OptimizationConfig) -> dict[str, Any]:
    hs: list[float] = []
    assignments: list[int] = []
    for p, count in enumerate(layout.intervals_per_phase):
        h = float(phase_lengths[p]) / count
        hs.extend([h] * count)
        assignments.extend([layout.phases[p].cell_index] * count)
    h_arr = np.asarray(hs, dtype=float)
    sigma = np.diff(curvatures) / h_arr
    raw = np.empty(2 * len(h_arr), dtype=float)
    raw[0::2] = h_arr
    raw[1::2] = sigma
    stations = np.concatenate(([0.0], np.cumsum(h_arr)))
    params = np.empty(2 * len(h_arr), dtype=float)
    params[0::2] = stations[1:]
    params[1::2] = curvatures[1:]
    expanded_corridor = CorridorModel(
        problem.corridor.cells,
        tuple(assignments),
        problem.corridor.body,
        problem.corridor.clearance,
    )
    expanded_problem = dataclasses.replace(problem, corridor=expanded_corridor)
    geometry_cert = certify_parameters_on_problem(
        expanded_problem,
        params,
        tolerance=config.feasibility_tolerance,
        maximum_abs_sigma=config.curvature_slope_limit,
    )
    separation_report = separate_rectangle_path(
        params, problem.initial_state, expanded_corridor
    )
    geometry = geometry_from_raw(raw, initial_k=problem.initial_state.k)
    speed = certify_piecewise_linear_profile(
        geometry,
        stations,
        squared_speed,
        init_w=config.init_w,
        tolerance=config.feasibility_tolerance,
    )
    certificate = {
        "certified": bool(geometry_cert["certified"] and speed["certified"]),
        "geometry_certified": bool(geometry_cert["certified"]),
        "speed_certified": bool(speed["certified"]),
        "endpoint_error": float(geometry_cert["endpoint_error"]),
        "corridor_upper_bound": float(geometry_cert["corridor_upper_bound"]),
        "final_state": geometry_cert["final_state"],
        "maximum_abs_sigma": float(np.max(np.abs(sigma))),
        "speed": speed,
        "raw_parameters": raw.tolist(),
        "parameters": params.tolist(),
        "stations": stations.tolist(),
    }
    return certificate, separation_report


def solve_full_ocp(problem, route_record: dict[str, Any], *, base_intervals: int,
                   initialization: str, config: OptimizationConfig) -> dict[str, Any]:
    layout = build_mesh_layout(problem, base_intervals)
    if initialization == "cold":
        guess = _initial_guess(problem, layout, route_record=None, config=config)
    elif initialization == "production_sampled":
        guess = _initial_guess(problem, layout, route_record=route_record, config=config)
    else:
        raise ValueError(initialization)

    wall_cuts: list[tuple[int, int, int, float]] = []
    cut_keys: set[tuple[int, int, int, int]] = set()
    exchange_records: list[dict[str, Any]] = []
    total_build_seconds = 0.0
    total_solve_seconds = 0.0
    total_iterations = 0
    final_nlp_variables = 0
    final_nlp_constraints = 0
    success = False
    status = None
    certificate = None
    z_value = states = sigma_value = accel_value = phase_lengths = stations = None

    current_guess = guess
    for exchange_round in range(_MAX_WALL_EXCHANGE_ROUNDS + 1):
        opti, z, X, sigma, accel, objective, phase_exprs, build_seconds = _build_problem(
            problem, layout, config=config, wall_cuts=wall_cuts
        )
        final_nlp_variables = int(opti.nx)
        final_nlp_constraints = int(opti.ng)
        total_build_seconds += float(build_seconds)
        opti.set_initial(z, current_guess["z"])
        opti.set_initial(X, current_guess["states"])
        opti.set_initial(sigma, current_guess["sigma"])
        opti.set_initial(accel, current_guess["acceleration"])

        started = time.perf_counter()
        round_success = False
        try:
            sol = opti.solve()
            round_success = True
            stats = opti.stats()
            status = stats.get("return_status")
            iterations = int(stats.get("iter_count", -1))
            value = lambda expr: np.asarray(sol.value(expr), dtype=float)
        except RuntimeError as exc:
            stats = opti.stats()
            status = stats.get("return_status") or type(exc).__name__
            iterations = int(stats.get("iter_count", -1))
            value = lambda expr: np.asarray(opti.debug.value(expr), dtype=float)
        round_solve_seconds = float(time.perf_counter() - started)
        total_solve_seconds += round_solve_seconds
        total_iterations += max(0, iterations)

        z_value = value(z).reshape(-1)
        states = value(X)
        sigma_value = value(sigma).reshape(-1)
        accel_value = value(accel).reshape(-1)
        phase_lengths = np.asarray([
            layout.phases[p].initial_length * math.exp(float(z_value[p]))
            for p in range(len(layout.phases))
        ], dtype=float)
        stations = _mesh_stations(layout, phase_lengths)
        certificate, separation_report = _expanded_certificate(
            problem, layout, phase_lengths, states[3, :], states[4, :], config=config
        )
        success = bool(round_success)
        exchange_records.append({
            "round": int(exchange_round),
            "solver_success": bool(round_success),
            "solver_return_status": status,
            "iterations": int(iterations),
            "solve_seconds": round_solve_seconds,
            "wall_cuts_before_round": int(len(wall_cuts)),
            "corridor_upper_bound": float(certificate["corridor_upper_bound"]),
            "geometry_certified": bool(certificate["geometry_certified"]),
        })
        if not round_success or certificate["geometry_certified"]:
            break
        if exchange_round >= _MAX_WALL_EXCHANGE_ROUNDS:
            break

        added = 0
        for maximum in separation_report.violating_maxima:
            segment = int(maximum.family.segment)
            wall_index = int(maximum.family.wall)
            corner_index = int(maximum.family.corner)
            tau = float(maximum.tau)
            # Quantize only for de-duplication; retain the exact tau in the cut.
            key = (segment, wall_index, corner_index, int(round(tau * 1.0e12)))
            if key in cut_keys:
                continue
            cut_keys.add(key)
            wall_cuts.append((segment, wall_index, corner_index, tau))
            added += 1
        exchange_records[-1]["wall_cuts_added"] = int(added)
        if added == 0:
            break
        current_guess = {
            "z": z_value.copy(),
            "states": states.copy(),
            "sigma": sigma_value.copy(),
            "acceleration": accel_value.copy(),
        }

    assert certificate is not None
    assert z_value is not None and states is not None and sigma_value is not None
    assert accel_value is not None and phase_lengths is not None and stations is not None
    # Evaluate the objective independently from the reconstructed piecewise-linear w.
    objective_value = float(np.sum(
        2.0 * np.diff(stations) /
        (np.sqrt(states[4, :-1]) + np.sqrt(states[4, 1:]))
    ))
    exact_time = float(certificate["speed"]["independent_piecewise_linear_time"])
    continuous_hybrid_time = None
    continuous_hybrid_eval_seconds = None
    if certificate["geometry_certified"]:
        raw_cert = certificate["raw_parameters"]
        started_hybrid = time.perf_counter()
        try:
            with reverse_backend("native"):
                continuous_hybrid_time = float(scalar_reverse_solver.evaluate_time_scalar(
                    raw_cert,
                    init_w=config.init_w,
                    terminal_w_max=config.init_w,
                    initial_k=problem.initial_state.k,
                    n_scan=config.n_scan,
                    envelope_scan=config.envelope_scan,
                    domain_scan=config.domain_scan,
                ))
        except Exception:
            continuous_hybrid_time = None
        continuous_hybrid_eval_seconds = float(time.perf_counter() - started_hybrid)
    phase_log_bound_margin = float(min(np.min(z_value + 16.0), np.min(3.0 - z_value)))
    return {
        "base_intervals": int(base_intervals),
        "actual_intervals": int(layout.intervals),
        "decision_nodes": int(layout.intervals + 1),
        "nlp_variables": final_nlp_variables,
        "intrinsic_geometry_dof": int(layout.intervals + len(layout.phases)),
        "intrinsic_speed_dof": int(layout.intervals),
        "intrinsic_total_dof": int(2 * layout.intervals + len(layout.phases)),
        "nlp_constraints": final_nlp_constraints,
        "corridor_phases": len(layout.phases),
        "intervals_per_phase": list(layout.intervals_per_phase),
        "phase_cell_indices": [p.cell_index for p in layout.phases],
        "phase_lengths": phase_lengths.tolist(),
        "total_length": float(np.sum(phase_lengths)),
        "initialization": initialization,
        "solver_success": bool(success),
        "solver_return_status": status,
        "execution_status": (
            SUCCESS if bool(success) and certificate["certified"]
            else CERTIFICATION_FAILURE if bool(success)
            else OPTIMIZER_FAILURE
        ),
        "ipopt_iterations": int(total_iterations),
        "nlp_build_seconds": float(total_build_seconds),
        "ipopt_solve_seconds": float(total_solve_seconds),
        "build_plus_solve_seconds": float(total_build_seconds + total_solve_seconds),
        "objective_time": objective_value,
        "exact_piecewise_speed_time": exact_time,
        "objective_crosscheck_absolute_error": abs(objective_value - exact_time),
        "continuous_hybrid_time_on_ocp_geometry": continuous_hybrid_time,
        "continuous_hybrid_eval_seconds": continuous_hybrid_eval_seconds,
        "phase_log_bound_margin": phase_log_bound_margin,
        "maximum_abs_acceleration": float(np.max(np.abs(accel_value))),
        "maximum_abs_sigma_decision": float(np.max(np.abs(sigma_value))),
        "wall_exchange_rounds": int(max(0, len(exchange_records) - 1)),
        "wall_exchange_cuts": int(len(wall_cuts)),
        "wall_exchange_records": exchange_records,
        "certificate": certificate,
        "states": states.tolist(),
        "sigma": sigma_value.tolist(),
        "acceleration": accel_value.tolist(),
        "comparison_times": {
            "generic_ocp_objective": objective_value,
            "generic_piecewise_speed_recheck": exact_time,
            "production_hybrid_on_ocp_geometry": continuous_hybrid_time,
        },
    }



def _select_case(topology: dict[str, Any], *, profile: str) -> dict[str, Any]:
    name = FULL_OCP_CASE_NAME if profile == "core" else SMOKE_FULL_OCP_CASE_NAME
    for case in topology["cases"]:
        if case["case"]["name"] == name:
            return case
    raise RuntimeError(f"predeclared full-OCP case missing from topology results: {name}")


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    certified = certified_only(rows, key="certificate")
    cold = [row for row in certified if row["initialization"] == "cold"]
    warm = [row for row in certified if row["initialization"] == "production_sampled"]
    finest = max((row["base_intervals"] for row in rows), default=None)
    finest_cert = [row for row in certified if row["base_intervals"] == finest]
    best_cert = min(certified, key=lambda r: r["objective_time"]) if certified else None
    return {
        "solves": len(rows),
        "certified_solves": len(certified),
        "certified_cold_solves": len(cold),
        "certified_production_sampled_solves": len(warm),
        "mesh_sizes": sorted({row["base_intervals"] for row in rows}),
        "finest_mesh": finest,
        "finest_mesh_certified_solves": len(finest_cert),
        "best_certified_ocp_time": None if best_cert is None else float(best_cert["objective_time"]),
        "best_certified_runtime_seconds": None if best_cert is None else float(best_cert["ipopt_solve_seconds"]),
        "best_certified_initialization": None if best_cert is None else best_cert["initialization"],
        "best_certified_mesh": None if best_cert is None else best_cert["base_intervals"],
        "maximum_certified_endpoint_error": max((r["certificate"]["endpoint_error"] for r in certified), default=None),
        "maximum_certified_corridor_upper_bound": max((r["certificate"]["corridor_upper_bound"] for r in certified), default=None),
    }


def assemble(output_dir: Path, *, profile: str) -> dict[str, Any]:
    topology_path = output_dir / "topology_search.json"
    topology = read_json(topology_path)
    case = _select_case(topology, profile=profile)
    source_route = case["branch_and_bound"]["route"]
    config = FULL_OCP_OPTIMIZATION if profile == "core" else SMOKE_OPTIMIZATION

    meshes = FULL_OCP_MESHES if profile == "core" else SMOKE_FULL_OCP_MESHES
    rows = [read_json(output_dir / f"full_ocp_ocp_{int(mesh)}.json")["row"] for mesh in meshes]
    refinements = FULL_OCP_STRUCTURED_REFINEMENTS if profile == "core" else (1,)
    structured_references = [
        read_json(output_dir / f"full_ocp_structured_ref{refinement}.json")
        for refinement in refinements
    ]
    production_reference = next(r for r in structured_references if r["geometry_refinement"] == 1)
    production_certification = production_reference["certification"]
    if not production_certification["certified"]:
        raise RuntimeError("full-budget production fixed-topology reference failed certification")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "full_ocp",
        "profile": profile,
        "environment": {**environment_metadata(config), **_casadi_ipopt_metadata()},
        "case": {
            "name": case["case"]["name"],
            "maze": case["maze"],
            "cells": source_route["cells"],
            "topology_source_route_time": float(source_route["time"]),
            "initial_state": production_reference["route"]["initial_state"],
        },
        "formulation": {
            "independent_variable": "arclength within variable-length corridor phases",
            "states": ["x", "y", "theta", "k", "w=v^2"],
            "controls": ["sigma=dk/ds", "a=0.5*dw/ds"],
            "geometry_integration": "8-point Gauss-Legendre per interval; independent certificate uses production exact clothoid/Fresnel geometry",
            "corridor": "same production overlapping convex cover; one variable-length OCP phase per consecutive cover cell; oriented rectangle constrained at 5 base samples/interval plus deterministic exact-certifier wall-cut exchange",
            "dynamics": "same motor, braking, speed, friction-circle and endpoint-speed limits; motor/brake/friction are continuously bounded per interval and rechecked independently",
            "initialization_cold": "analytic Euler/clothoid route initializer with constant endpoint squared speed; does not use the optimized production trajectory",
            "phase_length_log_bounds": [-16.0, 3.0],
            "curvature_bounds": [-10.0, 10.0],
            "wall_samples_per_interval": list(_WALL_SAMPLES),
            "wall_constraint_exchange": f"up to {_MAX_WALL_EXCHANGE_ROUNDS} deterministic rounds; exact continuous separator maxima are inserted as additional CasADi wall constraints",
            "production_reference_budget": config.to_dict(),
            "process_isolation": "each CasADi/IPOPT mesh and each supervised structured reference runs in a fresh interpreter; process startup is excluded from solver timing",
        },
        "structured_references": structured_references,
        "structured_reference_note": "Independent structured refinement solves are optimizer-robustness diagnostics only. Canonical OCP-vs-structured comparison values live in full_ocp_resolution_control.json.",
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    certified_structured = [r for r in structured_references if r["certification"]["certified"]]
    result["aggregate"]["structured_references_certified"] = len(certified_structured)
    result["aggregate"]["structured_refinements"] = [r["geometry_refinement"] for r in structured_references]
    write_json(output_dir / "full_ocp.json", result)
    return result
