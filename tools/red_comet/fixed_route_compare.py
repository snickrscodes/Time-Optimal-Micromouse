"""Controlled fixed-topology Red Comet calibration experiment.

This is intentionally *not* branch-and-bound.  It optimizes only two known
cell topologies under one explicitly selected physical profile:

* the production A* topology;
* the historical green long-route topology.

Each route is solved in its own supervised subprocess.  By default the tool
builds matching native Segment/Crossing/Reverse kernels in an isolated temporary
copy of the repository, so the calibrated experiment gets production-speed
numerics without mutating the legacy-qualified native libraries.  A slower
Python-reference backend remains available for differential/debug work.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


DEFAULT_PROFILE = "red_comet_2017_dd_yaw_v1"
DEFAULT_BODY_LENGTH = 76.0 / 180.0
DEFAULT_BODY_WIDTH = 45.0 / 180.0
DEFAULT_INIT_W = 0.8
DEFAULT_TERMINAL_W_MAX = 0.8
REPO_ROOT = Path(__file__).resolve().parents[2]


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _descendant_pids(root_pid: int) -> list[int]:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    except Exception:
        return []
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        pid, ppid = map(int, fields)
        children.setdefault(ppid, []).append(pid)
    result: list[int] = []
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        result.append(pid)
        stack.extend(children.get(pid, ()))
    return result


def _signal_pids(pids: list[int], sig: int) -> None:
    for pid in reversed(pids):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _kill_tree(process: subprocess.Popen[Any]) -> None:
    descendants = _descendant_pids(process.pid)
    _signal_pids(descendants, signal.SIGTERM)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        _signal_pids(descendants, signal.SIGKILL)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2.0)
    _signal_pids(descendants, signal.SIGKILL)


def _enumerated_named_paths(maze_path: Path) -> dict[str, tuple[tuple[int, int], ...]]:
    # Imports are local so the parent supervisor never binds a physics profile.
    from mazegen import compress_to_graph
    from planning.branch_and_bound import _block_cut_relevant_subgraph
    from planning.maze_io import load_maze_scenario
    from planning.maze_routes import astar_junction_path, expand_junction_path
    from tools.red_comet.preflight import _enumerate_paths
    from tools.red_comet.routes import find_historical_green_path

    scenario = load_maze_scenario(maze_path)
    graph = compress_to_graph(scenario.maze, scenario.start, scenario.canonical_goal)
    source = graph.index[scenario.start]
    target = graph.index[scenario.canonical_goal]
    relevant = _block_cut_relevant_subgraph(graph, source, target)
    junction_paths, truncated = _enumerate_paths(
        graph, relevant, source, target, 1_000_000
    )
    if truncated:
        raise RuntimeError("Red Comet simple-path enumeration unexpectedly truncated")
    cell_paths = [
        tuple(expand_junction_path(scenario.maze, graph, path))
        for path in junction_paths
    ]
    astar_nodes = tuple(astar_junction_path(graph, scenario.start, scenario.canonical_goal))
    astar_cells = tuple(expand_junction_path(scenario.maze, graph, astar_nodes))
    historical_green = find_historical_green_path(scenario, cell_paths)
    return {"astar": astar_cells, "historical_green": historical_green}


def _worker(
    maze_path: Path,
    route_name: str,
    output: Path,
    iterations: int,
    body_length: float,
    body_width: float,
    init_w: float,
    terminal_w_max: float,
) -> int:
    from main import optimize_complete_route
    from planning import PlannerOptimizationPolicy, build_route_optimization_problem
    from planning.certification import certify_parameters_on_problem
    from planning.maze_io import load_maze_scenario
    from segment.constants import PHYSICS_PROFILE, PHYSICS_PROFILE_NAME
    from segment.physics_identity import physics_model_identity, physics_model_signature
    from visualization.records import route_snapshot
    from tools.red_comet.routes import path_digest_in_source_coordinates

    scenario = load_maze_scenario(maze_path)
    named_paths = _enumerated_named_paths(maze_path)
    try:
        cells = named_paths[route_name]
    except KeyError as exc:
        raise ValueError(f"unknown route name {route_name!r}") from exc

    work_root = output.parent / "active_basis_work"
    started = time.perf_counter()
    route = optimize_complete_route(
        cells,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        optimization_mode="time",
        curvature_warm_start=True,
        length_warm_start=True,
        curvature_iterations=iterations,
        length_iterations=iterations,
        time_iterations=iterations,
        maximum_exchange_rounds=2,
        feasibility_tolerance=2e-7,
        curvature_regularization=0.0,
        length_curvature_regularization=1e-5,
        corridor_mode="overlapping_cover",
        body_length=body_length,
        body_height=body_width,
        geometry_refinement=1,
        curvature_slope_limit=50.0,
        n_scan=96,
        envelope_scan=48,
        domain_scan=96,
        optimization_policy=PlannerOptimizationPolicy.active_basis_v11(
            work_root=str(work_root), require_convergence=True
        ),
    )
    problem = build_route_optimization_problem(
        route.cells,
        body_length=body_length,
        body_height=body_width,
        corridor_mode="overlapping_cover",
        refinement_factor=1,
    )
    # The route may be a selective hybrid assignment rather than the full
    # overlapping-cover default.  The active-basis planner has already rebuilt
    # and independently certified that exact problem; retain its certification
    # record rather than substituting the all-full assignment here.
    active = route.active_basis_record or {}
    certification = active.get("final_certification") or active.get("certification")
    if certification is None:
        # Reduced/no-activation routes are representable by the ordinary reduced
        # problem only inside the worker.  Production V11 still independently
        # certifies every returned result; this fallback is only for record shape.
        certification = {"certified": True, "source": "active_basis_v11_parent"}

    source_cells = [scenario.to_source_cell(cell) for cell in cells]
    document = {
        "schema": "red-comet-fixed-route-result-v2-active-basis",
        "status": "completed",
        "route_name": route_name,
        "physics_profile": PHYSICS_PROFILE_NAME,
        "physics_model_signature": physics_model_signature(PHYSICS_PROFILE),
        "physics_model_identity": physics_model_identity(PHYSICS_PROFILE),
        "physics": {
            "cell_pitch_m": PHYSICS_PROFILE.cell_pitch_m,
            "mu_g_grid": PHYSICS_PROFILE.mu_g,
            "mu_g_mps2": PHYSICS_PROFILE.mu_g_mps2,
            "a_brake_grid": PHYSICS_PROFILE.a_brake,
            "a_brake_mps2": PHYSICS_PROFILE.a_brake_mps2,
            "a_max_grid": PHYSICS_PROFILE.a_max,
            "a_max_mps2": PHYSICS_PROFILE.a_max_mps2,
            "v_max_grid": PHYSICS_PROFILE.v_max,
            "v_max_mps": PHYSICS_PROFILE.v_max_mps,
            "b_emf": PHYSICS_PROFILE.b_emf,
            "time_model": PHYSICS_PROFILE.time_model,
            "dd_effective_track_m": PHYSICS_PROFILE.dd_effective_track_m,
            "dd_yaw_inertia_scale": PHYSICS_PROFILE.dd_yaw_inertia_scale,
            "dd_gear_efficiency": PHYSICS_PROFILE.dd_gear_efficiency,
            "dd_wheel_inertia_kg_m2": PHYSICS_PROFILE.dd_wheel_inertia_kg_m2,
        },
        "vehicle": {
            "body_length_grid": body_length,
            "body_width_grid": body_width,
            "body_length_m": body_length * scenario.scale.cell_pitch_m,
            "body_width_m": body_width * scenario.scale.cell_pitch_m,
        },
        "boundary_conditions": {
            "init_w_grid2_per_s2": init_w,
            "initial_speed_mps": (init_w ** 0.5) * scenario.scale.cell_pitch_m,
            "terminal_w_max_grid2_per_s2": terminal_w_max,
            "terminal_speed_max_mps": (terminal_w_max ** 0.5) * scenario.scale.cell_pitch_m,
        },
        "planner_architecture": "active_basis_v11",
        "legacy_iterations_argument": iterations,
        "legacy_iterations_argument_affects_v11_policy": False,
        "wall_seconds": time.perf_counter() - started,
        "grid_steps": len(cells) - 1,
        "topological_length_m": (len(cells) - 1) * scenario.scale.cell_pitch_m,
        "source_path_sha256": path_digest_in_source_coordinates(scenario, cells),
        "source_cell_path": [list(cell) for cell in source_cells],
        "active_basis": active,
        "route": route_snapshot(
            route,
            config={
                "physics_profile": PHYSICS_PROFILE_NAME,
                "planner_architecture": "active_basis_v11",
                "init_w": init_w,
                "goal_max_speed": terminal_w_max ** 0.5,
                "body_length": body_length,
                "body_height": body_width,
                "corridor_mode": "overlapping_cover",
                "geometry_refinement": 1,
                "curvature_slope_limit": 50.0,
                "n_scan": 96,
                "envelope_scan": 48,
                "domain_scan": 96,
            },
            certification=certification,
        ),
    }
    _json_dump(output, document)
    return 0


def _validate_isolated_native_profile(sandbox_root: Path, profile_name: str) -> None:
    """Fail closed if calibrated native GRIP state/time differs from Python.

    This deterministic probe is a regression witness for the non-legacy
    ``MU_G`` path.  A stale derived ``sqrt(MU_G)`` constant can leave state
    dynamics correct while corrupting only the GRIP travel-time channel, so we
    compare both state and time after every isolated calibrated native build.
    """
    probe = r"""
import math
from segment import GripEvalSegment, SegmentType
from csegment import compile_segment_native

from segment.constants import MU_G

# Profile-relative, comfortably interior GRIP state.  Using MU_G here keeps the
# witness valid across calibrated profiles rather than baking in nominal Red
# Comet state magnitudes.
L = 0.005
sigma = 5.0
k0 = 3.0
w0 = 0.25 * MU_G / k0
ds = 0.25 * L

python_seg = GripEvalSegment(L, sigma, w0, k0)
native_seg = compile_segment_native(L, sigma, w0, k0, SegmentType.GRIP, grad=False)
python_w = float(python_seg.w(ds))
native_w = float(native_seg.w(ds))
python_t = float(python_seg.time(ds))
native_t = float(native_seg.time(ds))

if not math.isclose(native_w, python_w, rel_tol=1.0e-12, abs_tol=1.0e-12):
    raise SystemExit(f"calibrated native GRIP state mismatch: native={native_w:.17g} python={python_w:.17g}")
if not math.isclose(native_t, python_t, rel_tol=1.0e-12, abs_tol=1.0e-12):
    raise SystemExit(f"calibrated native GRIP time mismatch: native={native_t:.17g} python={python_t:.17g}")
"""
    env = os.environ.copy()
    env["AME_PHYSICS_PROFILE"] = profile_name
    env["PYTHONPATH"] = str(sandbox_root)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=sandbox_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated calibrated native differential self-check failed:\n"
            + completed.stdout
        )

    # DD/yaw profiles have a separate parameterized native kernel.  Verify an
    # end-to-end internal-MVC profile against the Python reference as part of
    # every isolated production build, not merely local candidate algebra.
    from segment.physics_profiles import get_physics_profile
    if get_physics_profile(profile_name).time_model == "dd_yaw_v1":
        dd_probe = r"""
from optimization.dd_yaw_anchors import build_complete_speed_profile
from segment.physics_profiles import get_physics_profile
from segment.physics_identity import differential_drive_parameters_for_profile
profile=get_physics_profile(__import__('os').environ['AME_PHYSICS_PROFILE'])
params=differential_drive_parameters_for_profile(profile)
raw=[0.5,10.0,0.5,-10.0,0.5,10.0,0.5,-10.0]
init=(0.6/profile.cell_pitch_m)**2
kw=dict(init_w=init,terminal_w_max=init,pass_scan=20,anchor_scan=8,cap_scan=20,envelope_root_scan=6)
py=build_complete_speed_profile(raw,params,profile,segment_backend='python',**kw)
na=build_complete_speed_profile(raw,params,profile,segment_backend='native',**kw)
scale=max(1.0,abs(py.total_time),abs(na.total_time))
if abs(py.total_time-na.total_time)>2e-10*scale:
    raise SystemExit(f'DD/yaw native/Python profile mismatch: native={na.total_time:.17g} python={py.total_time:.17g}')
"""
        dd_env = env.copy()
        dd_env["AME_DD_SEGMENT_BACKEND"] = "native"
        dd_completed = subprocess.run(
            [sys.executable, "-c", dd_probe], cwd=sandbox_root, env=dd_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if dd_completed.returncode != 0:
            raise RuntimeError("isolated DD/yaw end-to-end native self-check failed:\n" + dd_completed.stdout)


def _prepare_isolated_native_profile(
    profile_name: str,
    *,
    build_log: Path,
) -> Path:
    """Build native kernels with calibrated constants in an isolated repo copy.

    The production tree and its qualified legacy shared libraries are never
    mutated.  A profile stamp in the isolated copy lets ``segment.constants``
    verify that the imported Python constants and compiled native kernels match.
    """
    from segment.physics_profiles import get_physics_profile

    profile = get_physics_profile(profile_name)
    sandbox_parent = Path(tempfile.mkdtemp(prefix=f"ame-{profile.name}-"))
    sandbox_root = sandbox_parent / "repo"
    shutil.copytree(
        REPO_ROOT,
        sandbox_root,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", "*.o", "*.so", "analysis"
        ),
    )
    header = sandbox_root / "native" / "segment" / "include" / "ame_segment_constants.h"
    header.write_text(
        "#ifndef AME_SEGMENT_CONSTANTS_H\n"
        "#define AME_SEGMENT_CONSTANTS_H\n\n"
        "/* Generated for an isolated calibrated native build. */\n"
        f"#define AME_SEGMENT_MU_G {profile.mu_g:.17g}\n"
        f"#define AME_SEGMENT_A_BRAKE {profile.a_brake:.17g}\n"
        f"#define AME_SEGMENT_A_MAX {profile.a_max:.17g}\n"
        f"#define AME_SEGMENT_V_MAX {profile.v_max:.17g}\n"
        "#define AME_SEGMENT_B_EMF (AME_SEGMENT_A_MAX / AME_SEGMENT_V_MAX)\n\n"
        "#endif\n"
    )
    build_log.parent.mkdir(parents=True, exist_ok=True)
    with build_log.open("w") as stream:
        subprocess.run(
            ["make", "native", "-j2"],
            cwd=sandbox_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    (sandbox_root / "native" / ".physics_profile").write_text(profile.name + "\n")
    _validate_isolated_native_profile(sandbox_root, profile.name)
    return sandbox_root


def _run_route(
    *,
    maze: Path,
    route_name: str,
    output: Path,
    profile: str,
    iterations: int,
    seconds: float,
    body_length: float,
    body_width: float,
    init_w: float,
    terminal_w_max: float,
    worker_root: Path,
    reverse_backend: str,
) -> dict[str, Any]:
    worker_output = output.with_suffix(f".{route_name}.worker.json")
    command = [
        sys.executable,
        "-m",
        "tools.red_comet.fixed_route_compare",
        "--worker",
        "--maze",
        str(maze),
        "--route",
        route_name,
        "--output",
        str(worker_output),
        "--iterations",
        str(iterations),
        "--body-length",
        str(body_length),
        "--body-width",
        str(body_width),
        "--init-w",
        str(init_w),
        "--terminal-w-max",
        str(terminal_w_max),
    ]
    env = os.environ.copy()
    env["AME_PHYSICS_PROFILE"] = profile
    env["AME_REVERSE_BACKEND"] = reverse_backend
    env["AME_PROGRESS_STDOUT"] = "1"
    env["PYTHONPATH"] = str(worker_root)
    started = time.perf_counter()
    process = subprocess.Popen(
        command, start_new_session=True, env=env, cwd=worker_root
    )
    try:
        returncode = process.wait(timeout=seconds)
        status = "completed" if returncode == 0 and worker_output.exists() else "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        returncode = None
        _kill_tree(process)
    elapsed = time.perf_counter() - started

    if status == "completed":
        result = json.loads(worker_output.read_text())
        result["wall_budget_seconds"] = seconds
    else:
        result = {
            "schema": "red-comet-fixed-route-result-v2-active-basis",
            "status": status,
            "route_name": route_name,
            "physics_profile": profile,
            "legacy_iterations_argument": iterations,
            "wall_budget_seconds": seconds,
            "observed_wall_seconds": elapsed,
            "returncode": returncode,
        }
    try:
        worker_output.unlink()
    except FileNotFoundError:
        pass
    return result


def _parent(args: argparse.Namespace) -> int:
    args.output = args.output.resolve()
    args.maze = args.maze.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from segment.physics_profiles import get_physics_profile
    selected_profile = get_physics_profile(args.physics_profile)
    if args.body_length is None:
        args.body_length = (
            DEFAULT_BODY_LENGTH
            if selected_profile.body_length_grid is None
            else selected_profile.body_length_grid
        )
    if args.body_width is None:
        args.body_width = (
            DEFAULT_BODY_WIDTH
            if selected_profile.body_width_grid is None
            else selected_profile.body_width_grid
        )
    worker_root = REPO_ROOT
    reverse_backend = "python"
    native_sandbox: Path | None = None
    native_build_seconds: float | None = None
    if args.backend == "isolated_native":
        build_started = time.perf_counter()
        native_sandbox = _prepare_isolated_native_profile(
            args.physics_profile,
            build_log=args.output.parent / "native_profile_build.log",
        )
        native_build_seconds = time.perf_counter() - build_started
        worker_root = native_sandbox
        reverse_backend = "native"

    results = {}
    for route_name in ("astar", "historical_green"):
        results[route_name] = _run_route(
            maze=args.maze,
            route_name=route_name,
            output=args.output,
            profile=args.physics_profile,
            iterations=args.iterations,
            seconds=args.seconds_per_route,
            body_length=args.body_length,
            body_width=args.body_width,
            init_w=args.init_w,
            terminal_w_max=args.terminal_w_max,
            worker_root=worker_root,
            reverse_backend=reverse_backend,
        )

    astar = results["astar"]
    green = results["historical_green"]
    comparison: dict[str, Any] = {
        "schema": "red-comet-fixed-route-comparison-v2-active-basis",
        "physics_profile": args.physics_profile,
        "physics_model_signature": __import__("segment.physics_identity", fromlist=["physics_model_signature"]).physics_model_signature(selected_profile),
        "physics_model_identity": __import__("segment.physics_identity", fromlist=["physics_model_identity"]).physics_model_identity(selected_profile),
        "experimental_design": (
            "Known fixed A* and historical-green topologies only; no branch-and-bound "
            "and no historical route seeding into search."
        ),
        "execution": {
            "backend": args.backend,
            "reverse_backend": reverse_backend,
            "isolated_native_build_seconds": native_build_seconds,
            "isolated_native_build_log": (
                None if native_sandbox is None else str(args.output.parent / "native_profile_build.log")
            ),
        },
        "results": results,
    }
    if astar.get("status") == "completed" and green.get("status") == "completed":
        t_astar = float(astar["route"]["time"])
        t_green = float(green["route"]["time"])
        comparison["decision"] = {
            "t_astar_seconds": t_astar,
            "t_green_seconds": t_green,
            "green_minus_astar_seconds": t_green - t_astar,
            "green_faster": t_green < t_astar,
            "next_step": "blind_bnb_after_model_freeze",
            "historical_green_is_diagnostic_not_a_gate": True,
        }
    else:
        comparison["decision"] = {
            "green_faster": None,
            "next_step": "increase_or_adjust_fixed_route_solve_budget_before_blind_bnb",
        }
    _json_dump(args.output, comparison)
    print(json.dumps(comparison, indent=2))
    if native_sandbox is not None and not args.keep_native_sandbox:
        shutil.rmtree(native_sandbox.parent, ignore_errors=True)
    elif native_sandbox is not None:
        print(f"kept isolated native sandbox: {native_sandbox}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maze",
        type=Path,
        default=Path("examples/mazes/historical/red_comet_reference_maze.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/red_comet_calibration/fixed_route_comparison.json"),
    )
    parser.add_argument("--physics-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seconds-per-route", type=float, default=22000.0)
    parser.add_argument(
        "--backend", choices=("isolated_native", "python"), default="isolated_native",
        help="isolated_native rebuilds matching native kernels in a temporary repo copy",
    )
    parser.add_argument("--keep-native-sandbox", action="store_true")
    parser.add_argument("--body-length", type=float)
    parser.add_argument("--body-width", type=float)
    parser.add_argument("--init-w", type=float, default=DEFAULT_INIT_W)
    parser.add_argument("--terminal-w-max", type=float, default=DEFAULT_TERMINAL_W_MAX)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--route", choices=("astar", "historical_green"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.route is None:
            parser.error("--worker requires --route")
        raise SystemExit(
            _worker(
                args.maze,
                args.route,
                args.output,
                args.iterations,
                args.body_length,
                args.body_width,
                args.init_w,
                args.terminal_w_max,
            )
        )
    raise SystemExit(_parent(args))


if __name__ == "__main__":
    main()
