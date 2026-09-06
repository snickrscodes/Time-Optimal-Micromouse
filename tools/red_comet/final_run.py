"""Resumable held-out Red Comet campaign for active-basis V11.

The campaign is deliberately serial:

1. build matching calibrated native kernels in an isolated temporary worktree;
2. solve A* and the verified historical-green fixed topologies with the exact
   qualified V11 optimizer and one shared persistent active-basis cache;
3. record A* versus historical-green as a diagnostic under the independently frozen model;
4. optionally stop after the fixed-route diagnostic, otherwise launch blind B&B with no historical route seeded into search.

The expensive route cache lives in ``OUTPUT/active_basis_work`` outside the
isolated native copy. Restarting this command therefore reuses every completed
fixed topology by request digest; only the cheap discrete search is replayed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

DEFAULT_PROFILE = "red_comet_2017_dd_yaw_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", type=Path, default=Path("examples/mazes/historical/red_comet_reference_maze.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/red_comet_final"))
    parser.add_argument("--physics-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--init-w", type=float, default=0.8)
    parser.add_argument("--terminal-w-max", type=float, default=0.8)
    parser.add_argument("--seconds-per-fixed-route", type=float, default=22000.0)
    parser.add_argument("--maximum-expansions", type=int, default=20_000)
    parser.add_argument("--fixed-only", action="store_true", help="solve/certify A* and historical green, write visual-ready records, then stop before B&B")
    parser.add_argument("--bnb-only", action="store_true", help="reuse an existing certified fixed_route_comparison.json with the same physics signature and launch only blind B&B")
    parser.add_argument("--force-bnb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-native-sandbox", action="store_true")
    args = parser.parse_args()

    args.maze = args.maze.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parent intentionally imports only profile metadata and supervisor helpers;
    # it never binds the calibrated segment constants itself.
    from segment.physics_profiles import get_physics_profile
    from tools.red_comet.fixed_route_compare import _prepare_isolated_native_profile, _run_route

    profile = get_physics_profile(args.physics_profile)
    from segment.physics_identity import physics_model_identity, physics_model_signature
    model_identity = physics_model_identity(profile)
    model_signature = physics_model_signature(profile)
    body_length = profile.body_length_grid or 5.0 / 9.0
    body_width = profile.body_width_grid or 4.0 / 9.0
    build_log = args.output_dir / "native_profile_build.log"
    build_started = time.perf_counter()
    sandbox = _prepare_isolated_native_profile(args.physics_profile, build_log=build_log)
    build_seconds = time.perf_counter() - build_started

    env = os.environ.copy()
    env["AME_PHYSICS_PROFILE"] = args.physics_profile
    env["AME_REVERSE_BACKEND"] = "native"
    env["AME_PROGRESS_STDOUT"] = "1"
    env["PYTHONPATH"] = str(sandbox)
    env["PYTHONHASHSEED"] = "0"
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        env[name] = "1"

    manifest: dict[str, Any] = {
        "schema": "red-comet-heldout-active-basis-v11-dd-yaw-run-v3-bnb-recovery",
        "planner_recovery_revision": "bnb_cache_boundary_and_basis_resume_v1",
        "physics_profile": args.physics_profile,
        "physics_model_signature": model_signature,
        "physics_model_identity": model_identity,
        "body_length_grid": body_length,
        "body_width_grid": body_width,
        "init_w": args.init_w,
        "terminal_w_max": args.terminal_w_max,
        "native_build_seconds": build_seconds,
        "native_build_log": str(build_log),
        "active_basis_work": str(args.output_dir / "active_basis_work"),
        "fixed_gate": {},
        "bnb": None,
    }
    _dump(args.output_dir / "run_manifest.json", manifest)

    try:
        fixed: dict[str, Any] = {}
        if args.bnb_only:
            fixed_path = args.output_dir / "fixed_route_comparison.json"
            if not fixed_path.exists():
                raise RuntimeError("--bnb-only requires an existing fixed_route_comparison.json")
            fixed_document = json.loads(fixed_path.read_text(encoding="utf-8"))
            if fixed_document.get("physics_model_signature") != model_signature:
                raise RuntimeError("--bnb-only fixed-route physics signature does not match current production model")
            decision = dict(fixed_document.get("decision") or {})
            if not bool(decision.get("bnb_ready", False)):
                raise RuntimeError("--bnb-only requires fixed_route_comparison.json with bnb_ready=true")
            fixed = dict(fixed_document.get("results") or {})
            if set(fixed) < {"astar", "historical_green"}:
                raise RuntimeError("--bnb-only fixed-route document is missing astar/historical_green results")
            manifest["fixed_gate"] = fixed
            manifest["fixed_gate_decision"] = decision
            manifest["fixed_gate_reused"] = True
            _dump(args.output_dir / "run_manifest.json", manifest)
        else:
            for route_name in ("astar", "historical_green"):
                print(f"[fixed gate] {route_name}", flush=True)
                fixed[route_name] = _run_route(
                    maze=args.maze,
                    route_name=route_name,
                    output=args.output_dir / "fixed_route_comparison.json",
                    profile=args.physics_profile,
                    iterations=0,
                    seconds=args.seconds_per_fixed_route,
                    body_length=body_length,
                    body_width=body_width,
                    init_w=args.init_w,
                    terminal_w_max=args.terminal_w_max,
                    worker_root=sandbox,
                    reverse_backend="native",
                )
                manifest["fixed_gate"] = fixed
                _dump(args.output_dir / "run_manifest.json", manifest)

        astar = fixed["astar"]
        green = fixed["historical_green"]
        decision: dict[str, Any]
        if astar.get("status") == "completed" and green.get("status") == "completed":
            ta = float(astar["route"]["time"])
            tg = float(green["route"]["time"])
            fixed_certified = all(
                bool((fixed[name].get("active_basis") or {}).get("parent_physics_certification", {}).get("certified", False))
                for name in ("astar", "historical_green")
            )
            decision = {
                "t_astar_seconds": ta,
                "t_historical_green_seconds": tg,
                "green_minus_astar_seconds": tg - ta,
                "green_faster": tg < ta,
                "fixed_routes_physics_certified": fixed_certified,
                "bnb_ready": fixed_certified,
                "historical_green_is_diagnostic_not_a_gate": True,
            }
        else:
            decision = {"green_faster": None, "fixed_routes_physics_certified": False, "bnb_ready": False}
        fixed_document = {
            "schema": "red-comet-fixed-route-comparison-v2-active-basis",
            "physics_profile": args.physics_profile,
            "physics_model_signature": model_signature,
            "physics_model_identity": model_identity,
            "planner_architecture": "active_basis_v11",
            "execution": {
                "backend": "isolated_native",
                "reverse_backend": "native",
                "isolated_native_build_seconds": build_seconds,
                "shared_active_basis_cache": str(args.output_dir / "active_basis_work"),
            },
            "results": fixed,
            "decision": decision,
        }
        _dump(args.output_dir / "fixed_route_comparison.json", fixed_document)
        manifest["fixed_gate_decision"] = decision
        _dump(args.output_dir / "run_manifest.json", manifest)

        if not decision.get("bnb_ready"):
            manifest["status"] = "fixed_route_certification_failed"
            manifest["next_step"] = "do not launch B&B until both fixed routes pass independent DD/yaw physics certification"
            _dump(args.output_dir / "run_manifest.json", manifest)
            print("fixed-route DD/yaw certification did not pass; blind B&B not launched", flush=True)
            return 2
        if args.fixed_only:
            manifest["status"] = "fixed_routes_complete"
            manifest["next_step"] = "blind B&B may now be launched with the same frozen physics signature"
            _dump(args.output_dir / "run_manifest.json", manifest)
            print("fixed-route DD/yaw diagnostic complete; --fixed-only requested", flush=True)
            return 0

        # Blind B&B receives only the production A* seed. The historical route
        # is *not* passed to the search; if encountered, its expensive fixed
        # solve is simply reused by the content-addressed V11 route cache.
        planner_png = args.output_dir / "red_comet_astar_vs_bnb.png"
        planner_json = args.output_dir / "red_comet_astar_vs_bnb.json"
        search_trace = args.output_dir / "red_comet_search_trace.json"
        command = [
            sys.executable, str(sandbox / "main.py"),
            "--maze-file", str(args.maze),
            "--planner-architecture", "active_basis_v11",
            "--active-basis-work-dir", str(args.output_dir / "active_basis_work"),
            "--no-topology-quotient",
            "--body-length", repr(body_length),
            "--body-height", repr(body_width),
            "--init-w", repr(args.init_w),
            "--goal-max-speed", repr(math.sqrt(args.terminal_w_max)),
            "--geometry-refinement", "1",
            "--n-scan", "96", "--envelope-scan", "48", "--domain-scan", "96",
            "--maximum-node-visits", "1",
            "--maximum-expansions", str(args.maximum_expansions),
            "--output", str(planner_png),
            "--metadata-output", str(planner_json),
            "--search-trace-output", str(search_trace),
        ]
        print("[blind B&B] launching production search", flush=True)
        bnb_started = time.perf_counter()
        _run_checked(command, cwd=sandbox, env=env, log=args.output_dir / "bnb_stdout_stderr.log")
        bnb_seconds = time.perf_counter() - bnb_started
        manifest["bnb"] = {
            "status": "completed",
            "wall_seconds": bnb_seconds,
            "metadata": str(planner_json),
            "figure": str(planner_png),
            "search_trace": str(search_trace),
            "log": str(args.output_dir / "bnb_stdout_stderr.log"),
        }
        manifest["status"] = "complete"
        _dump(args.output_dir / "run_manifest.json", manifest)
        return 0
    finally:
        if args.keep_native_sandbox:
            print(f"kept isolated native sandbox: {sandbox}", flush=True)
        else:
            shutil.rmtree(sandbox.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
