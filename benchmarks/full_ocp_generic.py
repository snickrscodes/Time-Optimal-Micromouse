from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planning import build_route_optimization_problem

from .common.io import read_json, write_json
from .common.status import SUCCESS, classify_exception
from .config import FULL_OCP_OPTIMIZATION, SMOKE_OPTIMIZATION
from .full_ocp import solve_full_ocp


def _select_case(topology: dict, name: str) -> dict:
    for case in topology["cases"]:
        if case["case"]["name"] == name:
            return case
    raise RuntimeError(f"predeclared OCP case missing from topology results: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated single-mesh CasADi/IPOPT full-OCP worker.")
    parser.add_argument("--profile", choices=("core", "smoke"), required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--mesh", type=int, required=True)
    parser.add_argument("--topology-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()

    topology = read_json(args.topology_path)
    case = _select_case(topology, args.case_name)
    route = case["branch_and_bound"]["route"]
    config = FULL_OCP_OPTIMIZATION if args.profile == "core" else SMOKE_OPTIMIZATION
    cells = tuple(tuple(c) for c in route["cells"])
    problem = build_route_optimization_problem(
        cells,
        body_length=config.body_length,
        body_height=config.body_height,
        corridor_mode=config.corridor_mode,
        refinement_factor=config.geometry_refinement,
    )
    try:
        row = solve_full_ocp(
            problem,
            route,
            base_intervals=int(args.mesh),
            initialization="cold",
            config=config,
        )
        payload = {
            "component": "generic_full_ocp",
            "execution_status": SUCCESS,
            "case": args.case_name,
            "cells": list(cells),
            "source_time": float(route["time"]),
            "row": row,
        }
    except Exception as exc:
        payload = {
            "component": "generic_full_ocp",
            "execution_status": classify_exception(exc),
            "case": args.case_name,
            "cells": list(cells),
            "source_time": float(route["time"]),
            "mesh": int(args.mesh),
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(args.output, payload)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")
    sys.stdout.flush(); sys.stderr.flush()


if __name__ == "__main__":
    main()
