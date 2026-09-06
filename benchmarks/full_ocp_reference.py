from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path

from .common.certification import certify_optimized_route as independent_certify
from .common.io import write_json
from .common.status import SUCCESS, classify_exception
from .common.routes import RouteEvaluator, route_to_record
from .config import FULL_OCP_OPTIMIZATION, SMOKE_OPTIMIZATION, OptimizationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated structured fixed-topology production reference worker.")
    parser.add_argument("--profile", choices=("core", "smoke"), required=True)
    parser.add_argument("--refinement", type=int, required=True)
    parser.add_argument("--cells-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()

    base = FULL_OCP_OPTIMIZATION if args.profile == "core" else SMOKE_OPTIMIZATION
    cfg = OptimizationConfig(**{**base.to_dict(), "geometry_refinement": int(args.refinement)})
    cells = tuple(tuple(int(v) for v in cell) for cell in json.loads(args.cells_json))
    started = time.perf_counter()
    try:
        with RouteEvaluator(cfg) as evaluator:
            route = evaluator.optimize(cells)
        wall = float(time.perf_counter() - started)
        cert = independent_certify(route, cfg)
        payload = {
            "component": "structured_fixed_topology_reference",
            "execution_status": SUCCESS,
            "geometry_refinement": int(args.refinement),
            "segments": int(len(route.parameters) // 2),
            "geometry_variables": int(len(route.parameters)),
            "intrinsic_geometry_dof": int(len(route.parameters)),
            "speed_profile_decision_variables": 0,
            "time": float(route.time),
            "wall_seconds": wall,
            "certification": cert,
            "selected_stage": route.selected_stage,
            "route": route_to_record(route, cfg),
        }
    except Exception as exc:
        payload = {
            "component": "structured_fixed_topology_reference",
            "execution_status": classify_exception(exc),
            "geometry_refinement": int(args.refinement),
            "wall_seconds": float(time.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(args.output, payload)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")
    sys.stdout.flush(); sys.stderr.flush()


if __name__ == "__main__":
    main()
