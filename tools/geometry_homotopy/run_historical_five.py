#!/usr/bin/env python3
"""CLI for the strictly serial historical-five convergence campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.geometry_homotopy.serial_runtime import force_single_thread_environment
force_single_thread_environment()

from tools.geometry_homotopy.state_machine import HomotopyPolicy, run_historical_five


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--route-wall", type=float, default=1800.0)
    parser.add_argument("--basis-wall", type=float, default=600.0)
    args = parser.parse_args()
    policy = HomotopyPolicy(
        maximum_route_wall_seconds=float(args.route_wall),
        basis_branch_worker_timeout_seconds=float(args.basis_wall),
    )
    payload = run_historical_five(args.output, seeds=args.seeds, policy=policy)
    print(json.dumps({
        "complete": payload["complete"],
        "execution_mode": payload["execution_mode"],
        "campaign_wall_seconds": payload["campaign_wall_seconds"],
        "routes": [
            {
                "seed": row["seed"],
                "status": row["status"],
                "converged": row["converged"],
                "reduced_converged": row["reduced_converged"],
                "basis_branch_converged": row["basis_branch_converged"],
                "basis_branch_attempts": row["basis_branch_attempts"],
                "initial_time": row["initial_time"],
                "final_time": row["final_time"],
                "final_basis_kind": row["final_basis_kind"],
                "final_segments": row["final_segments"],
                "wall_seconds": row["total_wall_seconds"],
                "error": row["error"],
            }
            for row in payload["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
