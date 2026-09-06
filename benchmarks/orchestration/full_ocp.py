from __future__ import annotations

"""Clean stdlib-level orchestration for the canonical simultaneous full OCP.

No production optimizer or CasADi module is imported here. Every numerical
component receives a fresh interpreter and process group.
"""

import argparse
import json
from pathlib import Path

from benchmarks.common.io import read_json
from benchmarks.common.status import SUCCESS
from benchmarks.config import (
    FULL_OCP_CASE_NAME,
    FULL_OCP_MESHES,
    FULL_OCP_STRUCTURED_REFINEMENTS,
    SMOKE_FULL_OCP_CASE_NAME,
    SMOKE_FULL_OCP_MESHES,
)
from benchmarks.orchestration.process import run_python_module


def _select_case(topology: dict, name: str) -> dict:
    return next(c for c in topology["cases"] if c["case"]["name"] == name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated canonical Level-2 OCP orchestration.")
    parser.add_argument("--profile", choices=("core", "smoke"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology_path = args.output_dir / "topology_search.json"
    if not topology_path.exists():
        raise FileNotFoundError("full_ocp requires topology_search.json in the same output directory")
    topology = read_json(topology_path)
    case_name = FULL_OCP_CASE_NAME if args.profile == "core" else SMOKE_FULL_OCP_CASE_NAME
    case = _select_case(topology, case_name)
    cells_json = json.dumps(case["branch_and_bound"]["route"]["cells"], separators=(",", ":"))

    refinements = FULL_OCP_STRUCTURED_REFINEMENTS if args.profile == "core" else (1,)
    meshes = FULL_OCP_MESHES if args.profile == "core" else SMOKE_FULL_OCP_MESHES
    ref_timeout = 180.0 if args.profile == "core" else 40.0
    ocp_timeout = 150.0 if args.profile == "core" else 60.0

    # Structured references first; each cleans up normally in its own process group.
    for refinement in refinements:
        done = args.output_dir / ".components" / f"full_ocp_ref{refinement}.done"
        component_path = args.output_dir / f"full_ocp_structured_ref{refinement}.json"
        run_python_module(
            "benchmarks.full_ocp_reference",
            [
                "--profile", args.profile,
                "--refinement", str(refinement),
                "--cells-json", cells_json,
                "--output", str(component_path),
                "--done", str(done),
            ],
            cwd=root,
            timeout=ref_timeout,
            completion_marker=done,
        )
        component = read_json(component_path)
        if component.get("execution_status", SUCCESS) != SUCCESS:
            raise RuntimeError(
                f"structured reference refinement {refinement} failed: "
                f"{component.get('execution_status')}: {component.get('error')}"
            )

    # Each mesh is a separate CasADi/IPOPT interpreter: no plugin teardown is
    # shared with another mesh or with the production warm-start supervisor.
    for mesh in meshes:
        done = args.output_dir / ".components" / f"full_ocp_ocp{mesh}.done"
        component_path = args.output_dir / f"full_ocp_ocp_{mesh}.json"
        run_python_module(
            "benchmarks.full_ocp_generic",
            [
                "--profile", args.profile,
                "--case-name", case_name,
                "--mesh", str(mesh),
                "--topology-path", str(topology_path),
                "--output", str(component_path),
                "--done", str(done),
            ],
            cwd=root,
            timeout=ocp_timeout,
            completion_marker=done,
        )
        component = read_json(component_path)
        if component.get("execution_status", SUCCESS) != SUCCESS:
            raise RuntimeError(
                f"generic OCP mesh {mesh} failed: "
                f"{component.get('execution_status')}: {component.get('error')}"
            )

    done = args.output_dir / ".components" / "full_ocp_assemble.done"
    run_python_module(
        "benchmarks.full_ocp_assemble",
        ["--profile", args.profile, "--output-dir", str(args.output_dir), "--done", str(done)],
        cwd=root,
        timeout=30.0,
        completion_marker=done,
    )

    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")



if __name__ == "__main__":
    main()
