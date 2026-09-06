from __future__ import annotations

"""Stdlib-oriented orchestrator for supplemental cross-topology OCP checks."""

import argparse
import json
from pathlib import Path

from benchmarks.common.io import read_json, write_json
from benchmarks.common.status import NUMERICAL_FAILURE, SUCCESS, TIMEOUT
from benchmarks.config import FULL_OCP_SENSITIVITY_CASES, SMOKE_FULL_OCP_SENSITIVITY_CASES
from benchmarks.orchestration.process import (
    BenchmarkProcessError,
    BenchmarkTimeoutError,
    run_python_module,
)


def _component_status(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return NUMERICAL_FAILURE, "component exited without writing its result"
    payload = read_json(path)
    return str(payload.get("execution_status", SUCCESS)), payload.get("error")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplemental multi-topology full-OCP orchestration.")
    parser.add_argument("--profile", choices=("core", "smoke"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    topology_path = args.output_dir / "topology_search.json"
    topology = read_json(topology_path)
    definitions = FULL_OCP_SENSITIVITY_CASES if args.profile == "core" else SMOKE_FULL_OCP_SENSITIVITY_CASES
    status: dict = {"cases": {}}

    for definition in definitions:
        case = next(c for c in topology["cases"] if c["case"]["name"] == definition.name)
        cells_json = json.dumps(case["branch_and_bound"]["route"]["cells"], separators=(",", ":"))
        case_status = {"structured": {}, "ocp": []}
        if definition.structured_refinement is None:
            case_status["structured"] = {"status": "not_requested"}
        else:
            output = args.output_dir / f"full_ocp_sensitivity_{definition.name}_structured_ref{definition.structured_refinement}.json"
            try:
                done = args.output_dir / ".components" / f"sensitivity_{definition.name}_ref.done"
                run_python_module(
                    "benchmarks.full_ocp_reference",
                    [
                        "--profile", args.profile,
                        "--refinement", str(definition.structured_refinement),
                        "--cells-json", cells_json,
                        "--output", str(output),
                        "--done", str(done),
                    ],
                    cwd=root,
                    timeout=90.0 if args.profile == "core" else 35.0,
                    completion_marker=done,
                )
                component_status, error = _component_status(output)
                case_status["structured"] = {"status": component_status, "error": error, "output": output.name}
            except BenchmarkTimeoutError as exc:
                case_status["structured"] = {"status": TIMEOUT, "error": str(exc), "output": output.name}
            except BenchmarkProcessError as exc:
                case_status["structured"] = {"status": NUMERICAL_FAILURE, "error": str(exc), "output": output.name}

        for mesh in definition.ocp_meshes:
            output = args.output_dir / f"full_ocp_sensitivity_{definition.name}_ocp{mesh}.json"
            try:
                done = args.output_dir / ".components" / f"sensitivity_{definition.name}_ocp{mesh}.done"
                run_python_module(
                    "benchmarks.full_ocp_generic",
                    [
                        "--profile", args.profile,
                        "--case-name", definition.name,
                        "--mesh", str(mesh),
                        "--topology-path", str(topology_path),
                        "--output", str(output),
                        "--done", str(done),
                    ],
                    cwd=root,
                    timeout=150.0 if args.profile == "core" else 60.0,
                    completion_marker=done,
                )
                component_status, error = _component_status(output)
                case_status["ocp"].append({"mesh": mesh, "status": component_status, "error": error, "output": output.name})
            except BenchmarkTimeoutError as exc:
                case_status["ocp"].append({"mesh": mesh, "status": TIMEOUT, "error": str(exc), "output": output.name})
            except BenchmarkProcessError as exc:
                case_status["ocp"].append({"mesh": mesh, "status": NUMERICAL_FAILURE, "error": str(exc), "output": output.name})
        status["cases"][definition.name] = case_status

    status_path = args.output_dir / "full_ocp_sensitivity_components.json"
    write_json(status_path, status)
    done = args.output_dir / ".components" / "full_ocp_sensitivity_assemble.done"
    run_python_module(
        "benchmarks.full_ocp_sensitivity_assemble",
        ["--profile", args.profile, "--output-dir", str(args.output_dir), "--status", str(status_path), "--done", str(done)],
        cwd=root,
        timeout=30.0,
        completion_marker=done,
    )

    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text("complete\n", encoding="utf-8")



if __name__ == "__main__":
    main()
