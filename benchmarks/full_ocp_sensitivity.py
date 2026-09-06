from __future__ import annotations

from . import SCHEMA_VERSION

from pathlib import Path
from typing import Any

from .common.certification import certified_only
from .common.environment import environment_metadata
from .common.io import read_json, write_json
from .common.status import CERTIFICATION_FAILURE, SUCCESS
from .config import FULL_OCP_OPTIMIZATION, FULL_OCP_SENSITIVITY_CASES, SMOKE_FULL_OCP_SENSITIVITY_CASES


def assemble(output_dir: Path, *, profile: str, status_path: Path) -> dict[str, Any]:
    topology = read_json(output_dir / "topology_search.json")
    status = read_json(status_path)
    definitions = FULL_OCP_SENSITIVITY_CASES if profile == "core" else SMOKE_FULL_OCP_SENSITIVITY_CASES
    cases: list[dict[str, Any]] = []

    for definition in definitions:
        topology_case = next(c for c in topology["cases"] if c["case"]["name"] == definition.name)
        baseline = topology_case["branch_and_bound"]
        baseline_route = baseline["route"]
        baseline_cert = baseline["certification"]
        state = status["cases"][definition.name]
        structured = None
        if state["structured"]["status"] == "success":
            structured_path = Path(state["structured"]["output"])
            if not structured_path.is_absolute():
                structured_path = output_dir / structured_path
            structured = read_json(structured_path)

        rows = []
        for mesh_state in state["ocp"]:
            if mesh_state["status"] != "success":
                rows.append({
                    "base_intervals": mesh_state["mesh"],
                    "execution_status": mesh_state["status"],
                    "error": mesh_state.get("error"),
                })
                continue
            component_path = Path(mesh_state["output"])
            if not component_path.is_absolute():
                component_path = output_dir / component_path
            component = read_json(component_path)
            row = component["row"]
            row["execution_status"] = (SUCCESS if row["certificate"]["certified"] else CERTIFICATION_FAILURE)
            baseline_time = float(baseline_route["time"])
            row["comparison_to_existing_production"] = {
                "production_time": baseline_time,
                "ocp_difference_percent": 100.0 * (float(row["objective_time"]) - baseline_time) / baseline_time,
                "hybrid_on_ocp_geometry_difference_percent": None if row.get("continuous_hybrid_time_on_ocp_geometry") is None else 100.0 * (float(row["continuous_hybrid_time_on_ocp_geometry"]) - baseline_time) / baseline_time,
            }
            if structured is not None and structured["certification"]["certified"]:
                structured_time = float(structured["time"])
                row["comparison_to_structured_control"] = {
                    "structured_segments": int(structured["segments"]),
                    "structured_time": structured_time,
                    "ocp_difference_percent": 100.0 * (float(row["objective_time"]) - structured_time) / structured_time,
                    "hybrid_on_ocp_geometry_difference_percent": None if row.get("continuous_hybrid_time_on_ocp_geometry") is None else 100.0 * (float(row["continuous_hybrid_time_on_ocp_geometry"]) - structured_time) / structured_time,
                }
            rows.append(row)

        cases.append({
            "case": definition.name,
            "note": definition.note,
            "cells": baseline_route["cells"],
            "existing_production": {
                "time": float(baseline_route["time"]),
                "certification": baseline_cert,
                "segments": len(baseline_route["raw_parameters"]) // 2,
            },
            "structured_control": structured,
            "structured_execution": state["structured"],
            "rows": rows,
        })

    successful_rows = [r for c in cases for r in c["rows"] if r.get("execution_status") == "success"]
    certified_rows = certified_only(successful_rows, key="certificate")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "full_ocp_sensitivity",
        "profile": profile,
        "environment": environment_metadata(FULL_OCP_OPTIMIZATION),
        "methodology": {
            "purpose": "supplemental cross-topology validation of the canonical simultaneous full-OCP finding",
            "selection": "predeclared deterministic topology cases; failed/timeout structured controls remain explicit",
            "quality_rule": "only independently certified OCP reconstructions enter quality counts",
        },
        "cases": cases,
        "aggregate": {
            "cases": len(cases),
            "ocp_attempts": len(successful_rows) + sum(1 for c in cases for r in c["rows"] if r.get("execution_status") != "success"),
            "ocp_completed": len(successful_rows),
            "ocp_certified": len(certified_rows),
            "structured_controls_certified": sum(bool(c["structured_control"] and c["structured_control"]["certification"]["certified"]) for c in cases),
            "structured_controls_not_completed": sum(c["structured_execution"]["status"] != "success" for c in cases),
            "certified_ocp_geometries_with_hybrid_time": sum(r.get("continuous_hybrid_time_on_ocp_geometry") is not None for r in certified_rows),
        },
    }
    write_json(output_dir / "full_ocp_sensitivity.json", result)
    return result
