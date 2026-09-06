from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import SCHEMA_VERSION
from benchmarks.common.routes import build_case
from benchmarks.common.runtime import reverse_backend as benchmark_reverse_backend
from benchmarks.common.environment import environment_metadata
from benchmarks.common.provenance import provenance_metadata, source_tree_sha256
from benchmarks.config import SMOKE_TOPOLOGY_CASES
from benchmarks.direct_transcription import certify_piecewise_linear_profile, geometry_from_raw
from benchmarks.full_ocp import Phase, _allocate_intervals, aggregate as full_ocp_aggregate
from benchmarks.gradients import five_point_derivative
from benchmarks.lower_bounds import check_admissibility
from benchmarks.run import _resolve_suites
from benchmarks.schema import ResultSchemaError, validate_benchmark_result, validate_manifest
from benchmarks.topology_search import aggregate as topology_aggregate
from optimization import reverse_solver
from planning import astar_junction_path, expand_junction_path


def test_admissibility_violation_fails_loudly():
    with pytest.raises(AssertionError):
        check_admissibility(1.000001, 1.0, tolerance=1e-9)


def test_reverse_backend_context_restores_previous():
    original = reverse_solver.reverse_backend()
    other = "python" if original == "native" else "native"
    with benchmark_reverse_backend(other):
        assert reverse_solver.reverse_backend() == other
    assert reverse_solver.reverse_backend() == original


def test_five_point_helper_on_analytic_function():
    x = np.array([0.7, -1.2])
    fn = lambda z: float(z[0] ** 3 + 2.0 * z[1] ** 2)
    derivative = five_point_derivative(fn, x, 0, 1e-4)
    assert derivative == pytest.approx(3.0 * x[0] ** 2, rel=1e-9, abs=1e-10)


def test_deterministic_maze_reproduces_graph_and_astar_topology():
    case = SMOKE_TOPOLOGY_CASES[0]
    m1, g1 = build_case(case)
    m2, g2 = build_case(case)
    p1 = tuple(astar_junction_path(g1, (0, 0), m1.goal))
    p2 = tuple(astar_junction_path(g2, (0, 0), m2.goal))
    assert m1.connection_dict() == m2.connection_dict()
    assert g1.nodes == g2.nodes
    assert p1 == p2
    assert tuple(expand_junction_path(m1, g1, p1)) == tuple(expand_junction_path(m2, g2, p2))


def _synthetic_case(astar_certified: bool, bb_certified: bool, improvement: float):
    return {
        "astar": {"certification": {"certified": astar_certified}},
        "branch_and_bound": {
            "certification": {"certified": bb_certified},
            "complete_route_optimizations_including_seed": 1,
            "search_wall_seconds": 1.0,
        },
        "comparison": {
            "percentage_time_improvement": improvement,
            "absolute_time_improvement_seconds": improvement / 100.0,
            "topology_changed": True,
            "selected_topology_geometrically_longer": False,
        },
    }


def test_quality_aggregate_uses_only_doubly_certified_routes():
    result = topology_aggregate([
        _synthetic_case(True, True, 2.0),
        _synthetic_case(False, True, 99.0),
        _synthetic_case(True, False, 88.0),
    ])
    assert result["cases_in_quality_aggregate"] == 1
    assert result["time_improvement_percent"]["max"] == 2.0


def _minimal_environment() -> dict:
    return {"provenance": {"git_commit": "abc", "source_tree_sha256": None}}


def test_result_schema_accepts_valid_and_rejects_missing_provenance():
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "x",
        "status": "complete",
        "profile": "smoke",
        "environment": _minimal_environment(),
        "aggregate": {},
    }
    validate_benchmark_result(payload, expected_benchmark="x", expected_profile="smoke")
    payload["environment"] = {}
    with pytest.raises(ResultSchemaError):
        validate_benchmark_result(payload)


def test_direct_transcription_continuous_certificate_detects_interior_friction_peak():
    geometry = geometry_from_raw([1.0, 2.0], initial_k=-1.0)
    stations = np.array([0.0, 1.0])
    result = certify_piecewise_linear_profile(
        geometry, stations, np.array([0.8, 20.0]), init_w=0.8, tolerance=2e-7
    )
    assert not result["certified"]
    assert result["maximum_residual"] > 0.0


def test_full_ocp_mesh_allocation_is_deterministic_and_respects_phase_minimums():
    phases = (
        Phase(0, 0, 2, 1.0),
        Phase(1, 2, 4, 0.2),
        Phase(2, 4, 7, 1.8),
    )
    first = _allocate_intervals(phases, 24)
    second = _allocate_intervals(phases, 24)
    assert first == second
    assert sum(first) == 24
    assert all(count >= 2 for count in first)


def test_full_ocp_aggregate_excludes_uncertified_solution_quality():
    rows = [
        {
            "base_intervals": 24,
            "initialization": "cold",
            "objective_time": 1.2,
            "ipopt_solve_seconds": 2.0,
            "certificate": {"certified": True, "endpoint_error": 1e-10, "corridor_upper_bound": 1e-10},
        },
        {
            "base_intervals": 48,
            "initialization": "cold",
            "objective_time": 0.1,
            "ipopt_solve_seconds": 3.0,
            "certificate": {"certified": False, "endpoint_error": 1.0, "corridor_upper_bound": 1.0},
        },
    ]
    result = full_ocp_aggregate(rows)
    assert result["certified_solves"] == 1
    assert result["best_certified_ocp_time"] == pytest.approx(1.2)


def test_profile_dependency_resolution_reuses_existing_topology():
    assert _resolve_suites("transcription", None, topology_available=False) == (
        "topology_search", "direct_transcription"
    )
    assert _resolve_suites("transcription", None, topology_available=True) == (
        "direct_transcription",
    )


def test_source_tree_hash_ignores_generated_benchmark_results(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)
    generated = tmp_path / "benchmarks" / "results" / "run.json"
    generated.write_text("one")
    first = source_tree_sha256(tmp_path)
    generated.write_text("two")
    assert source_tree_sha256(tmp_path) == first
    (tmp_path / "a.py").write_text("x = 2\n")
    assert source_tree_sha256(tmp_path) != first


def test_git_provenance_preferred_when_repository_is_available(tmp_path: Path):
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "bench@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Benchmark Test"], cwd=tmp_path, check=True)
    (tmp_path / "x.txt").write_text("x\n")
    subprocess.run(["git", "add", "x.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    clean = provenance_metadata(tmp_path)
    assert clean["source_kind"] == "git"
    assert clean["git_commit"]
    assert clean["git_dirty"] is False
    (tmp_path / "x.txt").write_text("changed\n")
    assert provenance_metadata(tmp_path)["git_dirty"] is True


def test_environment_metadata_contains_reproducibility_fields():
    env = environment_metadata(include_reverse_backend=True)
    assert env["timestamp_utc"]
    assert env["os"]["system"]
    assert env["cpu"]["logical_count"]
    assert env["python"]["version"]
    assert env["numpy"]
    assert env["scipy"]
    assert "cc" in env["compilers"] and "cxx" in env["compilers"]
    assert env["active_reverse_backend"] in {"python", "native"}
    assert env["provenance"]["git_commit"] or env["provenance"]["source_tree_sha256"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "suite",
    (
        "topology_search", "lower_bounds", "gradients", "warm_start",
        "native_stack", "resolution", "direct_transcription", "full_ocp",
        "full_ocp_sensitivity",
    ),
)
def test_smoke_run_each_suite_and_validate_schema(tmp_path: Path, suite: str):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / suite
    subprocess.run(
        [
            sys.executable, "-m", "benchmarks.run",
            "--profile", "smoke", "--suite", suite,
            "--output-dir", str(output), "--no-plots", "--no-report",
        ],
        cwd=root,
        check=True,
        timeout=120,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    path = output / f"{suite}.json"
    assert path.exists(), suite
    data = json.loads(path.read_text())
    validate_benchmark_result(data, expected_benchmark=suite, expected_profile="smoke")
    manifest = json.loads((output / "manifest.json").read_text())
    validate_manifest(manifest)
    assert suite in manifest["resolved_suites"]


def test_execution_status_schema_rejects_unknown_state():
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "x",
        "status": "complete",
        "profile": "smoke",
        "environment": _minimal_environment(),
        "aggregate": {},
        "rows": [{"execution_status": "mystery_failure"}],
    }
    with pytest.raises(ResultSchemaError):
        validate_benchmark_result(payload)


def test_failure_taxonomy_classifies_internal_cap():
    from benchmarks.common.status import INTERNAL_CAP, classify_exception
    assert classify_exception(RuntimeError("reverse solver internal cap reached")) == INTERNAL_CAP


def test_native_preflight_fails_cleanly_when_artifacts_missing(tmp_path: Path):
    from benchmarks.common.native import NativeBuildError, ensure_native_available
    with pytest.raises(NativeBuildError, match="make native"):
        ensure_native_available(tmp_path)


def test_source_tree_hash_ignores_benchmark_result_directories(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    reference = tmp_path / "benchmark_results" / "reference"
    reference.mkdir(parents=True)
    generated = reference / "manifest.json"
    generated.write_text("one")
    first = source_tree_sha256(tmp_path)
    generated.write_text("two")
    assert source_tree_sha256(tmp_path) == first


def test_public_suite_tiers_and_names_are_explicit():
    from benchmarks.config import SUITES
    assert SUITES["direct_transcription"].public_title.startswith("Benchmark 7 —")
    assert SUITES["full_ocp"].public_title.startswith("Benchmark 8 —")
    assert SUITES["full_ocp_sensitivity"].tier == "supplemental"
    assert all(SUITES[name].tier == "official" for name in SUITES if name != "full_ocp_sensitivity")


def test_default_run_directory_uses_benchmark_results_runs(tmp_path: Path):
    from benchmarks.run import _unique_run_directory
    output = _unique_run_directory(tmp_path, "core")
    assert output.parent == tmp_path / "benchmark_results" / "runs"
    assert output.name.endswith("_core")


def test_reference_flag_rejects_partial_profile_before_execution():
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.run", "--profile", "core", "--reference"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "--reference requires --profile all" in proc.stderr


def test_report_generator_operates_from_json_only(tmp_path: Path):
    from benchmarks.report import generate
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "full_ocp_sensitivity",
        "status": "complete",
        "profile": "smoke",
        "environment": _minimal_environment(),
        "aggregate": {},
        "cases": [],
    }
    (tmp_path / "full_ocp_sensitivity.json").write_text(json.dumps(payload))
    output = tmp_path / "BENCHMARK_REPORT.md"
    generate(tmp_path, output, tmp_path / "plots")
    text = output.read_text()
    assert "machine-readable JSON" in text
    assert "Supplemental sensitivity experiments" in text
