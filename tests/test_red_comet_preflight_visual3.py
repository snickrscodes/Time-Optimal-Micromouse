from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from tools.red_comet.preflight import run_preflight
from visualization.case_study import load_case_study, render_astar_vs_bnb_case_study


ROOT = Path(__file__).resolve().parents[1]
RED_COMET = ROOT / "examples/mazes/historical/red_comet_reference_maze.json"


def test_red_comet_preflight_exact_topology_count(tmp_path: Path) -> None:
    result = run_preflight(RED_COMET, output_dir=tmp_path, render_gallery=False)
    assert result["enumeration"]["truncated"] is False
    assert result["enumeration"]["simple_paths"] == 10
    assert result["graph"]["relevant_cycle_rank"] == 4
    assert result["astar"]["grid_steps"] == 99
    assert abs(result["astar"]["topological_length_m"] - 17.82) < 1e-12
    assert result["search_preflight"]["minimum_non_astar_complete_lb_seconds"] > result["astar"]["complete_lower_bound_seconds"]
    assert result["search_preflight"]["minimum_121plus_step_complete_lb_seconds"] > 23.0
    dry = result["search_preflight"]["conditional_dry_bnb"]
    assert dry
    assert all(row["exhausted"] for row in dry)
    assert max(row["complete_alternative_evaluations"] for row in dry) <= 9
    assert (tmp_path / "preflight.json").exists()
    assert (tmp_path / "PREFLIGHT.md").exists()


def test_visual3_renderer_consumes_saved_benchmark_only(tmp_path: Path) -> None:
    result = ROOT / "benchmark_results/reference/topology_search.json"
    case = load_case_study(result, case_name="cyclic_4x4_s043")
    assert case.topology_changed is True
    assert case.improvement_percent > 20.0
    output = tmp_path / "visual3.svg"
    render_astar_vs_bnb_case_study(case, output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_visual3_main_metadata_schema_resolves_custom_maze(tmp_path: Path) -> None:
    # Reuse a real saved route snapshot solely to exercise metadata parsing; no
    # planning/optimization is invoked by this test.
    benchmark = json.loads((ROOT / "benchmark_results/reference/topology_search.json").read_text())
    row = next(case for case in benchmark["cases"] if case["case"]["name"] == "cyclic_4x4_s043")
    document = {
        "configuration": {"maze_file": str(RED_COMET)},
        "search": {"expanded": 1, "complete_paths_evaluated": 0, "pruned_by_bound": 0, "pruned_by_reachability": 0},
        "seed_route": row["astar"]["route"],
        "best_route": row["branch_and_bound"]["route"],
    }
    path = tmp_path / "main_result.json"
    path.write_text(json.dumps(document))
    case = load_case_study(path)
    assert case.scenario is not None
    assert case.scenario.name.startswith("Red Comet")
