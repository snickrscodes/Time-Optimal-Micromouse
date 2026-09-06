from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planning.maze_io import load_maze_scenario
from tools.red_comet.routes import is_historical_green

RESULT = ROOT / "analysis/red_comet_2017/final_result.json"
MAZE = ROOT / "examples/mazes/historical/red_comet_reference_maze.json"


def test_corrected_historical_green_route_is_canonical() -> None:
    doc = json.loads(RESULT.read_text())
    scenario = load_maze_scenario(MAZE)
    hist_cells = [tuple(cell) for cell in doc["historical_green"]["cells"]]
    astar_cells = [tuple(cell) for cell in doc["astar"]["cells"]]
    assert is_historical_green(scenario, hist_cells) is True
    assert is_historical_green(scenario, astar_cells) is False


def test_topology_summary_marks_only_the_corrected_historical_route() -> None:
    doc = json.loads(RESULT.read_text())
    summary = doc["topology_search"]["complete_topologies"]
    canonical = [row for row in summary if row["historical_green"]]
    assert len(canonical) == 1
    assert canonical[0]["label"] == "Historical Red Comet route"
    assert canonical[0]["optimized_time_seconds"] == doc["historical_green"]["time"]
    assert all(row["label"] != "Historical green" for row in summary)
