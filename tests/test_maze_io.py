from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mazegen import compress_to_graph
from optimization import GeometryState
from planning import (
    MazeFormatError,
    astar_junction_path,
    build_route_optimization_problem,
    expand_junction_path,
    load_maze_scenario,
)


EXAMPLE = Path(__file__).parents[1] / "examples" / "mazes" / "ame_maze_v1_example.json"


def test_loader_normalizes_southwest_and_derives_unique_goal_entry() -> None:
    scenario = load_maze_scenario(EXAMPLE)
    assert scenario.start == (1, 3)
    assert set(scenario.goal_region.cells) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert len(scenario.goal_region.entrances) == 1
    entrance = scenario.goal_region.entrances[0]
    assert entrance.outside_cell == (1, 3)
    assert entrance.inside_cell == (1, 2)
    assert scenario.canonical_goal == (1, 2)
    assert scenario.to_source_cell(scenario.start) == (1, 0)
    assert scenario.to_source_cell(scenario.canonical_goal) == (1, 1)
    assert scenario.maze.goal == scenario.canonical_goal
    assert scenario.scale.wall_thickness_units == 1.0
    assert scenario.scale.unit_mm == 12.0
    assert scenario.scale.cell_pitch_m == pytest.approx(0.18)


def test_loaded_maze_uses_existing_graph_and_astar_unchanged() -> None:
    scenario = load_maze_scenario(EXAMPLE)
    graph = compress_to_graph(scenario.maze, scenario.start, scenario.canonical_goal)
    nodes = astar_junction_path(graph, scenario.start, scenario.canonical_goal)
    cells = tuple(expand_junction_path(scenario.maze, graph, nodes))
    assert cells == ((1, 3), (1, 2))


def test_loader_is_source_deterministic() -> None:
    first = load_maze_scenario(EXAMPLE)
    second = load_maze_scenario(EXAMPLE)
    assert first.source_sha256 == second.source_sha256
    assert first.maze.connection_dict() == second.maze.connection_dict()
    assert first.goal_region == second.goal_region


def test_loader_rejects_multiple_goal_entrances(tmp_path: Path) -> None:
    doc = json.loads(EXAMPLE.read_text())
    # Open the south edge beneath the second lower goal cell as well.
    row = list(doc["walls"]["horizontal"][3])
    row[2] = "0"
    doc["walls"]["horizontal"][3] = "".join(row)
    path = tmp_path / "two_entries.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(MazeFormatError, match="exactly one external opening"):
        load_maze_scenario(path)


def test_loader_rejects_open_exterior_boundary(tmp_path: Path) -> None:
    doc = json.loads(EXAMPLE.read_text())
    doc["walls"]["horizontal"][0] = "0111"
    path = tmp_path / "open_boundary.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(MazeFormatError, match="exterior boundaries"):
        load_maze_scenario(path)


def test_terminal_goal_is_region_based_and_heading_curvature_are_free() -> None:
    problem = build_route_optimization_problem(((0, 0), (1, 0), (1, 1)))
    # Final transition is downward into (1,1), hence the entry edge is y=1.
    assert problem.endpoint_target == (None, 1.0, None, None)

    gx, gy = problem.terminal_cell
    portal_a = GeometryState(gx + 0.2, gy, 2.7, -8.0)
    portal_b = GeometryState(gx + 0.9, gy, -2.1, 9.5)
    inside_but_past_entry = GeometryState(gx + 0.5, gy + 0.1, 0.0, 0.0)
    assert problem.terminal_violation(portal_a) == 0.0
    assert problem.terminal_violation(portal_b) == 0.0
    assert problem.terminal_violation(inside_but_past_entry) == pytest.approx(0.1)

    values_a, jac_a = problem.terminal_constraint()(problem.initial_parameters)
    assert values_a.shape == (4,)
    assert jac_a.shape == (4, problem.initial_parameters.size)
    assert np.all(np.isfinite(values_a))
    assert np.all(np.isfinite(jac_a))


def test_terminal_speed_is_an_upper_cap_and_backends_agree() -> None:
    from optimization.reverse_solver import time_value_and_gradient, using_reverse_backend
    from optimization.scalar_reverse_solver import evaluate_time_scalar_result

    raw = [0.25, 0.0]
    values = {}
    gradients = {}
    for backend in ("python", "native"):
        with using_reverse_backend(backend):
            slow = evaluate_time_scalar_result(
                raw, init_w=0.8, terminal_w_max=0.2, initial_k=0.0
            ).value
            loose = evaluate_time_scalar_result(
                raw, init_w=0.8, terminal_w_max=0.8, initial_k=0.0
            ).value
            value, gradient = time_value_and_gradient(
                raw, init_w=0.8, terminal_w_max=0.2, initial_k=0.0
            )
        assert slow > loose
        assert value == pytest.approx(slow, abs=1e-12)
        values[backend] = value
        gradients[backend] = np.asarray(gradient)

    assert values["native"] == pytest.approx(values["python"], abs=1e-12)
    assert np.allclose(gradients["native"], gradients["python"], atol=1e-11, rtol=1e-11)


HISTORICAL = Path(__file__).parents[1] / "examples" / "mazes" / "historical"


@pytest.mark.parametrize(
    ("filename", "source_entrance", "source_goal"),
    [
        ("micromouse_maze_01.json", ((9, 7), (8, 7)), (8, 7)),
        ("micromouse_maze_02.json", ((7, 9), (7, 8)), (7, 8)),
        ("red_comet_reference_maze.json", ((7, 9), (7, 8)), (7, 8)),
    ],
)
def test_verified_16x16_transcriptions_load_deterministically(
    filename: str,
    source_entrance: tuple[tuple[int, int], tuple[int, int]],
    source_goal: tuple[int, int],
) -> None:
    scenario = load_maze_scenario(HISTORICAL / filename)
    assert (scenario.maze.width, scenario.maze.height) == (16, 16)
    assert scenario.to_source_cell(scenario.start) == (0, 0)
    assert {scenario.to_source_cell(cell) for cell in scenario.goal_region.cells} == {
        (7, 7), (8, 7), (7, 8), (8, 8)
    }
    assert len(scenario.goal_region.entrances) == 1
    entry = scenario.goal_region.entrances[0]
    assert (scenario.to_source_cell(entry.outside_cell), scenario.to_source_cell(entry.inside_cell)) == source_entrance
    assert scenario.to_source_cell(scenario.canonical_goal) == source_goal
    graph = compress_to_graph(scenario.maze, scenario.start, scenario.canonical_goal)
    path = astar_junction_path(graph, scenario.start, scenario.canonical_goal)
    assert graph.nodes[path[0]] == scenario.start
    assert graph.nodes[path[-1]] == scenario.canonical_goal
