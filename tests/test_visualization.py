from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mazegen import Maze
from optimization import ConvexCell, CorridorModel, GeometryState, RectangleBody
from optimization import reverse_solver
from visualization.comparison import draw_solution
from visualization.maze import draw_maze_walls
from visualization.records import initial_state_from_record, route_snapshot
from visualization.sampling import (
    sample_geometry_parameters,
    sample_geometry_stations,
    sample_raw_geometry,
)
from visualization.speed import build_speed_profile_trace
from visualization.trajectory import body_polygon, convex_cell_polygon, draw_corridor


def test_geometry_trace_exact_endpoints_and_knots():
    initial = GeometryState(0.5, 0.5, 0.0, 0.0)
    raw = [0.4, 0.0, 0.6, 0.0]
    trace = sample_raw_geometry(raw, initial, samples_per_unit=20.0, minimum_samples_per_segment=4)
    assert trace.s[0] == pytest.approx(0.0)
    assert trace.s[-1] == pytest.approx(1.0)
    assert trace.knot_s.tolist() == pytest.approx([0.0, 0.4, 1.0])
    assert trace.x[0] == pytest.approx(0.5)
    assert trace.y[0] == pytest.approx(0.5)
    assert trace.x[-1] == pytest.approx(1.5)
    assert trace.y[-1] == pytest.approx(0.5)
    assert trace.theta[-1] == pytest.approx(0.0)
    assert trace.kappa[-1] == pytest.approx(0.0)
    assert np.all(np.diff(trace.s) > 0.0)


def test_explicit_station_sampling_assigns_final_segment_safely():
    initial = GeometryState(0.0, 0.0, 0.0, 0.0)
    trace = sample_geometry_stations([0.25, 0.0, 0.75, 0.0], initial, [0.0, 0.25, 1.0])
    assert trace.segment_index.tolist() == [0, 1, 1]
    assert trace.x.tolist() == pytest.approx([0.0, 0.25, 1.0])


def test_body_polygon_preserves_rectangle_dimensions():
    body = RectangleBody(front=0.4, rear=0.2, left=0.15, right=0.10)
    state = GeometryState(2.0, 3.0, 0.7, 0.0)
    polygon = body_polygon(state, body)
    edges = [np.linalg.norm(polygon[(i + 1) % 4] - polygon[i]) for i in range(4)]
    assert edges == pytest.approx([body.height, body.length, body.height, body.length])
    assert np.mean(polygon, axis=0).shape == (2,)


def test_convex_cell_polygon_and_corridor_layer():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = ConvexCell.axis_aligned_rectangle(1.0, 3.0, 2.0, 4.0)
    polygon = convex_cell_polygon(cell)
    assert polygon.shape == (4, 2)
    assert set(map(tuple, np.round(polygon, 12))) == {
        (1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)
    }
    corridor = CorridorModel((cell,), (0,), RectangleBody.centered(0.5, 0.4))
    fig, ax = plt.subplots()
    patches = draw_corridor(ax, corridor)
    assert len(patches) == 1
    plt.close(fig)


def test_route_snapshot_contains_reconstructable_geometry():
    route = SimpleNamespace(
        cells=((0, 0), (1, 0)),
        parameters=np.asarray([1.0, 0.0]),
        initial_state=GeometryState(0.5, 0.5, 0.0, 0.0),
        time=0.5,
        selected_stage="test",
        architecture="test_arch",
        stages=(),
        warm_start_record=None,
        sparse_specialist_record=None,
    )
    record = route_snapshot(route, config={"init_w": 0.8})
    assert record["raw_parameters"] == pytest.approx([1.0, 0.0])
    assert record["parameters"] == pytest.approx([1.0, 0.0])
    assert record["optimized_clothoid_length"] == pytest.approx(1.0)
    assert initial_state_from_record(record) == route.initial_state


def test_speed_trace_matches_authoritative_time_and_restores_backend():
    previous = reverse_solver.reverse_backend()
    trace = build_speed_profile_trace(
        [1.0, 0.0],
        init_w=0.8,
        terminal_w_max=0.8,
        samples_per_unit=100.0,
        time_tolerance=1e-5,
    )
    assert reverse_solver.reverse_backend() == previous
    assert trace.integration_error <= 1e-5
    assert trace.exact_total_time == pytest.approx(0.489447715151901, abs=1e-12)
    assert trace.s[0] == pytest.approx(0.0)
    assert trace.s[-1] == pytest.approx(1.0)
    assert trace.w[0] == pytest.approx(0.8)
    assert trace.w[-1] <= 0.8 + 1e-10
    assert {interval.mode for interval in trace.intervals} == {"MOTOR", "BRAKE"}
    assert any(event.kind == "mode_switch" for event in trace.events)
    assert trace.station_at_time([0.0, trace.exact_total_time]).tolist() == pytest.approx([0.0, 1.0])


def test_speed_intervals_cover_path_without_gaps():
    trace = build_speed_profile_trace(
        [1.0, 0.0], init_w=0.8, terminal_w_max=0.8, time_tolerance=1e-5
    )
    assert trace.intervals[0].station0 == pytest.approx(0.0)
    assert trace.intervals[-1].station1 == pytest.approx(trace.total_length)
    for left, right in zip(trace.intervals[:-1], trace.intervals[1:]):
        assert left.station1 == pytest.approx(right.station0, abs=1e-12)


def test_static_renderer_smoke_saves_svg(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    connections = {(0, 0): {(1, 0)}, (1, 0): {(0, 0)}}
    maze = Maze.from_connections(2, 1, connections, (1, 0))
    route = SimpleNamespace(
        cells=((0, 0), (1, 0)),
        parameters=np.asarray([1.0, 0.0]),
        initial_state=GeometryState(0.5, 0.5, 0.0, 0.0),
        time=0.5,
        selected_stage="test",
    )
    fig, ax = plt.subplots()
    draw_solution(ax, maze, route, title="smoke")
    output = tmp_path / "smoke.svg"
    fig.savefig(output)
    plt.close(fig)
    assert output.exists()
    assert output.stat().st_size > 500


def test_speed_plot_and_geometry_detail_smoke(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from planning import build_route_optimization_problem
    from visualization.detail import draw_geometry_detail
    from visualization.speed_plot import draw_speed_profile

    cells = ((0, 0), (1, 0))
    connections = {(0, 0): {(1, 0)}, (1, 0): {(0, 0)}}
    maze = Maze.from_connections(2, 1, connections, (1, 0))
    problem = build_route_optimization_problem(cells)

    fig, ax = plt.subplots()
    draw_geometry_detail(
        ax, maze=maze, problem=problem,
        parameters=problem.initial_parameters, cells=cells, footprint_count=3,
    )
    geometry_path = tmp_path / "geometry.svg"
    fig.savefig(geometry_path)
    plt.close(fig)
    assert geometry_path.stat().st_size > 500

    trace = build_speed_profile_trace(
        [1.0, 0.0], init_w=0.8, terminal_w_max=0.8, time_tolerance=1e-5
    )
    fig, axes = plt.subplots(2, 1)
    draw_speed_profile(axes[0], axes[1], trace)
    speed_path = tmp_path / "speed.svg"
    fig.savefig(speed_path)
    plt.close(fig)
    assert speed_path.stat().st_size > 500


def test_playback_trace_uses_real_time_and_exact_geometry():
    from visualization.animation import build_playback_trace
    from optimization import GeometryState

    speed = build_speed_profile_trace([1.0, 0.0], init_w=0.8, terminal_w_max=0.8, time_tolerance=1e-5)
    playback = build_playback_trace([1.0, 0.0], GeometryState(0.0, 0.0, 0.0, 0.0), speed, fps=10.0)
    assert playback.time[0] == pytest.approx(0.0)
    assert playback.time[-1] == pytest.approx(speed.exact_total_time)
    assert playback.station[0] == pytest.approx(0.0)
    assert playback.station[-1] == pytest.approx(1.0)
    assert playback.geometry.x[-1] == pytest.approx(1.0)
    assert len(playback.mode) == len(playback.time)


def test_animation_smoke_produces_gif(tmp_path: Path):
    from PIL import Image
    from visualization.animation import build_playback_trace, render_trajectory_gif

    connections = {(0, 0): {(1, 0)}, (1, 0): {(0, 0)}}
    maze = Maze.from_connections(2, 1, connections, (1, 0))
    speed = build_speed_profile_trace([1.0, 0.0], init_w=0.8, terminal_w_max=0.8, time_tolerance=1e-5)
    initial = GeometryState(0.5, 0.5, 0.0, 0.0)
    playback = build_playback_trace([1.0, 0.0], initial, speed, fps=5.0)
    output = render_trajectory_gif(
        tmp_path / "playback.gif", maze=maze, playback=playback,
        body=RectangleBody.centered(0.5, 0.4), cells=((0, 0), (1, 0)), fps=5.0, dpi=50,
    )
    assert output.stat().st_size > 1000
    image = Image.open(output)
    assert getattr(image, "n_frames", 1) >= 2


def test_search_trace_counts_reconcile_with_branch_and_bound():
    from mazegen import MazeGenerator, compress_to_graph
    from main import add_random_openings
    from planning import SearchSettings, branch_and_bound_junction_paths
    from planning.time_bounds import TimeBoundResult

    class ZeroLowerBound:
        name = "zero"
        def evaluate(self, request):
            return TimeBoundResult("zero", 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, ())

    maze = MazeGenerator(3, 3, 19).generate()
    maze = add_random_openings(maze, 2, seed=19 ^ 0x5EED5EED)
    graph = compress_to_graph(maze, (0, 0), maze.goal)
    events = []
    result = branch_and_bound_junction_paths(
        maze, graph, (0, 0), maze.goal,
        lower_bound=ZeroLowerBound(),
        complete_path_time=lambda _j, cells: float(len(cells) - 1),
        init_w=0.8,
        settings=SearchSettings(maximum_expansions=200, use_complete_bound_refinement=False),
        search_trace_observer=events.append,
    )
    assert sum(e.kind == "generated" for e in events) == result.generated
    assert sum(e.kind == "expanded" for e in events) == result.expanded
    assert sum(e.kind == "complete" for e in events) == result.complete_paths_evaluated
    assert any(e.kind == "seed" for e in events)


def test_search_and_architecture_renderers_smoke(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from planning.branch_and_bound import SearchTraceEvent
    from visualization.search import draw_search_tree
    from visualization.architecture import draw_system_architecture

    events = [
        SearchTraceEvent("root", (0,), None, 0.0, 5.0),
        SearchTraceEvent("generated", (0, 1), (0,), 1.0, 5.0),
        SearchTraceEvent("complete", (0, 1), (0,), 1.0, 5.0, complete_time=3.0),
    ]
    fig, ax = plt.subplots()
    draw_search_tree(ax, events)
    p = tmp_path / "tree.svg"
    fig.savefig(p)
    plt.close(fig)
    assert p.stat().st_size > 500

    fig, ax = plt.subplots()
    draw_system_architecture(ax)
    p = tmp_path / "architecture.svg"
    fig.savefig(p)
    plt.close(fig)
    assert p.stat().st_size > 500


def test_route_snapshot_round_trip_preserves_raw_geometry_and_endpoint():
    from visualization.records import route_view_from_record

    route = SimpleNamespace(
        cells=((0, 0), (1, 0), (1, 1)),
        parameters=np.asarray([0.45, 0.0, 1.00, 0.6, 1.40, 0.0]),
        initial_state=GeometryState(0.5, 0.5, 0.0, 0.0),
        time=0.75,
        selected_stage="roundtrip",
        architecture="test_arch",
        stages=(),
        warm_start_record=None,
        sparse_specialist_record=None,
    )
    record = route_snapshot(route, config={"init_w": 0.8})
    view = route_view_from_record(record)
    assert view.raw_parameters.tolist() == pytest.approx(record["raw_parameters"])
    assert view.parameters.tolist() == pytest.approx(route.parameters.tolist())
    original = sample_raw_geometry(record["raw_parameters"], route.initial_state)
    restored = sample_raw_geometry(view.raw_parameters, view.initial_state)
    assert restored.x[-1] == pytest.approx(original.x[-1], abs=1e-13)
    assert restored.y[-1] == pytest.approx(original.y[-1], abs=1e-13)
    assert restored.theta[-1] == pytest.approx(original.theta[-1], abs=1e-13)
    assert restored.kappa[-1] == pytest.approx(original.kappa[-1], abs=1e-13)


def test_canonical_ocp_resolution_control_renderer_saves_svg(tmp_path: Path):
    from tools.visuals.ocp_comparison import render

    source = Path("benchmark_results/reference/full_ocp_resolution_control.json")
    output = tmp_path / "ocp.svg"
    render(source, output)
    assert output.exists() and output.stat().st_size > 500


def test_search_trace_all_public_counters_reconcile():
    from mazegen import MazeGenerator, compress_to_graph
    from main import add_random_openings
    from planning import SearchSettings, branch_and_bound_junction_paths
    from planning.time_bounds import TimeBoundResult

    class ZeroLowerBound:
        name = "zero"
        def evaluate(self, request):
            return TimeBoundResult("zero", 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, ())

    maze = MazeGenerator(3, 3, 19).generate()
    maze = add_random_openings(maze, 2, seed=19 ^ 0x5EED5EED)
    graph = compress_to_graph(maze, (0, 0), maze.goal)
    events = []
    result = branch_and_bound_junction_paths(
        maze, graph, (0, 0), maze.goal,
        lower_bound=ZeroLowerBound(),
        complete_path_time=lambda _j, cells: float(len(cells) - 1),
        init_w=0.8,
        settings=SearchSettings(maximum_expansions=200, use_complete_bound_refinement=False),
        search_trace_observer=events.append,
    )
    counts = {kind: sum(e.kind == kind for e in events) for kind in {
        "generated", "expanded", "complete", "pruned_bound",
        "pruned_complete_bound", "pruned_reachability", "rejected_visit",
    }}
    assert counts["generated"] == result.generated
    assert counts["expanded"] == result.expanded
    assert counts["complete"] == result.complete_paths_evaluated
    assert counts["pruned_bound"] + counts["pruned_complete_bound"] == result.pruned_by_bound
    assert counts["pruned_complete_bound"] == result.pruned_by_complete_bound
    assert counts["pruned_reachability"] == result.pruned_by_reachability
    assert counts["rejected_visit"] == result.rejected_by_visit_limit
