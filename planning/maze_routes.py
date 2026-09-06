"""Maze topology helpers and feasible clothoid initializers for benchmarks."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from mazegen import Cell, JunctionGraph, Maze
from optimization import (
    ConvexCell,
    CorridorModel,
    CurvatureSlopeConstraint,
    EndpointBoxConstraint,
    ExchangeSettings,
    GeometryState,
    PathOptimizationResult,
    PathOptimizerSettings,
    RectangleBody,
    SLSQPSettings,
    stack_vector_constraints,
    compile_geometry_path,
    optimize_path,
)

Array = NDArray[np.float64]
Direction = tuple[int, int]

# Vehicle footprint in grid-cell units.  The path reference point is centered
# in this axis-aligned body frame; orientation is applied by the separator.
DEFAULT_BODY_LENGTH = 5.0 / 9.0
DEFAULT_BODY_HEIGHT = 4.0 / 9.0


def shortest_cell_path(maze: Maze, start: Cell, goal: Cell) -> list[Cell]:
    """Unweighted shortest path without assuming the maze is a tree."""
    queue = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in maze.neighbors(current):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if goal not in parent:
        raise ValueError("goal is unreachable")
    path: list[Cell] = []
    current: Cell | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def shortest_junction_path(
    graph: JunctionGraph,
    start: Cell,
    goal: Cell,
) -> list[int]:
    """Dijkstra path on the compact junction graph."""
    source = graph.index[start]
    target = graph.index[goal]
    distance = [math.inf] * len(graph.nodes)
    parent = [-1] * len(graph.nodes)
    distance[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        value, node = heapq.heappop(heap)
        if value != distance[node]:
            continue
        if node == target:
            break
        for edge in graph.adj[node]:
            candidate = value + edge.length
            if candidate < distance[edge.to]:
                distance[edge.to] = candidate
                parent[edge.to] = node
                heapq.heappush(heap, (candidate, edge.to))
    if not math.isfinite(distance[target]):
        raise ValueError("goal is unreachable in junction graph")
    path = []
    node = target
    while node >= 0:
        path.append(node)
        if node == source:
            break
        node = parent[node]
    path.reverse()
    return path


def astar_junction_path(
    graph: JunctionGraph,
    start: Cell,
    goal: Cell,
) -> list[int]:
    """Naive A* seed path on the compact junction graph.

    Edge costs are corridor lengths in grid-cell units.  Manhattan distance
    between junction cells is an admissible and consistent heuristic for any
    axis-grid maze, including mazes with cycles.
    """
    source = graph.index[start]
    target = graph.index[goal]
    target_cell = graph.nodes[target]

    def heuristic(node: int) -> int:
        x, y = graph.nodes[node]
        return abs(target_cell[0] - x) + abs(target_cell[1] - y)

    distance = [math.inf] * len(graph.nodes)
    parent = [-1] * len(graph.nodes)
    distance[source] = 0.0
    heap: list[tuple[float, float, int]] = [
        (float(heuristic(source)), 0.0, source)
    ]
    while heap:
        _estimate, value, node = heapq.heappop(heap)
        if value != distance[node]:
            continue
        if node == target:
            break
        for edge in graph.adj[node]:
            candidate = value + edge.length
            if candidate < distance[edge.to]:
                distance[edge.to] = candidate
                parent[edge.to] = node
                heapq.heappush(
                    heap,
                    (candidate + heuristic(edge.to), candidate, edge.to),
                )
    if not math.isfinite(distance[target]):
        raise ValueError("goal is unreachable in junction graph")

    path: list[int] = []
    node = target
    while node >= 0:
        path.append(node)
        if node == source:
            break
        node = parent[node]
    path.reverse()
    return path


def expand_junction_edge(
    maze: Maze,
    graph: JunctionGraph,
    source_id: int,
    destination_id: int,
) -> tuple[Cell, ...]:
    """Expand one compact graph edge, excluding its source cell.

    Branch-and-bound appends this tuple to the parent cell path instead of
    rebuilding the entire expanded route for every child.
    """
    source = graph.nodes[source_id]
    edge = next((item for item in graph.adj[source_id] if item.to == destination_id), None)
    if edge is None:
        raise ValueError(f"junction nodes are not adjacent: {source_id}->{destination_id}")
    direction = ((0, -1), (1, 0), (0, 1), (-1, 0))[edge.direction]
    current = source
    cells: list[Cell] = []
    previous = source
    for _ in range(edge.length):
        current = current[0] + direction[0], current[1] + direction[1]
        if not maze.is_path_between(previous, current):
            raise AssertionError("junction edge expansion crossed a closed wall")
        cells.append(current)
        previous = current
    if current != graph.nodes[destination_id]:
        raise AssertionError("junction edge length/direction is inconsistent")
    return tuple(cells)


def expand_junction_path(
    maze: Maze,
    graph: JunctionGraph,
    node_path: Sequence[int],
) -> list[Cell]:
    """Expand compact straight-corridor edges back into grid cells."""
    if not node_path:
        raise ValueError("node_path must not be empty")
    cells = [graph.nodes[node_path[0]]]
    for source_id, destination_id in zip(node_path[:-1], node_path[1:]):
        cells.extend(expand_junction_edge(maze, graph, source_id, destination_id))
    return cells


def _angle(direction: Direction) -> float:
    return math.atan2(direction[1], direction[0])


def _turn_sign(incoming: Direction, outgoing: Direction) -> float:
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    return 1.0 if cross > 0 else -1.0 if cross < 0 else 0.0


def _unit_right_angle_euler_constants() -> tuple[float, float]:
    # Normalize each half to length one and total heading to pi/2.  The final
    # displacement is symmetric; scaling it to (1/2,1/2) yields an exact turn
    # from one portal midpoint to the adjacent portal midpoint of a unit cell.
    normalized = compile_geometry_path(
        (1.0, 0.5 * math.pi, 1.0, -0.5 * math.pi),
        GeometryState(0.0, 0.0, 0.0, 0.0),
    )
    half_length = 0.5 / normalized.final_state.x
    peak_curvature = 0.5 * math.pi / half_length
    return half_length, peak_curvature


TURN_HALF_LENGTH, TURN_PEAK_CURVATURE = _unit_right_angle_euler_constants()


@dataclass(frozen=True, slots=True)
class OpenRoomSpan:
    """A contiguous route interval relaxed to one fully open rectangular room.

    ``start_index`` and ``end_index`` are inclusive indices into the member
    cell path.  ``cells`` must exactly fill the stated axis-aligned rectangle.
    The continuous separator still checks the real oriented body against the
    rectangle boundary; only the artificial internal cell-path cages are
    removed.
    """

    start_index: int
    end_index: int
    cells: tuple[Cell, ...]
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    block_index: int = -1

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("open-room span indices are invalid")
        expected = {
            (x, y)
            for x in range(self.xmin, self.xmax + 1)
            for y in range(self.ymin, self.ymax + 1)
        }
        if set(self.cells) != expected:
            raise ValueError("open-room cells must exactly fill their rectangle")


@dataclass(frozen=True, slots=True)
class RouteOptimizationProblem:
    cells: tuple[Cell, ...]
    initial_parameters: Array
    corridor: CorridorModel
    initial_state: GeometryState
    endpoint_target: tuple[float | None, float | None, float | None, float | None]
    terminal_cell: Cell
    corridor_groups: tuple[tuple[Cell, ...], ...]
    open_room_spans: tuple[OpenRoomSpan, ...] = ()

    def terminal_constraint(self) -> EndpointBoxConstraint:
        """Keep the endpoint on the finite entry edge of the goal cell.

        One Cartesian coordinate is enforced exactly through ``endpoint_target``;
        this box constrains the free coordinate to the physical shared edge.
        Heading and curvature remain unconstrained.
        """
        gx, gy = self.terminal_cell
        return EndpointBoxConstraint(
            self.initial_state, float(gx), float(gx + 1), float(gy), float(gy + 1)
        )

    def inequalities(self, extra=None):
        return stack_vector_constraints(self.terminal_constraint(), extra)

    def terminal_violation(self, state: GeometryState) -> float:
        gx, gy = self.terminal_cell
        tx, ty, _theta, _k = self.endpoint_target
        terms = [
            gx - state.x,
            state.x - (gx + 1.0),
            gy - state.y,
            state.y - (gy + 1.0),
            0.0,
        ]
        if tx is not None:
            terms.append(abs(state.x - tx))
        if ty is not None:
            terms.append(abs(state.y - ty))
        return float(max(terms))


def _maximal_direction_runs(
    cells: tuple[Cell, ...],
    directions: Sequence[Direction],
) -> tuple[tuple[tuple[Cell, ...], ...], tuple[int, ...]]:
    """Return a convex straight-run cover and one run index per route edge.

    A maximal run is a consecutive block of equally directed grid edges.  Its
    cells form a one-cell-wide axis-aligned rectangle.  Consecutive runs overlap
    in the complete turn cell, so a continuous path can choose its transition
    point anywhere in that cell rather than at a prescribed portal or knot.
    """
    groups: list[list[Cell]] = []
    edge_runs: list[int] = []
    for edge, direction in enumerate(directions):
        if edge == 0 or direction != directions[edge - 1]:
            groups.append([cells[edge], cells[edge + 1]])
        else:
            groups[-1].append(cells[edge + 1])
        edge_runs.append(len(groups) - 1)
    return tuple(tuple(group) for group in groups), tuple(edge_runs)


def _refine_knot_initializer(
    parameters: Sequence[float],
    segment_cells: Sequence[int],
    factor: int,
    *,
    initial_s: float = 0.0,
    initial_k: float = 0.0,
) -> tuple[Array, tuple[int, ...]]:
    """Exactly split each linear-curvature segment into ``factor`` children."""
    if factor < 1:
        raise ValueError("refinement_factor must be a positive integer")
    source = np.asarray(parameters, dtype=float)
    if source.ndim != 1 or source.size == 0 or source.size % 2:
        raise ValueError("parameters must be a nonempty [s1,k1,...] vector")
    if len(segment_cells) != source.size // 2:
        raise ValueError("segment assignment length does not match parameters")
    if factor == 1:
        return source.copy(), tuple(int(value) for value in segment_cells)

    refined: list[float] = []
    assignments: list[int] = []
    s0 = float(initial_s)
    k0 = float(initial_k)
    for (s1_in, k1_in), cell_index in zip(
        source.reshape(-1, 2), segment_cells, strict=True
    ):
        s1 = float(s1_in)
        k1 = float(k1_in)
        for child in range(1, factor + 1):
            fraction = child / factor
            refined.extend(
                (
                    math.fma(fraction, s1 - s0, s0),
                    math.fma(fraction, k1 - k0, k0),
                )
            )
            assignments.append(int(cell_index))
        s0 = s1
        k0 = k1
    return np.asarray(refined, dtype=float), tuple(assignments)


def build_route_optimization_problem(
    cells: Sequence[Cell],
    *,
    body_length: float = DEFAULT_BODY_LENGTH,
    body_height: float = DEFAULT_BODY_HEIGHT,
    body_width: float | None = None,
    clearance: float = 0.0,
    corridor_mode: Literal[
        "overlapping_cover", "maximal_runs", "per_cell"
    ] = "overlapping_cover",
    refinement_factor: int = 1,
    open_room_spans: Sequence[OpenRoomSpan] = (),
) -> RouteOptimizationProblem:
    """Construct an exact G2 initializer and a topology-preserving corridor.

    ``body_length`` and ``body_height`` are the longitudinal and lateral
    footprint dimensions in grid-cell units.  ``body_width`` is retained as a
    compatibility alias for ``body_height``.

    ``corridor_mode="overlapping_cover"`` is the production formulation.  It
    builds an ordered overlapping convex cover from maximal straight-run
    rectangles plus the complete cell at every turn.  The analytic Euler turn
    is exactly subdivided so its outer pieces belong to the adjacent run cells
    and its inner pieces belong to the turn cell.  All transitions occur inside
    two-dimensional overlaps and can move during optimization; no portal,
    centerline point, diagonal graph edge, or explicit corner-cut primitive is
    imposed on the final curve.

    ``corridor_mode="maximal_runs"`` is a smaller legacy-improvement cover that
    assigns each half-turn directly to its incoming/outgoing run.  It permits
    corner cutting but still forces the curvature-peak knot into the run-cell
    overlap.  ``corridor_mode="per_cell"`` retains the original segment cages
    for regression comparisons.  ``refinement_factor`` exactly subdivides the
    initializer without changing its geometry, decoupling curvature resolution
    from graph-cell count.
    """
    cells = tuple(cells)
    if len(cells) < 2:
        raise ValueError("a route needs at least two cells")
    spans = tuple(open_room_spans)
    previous_end = -1
    for span in spans:
        if span.end_index >= len(cells):
            raise ValueError("open-room span extends beyond the route")
        if span.start_index <= previous_end:
            raise ValueError("open-room spans must be ordered and disjoint")
        room_cells = set(span.cells)
        if not all(
            cell in room_cells
            for cell in cells[span.start_index:span.end_index + 1]
        ):
            raise ValueError("open-room member interval leaves the room rectangle")
        previous_end = span.end_index
    if corridor_mode not in {"overlapping_cover", "maximal_runs", "per_cell"}:
        raise ValueError(
            "corridor_mode must be 'overlapping_cover', 'maximal_runs', or 'per_cell'"
        )
    if isinstance(refinement_factor, bool) or int(refinement_factor) != refinement_factor:
        raise ValueError("refinement_factor must be a positive integer")
    refinement_factor = int(refinement_factor)
    if refinement_factor < 1:
        raise ValueError("refinement_factor must be a positive integer")

    directions: list[Direction] = []
    for first, second in zip(cells[:-1], cells[1:]):
        direction = second[0] - first[0], second[1] - first[1]
        if abs(direction[0]) + abs(direction[1]) != 1:
            raise ValueError(f"non-adjacent route transition: {first}->{second}")
        directions.append(direction)

    if body_width is not None:
        width = float(body_width)
        if (
            body_height != DEFAULT_BODY_HEIGHT
            and not math.isclose(float(body_height), width, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("body_height and body_width specify different values")
        body_height = width
    body = RectangleBody.centered(body_length, body_height)
    run_groups: tuple[tuple[Cell, ...], ...] = ()
    edge_runs: tuple[int, ...] = ()
    run_cells: tuple[int, ...] = ()
    turn_cells: dict[int, int] = {}
    if corridor_mode in {"overlapping_cover", "maximal_runs"}:
        run_groups, edge_runs = _maximal_direction_runs(cells, directions)
        convex_cells = []
        groups: list[tuple[Cell, ...]] = []
        run_cell_indices: list[int] = []
        # Store cover cells in route order: run, optional turn cell, next run.
        # This is not required by the separator, but makes diagnostics readable.
        turn_route_by_run = {
            edge_runs[i - 1]: i
            for i in range(1, len(cells) - 1)
            if directions[i - 1] != directions[i]
        }
        for run_index, group in enumerate(run_groups):
            xs = [cell[0] for cell in group]
            ys = [cell[1] for cell in group]
            run_cell_indices.append(len(convex_cells))
            convex_cells.append(
                ConvexCell.axis_aligned_rectangle(
                    float(min(xs)),
                    float(max(xs) + 1),
                    float(min(ys)),
                    float(max(ys) + 1),
                    name=f"route_run_{run_index}",
                )
            )
            groups.append(group)
            if corridor_mode == "overlapping_cover" and run_index + 1 < len(run_groups):
                # The common endpoint cell of adjacent direction runs is the
                # full two-dimensional overlap through which their transition
                # is free to move.
                route_index = turn_route_by_run[run_index]
                x, y = cells[route_index]
                turn_cells[route_index] = len(convex_cells)
                convex_cells.append(
                    ConvexCell.axis_aligned_rectangle(
                        float(x), float(x + 1), float(y), float(y + 1),
                        name=f"route_turn_{route_index}_{x}_{y}",
                    )
                )
                groups.append(((x, y),))
        corridor_groups = tuple(groups)
        run_cells = tuple(run_cell_indices)
    else:
        # Legacy formulation: each geometry segment is caged in one route cell.
        # Open-side padding creates a narrow overlap in which the mandatory knot
        # between adjacent cages can cross the portal.
        pad = body.maximum_radius + clearance + 1.0e-8
        groups: list[tuple[Cell, ...]] = []
        convex_cells = []
        for i, (x, y) in enumerate(cells):
            xmin, xmax = float(x), float(x + 1)
            ymin, ymax = float(y), float(y + 1)
            used_sides: list[Direction] = []
            if i > 0:
                used_sides.append((-directions[i - 1][0], -directions[i - 1][1]))
            if i + 1 < len(cells):
                used_sides.append(directions[i])
            for dx, dy in used_sides:
                if dx < 0:
                    xmin -= pad
                elif dx > 0:
                    xmax += pad
                if dy < 0:
                    ymin -= pad
                elif dy > 0:
                    ymax += pad
            convex_cells.append(
                ConvexCell.axis_aligned_rectangle(
                    xmin, xmax, ymin, ymax, name=f"route_cell_{i}_{x}_{y}"
                )
            )
            groups.append(((x, y),))
        corridor_groups = tuple(groups)

    # Append one exact convex room for each quotient span.  The ordinary
    # path-derived cover is retained as an overlapping subset, which preserves
    # robust transitions to the fixed outside corridor without admitting any
    # point outside the quotient class's wall-aware cell union.
    room_cell_indices: list[int] = []
    if spans:
        if corridor_mode != "overlapping_cover":
            raise ValueError("open-room quotienting requires overlapping_cover mode")
        for span in spans:
            room_cell_indices.append(len(convex_cells))
            convex_cells.append(
                ConvexCell.axis_aligned_rectangle(
                    float(span.xmin),
                    float(span.xmax + 1),
                    float(span.ymin),
                    float(span.ymax + 1),
                    name=f"quotient_room_{span.block_index}_{span.xmin}_{span.ymin}",
                )
            )
            groups.append(tuple(span.cells))
        corridor_groups = tuple(groups)

    parameters: list[float] = []
    segment_cells: list[int] = []
    segment_route_indices: list[int] = []
    station = 0.0

    def append_segment(
        length: float,
        end_curvature: float,
        group_index: int,
        route_index: int,
    ) -> None:
        nonlocal station
        station += length
        parameters.extend((station, end_curvature))
        segment_cells.append(group_index)
        segment_route_indices.append(route_index)

    if corridor_mode in {"overlapping_cover", "maximal_runs"}:
        append_segment(0.5, 0.0, run_cells[edge_runs[0]], 0)
    else:
        append_segment(0.5, 0.0, 0, 0)

    for i in range(1, len(cells) - 1):
        incoming = directions[i - 1]
        outgoing = directions[i]
        if incoming == outgoing:
            group_index = (
                run_cells[edge_runs[i]]
                if corridor_mode in {"overlapping_cover", "maximal_runs"}
                else i
            )
            append_segment(1.0, 0.0, group_index, i)
            continue
        sign = _turn_sign(incoming, outgoing)
        if sign == 0.0:
            raise ValueError("immediate U-turns need a separate initializer")
        peak = sign * TURN_PEAK_CURVATURE
        if corridor_mode == "overlapping_cover":
            # Exact four-way split of the same linear-curvature Euler turn.
            # The peak-curvature knot now lies in the turn cell rather than
            # serving as the transition between incoming/outgoing run cages.
            quarter = 0.5 * TURN_HALF_LENGTH
            incoming_cell = run_cells[edge_runs[i - 1]]
            turn_cell = turn_cells[i]
            outgoing_cell = run_cells[edge_runs[i]]
            append_segment(quarter, 0.5 * peak, incoming_cell, i)
            append_segment(quarter, peak, turn_cell, i)
            append_segment(quarter, 0.5 * peak, turn_cell, i)
            append_segment(quarter, 0.0, outgoing_cell, i)
        elif corridor_mode == "maximal_runs":
            append_segment(
                TURN_HALF_LENGTH,
                peak,
                run_cells[edge_runs[i - 1]],
                i,
            )
            append_segment(TURN_HALF_LENGTH, 0.0, run_cells[edge_runs[i]], i)
        else:
            append_segment(TURN_HALF_LENGTH, peak, i, i)
            append_segment(TURN_HALF_LENGTH, 0.0, i, i)

    final_group = (
        run_cells[edge_runs[-1]]
        if corridor_mode in {"overlapping_cover", "maximal_runs"}
        else len(cells) - 1
    )
    append_segment(0.5, 0.0, final_group, len(cells) - 1)

    if spans:
        # Reassign whole interior cell traversals to the room.  At entry and
        # exit cells, split the exact initializer at half of that cell's local
        # arclength so the transition occurs inside the complete shared cell,
        # where both the outside cover and room cover are simultaneously valid.
        source_stations = np.asarray(parameters[0::2], dtype=float)
        source_curvatures = np.asarray(parameters[1::2], dtype=float)
        source_previous = np.concatenate(([0.0], source_stations[:-1]))
        source_lengths = source_stations - source_previous
        rebuilt_parameters: list[float] = []
        rebuilt_assignments: list[int] = []
        rebuilt_station = 0.0
        segment_start = 0
        while segment_start < len(segment_route_indices):
            route_index = segment_route_indices[segment_start]
            segment_end = segment_start + 1
            while (
                segment_end < len(segment_route_indices)
                and segment_route_indices[segment_end] == route_index
            ):
                segment_end += 1
            role: tuple[str, int] | None = None
            for span, room_cell in zip(spans, room_cell_indices, strict=True):
                if span.start_index < route_index < span.end_index:
                    role = ("inside", room_cell)
                    break
                if route_index == span.start_index:
                    role = ("entry", room_cell)
                    break
                if route_index == span.end_index:
                    role = ("exit", room_cell)
                    break

            local_total = float(math.fsum(map(float, source_lengths[segment_start:segment_end])))
            local_half = 0.5 * local_total
            local_position = 0.0
            k_start = (
                0.0
                if segment_start == 0
                else float(source_curvatures[segment_start - 1])
            )
            for source_index in range(segment_start, segment_end):
                length = float(source_lengths[source_index])
                k_end = float(source_curvatures[source_index])
                base_cell = int(segment_cells[source_index])
                cuts = [0.0, length]
                if role is not None and role[0] in {"entry", "exit"}:
                    cut = local_half - local_position
                    if 0.0 < cut < length:
                        cuts.insert(1, cut)
                for left, right in zip(cuts[:-1], cuts[1:]):
                    fraction = right / length
                    child_k = math.fma(fraction, k_end - k_start, k_start)
                    midpoint = local_position + 0.5 * (left + right)
                    assignment = base_cell
                    if role is not None:
                        kind, room_cell = role
                        if (
                            kind == "inside"
                            or (kind == "entry" and midpoint >= local_half)
                            or (kind == "exit" and midpoint <= local_half)
                        ):
                            assignment = room_cell
                    rebuilt_station += right - left
                    rebuilt_parameters.extend((rebuilt_station, child_k))
                    rebuilt_assignments.append(assignment)
                    k_start = child_k
                local_position += length
            segment_start = segment_end
        parameters = rebuilt_parameters
        segment_cells = rebuilt_assignments

    initial_parameters, refined_assignments = _refine_knot_initializer(
        parameters,
        segment_cells,
        refinement_factor,
    )

    sx, sy = cells[0]
    gx, gy = cells[-1]
    initial_state = GeometryState(
        sx + 0.5,
        sy + 0.5,
        _angle(directions[0]),
        0.0,
    )
    # Terminate on the shared edge by which this topology enters the final
    # goal cell.  The along-edge coordinate is free, as are heading and
    # curvature.  The hybrid speed solver separately enforces an upper bound
    # on speed at this exact entry event.
    px, py = cells[-2]
    dx, dy = gx - px, gy - py
    if (dx, dy) == (1, 0):       # enter goal from its west side
        endpoint_target = (float(gx), None, None, None)
    elif (dx, dy) == (-1, 0):    # enter from east
        endpoint_target = (float(gx + 1), None, None, None)
    elif (dx, dy) == (0, 1):     # enter from north (internal y grows down)
        endpoint_target = (None, float(gy), None, None)
    elif (dx, dy) == (0, -1):    # enter from south
        endpoint_target = (None, float(gy + 1), None, None)
    else:  # defensive; route validation should already forbid this
        raise ValueError("final route transition must be axis-adjacent")
    corridor = CorridorModel(
        tuple(convex_cells),
        refined_assignments,
        body,
        clearance,
    )
    return RouteOptimizationProblem(
        cells=cells,
        initial_parameters=initial_parameters,
        corridor=corridor,
        initial_state=initial_state,
        endpoint_target=endpoint_target,
        terminal_cell=(gx, gy),
        corridor_groups=corridor_groups,
        open_room_spans=spans,
    )


def stable_knot_bounds(
    parameters: Sequence[float],
    *,
    curvature_limit: float = 10.0,
    station_fraction: float | None = None,
) -> list[tuple[float | None, float | None]]:
    """Production bounds for the knot optimizer.

    The default log-length coordinate map already guarantees positive ordered
    segment lengths, so cumulative stations are intentionally left unbounded.
    Curvature remains box-bounded.  Pass an explicit ``station_fraction`` in
    ``[0, 0.5)`` to recover the legacy local cumulative-station boxes, for
    example when using ``KnotCoordinateSettings(mode="stations")``.
    """
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 1 or x.size == 0 or x.size % 2:
        raise ValueError("parameters must be a nonempty [s1,k1,...] vector")
    stations = x[0::2]
    previous = np.concatenate(([0.0], stations[:-1]))
    lengths = stations - previous
    if np.any(~np.isfinite(x)) or np.any(lengths <= 0.0):
        raise ValueError("parameters do not have finite increasing knot stations")
    curvature_limit = float(curvature_limit)
    if not math.isfinite(curvature_limit) or curvature_limit <= 0.0:
        raise ValueError("curvature_limit must be finite and positive")

    bounds: list[tuple[float | None, float | None]] = []
    if station_fraction is None:
        for _station in stations:
            bounds.append((None, None))
            bounds.append((-curvature_limit, curvature_limit))
        return bounds

    station_fraction = float(station_fraction)
    if (
        not math.isfinite(station_fraction)
        or station_fraction < 0.0
        or station_fraction >= 0.5
    ):
        raise ValueError("station_fraction must be finite and lie in [0, 0.5)")
    for i, station in enumerate(stations):
        left_room = station_fraction * lengths[i]
        if i + 1 < len(stations):
            right_room = station_fraction * lengths[i + 1]
        else:
            right_room = station_fraction * lengths[i]
        bounds.append((float(station - left_room), float(station + right_room)))
        bounds.append((-curvature_limit, curvature_limit))
    return bounds


def legacy_stable_knot_bounds(
    parameters: Sequence[float],
    *,
    curvature_limit: float = 10.0,
    station_fraction: float = 0.45,
) -> list[tuple[float | None, float | None]]:
    """Explicit alias for the former local cumulative-station bounds."""
    return stable_knot_bounds(
        parameters,
        curvature_limit=curvature_limit,
        station_fraction=station_fraction,
    )


def optimize_route_curvature(
    problem: RouteOptimizationProblem,
    *,
    init_w: float = 0.8,
    maximum_iterations: int = 120,
    maximum_exchange_rounds: int = 6,
    curvature_slope_limit: float | None = 50.0,
) -> PathOptimizationResult:
    settings = PathOptimizerSettings(
        exchange=ExchangeSettings(
            maximum_rounds=maximum_exchange_rounds,
            require_solver_success=False,
            slsqp=SLSQPSettings(
                max_iterations=maximum_iterations,
                ftol=1.0e-11,
                display=False,
            ),
        ),
        n_scan=96,
        envelope_scan=48,
        domain_scan=96,
    )
    return optimize_path(
        problem.initial_parameters,
        problem.corridor,
        problem.initial_state,
        init_w=init_w,
        time_weight=0.0,
        curvature_weight=1.0,
        endpoint_target=problem.endpoint_target,
        bounds=stable_knot_bounds(problem.initial_parameters),
        additional_inequalities=problem.inequalities(
            None
            if curvature_slope_limit is None
            else CurvatureSlopeConstraint(
                curvature_slope_limit,
                initial_k=problem.initial_state.k,
            )
        ),
        settings=settings,
    )


__all__ = [
    "DEFAULT_BODY_HEIGHT",
    "DEFAULT_BODY_LENGTH",
    "OpenRoomSpan",
    "RouteOptimizationProblem",
    "astar_junction_path",
    "TURN_HALF_LENGTH",
    "TURN_PEAK_CURVATURE",
    "build_route_optimization_problem",
    "expand_junction_path",
    "optimize_route_curvature",
    "shortest_cell_path",
    "shortest_junction_path",
    "stable_knot_bounds",
    "legacy_stable_knot_bounds",
]
