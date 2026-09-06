"""Path-indexed branch-and-bound over a compact junction graph.

The production search removes graph blocks that cannot lie on a simple
start-to-goal path, checks residual reachability before evaluating a child,
and asks the lower-bound hierarchy for an optional stronger complete-leaf
certificate before launching the expensive nonlinear optimizer.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from mazegen import Cell, Edge, JunctionGraph, Maze

from .graph_blocks import biconnected_edge_components, undirected_edges
from .maze_routes import expand_junction_edge, expand_junction_path, shortest_junction_path
from .time_bounds import TimeBoundRequest, TimeBoundResult, TimeLowerBound
from .topology_quotient import OpenRoomTopologyQuotient, TopologyClassKey


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Finite and exact graph-search policy."""

    maximum_node_visits: int = 1
    maximum_expansions: int = 100_000
    pruning_tolerance: float = 1.0e-12
    use_block_cut_pruning: bool = True
    use_no_revisit_reachability: bool = True
    use_complete_bound_refinement: bool = True

    def __post_init__(self) -> None:
        if self.maximum_node_visits <= 0:
            raise ValueError("maximum_node_visits must be positive")
        if self.maximum_expansions <= 0:
            raise ValueError("maximum_expansions must be positive")
        if not math.isfinite(self.pruning_tolerance) or self.pruning_tolerance < 0.0:
            raise ValueError("pruning_tolerance must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SearchNode:
    junction_path: tuple[int, ...]
    cell_path: tuple[Cell, ...]
    bound: TimeBoundResult
    visited_mask: int = 0
    portal_mask: tuple[bool, ...] | None = None


@dataclass(frozen=True, slots=True)
class BranchAndBoundResult:
    best_time: float
    best_junction_path: tuple[int, ...]
    best_cell_path: tuple[Cell, ...]
    seed_time: float
    expanded: int
    generated: int
    pruned_by_bound: int
    rejected_by_visit_limit: int
    complete_paths_evaluated: int
    minimum_open_bound: float
    exhausted: bool
    pruned_by_reachability: int = 0
    pruned_by_complete_bound: int = 0
    block_cut_removed_nodes: int = 0
    block_cut_removed_edges: int = 0
    relevant_cycle_rank: int = 0
    bound_statistics: dict[str, int | float] | None = None
    quotient_rooms: int = 0
    quotient_classes_evaluated: int = 0
    quotient_leaf_reuses: int = 0
    quotient_classes_pruned: int = 0
    quotient_variants_generated: int = 0


@dataclass(frozen=True, slots=True)
class CompletePathEvaluation:
    """A complete evaluator result with the selected class representative."""

    time: float
    junction_path: tuple[int, ...]
    cell_path: tuple[Cell, ...]


CompletePathEvaluator = Callable[
    [tuple[int, ...], tuple[Cell, ...]], float | CompletePathEvaluation
]
CompleteCandidateObserver = Callable[
    [tuple[int, ...], tuple[Cell, ...], TimeBoundResult, float], None
]


@dataclass(frozen=True, slots=True)
class SearchTraceEvent:
    """Optional diagnostic event emitted by branch-and-bound.

    The observer is disabled by default and has no role in search decisions.
    Paths provide stable node identities for small-search visualization.
    """

    kind: str
    junction_path: tuple[int, ...]
    parent_path: tuple[int, ...] | None
    lower_bound: float | None
    incumbent: float
    complete_time: float | None = None
    prune_reason: str | None = None


SearchTraceObserver = Callable[[SearchTraceEvent], None]


def _normalize_complete_evaluation(
    value: float | CompletePathEvaluation,
    junction_path: tuple[int, ...],
    cell_path: tuple[Cell, ...],
) -> CompletePathEvaluation:
    if isinstance(value, CompletePathEvaluation):
        result = value
    else:
        result = CompletePathEvaluation(float(value), junction_path, cell_path)
    if not math.isfinite(result.time):
        raise ValueError("complete-path evaluator did not return a finite time")
    if not result.junction_path or not result.cell_path:
        raise ValueError("complete-path evaluator returned an empty selected route")
    return result


def _validate_selected_class_route(
    evaluation: CompletePathEvaluation,
    class_key: TopologyClassKey | tuple[int, ...],
    *,
    maze: Maze,
    graph: JunctionGraph,
    topology_quotient: OpenRoomTopologyQuotient | None,
) -> None:
    expanded = tuple(
        expand_junction_path(maze, graph, evaluation.junction_path)
    )
    if expanded != evaluation.cell_path:
        raise ValueError(
            "complete-path evaluator returned a cell path that does not match "
            "its junction path"
        )
    if topology_quotient is not None:
        selected_key = topology_quotient.signature(evaluation.junction_path)
        if selected_key != class_key:
            raise ValueError(
                "complete-path evaluator selected a route outside the evaluated "
                "topology quotient class"
            )


@dataclass(frozen=True, slots=True)
class _RelevantSubgraph:
    adjacency: tuple[tuple[Edge, ...], ...]
    relevant_vertices: frozenset[int]
    relevant_edges: frozenset[tuple[int, int]]
    removed_nodes: int
    removed_edges: int
    cycle_rank: int


def _undirected_edges(graph: JunctionGraph) -> tuple[tuple[int, int], ...]:
    """Compatibility wrapper for the shared graph-block utility."""
    return undirected_edges(graph)


def _biconnected_edge_components(graph: JunctionGraph) -> list[list[tuple[int, int]]]:
    """Compatibility wrapper for the shared graph-block utility."""
    return biconnected_edge_components(graph)

def _block_cut_relevant_subgraph(
    graph: JunctionGraph,
    source: int,
    target: int,
) -> _RelevantSubgraph:
    """Retain exactly the block-cut-tree path between source and target."""
    all_edges = _undirected_edges(graph)
    if source == target:
        adjacency = tuple(
            tuple(edge for edge in edges if edge.to == source)
            for edges in graph.adj
        )
        return _RelevantSubgraph(
            adjacency,
            frozenset((source,)),
            frozenset(),
            len(graph.nodes) - 1,
            len(all_edges),
            0,
        )

    components = _biconnected_edge_components(graph)
    vertex_blocks: list[list[int]] = [[] for _ in graph.nodes]
    component_vertices: list[set[int]] = []
    for block_index, edges in enumerate(components):
        vertices: set[int] = set()
        for first, second in edges:
            vertices.add(first)
            vertices.add(second)
        component_vertices.append(vertices)
        for vertex in vertices:
            vertex_blocks[vertex].append(block_index)

    articulation_vertices = [
        vertex for vertex, blocks in enumerate(vertex_blocks) if len(blocks) > 1
    ]
    articulation_node = {
        vertex: len(components) + index
        for index, vertex in enumerate(articulation_vertices)
    }
    tree_size = len(components) + len(articulation_vertices)
    tree: list[list[int]] = [[] for _ in range(tree_size)]
    for vertex in articulation_vertices:
        art_node = articulation_node[vertex]
        for block in vertex_blocks[vertex]:
            tree[art_node].append(block)
            tree[block].append(art_node)

    def representation(vertex: int) -> int:
        if vertex in articulation_node:
            return articulation_node[vertex]
        blocks = vertex_blocks[vertex]
        if not blocks:
            raise ValueError("start or goal is isolated in junction graph")
        return blocks[0]

    tree_source = representation(source)
    tree_target = representation(target)
    parent = [-1] * tree_size
    parent[tree_source] = tree_source
    queue = [tree_source]
    for node in queue:
        if node == tree_target:
            break
        for neighbor in tree[node]:
            if parent[neighbor] >= 0:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    if parent[tree_target] < 0:
        raise ValueError("goal is unreachable in block-cut tree")

    path_nodes: set[int] = set()
    node = tree_target
    while True:
        path_nodes.add(node)
        if node == tree_source:
            break
        node = parent[node]
    relevant_blocks = {node for node in path_nodes if node < len(components)}
    relevant_edges: set[tuple[int, int]] = set()
    relevant_vertices: set[int] = {source, target}
    for block in relevant_blocks:
        relevant_vertices.update(component_vertices[block])
        for first, second in components[block]:
            relevant_edges.add((min(first, second), max(first, second)))

    adjacency: list[tuple[Edge, ...]] = []
    for source_id, edges in enumerate(graph.adj):
        adjacency.append(tuple(
            edge
            for edge in edges
            if (min(source_id, edge.to), max(source_id, edge.to)) in relevant_edges
        ))
    edge_count = len(relevant_edges)
    vertex_count = len(relevant_vertices)
    cycle_rank = max(0, edge_count - vertex_count + 1)
    return _RelevantSubgraph(
        tuple(adjacency),
        frozenset(relevant_vertices),
        frozenset(relevant_edges),
        len(graph.nodes) - vertex_count,
        len(all_edges) - edge_count,
        cycle_rank,
    )


def _full_subgraph(graph: JunctionGraph) -> _RelevantSubgraph:
    edges = frozenset(_undirected_edges(graph))
    vertices = frozenset(range(len(graph.nodes)))
    return _RelevantSubgraph(
        tuple(tuple(items) for items in graph.adj),
        vertices,
        edges,
        0,
        0,
        max(0, len(edges) - len(vertices) + 1),
    )


class _ReachabilityWorkspace:
    __slots__ = ("adjacency", "target", "marks", "epoch", "stack")

    def __init__(self, adjacency: tuple[tuple[Edge, ...], ...], target: int):
        self.adjacency = adjacency
        self.target = target
        self.marks = [0] * len(adjacency)
        self.epoch = 0
        self.stack: list[int] = []

    def reachable(self, source: int, blocked_mask: int) -> bool:
        if source == self.target:
            return True
        self.epoch += 1
        if self.epoch >= (1 << 30):
            self.marks[:] = [0] * len(self.marks)
            self.epoch = 1
        epoch = self.epoch
        marks = self.marks
        stack = self.stack
        stack.clear()
        stack.append(source)
        marks[source] = epoch
        while stack:
            node = stack.pop()
            for edge in self.adjacency[node]:
                neighbor = edge.to
                if neighbor == self.target:
                    return True
                if marks[neighbor] == epoch or (blocked_mask >> neighbor) & 1:
                    continue
                marks[neighbor] = epoch
                stack.append(neighbor)
        return False


def _bound_statistics(lower_bound: TimeLowerBound) -> dict[str, int | float] | None:
    statistics = getattr(lower_bound, "statistics", None)
    if statistics is None:
        return None
    try:
        return asdict(statistics)
    except TypeError:
        return None


def branch_and_bound_junction_paths(
    maze: Maze,
    graph: JunctionGraph,
    start: Cell,
    goal: Cell,
    *,
    lower_bound: TimeLowerBound,
    complete_path_time: CompletePathEvaluator,
    init_w: float,
    terminal_w_max: float | None = None,
    terminal_goal_radius: float = math.sqrt(0.5),
    settings: SearchSettings = SearchSettings(),
    seed_junction_path: Sequence[int] | None = None,
    complete_candidate_observer: CompleteCandidateObserver | None = None,
    search_trace_observer: SearchTraceObserver | None = None,
    topology_quotient: OpenRoomTopologyQuotient | None = None,
) -> BranchAndBoundResult:
    """Best-first branch-and-bound over graph paths or quotient classes.

    When ``topology_quotient`` is supplied, internal portals of certified open
    rooms are omitted from the lower-bound relaxation and complete members are
    evaluated once per continuous class.  Graph expansion remains path-based,
    keeping the change local and preserving all nonquotient alternatives.
    """
    if topology_quotient is not None and settings.maximum_node_visits != 1:
        raise ValueError("topology quotienting requires maximum_node_visits=1")
    if seed_junction_path is None:
        seed_nodes = tuple(shortest_junction_path(graph, start, goal))
    else:
        seed_nodes = tuple(int(node) for node in seed_junction_path)
        if not seed_nodes:
            raise ValueError("seed_junction_path must not be empty")
        if seed_nodes[0] != graph.index[start] or seed_nodes[-1] != graph.index[goal]:
            raise ValueError("seed_junction_path must run from start to goal")
        for first, second in zip(seed_nodes[:-1], seed_nodes[1:]):
            if not any(edge.to == second for edge in graph.adj[first]):
                raise ValueError(f"seed_junction_path contains a non-edge: {first}->{second}")
    seed_counts: dict[int, int] = {}
    for node in seed_nodes:
        seed_counts[node] = seed_counts.get(node, 0) + 1
        if seed_counts[node] > settings.maximum_node_visits:
            raise ValueError("seed_junction_path exceeds maximum_node_visits")

    seed_cells = tuple(expand_junction_path(maze, graph, seed_nodes))
    seed_evaluation = _normalize_complete_evaluation(
        complete_path_time(seed_nodes, seed_cells),
        seed_nodes,
        seed_cells,
    )
    incumbent = seed_evaluation.time
    seed_time = incumbent
    best_nodes = seed_evaluation.junction_path
    best_cells = seed_evaluation.cell_path
    seed_class_key: TopologyClassKey | tuple[int, ...] = (
        topology_quotient.signature(seed_nodes)
        if topology_quotient is not None
        else seed_nodes
    )
    _validate_selected_class_route(
        seed_evaluation,
        seed_class_key,
        maze=maze,
        graph=graph,
        topology_quotient=topology_quotient,
    )
    if search_trace_observer is not None:
        search_trace_observer(SearchTraceEvent(
            "seed", seed_evaluation.junction_path,
            (seed_evaluation.junction_path[:-1] or None), None, incumbent,
            complete_time=seed_evaluation.time,
        ))

    source = graph.index[start]
    target = graph.index[goal]
    relevant = (
        _block_cut_relevant_subgraph(graph, source, target)
        if settings.use_block_cut_pruning and settings.maximum_node_visits == 1
        else _full_subgraph(graph)
    )

    # A cycle-free relevant subgraph has one simple start-to-goal route.  The
    # seed has already been optimized, so queue construction would only prove
    # the same fact expensively.
    if settings.maximum_node_visits == 1 and relevant.cycle_rank == 0:
        return BranchAndBoundResult(
            incumbent,
            best_nodes,
            best_cells,
            seed_time,
            0,
            0,
            0,
            0,
            0,
            incumbent,
            True,
            0,
            0,
            relevant.removed_nodes,
            relevant.removed_edges,
            relevant.cycle_rank,
            _bound_statistics(lower_bound),
            0 if topology_quotient is None else len(topology_quotient.rooms),
            1 if topology_quotient is not None else 0,
            0,
            0,
            (
                0
                if topology_quotient is None
                else topology_quotient.statistics.generated_variants
            ),
        )

    start_point = (start[0] + 0.5, start[1] + 0.5)
    goal_point = (goal[0] + 0.5, goal[1] + 0.5)
    root_cells = (start,)
    root_portal_mask: tuple[bool, ...] | None = (
        () if topology_quotient is not None else None
    )
    root_bound = lower_bound.evaluate(
        TimeBoundRequest(
            root_cells, start_point, goal_point, init_w,
            terminal_w_max=terminal_w_max, portal_mask=root_portal_mask,
            goal_radius=terminal_goal_radius,
        )
    )
    counter = itertools.count()
    root_mask = 1 << source
    heap: list[tuple[float, int, SearchNode]] = [
        (
            root_bound.time_lower_bound,
            next(counter),
            SearchNode(
                (source,), root_cells, root_bound, root_mask, root_portal_mask
            ),
        )
    ]
    if search_trace_observer is not None:
        search_trace_observer(SearchTraceEvent(
            "root", (source,), None, root_bound.time_lower_bound, incumbent
        ))

    reachability = _ReachabilityWorkspace(relevant.adjacency, target)
    expanded = generated = pruned = rejected = completed = 0
    reachability_pruned = complete_bound_pruned = 0
    quotient_leaf_reuses = quotient_classes_pruned = 0
    completed_classes: dict[
        TopologyClassKey | tuple[int, ...], CompletePathEvaluation
    ] = {seed_class_key: seed_evaluation}
    pruned_classes: set[TopologyClassKey | tuple[int, ...]] = set()

    while heap and expanded < settings.maximum_expansions:
        bound_value, _serial, node = heapq.heappop(heap)
        if bound_value >= incumbent - settings.pruning_tolerance:
            pruned += 1
            if search_trace_observer is not None:
                search_trace_observer(SearchTraceEvent(
                    "pruned_bound", node.junction_path,
                    (node.junction_path[:-1] or None), float(bound_value), incumbent,
                    prune_reason="incumbent_bound",
                ))
            continue
        expanded += 1
        if search_trace_observer is not None:
            search_trace_observer(SearchTraceEvent(
                "expanded", node.junction_path,
                (node.junction_path[:-1] or None), float(bound_value), incumbent
            ))
        current = node.junction_path[-1]
        if current == target:
            class_key: TopologyClassKey | tuple[int, ...] = (
                topology_quotient.signature(node.junction_path)
                if topology_quotient is not None
                else node.junction_path
            )
            if class_key in completed_classes or class_key in pruned_classes:
                if topology_quotient is not None:
                    quotient_leaf_reuses += 1
                continue
            complete_bound = node.bound
            if settings.use_complete_bound_refinement:
                refine = getattr(lower_bound, "refine_complete", None)
                if callable(refine):
                    complete_bound = refine(
                        TimeBoundRequest(
                            node.cell_path,
                            start_point,
                            goal_point,
                            init_w,
                            terminal_w_max=terminal_w_max, portal_mask=node.portal_mask,
                            goal_radius=terminal_goal_radius,
                        ),
                        incumbent,
                        node.bound,
                    )
                    if (
                        complete_bound.time_lower_bound
                        >= incumbent - settings.pruning_tolerance
                    ):
                        pruned += 1
                        complete_bound_pruned += 1
                        if search_trace_observer is not None:
                            search_trace_observer(SearchTraceEvent(
                                "pruned_complete_bound", node.junction_path,
                                (node.junction_path[:-1] or None),
                                complete_bound.time_lower_bound, incumbent,
                                prune_reason="complete_bound",
                            ))
                        if topology_quotient is not None:
                            quotient_classes_pruned += 1
                            pruned_classes.add(class_key)
                        continue
            if complete_candidate_observer is not None:
                complete_candidate_observer(
                    node.junction_path,
                    node.cell_path,
                    complete_bound,
                    incumbent,
                )
            evaluation = _normalize_complete_evaluation(
                complete_path_time(node.junction_path, node.cell_path),
                node.junction_path,
                node.cell_path,
            )
            _validate_selected_class_route(
                evaluation,
                class_key,
                maze=maze,
                graph=graph,
                topology_quotient=topology_quotient,
            )
            completed_classes[class_key] = evaluation
            completed += 1
            if search_trace_observer is not None:
                search_trace_observer(SearchTraceEvent(
                    "complete", evaluation.junction_path,
                    (evaluation.junction_path[:-1] or None),
                    complete_bound.time_lower_bound, incumbent,
                    complete_time=evaluation.time,
                ))
            if evaluation.time < incumbent:
                incumbent = evaluation.time
                best_nodes = evaluation.junction_path
                best_cells = evaluation.cell_path
                if search_trace_observer is not None:
                    search_trace_observer(SearchTraceEvent(
                        "incumbent_update", evaluation.junction_path,
                        (evaluation.junction_path[:-1] or None),
                        complete_bound.time_lower_bound, incumbent,
                        complete_time=evaluation.time,
                    ))
            continue

        visit_counts: dict[int, int] | None = None
        if settings.maximum_node_visits != 1:
            visit_counts = {}
            for item in node.junction_path:
                visit_counts[item] = visit_counts.get(item, 0) + 1

        for edge in relevant.adjacency[current]:
            if settings.maximum_node_visits == 1:
                if (node.visited_mask >> edge.to) & 1:
                    rejected += 1
                    if search_trace_observer is not None:
                        rejected_path = (*node.junction_path, edge.to)
                        search_trace_observer(SearchTraceEvent(
                            "rejected_visit", rejected_path, node.junction_path,
                            None, incumbent, prune_reason="visit_limit",
                        ))
                    continue
                child_mask = node.visited_mask | (1 << edge.to)
                if (
                    settings.use_no_revisit_reachability
                    and edge.to != target
                    and not reachability.reachable(
                        edge.to,
                        node.visited_mask,
                    )
                ):
                    reachability_pruned += 1
                    if search_trace_observer is not None:
                        rejected_path = (*node.junction_path, edge.to)
                        search_trace_observer(SearchTraceEvent(
                            "pruned_reachability", rejected_path, node.junction_path,
                            None, incumbent, prune_reason="no_revisit_reachability",
                        ))
                    continue
            else:
                assert visit_counts is not None
                if visit_counts.get(edge.to, 0) >= settings.maximum_node_visits:
                    rejected += 1
                    if search_trace_observer is not None:
                        rejected_path = (*node.junction_path, edge.to)
                        search_trace_observer(SearchTraceEvent(
                            "rejected_visit", rejected_path, node.junction_path,
                            None, incumbent, prune_reason="visit_limit",
                        ))
                    continue
                child_mask = node.visited_mask | (1 << edge.to)

            child_nodes = (*node.junction_path, edge.to)
            child_cells = (
                *node.cell_path,
                *expand_junction_edge(maze, graph, current, edge.to),
            )
            child_portal_mask = (
                topology_quotient.portal_mask(child_nodes)
                if topology_quotient is not None
                else None
            )
            result = lower_bound.evaluate(
                TimeBoundRequest(
                    child_cells,
                    start_point,
                    goal_point,
                    init_w,
                    terminal_w_max=terminal_w_max, portal_mask=child_portal_mask,
                    goal_radius=terminal_goal_radius,
                )
            )
            generated += 1
            if search_trace_observer is not None:
                search_trace_observer(SearchTraceEvent(
                    "generated", child_nodes, node.junction_path,
                    result.time_lower_bound, incumbent
                ))
            if result.time_lower_bound >= incumbent - settings.pruning_tolerance:
                pruned += 1
                if search_trace_observer is not None:
                    search_trace_observer(SearchTraceEvent(
                        "pruned_bound", child_nodes, node.junction_path,
                        result.time_lower_bound, incumbent, prune_reason="incumbent_bound",
                    ))
                continue
            heapq.heappush(
                heap,
                (
                    result.time_lower_bound,
                    next(counter),
                    SearchNode(
                        child_nodes, child_cells, result, child_mask,
                        child_portal_mask,
                    ),
                ),
            )

    minimum_open = heap[0][0] if heap else incumbent
    return BranchAndBoundResult(
        incumbent,
        best_nodes,
        best_cells,
        seed_time,
        expanded,
        generated,
        pruned,
        rejected,
        completed,
        minimum_open,
        not heap,
        reachability_pruned,
        complete_bound_pruned,
        relevant.removed_nodes,
        relevant.removed_edges,
        relevant.cycle_rank,
        _bound_statistics(lower_bound),
        0 if topology_quotient is None else len(topology_quotient.rooms),
        (len(completed_classes) if topology_quotient is not None else 0),
        quotient_leaf_reuses,
        quotient_classes_pruned,
        (
            0
            if topology_quotient is None
            else topology_quotient.statistics.generated_variants
        ),
    )


__all__ = [
    "BranchAndBoundResult",
    "CompleteCandidateObserver",
    "CompletePathEvaluation",
    "SearchNode",
    "SearchSettings",
    "SearchTraceEvent",
    "SearchTraceObserver",
    "branch_and_bound_junction_paths",
]
