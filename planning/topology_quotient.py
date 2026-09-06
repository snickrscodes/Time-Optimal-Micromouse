"""Conservative production topology quotient for fully open 2x2 rooms.

The discrete maze graph may contain two simple paths around an open 2x2 block
although the corresponding continuous free-space region is one contractible
room.  This module identifies only that rigorously simple case, collapses it in
class signatures and lower-bound portal sequences, and supplies every member
initializer together with one wall-aware union corridor.

The quotient is deliberately conservative:

* only simple-path search (maximum visit count one) is supported;
* a room must be exactly 2x2 cells with all four internal adjacencies open;
* its graph block must have cycle rank one; and
* the oriented rectangle's circumscribed radius plus clearance must fit inside
  half a cell, providing an all-orientation refuge in every room cell.

Anything ambiguous remains an ordinary graph path.  Failing to quotient can
cost performance but cannot change the admitted route set.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence, TypeAlias

from mazegen import Cell, JunctionGraph, Maze

from .graph_blocks import EdgeKey, biconnected_edge_components
from .maze_routes import OpenRoomSpan, expand_junction_edge, expand_junction_path

ClassToken: TypeAlias = tuple[object, ...]
TopologyClassKey: TypeAlias = tuple[ClassToken, ...]


@dataclass(frozen=True, slots=True)
class OpenRoomBlock:
    index: int
    vertices: frozenset[int]
    edges: frozenset[EdgeKey]
    cells: tuple[Cell, ...]
    xmin: int
    xmax: int
    ymin: int
    ymax: int


@dataclass(frozen=True, slots=True)
class QuotientRouteVariant:
    junction_path: tuple[int, ...]
    cell_path: tuple[Cell, ...]
    open_room_spans: tuple[OpenRoomSpan, ...]


@dataclass(frozen=True, slots=True)
class QuotientRouteClass:
    key: TopologyClassKey
    variants: tuple[QuotientRouteVariant, ...]
    quotient_room_indices: tuple[int, ...]

    @property
    def representative(self) -> QuotientRouteVariant:
        return self.variants[0]


@dataclass(slots=True)
class TopologyQuotientStatistics:
    detected_rooms: int = 0
    signature_evaluations: int = 0
    signature_cache_hits: int = 0
    class_expansions: int = 0
    class_cache_hits: int = 0
    generated_variants: int = 0
    portal_mask_evaluations: int = 0
    portal_mask_cache_hits: int = 0
    edge_expansion_cache_hits: int = 0
    edge_expansion_cache_misses: int = 0


@dataclass(slots=True)
class OpenRoomTopologyQuotient:
    """Production-safe quotient of graph paths through open 2x2 rooms."""

    maze: Maze
    graph: JunctionGraph
    body_length: float
    body_height: float
    clearance: float = 0.0
    maximum_class_variants: int = 16
    statistics: TopologyQuotientStatistics = field(
        default_factory=TopologyQuotientStatistics, init=False
    )
    rooms: tuple[OpenRoomBlock, ...] = field(init=False)
    _edge_room: dict[EdgeKey, int] = field(default_factory=dict, init=False, repr=False)
    _signature_cache: dict[tuple[int, ...], TopologyClassKey] = field(
        default_factory=dict, init=False, repr=False
    )
    _class_cache: dict[TopologyClassKey, QuotientRouteClass] = field(
        default_factory=dict, init=False, repr=False
    )
    _local_paths: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _edge_expansions: dict[tuple[int, int], tuple[Cell, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _portal_mask_cache: dict[tuple[int, ...], tuple[bool, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("body_length", self.body_length),
            ("body_height", self.body_height),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.clearance) or self.clearance < 0.0:
            raise ValueError("clearance must be finite and nonnegative")
        if self.maximum_class_variants <= 0:
            raise ValueError("maximum_class_variants must be positive")
        self.rooms = self._detect_rooms()
        for room in self.rooms:
            for edge in room.edges:
                self._edge_room[edge] = room.index
        self.statistics.detected_rooms = len(self.rooms)

    @staticmethod
    def _edge_key(first: int, second: int) -> EdgeKey:
        return (first, second) if first < second else (second, first)

    @property
    def active(self) -> bool:
        return bool(self.rooms)

    def room_for_edge(self, first: int, second: int) -> int | None:
        return self._edge_room.get(self._edge_key(first, second))

    def _expanded_edge(self, first: int, second: int) -> tuple[Cell, ...]:
        key = (first, second)
        cached = self._edge_expansions.get(key)
        if cached is not None:
            self.statistics.edge_expansion_cache_hits += 1
            return cached
        self.statistics.edge_expansion_cache_misses += 1
        expanded = expand_junction_edge(self.maze, self.graph, first, second)
        self._edge_expansions[key] = expanded
        self._edge_expansions[(second, first)] = tuple(reversed((self.graph.nodes[first], *expanded[:-1])))
        return expanded

    def _expanded_block_cells(self, edges: Iterable[EdgeKey]) -> frozenset[Cell]:
        cells: set[Cell] = set()
        for first, second in edges:
            cells.add(self.graph.nodes[first])
            cells.add(self.graph.nodes[second])
            cells.update(self._expanded_edge(first, second))
        return frozenset(cells)

    def _all_orientation_refuge(self) -> bool:
        radius = 0.5 * math.hypot(self.body_length, self.body_height) + self.clearance
        guard = 256.0 * math.ulp(max(1.0, radius)) + 1.0e-14
        return radius <= math.nextafter(0.5 - guard, 0.0)

    def _detect_rooms(self) -> tuple[OpenRoomBlock, ...]:
        if not self._all_orientation_refuge():
            return ()
        rooms: list[OpenRoomBlock] = []
        for component in biconnected_edge_components(self.graph):
            edges = frozenset(self._edge_key(*edge) for edge in component)
            vertices = frozenset(vertex for edge in edges for vertex in edge)
            if len(edges) - len(vertices) + 1 != 1:
                continue
            cells = self._expanded_block_cells(edges)
            if len(cells) != 4:
                continue
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmax - xmin != 1 or ymax - ymin != 1:
                continue
            expected = {
                (x, y)
                for x in range(xmin, xmax + 1)
                for y in range(ymin, ymax + 1)
            }
            if cells != expected:
                continue
            fully_open = all(
                self.maze.is_path_between((x, y), (x + 1, y))
                for x in range(xmin, xmax)
                for y in range(ymin, ymax + 1)
            ) and all(
                self.maze.is_path_between((x, y), (x, y + 1))
                for x in range(xmin, xmax + 1)
                for y in range(ymin, ymax)
            )
            if not fully_open:
                continue
            room_index = len(rooms)
            rooms.append(
                OpenRoomBlock(
                    room_index,
                    vertices,
                    edges,
                    tuple(sorted(cells)),
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )
            )
        return tuple(rooms)

    def _local_room_paths(
        self, room_index: int, entry: int, exit: int
    ) -> tuple[tuple[int, ...], ...]:
        key = (room_index, entry, exit)
        cached = self._local_paths.get(key)
        if cached is not None:
            return cached
        room = self.rooms[room_index]
        if entry not in room.vertices or exit not in room.vertices or entry == exit:
            raise ValueError("invalid room traversal terminals")
        adjacency: dict[int, list[int]] = {vertex: [] for vertex in room.vertices}
        for first, second in room.edges:
            adjacency[first].append(second)
            adjacency[second].append(first)
        for neighbors in adjacency.values():
            neighbors.sort()
        paths: list[tuple[int, ...]] = []
        stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
            (entry, (entry,), frozenset((entry,)))
        ]
        while stack:
            node, path, visited = stack.pop()
            if node == exit:
                paths.append(path)
                continue
            for neighbor in reversed(adjacency[node]):
                if neighbor in visited:
                    continue
                stack.append((neighbor, (*path, neighbor), visited | {neighbor}))
        cached = tuple(sorted(paths, key=lambda item: (len(item), item)))
        if len(cached) != 2:
            # A certified 2x2 cycle must have exactly two simple terminal paths.
            # Treating any unexpected block as nonquotient is safer than trying
            # to infer a broader equivalence class.
            cached = ()
        self._local_paths[key] = cached
        return cached

    def signature(self, path: Sequence[int]) -> TopologyClassKey:
        frozen = tuple(int(node) for node in path)
        self.statistics.signature_evaluations += 1
        cached = self._signature_cache.get(frozen)
        if cached is not None:
            self.statistics.signature_cache_hits += 1
            return cached
        if len(frozen) < 2 or not self.rooms:
            result: TopologyClassKey = (("PATH", frozen),)
            self._signature_cache[frozen] = result
            return result

        tokens: list[ClassToken] = []
        normal_start = 0
        index = 0
        variant_count = 1
        while index + 1 < len(frozen):
            room_index = self.room_for_edge(frozen[index], frozen[index + 1])
            if room_index is None:
                index += 1
                continue
            end = index + 1
            while end + 1 < len(frozen):
                next_room = self.room_for_edge(frozen[end], frozen[end + 1])
                if next_room != room_index:
                    break
                end += 1
            alternatives = self._local_room_paths(
                room_index, frozen[index], frozen[end]
            )
            previous_room = (
                self.room_for_edge(frozen[index - 1], frozen[index])
                if index > 0
                else None
            )
            next_room = (
                self.room_for_edge(frozen[end], frozen[end + 1])
                if end + 1 < len(frozen)
                else None
            )
            can_collapse = (
                bool(alternatives)
                and index > 0
                and end < len(frozen) - 1
                # Adjacent room spans share an articulation cell.  The current
                # exact initializer splitter intentionally supports disjoint
                # room intervals only, so leave this rarer compound case
                # unquotiented rather than introducing a configuration-space
                # assumption at the shared cell.
                and previous_room is None
                and next_room is None
                and variant_count * len(alternatives) <= self.maximum_class_variants
            )
            if can_collapse:
                if normal_start < index:
                    tokens.append(("PATH", frozen[normal_start:index + 1]))
                tokens.append(("ROOM", room_index, frozen[index], frozen[end]))
                variant_count *= len(alternatives)
                normal_start = end
            index = end
        if normal_start < len(frozen) - 1:
            tokens.append(("PATH", frozen[normal_start:]))
        elif not tokens:
            tokens.append(("PATH", frozen))
        result = tuple(tokens)
        self._signature_cache[frozen] = result
        return result

    def _assemble_variant_paths(
        self, signature: TopologyClassKey
    ) -> tuple[tuple[int, ...], ...]:
        options: list[tuple[tuple[int, ...], ...]] = []
        for token in signature:
            if token[0] == "PATH":
                options.append((tuple(token[1]),))
            else:
                room_index, entry, exit = map(int, token[1:])
                local = self._local_room_paths(room_index, entry, exit)
                if not local:
                    raise RuntimeError("quotient signature references an invalid room")
                options.append(local)
        assembled: list[tuple[int, ...]] = []
        for pieces in itertools.product(*options):
            path: list[int] = []
            for piece in pieces:
                if not path:
                    path.extend(piece)
                elif path[-1] == piece[0]:
                    path.extend(piece[1:])
                else:
                    raise RuntimeError("quotient signature pieces are disconnected")
            assembled.append(tuple(path))
        return tuple(sorted(set(assembled)))

    def _room_spans(self, path: tuple[int, ...]) -> tuple[OpenRoomSpan, ...]:
        spans: list[OpenRoomSpan] = []
        cell_index = 0
        edge_index = 0
        while edge_index + 1 < len(path):
            room_index = self.room_for_edge(path[edge_index], path[edge_index + 1])
            expanded = self._expanded_edge(path[edge_index], path[edge_index + 1])
            if room_index is None:
                cell_index += len(expanded)
                edge_index += 1
                continue
            start_cell = cell_index
            current_room = room_index
            while edge_index + 1 < len(path):
                found = self.room_for_edge(path[edge_index], path[edge_index + 1])
                if found != current_room:
                    break
                expanded = self._expanded_edge(path[edge_index], path[edge_index + 1])
                cell_index += len(expanded)
                edge_index += 1
            room = self.rooms[current_room]
            spans.append(
                OpenRoomSpan(
                    start_cell,
                    cell_index,
                    room.cells,
                    room.xmin,
                    room.xmax,
                    room.ymin,
                    room.ymax,
                    room.index,
                )
            )
        return tuple(spans)

    def route_class(self, path: Sequence[int]) -> QuotientRouteClass:
        signature = self.signature(path)
        cached = self._class_cache.get(signature)
        if cached is not None:
            self.statistics.class_cache_hits += 1
            return cached
        self.statistics.class_expansions += 1
        variants: list[QuotientRouteVariant] = []
        for member in self._assemble_variant_paths(signature):
            cells = tuple(expand_junction_path(self.maze, self.graph, member))
            variants.append(
                QuotientRouteVariant(member, cells, self._room_spans(member))
            )
        room_indices = tuple(
            int(token[1]) for token in signature if token[0] == "ROOM"
        )
        result = QuotientRouteClass(signature, tuple(variants), room_indices)
        self.statistics.generated_variants += len(variants)
        self._class_cache[signature] = result
        return result

    def portal_mask(self, path: Sequence[int]) -> tuple[bool, ...]:
        """Mask member portals that remain mandatory for the quotient class.

        Every transition internal to a collapsed room is omitted.  All outside
        portals are retained, so the resulting ordered-portal problem is a
        relaxation of the full room-union class rather than of one member path.
        """
        frozen = tuple(int(node) for node in path)
        self.statistics.portal_mask_evaluations += 1
        cached = self._portal_mask_cache.get(frozen)
        if cached is not None:
            self.statistics.portal_mask_cache_hits += 1
            return cached
        signature = self.signature(frozen)
        collapsed = {
            int(token[1]) for token in signature if token[0] == "ROOM"
        }
        mask: list[bool] = []
        for first, second in zip(frozen[:-1], frozen[1:]):
            room_index = self.room_for_edge(first, second)
            keep = room_index not in collapsed
            mask.extend((keep,) * len(self._expanded_edge(first, second)))
        result = tuple(mask)
        self._portal_mask_cache[frozen] = result
        return result


__all__ = [
    "OpenRoomBlock",
    "OpenRoomTopologyQuotient",
    "QuotientRouteClass",
    "QuotientRouteVariant",
    "TopologyClassKey",
    "TopologyQuotientStatistics",
]
