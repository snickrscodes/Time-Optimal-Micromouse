from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


Cell = Tuple[int, int]
Wall = Tuple[Cell, Cell]

# Direction order: up, right, down, left.
DIRS: Tuple[Cell, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)

UP, RIGHT, DOWN, LEFT = range(4)

OPPOSITE: Tuple[int, ...] = (
    DOWN,
    LEFT,
    UP,
    RIGHT,
)

DIR_BITS: Tuple[int, ...] = tuple(
    1 << direction
    for direction in range(4)
)

DELTA_TO_DIR = {
    delta: direction
    for direction, delta in enumerate(DIRS)
}

BIT_TO_DIR = {
    bit: direction
    for direction, bit in enumerate(DIR_BITS)
}

VERTICAL_MASK = DIR_BITS[UP] | DIR_BITS[DOWN]
HORIZONTAL_MASK = DIR_BITS[RIGHT] | DIR_BITS[LEFT]
ALL_DIRECTIONS_MASK = sum(DIR_BITS)


@dataclass(frozen=True, slots=True)
class Edge:
    to: int
    direction: int
    length: int


@dataclass(slots=True)
class JunctionGraph:
    nodes: List[Cell]
    index: Dict[Cell, int]
    adj: List[List[Edge]]


class Maze:
    """
    Grid maze backed by four-bit connection masks.

    Each cell uses one bit per direction:

        bit 0: up
        bit 1: right
        bit 2: down
        bit 3: left

    Goal distances and wall segments are computed lazily and cached.
    """

    __slots__ = (
        "width",
        "height",
        "goal",
        "_masks",
        "_index_deltas",
        "_dist_from_goal",
        "_walls",
    )

    def __init__(
        self,
        width: int,
        height: int,
        connection_masks: Sequence[int],
        goal: Cell,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        if len(connection_masks) != width * height:
            raise ValueError(
                "connection_masks must contain width * height entries"
            )

        if not (
            0 <= goal[0] < width
            and 0 <= goal[1] < height
        ):
            raise ValueError("goal is outside the maze")

        self.width = width
        self.height = height
        self.goal = goal

        # Masks are immutable after construction, which keeps cached derived
        # data such as distances and walls valid.
        self._masks = bytes(connection_masks)

        # Index changes corresponding to DIRS.
        self._index_deltas = (
            -width,
            1,
            width,
            -1,
        )

        self._dist_from_goal: Optional[Dict[Cell, int]] = None
        self._walls: Optional[List[Wall]] = None

    @classmethod
    def from_connections(
        cls,
        width: int,
        height: int,
        connections: Mapping[Cell, Set[Cell]],
        goal: Cell,
    ) -> Maze:
        """
        Convert the old dictionary-of-sets representation into masks.

        Connections are made symmetric even if only one direction appears
        in the supplied mapping.
        """
        masks = bytearray(width * height)

        for cell, neighbors in connections.items():
            x, y = cell

            if not (
                0 <= x < width
                and 0 <= y < height
            ):
                raise ValueError(f"cell outside maze: {cell}")

            cell_index = y * width + x

            for neighbor in neighbors:
                nx, ny = neighbor

                if not (
                    0 <= nx < width
                    and 0 <= ny < height
                ):
                    raise ValueError(
                        f"neighbor outside maze: {neighbor}"
                    )

                direction = DELTA_TO_DIR.get(
                    (nx - x, ny - y)
                )

                if direction is None:
                    raise ValueError(
                        f"non-adjacent connection: "
                        f"{cell} -> {neighbor}"
                    )

                neighbor_index = ny * width + nx

                masks[cell_index] |= DIR_BITS[direction]
                masks[neighbor_index] |= DIR_BITS[
                    OPPOSITE[direction]
                ]

        return cls(
            width=width,
            height=height,
            connection_masks=masks,
            goal=goal,
        )

    @property
    def dist_from_goal(self) -> Dict[Cell, int]:
        """Shortest cell distance from the goal, computed on first access."""
        distances = self._dist_from_goal

        if distances is None:
            distances = self._bfs_from(self.goal)
            self._dist_from_goal = distances

        return distances

    @property
    def walls(self) -> List[Wall]:
        """Maze wall segments, computed on first access."""
        walls = self._walls

        if walls is None:
            walls = self.gen_walls()
            self._walls = walls

        return walls

    def _index(self, cell: Cell) -> int:
        x, y = cell

        if not (
            0 <= x < self.width
            and 0 <= y < self.height
        ):
            raise IndexError(f"cell outside maze: {cell}")

        return y * self.width + x

    def _cell(self, index: int) -> Cell:
        return (
            index % self.width,
            index // self.width,
        )

    def _neighbor_indices(
        self,
        index: int,
    ) -> Iterator[int]:
        mask = self._masks[index]

        for direction, bit in enumerate(DIR_BITS):
            if mask & bit:
                yield index + self._index_deltas[direction]

    def neighbors(
        self,
        cell: Cell,
    ) -> Iterator[Cell]:
        index = self._index(cell)

        for neighbor_index in self._neighbor_indices(index):
            yield self._cell(neighbor_index)

    def connection_dict(
        self,
    ) -> Dict[Cell, Set[Cell]]:
        """
        Materialize the legacy dictionary-of-sets representation.

        This is intended for interoperability. Internally, masks are much
        more compact and faster.
        """
        connections: Dict[Cell, Set[Cell]] = {}

        for index in range(self.width * self.height):
            cell = self._cell(index)
            connections[cell] = set(self.neighbors(cell))

        return connections

    def _bfs_from(
        self,
        source: Cell,
    ) -> Dict[Cell, int]:
        source_index = self._index(source)
        cell_count = self.width * self.height

        distances = [-1] * cell_count
        distances[source_index] = 0

        queue = deque([source_index])

        while queue:
            current = queue.popleft()
            next_distance = distances[current] + 1

            for neighbor in self._neighbor_indices(current):
                if distances[neighbor] != -1:
                    continue

                distances[neighbor] = next_distance
                queue.append(neighbor)

        return {
            self._cell(index): distance
            for index, distance in enumerate(distances)
            if distance >= 0
        }

    def gen_walls(self) -> List[Wall]:
        width = self.width
        height = self.height

        segments: List[Wall] = [
            ((0, 0), (width, 0)),
            ((0, height), (width, height)),
            ((0, 0), (0, height)),
            ((width, 0), (width, height)),
        ]

        for y in range(height):
            row_start = y * width

            for x in range(width):
                mask = self._masks[row_start + x]

                if (
                    x + 1 < width
                    and not mask & DIR_BITS[RIGHT]
                ):
                    segments.append(
                        (
                            (x + 1, y),
                            (x + 1, y + 1),
                        )
                    )

                if (
                    y + 1 < height
                    and not mask & DIR_BITS[DOWN]
                ):
                    segments.append(
                        (
                            (x, y + 1),
                            (x + 1, y + 1),
                        )
                    )

        return segments

    def is_path_between(
        self,
        a: Cell,
        b: Cell,
    ) -> bool:
        direction = DELTA_TO_DIR.get(
            (
                b[0] - a[0],
                b[1] - a[1],
            )
        )

        if direction is None:
            return False

        try:
            index = self._index(a)
        except IndexError:
            return False

        return bool(
            self._masks[index]
            & DIR_BITS[direction]
        )

    def is_junction(
        self,
        cell: Cell,
        start: Cell,
        goal: Cell,
    ) -> bool:
        if cell == start or cell == goal:
            return True

        mask = self._masks[self._index(cell)]

        # A degree-two cell is a corridor only when its passages
        # are directly opposite each other.
        return (
            mask.bit_count() != 2
            or mask not in (
                VERTICAL_MASK,
                HORIZONTAL_MASK,
            )
        )

    def __str__(self) -> str:
        lines = [
            "+" + "---+" * self.width
        ]

        for y in range(self.height):
            row = ["|"]
            floor = ["+"]
            row_start = y * self.width

            for x in range(self.width):
                mask = self._masks[row_start + x]

                row.append("   ")
                row.append(
                    " "
                    if mask & DIR_BITS[RIGHT]
                    else "|"
                )

                floor.append(
                    "   +"
                    if mask & DIR_BITS[DOWN]
                    else "---+"
                )

            lines.append("".join(row))
            lines.append("".join(floor))

        return "\n".join(lines)


class MazeGenerator:
    """
    Wilson's loop-erased random-walk maze generator.

    Internally uses integer cell indexes, compact valid-direction masks,
    O(1) loop detection, and O(1) unvisited-cell removal.
    """

    __slots__ = (
        "width",
        "height",
        "rng",
        "_valid_directions",
    )

    def __init__(
        self,
        width: int,
        height: int,
        seed: Optional[int] = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive"
            )

        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        cell_count = width * height
        valid_directions = bytearray(
            [ALL_DIRECTIONS_MASK]
        ) * cell_count

        without_up = ALL_DIRECTIONS_MASK ^ DIR_BITS[UP]
        without_right = ALL_DIRECTIONS_MASK ^ DIR_BITS[RIGHT]
        without_down = ALL_DIRECTIONS_MASK ^ DIR_BITS[DOWN]
        without_left = ALL_DIRECTIONS_MASK ^ DIR_BITS[LEFT]

        for x in range(width):
            valid_directions[x] &= without_up
            valid_directions[(height - 1) * width + x] &= without_down

        for y in range(height):
            row_start = y * width
            valid_directions[row_start] &= without_left
            valid_directions[row_start + width - 1] &= without_right

        self._valid_directions = bytes(valid_directions)

    def generate(self) -> Maze:
        width = self.width
        height = self.height
        cell_count = width * height

        # Four-bit connection mask per cell.
        connection_masks = bytearray(cell_count)

        # Dense indexed set:
        #
        #     unvisited[position[cell]] == cell
        #
        # position[cell] becomes -1 once the cell has joined the maze.
        unvisited = list(range(cell_count))
        position = list(range(cell_count))

        def remove_unvisited(cell: int) -> None:
            remove_at = position[cell]

            if remove_at < 0:
                return

            last = unvisited.pop()

            if remove_at < len(unvisited):
                unvisited[remove_at] = last
                position[last] = remove_at

            position[cell] = -1

        # Wilson's algorithm starts with one arbitrary root cell.
        root = self.rng.choice(unvisited)
        remove_unvisited(root)

        # Entries are not cleared between walks. A stored position is valid
        # only when it still points to that cell in the current path.
        path_position = [-1] * cell_count

        choice = self.rng.choice
        getrandbits = self.rng.getrandbits
        valid_directions = self._valid_directions
        index_deltas = (
            -width,
            1,
            width,
            -1,
        )

        while unvisited:
            start = choice(unvisited)
            path = [start]
            path_position[start] = 0

            # Walk until reaching a cell already incorporated into the maze.
            while position[path[-1]] >= 0:
                current = path[-1]
                valid_mask = valid_directions[current]

                # Two random bits choose a direction. Boundary directions are
                # rejected; among valid directions the result stays uniform.
                while True:
                    direction = getrandbits(2)

                    if valid_mask & DIR_BITS[direction]:
                        break

                next_cell = current + index_deltas[direction]
                loop_at = path_position[next_cell]

                if (
                    0 <= loop_at < len(path)
                    and path[loop_at] == next_cell
                ):
                    # C-level slice deletion erases the loop. Stale entries in
                    # path_position are harmless because they are validated.
                    del path[loop_at + 1:]
                else:
                    path_position[next_cell] = len(path)
                    path.append(next_cell)

            # Connect the loop-erased path to the existing maze. Directions
            # are derived here, avoiding one append per random-walk step.
            for path_index in range(len(path) - 1):
                u = path[path_index]
                v = path[path_index + 1]
                delta = v - u

                # Vertical checks must come first when width == 1.
                if delta == -width:
                    direction = UP
                elif delta == width:
                    direction = DOWN
                elif delta == 1:
                    direction = RIGHT
                elif delta == -1:
                    direction = LEFT
                else:
                    raise RuntimeError(
                        "Wilson path contains non-adjacent cells"
                    )

                connection_masks[u] |= DIR_BITS[direction]
                connection_masks[v] |= DIR_BITS[
                    OPPOSITE[direction]
                ]

                remove_unvisited(u)

        goal = (
            width // 2,
            height // 2,
        )

        return Maze(
            width=width,
            height=height,
            connection_masks=connection_masks,
            goal=goal,
        )


def compress_to_graph(
    maze: Maze,
    start: Cell,
    goal: Cell,
) -> JunctionGraph:
    """
    Compress straight maze corridors into weighted graph edges.

    Every physical corridor is traversed only once. Both directed graph
    edges are emitted during that traversal.
    """
    start_index = maze._index(start)
    goal_index = maze._index(goal)
    cell_count = maze.width * maze.height

    nodes: List[Cell] = []
    index: Dict[Cell, int] = {}

    # Maps every maze-cell index to its junction node ID.
    # Non-junction cells remain -1.
    junction_id = [-1] * cell_count

    for cell_index, mask in enumerate(maze._masks):
        is_junction = (
            cell_index == start_index
            or cell_index == goal_index
            or mask.bit_count() != 2
            or mask not in (
                VERTICAL_MASK,
                HORIZONTAL_MASK,
            )
        )

        if not is_junction:
            continue

        cell = maze._cell(cell_index)
        node_id = len(nodes)

        nodes.append(cell)
        index[cell] = node_id
        junction_id[cell_index] = node_id

    adj: List[List[Edge]] = [
        []
        for _ in nodes
    ]

    # Four-bit mask per junction. Each bit marks an entrance that
    # has already been followed.
    visited_directions = bytearray(len(nodes))

    for source_id, source_cell in enumerate(nodes):
        source_index = maze._index(source_cell)
        source_mask = maze._masks[source_index]

        for direction, direction_bit in enumerate(DIR_BITS):
            if not source_mask & direction_bit:
                continue

            if visited_directions[source_id] & direction_bit:
                continue

            current = (
                source_index
                + maze._index_deltas[direction]
            )

            travel_direction = direction
            length = 1

            while junction_id[current] < 0:
                corridor_mask = maze._masks[current]

                # Remove the direction pointing back to the cell
                # we came from. The remaining bit is the direction
                # forward through the corridor.
                back_bit = DIR_BITS[
                    OPPOSITE[travel_direction]
                ]

                forward_bit = corridor_mask & ~back_bit
                travel_direction = BIT_TO_DIR[forward_bit]

                current += maze._index_deltas[
                    travel_direction
                ]

                length += 1

            destination_id = junction_id[current]

            # Direction leaving the destination junction back
            # through the same corridor.
            reverse_direction = OPPOSITE[
                travel_direction
            ]

            adj[source_id].append(
                Edge(
                    to=destination_id,
                    direction=direction,
                    length=length,
                )
            )

            adj[destination_id].append(
                Edge(
                    to=source_id,
                    direction=reverse_direction,
                    length=length,
                )
            )

            visited_directions[source_id] |= direction_bit
            visited_directions[destination_id] |= DIR_BITS[
                reverse_direction
            ]

    return JunctionGraph(
        nodes=nodes,
        index=index,
        adj=adj,
    )
