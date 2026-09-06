"""Project-native JSON maze loader.

Wall matrices are written in visual north-to-south order so a maze can be
transcribed directly from an image.  Cell coordinates in ``start``/``goals``
are normalized independently according to ``coordinates.origin``.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from mazegen import Cell, Maze

from .maze_scenario import GoalEntrance, GoalRegion, MazeScale, MazeScenario

FORMAT_ID = "ame-maze-v1"
_ALLOWED_ORIGINS = {"northwest", "southwest"}
_ALLOWED_HEADINGS = {"north", "east", "south", "west"}

_HEADING_INTERNAL_DELTA = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}


class MazeFormatError(ValueError):
    pass


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MazeFormatError(f"{where} must be an object")
    return value


def _require_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MazeFormatError(f"{where} must be an integer")
    return int(value)


def _external_cell(value: Any, *, width: int, height: int, origin: str, where: str) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MazeFormatError(f"{where} must be [x, y]")
    x = _require_int(value[0], f"{where}[0]")
    y = _require_int(value[1], f"{where}[1]")
    if not (0 <= x < width and 0 <= y < height):
        raise MazeFormatError(f"{where} is outside the maze: {(x, y)}")
    if origin == "southwest":
        y = height - 1 - y
    return x, y


def _wall_rows(value: Any, *, count: int, width: int, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise MazeFormatError(f"{where} must contain exactly {count} rows")
    rows: list[str] = []
    for index, row in enumerate(value):
        if not isinstance(row, str) or len(row) != width:
            raise MazeFormatError(f"{where}[{index}] must be a {width}-character string")
        if any(ch not in "01" for ch in row):
            raise MazeFormatError(f"{where}[{index}] may contain only '0' and '1'")
        rows.append(row)
    return tuple(rows)


def _build_connections(width: int, height: int, horizontal: tuple[str, ...], vertical: tuple[str, ...]) -> dict[Cell, set[Cell]]:
    if any(ch != "1" for ch in horizontal[0]) or any(ch != "1" for ch in horizontal[-1]):
        raise MazeFormatError("north and south exterior boundaries must be walls")
    if any(row[0] != "1" or row[-1] != "1" for row in vertical):
        raise MazeFormatError("west and east exterior boundaries must be walls")

    connections: dict[Cell, set[Cell]] = {(x, y): set() for y in range(height) for x in range(width)}
    for y in range(height):
        for x in range(width):
            here = (x, y)
            if x + 1 < width and vertical[y][x + 1] == "0":
                other = (x + 1, y)
                connections[here].add(other); connections[other].add(here)
            if y + 1 < height and horizontal[y + 1][x] == "0":
                other = (x, y + 1)
                connections[here].add(other); connections[other].add(here)
    return connections


def _goal_entrances(goals: set[Cell], connections: Mapping[Cell, set[Cell]]) -> tuple[GoalEntrance, ...]:
    entries: list[GoalEntrance] = []
    for inside in sorted(goals):
        for outside in sorted(connections[inside]):
            if outside not in goals:
                entries.append(GoalEntrance(outside, inside))
    return tuple(entries)


def _goal_region_connected(goals: set[Cell], connections: Mapping[Cell, set[Cell]]) -> bool:
    if not goals:
        return False
    first = next(iter(goals)); seen = {first}; queue = deque([first])
    while queue:
        cell = queue.popleft()
        for neighbor in connections[cell]:
            if neighbor in goals and neighbor not in seen:
                seen.add(neighbor); queue.append(neighbor)
    return seen == goals


def _reachable(start: Cell, connections: Mapping[Cell, set[Cell]]) -> set[Cell]:
    seen = {start}; queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in connections[cell]:
            if neighbor not in seen:
                seen.add(neighbor); queue.append(neighbor)
    return seen


def load_maze_scenario(path: str | Path) -> MazeScenario:
    source = Path(path)
    raw_bytes = source.read_bytes()
    try:
        document = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MazeFormatError(f"invalid UTF-8 JSON maze: {exc}") from exc
    root = _require_mapping(document, "root")
    if root.get("format") != FORMAT_ID:
        raise MazeFormatError(f"format must be {FORMAT_ID!r}")

    name = root.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MazeFormatError("name must be a nonempty string")
    dims = _require_mapping(root.get("dimensions"), "dimensions")
    width = _require_int(dims.get("width"), "dimensions.width")
    height = _require_int(dims.get("height"), "dimensions.height")
    if width <= 0 or height <= 0:
        raise MazeFormatError("maze dimensions must be positive")

    coords = _require_mapping(root.get("coordinates", {}), "coordinates")
    origin = coords.get("origin", "southwest")
    if origin not in _ALLOWED_ORIGINS:
        raise MazeFormatError(f"coordinates.origin must be one of {sorted(_ALLOWED_ORIGINS)}")

    walls = _require_mapping(root.get("walls"), "walls")
    if walls.get("row_order", "north_to_south") != "north_to_south":
        raise MazeFormatError("walls.row_order currently must be 'north_to_south'")
    horizontal = _wall_rows(walls.get("horizontal"), count=height + 1, width=width, where="walls.horizontal")
    vertical = _wall_rows(walls.get("vertical"), count=height, width=width + 1, where="walls.vertical")
    connections = _build_connections(width, height, horizontal, vertical)

    start_obj = _require_mapping(root.get("start"), "start")
    start = _external_cell(start_obj.get("cell"), width=width, height=height, origin=origin, where="start.cell")
    heading = start_obj.get("heading")
    if heading is not None:
        if not isinstance(heading, str) or heading.lower() not in _ALLOWED_HEADINGS:
            raise MazeFormatError(f"start.heading must be one of {sorted(_ALLOWED_HEADINGS)}")
        heading = heading.lower()
        dx, dy = _HEADING_INTERNAL_DELTA[heading]
        expected = (start[0] + dx, start[1] + dy)
        start_neighbors = connections[start]
        if len(start_neighbors) != 1 or expected not in start_neighbors:
            raise MazeFormatError(
                "when start.heading is supplied, the start cell must have "
                "exactly one opening and it must point in that heading"
            )

    goal_values = root.get("goals")
    if not isinstance(goal_values, list) or not goal_values:
        raise MazeFormatError("goals must be a nonempty array of [x, y] cells")
    goals = {
        _external_cell(value, width=width, height=height, origin=origin, where=f"goals[{index}]")
        for index, value in enumerate(goal_values)
    }
    if len(goals) != len(goal_values):
        raise MazeFormatError("goals must not contain duplicates")
    if not _goal_region_connected(goals, connections):
        raise MazeFormatError("goal cells must form one connected open region")
    entrances = _goal_entrances(goals, connections)
    policy = root.get("goal_policy", "single_entry")
    if policy != "single_entry":
        raise MazeFormatError("goal_policy currently must be 'single_entry'")
    if len(entrances) != 1:
        raise MazeFormatError(f"single_entry goal region must have exactly one external opening; found {len(entrances)}")
    canonical = entrances[0].inside_cell

    reachable = _reachable(start, connections)
    if canonical not in reachable:
        raise MazeFormatError("canonical goal is unreachable from start")
    if root.get("require_all_cells_reachable", False) and len(reachable) != width * height:
        raise MazeFormatError(f"require_all_cells_reachable=true but only {len(reachable)}/{width*height} cells are reachable")

    scale_obj = _require_mapping(root.get("scale", {}), "scale")
    scale = MazeScale(
        unit_mm=float(scale_obj.get("unit_mm", 12.0)),
        wall_thickness_units=float(scale_obj.get("wall_thickness_units", 1.0)),
        cell_pitch_units=float(scale_obj.get("cell_pitch_units", 15.0)),
    )
    if abs(scale.unit_mm - 12.0) > 1e-12 or abs(scale.wall_thickness_units - 1.0) > 1e-12:
        raise MazeFormatError("project convention requires 12 mm per unit and wall_thickness_units=1")
    if scale.cell_pitch_units <= 0.0:
        raise MazeFormatError("scale.cell_pitch_units must be positive")

    maze = Maze.from_connections(width, height, connections, canonical)
    metadata = root.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise MazeFormatError("metadata must be an object")
    return MazeScenario(
        maze=maze,
        name=name.strip(),
        start=start,
        start_heading=heading,
        goal_region=GoalRegion(tuple(sorted(goals)), entrances, canonical),
        coordinate_origin=origin,
        source_path=str(source),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        scale=scale,
        metadata=dict(metadata),
    )
