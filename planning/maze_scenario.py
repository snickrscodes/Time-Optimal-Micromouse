"""Rich maze metadata layered over the production :class:`mazegen.Maze`.

The discrete planner intentionally retains one canonical goal cell.  A loaded
scenario may preserve a larger semantic goal region; when that region has one
external opening, the inside cell at that opening becomes ``maze.goal``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mazegen import Cell, Maze


@dataclass(frozen=True, slots=True)
class GoalEntrance:
    outside_cell: Cell
    inside_cell: Cell


@dataclass(frozen=True, slots=True)
class GoalRegion:
    cells: tuple[Cell, ...]
    entrances: tuple[GoalEntrance, ...]
    canonical_goal: Cell


@dataclass(frozen=True, slots=True)
class MazeScale:
    """Physical metadata for classic micromouse-style mazes.

    One project wall-thickness unit is 12 mm.  A standard 180 mm cell pitch is
    therefore 15 units.  The current grid planner still works in *cell units*;
    this object records the physical conversion without silently rescaling the
    established dynamics model.
    """

    unit_mm: float = 12.0
    wall_thickness_units: float = 1.0
    cell_pitch_units: float = 15.0

    @property
    def cell_pitch_mm(self) -> float:
        return self.unit_mm * self.cell_pitch_units

    @property
    def cell_pitch_m(self) -> float:
        return self.cell_pitch_mm / 1000.0


@dataclass(frozen=True, slots=True)
class MazeScenario:
    maze: Maze
    name: str
    start: Cell
    start_heading: str | None
    goal_region: GoalRegion
    coordinate_origin: str
    source_path: str
    source_sha256: str
    scale: MazeScale = MazeScale()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_goal(self) -> Cell:
        return self.goal_region.canonical_goal

    def to_source_cell(self, cell: Cell) -> Cell:
        """Convert a normalized internal cell back to the file's coordinates."""
        x, y = cell
        if self.coordinate_origin == "southwest":
            return x, self.maze.height - 1 - y
        return x, y
