"""Time-parameterized playback and GIF rendering for certified trajectories.

Playback is postprocessing only.  Stations come from :class:`SpeedProfileTrace`
and geometry states are evaluated by the exact production clothoid compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import GeometryState, RectangleBody
from visualization.sampling import GeometryTrace, sample_geometry_stations
from visualization.trajectory import body_polygon, draw_geometry_trace
from visualization.maze import draw_goal_region, draw_maze_walls, draw_start

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlaybackTrace:
    time: Array
    station: Array
    geometry: GeometryTrace
    speed: Array
    mode: tuple[str, ...]

    @property
    def duration(self) -> float:
        return float(self.time[-1])


def build_playback_trace(
    raw_parameters: Sequence[float],
    initial_state: GeometryState,
    speed_trace: Any,
    *,
    fps: float = 24.0,
    minimum_frames: int = 2,
) -> PlaybackTrace:
    if fps <= 0.0 or minimum_frames < 2:
        raise ValueError("fps must be positive and minimum_frames >= 2")
    duration = float(speed_trace.exact_total_time)
    frame_count = max(minimum_frames, int(np.ceil(duration * fps)) + 1)
    times = np.linspace(0.0, duration, frame_count)
    stations = speed_trace.station_at_time(times)
    geometry = sample_geometry_stations(raw_parameters, initial_state, stations)
    speeds = np.interp(stations, speed_trace.s, speed_trace.v)
    indices = np.searchsorted(speed_trace.s, stations, side="right") - 1
    indices = np.clip(indices, 0, len(speed_trace.mode) - 1)
    modes = tuple(speed_trace.mode[int(i)] for i in indices)
    return PlaybackTrace(times, stations, geometry, speeds, modes)


def render_trajectory_gif(
    output: str | Path,
    *,
    maze: Any,
    playback: PlaybackTrace,
    body: RectangleBody,
    cells: Sequence[tuple[int, int]] | None = None,
    scenario: Any | None = None,
    fps: float = 24.0,
    dpi: int = 100,
    title: str | None = None,
    cell_pitch_m: float = 1.0,
) -> Path:
    """Render a fixed-camera, real-time trajectory GIF using PillowWriter."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_maze_walls(ax, maze)
    if scenario is not None:
        draw_goal_region(ax, scenario=scenario)
        start = scenario.start
    else:
        draw_goal_region(ax, goal_cell=maze.goal)
        start = cells[0] if cells else (0, maze.height - 1)
    draw_start(ax, start)
    draw_geometry_trace(ax, playback.geometry, linewidth=1.15, alpha=0.4)
    if title:
        ax.set_title(title)

    body_patch = Polygon(body_polygon(playback.geometry.state(0), body), closed=True, fill=False, linewidth=2.0, zorder=8)
    ax.add_patch(body_patch)
    traversed, = ax.plot([], [], linewidth=2.4, zorder=7)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, zorder=10)

    if cell_pitch_m <= 0.0:
        raise ValueError("cell_pitch_m must be positive")

    def update(frame: int):
        body_patch.set_xy(body_polygon(playback.geometry.state(frame), body))
        traversed.set_data(playback.geometry.x[: frame + 1], playback.geometry.y[: frame + 1])
        speed_mps = playback.speed[frame] * cell_pitch_m
        status.set_text(
            f"t = {playback.time[frame]:.3f} s\n"
            f"v = {speed_mps:.3f} m/s\n"
            f"{playback.mode[frame]}"
        )
        return body_patch, traversed, status

    animation = FuncAnimation(fig, update, frames=len(playback.time), interval=1000.0 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output


__all__ = ["PlaybackTrace", "build_playback_trace", "render_trajectory_gif"]
