"""Maze/scenario drawing primitives shared by CLI and artifact scripts."""

from __future__ import annotations

from typing import Any


def draw_maze_walls(ax: Any, maze: Any, **collection_kwargs: Any) -> Any:
    from matplotlib.collections import LineCollection

    defaults = {
        "linewidths": 1.6,
        "capstyle": "butt",
        "joinstyle": "miter",
        "zorder": 1,
    }
    defaults.update(collection_kwargs)
    collection = LineCollection(
        [[first, second] for first, second in maze.walls],
        **defaults,
    )
    ax.add_collection(collection)
    ax.set_xlim(-0.05, maze.width + 0.05)
    ax.set_ylim(maze.height + 0.05, -0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return collection


def draw_start(ax: Any, start: tuple[int, int], **scatter_kwargs: Any) -> Any:
    defaults = {"s": 42, "marker": "o", "zorder": 6}
    defaults.update(scatter_kwargs)
    return ax.scatter([start[0] + 0.5], [start[1] + 0.5], **defaults)


def draw_goal_region(
    ax: Any,
    *,
    scenario: Any | None = None,
    goal_cell: tuple[int, int] | None = None,
    show_entry: bool = True,
    **patch_kwargs: Any,
) -> list[Any]:
    """Draw semantic goal cells when available, otherwise one canonical cell."""

    from matplotlib.patches import Rectangle

    if scenario is not None:
        cells = scenario.goal_region.cells
    elif goal_cell is not None:
        cells = (goal_cell,)
    else:
        raise ValueError("scenario or goal_cell is required")
    defaults = {"fill": True, "alpha": 0.12, "linewidth": 0.8, "zorder": 0}
    defaults.update(patch_kwargs)
    artists: list[Any] = []
    for x, y in cells:
        patch = Rectangle((x, y), 1.0, 1.0, **defaults)
        ax.add_patch(patch)
        artists.append(patch)

    if scenario is not None and show_entry:
        for entry in scenario.goal_region.entrances:
            ox, oy = entry.outside_cell
            ix, iy = entry.inside_cell
            p0 = (ox + 0.5, oy + 0.5)
            p1 = (ix + 0.5, iy + 0.5)
            # Mark the shared edge by taking the midpoint between cell centers
            # and drawing a short segment tangent to that edge.
            mx = 0.5 * (p0[0] + p1[0])
            my = 0.5 * (p0[1] + p1[1])
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            tx, ty = -dy, dx
            artist = ax.plot(
                [mx - 0.28 * tx, mx + 0.28 * tx],
                [my - 0.28 * ty, my + 0.28 * ty],
                linewidth=2.6,
                zorder=7,
            )
            artists.extend(artist)
    return artists
