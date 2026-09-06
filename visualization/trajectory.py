"""Trajectory, topology, body-footprint, and corridor drawing primitives."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from mazegen import Cell
from optimization import ConvexCell, GeometryState, RectangleBody

Array = NDArray[np.float64]


def topology_centerline(cells: Sequence[Cell]) -> Array:
    """Cell-center polyline used only to visualize a discrete topology."""

    return np.asarray([(x + 0.5, y + 0.5) for x, y in cells], dtype=float)


def body_polygon(state: GeometryState, body: RectangleBody) -> Array:
    """Return body corners in a non-self-crossing world-frame polygon order."""

    c = math.cos(state.theta)
    s = math.sin(state.theta)
    # front-left -> front-right -> rear-right -> rear-left
    local = (
        (body.front, body.left),
        (body.front, -body.right),
        (-body.rear, -body.right),
        (-body.rear, body.left),
    )
    points = []
    for u, v in local:
        points.append((state.x + u * c - v * s, state.y + u * s + v * c))
    return np.asarray(points, dtype=float)


def convex_cell_polygon(cell: ConvexCell, *, tolerance: float = 1e-9) -> Array:
    """Recover a finite convex polygon from a 2-D normalized half-space cell.

    Production corridor cells are rectangles/open-room rectangles, but using
    the half-space representation here keeps visualization decoupled from that
    construction detail.
    """

    points: list[tuple[float, float]] = []
    walls = cell.walls
    for i, first in enumerate(walls):
        a1, b1 = first.normal
        c1 = first.offset
        for second in walls[i + 1 :]:
            a2, b2 = second.normal
            c2 = second.offset
            det = a1 * b2 - a2 * b1
            if abs(det) <= 1e-14:
                continue
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            if all(
                wall.normal[0] * x + wall.normal[1] * y
                <= wall.offset + tolerance
                for wall in walls
            ):
                points.append((float(x), float(y)))
    if len(points) < 3:
        raise ValueError(f"convex cell {cell.name!r} does not yield a finite polygon")
    unique: list[tuple[float, float]] = []
    for point in points:
        if not any(math.hypot(point[0] - q[0], point[1] - q[1]) <= tolerance for q in unique):
            unique.append(point)
    center = np.mean(np.asarray(unique, dtype=float), axis=0)
    unique.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    return np.asarray(unique, dtype=float)


def draw_topology(ax: Any, cells: Sequence[Cell], **plot_kwargs: Any) -> Any:
    centerline = topology_centerline(cells)
    defaults = {"linestyle": "--", "linewidth": 1.2, "alpha": 0.75, "zorder": 2}
    defaults.update(plot_kwargs)
    return ax.plot(centerline[:, 0], centerline[:, 1], **defaults)


def draw_geometry_trace(ax: Any, trace: Any, **plot_kwargs: Any) -> Any:
    defaults = {"linewidth": 2.2, "zorder": 4}
    defaults.update(plot_kwargs)
    return ax.plot(trace.x, trace.y, **defaults)


def draw_body_footprint(
    ax: Any,
    state: GeometryState,
    body: RectangleBody,
    **patch_kwargs: Any,
) -> Any:
    from matplotlib.patches import Polygon

    defaults = {"fill": False, "linewidth": 1.0, "zorder": 5}
    defaults.update(patch_kwargs)
    patch = Polygon(body_polygon(state, body), closed=True, **defaults)
    ax.add_patch(patch)
    return patch


def draw_corridor(
    ax: Any,
    corridor: Any,
    *,
    indices: Iterable[int] | None = None,
    annotate: bool = False,
    **patch_kwargs: Any,
) -> list[Any]:
    """Draw corridor cover cells; repeated segment assignments are not redrawn."""

    from matplotlib.patches import Polygon

    selected = range(len(corridor.cells)) if indices is None else indices
    defaults = {"alpha": 0.12, "linewidth": 0.8, "zorder": 0}
    defaults.update(patch_kwargs)
    patches: list[Any] = []
    for index in selected:
        polygon = convex_cell_polygon(corridor.cells[int(index)])
        patch = Polygon(polygon, closed=True, **defaults)
        ax.add_patch(patch)
        patches.append(patch)
        if annotate:
            center = np.mean(polygon, axis=0)
            ax.text(center[0], center[1], str(index), ha="center", va="center", fontsize=7, zorder=6)
    return patches
