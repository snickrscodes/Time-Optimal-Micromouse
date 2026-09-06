"""Development renderer for the qualified active-basis V11 architecture."""
from __future__ import annotations
from typing import Any


def draw_system_architecture(ax: Any) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nodes = {
        "maze": (0.50, 0.95, "Maze / physical scenario"),
        "graph": (0.50, 0.85, "Compressed junction graph"),
        "astar": (0.25, 0.73, "Shortest-distance A*\nseed incumbent"),
        "bnb": (0.70, 0.73, "Kinodynamic B&B\nreachability + time LBs"),
        "topology": (0.50, 0.61, "Fixed cell topology"),
        "reduced": (0.25, 0.47, "Reduced maximal-run basis\ncertified analytic initializer"),
        "curv": (0.25, 0.33, "Curvature guard homotopy\nq=.01 → .005 → [.0025]"),
        "time": (0.55, 0.42, "Reduced time transactions\nuntil exhaustion"),
        "speed": (0.82, 0.42, "Hybrid reverse speed solver\nprofile-selected dynamics"),
        "basis": (0.55, 0.27, "Support analysis + selective\nturn-pair basis activation"),
        "floor": (0.55, 0.14, "Conditioning-floor continuation\n7.5e-4 → 6.0e-4 + closure"),
        "cert": (0.25, 0.14, "Independent certification\ngeometry + dynamics replay"),
        "out": (0.50, 0.035, "Certified minimum-time trajectory"),
    }
    widths = {key: (0.25 if key not in {"speed", "cert"} else 0.27) for key in nodes}
    height = 0.073
    artists: dict[str, tuple[float, float]] = {}
    for key, (x, y, label) in nodes.items():
        w = widths[key]
        box = FancyBboxPatch(
            (x - w / 2, y - height / 2), w, height,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            fill=False, linewidth=1.15,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.2)
        artists[key] = (x, y)

    def arrow(a: str, b: str, *, rad: float = 0.0, label: str | None = None) -> None:
        x0, y0 = artists[a]; x1, y1 = artists[b]
        patch = FancyArrowPatch(
            (x0, y0 - height / 2), (x1, y1 + height / 2),
            arrowstyle="-|>", mutation_scale=10, linewidth=0.95,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)
        if label:
            ax.text((x0+x1)/2, (y0+y1)/2, label, fontsize=6.0,
                    ha="center", va="center")

    arrow("maze", "graph")
    arrow("graph", "astar", rad=0.04)
    arrow("graph", "bnb", rad=-0.04)
    arrow("astar", "bnb", rad=-0.16, label="initial incumbent")
    arrow("bnb", "topology")
    arrow("topology", "reduced", rad=0.04)
    arrow("reduced", "curv")
    arrow("curv", "time", rad=-0.04)
    arrow("time", "basis")
    arrow("basis", "floor")
    arrow("speed", "time", rad=0.16, label="T + gradient")
    arrow("time", "speed", rad=0.16)
    arrow("speed", "basis", rad=-0.14)
    arrow("curv", "cert", rad=0.03)
    arrow("floor", "cert", rad=-0.04)
    arrow("cert", "out", rad=-0.03)
    arrow("floor", "out", rad=0.03)

    ax.set_xlim(0.04, 0.96)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.set_title("AME planner · qualified active-basis V11")


__all__ = ["draw_system_architecture"]
