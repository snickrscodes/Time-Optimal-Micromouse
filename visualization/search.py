"""Small-search visualization helpers for branch-and-bound trace events."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def draw_search_tree(ax: Any, events: Iterable[Any]) -> None:
    """Draw one compact tree from SearchTraceEvent records.

    This is intended for small exhaustive validation cases, not full 16x16
    Micromouse searches.
    """
    events = list(events)
    state: dict[tuple[int, ...], Any] = {}
    for event in events:
        path = tuple(event.junction_path)
        # Last event for a node intentionally wins: it carries the most useful
        # final status (pruned/complete/incumbent) for a static diagram.
        state[path] = event
    if not state:
        raise ValueError("search trace is empty")

    by_depth: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for path in state:
        by_depth[len(path) - 1].append(path)
    position: dict[tuple[int, ...], tuple[float, float]] = {}
    for depth in sorted(by_depth):
        paths = sorted(by_depth[depth])
        n = len(paths)
        for index, path in enumerate(paths):
            x = (index + 1) / (n + 1)
            y = -float(depth)
            position[path] = (x, y)

    for path, event in state.items():
        parent = tuple(event.parent_path) if event.parent_path is not None else None
        if parent in position:
            x0, y0 = position[parent]
            x1, y1 = position[path]
            ax.plot([x0, x1], [y0, y1], linewidth=0.8, alpha=0.6, zorder=1)

    terminal_kinds = {"pruned_bound", "pruned_reachability", "pruned_complete_bound", "rejected_visit", "complete", "incumbent_update"}
    for path, event in state.items():
        x, y = position[path]
        marker = "s" if event.kind in terminal_kinds else "o"
        ax.scatter([x], [y], s=70, marker=marker, zorder=3)
        label = "→".join(str(v) for v in path)
        if event.lower_bound is not None:
            label += f"\nLB {event.lower_bound:.3g}"
        if event.complete_time is not None:
            label += f"\nT {event.complete_time:.3g}"
        if event.kind not in {"generated", "expanded", "root", "seed"}:
            label += f"\n{event.kind.replace('_', ' ')}"
        ax.text(x, y - 0.11, label, ha="center", va="top", fontsize=6.5)
    ax.set_axis_off()
    ax.set_title("Branch-and-bound trace")


__all__ = ["draw_search_tree"]
