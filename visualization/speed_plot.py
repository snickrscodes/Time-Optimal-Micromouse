"""Reusable plotting layer for :class:`SpeedProfileTrace`.

The module renders the semantic layers used by the release figures: mode spans,
speed, curvature, and event markers.
"""

from __future__ import annotations

from typing import Any, Mapping


def draw_speed_profile(
    ax_speed: Any,
    ax_curvature: Any,
    trace: Any,
    *,
    mode_colors: Mapping[str, Any] | None = None,
    shade_modes: bool = True,
    show_events: bool = True,
    cell_pitch_m: float | None = None,
) -> None:
    if mode_colors is None:
        mode_colors = {
            "MOTOR": "tab:blue",
            "GRIP": "tab:orange",
            "BRAKE": "tab:gray",
            "SIDE_LEFT": "tab:green",
            "SIDE_RIGHT": "tab:red",
        }
    station_scale = 1.0 if cell_pitch_m is None else float(cell_pitch_m)
    if station_scale <= 0.0:
        raise ValueError("cell_pitch_m must be positive when provided")
    x = trace.s * station_scale
    v = trace.v * station_scale
    kappa = trace.kappa / station_scale
    x_label = "Path station $s$" if cell_pitch_m is None else "Path station $s$ (m)"
    speed_label = "Speed $v(s)$" if cell_pitch_m is None else "Speed $v(s)$ (m/s)"
    curvature_label = r"Curvature $\kappa(s)$" if cell_pitch_m is None else r"Curvature $\kappa(s)$ (1/m)"
    if shade_modes:
        seen: set[str] = set()
        for interval in trace.intervals:
            label = interval.mode if interval.mode not in seen else None
            seen.add(interval.mode)
            color = mode_colors.get(interval.mode)
            ax_speed.axvspan(
                interval.station0 * station_scale,
                interval.station1 * station_scale,
                alpha=0.10,
                color=color,
                label=label,
                zorder=0,
            )
    ax_speed.plot(x, v, linewidth=1.8, zorder=3)
    ax_speed.set_ylabel(speed_label)
    ax_speed.grid(True, alpha=0.2)
    ax_curvature.plot(x, kappa, linewidth=1.6, zorder=3)
    ax_curvature.set_xlabel(x_label)
    ax_curvature.set_ylabel(curvature_label)
    ax_curvature.grid(True, alpha=0.2)
    if show_events:
        for event in trace.events:
            if event.kind != "mode_switch":
                continue
            x_event = event.station * station_scale
            ax_speed.axvline(x_event, linewidth=0.7, alpha=0.35)
            ax_curvature.axvline(x_event, linewidth=0.7, alpha=0.35)
