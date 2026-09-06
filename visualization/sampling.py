"""Exact-geometry sampling utilities for plotting and animation.

All path states come from the production clothoid compiler.  Sampling changes
only how densely an already-defined continuous curve is observed; it never
changes the curve or participates in optimization/certification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import GeometryState, compile_geometry_path, knot_parameters_to_raw

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GeometryTrace:
    """Dense samples of one exact piecewise-clothoid geometry."""

    s: Array
    x: Array
    y: Array
    theta: Array
    kappa: Array
    segment_index: IntArray
    knot_s: Array
    total_length: float

    def __post_init__(self) -> None:
        n = self.s.size
        for name in ("x", "y", "theta", "kappa", "segment_index"):
            if getattr(self, name).size != n:
                raise ValueError(f"GeometryTrace.{name} length mismatch")
        if n == 0:
            raise ValueError("GeometryTrace may not be empty")
        if np.any(np.diff(self.s) < 0.0):
            raise ValueError("GeometryTrace stations must be nondecreasing")
        if abs(float(self.s[0])) > 1e-12:
            raise ValueError("GeometryTrace must start at s=0")
        if abs(float(self.s[-1]) - float(self.total_length)) > 1e-10:
            raise ValueError("GeometryTrace must end at total_length")

    @property
    def xy(self) -> Array:
        return np.column_stack((self.x, self.y))

    def state(self, index: int) -> GeometryState:
        """Return one sampled state as the production GeometryState type."""
        return GeometryState(
            float(self.x[index]),
            float(self.y[index]),
            float(self.theta[index]),
            float(self.kappa[index]),
        )


def _raw_lengths(raw_parameters: Sequence[float]) -> Array:
    raw = np.asarray(raw_parameters, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or raw.size % 2:
        raise ValueError("raw parameters must be flat [L0, sigma0, ...]")
    lengths = raw[0::2].copy()
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("all segment lengths must be finite and positive")
    return lengths


def _states_at_stations(
    raw_parameters: Sequence[float],
    initial_state: GeometryState,
    stations: Array,
) -> GeometryTrace:
    raw = np.asarray(raw_parameters, dtype=float)
    lengths = _raw_lengths(raw)
    knots = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(knots[-1])
    if stations.ndim != 1 or stations.size == 0:
        raise ValueError("stations must be a nonempty one-dimensional array")
    tolerance = 64.0 * math.ulp(max(1.0, total))
    if float(stations[0]) < -tolerance or float(stations[-1]) > total + tolerance:
        raise ValueError("stations lie outside the geometry domain")
    stations = np.clip(stations.astype(float, copy=True), 0.0, total)
    if np.any(np.diff(stations) < 0.0):
        raise ValueError("stations must be sorted")

    path = compile_geometry_path(raw.tolist(), initial_state)
    # searchsorted(..., side='right') makes an interior knot belong to the next
    # segment.  The final knot is clamped back to the final segment.
    segment_index = np.searchsorted(knots[1:], stations, side="right").astype(np.int64)
    segment_index = np.minimum(segment_index, len(lengths) - 1)

    x = np.empty_like(stations)
    y = np.empty_like(stations)
    theta = np.empty_like(stations)
    kappa = np.empty_like(stations)
    for i, (station, segment) in enumerate(zip(stations, segment_index, strict=True)):
        local = float(station - knots[int(segment)])
        length = float(lengths[int(segment)])
        fraction = min(1.0, max(0.0, local / length))
        state = path.state_at_fraction(int(segment), fraction)
        x[i] = state.x
        y[i] = state.y
        theta[i] = state.theta
        kappa[i] = state.k

    return GeometryTrace(
        s=stations,
        x=x,
        y=y,
        theta=theta,
        kappa=kappa,
        segment_index=segment_index,
        knot_s=knots,
        total_length=total,
    )


def sample_geometry_stations(
    raw_parameters: Sequence[float],
    initial_state: GeometryState,
    stations: Sequence[float],
) -> GeometryTrace:
    """Evaluate the exact clothoid geometry at explicitly supplied stations."""

    return _states_at_stations(
        raw_parameters,
        initial_state,
        np.asarray(stations, dtype=float),
    )


def sample_raw_geometry(
    raw_parameters: Sequence[float],
    initial_state: GeometryState,
    *,
    samples_per_unit: float = 100.0,
    minimum_samples_per_segment: int = 12,
) -> GeometryTrace:
    """Sample each exact clothoid segment with deterministic station density."""

    if not math.isfinite(samples_per_unit) or samples_per_unit <= 0.0:
        raise ValueError("samples_per_unit must be finite and positive")
    if minimum_samples_per_segment < 2:
        raise ValueError("minimum_samples_per_segment must be at least 2")
    lengths = _raw_lengths(raw_parameters)
    stations: list[float] = []
    offset = 0.0
    for segment, length in enumerate(lengths):
        count = max(
            int(minimum_samples_per_segment),
            int(math.ceil(samples_per_unit * float(length))) + 1,
        )
        local = np.linspace(0.0, float(length), count)
        if segment:
            local = local[1:]
        stations.extend((offset + local).tolist())
        offset += float(length)
    return _states_at_stations(
        raw_parameters,
        initial_state,
        np.asarray(stations, dtype=float),
    )


def sample_geometry_parameters(
    parameters: Sequence[float],
    initial_state: GeometryState,
    *,
    samples_per_unit: float = 100.0,
    minimum_samples_per_segment: int = 12,
) -> GeometryTrace:
    """Sample knot-parameter geometry using the production raw conversion."""

    raw = knot_parameters_to_raw(parameters, initial_k=initial_state.k)
    return sample_raw_geometry(
        raw,
        initial_state,
        samples_per_unit=samples_per_unit,
        minimum_samples_per_segment=minimum_samples_per_segment,
    )
