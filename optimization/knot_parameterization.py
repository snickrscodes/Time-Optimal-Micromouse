"""Optimizer-coordinate maps for piecewise-linear-curvature paths.

The public knot basis remains::

    [s1, k1, s2, k2, ..., sn, kn]

with cumulative stations ``s_j``.  The production optimizer uses an internal
log-length basis::

    [z0, k1, z1, k2, ..., z(n-1), kn]

where ``L_i = L_i_ref * exp(z_i)`` and cumulative stations are reconstructed by
summing the positive segment lengths.  This guarantees strict knot ordering
without imposing local boxes on the cumulative stations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KnotCoordinateSettings:
    """Choice of optimizer coordinates for knot positions.

    ``mode="log_lengths"`` is the production default.  Public inputs and
    outputs are still cumulative stations; only the nonlinear solver sees log
    segment lengths.  ``minimum_length_ratio`` is a numerical and conditioning
    floor relative to each reference segment.  The default permits a segment
    to shrink by eight orders of magnitude, which is geometrically negligible
    at ordinary maze scales while preventing cumulative-station collapse.  A
    separate curvature-slope constraint controls the near-discontinuous limit
    ``sigma = delta_k / L`` rather than treating a numerical floor as physics.

    ``maximum_abs_log_length`` is a final exponent guard.  The effective lower
    log bound is the stronger of ``log(minimum_length_ratio)`` and
    ``-maximum_abs_log_length``.  ``maximum_length_ratio`` is a one-sided trust
    bound that prevents expensive
    trial paths from making one segment arbitrarily longer than its reference.
    It does not limit shortening.  ``mode="stations"`` retains the legacy
    direct cumulative-station solve.
    """

    mode: Literal["log_lengths", "stations"] = "log_lengths"
    maximum_abs_log_length: float = 50.0
    minimum_length_ratio: float = 1.0e-8
    maximum_length_ratio: float | None = 4.0

    def __post_init__(self) -> None:
        if self.mode not in {"log_lengths", "stations"}:
            raise ValueError("mode must be 'log_lengths' or 'stations'")
        if (
            not math.isfinite(self.maximum_abs_log_length)
            or self.maximum_abs_log_length <= 0.0
        ):
            raise ValueError("maximum_abs_log_length must be finite and positive")
        if (
            not math.isfinite(self.minimum_length_ratio)
            or not 0.0 < self.minimum_length_ratio <= 1.0
        ):
            raise ValueError(
                "minimum_length_ratio must be finite and lie in (0, 1]"
            )
        if self.maximum_length_ratio is not None and (
            not math.isfinite(self.maximum_length_ratio)
            or self.maximum_length_ratio <= 1.0
        ):
            raise ValueError("maximum_length_ratio must be None or finite and greater than one")


@dataclass(frozen=True, slots=True)
class LogLengthKnotMap:
    """Invertible map between cumulative stations and log segment lengths."""

    reference_lengths: Array
    initial_s: float = 0.0

    def __post_init__(self) -> None:
        lengths = np.asarray(self.reference_lengths, dtype=float)
        if lengths.ndim != 1 or lengths.size == 0:
            raise ValueError("reference_lengths must be a nonempty vector")
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError("reference_lengths must be finite and positive")
        initial_s = float(self.initial_s)
        if not math.isfinite(initial_s):
            raise ValueError("initial_s must be finite")
        object.__setattr__(self, "reference_lengths", lengths.copy())
        object.__setattr__(self, "initial_s", initial_s)

    @classmethod
    def from_knot_parameters(
        cls,
        knot_params: Sequence[float],
        *,
        initial_s: float = 0.0,
    ) -> "LogLengthKnotMap":
        x = np.asarray(knot_params, dtype=float)
        if x.ndim != 1 or x.size == 0 or x.size % 2:
            raise ValueError("knot parameters must be a nonempty [s1,k1,...] vector")
        if not np.all(np.isfinite(x)):
            raise ValueError("knot parameters must be finite")
        stations = x[0::2]
        previous = np.concatenate(([float(initial_s)], stations[:-1]))
        lengths = stations - previous
        if np.any(lengths <= 0.0):
            raise ValueError("knot stations must be strictly increasing")
        return cls(lengths, initial_s=float(initial_s))

    @property
    def n_segments(self) -> int:
        return int(self.reference_lengths.size)

    @property
    def width(self) -> int:
        return 2 * self.n_segments

    def _checked_internal(self, variables: Sequence[float]) -> Array:
        u = np.asarray(variables, dtype=float)
        if u.shape != (self.width,):
            raise ValueError("optimizer-coordinate vector has wrong shape")
        if not np.all(np.isfinite(u)):
            raise ValueError("optimizer coordinates must be finite")
        return u

    def _checked_physical(self, knot_params: Sequence[float]) -> Array:
        x = np.asarray(knot_params, dtype=float)
        if x.shape != (self.width,):
            raise ValueError("knot-parameter vector has wrong shape")
        if not np.all(np.isfinite(x)):
            raise ValueError("knot parameters must be finite")
        return x

    def lengths(self, variables: Sequence[float]) -> Array:
        u = self._checked_internal(variables)
        lengths = self.reference_lengths * np.exp(u[0::2])
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise FloatingPointError("log-length map produced invalid segment lengths")
        return lengths

    def to_optimizer(self, knot_params: Sequence[float]) -> Array:
        """Map physical ``[s,k]`` knots to internal ``[log(L/Lref),k]``."""
        x = self._checked_physical(knot_params)
        stations = x[0::2]
        previous = np.concatenate(([self.initial_s], stations[:-1]))
        lengths = stations - previous
        if np.any(lengths <= 0.0):
            raise ValueError("knot stations must be strictly increasing")
        u = np.empty_like(x)
        u[0::2] = np.log(lengths / self.reference_lengths)
        u[1::2] = x[1::2]
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("physical-to-log-length map produced nonfinite data")
        return u

    def to_physical(self, variables: Sequence[float]) -> Array:
        """Map internal ``[z,k]`` coordinates to cumulative-station knots."""
        u = self._checked_internal(variables)
        lengths = self.lengths(u)
        x = np.empty_like(u)
        stations = self.initial_s + np.cumsum(lengths)
        previous = np.concatenate(([self.initial_s], stations[:-1]))
        if np.any(stations <= previous):
            raise FloatingPointError(
                "positive log lengths collapsed while forming cumulative "
                "binary64 stations; increase minimum_length_ratio"
            )
        x[0::2] = stations
        x[1::2] = u[1::2]
        return x

    def pullback_gradient(
        self,
        variables: Sequence[float],
        physical_gradient: Sequence[float],
    ) -> Array:
        """Pull a scalar gradient from cumulative stations to log lengths."""
        u = self._checked_internal(variables)
        gradient = np.asarray(physical_gradient, dtype=float)
        if gradient.shape != (self.width,):
            raise ValueError("physical gradient has wrong shape")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("physical gradient must be finite")
        lengths = self.lengths(u)
        out = np.empty_like(gradient)
        station_gradient = gradient[0::2]
        suffix = np.cumsum(station_gradient[::-1])[::-1]
        out[0::2] = lengths * suffix
        out[1::2] = gradient[1::2]
        return out

    def pullback_jacobian(
        self,
        variables: Sequence[float],
        physical_jacobian: Sequence[Sequence[float]] | Array,
    ) -> Array:
        """Pull Jacobian rows from cumulative stations to log lengths."""
        u = self._checked_internal(variables)
        jacobian = np.asarray(physical_jacobian, dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[1] != self.width:
            raise ValueError("physical Jacobian has wrong shape")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("physical Jacobian must be finite")
        lengths = self.lengths(u)
        out = np.empty_like(jacobian)
        station_rows = jacobian[:, 0::2]
        suffix = np.cumsum(station_rows[:, ::-1], axis=1)[:, ::-1]
        out[:, 0::2] = suffix * lengths[np.newaxis, :]
        out[:, 1::2] = jacobian[:, 1::2]
        return out

    def pullback_sparse_jacobian(
        self,
        variables: Sequence[float],
        physical_jacobian: object,
    ):
        """Pull sparse Jacobian rows into log-length coordinates.

        The coordinate Jacobian is block lower-triangular: cumulative station
        ``s_i`` depends on log lengths ``z_0, ..., z_i``.  Sparse matrix
        multiplication preserves the path-prefix structure without first
        materializing a dense row matrix.
        """
        from scipy.sparse import coo_matrix, csr_matrix

        u = self._checked_internal(variables)
        jacobian = csr_matrix(physical_jacobian, dtype=float)
        if jacobian.shape[1] != self.width:
            raise ValueError("physical Jacobian has wrong shape")
        if not np.all(np.isfinite(jacobian.data)):
            raise ValueError("physical Jacobian must be finite")
        lengths = self.lengths(u)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for i in range(self.n_segments):
            station_row = 2 * i
            for j in range(i + 1):
                rows.append(station_row)
                cols.append(2 * j)
                data.append(float(lengths[j]))
            rows.append(station_row + 1)
            cols.append(station_row + 1)
            data.append(1.0)
        transform = coo_matrix(
            (np.asarray(data, dtype=float), (rows, cols)),
            shape=(self.width, self.width),
        ).tocsr()
        result = (jacobian @ transform).tocsr()
        result.eliminate_zeros()
        return result


__all__ = ["KnotCoordinateSettings", "LogLengthKnotMap"]
