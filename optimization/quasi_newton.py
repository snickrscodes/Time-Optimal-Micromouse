"""Structured positive-definite quasi-Newton models for sparse SQP.

The path variables are ordered along arclength, so a narrow symmetric band is
a natural candidate Hessian model.  ``BandedHessian`` stores only the lower
band, supports O(n*b) products and BFGS projection, and enforces positive definiteness with LAPACK's symmetric-band eigensolver,
avoiding a dense matrix while retaining substantially better scaling than a
Gershgorin-only shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(slots=True)
class BandedHessian:
    """Symmetric lower-band matrix.

    ``bands[d, i]`` stores ``H[i, i-d]`` for ``d <= i``.  The first row is the
    diagonal.  Entries outside ``bandwidth`` are exactly zero.
    """

    bands: Array

    def __post_init__(self) -> None:
        bands = np.asarray(self.bands, dtype=float)
        if bands.ndim != 2 or bands.shape[1] == 0:
            raise ValueError("bands must have shape (bandwidth+1, n)")
        if not np.all(np.isfinite(bands)):
            raise ValueError("bands must be finite")
        for d in range(1, bands.shape[0]):
            bands[d, :d] = 0.0
        self.bands = bands.copy()

    @classmethod
    def scaled_identity(cls, n: int, scale: float, bandwidth: int) -> "BandedHessian":
        if n <= 0 or bandwidth < 0 or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("invalid banded Hessian dimensions or scale")
        bandwidth = min(int(bandwidth), n - 1)
        bands = np.zeros((bandwidth + 1, n), dtype=float)
        bands[0] = float(scale)
        return cls(bands)

    @property
    def n(self) -> int:
        return int(self.bands.shape[1])

    @property
    def bandwidth(self) -> int:
        return int(self.bands.shape[0] - 1)

    @property
    def nnz(self) -> int:
        n = self.n
        b = self.bandwidth
        return n + 2 * sum(n - d for d in range(1, b + 1))

    def copy(self) -> "BandedHessian":
        return BandedHessian(self.bands.copy())

    def matvec(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError("vector has wrong shape")
        out = self.bands[0] * x
        for d in range(1, self.bandwidth + 1):
            vals = self.bands[d, d:]
            out[d:] += vals * x[:-d]
            out[:-d] += vals * x[d:]
        return out

    def quadratic(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.matvec(x))

    def to_csc(self):
        from scipy.sparse import diags

        diagonals: list[Array] = [self.bands[0].copy()]
        offsets: list[int] = [0]
        for d in range(1, self.bandwidth + 1):
            vals = self.bands[d, d:].copy()
            diagonals.extend((vals, vals))
            offsets.extend((-d, d))
        return diags(diagonals, offsets, shape=(self.n, self.n), format="csc")

    def gershgorin_lower_bound(self) -> float:
        radii = np.zeros(self.n, dtype=float)
        for d in range(1, self.bandwidth + 1):
            vals = np.abs(self.bands[d, d:])
            radii[d:] += vals
            radii[:-d] += vals
        return float(np.min(self.bands[0] - radii))

    def regularize(self, minimum_eigenvalue: float, maximum_eigenvalue: float) -> None:
        """Spectrally bound the band without forming a dense matrix.

        LAPACK's symmetric-band eigensolver costs O(n*b^2), versus O(n^3) for
        the former dense eigendecomposition.  A uniform scale and diagonal
        shift preserve the band and avoid the severe over-regularization of a
        Gershgorin-only projection.
        """
        from scipy.linalg import eig_banded

        if minimum_eigenvalue <= 0.0 or maximum_eigenvalue < minimum_eigenvalue:
            raise ValueError("invalid regularization bounds")
        ab = np.zeros_like(self.bands)
        ab[0] = self.bands[0]
        for d in range(1, self.bandwidth + 1):
            ab[d, : self.n - d] = self.bands[d, d:]
        eigenvalues = eig_banded(
            ab, lower=True, eigvals_only=True, check_finite=False
        )
        largest = float(eigenvalues[-1])
        if largest > maximum_eigenvalue:
            factor = maximum_eigenvalue / largest
            self.bands *= factor
            eigenvalues *= factor
        smallest = float(eigenvalues[0])
        if smallest < minimum_eigenvalue:
            self.bands[0] += minimum_eigenvalue - smallest

    def damped_bfgs_update(
        self,
        step: Array,
        gradient_change: Array,
        *,
        curvature_tolerance: float,
        minimum_eigenvalue: float,
        maximum_eigenvalue: float,
    ) -> bool:
        """Apply Powell-damped BFGS and project directly into the band."""
        s = np.asarray(step, dtype=float)
        y_in = np.asarray(gradient_change, dtype=float)
        if s.shape != (self.n,) or y_in.shape != (self.n,):
            raise ValueError("BFGS vectors have wrong shape")
        norm = float(np.linalg.norm(s))
        if norm == 0.0:
            return False
        bs = self.matvec(s)
        sbs = float(s @ bs)
        sy = float(s @ y_in)
        tol = curvature_tolerance * max(1.0, norm * norm)
        if not math.isfinite(sbs) or sbs <= tol:
            return False
        y = y_in.copy()
        if sy < 0.2 * sbs:
            denominator = sbs - sy
            if denominator <= tol:
                return False
            theta = 0.8 * sbs / denominator
            y = theta * y + (1.0 - theta) * bs
            sy = float(s @ y)
        if not math.isfinite(sy) or sy <= tol:
            return False

        # Only entries inside the retained band are formed.
        self.bands[0] += -(bs * bs) / sbs + (y * y) / sy
        for d in range(1, self.bandwidth + 1):
            self.bands[d, d:] += (
                -(bs[d:] * bs[:-d]) / sbs
                + (y[d:] * y[:-d]) / sy
            )
        self.regularize(minimum_eigenvalue, maximum_eigenvalue)
        return True


__all__ = ["BandedHessian"]
