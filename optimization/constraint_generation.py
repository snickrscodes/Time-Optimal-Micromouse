"""Production rectangular-corridor constraint generation for clothoid paths.

The optimization variables are flat knot parameters::

    [s1, k1, s2, k2, ..., sn, kn]

with fixed ``s0`` and ``k0`` supplied separately.  Geometry kernels consume the
corresponding flat raw parameters::

    [L0, sigma0, L1, sigma1, ...]

This module provides:

* normalized convex half-space corridor cells,
* an exact four-corner rectangular vehicle model,
* grouped smooth finite point constraints and exact Jacobians,
* exhaustive continuous corner-wall separation with shared state/root work,
* a persistent normalized-coordinate constraint pool,
* a backend-neutral finite constraint oracle for optimizer callbacks.

The optimizer dimension never changes.  Heading-quadrant changes, corner
support switches, and changes in the number of stationary points are confined
to the separation oracle.
"""

from __future__ import annotations

import math
import operator
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, NamedTuple, Protocol, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .geometry_gradients import (
    GeometryPath,
    GeometryState,
    compile_geometry_path,
    knot_parameters_to_raw,
    pullback_raw_gradient_to_knot_parameters,
    pullback_raw_gradient_from_precomputed_raw,
)

Array: TypeAlias = NDArray[np.float64]
Float2: TypeAlias = tuple[float, float]
Float4: TypeAlias = tuple[float, float, float, float]

_PI = math.pi
_HALF_PI = 0.5 * math.pi


# =============================================================================
# Public callback protocols
# =============================================================================


class ScalarObjective(Protocol):
    """Return ``(value, gradient)`` in the optimizer's knot basis."""

    def __call__(self, x: Sequence[float]) -> tuple[float, Sequence[float]]: ...


class VectorConstraint(Protocol):
    """Return ``(values, jacobian)`` with one Jacobian row per value."""

    def __call__(self, x: Sequence[float]) -> tuple[Sequence[float], Sequence[Sequence[float]]]: ...


# =============================================================================
# Corridor and body models
# =============================================================================


@dataclass(frozen=True, slots=True)
class HalfSpace:
    """Normalized half-space ``normal dot p <= offset``."""

    normal: Float2
    offset: float
    name: str = ""

    def __post_init__(self) -> None:
        nx, ny = map(float, self.normal)
        offset = float(self.offset)
        norm = math.hypot(nx, ny)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("half-space normal must be finite and nonzero")
        if not math.isfinite(offset):
            raise ValueError("half-space offset must be finite")
        inv = 1.0 / norm
        object.__setattr__(self, "normal", (nx * inv, ny * inv))
        object.__setattr__(self, "offset", offset * inv)


@dataclass(frozen=True, slots=True)
class ConvexCell:
    """One convex free-space cell represented by normalized half-spaces."""

    walls: tuple[HalfSpace, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.walls:
            raise ValueError("a convex cell must contain at least one wall")

    @classmethod
    def axis_aligned_rectangle(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        *,
        name: str = "",
    ) -> "ConvexCell":
        xmin, xmax, ymin, ymax = map(float, (xmin, xmax, ymin, ymax))
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            raise ValueError("rectangle bounds must be finite")
        if not xmin < xmax or not ymin < ymax:
            raise ValueError("rectangle bounds must satisfy xmin<xmax and ymin<ymax")
        return cls(
            (
                HalfSpace((1.0, 0.0), xmax, "right"),
                HalfSpace((-1.0, 0.0), -xmin, "left"),
                HalfSpace((0.0, 1.0), ymax, "top"),
                HalfSpace((0.0, -1.0), -ymin, "bottom"),
            ),
            name,
        )


@dataclass(frozen=True, slots=True)
class BodyCorner:
    """A fixed body-frame corner ``u*tangent + v*left_normal``."""

    u: float
    v: float
    name: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.u) or not math.isfinite(self.v):
            raise ValueError("body corner coordinates must be finite")


@dataclass(frozen=True, slots=True)
class RectangleBody:
    """Rectangular vehicle relative to the path reference point.

    ``front`` and ``rear`` are nonnegative longitudinal distances from the
    reference point. ``left`` and ``right`` are nonnegative lateral distances.
    """

    front: float
    rear: float
    left: float
    right: float

    def __post_init__(self) -> None:
        values = tuple(map(float, (self.front, self.rear, self.left, self.right)))
        if not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("rectangle extents must be finite and nonnegative")
        if values[0] + values[1] <= 0.0 or values[2] + values[3] <= 0.0:
            raise ValueError("rectangle must have positive length and width")
        object.__setattr__(self, "front", values[0])
        object.__setattr__(self, "rear", values[1])
        object.__setattr__(self, "left", values[2])
        object.__setattr__(self, "right", values[3])

    @classmethod
    def centered(cls, length: float, width: float) -> "RectangleBody":
        length = float(length)
        width = float(width)
        if not math.isfinite(length) or not math.isfinite(width) or length <= 0.0 or width <= 0.0:
            raise ValueError("length and width must be finite and positive")
        return cls(0.5 * length, 0.5 * length, 0.5 * width, 0.5 * width)

    @property
    def length(self) -> float:
        """Total longitudinal footprint length."""
        return self.front + self.rear

    @property
    def height(self) -> float:
        """Total lateral footprint height."""
        return self.left + self.right

    @property
    def width(self) -> float:
        """Compatibility alias for :attr:`height`."""
        return self.height

    @property
    def corners(self) -> tuple[BodyCorner, BodyCorner, BodyCorner, BodyCorner]:
        return (
            BodyCorner(self.front, self.left, "front_left"),
            BodyCorner(self.front, -self.right, "front_right"),
            BodyCorner(-self.rear, self.left, "rear_left"),
            BodyCorner(-self.rear, -self.right, "rear_right"),
        )

    @property
    def maximum_radius(self) -> float:
        return max(math.hypot(c.u, c.v) for c in self.corners)


@dataclass(frozen=True, slots=True)
class CorridorModel:
    """Fixed ordered corridor and segment-to-cell assignment."""

    cells: tuple[ConvexCell, ...]
    segment_cells: tuple[int, ...]
    body: RectangleBody
    clearance: float = 0.0

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("corridor must contain at least one cell")
        if not self.segment_cells:
            raise ValueError("segment_cells must not be empty")
        normalized_indices: list[int] = []
        for i, cell_index in enumerate(self.segment_cells):
            try:
                index = operator.index(cell_index)
            except TypeError as exc:
                raise ValueError(
                    f"segment {i} cell index must be an integer, got {cell_index!r}"
                ) from exc
            if not 0 <= index < len(self.cells):
                raise ValueError(f"segment {i} has invalid cell index {cell_index}")
            normalized_indices.append(index)
        object.__setattr__(self, "segment_cells", tuple(normalized_indices))
        if not math.isfinite(self.clearance) or self.clearance < 0.0:
            raise ValueError("clearance must be finite and nonnegative")
        object.__setattr__(self, "clearance", float(self.clearance))

    @property
    def n_segments(self) -> int:
        return len(self.segment_cells)


# =============================================================================
# Stable scalar polynomial/root helpers
# =============================================================================


def _poly3(a3: float, a2: float, a1: float, a0: float, x: float) -> float:
    return math.fma(math.fma(math.fma(a3, x, a2), x, a1), x, a0)


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """All finite real roots with stable cancellation handling."""
    scale = max(1.0, abs(a), abs(b), abs(c))
    tol = 32.0 * math.ulp(scale)
    if abs(a) <= tol:
        if abs(b) <= tol:
            return []
        r = -c / b
        return [r] if math.isfinite(r) else []
    disc = math.fma(-4.0 * a, c, b * b)
    disc_tol = 64.0 * math.ulp(max(1.0, b * b, abs(4.0 * a * c)))
    if disc < -disc_tol:
        return []
    root = math.sqrt(max(0.0, disc))
    if root == 0.0:
        r = -0.5 * b / a
        return [r] if math.isfinite(r) else []
    q = -0.5 * (b + math.copysign(root, b))
    roots = [q / a]
    if q != 0.0:
        roots.append(c / q)
    roots = [r for r in roots if math.isfinite(r)]
    roots.sort()
    if len(roots) == 2 and abs(roots[1] - roots[0]) <= 32.0 * math.ulp(max(1.0, abs(roots[0]), abs(roots[1]))):
        roots.pop()
    return roots


def _bracketed_root(
    f: Callable[[float], float],
    df: Callable[[float], float] | None,
    lo: float,
    hi: float,
    f_lo: float | None = None,
    f_hi: float | None = None,
    *,
    x_tol: float = 2.0e-13,
    f_tol: float = 2.0e-13,
    max_iter: int = 80,
) -> float:
    if hi < lo:
        lo, hi = hi, lo
        f_lo, f_hi = f_hi, f_lo
    a = float(lo)
    b = float(hi)
    fa = f(a) if f_lo is None else float(f_lo)
    fb = f(b) if f_hi is None else float(f_hi)
    if abs(fa) <= f_tol:
        return a
    if abs(fb) <= f_tol:
        return b
    if fa * fb > 0.0:
        raise ValueError("root is not bracketed")
    x = 0.5 * (a + b)
    for _ in range(max_iter):
        fx = f(x)
        if not math.isfinite(fx):
            x = 0.5 * (a + b)
            fx = f(x)
        if abs(fx) <= f_tol:
            return x
        if fa * fx <= 0.0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        mid = 0.5 * (a + b)
        if b - a <= x_tol * max(1.0, abs(mid)):
            return mid
        if df is not None:
            dfx = df(x)
            if math.isfinite(dfx) and dfx != 0.0:
                xn = x - fx / dfx
                if a < xn < b and math.isfinite(xn):
                    x = xn
                    continue
        if fb != fa:
            xs = b - fb * (b - a) / (fb - fa)
            if a < xs < b and math.isfinite(xs):
                x = xs
                continue
        x = mid
    return 0.5 * (a + b)


def _cubic_roots_in_interval(
    a3: float,
    a2: float,
    a1: float,
    a0: float,
    lo: float,
    hi: float,
) -> list[float]:
    """Exhaustively isolate all real cubic roots on ``[lo, hi]``.

    The derivative's real roots partition the cubic into monotone intervals.
    Repeated roots are detected at derivative critical points.
    """
    if hi < lo:
        lo, hi = hi, lo
    points = [float(lo), float(hi)]
    for r in _quadratic_roots(3.0 * a3, 2.0 * a2, a1):
        if lo < r < hi:
            points.append(r)
    points.sort()

    max_x = max(1.0, abs(lo), abs(hi))
    scale = abs(a3) * max_x**3 + abs(a2) * max_x**2 + abs(a1) * max_x + abs(a0) + 1.0
    f_tol = 2.0e-13 * scale
    roots: list[float] = []

    def add(r: float) -> None:
        if lo - 1e-14 * max_x <= r <= hi + 1e-14 * max_x:
            r = min(hi, max(lo, r))
            if not any(abs(r - old) <= 2.0e-12 * max(1.0, abs(r), abs(old)) for old in roots):
                roots.append(r)

    values = [_poly3(a3, a2, a1, a0, x) for x in points]
    for x, fx in zip(points, values):
        if abs(fx) <= f_tol:
            add(x)
    derivative = lambda x: math.fma(math.fma(3.0 * a3, x, 2.0 * a2), x, a1)
    polynomial = lambda x: _poly3(a3, a2, a1, a0, x)
    for xa, xb, fa, fb in zip(points[:-1], points[1:], values[:-1], values[1:]):
        if fa * fb < 0.0:
            add(_bracketed_root(polynomial, derivative, xa, xb, fa, fb, f_tol=f_tol))
    roots.sort()
    return roots


def _unique_sorted(values: Iterable[float], *, tol: float = 2.0e-12) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not out or abs(value - out[-1]) > tol * max(1.0, abs(value), abs(out[-1])):
            out.append(value)
    return out


# =============================================================================
# Geometry value/Jacobian cache
# =============================================================================


class PoseJacobian(NamedTuple):
    state: GeometryState
    jx: Array
    jy: Array
    jtheta: Array


@dataclass(slots=True)
class KnotGeometryCache:
    initial_state: GeometryState | Sequence[float]
    initial_s: float = 0.0
    _x: Array | None = field(default=None, init=False, repr=False)
    _raw: list[float] | None = field(default=None, init=False, repr=False)
    _path: GeometryPath | None = field(default=None, init=False, repr=False)
    _point_cache: dict[tuple[int, float], PoseJacobian] = field(default_factory=dict, init=False, repr=False)
    _endpoint_cache: tuple[GeometryState, tuple[Array, Array, Array, Array]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.initial_state) != 4:
            raise ValueError("initial_state must be (x0, y0, theta0, k0)")
        self.initial_state = GeometryState(*map(float, self.initial_state))
        if not all(math.isfinite(v) for v in self.initial_state):
            raise ValueError("initial_state must be finite")
        self.initial_s = float(self.initial_s)
        if not math.isfinite(self.initial_s):
            raise ValueError("initial_s must be finite")

    def update(self, knot_params: Sequence[float]) -> GeometryPath:
        x = np.asarray(knot_params, dtype=float)
        if x.ndim != 1:
            raise ValueError("knot parameter vector must be one-dimensional")
        if self._x is not None and np.array_equal(x, self._x):
            assert self._path is not None
            return self._path
        raw = knot_parameters_to_raw(
            x,
            initial_k=self.initial_state.k,
            initial_s=self.initial_s,
        )
        path = compile_geometry_path(raw, self.initial_state)
        self._x = x.copy()
        self._raw = raw
        self._path = path
        self._point_cache.clear()
        self._endpoint_cache = None
        return path

    @property
    def knot_params(self) -> Array:
        if self._x is None:
            raise RuntimeError("geometry cache has not been initialized")
        return self._x

    @property
    def raw_params(self) -> list[float]:
        if self._raw is None:
            raise RuntimeError("geometry cache has not been initialized")
        return self._raw

    @property
    def path(self) -> GeometryPath:
        if self._path is None:
            raise RuntimeError("geometry cache has not been initialized")
        return self._path

    def point_pose_jacobian(self, segment: int, tau: float) -> PoseJacobian:
        tau = float(tau)
        key = (int(segment), tau)
        cached = self._point_cache.get(key)
        if cached is not None:
            return cached
        path = self.path
        point, raw_jx, raw_jy = path.point_xy_jacobian(segment, tau)
        state, raw_jtheta = path.point_pose_vjp(segment, tau, (0.0, 0.0, 1.0, 0.0))
        raw = self.raw_params
        jx = np.asarray(pullback_raw_gradient_from_precomputed_raw(raw, raw_jx), dtype=float)
        jy = np.asarray(pullback_raw_gradient_from_precomputed_raw(raw, raw_jy), dtype=float)
        jtheta = np.asarray(pullback_raw_gradient_from_precomputed_raw(raw, raw_jtheta), dtype=float)
        # point_xy_jacobian and point_pose_vjp use the same state; use the latter.
        if abs(point[0] - state.x) > 2e-12 or abs(point[1] - state.y) > 2e-12:
            raise AssertionError("geometry point evaluators disagree")
        result = PoseJacobian(state, jx, jy, jtheta)
        self._point_cache[key] = result
        return result

    def endpoint_pose_jacobian(self) -> tuple[GeometryState, tuple[Array, Array, Array, Array]]:
        if self._endpoint_cache is not None:
            return self._endpoint_cache
        path = self.path
        raw_rows = path.endpoint_jacobian()
        raw = self.raw_params
        row_x, row_y, row_theta, row_k = (
            np.asarray(pullback_raw_gradient_from_precomputed_raw(raw, row), dtype=float)
            for row in raw_rows
        )
        rows = (row_x, row_y, row_theta, row_k)
        self._endpoint_cache = (path.final_state, rows)
        return self._endpoint_cache


# =============================================================================
# Finite smooth rectangular constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConstraintFamily:
    segment: int
    wall: int
    corner: int


@dataclass(frozen=True, slots=True)
class PointConstraint:
    segment: int
    wall: int
    corner: int
    tau: float

    @property
    def family(self) -> ConstraintFamily:
        return ConstraintFamily(self.segment, self.wall, self.corner)


@dataclass(slots=True)
class ConstraintPool:
    """Persistent normalized-coordinate cuts grouped by constraint family."""

    corridor: CorridorModel
    taus: dict[ConstraintFamily, list[float]] = field(default_factory=dict)

    @classmethod
    def seeded(
        cls,
        corridor: CorridorModel,
        *,
        include_midpoints: bool = False,
    ) -> "ConstraintPool":
        pool = cls(corridor)
        seeds = (0.0, 0.5, 1.0) if include_midpoints else (0.0, 1.0)
        for i, cell_index in enumerate(corridor.segment_cells):
            cell = corridor.cells[cell_index]
            for wall_index in range(len(cell.walls)):
                for corner_index in range(4):
                    pool.taus[ConstraintFamily(i, wall_index, corner_index)] = list(seeds)
        return pool

    def copy(self) -> "ConstraintPool":
        return ConstraintPool(self.corridor, {family: list(values) for family, values in self.taus.items()})

    def entries(self) -> list[PointConstraint]:
        return [
            PointConstraint(f.segment, f.wall, f.corner, tau)
            for f in sorted(self.taus, key=lambda q: (q.segment, q.wall, q.corner))
            for tau in self.taus[f]
        ]

    @property
    def size(self) -> int:
        return sum(len(values) for values in self.taus.values())

    def add(
        self,
        family: ConstraintFamily,
        tau: float,
        *,
        segment_length: float,
        merge_distance: float,
    ) -> bool:
        if not 0 <= family.segment < self.corridor.n_segments:
            raise ValueError(f"invalid constraint segment {family.segment}")
        cell = self.corridor.cells[
            self.corridor.segment_cells[family.segment]
        ]
        if not 0 <= family.wall < len(cell.walls):
            raise ValueError(
                f"invalid wall index {family.wall} for segment {family.segment}"
            )
        if not 0 <= family.corner < len(self.corridor.body.corners):
            raise ValueError(f"invalid corner index {family.corner}")
        tau = min(1.0, max(0.0, float(tau)))
        values = self.taus.setdefault(family, [])
        tau_tol = merge_distance / max(float(segment_length), 1.0e-300)
        if any(abs(tau - old) <= tau_tol for old in values):
            return False
        values.append(tau)
        values.sort()
        return True


@dataclass(slots=True)
class _ConstraintPoseGroupBuilder:
    rows: list[int] = field(default_factory=list)
    u: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    nx: list[float] = field(default_factory=list)
    ny: list[float] = field(default_factory=list)
    effective_offset: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ConstraintPoseGroup:
    """All finite rows sharing one geometry pose evaluation."""

    segment: int
    tau: float
    rows: NDArray[np.intp]
    u: Array
    v: Array
    nx: Array
    ny: Array
    effective_offset: Array


@dataclass(slots=True)
class RectangleConstraintOracle:
    corridor: CorridorModel
    pool: ConstraintPool
    cache: KnotGeometryCache
    calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    _x: Array | None = field(default=None, init=False, repr=False)
    _values: Array | None = field(default=None, init=False, repr=False)
    _jacobian: Array | None = field(default=None, init=False, repr=False)
    _sparse_jacobian: object | None = field(default=None, init=False, repr=False)
    _groups: tuple[_ConstraintPoseGroup, ...] = field(default=(), init=False, repr=False)
    _row_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.pool.corridor != self.corridor:
            raise ValueError("constraint pool belongs to a different corridor")

        # The pool is fixed for the lifetime of one finite solve.  Compile its
        # stable row layout once, while grouping rows that share (segment, tau).
        # This preserves the public row ordering produced by entries().
        corners = self.corridor.body.corners
        clearance = self.corridor.clearance
        grouped: dict[tuple[int, float], _ConstraintPoseGroupBuilder] = {}
        row = 0
        for family in sorted(self.pool.taus, key=lambda q: (q.segment, q.wall, q.corner)):
            if not 0 <= family.segment < self.corridor.n_segments:
                raise ValueError(f"invalid constraint segment {family.segment}")
            cell = self.corridor.cells[self.corridor.segment_cells[family.segment]]
            if not 0 <= family.wall < len(cell.walls):
                raise ValueError(f"invalid wall index {family.wall} for segment {family.segment}")
            if not 0 <= family.corner < len(corners):
                raise ValueError(f"invalid corner index {family.corner}")
            wall = cell.walls[family.wall]
            corner = corners[family.corner]
            for tau_in in self.pool.taus[family]:
                tau = float(tau_in)
                if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
                    raise ValueError("constraint tau must be finite and lie in [0, 1]")
                data = grouped.setdefault(
                    (family.segment, tau), _ConstraintPoseGroupBuilder()
                )
                data.rows.append(row)
                data.u.append(corner.u)
                data.v.append(corner.v)
                data.nx.append(wall.normal[0])
                data.ny.append(wall.normal[1])
                data.effective_offset.append(wall.offset - clearance)
                row += 1

        groups: list[_ConstraintPoseGroup] = []
        for (segment, tau), data in grouped.items():
            groups.append(
                _ConstraintPoseGroup(
                    segment,
                    tau,
                    np.asarray(data.rows, dtype=np.intp),
                    np.asarray(data.u, dtype=float),
                    np.asarray(data.v, dtype=float),
                    np.asarray(data.nx, dtype=float),
                    np.asarray(data.ny, dtype=float),
                    np.asarray(data.effective_offset, dtype=float),
                )
            )
        self._groups = tuple(groups)
        self._row_count = row

    @property
    def pose_groups(self) -> int:
        """Number of unique ``(segment, tau)`` evaluations in the pool."""
        return len(self._groups)

    def evaluate(self, knot_params: Sequence[float]) -> tuple[Array, Array]:
        x = np.asarray(knot_params, dtype=float)
        if self._x is not None and np.array_equal(x, self._x):
            self.cache_hits += 1
            assert self._values is not None
            if self._jacobian is None:
                assert self._sparse_jacobian is not None
                self._jacobian = self._sparse_jacobian.toarray()
            return self._values, self._jacobian
        started = time.perf_counter()
        self.calls += 1
        path = self.cache.update(x)
        if path.n_segments != self.corridor.n_segments:
            raise ValueError("corridor assignment length does not match path segment count")
        values = np.empty(self._row_count, dtype=float)
        jacobian = np.empty((self._row_count, x.size), dtype=float)

        for group in self._groups:
            pose = self.cache.point_pose_jacobian(group.segment, group.tau)
            theta = pose.state.theta
            c = math.cos(theta)
            s = math.sin(theta)

            corner_x = pose.state.x + group.u * c - group.v * s
            corner_y = pose.state.y + group.u * s + group.v * c
            values[group.rows] = group.nx * corner_x + group.ny * corner_y - group.effective_offset

            dcorner_dtheta_x = -group.u * s - group.v * c
            dcorner_dtheta_y = group.u * c - group.v * s
            theta_seed = group.nx * dcorner_dtheta_x + group.ny * dcorner_dtheta_y
            coefficients = np.column_stack((group.nx, group.ny, theta_seed))
            pose_jacobian = np.vstack((pose.jx, pose.jy, pose.jtheta))
            jacobian[group.rows] = coefficients @ pose_jacobian

        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("nonfinite rectangular constraint evaluation")
        self.seconds += time.perf_counter() - started
        self._x = x.copy()
        self._values = values
        self._jacobian = jacobian
        self._sparse_jacobian = None
        return values, jacobian

    def evaluate_sparse(self, knot_params: Sequence[float]):
        """Evaluate finite rectangle rows with a CSR Jacobian.

        A pose on segment ``j`` depends only on the knot prefix through that
        segment.  Building those prefixes directly avoids allocating the
        dense rectangular Jacobian used by SLSQP.
        """
        from scipy.sparse import coo_matrix

        x = np.asarray(knot_params, dtype=float)
        if self._x is not None and np.array_equal(x, self._x):
            assert self._values is not None
            if self._sparse_jacobian is None:
                assert self._jacobian is not None
                from scipy.sparse import csr_matrix
                self._sparse_jacobian = csr_matrix(self._jacobian)
            return self._values, self._sparse_jacobian
        started = time.perf_counter()
        self.calls += 1
        path = self.cache.update(x)
        if path.n_segments != self.corridor.n_segments:
            raise ValueError(
                "corridor assignment length does not match path segment count"
            )
        values = np.empty(self._row_count, dtype=float)
        data_parts: list[Array] = []
        row_parts: list[NDArray[np.intp]] = []
        col_parts: list[NDArray[np.intp]] = []

        for group in self._groups:
            pose = self.cache.point_pose_jacobian(group.segment, group.tau)
            theta = pose.state.theta
            c = math.cos(theta)
            s = math.sin(theta)
            corner_x = pose.state.x + group.u * c - group.v * s
            corner_y = pose.state.y + group.u * s + group.v * c
            values[group.rows] = (
                group.nx * corner_x
                + group.ny * corner_y
                - group.effective_offset
            )
            dcorner_dtheta_x = -group.u * s - group.v * c
            dcorner_dtheta_y = group.u * c - group.v * s
            theta_seed = (
                group.nx * dcorner_dtheta_x
                + group.ny * dcorner_dtheta_y
            )
            coefficients = np.column_stack((group.nx, group.ny, theta_seed))
            prefix = min(x.size, 2 * (group.segment + 1))
            pose_jacobian = np.vstack(
                (pose.jx[:prefix], pose.jy[:prefix], pose.jtheta[:prefix])
            )
            block = coefficients @ pose_jacobian
            n_rows = int(group.rows.size)
            row_parts.append(np.repeat(group.rows, prefix))
            col_parts.append(
                np.tile(np.arange(prefix, dtype=np.intp), n_rows)
            )
            data_parts.append(block.reshape(-1))

        if data_parts:
            data = np.concatenate(data_parts)
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            jacobian = coo_matrix(
                (data, (rows, cols)), shape=(self._row_count, x.size)
            ).tocsr()
            jacobian.eliminate_zeros()
        else:
            from scipy.sparse import csr_matrix
            jacobian = csr_matrix((self._row_count, x.size), dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(
            np.isfinite(jacobian.data)
        ):
            raise FloatingPointError(
                "nonfinite rectangular constraint evaluation"
            )
        self.seconds += time.perf_counter() - started
        self._x = x.copy()
        self._values = values
        self._jacobian = None
        self._sparse_jacobian = jacobian
        return values, jacobian

    def values(self, knot_params: Sequence[float]) -> Array:
        return self.evaluate(knot_params)[0]

    def jacobian(self, knot_params: Sequence[float]) -> Array:
        return self.evaluate(knot_params)[1]


@dataclass(slots=True)
class EndpointPoseEquality:
    """Selected final-pose equalities using the shared geometry cache.

    ``target`` may contain ``None`` for unconstrained components in the order
    ``(x, y, theta, k)``.
    """

    cache: KnotGeometryCache
    target: tuple[float | None, float | None, float | None, float | None]
    periodic_heading: bool = False

    def __post_init__(self) -> None:
        if len(self.target) != 4:
            raise ValueError("target must have four components")
        for value in self.target:
            if value is not None and not math.isfinite(value):
                raise ValueError("endpoint targets must be finite or None")

    def __call__(self, knot_params: Sequence[float]) -> tuple[Array, Array]:
        self.cache.update(knot_params)
        state, rows = self.cache.endpoint_pose_jacobian()
        values: list[float] = []
        jac: list[Array] = []
        for component, (value, target, row) in enumerate(
            zip(state, self.target, rows)
        ):
            if target is not None:
                residual = float(value) - target
                if component == 2 and self.periodic_heading:
                    # Phase I must use the same periodic heading semantics as
                    # the authoritative planner certificate.  Later local
                    # polishing retains its historical branch-aligned residual
                    # to avoid changing established continuation trajectories.
                    residual = math.atan2(math.sin(residual), math.cos(residual))
                values.append(residual)
                jac.append(row)
        if not values:
            return np.empty(0, dtype=float), np.empty((0, len(knot_params)), dtype=float)
        return np.asarray(values, dtype=float), np.vstack(jac)


@dataclass(slots=True)
class EndpointBoxConstraint:
    """Keep the final path reference point inside an axis-aligned box.

    The package inequality convention is ``c <= 0``.  Heading and curvature
    are deliberately unconstrained; this is intended for terminal regions such
    as a micromouse goal cell where the robot only needs to enter the cell.
    """

    initial_state: GeometryState
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    initial_s: float = 0.0
    _cache: KnotGeometryCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.xmin, self.xmax, self.ymin, self.ymax, self.initial_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("endpoint box values must be finite")
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise ValueError("endpoint box must be nonempty")
        self._cache = KnotGeometryCache(self.initial_state, self.initial_s)

    def __call__(self, knot_params: Sequence[float]) -> tuple[Array, Array]:
        x = np.asarray(knot_params, dtype=float)
        self._cache.update(x)
        state, rows = self._cache.endpoint_pose_jacobian()
        jx, jy, _jtheta, _jk = rows
        values = np.asarray((
            self.xmin - state.x,
            state.x - self.xmax,
            self.ymin - state.y,
            state.y - self.ymax,
        ), dtype=float)
        jacobian = np.vstack((-jx, jx, -jy, jy))
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("endpoint box constraint produced nonfinite data")
        return values, jacobian


@dataclass(frozen=True, slots=True)
class CurvatureSlopeConstraint:
    """Two-sided bounds on every linear-curvature slope.

    For knot parameters ``[s1,k1,...]`` and fixed ``(s0,k0)``, segment ``i``
    has ``sigma_i = (k_{i+1}-k_i)/(s_{i+1}-s_i)``.  The returned inequality
    rows use the package convention ``c <= 0`` and enforce
    ``abs(sigma_i) <= maximum_abs_sigma`` with an exact knot-basis Jacobian.
    """

    maximum_abs_sigma: float
    initial_k: float = 0.0
    initial_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_abs_sigma)
            or self.maximum_abs_sigma <= 0.0
        ):
            raise ValueError("maximum_abs_sigma must be finite and positive")
        if not math.isfinite(self.initial_k) or not math.isfinite(self.initial_s):
            raise ValueError("initial knot must be finite")

    def __call__(self, knot_params: Sequence[float]) -> tuple[Array, Array]:
        x = np.asarray(knot_params, dtype=float)
        if x.ndim != 1 or x.size == 0 or x.size % 2:
            raise ValueError("knot parameters must be a nonempty [s1,k1,...] vector")
        if not np.all(np.isfinite(x)):
            raise ValueError("knot parameters must be finite")

        stations = x[0::2]
        curvatures = x[1::2]
        previous_stations = np.concatenate(
            ([float(self.initial_s)], stations[:-1])
        )
        previous_curvatures = np.concatenate(
            ([float(self.initial_k)], curvatures[:-1])
        )
        lengths = stations - previous_stations
        if np.any(lengths <= 0.0):
            raise ValueError("knot stations must be strictly increasing")
        delta_k = curvatures - previous_curvatures
        sigma = delta_k / lengths
        n = stations.size
        values = np.empty(2 * n, dtype=float)
        values[0::2] = sigma - self.maximum_abs_sigma
        values[1::2] = -sigma - self.maximum_abs_sigma

        jacobian = np.zeros((2 * n, x.size), dtype=float)
        for i in range(n):
            inv_length = 1.0 / lengths[i]
            dsigma_ds1 = -sigma[i] * inv_length
            dsigma_dk1 = inv_length
            plus = 2 * i
            minus = plus + 1
            station_col = 2 * i
            curvature_col = station_col + 1
            jacobian[plus, station_col] = dsigma_ds1
            jacobian[plus, curvature_col] = dsigma_dk1
            jacobian[minus, station_col] = -dsigma_ds1
            jacobian[minus, curvature_col] = -dsigma_dk1
            if i:
                previous_station_col = station_col - 2
                previous_curvature_col = curvature_col - 2
                dsigma_ds0 = -dsigma_ds1
                dsigma_dk0 = -dsigma_dk1
                jacobian[plus, previous_station_col] = dsigma_ds0
                jacobian[plus, previous_curvature_col] = dsigma_dk0
                jacobian[minus, previous_station_col] = -dsigma_ds0
                jacobian[minus, previous_curvature_col] = -dsigma_dk0

        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("curvature-slope constraint produced nonfinite data")
        return values, jacobian


@dataclass(frozen=True, slots=True)
class StackedVectorConstraint:
    """Stack several vector constraints into one callback."""

    constraints: tuple[VectorConstraint, ...]

    def __call__(self, x: Sequence[float]) -> tuple[Array, Array]:
        values: list[Array] = []
        jacobians: list[Array] = []
        width = len(x)
        for constraint in self.constraints:
            value, jacobian = constraint(x)
            value_array = np.asarray(value, dtype=float)
            jacobian_array = np.asarray(jacobian, dtype=float)
            if jacobian_array.shape != (value_array.size, width):
                raise ValueError("stacked constraint Jacobian has wrong shape")
            if value_array.size:
                values.append(value_array)
                jacobians.append(jacobian_array)
        if not values:
            return np.empty(0, dtype=float), np.empty((0, width), dtype=float)
        return np.concatenate(values), np.vstack(jacobians)


def stack_vector_constraints(*constraints: VectorConstraint | None) -> VectorConstraint | None:
    active = tuple(c for c in constraints if c is not None)
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return StackedVectorConstraint(active)


# =============================================================================
# Exact continuous separation
# =============================================================================


@dataclass(frozen=True, slots=True)
class SeparationMaximum:
    family: ConstraintFamily
    tau: float
    violation: float
    center: Float2
    corner_point: Float2
    stationary: bool


@dataclass(frozen=True, slots=True)
class WallSeparation:
    segment: int
    wall: int
    pruned_by_circle_bound: bool
    upper_bound: float
    maxima: tuple[SeparationMaximum, ...]

    @property
    def worst_violation(self) -> float:
        if self.maxima:
            return max(m.violation for m in self.maxima)
        return self.upper_bound


@dataclass(frozen=True, slots=True)
class SeparationReport:
    walls: tuple[WallSeparation, ...]
    violating_maxima: tuple[SeparationMaximum, ...]
    worst_upper_bound: float
    worst_exact_violation: float
    pruned_walls: int
    exact_corner_families: int

    def certified(self, tolerance: float) -> bool:
        return self.worst_upper_bound <= tolerance


@dataclass(frozen=True, slots=True)
class SeparationSettings:
    # Any violation too large to certify must also be eligible to become a cut.
    add_tolerance: float = 1.0e-9
    certificate_tolerance: float = 1.0e-9
    broad_phase_tolerance: float = 2.0e-13
    root_x_tolerance: float = 2.0e-13
    root_function_tolerance: float = 2.0e-13
    maximum_stationary_points_per_family: int = 4096

    def __post_init__(self) -> None:
        values = (
            self.add_tolerance,
            self.certificate_tolerance,
            self.broad_phase_tolerance,
            self.root_x_tolerance,
            self.root_function_tolerance,
        )
        if not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("separation tolerances must be finite and nonnegative")
        if self.add_tolerance > self.certificate_tolerance:
            raise ValueError("add_tolerance must not exceed certificate_tolerance")
        if self.maximum_stationary_points_per_family <= 0:
            raise ValueError("maximum_stationary_points_per_family must be positive")


def _theta_at(theta0: float, k0: float, sigma: float, s: float) -> float:
    return math.fma(s, math.fma(0.5 * sigma, s, k0), theta0)


def _k_at(k0: float, sigma: float, s: float) -> float:
    return math.fma(sigma, s, k0)


def _heading_stationary_taus(
    theta0: float,
    k0: float,
    sigma: float,
    length: float,
    normal_angle: float,
    *,
    maximum_points: int | None = None,
) -> list[float]:
    theta_values = [theta0, _theta_at(theta0, k0, sigma, length)]
    if sigma != 0.0:
        vertex = -k0 / sigma
        if 0.0 < vertex < length:
            theta_values.append(_theta_at(theta0, k0, sigma, vertex))
    lo = min(theta_values)
    hi = max(theta_values)
    base = normal_angle + _HALF_PI
    eps = 4.0e-13
    m_lo = math.ceil((lo - base) / _PI - eps)
    m_hi = math.floor((hi - base) / _PI + eps)
    count = max(0, m_hi - m_lo + 1)
    if maximum_points is not None and count > maximum_points:
        raise ArithmeticError("too many centerline stationary points")
    roots: list[float] = []
    for m in range(m_lo, m_hi + 1):
        target = base + m * _PI
        for s in _quadratic_roots(0.5 * sigma, k0, theta0 - target):
            if 0.0 < s < length:
                roots.append(s / length)
    return _unique_sorted(roots)


def _corner_tangent_angle(
    theta0: float,
    k0: float,
    sigma: float,
    s: float,
    u: float,
    v: float,
) -> float:
    k = _k_at(k0, sigma, s)
    a0 = math.fma(-v, k0, 1.0)
    b0 = u * k0
    a = math.fma(-v, k, 1.0)
    b = u * k
    delta0 = math.atan2(b0, a0)
    cross = math.fma(a0, b, -b0 * a)
    dot = math.fma(a0, a, b0 * b)
    delta = math.atan2(cross, dot)
    return _theta_at(theta0, k0, sigma, s) + delta0 + delta


def _corner_tangent_angle_derivative(k: float, sigma: float, u: float, v: float) -> float:
    a = math.fma(-v, k, 1.0)
    b = u * k
    d = math.fma(a, a, b * b)
    return k + u * sigma / d


def _corner_monotonic_partitions(
    k0: float,
    sigma: float,
    length: float,
    corner: BodyCorner,
) -> tuple[float, ...]:
    """Wall-independent monotonicity intervals of the corner tangent angle."""
    u = corner.u
    if u == 0.0 or sigma == 0.0:
        return (0.0, length)
    v = corner.v
    radius2 = math.fma(u, u, v * v)
    k1 = _k_at(k0, sigma, length)
    k_lo, k_hi = sorted((k0, k1))
    critical_k = _cubic_roots_in_interval(
        radius2, -2.0 * v, 1.0, u * sigma, k_lo, k_hi
    )
    partitions = [0.0, length]
    for kr in critical_k:
        s = (kr - k0) / sigma
        if 0.0 < s < length:
            partitions.append(s)
    return tuple(_unique_sorted(partitions))


def _corner_stationary_taus(
    theta0: float,
    k0: float,
    sigma: float,
    length: float,
    normal_angle: float,
    corner: BodyCorner,
    settings: SeparationSettings,
    *,
    partitions: Sequence[float] | None = None,
) -> list[float]:
    u = corner.u
    v = corner.v
    if u == 0.0:
        roots = _heading_stationary_taus(
            theta0,
            k0,
            sigma,
            length,
            normal_angle,
            maximum_points=settings.maximum_stationary_points_per_family,
        )
        if v != 0.0 and sigma != 0.0:
            zero_velocity_s = (1.0 / v - k0) / sigma
            if 0.0 < zero_velocity_s < length:
                roots.append(zero_velocity_s / length)
        return _unique_sorted(roots)

    if sigma == 0.0:
        # Constant curvature means the corner tangent offset is constant.
        k = k0
        a = math.fma(-v, k, 1.0)
        b = u * k
        chi0 = theta0 + math.atan2(b, a)
        if k == 0.0:
            return []
        lo = min(chi0, chi0 + k * length)
        hi = max(chi0, chi0 + k * length)
        base = normal_angle + _HALF_PI
        m_lo = math.ceil((lo - base) / _PI - 4e-13)
        m_hi = math.floor((hi - base) / _PI + 4e-13)
        if m_hi - m_lo + 1 > settings.maximum_stationary_points_per_family:
            raise ArithmeticError("too many rectangular corner stationary points")
        roots = []
        for m in range(m_lo, m_hi + 1):
            s = (base + m * _PI - chi0) / k
            if 0.0 < s < length:
                roots.append(s / length)
        return _unique_sorted(roots)

    if partitions is None:
        partitions = _corner_monotonic_partitions(k0, sigma, length, corner)

    base = normal_angle + _HALF_PI
    roots: list[float] = []
    for sa, sb in zip(partitions[:-1], partitions[1:]):
        chi_a = _corner_tangent_angle(theta0, k0, sigma, sa, u, v)
        chi_b = _corner_tangent_angle(theta0, k0, sigma, sb, u, v)
        lo = min(chi_a, chi_b)
        hi = max(chi_a, chi_b)
        m_lo = math.ceil((lo - base) / _PI - 4e-13)
        m_hi = math.floor((hi - base) / _PI + 4e-13)
        if len(roots) + max(0, m_hi - m_lo + 1) > settings.maximum_stationary_points_per_family:
            raise ArithmeticError("too many rectangular corner stationary points")
        for m in range(m_lo, m_hi + 1):
            target = base + m * _PI

            def f(s: float) -> float:
                return _corner_tangent_angle(theta0, k0, sigma, s, u, v) - target

            def df(s: float) -> float:
                return _corner_tangent_angle_derivative(_k_at(k0, sigma, s), sigma, u, v)

            fa = chi_a - target
            fb = chi_b - target
            f_tol = settings.root_function_tolerance * max(1.0, abs(target), abs(chi_a), abs(chi_b))
            if abs(fa) <= f_tol:
                root = sa
            elif abs(fb) <= f_tol:
                root = sb
            elif fa * fb < 0.0:
                root = _bracketed_root(
                    f,
                    df,
                    sa,
                    sb,
                    fa,
                    fb,
                    x_tol=settings.root_x_tolerance,
                    f_tol=f_tol,
                )
            else:
                continue
            if 0.0 < root < length:
                roots.append(root / length)
                if len(roots) > settings.maximum_stationary_points_per_family:
                    raise ArithmeticError("too many rectangular corner stationary points")
    return _unique_sorted(roots)


def _corner_point(state: GeometryState, corner: BodyCorner) -> Float2:
    c = math.cos(state.theta)
    s = math.sin(state.theta)
    return (
        math.fma(corner.u, c, math.fma(-corner.v, s, state.x)),
        math.fma(corner.u, s, math.fma(corner.v, c, state.y)),
    )


def _wall_value(point: Float2, wall: HalfSpace, clearance: float) -> float:
    nx, ny = wall.normal
    return math.fma(nx, point[0], math.fma(ny, point[1], -(wall.offset - clearance)))


def _unoriented_normal_key(normal: Float2) -> Float2:
    """Canonicalize a wall normal modulo sign (the stationary set is pi-periodic)."""
    nx, ny = normal
    if nx < 0.0 or (nx == 0.0 and ny < 0.0):
        nx, ny = -nx, -ny
    # Collapse signed zero so opposite axis-aligned normals share a key.
    return (0.0 if nx == 0.0 else nx, 0.0 if ny == 0.0 else ny)


def _centerline_wall_max(
    segment: int,
    wall: HalfSpace,
    taus: Sequence[float],
    state_at: Callable[[int, float], GeometryState],
) -> float:
    nx, ny = wall.normal
    best = -math.inf
    for tau in taus:
        state = state_at(segment, tau)
        best = max(best, math.fma(nx, state.x, ny * state.y))
    return best


def _local_maxima_from_candidates(
    segment: int,
    wall_index: int,
    wall: HalfSpace,
    corner_index: int,
    corner: BodyCorner,
    taus: Sequence[float],
    clearance: float,
    state_at: Callable[[int, float], GeometryState],
) -> tuple[SeparationMaximum, ...]:
    candidates = _unique_sorted((0.0, *taus, 1.0))
    states = [state_at(segment, tau) for tau in candidates]
    points = [_corner_point(state, corner) for state in states]
    values = [_wall_value(point, wall, clearance) for point in points]
    maxima: list[SeparationMaximum] = []
    local_tol = 2.0e-13 * max(1.0, *(abs(v) for v in values))
    for j, (tau, state, point, value) in enumerate(zip(candidates, states, points, values)):
        left = values[j - 1] if j > 0 else -math.inf
        right = values[j + 1] if j + 1 < len(values) else -math.inf
        if value + local_tol >= left and value + local_tol >= right:
            maxima.append(
                SeparationMaximum(
                    ConstraintFamily(segment, wall_index, corner_index),
                    tau,
                    value,
                    (state.x, state.y),
                    point,
                    0.0 < tau < 1.0,
                )
            )
    return tuple(maxima)


def separate_compiled_path(
    path: GeometryPath,
    corridor: CorridorModel,
    *,
    settings: SeparationSettings = SeparationSettings(),
    use_broad_phase: bool = True,
) -> SeparationReport:
    """Exact rectangular-body separation for an already compiled path.

    Geometry states are memoized by ``(segment, tau)``.  Heading stationary
    sets are shared by opposite/parallel walls, and corner monotonicity
    partitions are computed once per segment/corner.
    """
    if path.n_segments != corridor.n_segments:
        raise ValueError("corridor assignment length does not match path")

    wall_reports: list[WallSeparation] = []
    violating: list[SeparationMaximum] = []
    worst_upper = -math.inf
    worst_exact = -math.inf
    pruned = 0
    exact_families = 0
    corners = corridor.body.corners
    radius = corridor.body.maximum_radius
    state_cache: dict[tuple[int, float], GeometryState] = {}

    def state_at(segment: int, tau: float) -> GeometryState:
        key = (segment, float(tau))
        state = state_cache.get(key)
        if state is None:
            state = path.state_at_fraction(segment, tau)
            state_cache[key] = state
        return state

    for i, cell_index in enumerate(corridor.segment_cells):
        cell = corridor.cells[cell_index]
        theta0 = path.thetas[i]
        k0 = path.curvatures[i]
        sigma = path.sigmas[i]
        length = path.lengths[i]

        heading_taus: dict[Float2, tuple[float, ...]] = {}
        corner_roots: dict[tuple[int, Float2], tuple[float, ...]] = {}
        corner_partitions: list[tuple[float, ...] | None] = [None] * len(corners)

        for wall_index, wall in enumerate(cell.walls):
            direction = _unoriented_normal_key(wall.normal)
            taus = heading_taus.get(direction)
            if taus is None:
                phi = math.atan2(direction[1], direction[0])
                taus = tuple(
                    _unique_sorted(
                        (
                            0.0,
                            *_heading_stationary_taus(
                                theta0,
                                k0,
                                sigma,
                                length,
                                phi,
                                maximum_points=settings.maximum_stationary_points_per_family,
                            ),
                            1.0,
                        )
                    )
                )
                heading_taus[direction] = taus
            center_max = _centerline_wall_max(i, wall, taus, state_at)
            circle_upper = center_max + radius - (wall.offset - corridor.clearance)
            if (
                use_broad_phase
                and circle_upper
                <= settings.certificate_tolerance - settings.broad_phase_tolerance
            ):
                pruned += 1
                worst_upper = max(worst_upper, circle_upper)
                wall_reports.append(WallSeparation(i, wall_index, True, circle_upper, ()))
                continue

            phi = math.atan2(direction[1], direction[0])
            maxima: list[SeparationMaximum] = []
            for corner_index, corner in enumerate(corners):
                exact_families += 1
                root_key = (corner_index, direction)
                roots = corner_roots.get(root_key)
                if roots is None:
                    partitions = corner_partitions[corner_index]
                    if partitions is None:
                        partitions = _corner_monotonic_partitions(
                            k0, sigma, length, corner
                        )
                        corner_partitions[corner_index] = partitions
                    roots = tuple(
                        _corner_stationary_taus(
                            theta0,
                            k0,
                            sigma,
                            length,
                            phi,
                            corner,
                            settings,
                            partitions=partitions,
                        )
                    )
                    corner_roots[root_key] = roots
                family_maxima = _local_maxima_from_candidates(
                    i,
                    wall_index,
                    wall,
                    corner_index,
                    corner,
                    roots,
                    corridor.clearance,
                    state_at,
                )
                maxima.extend(family_maxima)
                for maximum in family_maxima:
                    worst_exact = max(worst_exact, maximum.violation)
                    if maximum.violation > settings.add_tolerance:
                        violating.append(maximum)
            exact_worst = max((m.violation for m in maxima), default=-math.inf)
            worst_upper = max(worst_upper, exact_worst)
            wall_reports.append(
                WallSeparation(i, wall_index, False, exact_worst, tuple(maxima))
            )

    return SeparationReport(
        tuple(wall_reports),
        tuple(sorted(violating, key=lambda m: m.violation, reverse=True)),
        worst_upper,
        worst_exact,
        pruned,
        exact_families,
    )


def separate_rectangle_path(
    knot_params: Sequence[float],
    initial_state: GeometryState | Sequence[float],
    corridor: CorridorModel,
    *,
    initial_s: float = 0.0,
    settings: SeparationSettings = SeparationSettings(),
    use_broad_phase: bool = True,
) -> SeparationReport:
    """Compile a knot-basis path and run exact rectangular separation."""
    raw = knot_parameters_to_raw(
        knot_params,
        initial_k=float(initial_state[3]),
        initial_s=initial_s,
    )
    path = compile_geometry_path(raw, initial_state)
    return separate_compiled_path(
        path, corridor, settings=settings, use_broad_phase=use_broad_phase
    )


def add_report_violations(
    pool: ConstraintPool,
    report: SeparationReport,
    lengths: Sequence[float],
    *,
    merge_distance: float = 1.0e-8,
) -> int:
    if not math.isfinite(merge_distance) or merge_distance < 0.0:
        raise ValueError("merge_distance must be finite and nonnegative")
    added = 0
    for maximum in report.violating_maxima:
        added += int(
            pool.add(
                maximum.family,
                maximum.tau,
                segment_length=float(lengths[maximum.family.segment]),
                merge_distance=merge_distance,
            )
        )
    return added



# =============================================================================
# Combined finite constraint adapter
# =============================================================================

@dataclass(slots=True)
class CombinedConstraintOracle:
    rectangle: RectangleConstraintOracle
    equalities: VectorConstraint | None = None
    additional_inequalities: VectorConstraint | None = None
    _x: Array | None = field(default=None, init=False, repr=False)
    _c: Array | None = field(default=None, init=False, repr=False)
    _jc: Array | None = field(default=None, init=False, repr=False)
    _h: Array | None = field(default=None, init=False, repr=False)
    _jh: Array | None = field(default=None, init=False, repr=False)
    _sparse_result: tuple[Array, object, Array, object] | None = field(
        default=None, init=False, repr=False
    )

    def evaluate(self, x: Sequence[float]) -> tuple[Array, Array, Array, Array]:
        array = np.asarray(x, dtype=float)
        if self._x is not None and np.array_equal(array, self._x):
            assert self._c is not None and self._h is not None
            if self._jc is None or self._jh is None:
                assert self._sparse_result is not None
                self._jc = self._sparse_result[1].toarray()
                self._jh = self._sparse_result[3].toarray()
            return self._c, self._jc, self._h, self._jh
        c_rect, jc_rect = self.rectangle.evaluate(array)
        c_parts = [c_rect]
        jc_parts = [jc_rect]
        if self.additional_inequalities is not None:
            c_extra, jc_extra = self.additional_inequalities(array)
            c_extra = np.asarray(c_extra, dtype=float)
            jc_extra = np.asarray(jc_extra, dtype=float)
            if jc_extra.shape != (c_extra.size, array.size):
                raise ValueError("additional inequality Jacobian has wrong shape")
            c_parts.append(c_extra)
            jc_parts.append(jc_extra)
        c = np.concatenate(c_parts) if c_parts else np.empty(0, dtype=float)
        jc = np.vstack(jc_parts) if jc_parts else np.empty((0, array.size), dtype=float)
        if self.equalities is None:
            h = np.empty(0, dtype=float)
            jh = np.empty((0, array.size), dtype=float)
        else:
            h_in, jh_in = self.equalities(array)
            h = np.asarray(h_in, dtype=float)
            jh = np.asarray(jh_in, dtype=float)
            if jh.shape != (h.size, array.size):
                raise ValueError("equality Jacobian has wrong shape")
        if not all(np.all(np.isfinite(v)) for v in (c, jc, h, jh)):
            raise FloatingPointError("constraint callbacks returned nonfinite data")
        self._x = array.copy()
        self._c, self._jc, self._h, self._jh = c, jc, h, jh
        self._sparse_result = None
        return c, jc, h, jh

    def evaluate_sparse(self, x: Sequence[float]):
        from scipy.sparse import csr_matrix, vstack

        array = np.asarray(x, dtype=float)
        if self._x is not None and np.array_equal(array, self._x):
            if self._sparse_result is not None:
                return self._sparse_result
            assert (
                self._c is not None
                and self._jc is not None
                and self._h is not None
                and self._jh is not None
            )
            self._sparse_result = (
                self._c, csr_matrix(self._jc), self._h, csr_matrix(self._jh)
            )
            return self._sparse_result
        c_rect, jc_rect = self.rectangle.evaluate_sparse(array)
        c_parts = [c_rect]
        jc_parts = [jc_rect]
        if self.additional_inequalities is not None:
            c_extra, jc_extra = self.additional_inequalities(array)
            c_extra = np.asarray(c_extra, dtype=float)
            jc_extra = csr_matrix(jc_extra, dtype=float)
            if jc_extra.shape != (c_extra.size, array.size):
                raise ValueError(
                    "additional inequality Jacobian has wrong shape"
                )
            c_parts.append(c_extra)
            jc_parts.append(jc_extra)
        c = np.concatenate(c_parts) if c_parts else np.empty(0, dtype=float)
        jc = (
            vstack(jc_parts, format="csr")
            if jc_parts
            else csr_matrix((0, array.size), dtype=float)
        )
        if self.equalities is None:
            h = np.empty(0, dtype=float)
            jh = csr_matrix((0, array.size), dtype=float)
        else:
            h_in, jh_in = self.equalities(array)
            h = np.asarray(h_in, dtype=float)
            jh = csr_matrix(jh_in, dtype=float)
            if jh.shape != (h.size, array.size):
                raise ValueError("equality Jacobian has wrong shape")
        if not all(
            np.all(np.isfinite(v))
            for v in (c, jc.data, h, jh.data)
        ):
            raise FloatingPointError(
                "constraint callbacks returned nonfinite data"
            )
        self._x = array.copy()
        self._c, self._h = c, h
        self._jc = self._jh = None
        self._sparse_result = (c, jc, h, jh)
        return self._sparse_result

    def inequality_values(self, x: Sequence[float]) -> Array:
        return self.evaluate(x)[0]

    def inequality_jacobian(self, x: Sequence[float]) -> Array:
        return self.evaluate(x)[1]

    def equality_values(self, x: Sequence[float]) -> Array:
        return self.evaluate(x)[2]

    def equality_jacobian(self, x: Sequence[float]) -> Array:
        return self.evaluate(x)[3]

__all__ = [
    "BodyCorner",
    "CombinedConstraintOracle",
    "ConstraintFamily",
    "ConstraintPool",
    "CurvatureSlopeConstraint",
    "ConvexCell",
    "CorridorModel",
    "EndpointPoseEquality",
    "HalfSpace",
    "KnotGeometryCache",
    "PointConstraint",
    "PoseJacobian",
    "RectangleBody",
    "RectangleConstraintOracle",
    "ScalarObjective",
    "SeparationMaximum",
    "SeparationReport",
    "SeparationSettings",
    "StackedVectorConstraint",
    "VectorConstraint",
    "WallSeparation",
    "add_report_violations",
    "separate_compiled_path",
    "separate_rectangle_path",
    "stack_vector_constraints",
]
