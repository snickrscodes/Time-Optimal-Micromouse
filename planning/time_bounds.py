"""Certified geometric/dynamic lower bounds for topology-prefix search.

The production hierarchy combines three admissible relaxations:

* every prefix uses a cached inscribed-disk contraction of each mandatory
  portal and an exact straight-line accelerate/brake relaxation;
* a fixed 12-axis projection maximum captures mandatory scalar reversals under
  the global speed and friction-acceleration limits; and
* complete leaves close to the incumbent may additionally use the shortest
  polyline through the disk-eroded overlapping corridor cover used by the
  continuous optimizer.

Every numerical certificate is rounded toward ``-inf``.  Numerical ambiguity
always enlarges the relaxed feasible set (for example by falling back to a full
portal), so failed classification can weaken pruning but cannot remove a
feasible route.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from mazegen import Cell, Maze
from segment import SegmentType, compile_segment
from segment.constants import A_BRAKE, B_EMF, MU_G, V_MAX

Point2: TypeAlias = tuple[float, float]
Array = NDArray[np.float64]
WorldWall: TypeAlias = tuple[Point2, Point2]

_EPS = np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class Portal:
    """Closed line-segment portal between consecutive convex cells."""

    a: Point2
    b: Point2

    def __post_init__(self) -> None:
        values = (*self.a, *self.b)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("portal endpoints must be finite")
        if self.a == self.b:
            raise ValueError("portal must have positive length")

    @property
    def direction(self) -> Point2:
        return self.b[0] - self.a[0], self.b[1] - self.a[1]

    def point(self, parameter: float) -> Point2:
        parameter = float(parameter)
        return (
            math.fma(parameter, self.b[0] - self.a[0], self.a[0]),
            math.fma(parameter, self.b[1] - self.a[1], self.a[1]),
        )


@dataclass(frozen=True, slots=True)
class AxisAlignedBox:
    """Closed nonempty axis-aligned rectangle used by the leaf relaxation."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        values = (self.xmin, self.xmax, self.ymin, self.ymax)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite")
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise ValueError("box must be nonempty")

    def intersection(self, other: AxisAlignedBox) -> AxisAlignedBox | None:
        xmin = max(self.xmin, other.xmin)
        xmax = min(self.xmax, other.xmax)
        ymin = max(self.ymin, other.ymin)
        ymax = min(self.ymax, other.ymax)
        if xmin > xmax or ymin > ymax:
            return None
        return AxisAlignedBox(xmin, xmax, ymin, ymax)

    def contains(self, point: Point2) -> bool:
        return (
            self.xmin <= point[0] <= self.xmax
            and self.ymin <= point[1] <= self.ymax
        )


@dataclass(frozen=True, slots=True)
class TimeBoundRequest:
    """Geometry-independent input shared by present and future lower bounds."""

    cell_path: tuple[Cell, ...]
    start: Point2
    goal: Point2
    init_w: float
    terminal_w_max: float | None = None
    cell_size: float = 1.0
    origin: Point2 = (0.0, 0.0)
    portal_mask: tuple[bool, ...] | None = None
    goal_radius: float = 0.0

    def __post_init__(self) -> None:
        if not self.cell_path:
            raise ValueError("cell_path must not be empty")
        if not math.isfinite(self.init_w) or not 0.0 < self.init_w <= V_MAX * V_MAX:
            raise ValueError("init_w must lie in (0, V_MAX^2]")
        if self.terminal_w_max is None:
            object.__setattr__(self, "terminal_w_max", self.init_w)
        if not math.isfinite(float(self.terminal_w_max)) or not 0.0 < float(self.terminal_w_max) <= V_MAX * V_MAX:
            raise ValueError("terminal_w_max must lie in (0, V_MAX^2]")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0.0:
            raise ValueError("cell_size must be finite and positive")
        if not all(math.isfinite(value) for value in (*self.start, *self.goal, *self.origin)):
            raise ValueError("request coordinates must be finite")
        if not math.isfinite(self.goal_radius) or self.goal_radius < 0.0:
            raise ValueError("goal_radius must be finite and nonnegative")
        if self.portal_mask is not None:
            if len(self.portal_mask) != len(self.cell_path) - 1:
                raise ValueError("portal_mask must match cell-path transitions")
            if not all(isinstance(value, bool) for value in self.portal_mask):
                raise ValueError("portal_mask entries must be bool")


@dataclass(frozen=True, slots=True)
class TimeBoundResult:
    """One certified lower-bound evaluation and its relaxation diagnostics."""

    name: str
    time_lower_bound: float
    distance_lower_bound: float
    relaxed_distance_upper_bound: float
    distance_certificate_gap: float
    solve_seconds: float
    iterations: int
    converged: bool
    portal_parameters: tuple[float, ...]


class TimeLowerBound(Protocol):
    name: str

    def evaluate(self, request: TimeBoundRequest) -> TimeBoundResult: ...


class CompleteTimeLowerBound(TimeLowerBound, Protocol):
    def refine_complete(
        self,
        request: TimeBoundRequest,
        incumbent: float,
        base_result: TimeBoundResult | None = None,
    ) -> TimeBoundResult: ...


@dataclass(slots=True)
class TimeBoundStatistics:
    evaluations: int = 0
    result_cache_hits: int = 0
    gate_cache_hits: int = 0
    gate_cache_misses: int = 0
    portal_seconds: float = 0.0
    complete_refinements: int = 0
    complete_refinement_cache_hits: int = 0
    complete_refinement_skipped_by_gap: int = 0
    complete_refinement_seconds: float = 0.0
    dual_certificates: int = 0
    projection_evaluations: int = 0
    projection_strengthened: int = 0
    projection_seconds: float = 0.0
    projection_maximum_gain: float = 0.0
    projection_gate_cache_hits: int = 0
    projection_gate_cache_misses: int = 0


def portal_between_cells(
    first: Cell,
    second: Cell,
    *,
    cell_size: float = 1.0,
    origin: Point2 = (0.0, 0.0),
) -> Portal:
    """Return the full shared edge of two axis-adjacent grid cells."""
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    if abs(dx) + abs(dy) != 1:
        raise ValueError(f"cells are not axis-adjacent: {first} -> {second}")

    ox, oy = origin
    h = float(cell_size)
    x = ox + first[0] * h
    y = oy + first[1] * h
    if dx == 1:
        return Portal((x + h, y), (x + h, y + h))
    if dx == -1:
        return Portal((x, y + h), (x, y))
    if dy == 1:
        return Portal((x + h, y + h), (x, y + h))
    return Portal((x, y), (x + h, y))


def portals_from_cell_path(
    cells: Sequence[Cell],
    *,
    cell_size: float = 1.0,
    origin: Point2 = (0.0, 0.0),
) -> tuple[Portal, ...]:
    return tuple(
        portal_between_cells(
            first,
            second,
            cell_size=cell_size,
            origin=origin,
        )
        for first, second in zip(cells[:-1], cells[1:])
    )


def _distance_objective_and_gradient(
    parameters: Array,
    start: Point2,
    portals: Sequence[Portal],
    goal: Point2,
) -> tuple[float, Array]:
    count = len(portals)
    points = np.empty((count + 2, 2), dtype=float)
    points[0] = start
    points[-1] = goal
    directions = np.empty((count, 2), dtype=float)
    for i, (parameter, portal) in enumerate(zip(parameters, portals), start=1):
        ax, ay = portal.a
        dx, dy = portal.direction
        points[i] = (math.fma(float(parameter), dx, ax), math.fma(float(parameter), dy, ay))
        directions[i - 1] = (dx, dy)

    chords = points[1:] - points[:-1]
    lengths = np.hypot(chords[:, 0], chords[:, 1])
    units = np.zeros_like(chords)
    nonzero = lengths > 0.0
    units[nonzero] = chords[nonzero] / lengths[nonzero, None]
    point_gradient = units[:-1] - units[1:]
    gradient = np.einsum("ij,ij->i", point_gradient, directions)
    return float(math.fsum(map(float, lengths))), gradient


def _portal_points(
    parameters: Array,
    start: Point2,
    portals: Sequence[Portal],
    goal: Point2,
) -> tuple[Array, Array, Array]:
    count = len(portals)
    points = np.empty((count + 2, 2), dtype=float)
    points[0] = start
    points[-1] = goal
    anchors = np.empty((count, 2), dtype=float)
    directions = np.empty((count, 2), dtype=float)
    for i, (parameter, portal) in enumerate(zip(parameters, portals), start=1):
        anchors[i - 1] = portal.a
        directions[i - 1] = portal.direction
        points[i] = anchors[i - 1] + float(parameter) * directions[i - 1]
    return points, anchors, directions


def _dual_distance_lower_bound(
    parameters: Array,
    start: Point2,
    portals: Sequence[Portal],
    goal: Point2,
    *,
    maximum_iterations: int,
) -> tuple[float, int, bool]:
    """Optimize and sanitize a feasible dual certificate for the portal SOCP."""
    from scipy.optimize import minimize

    points, anchors, directions = _portal_points(parameters, start, portals, goal)
    chords = points[1:] - points[:-1]
    lengths = np.hypot(chords[:, 0], chords[:, 1])
    y0 = np.zeros_like(chords)
    nonzero = lengths > 0.0
    y0[nonzero] = chords[nonzero] / lengths[nonzero, None]
    count = len(portals)

    def coefficients(y: Array) -> Array:
        return np.einsum("ij,ij->i", y[:-1] - y[1:], directions)

    initial_z = np.minimum(0.0, coefficients(y0))
    initial = np.concatenate((y0.ravel(), initial_z))
    linear_y = np.empty_like(y0)
    linear_y[0] = np.asarray(anchors[0] if count else goal) - np.asarray(start)
    for j in range(1, count):
        linear_y[j] = anchors[j] - anchors[j - 1]
    if count:
        linear_y[count] = np.asarray(goal) - anchors[-1]

    def unpack(variable: Array) -> tuple[Array, Array]:
        return variable[: 2 * (count + 1)].reshape(count + 1, 2), variable[2 * (count + 1):]

    def objective(variable: Array) -> tuple[float, Array]:
        y, z = unpack(variable)
        value = -(float(np.sum(linear_y * y)) + float(np.sum(z)))
        gradient = np.empty_like(variable)
        gradient[: 2 * (count + 1)] = -linear_y.ravel()
        gradient[2 * (count + 1):] = -1.0
        return value, gradient

    def constraints(variable: Array) -> Array:
        y, z = unpack(variable)
        c = coefficients(y)
        return np.concatenate((
            1.0 - np.einsum("ij,ij->i", y, y),
            -z,
            c - z,
        ))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": int(maximum_iterations), "ftol": 1.0e-13, "disp": False},
    )

    y, _z = unpack(np.asarray(result.x, dtype=float))
    norms = np.hypot(y[:, 0], y[:, 1])
    outside = norms > 1.0
    y[outside] /= norms[outside, None]
    c = coefficients(y)
    z = np.minimum(0.0, c)
    dual = float(np.sum(linear_y * y)) + float(np.sum(z))
    scale = 1.0 + abs(dual) + float(np.sum(np.abs(linear_y * y))) + float(np.sum(np.abs(z)))
    guard = 1024.0 * _EPS * scale
    return math.nextafter(dual - guard, -math.inf), int(getattr(result, "nit", 0)), bool(result.success)


@dataclass(frozen=True, slots=True)
class OrderedPortalDistanceResult:
    lower_bound: float
    upper_bound: float
    parameters: tuple[float, ...]
    iterations: int
    converged: bool
    solve_seconds: float
    dual_used: bool = False

    @property
    def gap(self) -> float:
        return max(0.0, self.upper_bound - self.lower_bound)


def ordered_portal_distance_bounds(
    start: Point2,
    portals: Sequence[Portal],
    goal: Point2,
    *,
    initial_parameters: Sequence[float] | None = None,
    maximum_iterations: int = 100,
    gradient_tolerance: float = 1.0e-11,
    dual_certificate: bool = True,
    dual_gap_trigger: float | None = None,
) -> OrderedPortalDistanceResult:
    """Bound the convex shortest-polyline-through-portals problem."""
    started = time.perf_counter()
    count = len(portals)
    if count == 0:
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        return OrderedPortalDistanceResult(distance, distance, (), 0, True, time.perf_counter() - started)

    if initial_parameters is None:
        x0 = np.full(count, 0.5, dtype=float)
    else:
        x0 = np.asarray(initial_parameters, dtype=float)
        if x0.shape != (count,) or not np.all(np.isfinite(x0)):
            raise ValueError("initial_parameters has wrong shape or nonfinite values")
        x0 = np.clip(x0, 0.0, 1.0)

    from scipy.optimize import minimize

    result = minimize(
        lambda x: _distance_objective_and_gradient(x, start, portals, goal),
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * count,
        options={
            "maxiter": int(maximum_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1.0e-15,
            "maxls": 40,
        },
    )
    parameters = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    upper, gradient = _distance_objective_and_gradient(parameters, start, portals, goal)
    minimizing_corner = np.where(gradient >= 0.0, 0.0, 1.0)
    affine_lower = upper + float(np.dot(gradient, minimizing_corner - parameters))

    iterations = int(getattr(result, "nit", 0))
    converged = bool(result.success)
    lower_candidates = [affine_lower]
    use_dual = dual_certificate and (
        dual_gap_trigger is None
        or upper - affine_lower > dual_gap_trigger * max(1.0, upper)
    )
    if use_dual:
        dual_lower, dual_iterations, dual_converged = _dual_distance_lower_bound(
            parameters,
            start,
            portals,
            goal,
            maximum_iterations=maximum_iterations,
        )
        lower_candidates.append(dual_lower)
        iterations += dual_iterations
        converged = converged and dual_converged

    scale = 1.0 + abs(upper) + float(np.sum(np.abs(gradient)))
    guard = 512.0 * _EPS * scale
    lower = math.nextafter(max(0.0, max(lower_candidates) - guard), -math.inf)
    lower = min(lower, upper)
    return OrderedPortalDistanceResult(
        lower,
        upper,
        tuple(map(float, parameters)),
        iterations,
        converged,
        time.perf_counter() - started,
        use_dual,
    )


# Twelve unoriented axes, evenly spaced over [0, pi).  These are module-level
# constants so a bound evaluation performs no trigonometric work or allocation.
_PROJECTION_DIRECTIONS_12: tuple[Point2, ...] = tuple(
    (math.cos(math.pi * index / 12.0), math.sin(math.pi * index / 12.0))
    for index in range(12)
)


def _outward_dot_interval(point: Point2, direction: Point2) -> tuple[float, float]:
    """Outward-rounded interval containing the exact binary64 dot product.

    The real quantity being relaxed is the dot product of the stored floating
    coordinates and the stored floating direction.  ``math.fsum`` substantially
    reduces cancellation, while the explicit first-order error enclosure makes
    the result independent of whether a platform's ``fsum`` happens to be
    correctly rounded in a particular cancellation case.
    """
    x, y = point
    nx, ny = direction
    first = nx * x
    second = ny * y
    value = math.fsum((first, second))
    error = 8.0 * _EPS * (abs(first) + abs(second) + abs(value) + 1.0)
    return (
        math.nextafter(value - error, -math.inf),
        math.nextafter(value + error, math.inf),
    )


def _project_portal_interval(
    portal: Portal,
    direction: Point2,
) -> tuple[float, float]:
    a_lo, a_hi = _outward_dot_interval(portal.a, direction)
    b_lo, b_hi = _outward_dot_interval(portal.b, direction)
    return min(a_lo, b_lo), max(a_hi, b_hi)


def _compress_monotone_extrema(values: list[float]) -> tuple[float, ...]:
    """Remove duplicate and same-direction interior values in-place."""
    result: list[float] = []
    for value in values:
        if result and value == result[-1]:
            continue
        while len(result) >= 2:
            previous = result[-1] - result[-2]
            following = value - result[-1]
            if previous * following < 0.0:
                break
            result.pop()
        result.append(value)
    return tuple(result)


def mandatory_projection_extrema(
    intervals: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    """Return the minimum-amplitude forced alternating-extrema skeleton.

    A scalar continuous path must visit the closed intervals in order.  The
    streaming intersection is the standard taut-string construction for this
    one-dimensional interval problem.  When a new interval lies wholly above
    the current feasible intersection, the old upper endpoint is unavoidable;
    the symmetric statement holds below.  The final active interval is closed
    by the nearest point to the last forced value.

    The returned values are a relaxation: intervals are already outward
    rounded, and same-direction excursions are compressed.
    """
    if not intervals:
        return ()
    lower, upper = map(float, intervals[0])
    if lower > upper or not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError('projection intervals must be finite and nonempty')
    forced: list[float] = []
    for interval in intervals[1:]:
        next_lower, next_upper = map(float, interval)
        if (
            next_lower > next_upper
            or not math.isfinite(next_lower)
            or not math.isfinite(next_upper)
        ):
            raise ValueError('projection intervals must be finite and nonempty')
        if next_lower > upper:
            forced.append(upper)
            lower, upper = next_lower, next_upper
        elif next_upper < lower:
            forced.append(lower)
            lower, upper = next_lower, next_upper
        else:
            lower = max(lower, next_lower)
            upper = min(upper, next_upper)

    if not forced:
        return ()
    last = forced[-1]
    terminal = min(upper, max(lower, last))
    if terminal != last:
        forced.append(terminal)
    return _compress_monotone_extrema(forced)


def _free_to_stop_time_lower_bound(distance: float) -> float:
    """Minimum scalar time from arbitrary bounded speed to rest."""
    distance = max(0.0, float(distance))
    threshold = V_MAX * V_MAX / (2.0 * MU_G)
    if distance <= threshold:
        value = math.sqrt(2.0 * distance / MU_G)
    else:
        value = distance / V_MAX + V_MAX / (2.0 * MU_G)
    guard = 128.0 * _EPS * (1.0 + value + distance)
    return max(0.0, math.nextafter(value - guard, -math.inf))


def _stop_to_stop_time_lower_bound(distance: float) -> float:
    """Minimum scalar time between two rest states."""
    distance = max(0.0, float(distance))
    threshold = V_MAX * V_MAX / MU_G
    if distance <= threshold:
        value = 2.0 * math.sqrt(distance / MU_G)
    else:
        value = distance / V_MAX + V_MAX / MU_G
    guard = 128.0 * _EPS * (1.0 + value + distance)
    return max(0.0, math.nextafter(value - guard, -math.inf))


def projected_reversal_time_lower_bound(
    intervals: Sequence[tuple[float, float]],
) -> float:
    """Certified scalar travel-time lower bound through ordered intervals.

    Consecutive values in the skeleton alternate direction.  Every internal
    direction reversal has an intervening zero projected velocity.  The first
    and last excursions therefore have one free endpoint velocity, while all
    interior excursions are rest-to-rest.
    """
    extrema = mandatory_projection_extrema(intervals)
    if len(extrema) < 2:
        return 0.0
    distances = tuple(abs(second - first) for first, second in zip(extrema[:-1], extrema[1:]))
    if len(distances) == 1:
        value = distances[0] / V_MAX
    else:
        terms = [_free_to_stop_time_lower_bound(distances[0])]
        terms.extend(_stop_to_stop_time_lower_bound(value) for value in distances[1:-1])
        terms.append(_free_to_stop_time_lower_bound(distances[-1]))
        value = math.fsum(terms)
    guard = 256.0 * _EPS * (1.0 + value + math.fsum(distances))
    return max(0.0, math.nextafter(value - guard, -math.inf))


def _projected_reversal_time_cached_axis(
    start_interval: tuple[float, float],
    portal_intervals: Sequence[tuple[tuple[float, float], ...]],
    goal_interval: tuple[float, float],
    direction_index: int,
) -> float:
    """Allocation-light cached-axis variant used by the production class."""
    lower, upper = start_interval
    forced: list[float] = []
    for values in portal_intervals:
        next_lower, next_upper = values[direction_index]
        if next_lower > upper:
            forced.append(upper)
            lower, upper = next_lower, next_upper
        elif next_upper < lower:
            forced.append(lower)
            lower, upper = next_lower, next_upper
        else:
            lower = max(lower, next_lower)
            upper = min(upper, next_upper)
    next_lower, next_upper = goal_interval
    if next_lower > upper:
        forced.append(upper)
        lower, upper = next_lower, next_upper
    elif next_upper < lower:
        forced.append(lower)
        lower, upper = next_lower, next_upper
    else:
        lower = max(lower, next_lower)
        upper = min(upper, next_upper)
    if not forced:
        return 0.0
    last = forced[-1]
    terminal = min(upper, max(lower, last))
    if terminal != last:
        forced.append(terminal)
    extrema = _compress_monotone_extrema(forced)
    if len(extrema) < 2:
        return 0.0
    distances = tuple(
        abs(second - first) for first, second in zip(extrema[:-1], extrema[1:])
    )
    if len(distances) == 1:
        value = distances[0] / V_MAX
    else:
        terms = [_free_to_stop_time_lower_bound(distances[0])]
        terms.extend(
            _stop_to_stop_time_lower_bound(value) for value in distances[1:-1]
        )
        terms.append(_free_to_stop_time_lower_bound(distances[-1]))
        value = math.fsum(terms)
    guard = 256.0 * _EPS * (1.0 + value + math.fsum(distances))
    return max(0.0, math.nextafter(value - guard, -math.inf))


def multidirectional_projection_time_lower_bound(
    start: Point2,
    portals: Sequence[Portal],
    goal: Point2,
    *,
    directions: Sequence[Point2] = _PROJECTION_DIRECTIONS_12,
) -> tuple[float, int]:
    """Maximum certified reversal bound over fixed projection axes.

    Returns ``(bound, winning_direction_index)``.  Start and goal are also
    represented by outward intervals, which can only weaken the relaxation.
    """
    if not directions:
        return 0.0, -1
    best = 0.0
    best_index = 0
    for index, direction in enumerate(directions):
        intervals = [_outward_dot_interval(start, direction)]
        intervals.extend(_project_portal_interval(portal, direction) for portal in portals)
        intervals.append(_outward_dot_interval(goal, direction))
        candidate = projected_reversal_time_lower_bound(intervals)
        if candidate > best:
            best = candidate
            best_index = index
    return best, best_index


def motor_only_time(distance: float, init_w: float) -> float:
    """Legacy exact optimistic one-sided motor time over a distance."""
    distance = float(distance)
    init_w = float(init_w)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and nonnegative")
    if not math.isfinite(init_w) or not 0.0 < init_w <= V_MAX * V_MAX:
        raise ValueError("init_w must lie in (0, V_MAX^2]")
    if distance == 0.0:
        return 0.0
    segment = compile_segment(distance, 0.0, init_w, 0.0, SegmentType.MOTOR, grad=True)
    value, _gradient = segment.time_and_jac(distance)
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError("motor-only time evaluator returned an invalid value")
    return float(value)


def _motor_acceleration_distance(v0: float, peak_v: float) -> float:
    if peak_v <= v0:
        return 0.0
    ratio = (V_MAX - v0) / (V_MAX - peak_v)
    return (V_MAX * math.log(ratio) - (peak_v - v0)) / B_EMF


def _motor_acceleration_time(v0: float, peak_v: float) -> float:
    if peak_v <= v0:
        return 0.0
    return math.log((V_MAX - v0) / (V_MAX - peak_v)) / B_EMF


def straight_two_sided_time_lower_bound(distance: float, init_w: float) -> float:
    """Certified straight accelerate/brake lower time with equal endpoint speed.

    The relaxed path has zero curvature, maximum motor acceleration, maximum
    braking, and no geometric restrictions.  A bisection maintains a peak
    speed whose accelerate+brake distance is no greater than ``distance``;
    any tiny remaining distance is credited at the unattainably favorable
    speed ``V_MAX``.  This one-sided construction makes the returned value a
    lower bound independent of root-solver convergence.
    """
    distance = float(distance)
    init_w = float(init_w)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and nonnegative")
    if not math.isfinite(init_w) or not 0.0 < init_w <= V_MAX * V_MAX:
        raise ValueError("init_w must lie in (0, V_MAX^2]")
    if distance == 0.0:
        return 0.0

    v0 = math.sqrt(init_w)
    if v0 >= V_MAX:
        value = distance / V_MAX
        return math.nextafter(value, -math.inf)

    def distance_to_peak(peak_v: float) -> float:
        return _motor_acceleration_distance(v0, peak_v) + (
            peak_v * peak_v - init_w
        ) / (2.0 * A_BRAKE)

    low = v0
    high = math.nextafter(V_MAX, 0.0)
    if distance_to_peak(high) <= distance:
        return math.nextafter(distance / V_MAX, -math.inf)

    # 56 iterations already over-resolve binary64 on this bounded interval;
    # 64 keeps the proof simple and costs far less than one portal solve.
    for _ in range(64):
        mid = 0.5 * (low + high)
        if mid == low or mid == high:
            break
        if distance_to_peak(mid) <= distance:
            low = mid
        else:
            high = mid

    used_distance = distance_to_peak(low)
    while used_distance > distance and low > v0:
        next_low = math.nextafter(low, v0)
        if next_low == low:
            break
        low = next_low
        used_distance = distance_to_peak(low)
    time_to_peak = _motor_acceleration_time(v0, low) + (low - v0) / A_BRAKE
    value = time_to_peak + max(0.0, distance - used_distance) / V_MAX
    scale = 1.0 + abs(value) + abs(time_to_peak) + distance / V_MAX
    guard = 256.0 * _EPS * scale
    return max(0.0, math.nextafter(value - guard, -math.inf))


def _world_walls(
    maze: Maze,
    *,
    cell_size: float,
    origin: Point2,
) -> tuple[WorldWall, ...]:
    """Return unit-grid wall segments in world coordinates.

    ``Maze.gen_walls`` stores each outer boundary as one long segment.  Splitting
    it at grid vertices makes the common small-radius portal query local: when
    ``r < cell_size/2``, only walls incident to a portal endpoint can matter.
    """
    ox, oy = origin
    h = cell_size
    result: list[WorldWall] = []
    for a, b in maze.walls:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = abs(dx) + abs(dy)
        if length <= 0:
            continue
        sx = 0 if dx == 0 else (1 if dx > 0 else -1)
        sy = 0 if dy == 0 else (1 if dy > 0 else -1)
        first = a
        for step in range(1, length + 1):
            second = (a[0] + sx * step, a[1] + sy * step)
            result.append((
                (math.fma(float(first[0]), h, ox), math.fma(float(first[1]), h, oy)),
                (math.fma(float(second[0]), h, ox), math.fma(float(second[1]), h, oy)),
            ))
            first = second
    return tuple(result)


def _blocked_interval(
    portal: Portal,
    wall: WorldWall,
    radius: float,
) -> tuple[float, float] | None:
    """Portal parameters whose points are within ``radius`` of an axis wall."""
    (ax, ay), (bx, by) = portal.a, portal.b
    (wx0, wy0), (wx1, wy1) = wall
    dx, dy = bx - ax, by - ay
    if dx == 0.0:
        if wx0 == wx1:
            perp = abs(ax - wx0)
            if perp > radius:
                return None
            reach = math.sqrt(max(0.0, radius * radius - perp * perp))
            lo, hi = min(wy0, wy1) - reach, max(wy0, wy1) + reach
        else:
            xlo, xhi = min(wx0, wx1), max(wx0, wx1)
            perp = 0.0 if xlo <= ax <= xhi else min(abs(ax - xlo), abs(ax - xhi))
            if perp > radius:
                return None
            reach = math.sqrt(max(0.0, radius * radius - perp * perp))
            lo, hi = wy0 - reach, wy0 + reach
        lo, hi = max(min(ay, by), lo), min(max(ay, by), hi)
        if lo > hi:
            return None
        u, v = (lo - ay) / dy, (hi - ay) / dy
    else:
        if wy0 == wy1:
            perp = abs(ay - wy0)
            if perp > radius:
                return None
            reach = math.sqrt(max(0.0, radius * radius - perp * perp))
            lo, hi = min(wx0, wx1) - reach, max(wx0, wx1) + reach
        else:
            ylo, yhi = min(wy0, wy1), max(wy0, wy1)
            perp = 0.0 if ylo <= ay <= yhi else min(abs(ay - ylo), abs(ay - yhi))
            if perp > radius:
                return None
            reach = math.sqrt(max(0.0, radius * radius - perp * perp))
            lo, hi = wx0 - reach, wx0 + reach
        lo, hi = max(min(ax, bx), lo), min(max(ax, bx), hi)
        if lo > hi:
            return None
        u, v = (lo - ax) / dx, (hi - ax) / dx
    return max(0.0, min(u, v)), min(1.0, max(u, v))


def clearance_gate(portal: Portal, walls: Sequence[WorldWall], radius: float) -> Portal:
    """Return an outward-relaxed inscribed-disk-safe subset of ``portal``.

    If interval classification is empty or numerically ambiguous, returning the
    full portal weakens the relaxation but preserves admissibility.
    """
    scale = 1.0 + max(abs(value) for value in (*portal.a, *portal.b))
    coordinate_guard = 8192.0 * _EPS * scale + 1.0e-14
    effective_radius = max(0.0, float(radius) - coordinate_guard)
    blocked = sorted(
        interval
        for wall in walls
        if (interval := _blocked_interval(portal, wall, effective_radius)) is not None
    )
    merged: list[list[float]] = []
    for lo, hi in blocked:
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return portal
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)

    feasible: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in merged:
        if lo > cursor:
            feasible.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < 1.0:
        feasible.append((cursor, 1.0))
    if not feasible:
        return portal

    # The convex hull of all feasible components is a superset and therefore
    # remains a valid relaxation when unusual large bodies create disjoint
    # intervals along one portal.
    parameter_guard = 8192.0 * _EPS + 1.0e-13
    lo = max(0.0, feasible[0][0] - parameter_guard)
    hi = min(1.0, feasible[-1][1] + parameter_guard)
    if not lo < hi:
        return portal
    a = portal.point(lo)
    b = portal.point(hi)
    if a == b:
        return portal
    return Portal(a, b)


def _directions(cells: Sequence[Cell]) -> tuple[Cell, ...]:
    result: list[Cell] = []
    for first, second in zip(cells[:-1], cells[1:]):
        direction = second[0] - first[0], second[1] - first[1]
        if abs(direction[0]) + abs(direction[1]) != 1:
            raise ValueError(f"non-adjacent route transition: {first}->{second}")
        result.append(direction)
    return tuple(result)


def _eroded_overlapping_cover(
    request: TimeBoundRequest,
    radius: float,
) -> tuple[AxisAlignedBox, ...] | None:
    """Build the disk-eroded run/turn cover used by the production optimizer."""
    cells = request.cell_path
    if len(cells) == 1:
        x, y = cells[0]
        h = request.cell_size
        ox, oy = request.origin
        guard = 8192.0 * _EPS * (1.0 + abs(ox) + abs(oy) + h)
        r = max(0.0, radius - guard)
        box = AxisAlignedBox(ox + x * h + r, ox + (x + 1) * h - r,
                             oy + y * h + r, oy + (y + 1) * h - r)
        return (box,)

    directions = _directions(cells)
    run_groups: list[list[Cell]] = []
    for edge, direction in enumerate(directions):
        if edge == 0 or direction != directions[edge - 1]:
            run_groups.append([cells[edge], cells[edge + 1]])
        else:
            run_groups[-1].append(cells[edge + 1])

    h = request.cell_size
    ox, oy = request.origin
    scale = 1.0 + abs(ox) + abs(oy) + h * (1.0 + max(max(x, y) for x, y in cells))
    r = max(0.0, radius - (8192.0 * _EPS * scale + 1.0e-14))
    if 2.0 * r >= h:
        # A zero-width eroded cell is still a valid necessary set, but tiny
        # negative widths caused by guards should not manufacture infeasibility.
        r = math.nextafter(0.5 * h, 0.0)

    boxes: list[AxisAlignedBox] = []
    for run_index, group in enumerate(run_groups):
        xs = [cell[0] for cell in group]
        ys = [cell[1] for cell in group]
        boxes.append(AxisAlignedBox(
            ox + min(xs) * h + r,
            ox + (max(xs) + 1) * h - r,
            oy + min(ys) * h + r,
            oy + (max(ys) + 1) * h - r,
        ))
        if run_index + 1 < len(run_groups):
            # Adjacent runs share exactly their turn cell.
            turn = run_groups[run_index][-1]
            x, y = turn
            boxes.append(AxisAlignedBox(
                ox + x * h + r,
                ox + (x + 1) * h - r,
                oy + y * h + r,
                oy + (y + 1) * h - r,
            ))
    return tuple(boxes)


@dataclass(frozen=True, slots=True)
class OrderedBoxDistanceResult:
    lower_bound: float
    upper_bound: float
    points: tuple[Point2, ...]
    iterations: int
    converged: bool
    solve_seconds: float

    @property
    def gap(self) -> float:
        return max(0.0, self.upper_bound - self.lower_bound)


def _box_distance_objective_and_gradient(
    variables: Array,
    start: Point2,
    goal: Point2,
) -> tuple[float, Array]:
    count = variables.size // 2
    points = np.empty((count + 2, 2), dtype=float)
    points[0] = start
    points[-1] = goal
    points[1:-1] = variables.reshape(count, 2)
    chords = points[1:] - points[:-1]
    lengths = np.hypot(chords[:, 0], chords[:, 1])
    units = np.zeros_like(chords)
    nonzero = lengths > 0.0
    units[nonzero] = chords[nonzero] / lengths[nonzero, None]
    gradient = (units[:-1] - units[1:]).ravel()
    return float(math.fsum(map(float, lengths))), gradient


def ordered_box_distance_bounds(
    start: Point2,
    transition_boxes: Sequence[AxisAlignedBox],
    goal: Point2,
    *,
    initial_points: Sequence[Point2] | None = None,
    maximum_iterations: int = 100,
    gradient_tolerance: float = 1.0e-11,
) -> OrderedBoxDistanceResult:
    """Certified shortest polyline through ordered axis-aligned boxes."""
    started = time.perf_counter()
    count = len(transition_boxes)
    if count == 0:
        distance = math.dist(start, goal)
        return OrderedBoxDistanceResult(distance, distance, (), 0, True, time.perf_counter() - started)

    bounds: list[tuple[float, float]] = []
    default = np.empty(2 * count, dtype=float)
    for i, box in enumerate(transition_boxes):
        bounds.extend(((box.xmin, box.xmax), (box.ymin, box.ymax)))
        default[2 * i] = 0.5 * (box.xmin + box.xmax)
        default[2 * i + 1] = 0.5 * (box.ymin + box.ymax)
    if initial_points is None:
        x0 = default
    else:
        array = np.asarray(initial_points, dtype=float)
        if array.shape != (count, 2) or not np.all(np.isfinite(array)):
            x0 = default
        else:
            x0 = array.ravel().copy()
            for i, (lo, hi) in enumerate(bounds):
                x0[i] = min(hi, max(lo, x0[i]))

    from scipy.optimize import minimize

    result = minimize(
        lambda x: _box_distance_objective_and_gradient(x, start, goal),
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": int(maximum_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1.0e-15,
            "maxls": 40,
        },
    )
    variables = np.asarray(result.x, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        variables[i] = min(hi, max(lo, variables[i]))
    upper, gradient = _box_distance_objective_and_gradient(variables, start, goal)
    affine_lower = upper
    for value, derivative, (lo, hi) in zip(variables, gradient, bounds):
        affine_lower += derivative * ((lo if derivative >= 0.0 else hi) - value)
    scale = 1.0 + abs(upper) + float(np.sum(np.abs(gradient)))
    guard = 512.0 * _EPS * scale
    lower = min(upper, math.nextafter(max(0.0, affine_lower - guard), -math.inf))
    points = tuple((float(x), float(y)) for x, y in variables.reshape(count, 2))
    return OrderedBoxDistanceResult(
        lower,
        upper,
        points,
        int(getattr(result, "nit", 0)),
        bool(result.success),
        time.perf_counter() - started,
    )


def eroded_cover_distance_bounds(
    request: TimeBoundRequest,
    radius: float,
    *,
    initial_points: Sequence[Point2] | None = None,
    maximum_iterations: int = 100,
    gradient_tolerance: float = 1.0e-11,
) -> OrderedBoxDistanceResult:
    cover = _eroded_overlapping_cover(request, radius)
    if cover is None or not cover:
        return OrderedBoxDistanceResult(0.0, math.inf, (), 0, False, 0.0)
    if not cover[0].contains(request.start) or not cover[-1].contains(request.goal):
        return OrderedBoxDistanceResult(math.inf, math.inf, (), 0, True, 0.0)
    overlaps: list[AxisAlignedBox] = []
    for first, second in zip(cover[:-1], cover[1:]):
        overlap = first.intersection(second)
        if overlap is None:
            return OrderedBoxDistanceResult(math.inf, math.inf, (), 0, True, 0.0)
        overlaps.append(overlap)
    return ordered_box_distance_bounds(
        request.start,
        overlaps,
        request.goal,
        initial_points=initial_points,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
    )


@dataclass(slots=True)
class PortalMotorTimeLowerBound:
    """Production hierarchical lower bound for path-indexed search.

    Without ``maze`` this class retains the legacy full-portal behavior.  With
    a maze it contracts each portal using the vehicle's centered inscribed disk
    and uses equal start/end speed in the straight dynamic relaxation.  The
    production default also takes the maximum with a cached 12-axis scalar
    reversal bound.
    """

    maximum_iterations: int = 100
    gradient_tolerance: float = 1.0e-11
    cache_warm_starts: bool = True
    maze: Maze | None = None
    body_length: float = 0.0
    body_height: float = 0.0
    use_inscribed_disk_gates: bool = True
    use_two_sided_time: bool = True
    use_dual_certificate: bool = False
    dual_gap_trigger: float = 1.0e-6
    use_complete_cover_bound: bool = True
    complete_refinement_gap: float = 2.0
    projection_directions: int = 12
    name: str = "inscribed_disk_portal_two_sided"
    statistics: TimeBoundStatistics = field(default_factory=TimeBoundStatistics, init=False)
    _warm_starts: dict[tuple[object, ...], tuple[float, ...]] = field(default_factory=dict, init=False, repr=False)
    _results: dict[tuple[object, ...], TimeBoundResult] = field(default_factory=dict, init=False, repr=False)
    _gate_cache: dict[tuple[object, ...], Portal] = field(default_factory=dict, init=False, repr=False)
    _wall_cache: dict[tuple[float, Point2], tuple[WorldWall, ...]] = field(default_factory=dict, init=False, repr=False)
    _wall_vertex_cache: dict[tuple[float, Point2], dict[Point2, tuple[WorldWall, ...]]] = field(default_factory=dict, init=False, repr=False)
    _complete_warm: dict[tuple[object, ...], tuple[Point2, ...]] = field(default_factory=dict, init=False, repr=False)
    _complete_results: dict[tuple[object, ...], TimeBoundResult] = field(default_factory=dict, init=False, repr=False)
    _projection_gate_cache: dict[Portal, tuple[tuple[float, float], ...]] = field(default_factory=dict, init=False, repr=False)
    _projection_point_cache: dict[Point2, tuple[tuple[float, float], ...]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if not math.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0.0:
            raise ValueError("gradient_tolerance must be finite and positive")
        if self.maze is not None:
            if not math.isfinite(self.body_length) or self.body_length <= 0.0:
                raise ValueError("body_length must be finite and positive when maze is set")
            if not math.isfinite(self.body_height) or self.body_height <= 0.0:
                raise ValueError("body_height must be finite and positive when maze is set")
        if not math.isfinite(self.complete_refinement_gap) or self.complete_refinement_gap < 0.0:
            raise ValueError("complete_refinement_gap must be finite and nonnegative")
        if not math.isfinite(self.dual_gap_trigger) or self.dual_gap_trigger < 0.0:
            raise ValueError("dual_gap_trigger must be finite and nonnegative")
        if self.projection_directions not in {0, 12}:
            raise ValueError("projection_directions must be 0 or 12")
        if self.maze is None or not self.use_inscribed_disk_gates:
            self.name = "ordered_portal_motor"
        elif not self.use_two_sided_time:
            self.name = "inscribed_disk_portal_motor"
        if self.projection_directions:
            self.name += "_projection12"

    @property
    def inscribed_radius(self) -> float:
        if self.maze is None:
            return 0.0
        return 0.5 * min(self.body_length, self.body_height)

    @staticmethod
    def _geometry_key(request: TimeBoundRequest) -> tuple[object, ...]:
        return (
            request.cell_path,
            request.start,
            request.goal,
            request.cell_size,
            request.origin,
            request.portal_mask,
            request.goal_radius,
        )

    def _walls(self, request: TimeBoundRequest) -> tuple[WorldWall, ...]:
        if self.maze is None:
            return ()
        key = (request.cell_size, request.origin)
        walls = self._wall_cache.get(key)
        if walls is None:
            walls = _world_walls(self.maze, cell_size=request.cell_size, origin=request.origin)
            self._wall_cache[key] = walls
        return walls

    def _portal_walls(
        self,
        request: TimeBoundRequest,
        portal: Portal,
    ) -> Sequence[WorldWall]:
        walls = self._walls(request)
        if self.inscribed_radius >= 0.5 * request.cell_size:
            return walls
        key = (request.cell_size, request.origin)
        by_vertex = self._wall_vertex_cache.get(key)
        if by_vertex is None:
            temporary: dict[Point2, list[WorldWall]] = {}
            for wall in walls:
                temporary.setdefault(wall[0], []).append(wall)
                temporary.setdefault(wall[1], []).append(wall)
            by_vertex = {vertex: tuple(items) for vertex, items in temporary.items()}
            self._wall_vertex_cache[key] = by_vertex
        local = [*by_vertex.get(portal.a, ()), *by_vertex.get(portal.b, ())]
        if not local:
            return ()
        # A wall can be incident to both portal endpoints only if it is the
        # portal itself; open portals are absent from the maze wall set, but
        # deduplication keeps this helper robust to external Maze objects.
        return tuple(dict.fromkeys(local))

    def _gate(self, first: Cell, second: Cell, request: TimeBoundRequest) -> Portal:
        if self.maze is None or not self.use_inscribed_disk_gates:
            return portal_between_cells(
                first, second, cell_size=request.cell_size, origin=request.origin
            )
        key = (first, second, request.cell_size, request.origin, self.inscribed_radius)
        gate = self._gate_cache.get(key)
        if gate is not None:
            self.statistics.gate_cache_hits += 1
            return gate
        self.statistics.gate_cache_misses += 1
        portal = portal_between_cells(
            first, second, cell_size=request.cell_size, origin=request.origin
        )
        gate = clearance_gate(portal, self._portal_walls(request, portal), self.inscribed_radius)
        self._gate_cache[key] = gate
        # Reverse traversal is common during graph search and is exactly the
        # same geometric interval with opposite orientation.
        reverse_key = (second, first, request.cell_size, request.origin, self.inscribed_radius)
        self._gate_cache[reverse_key] = Portal(gate.b, gate.a)
        return gate

    def _time_from_distance(self, distance: float, init_w: float, terminal_w_max: float | None = None) -> float:
        if self.use_two_sided_time and (terminal_w_max is None or terminal_w_max <= init_w):
            # Requiring the relaxed path to finish at init_w is exact when the
            # cap equals init_w and optimistic (therefore still admissible)
            # when the actual terminal cap is lower.  For a higher cap, fall
            # back to the one-sided motor relaxation to preserve admissibility.
            return straight_two_sided_time_lower_bound(distance, init_w)
        return motor_only_time(distance, init_w)

    def _projection_point_intervals(
        self, point: Point2
    ) -> tuple[tuple[float, float], ...]:
        cached = self._projection_point_cache.get(point)
        if cached is None:
            cached = tuple(
                _outward_dot_interval(point, direction)
                for direction in _PROJECTION_DIRECTIONS_12
            )
            self._projection_point_cache[point] = cached
        return cached

    def _projection_portal_intervals(
        self, portal: Portal
    ) -> tuple[tuple[float, float], ...]:
        cached = self._projection_gate_cache.get(portal)
        if cached is not None:
            self.statistics.projection_gate_cache_hits += 1
            return cached
        self.statistics.projection_gate_cache_misses += 1
        cached = tuple(
            _project_portal_interval(portal, direction)
            for direction in _PROJECTION_DIRECTIONS_12
        )
        self._projection_gate_cache[portal] = cached
        reverse = Portal(portal.b, portal.a)
        self._projection_gate_cache.setdefault(reverse, cached)
        return cached

    def _projection_time(
        self,
        start: Point2,
        portals: Sequence[Portal],
        goal: Point2,
        goal_radius: float = 0.0,
    ) -> float:
        start_intervals = self._projection_point_intervals(start)
        goal_intervals = self._projection_point_intervals(goal)
        if goal_radius > 0.0:
            # The true terminal set is the goal-cell rectangle.  Relax it to
            # a containing disk centered on ``goal``.  Expanding each scalar
            # projection by that radius is optimistic, hence admissible.
            goal_intervals = tuple(
                (lo - goal_radius, hi + goal_radius)
                for lo, hi in goal_intervals
            )
        portal_intervals = tuple(
            self._projection_portal_intervals(portal) for portal in portals
        )
        best = 0.0
        for direction_index in range(12):
            best = max(
                best,
                _projected_reversal_time_cached_axis(
                    start_intervals[direction_index],
                    portal_intervals,
                    goal_intervals[direction_index],
                    direction_index,
                ),
            )
        return best

    def evaluate(self, request: TimeBoundRequest) -> TimeBoundResult:
        self.statistics.evaluations += 1
        key = (
            *self._geometry_key(request),
            request.init_w,
            float(request.terminal_w_max),
        )
        cached = self._results.get(key)
        if cached is not None:
            self.statistics.result_cache_hits += 1
            return cached

        if request.portal_mask is None:
            transitions = zip(request.cell_path[:-1], request.cell_path[1:])
        else:
            transitions = (
                (first, second)
                for first, second, keep in zip(
                    request.cell_path[:-1],
                    request.cell_path[1:],
                    request.portal_mask,
                    strict=True,
                )
                if keep
            )
        portals = tuple(
            self._gate(first, second, request)
            for first, second in transitions
        )
        geometry_key = self._geometry_key(request)
        warm = self._warm_starts.get(geometry_key)
        if warm is None and self.cache_warm_starts and len(request.cell_path) > 1:
            parent_mask = (
                None
                if request.portal_mask is None
                else request.portal_mask[:-1]
            )
            parent_key = (
                request.cell_path[:-1],
                request.start,
                request.goal,
                request.cell_size,
                request.origin,
                parent_mask,
                request.goal_radius,
            )
            parent_warm = self._warm_starts.get(parent_key)
            if parent_warm is not None:
                appended = 1 if request.portal_mask is None or request.portal_mask[-1] else 0
                warm = (*parent_warm, *((0.5,) * appended))

        distance = ordered_portal_distance_bounds(
            request.start,
            portals,
            request.goal,
            initial_parameters=warm,
            maximum_iterations=self.maximum_iterations,
            gradient_tolerance=self.gradient_tolerance,
            dual_certificate=self.use_dual_certificate,
            dual_gap_trigger=self.dual_gap_trigger,
        )
        self.statistics.portal_seconds += distance.solve_seconds
        if distance.dual_used:
            self.statistics.dual_certificates += 1
        if self.cache_warm_starts:
            self._warm_starts[geometry_key] = distance.parameters

        # ``ordered_portal_distance_bounds`` terminates at the goal-cell
        # center.  The optimizer may stop anywhere inside the goal cell, so
        # subtract the radius of a disk containing that entire cell.  By the
        # triangle inequality this can only weaken the bound.
        distance_lower = max(0.0, distance.lower_bound - request.goal_radius)
        if len(request.cell_path) > 1:
            parent_mask = (
                None
                if request.portal_mask is None
                else request.portal_mask[:-1]
            )
            parent_key = (
                request.cell_path[:-1],
                request.start,
                request.goal,
                request.cell_size,
                request.origin,
                parent_mask,
                request.goal_radius,
                request.init_w,
                float(request.terminal_w_max),
            )
            parent_result = self._results.get(parent_key)
            if parent_result is not None:
                distance_lower = max(distance_lower, parent_result.distance_lower_bound)

        distance_time = self._time_from_distance(distance_lower, request.init_w, request.terminal_w_max)
        projection_time = 0.0
        if self.projection_directions:
            projection_started = time.perf_counter()
            projection_time = self._projection_time(
                request.start, portals, request.goal, request.goal_radius
            )
            self.statistics.projection_evaluations += 1
            self.statistics.projection_seconds += time.perf_counter() - projection_started
            gain = projection_time - distance_time
            if gain > 0.0:
                self.statistics.projection_strengthened += 1
                self.statistics.projection_maximum_gain = max(
                    self.statistics.projection_maximum_gain, gain
                )

        result = TimeBoundResult(
            self.name,
            max(distance_time, projection_time),
            distance_lower,
            distance.upper_bound,
            max(0.0, distance.upper_bound - distance_lower),
            distance.solve_seconds,
            distance.iterations,
            distance.converged,
            distance.parameters,
        )
        self._results[key] = result
        return result

    def refine_complete(
        self,
        request: TimeBoundRequest,
        incumbent: float,
        base_result: TimeBoundResult | None = None,
    ) -> TimeBoundResult:
        if base_result is None:
            base_result = self.evaluate(request)
        if self.maze is None or not self.use_complete_cover_bound:
            return base_result
        if request.portal_mask is not None and not all(request.portal_mask):
            # The ordinary cover is member-specific.  Reusing it after room
            # quotienting could overbound the larger union class, so masked
            # requests retain the already certified portal/projection result.
            return base_result
        gap = float(incumbent) - base_result.time_lower_bound
        if gap <= 0.0 or gap > self.complete_refinement_gap:
            self.statistics.complete_refinement_skipped_by_gap += 1
            return base_result

        key = (
            *self._geometry_key(request),
            request.init_w,
            float(request.terminal_w_max),
            self.inscribed_radius,
        )
        cached = self._complete_results.get(key)
        if cached is not None:
            self.statistics.complete_refinement_cache_hits += 1
            return cached if cached.time_lower_bound >= base_result.time_lower_bound else base_result

        started = time.perf_counter()
        warm = self._complete_warm.get(self._geometry_key(request))
        distance = eroded_cover_distance_bounds(
            request,
            self.inscribed_radius,
            initial_points=warm,
            maximum_iterations=self.maximum_iterations,
            gradient_tolerance=self.gradient_tolerance,
        )
        elapsed = time.perf_counter() - started
        self.statistics.complete_refinements += 1
        self.statistics.complete_refinement_seconds += elapsed
        if distance.points:
            self._complete_warm[self._geometry_key(request)] = distance.points

        relaxed_complete_lower = max(
            0.0, distance.lower_bound - request.goal_radius
        )
        distance_lower = max(
            base_result.distance_lower_bound, relaxed_complete_lower
        )
        distance_upper = min(base_result.relaxed_distance_upper_bound, distance.upper_bound)
        if math.isinf(distance_lower):
            distance_upper = math.inf
            time_lower = math.inf
        else:
            if not math.isfinite(distance_upper):
                distance_upper = base_result.relaxed_distance_upper_bound
            distance_upper = max(distance_lower, distance_upper)
            time_lower = max(
                base_result.time_lower_bound,
                self._time_from_distance(distance_lower, request.init_w, request.terminal_w_max),
            )
        result = TimeBoundResult(
            "eroded_cover_two_sided_projection12" if self.projection_directions else "eroded_cover_two_sided",
            time_lower,
            distance_lower,
            distance_upper,
            max(0.0, distance_upper - distance_lower),
            base_result.solve_seconds + elapsed,
            base_result.iterations + distance.iterations,
            base_result.converged and distance.converged,
            base_result.portal_parameters,
        )
        self._complete_results[key] = result
        return result


__all__ = [
    "AxisAlignedBox",
    "Cell",
    "CompleteTimeLowerBound",
    "OrderedBoxDistanceResult",
    "OrderedPortalDistanceResult",
    "Point2",
    "Portal",
    "PortalMotorTimeLowerBound",
    "TimeBoundRequest",
    "TimeBoundResult",
    "TimeBoundStatistics",
    "TimeLowerBound",
    "clearance_gate",
    "eroded_cover_distance_bounds",
    "mandatory_projection_extrema",
    "motor_only_time",
    "multidirectional_projection_time_lower_bound",
    "ordered_box_distance_bounds",
    "ordered_portal_distance_bounds",
    "portal_between_cells",
    "projected_reversal_time_lower_bound",
    "portals_from_cell_path",
    "straight_two_sided_time_lower_bound",
]
