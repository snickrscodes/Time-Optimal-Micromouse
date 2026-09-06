"""Production reverse-mode speed-profile solver for the segment API.

The scalar topology pass retains each full candidate segment and records the
actual emitted length separately, so an event-truncated record may satisfy
``record.seg.L > record.L_used``.  Internal friction-cap extremals are inserted
in exact batches only while the current candidate family leaves an uncovered
interval or cannot form a continuous profile.  Switching and friction-domain
detection share one scan, differentiable replay segments support arbitrary-
prefix state and time queries, and the reverse sweep contains only local
algebra.

The optional :class:`PhaseProfiler` measures segment construction, topology scanning,
envelope work, prefix queries, and reverse-sweep time.  All physical constants
are imported from ``segment.constants``; every other physical quantity in this
module is derived from those five constants.

Python 3.13+.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .crossings import (
    ScanResult,
    brake_grip_scan,
    bracketed_newton,
    grip_brake_scan,
    grip_brake_earliest_scan,
    grip_motor_scan,
    grip_motor_earliest_scan,
    motor_grip_scan,
)
from segment import SegmentType, compile_segment
from segment.constants import A_BRAKE, A_MAX, B_EMF, MU_G, V_MAX

# Derived physical constants.  The five imported constants above are the only
# independent physical constants used by this module.
MU_G2 = MU_G * MU_G
S_BRAKE = 2.0 * A_BRAKE
W_EQ = V_MAX * V_MAX

FRICTION_DOMAIN_MARGIN = 1e-12
# Internal cap anchors retain a small G^2 margin for robust switching/event
# differentiation. Exact repelling boundary starts are handled directly by Cflow.
INTERNAL_CAP_G2_MARGIN = 1.0e-8
FRICTION_DOMAIN_SCAN = 256
EVENT_RESIDUAL_TOL = 1e-8
EVENT_DERIV_TOL = 1e-10
# A directional switching root can be mathematically positive yet too short
# to change either state coordinate in binary64.  Such a root must be treated
# as an unresolved start boundary; otherwise the two mode scanners can bounce
# while repeatedly re-emitting an identical state.  The cap only limits this
# classifier to numerically microscopic intervals.
EVENT_START_MAX_SPATIAL_TOL = 1.0e-9

TEMP_SCALAR_COMPILE = "temporary_scalar_segment_constructions"
EMITTED_SCALAR_COMPILE = "emitted_scalar_segment_constructions"
DIFF_REPLAY_COMPILE = "differentiable_replay_constructions"
CROSSING_SCAN_EVAL = "crossing_scan_evaluations"
DOMAIN_SCAN_EVAL = "domain_scan_evaluations"
ENVELOPE_OVERLAP_EVAL = "envelope_overlap_evaluations"
INTERIOR_PREFIX_TIME_EVAL = "interior_prefix_time_evaluations"
REVERSE_SWEEP = "reverse_sweep_time"
ANCHOR_WITNESS_SCAN = "anchor_witness_scans"

NATIVE_SCALAR_BUILD = "native_scalar_profile_build"
NATIVE_REVERSE_PROMOTION = "native_reverse_promotion"
NATIVE_REVERSE_SWEEP = "native_reverse_sweep"

_REVERSE_BACKEND = os.environ.get("AME_REVERSE_BACKEND", "native").strip().lower() or "native"
if _REVERSE_BACKEND not in {"python", "native"}:
    raise RuntimeError(
        "AME_REVERSE_BACKEND must be 'native' or 'python'"
    )

def reverse_backend() -> str:
    """Return the active reverse-solver backend (``python`` or ``native``)."""
    return _REVERSE_BACKEND

def set_reverse_backend(name: str) -> str:
    """Select the reverse-solver backend and return the previous backend.

    The selection is process-local. Native is the packaged production default;
    the Python implementation remains available as an explicit reference backend
    for A/B qualification and low-level oracle tests.
    """
    global _REVERSE_BACKEND
    value = str(name).strip().lower()
    if value not in {"python", "native"}:
        raise ValueError("reverse backend must be 'python' or 'native'")
    old = _REVERSE_BACKEND
    _REVERSE_BACKEND = value
    return old

@contextmanager
def using_reverse_backend(name: str):
    old = set_reverse_backend(name)
    try:
        yield
    finally:
        set_reverse_backend(old)

_NATIVE_REVERSE_MODULE: Any | None = None

def _native_reverse_module():
    global _NATIVE_REVERSE_MODULE
    if _NATIVE_REVERSE_MODULE is not None:
        return _NATIVE_REVERSE_MODULE
    try:
        import creverse
    except Exception as exc:
        raise RuntimeError(
            "native reverse backend requested but unavailable; run `make native`"
        ) from exc
    _NATIVE_REVERSE_MODULE = creverse
    return creverse

PROFILE_PHASES = (
    TEMP_SCALAR_COMPILE,
    EMITTED_SCALAR_COMPILE,
    DIFF_REPLAY_COMPILE,
    CROSSING_SCAN_EVAL,
    DOMAIN_SCAN_EVAL,
    ENVELOPE_OVERLAP_EVAL,
    INTERIOR_PREFIX_TIME_EVAL,
    REVERSE_SWEEP,
    ANCHOR_WITNESS_SCAN,
    NATIVE_SCALAR_BUILD,
    NATIVE_REVERSE_PROMOTION,
    NATIVE_REVERSE_SWEEP,
)


@dataclass(slots=True)
class PhaseMetric:
    count: int = 0
    seconds: float = 0.0


@dataclass(slots=True)
class PhaseProfiler:
    """Low-overhead, opt-in phase profiler for topology and reverse work."""

    enabled: bool = True
    metrics: dict[str, PhaseMetric] = field(
        default_factory=lambda: {name: PhaseMetric() for name in PROFILE_PHASES}
    )

    def record(self, phase: str, elapsed: float, *, count: int = 1) -> None:
        if not self.enabled:
            return
        metric = self.metrics.get(phase)
        if metric is None:
            metric = self.metrics[phase] = PhaseMetric()
        metric.count += count
        metric.seconds += elapsed

    def reset(self) -> None:
        self.metrics = {name: PhaseMetric() for name in PROFILE_PHASES}

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": metric.count,
                "seconds": metric.seconds,
                "microseconds_per_call": (
                    1e6 * metric.seconds / metric.count if metric.count else 0.0
                ),
            }
            for name, metric in self.metrics.items()
        }

    def format_report(self) -> str:
        lines = [
            "phase | count | total ms | us/call",
            "--- | ---: | ---: | ---:",
        ]
        for name in PROFILE_PHASES:
            metric = self.metrics.get(name, PhaseMetric())
            per_call = 1e6 * metric.seconds / metric.count if metric.count else 0.0
            lines.append(
                f"{name} | {metric.count} | {1e3 * metric.seconds:.6f} | {per_call:.3f}"
            )
        return "\n".join(lines)


class PassKind(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class EventKind(Enum):
    PIECE_END = auto()
    GRIP_MOTOR = auto()
    MOTOR_GRIP = auto()
    GRIP_BRAKE = auto()
    BRAKE_GRIP = auto()


def zeros(n: int) -> List[float]:
    return [0.0] * n


def validate_raw_parameters(raw_params: Sequence[float]) -> list[float]:
    """Validate and copy flat ``[L0, sigma0, L1, sigma1, ...]`` parameters."""
    raw = [float(value) for value in raw_params]
    if len(raw) % 2:
        raise ValueError("raw parameter array must have even length: [L0, sigma0, ...]")
    for i in range(0, len(raw), 2):
        L = raw[i]
        sigma = raw[i + 1]
        if not math.isfinite(L) or L <= 0.0:
            raise ValueError(f"segment {i // 2} length must be finite and positive, got {L!r}")
        if not math.isfinite(sigma):
            raise ValueError(f"segment {i // 2} sigma must be finite, got {sigma!r}")
    return raw


def segment_count(raw_params: Sequence[float]) -> int:
    if len(raw_params) % 2:
        raise ValueError("raw parameter array must have even length")
    return len(raw_params) // 2


def design_dim(raw_params: Sequence[float]) -> int:
    if len(raw_params) % 2:
        raise ValueError("raw parameter array must have even length")
    return len(raw_params)


def total_length(raw_params: Sequence[float]) -> float:
    if len(raw_params) % 2:
        raise ValueError("raw parameter array must have even length")
    return math.fsum(float(raw_params[i]) for i in range(0, len(raw_params), 2))


def mode_name(mode: SegmentType) -> str:
    return getattr(mode, "name", str(mode))


def rows_from_flat_jac8(J: Sequence[float]) -> Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]:
    if len(J) != 8:
        raise ValueError(f"expected flattened 2x4 jacobian of length 8, got {len(J)}")
    return tuple(float(x) for x in J[:4]), tuple(float(x) for x in J[4:8])


def dot_rows_with_state_adj(
    Jw: Sequence[float],
    Jk: Sequence[float],
    aw: float,
    ak: float,
) -> List[float]:
    return [aw * Jw[i] + ak * Jk[i] for i in range(4)]


# =============================================================================
# Compact production records
# =============================================================================

@dataclass(slots=True)
class ProdTraversalPiece:
    L: float
    sigma: float
    piece_index: int
    abs0: float
    direction: float


@dataclass(slots=True)
class ProdScalarSegment:
    kind: PassKind
    mode: SegmentType
    event: EventKind

    traversal_index: int
    piece_index: int

    offset0: float
    L_used: float
    sigma: float

    abs0: float
    abs1: float
    direction: float

    w0: float
    k0: float
    w1: float
    k1: float

    # The first grip segment of a repelling internal-cap pass can use Cflow
    # exact inward-boundary continuation rather than numerical interiorization.
    boundary_start: bool = False

    # Scalar value segment retained for envelope/value queries only.
    seg: Any = None


@dataclass(slots=True)
class ProdScalarPass:
    kind: PassKind
    pieces: List[ProdTraversalPiece]
    segments: List[ProdScalarSegment]
    final_w: float
    final_k: float
    final_mode: SegmentType
    # A pass can terminate when its extremal trajectory reaches the friction
    # boundary before another ordinary switching surface.  Such a pass is
    # still useful up to that point; internal curvature-cap passes cover the
    # remainder of a general multi-bottleneck profile.
    terminated_at_domain: bool = False
    # Knot whose curvature initializes this pass.  ``None`` means the fixed
    # initial curvature.  ``n_pieces`` denotes the geometry endpoint.
    initial_knot_index: int | None = None
    # Derivative of the pass's initial speed with respect to its initial
    # curvature.  It is nonzero for an internal curvature-cap anchor.
    initial_w_dk: float = 0.0


@dataclass(slots=True)
class EnvelopePiece:
    source: PassKind
    source_index: int
    abs0: float
    abs1: float
    local0: float
    local1: float
    pass_index: int = 0


@dataclass(slots=True)
class ProdSpeedProfileBuild:
    raw_params: List[float]
    forward_pieces: List[ProdTraversalPiece]
    backward_pieces: List[ProdTraversalPiece]
    forward_scalar: ProdScalarPass
    backward_scalar: ProdScalarPass
    scalar_passes: List[ProdScalarPass]
    envelope: List[EnvelopePiece]
    # Optional only for backward compatibility with callers that manually
    # instantiate the compact build record.  Production builders always fill
    # this field.
    anchor_stats: "AnchorInsertionStats | None" = None


@dataclass(frozen=True, slots=True)
class InternalCapAnchor:
    """One internal curvature knot whose friction cap is dynamically active."""

    knot_index: int
    station: float
    signed_k: float
    cap_w: float
    initial_w_dk: float


def _cap_releases_into_piece(
    piece: ProdTraversalPiece,
    signed_k: float,
) -> bool:
    """Return whether an exact lateral-cap state enters the real domain.

    On the grip boundary ``w*|k| = MU_G`` the state derivative vanishes, so

        d/ds [MU_G**2 - (w*k)**2] = -2*w**2*k*sigma.

    A cap-launched extremal therefore exists into a traversal piece only when
    ``k*sigma < 0``.  The opposite sign makes the lateral cap tighten
    immediately; interiorizing the anchor merely creates a vanishing numerical
    prefix and should not be treated as a physical candidate pass.
    """
    return piece.sigma * signed_k < 0.0


@dataclass(frozen=True, slots=True)
class AnchorWitness:
    """A path interval that the current candidate family does not close."""

    abs0: float
    abs1: float
    kind: str
    magnitude: float = 0.0


@dataclass(slots=True)
class AnchorInsertionStats:
    """Diagnostics for exact batch internal-cap construction."""

    possible_anchors: int
    inserted_anchor_indices: List[int] = field(default_factory=list)
    rounds: int = 0
    coverage_witnesses: int = 0
    continuity_witnesses: int = 0

    @property
    def inserted_anchors(self) -> int:
        return len(self.inserted_anchor_indices)


# =============================================================================
# Traversal construction and endpoint geometry
# =============================================================================

def make_forward_traversal(raw_params: Sequence[float]) -> List[ProdTraversalPiece]:
    n = segment_count(raw_params)
    out: List[ProdTraversalPiece] = []
    s = 0.0
    for i in range(n):
        L = float(raw_params[2 * i])
        sigma = float(raw_params[2 * i + 1])
        out.append(ProdTraversalPiece(L, sigma, i, s, 1.0))
        s += L
    return out


def make_backward_traversal_flip_sigma(raw_params: Sequence[float]) -> List[ProdTraversalPiece]:
    n = segment_count(raw_params)
    starts = [0.0] * n
    s = 0.0
    for i in range(n):
        starts[i] = s
        s += float(raw_params[2 * i])

    out: List[ProdTraversalPiece] = []
    for i in range(n - 1, -1, -1):
        L = float(raw_params[2 * i])
        sigma = float(raw_params[2 * i + 1])
        out.append(ProdTraversalPiece(L, -sigma, i, starts[i] + L, -1.0))
    return out


def geometry_knot_curvatures(
    raw_params: Sequence[float],
    *,
    initial_k: float = 0.0,
) -> List[float]:
    """Return ``[k_0, ..., k_n]`` for the raw linear-curvature path."""
    raw = validate_raw_parameters(raw_params)
    out = [float(initial_k)]
    if not math.isfinite(out[0]):
        raise ValueError("initial_k must be finite")
    for i in range(len(raw) // 2):
        out.append(math.fma(raw[2 * i + 1], raw[2 * i], out[-1]))
    return out


def geometry_endpoint_k(raw_params: Sequence[float], *, initial_k: float = 0.0) -> float:
    k = float(initial_k)
    if not math.isfinite(k):
        raise ValueError("initial_k must be finite")
    for i in range(segment_count(raw_params)):
        k = math.fma(float(raw_params[2 * i + 1]), float(raw_params[2 * i]), k)
    return k


def add_geometry_endpoint_k_adjoint(
    raw_params: Sequence[float],
    grad: List[float],
    *,
    adj_k_end: float,
) -> None:
    if adj_k_end == 0.0:
        return
    n = segment_count(raw_params)
    if len(grad) != 2 * n:
        raise ValueError("gradient length does not match raw parameters")
    for i in range(n):
        L = float(raw_params[2 * i])
        sigma = float(raw_params[2 * i + 1])
        grad[2 * i] += adj_k_end * sigma
        grad[2 * i + 1] += adj_k_end * L


def add_geometry_knot_k_adjoint(
    raw_params: Sequence[float],
    grad: List[float],
    *,
    knot_index: int,
    adj_k: float,
) -> None:
    """Accumulate an adjoint on curvature at an arbitrary geometry knot."""
    if adj_k == 0.0:
        return
    n = segment_count(raw_params)
    if not 0 <= knot_index <= n:
        raise ValueError(f"knot_index={knot_index} outside [0, {n}]")
    if len(grad) != 2 * n:
        raise ValueError("gradient length does not match raw parameters")
    for i in range(knot_index):
        L = float(raw_params[2 * i])
        sigma = float(raw_params[2 * i + 1])
        grad[2 * i] += adj_k * sigma
        grad[2 * i + 1] += adj_k * L


# =============================================================================
# Raw gradient accumulation
# =============================================================================

@dataclass(slots=True)
class RawGradientAccumulator:
    n_pieces: int
    grad: List[float]
    length_prefix_diff: List[float]

    @classmethod
    def create(cls, n_pieces: int) -> "RawGradientAccumulator":
        return cls(n_pieces, zeros(2 * n_pieces), zeros(n_pieces + 1))

    def add_length(self, piece_index: int, adj: float) -> None:
        if adj != 0.0:
            self.grad[2 * piece_index] += adj

    def add_raw_sigma(self, piece_index: int, adj: float) -> None:
        if adj != 0.0:
            self.grad[2 * piece_index + 1] += adj

    def add_traversal_sigma(self, scalar: ProdScalarSegment, adj_sigma_traversal: float) -> None:
        if adj_sigma_traversal == 0.0:
            return
        if scalar.kind is PassKind.FORWARD:
            self.add_raw_sigma(scalar.piece_index, adj_sigma_traversal)
        elif scalar.kind is PassKind.BACKWARD:
            self.add_raw_sigma(scalar.piece_index, -adj_sigma_traversal)
        else:
            raise ValueError(scalar.kind)

    def add_prefix_exclusive(self, piece_index: int, adj: float) -> None:
        # adj * d/dL of sum_{j < piece_index} L_j
        if adj == 0.0 or piece_index <= 0:
            return
        self.length_prefix_diff[0] += adj
        self.length_prefix_diff[piece_index] -= adj

    def add_prefix_inclusive(self, piece_index: int, adj: float) -> None:
        # adj * d/dL of sum_{j <= piece_index} L_j
        if adj == 0.0:
            return
        end = piece_index + 1
        if not (0 < end <= self.n_pieces):
            raise IndexError((piece_index, self.n_pieces))
        self.length_prefix_diff[0] += adj
        self.length_prefix_diff[end] -= adj

    def add_abs0(self, scalar: ProdScalarSegment, adj_abs0: float) -> None:
        if adj_abs0 == 0.0:
            return
        if scalar.kind is PassKind.FORWARD:
            self.add_prefix_exclusive(scalar.piece_index, adj_abs0)
        elif scalar.kind is PassKind.BACKWARD:
            self.add_prefix_inclusive(scalar.piece_index, adj_abs0)
        else:
            raise ValueError(scalar.kind)

    def add_global_end(self, adj_end_abs: float) -> None:
        if adj_end_abs == 0.0 or self.n_pieces == 0:
            return
        self.length_prefix_diff[0] += adj_end_abs
        self.length_prefix_diff[self.n_pieces] -= adj_end_abs

    def finalize(self) -> List[float]:
        out = list(self.grad)
        running = 0.0
        for i in range(self.n_pieces):
            running += self.length_prefix_diff[i]
            out[2 * i] += running
        return out


# =============================================================================
# Domain helpers
# =============================================================================

def friction_g2(w: float, k: float) -> float:
    q = w * k
    return math.fma(-q, q, MU_G2)


def friction_domain_ok(w: float, k: float, *, margin: float = FRICTION_DOMAIN_MARGIN) -> bool:
    if not (math.isfinite(w) and math.isfinite(k)) or w <= 0.0:
        return False
    G2 = friction_g2(w, k)
    return math.isfinite(G2) and G2 > margin


def require_friction_domain_state(
    w: float,
    k: float,
    *,
    where: str = "",
    margin: float = FRICTION_DOMAIN_MARGIN,
) -> None:
    if not friction_domain_ok(w, k, margin=margin):
        G2 = friction_g2(w, k) if math.isfinite(w) and math.isfinite(k) else float("nan")
        raise FloatingPointError(
            f"friction domain violated {where}: w={w:.17g}, k={k:.17g}, "
            f"G2={G2:.17g}, margin={margin:.17g}"
        )


def segment_friction_g2(seg: Any, ds: float) -> float:
    w = seg.w(ds)
    k = math.fma(seg.sigma, ds, seg.k0)
    return friction_g2(w, k)


def _bisect_first_domain_edge(
    seg: Any,
    lo: float,
    hi: float,
    f_lo: float,
    f_hi: float,
    *,
    margin: float,
    max_iter: int = 80,
) -> float:
    a = lo
    b = hi
    _ = f_lo, f_hi
    for _i in range(max_iter):
        m = 0.5 * (a + b)
        try:
            fm = segment_friction_g2(seg, m) - margin
        except Exception:
            fm = -float("inf")
        if not math.isfinite(fm):
            fm = -float("inf")
        if fm > 0.0:
            a = m
        else:
            b = m
    return b


@dataclass(frozen=True, slots=True)
class DomainClip:
    """Certified interior prefix immediately before a friction-domain edge.

    ``safe`` evaluates strictly inside ``G² > margin`` and ``edge`` is the
    first refined point on or outside the requested margin.  The distinction
    matters near the lateral cap: using the invalid-side root directly can
    reproduce an endpoint violation after scalar or differentiable replay.
    """

    safe: float
    edge: float
    safe_g2: float
    edge_g2: float


def _segment_g2_or_negative_infinity(seg: Any, ds: float) -> float:
    try:
        value = segment_friction_g2(seg, ds)
    except Exception:
        return -math.inf
    return value if math.isfinite(value) else -math.inf



def _friction_replay_target(margin: float) -> float:
    """State-space cushion used for scalar replay before the friction edge."""
    return max(4.0 * margin, 128.0 * math.ulp(MU_G2))

def first_domain_clip(
    seg: Any,
    L: float,
    *,
    margin: float = FRICTION_DOMAIN_MARGIN,
    n_scan: int = FRICTION_DOMAIN_SCAN,
    max_iter: int = 96,
) -> DomainClip | None:
    """Return a replay-safe prefix before the first detected domain edge.

    This is an exceptional certification path, not the normal shared scan.
    It is used when a candidate switching root was rejected after the shared
    scanner stopped at it, or when the emitted endpoint itself disproves the
    scanner result.  Bisection tracks both sides of the boundary and returns
    the valid side, avoiding arbitrary station-length guards.
    """
    if L <= 0.0:
        return None
    if n_scan <= 0:
        raise ValueError("n_scan must be positive")

    g_prev = _segment_g2_or_negative_infinity(seg, 0.0)
    if g_prev <= margin:
        return DomainClip(0.0, 0.0, g_prev, g_prev)

    s_prev = 0.0
    for j in range(1, n_scan + 1):
        s = L * (j / n_scan)
        g = _segment_g2_or_negative_infinity(seg, s)
        if g <= margin:
            lo = s_prev
            hi = s
            g_lo = g_prev
            g_hi = g
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if mid == lo or mid == hi:
                    break
                g_mid = _segment_g2_or_negative_infinity(seg, mid)
                if g_mid > margin:
                    lo = mid
                    g_lo = g_mid
                else:
                    hi = mid
                    g_hi = g_mid

            # ``lo`` is mathematically on the valid side, but retain a modest
            # G² replay cushion when the local bracket contains one.  This is
            # a state-space guard, unlike the former fixed 1e-10 distance
            # subtraction, and therefore adapts to the local boundary slope.
            replay_target = _friction_replay_target(margin)
            if g_prev >= replay_target and g_lo < replay_target:
                a = s_prev
                b = lo
                g_a = g_prev
                for _ in range(max_iter):
                    mid = 0.5 * (a + b)
                    if mid == a or mid == b:
                        break
                    g_mid = _segment_g2_or_negative_infinity(seg, mid)
                    if g_mid >= replay_target:
                        a = mid
                        g_a = g_mid
                    else:
                        b = mid
                lo = a
                g_lo = g_a

            if not (0.0 <= lo <= hi <= L):
                raise AssertionError((lo, hi, L))
            if lo > 0.0 and not g_lo > margin:
                raise FloatingPointError(
                    "failed to recover a friction-domain interior prefix: "
                    f"safe={lo:.17g}, edge={hi:.17g}, "
                    f"safe_G2={g_lo:.17g}, edge_G2={g_hi:.17g}, "
                    f"margin={margin:.17g}"
                )
            return DomainClip(lo, hi, g_lo, g_hi)
        s_prev = s
        g_prev = g
    return None


def first_domain_edge(
    seg: Any,
    L: float,
    *,
    margin: float = FRICTION_DOMAIN_MARGIN,
    n_scan: int = FRICTION_DOMAIN_SCAN,
) -> Optional[float]:
    if L <= 0.0:
        return None

    try:
        f_prev = segment_friction_g2(seg, 0.0) - margin
    except Exception:
        return 0.0

    if not math.isfinite(f_prev) or f_prev <= 0.0:
        return 0.0

    s_prev = 0.0
    for j in range(1, n_scan + 1):
        s = L * (j / n_scan)
        try:
            f = segment_friction_g2(seg, s) - margin
        except Exception:
            return _bisect_first_domain_edge(seg, s_prev, s, f_prev, -float("inf"), margin=margin)
        if not math.isfinite(f) or f <= 0.0:
            return _bisect_first_domain_edge(
                seg,
                s_prev,
                s,
                f_prev,
                f if math.isfinite(f) else -float("inf"),
                margin=margin,
            )
        s_prev = s
        f_prev = f
    return None


def require_segment_friction_domain(
    seg: Any,
    L: float,
    *,
    where: str = "",
    margin: float = FRICTION_DOMAIN_MARGIN,
    n_scan: int = FRICTION_DOMAIN_SCAN,
) -> None:
    edge = first_domain_edge(seg, L, margin=margin, n_scan=n_scan)
    if edge is not None:
        G2 = segment_friction_g2(seg, edge if edge > 0.0 else 0.0)
        raise FloatingPointError(
            f"segment violates friction domain {where}: edge={edge:.17g}, "
            f"L={L:.17g}, G2={G2:.17g}, margin={margin:.17g}"
        )


# =============================================================================
# Segment wrappers and local value functions
# =============================================================================

def _compile_segment_profiled(
    mode: SegmentType,
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    *,
    grad: bool,
    profiler: PhaseProfiler | None,
    phase: str,
    boundary_start: bool = False,
    reverse_eta: bool = False,
    authoritative_w1: float | None = None,
) -> Any:
    if boundary_start:
        if mode is not SegmentType.GRIP or k0 == 0.0 or not sigma / k0 < 0.0:
            raise ValueError("invalid exact boundary-start segment request")
        expected_w0 = MU_G / abs(k0)
        tolerance = 64.0 * math.ulp(max(expected_w0, abs(w0), 1.0))
        if abs(w0 - expected_w0) > tolerance:
            raise ValueError("boundary-start speed is not the exact friction cap")
    else:
        require_friction_domain_state(
            w0,
            k0,
            where=f"compile_segment mode={mode_name(mode)} grad={grad}",
        )
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    try:
        options: dict[str, Any] = {}
        if boundary_start:
            options["boundary_start"] = True
        if grad and mode is SegmentType.GRIP and sigma != 0.0 and reverse_eta:
            options["reverse_eta"] = True
            options["authoritative_w1"] = authoritative_w1
        return compile_segment(
            L, sigma, w0, k0, mode, grad=grad, **options
        )
    finally:
        if profiler is not None and profiler.enabled:
            profiler.record(phase, perf_counter() - started)


def compile_eval(
    mode: SegmentType,
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    *,
    profiler: PhaseProfiler | None = None,
    phase: str = TEMP_SCALAR_COMPILE,
    boundary_start: bool = False,
) -> Any:
    return _compile_segment_profiled(
        mode,
        L,
        sigma,
        w0,
        k0,
        grad=False,
        profiler=profiler,
        phase=phase,
        boundary_start=boundary_start,
    )



@dataclass(frozen=True, slots=True)
class ProbeResult:
    segment: Any
    length: float
    real_domain_limited: bool
    event_position: float | None
    status: int | None = None


def compile_eval_probe(
    mode: SegmentType,
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    *,
    profiler: PhaseProfiler | None = None,
    min_prefix: float = 1e-13,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    boundary_start: bool = False,
) -> ProbeResult:
    """Construct one scalar segment and return its largest safe real prefix.

    Cflow segments are direct evaluators rather than interval compilers.  For a
    general GRIP request we inspect the structured endpoint/event status once;
    only a beyond-event result invokes the native first-event machinery.  No
    exception-text parsing or recompilation bisection remains.
    """
    try:
        seg = compile_eval(
            mode, L, sigma, w0, k0, profiler=profiler,
            phase=TEMP_SCALAR_COMPILE, boundary_start=boundary_start,
        )
    except ArithmeticError:
        # Exact constant-curvature GRIP remains routed through the existing
        # circular closed form.  Its constructor intentionally rejects an
        # interval extending past the first friction maximum.  Recover that
        # endpoint structurally from a zero-length circular object; never parse
        # exception text or bisect compiler failures.
        if mode is not SegmentType.GRIP or sigma != 0.0 or boundary_start:
            raise
        seed = compile_eval(
            mode, 0.0, sigma, w0, k0, profiler=profiler,
            phase=TEMP_SCALAR_COMPILE, boundary_start=False,
        )
        event = getattr(seed, "domain_end", None)
        if event is None or not math.isfinite(float(event)) or not (0.0 < float(event) <= L):
            raise
        event = float(event)
        event_seg = compile_eval(
            mode, event, sigma, w0, k0, profiler=profiler,
            phase=TEMP_SCALAR_COMPILE, boundary_start=False,
        )
        clip = first_domain_clip(
            event_seg, event, margin=domain_margin, n_scan=domain_scan
        )
        floor = max(min_prefix, 64.0 * math.ulp(max(1.0, L)))
        safe = clip.safe if clip is not None and clip.safe > 0.0 else 0.5 * event
        if safe < floor and event > floor:
            safe = floor
        if not safe < event:
            safe = math.nextafter(event, 0.0)
        seg = compile_eval(
            mode, safe, sigma, w0, k0, profiler=profiler,
            phase=TEMP_SCALAR_COMPILE, boundary_start=False,
        )
        return ProbeResult(seg, safe, True, event, None)

    if mode is not SegmentType.GRIP or not hasattr(seg, "domain_probe"):
        return ProbeResult(seg, L, False, None, None)

    probe = seg.domain_probe(L)
    if probe.pre_event:
        return ProbeResult(seg, L, False, None, probe.status)
    event = probe.event_position
    if event is None or not math.isfinite(event) or not (0.0 < event <= L):
        raise FloatingPointError(
            "Cflow reported a GRIP domain limit without a finite first-event position"
        )
    floor = max(min_prefix, 64.0 * math.ulp(max(1.0, L)))
    # The event-time operator and the authoritative state map can differ by a
    # few ulps of *state* extremely close to the cap.  Select the probe endpoint
    # using the same G^2 interior criterion used by the switching scanner, not
    # by subtracting an absolute or relative time epsilon from the event.
    clip = first_domain_clip(
        seg, event, margin=domain_margin, n_scan=domain_scan
    )
    safe = clip.safe if clip is not None and clip.safe > 0.0 else 0.5 * event
    if safe < floor and event > floor:
        safe = floor
    if not safe < event:
        safe = math.nextafter(event, 0.0)
    return ProbeResult(seg, safe, True, event, probe.status)


def compile_diff(
    mode: SegmentType,
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    *,
    profiler: PhaseProfiler | None = None,
    boundary_start: bool = False,
    reverse_eta: bool = False,
    authoritative_w1: float | None = None,
) -> Any:
    """Construct the direct differentiable segment; Cflow has no piece budget."""
    return _compile_segment_profiled(
        mode, L, sigma, w0, k0, grad=True, profiler=profiler,
        phase=DIFF_REPLAY_COMPILE, boundary_start=boundary_start,
        reverse_eta=reverse_eta, authoritative_w1=authoritative_w1,
    )


def segment_end_state(seg: Any, ds: float) -> Tuple[float, float]:
    return seg.w(ds), math.fma(seg.sigma, ds, seg.k0)


def local_dw_ds_for_mode(mode: SegmentType, seg: Any, ds: float) -> float:
    w = seg.w(ds)
    if mode is SegmentType.MOTOR:
        return 2.0 * math.fma(B_EMF, -math.sqrt(w), A_MAX)
    if mode is SegmentType.GRIP:
        k = math.fma(seg.sigma, ds, seg.k0)
        G2 = friction_g2(w, k)
        if G2 <= 0.0:
            raise FloatingPointError(f"GRIP dw/ds outside friction domain: ds={ds}, w={w}, k={k}, G2={G2}")
        return 2.0 * math.sqrt(G2)
    if mode is SegmentType.BRAKE:
        return S_BRAKE
    raise ValueError(mode)


def abs_to_local(rec: ProdScalarSegment, abs_s: float) -> float:
    """Convert a global station to the record's local coordinate.

    Long paths accumulate a few ulps of station-coordinate roundoff when
    independently constructed candidate intervals are intersected.  Clamp only
    those endpoint-sized discrepancies; a materially out-of-record query still
    reaches the segment evaluator and raises normally.
    """
    ds = rec.direction * (abs_s - rec.abs0)
    tolerance = 32.0 * math.ulp(
        max(abs(abs_s), abs(rec.abs0), abs(rec.abs1), abs(rec.L_used), 1.0)
    )
    if -tolerance <= ds < 0.0:
        return 0.0
    if rec.L_used < ds <= rec.L_used + tolerance:
        return rec.L_used
    return ds


def interval_low_high(rec: ProdScalarSegment) -> Tuple[float, float]:
    return min(rec.abs0, rec.abs1), max(rec.abs0, rec.abs1)


def segment_w_at_abs(rec: ProdScalarSegment, abs_s: float) -> float:
    return rec.seg.w(abs_to_local(rec, abs_s))


def segment_local_dw_ds(rec: ProdScalarSegment, ds: float) -> float:
    return local_dw_ds_for_mode(rec.mode, rec.seg, ds)


def segment_abs_dw_ds(rec: ProdScalarSegment, abs_s: float) -> float:
    ds = abs_to_local(rec, abs_s)
    return rec.direction * segment_local_dw_ds(rec, ds)


# =============================================================================
# Mode and event logic
# =============================================================================

def slack_fwd_value(w: float, k: float) -> float:
    require_friction_domain_state(w, k, where="slack_fwd_value")
    G = math.sqrt(friction_g2(w, k))
    return math.fma(B_EMF, math.sqrt(w), G - A_MAX)


def slack_bwd_value(w: float, k: float) -> float:
    require_friction_domain_state(w, k, where="slack_bwd_value")
    return math.sqrt(friction_g2(w, k)) - A_BRAKE


def choose_initial_mode(kind: PassKind, w: float, k: float) -> SegmentType:
    require_friction_domain_state(w, k, where=f"choose_initial_mode kind={kind}")
    if kind is PassKind.FORWARD:
        h = slack_fwd_value(w, k)
        if w < W_EQ and h >= 0.0:
            return SegmentType.MOTOR
        return SegmentType.GRIP
    if kind is PassKind.BACKWARD:
        h = slack_bwd_value(w, k)
        if h >= 0.0:
            return SegmentType.BRAKE
        return SegmentType.GRIP
    raise ValueError(kind)


def crossing_for_mode(kind: PassKind, mode: SegmentType):
    if kind is PassKind.FORWARD:
        if mode is SegmentType.MOTOR:
            return EventKind.MOTOR_GRIP, SegmentType.GRIP, motor_grip_scan
        if mode is SegmentType.GRIP:
            return EventKind.GRIP_MOTOR, SegmentType.MOTOR, grip_motor_scan
    if kind is PassKind.BACKWARD:
        if mode is SegmentType.BRAKE:
            return EventKind.BRAKE_GRIP, SegmentType.GRIP, brake_grip_scan
        if mode is SegmentType.GRIP:
            return EventKind.GRIP_BRAKE, SegmentType.BRAKE, grip_brake_scan
    raise ValueError((kind, mode))


def _event_partials_wk(event: EventKind, w: float, k: float) -> Tuple[float, float]:
    q = w * k
    G2 = math.fma(-q, q, MU_G2)
    if G2 <= 0.0:
        raise FloatingPointError("event partial requested at/outside friction boundary")
    G = math.sqrt(G2)

    if event in (EventKind.GRIP_MOTOR, EventKind.MOTOR_GRIP):
        y = math.sqrt(max(0.0, w))
        if y <= 0.0:
            raise FloatingPointError("forward slack derivative singular at w <= 0")
        return -q * k / G + B_EMF / (2.0 * y), -q * w / G

    if event in (EventKind.GRIP_BRAKE, EventKind.BRAKE_GRIP):
        return -q * k / G, -q * w / G

    raise ValueError(event)


def _coordinate_spatial_resolution(value: float, derivative: float) -> float:
    """First-order distance associated with one binary64 coordinate ulp."""
    if not (math.isfinite(value) and math.isfinite(derivative)) or derivative == 0.0:
        return math.inf
    target = math.nextafter(
        value,
        math.inf if derivative > 0.0 else -math.inf,
    )
    delta = abs(target - value)
    if not math.isfinite(delta) or delta <= 0.0:
        return math.inf
    distance = delta / abs(derivative)
    return distance if math.isfinite(distance) and distance > 0.0 else math.inf


def _initial_event_state_spatial_tolerance(
    event: EventKind,
    mode: SegmentType,
    seg: Any,
    L: float,
) -> float:
    """Return a state-aware tolerance for classifying a root at ``ds=0``.

    The event locator's nominal x tolerance does not account for the floating
    state map.  Near motor equilibrium or the friction cap, a positive event
    root can be shorter than either (a) the distance needed for ``w`` or ``k``
    to advance by one representable value, or (b) the root uncertainty induced
    by one-ulp uncertainty in those state coordinates.  Emitting such an
    interval leaves an indistinguishable state and permits an immediate
    opposite-mode switch.

    One full coordinate ulp is conservative under round-to-nearest.  The
    event-residual uncertainty is propagated with its analytic ``(F_w,F_k)``
    partials.  The cap confines this classifier to numerically microscopic
    intervals; resolved roots retain the ordinary bracketed solve.
    """
    w0 = float(seg.w0)
    k0 = float(seg.k0)
    sigma = float(seg.sigma)

    if mode is SegmentType.MOTOR:
        dw = 2.0 * math.fma(B_EMF, -math.sqrt(max(0.0, w0)), A_MAX)
    elif mode is SegmentType.GRIP:
        G2 = friction_g2(w0, k0)
        dw = 2.0 * math.sqrt(max(0.0, G2))
    elif mode is SegmentType.BRAKE:
        dw = S_BRAKE
    else:
        raise ValueError(mode)

    coordinate_resolution = min(
        _coordinate_spatial_resolution(w0, dw),
        _coordinate_spatial_resolution(k0, sigma),
    )
    if not math.isfinite(coordinate_resolution):
        coordinate_resolution = 0.0

    residual_resolution = 0.0
    try:
        F, dF, _ = event_residual_df_g2(event, mode, seg, 0.0)
        F_w, F_k = _event_partials_wk(event, w0, k0)
        if math.isfinite(F) and dF is not None and math.isfinite(dF) and dF != 0.0:
            residual_uncertainty = math.fsum(
                (
                    abs(F_w) * math.ulp(w0),
                    abs(F_k) * math.ulp(k0),
                    8.0 * math.ulp(max(abs(F), A_MAX, A_BRAKE, 1.0)),
                )
            )
            residual_resolution = residual_uncertainty / abs(dF)
    except (FloatingPointError, ValueError):
        pass

    resolution = max(coordinate_resolution, residual_resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        return 0.0
    cap = EVENT_START_MAX_SPATIAL_TOL * max(1.0, abs(L))
    return min(resolution, cap)


def event_residual_df_g2(event: EventKind, mode: SegmentType, seg: Any, ds: float):
    w = seg.w(ds)
    k = math.fma(seg.sigma, ds, seg.k0)
    G2 = friction_g2(w, k)
    if w <= 0.0 or G2 <= 0.0 or not math.isfinite(G2):
        return float("nan"), None, G2
    G = math.sqrt(G2)

    if event in (EventKind.MOTOR_GRIP, EventKind.GRIP_MOTOR):
        F = math.fma(B_EMF, math.sqrt(w), G - A_MAX)
    elif event in (EventKind.BRAKE_GRIP, EventKind.GRIP_BRAKE):
        F = G - A_BRAKE
    else:
        raise ValueError(event)

    F_w, F_k = _event_partials_wk(event, w, k)
    dF = math.fma(F_w, local_dw_ds_for_mode(mode, seg, ds), F_k * seg.sigma)
    return F, dF, G2


def validate_crossing_root(
    event: EventKind,
    mode: SegmentType,
    seg: Any,
    r: float | None,
    *,
    residual_tol: float = EVENT_RESIDUAL_TOL,
    deriv_tol: float = EVENT_DERIV_TOL,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
) -> float | None:
    if r is None or not math.isfinite(r):
        return None
    try:
        F, dF, G2 = event_residual_df_g2(event, mode, seg, r)
    except Exception:
        return None
    if dF is None:
        return None
    if not (math.isfinite(F) and math.isfinite(dF) and math.isfinite(G2)):
        return None
    if G2 <= domain_margin or abs(dF) <= deriv_tol or abs(F) > residual_tol:
        return None
    crosses_upward = event in (EventKind.GRIP_MOTOR, EventKind.GRIP_BRAKE)
    if (crosses_upward and dF <= deriv_tol) or (
        not crosses_upward and dF >= -deriv_tol
    ):
        return None
    return r


def local_event_columns(
    seg: Any,
    ds: float,
    event: EventKind,
    *,
    w_and_jac: tuple[float, Sequence[float]] | None = None,
) -> Tuple[float, float, float, float]:
    if w_and_jac is None:
        w1, jac = seg.w_and_jac(ds)
    else:
        w1, jac = w_and_jac
    k1 = math.fma(seg.sigma, ds, seg.k0)
    F_w, F_k = _event_partials_wk(event, w1, k1)
    Jw, Jk = rows_from_flat_jac8(jac)
    return tuple(math.fma(F_w, Jw[i], F_k * Jk[i]) for i in range(4))


# =============================================================================
# Scalar topology pass
# =============================================================================

def build_scalar_pass(
    pieces: Sequence[ProdTraversalPiece],
    init_w: float,
    init_k: float,
    kind: PassKind,
    *,
    init_mode: Optional[SegmentType] = None,
    n_scan: int = 256,
    root_eps: float = 1e-11,
    piece_eps: float = 1e-13,
    max_subsegments_per_piece: int = 128,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
    allow_domain_termination: bool = True,
    initial_knot_index: int | None = None,
    initial_w_dk: float = 0.0,
    fused_grip_discovery: bool = True,
) -> ProdScalarPass:
    require_friction_domain_state(
        init_w,
        init_k,
        where="build_scalar_pass initial",
        margin=domain_margin,
    )
    w = init_w
    k = init_k
    mode = init_mode or choose_initial_mode(kind, w, k)
    exact_boundary_pending = initial_knot_index is not None
    records: List[ProdScalarSegment] = []
    shared_scan = max(n_scan, domain_scan)
    terminated_at_domain = False

    for traversal_index, piece in enumerate(pieces):
        offset = 0.0
        for _seg_iter in range(max_subsegments_per_piece):
            remaining = piece.L - offset
            if remaining <= piece_eps:
                break

            boundary_here = (
                exact_boundary_pending
                and traversal_index == 0
                and offset == 0.0
                and mode is SegmentType.GRIP
                and k != 0.0
                and piece.sigma / k < 0.0
            )
            if boundary_here:
                # Replace the tiny numerical interiorization by the exact cap
                # only for the repelling first grip segment.  Other directions
                # retain the existing safe interior anchor behavior.
                w = MU_G / abs(k)
            else:
                require_friction_domain_state(
                    w,
                    k,
                    where=f"piece {traversal_index} offset {offset} start",
                    margin=domain_margin,
                )

            # One scalar segment supports event/domain discovery, the emitted
            # endpoint, and later scalar value queries.  For general GRIP, the
            # fused path deliberately avoids the former full-horizon
            # ``domain_probe -> first_domain_clip`` pre-pass: it traverses only
            # until the first switch or the same replay-safe G^2 cushion used
            # by ``first_domain_clip``.  Constant-curvature GRIP retains the
            # legacy probe because the circular implementation owns a separate
            # analytic endpoint contract.
            event, next_mode, legacy_scan_fn = crossing_for_mode(kind, mode)
            min_prefix = max(piece_eps, 1e-14 * max(1.0, remaining))
            fused_here = bool(
                fused_grip_discovery
                and mode is SegmentType.GRIP
                and piece.sigma != 0.0
            )
            if fused_here:
                try:
                    seg = compile_eval(
                        mode, remaining, piece.sigma, w, k,
                        profiler=profiler, phase=TEMP_SCALAR_COMPILE,
                        boundary_start=boundary_here,
                    )
                except ArithmeticError:
                    # Preserve the structured legacy recovery for any GRIP
                    # implementation that rejects a full requested horizon.
                    fused_here = False
            if fused_here:
                probe = ProbeResult(seg, remaining, False, None, None)
                probe_L = remaining
                scan_fn = (
                    grip_motor_earliest_scan
                    if kind is PassKind.FORWARD
                    else grip_brake_earliest_scan
                )
                scan_kwargs = dict(
                    n_scan=shared_scan,
                    domain_stop_margin=_friction_replay_target(domain_margin),
                    physical_domain_margin=domain_margin,
                    domain_safe_floor=min_prefix,
                    profiler=profiler,
                    initial_spatial_tol=_initial_event_state_spatial_tolerance(
                        event, mode, seg, probe_L
                    ),
                )
            else:
                probe = compile_eval_probe(
                    mode, remaining, piece.sigma, w, k,
                    profiler=profiler,
                    min_prefix=min_prefix,
                    domain_margin=domain_margin, domain_scan=shared_scan,
                    boundary_start=boundary_here,
                )
                seg = probe.segment
                probe_L = probe.length
                scan_fn = legacy_scan_fn
                scan_kwargs = dict(
                    n_scan=shared_scan, domain_margin=domain_margin,
                    profiler=profiler,
                    initial_spatial_tol=_initial_event_state_spatial_tolerance(
                        event, mode, seg, probe_L
                    ),
                )
            if boundary_here:
                exact_boundary_pending = False
                scan_kwargs["allow_initial_boundary"] = True
            scan = scan_fn(seg, probe_L, **scan_kwargs)

            if fused_here and scan.event is None and scan.domain_edge is None:
                # A long one-shot Cflow probe can terminate at its native
                # local-step budget even though the same trajectory is
                # traversable as short semigroup increments.  Preserve that
                # legacy conditioning outcome for no-event/no-domain horizons
                # without imposing the speculative long probe on ordinary
                # scans.  20,000 is the native CFLOW_MAX_STEPS contract.
                stats = getattr(seg, "_prefix_cache_stats", lambda: {})()
                if int(stats.get("local_steps", 0)) >= 20000:
                    seg.domain_probe(probe_L)

            if fused_here and scan.domain_edge is not None and scan.event is None:
                # The physical domain won before any valid switching event.
                # At that point there is no future propagation to save, so
                # delegate endpoint selection to the unchanged legacy domain
                # certifier.  This deliberately preserves its few-ulp
                # event/state reconciliation and half-event fallback exactly.
                # The fused optimization therefore changes only switch-first
                # and no-domain traversals, where the old full-horizon probe
                # was provably speculative work.
                probe = compile_eval_probe(
                    mode, remaining, piece.sigma, w, k,
                    profiler=profiler,
                    min_prefix=min_prefix,
                    domain_margin=domain_margin, domain_scan=shared_scan,
                    boundary_start=boundary_here,
                )
                seg = probe.segment
                probe_L = probe.length
                if scan.domain_switch_excluded:
                    # A structural theorem has already excluded the mode switch
                    # through the physical terminal.  Keep compile_eval_probe as
                    # the unchanged endpoint authority (including the historical
                    # half-event seam), but do not rescan its certified safe
                    # prefix for an impossible switch.
                    scan = ScanResult(None, None)
                else:
                    scan = legacy_scan_fn(
                        seg, probe_L,
                        n_scan=shared_scan, domain_margin=domain_margin,
                        profiler=profiler,
                        allow_initial_boundary=boundary_here,
                        initial_spatial_tol=_initial_event_state_spatial_tolerance(
                            event, mode, seg, probe_L
                        ),
                    )
                fused_here = False

            # A zero-location scanner sentinel means the current mode is
            # already invalid at this station (or a directional event lies
            # exactly at the start).  Switch without emitting a zero-length
            # segment.  A genuine positive root can also be much smaller than
            # ``root_eps``; it must *not* be treated as this sentinel, because
            # failing to consume it leaves the state on the pre-crossing side
            # and can toggle modes forever at an unchanged offset.
            if scan.initial_switch:
                mode = next_mode
                continue

            r = validate_crossing_root(
                event,
                mode,
                seg,
                scan.event,
                domain_margin=domain_margin,
            )
            event_limited = r is not None and 0.0 < r < remaining - piece_eps
            boundary_event = (
                r is not None
                and remaining - piece_eps <= r <= remaining + root_eps
            )
            edge = scan.domain_edge
            recovered_clip: DomainClip | None = None
            if scan.event is not None and r is None and edge is None:
                # The shared scan stops at the first candidate switching root,
                # while ``validate_crossing_root`` deliberately applies stricter
                # domain, residual, and transversality checks.  If validation
                # rejects that candidate, the scan has not inspected the rest
                # of the segment and therefore cannot certify that no later
                # friction edge exists.  Recover that edge only on this
                # exceptional path.
                recovered_clip = first_domain_clip(
                    seg,
                    probe_L,
                    margin=domain_margin,
                    n_scan=shared_scan,
                )
                if recovered_clip is not None:
                    edge = recovered_clip.edge

            domain_before_event = edge is not None and (
                not event_limited or edge <= float(r) + max(root_eps, piece_eps)
            )
            compiler_domain_stop = (not fused_here) and probe.real_domain_limited and r is None
            terminate_after = domain_before_event or compiler_domain_stop

            if terminate_after and not allow_domain_termination:
                if domain_before_event:
                    raise FloatingPointError(
                        f"friction-domain edge reached before valid event: kind={kind}, "
                        f"mode={mode_name(mode)}, traversal={traversal_index}, offset={offset:.17g}, "
                        f"remaining={remaining:.17g}, edge={edge:.17g}, r={r}"
                    )
                approx_edge = probe.event_position if probe.event_position is not None else probe_L
                raise FloatingPointError(
                    "grip segment reaches its friction endpoint before a valid "
                    f"switching event: kind={kind}, mode={mode_name(mode)}, "
                    f"traversal={traversal_index}, offset={offset:.17g}, "
                    f"remaining={remaining:.17g}, safe_prefix={probe_L:.17g}, "
                    f"event_position={approx_edge:.17g}, status={probe.status!r}"
                )

            if r is not None and r <= 0.0:
                # Exact zero roots should normally arrive as ``initial_switch``.
                # Retain this defensive path for custom scanners, but do not
                # classify a positive sub-tolerance root as zero.
                mode = next_mode
                continue

            if terminate_after:
                if fused_here and scan.domain_safe is not None:
                    # If Cflow itself terminated inside a coarse scan step,
                    # reproduce the legacy endpoint recertification from a
                    # fresh anchor.  This preserves the deliberate half-event
                    # fallback used when the event-time operator and sequential
                    # state replay disagree by a few ulps near the cap.  The
                    # important saving remains: this recertification is paid
                    # only when the *domain* wins; switch-first GRIP intervals
                    # never perform a speculative full-horizon domain probe.
                    if scan.domain_event is not None:
                        cert_seg = compile_eval(
                            mode, remaining, piece.sigma, w, k,
                            profiler=profiler, phase=TEMP_SCALAR_COMPILE,
                            boundary_start=boundary_here,
                        )
                        cert_probe = cert_seg.domain_probe(min(remaining, float(edge)))
                        cert_event = cert_probe.event_position
                        if cert_probe.pre_event or cert_event is None:
                            # The sampled physical-margin crossing is still an
                            # authoritative domain stop even when the native
                            # G^2=0 terminal lies just beyond the scan point.
                            L_used = min(probe_L, float(scan.domain_safe))
                        else:
                            cert_clip = first_domain_clip(
                                cert_seg, float(cert_event),
                                margin=domain_margin, n_scan=shared_scan,
                            )
                            floor = max(min_prefix, 64.0 * math.ulp(max(1.0, remaining)))
                            safe = (
                                cert_clip.safe
                                if cert_clip is not None and cert_clip.safe > 0.0
                                else 0.5 * float(cert_event)
                            )
                            if safe < floor and float(cert_event) > floor:
                                safe = floor
                            if not safe < float(cert_event):
                                safe = math.nextafter(float(cert_event), 0.0)
                            L_used = min(probe_L, safe)
                    else:
                        # A sampled G^2 margin crossing was refined without a
                        # native terminal.  The fused scanner has already
                        # reproduced the physical-edge and replay-cushion
                        # bisections, so no second full scan is necessary.
                        L_used = min(probe_L, float(scan.domain_safe))
                elif edge is None:
                    L_used = probe_L
                else:
                    clip = recovered_clip or first_domain_clip(
                        seg, min(probe_L, edge),
                        margin=domain_margin, n_scan=shared_scan,
                    )
                    if clip is None:
                        raise FloatingPointError(
                            "domain scanner reported an edge that could not be "
                            "recertified: "
                            f"kind={kind}, mode={mode_name(mode)}, "
                            f"traversal={traversal_index}, offset={offset:.17g}, "
                            f"probe_L={probe_L:.17g}, edge={edge:.17g}"
                        )
                    L_used = min(probe_L, clip.safe)
                if L_used <= piece_eps:
                    terminated_at_domain = True
                    return ProdScalarPass(
                        kind,
                        list(pieces),
                        records,
                        w,
                        k,
                        mode,
                        True,
                        initial_knot_index,
                        initial_w_dk,
                    )
                used_event = EventKind.PIECE_END
                mode_after = mode
            elif event_limited:
                L_used = float(r)
                used_event = event
                mode_after = next_mode
            else:
                L_used = remaining
                used_event = EventKind.PIECE_END
                mode_after = next_mode if boundary_event else mode

            w1, k1 = segment_end_state(seg, L_used)
            if not friction_domain_ok(w1, k1, margin=domain_margin):
                # Final certification is intentionally independent of the
                # event scanner.  It catches endpoint-evaluator error and any
                # future scanner/validator mismatch before an invalid state is
                # committed to the scalar pass.  A physical domain edge
                # supersedes a switching event at or beyond it.
                clip = first_domain_clip(
                    seg,
                    L_used,
                    margin=domain_margin,
                    n_scan=max(shared_scan, 2 * domain_scan),
                )
                if clip is None:
                    G2 = friction_g2(w1, k1)
                    raise FloatingPointError(
                        "friction-domain endpoint failed certification but no "
                        "edge bracket was recovered: "
                        f"kind={kind}, mode={mode_name(mode)}, "
                        f"traversal={traversal_index}, offset={offset:.17g}, "
                        f"L_used={L_used:.17g}, probe_L={probe_L:.17g}, "
                        f"candidate_event={scan.event}, validated_event={r}, "
                        f"scanner_edge={scan.domain_edge}, w={w1:.17g}, "
                        f"k={k1:.17g}, G2={G2:.17g}, "
                        f"margin={domain_margin:.17g}"
                    )
                L_used = clip.safe
                terminate_after = True
                used_event = EventKind.PIECE_END
                mode_after = mode
                if L_used <= piece_eps:
                    terminated_at_domain = True
                    return ProdScalarPass(
                        kind,
                        list(pieces),
                        records,
                        w,
                        k,
                        mode,
                        True,
                        initial_knot_index,
                        initial_w_dk,
                    )
                w1, k1 = segment_end_state(seg, L_used)

            require_friction_domain_state(
                w1,
                k1,
                where=(
                    "emitted endpoint after domain certification: "
                    f"kind={kind}, mode={mode_name(mode)}, "
                    f"traversal={traversal_index}, offset={offset:.17g}, "
                    f"L_used={L_used:.17g}, candidate_event={scan.event}, "
                    f"validated_event={r}, scanner_edge={scan.domain_edge}"
                ),
                margin=domain_margin,
            )

            abs0 = piece.abs0 + piece.direction * offset
            abs1 = piece.abs0 + piece.direction * (offset + L_used)
            records.append(
                ProdScalarSegment(
                    kind=kind,
                    mode=mode,
                    event=used_event,
                    traversal_index=traversal_index,
                    piece_index=piece.piece_index,
                    offset0=offset,
                    L_used=L_used,
                    sigma=piece.sigma,
                    abs0=abs0,
                    abs1=abs1,
                    direction=piece.direction,
                    w0=w,
                    k0=k,
                    w1=w1,
                    k1=k1,
                    boundary_start=boundary_here,
                    seg=seg,
                )
            )

            offset += L_used
            w, k = w1, k1
            if terminate_after:
                terminated_at_domain = True
                return ProdScalarPass(
                    kind,
                    list(pieces),
                    records,
                    w,
                    k,
                    mode,
                    True,
                    initial_knot_index,
                    initial_w_dk,
                )
            if used_event is EventKind.PIECE_END:
                mode = mode_after
                break
            mode = mode_after
        else:
            raise RuntimeError(
                "too many subsegments in traversal piece; possible switching "
                f"chatter: kind={kind.name}, traversal={traversal_index}, "
                f"mode={mode_name(mode)}, offset={offset:.17g}, "
                f"piece_length={piece.L:.17g}, sigma={piece.sigma:.17g}, "
                f"w={w:.17g}, k={k:.17g}"
            )

    return ProdScalarPass(
        kind,
        list(pieces),
        records,
        w,
        k,
        mode,
        terminated_at_domain,
        initial_knot_index,
        initial_w_dk,
    )


# =============================================================================
# Lower envelope
# =============================================================================

def _sorted_segment_refs(pass_: ProdScalarPass) -> List[Tuple[float, float, int]]:
    refs: List[Tuple[float, float, int]] = []
    for i, rec in enumerate(pass_.segments):
        lo, hi = interval_low_high(rec)
        if hi > lo:
            refs.append((lo, hi, i))
    refs.sort(key=lambda x: x[0])
    return refs


def _envelope_difference(
    f_rec: ProdScalarSegment,
    b_rec: ProdScalarSegment,
    abs_s: float,
    *,
    with_derivative: bool,
    profiler: PhaseProfiler | None,
) -> tuple[float, float | None]:
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    try:
        value = segment_w_at_abs(f_rec, abs_s) - segment_w_at_abs(b_rec, abs_s)
        if not with_derivative:
            return value, None
        deriv = segment_abs_dw_ds(f_rec, abs_s) - segment_abs_dw_ds(b_rec, abs_s)
        return value, deriv
    finally:
        if profiler is not None and profiler.enabled:
            profiler.record(ENVELOPE_OVERLAP_EVAL, perf_counter() - started)


def _envelope_root_between(
    f_rec: ProdScalarSegment,
    b_rec: ProdScalarSegment,
    lo: float,
    hi: float,
    f_lo: Optional[float] = None,
    f_hi: Optional[float] = None,
    *,
    profiler: PhaseProfiler | None = None,
) -> Optional[float]:
    def f_df(s: float) -> Tuple[float, Optional[float]]:
        return _envelope_difference(
            f_rec,
            b_rec,
            s,
            with_derivative=True,
            profiler=profiler,
        )

    if f_lo is None:
        f_lo, _ = f_df(lo)
    if f_hi is None:
        f_hi, _ = f_df(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        return None
    return bracketed_newton(f_df, lo, hi, f_lo=f_lo, f_hi=f_hi)


def _append_envelope_piece(
    out: List[EnvelopePiece],
    source: PassKind,
    source_index: int,
    rec: ProdScalarSegment,
    abs0: float,
    abs1: float,
    *,
    eps: float,
) -> None:
    if abs1 <= abs0 + eps:
        return
    local0 = abs_to_local(rec, abs0)
    local1 = abs_to_local(rec, abs1)
    if out:
        last = out[-1]
        if last.source is source and last.source_index == source_index and abs(last.abs1 - abs0) <= eps:
            last.abs1 = abs1
            last.local1 = local1
            return
    out.append(EnvelopePiece(source, source_index, abs0, abs1, local0, local1))


def _append_monotone_overlap(
    out: List[EnvelopePiece],
    f_rec: ProdScalarSegment,
    b_rec: ProdScalarSegment,
    f_idx: int,
    b_idx: int,
    a: float,
    b: float,
    F_a: float,
    F_b: float,
    *,
    eps: float,
    profiler: PhaseProfiler | None,
) -> bool:
    """Append one overlap using monotonicity; return False on a violation."""
    scale = max(1.0, abs(F_a), abs(F_b))
    monotone_tol = 64.0 * math.ulp(scale)
    if F_b + monotone_tol < F_a:
        return False

    root: float | None = None
    if abs(F_a) <= monotone_tol:
        root = a
    elif abs(F_b) <= monotone_tol:
        root = b
    elif F_a < 0.0 < F_b:
        root = _envelope_root_between(
            f_rec,
            b_rec,
            a,
            b,
            F_a,
            F_b,
            profiler=profiler,
        )

    if root is None:
        if F_a <= 0.0 and F_b <= 0.0:
            _append_envelope_piece(
                out, PassKind.FORWARD, f_idx, f_rec, a, b, eps=eps
            )
        elif F_a >= 0.0 and F_b >= 0.0:
            _append_envelope_piece(
                out, PassKind.BACKWARD, b_idx, b_rec, a, b, eps=eps
            )
        else:
            return False
        return True

    if root <= a + eps:
        _append_envelope_piece(
            out, PassKind.BACKWARD, b_idx, b_rec, a, b, eps=eps
        )
    elif root >= b - eps:
        _append_envelope_piece(
            out, PassKind.FORWARD, f_idx, f_rec, a, b, eps=eps
        )
    else:
        _append_envelope_piece(
            out, PassKind.FORWARD, f_idx, f_rec, a, root, eps=eps
        )
        _append_envelope_piece(
            out, PassKind.BACKWARD, b_idx, b_rec, root, b, eps=eps
        )
    return True


def _append_overlap_uniform_fallback(
    out: List[EnvelopePiece],
    f_rec: ProdScalarSegment,
    b_rec: ProdScalarSegment,
    f_idx: int,
    b_idx: int,
    a: float,
    b: float,
    *,
    n_scan: int,
    eps: float,
    profiler: PhaseProfiler | None,
) -> None:
    """Defensive fallback for a numerically non-monotone overlap."""
    cuts = [a]
    prev_s = a
    prev_F, _ = _envelope_difference(
        f_rec, b_rec, prev_s, with_derivative=False, profiler=profiler
    )
    for m in range(1, n_scan + 1):
        s = a + (b - a) * (m / n_scan)
        F, _ = _envelope_difference(
            f_rec, b_rec, s, with_derivative=False, profiler=profiler
        )
        if prev_F == 0.0:
            root = prev_s
        elif F == 0.0:
            root = s
        elif prev_F * F < 0.0:
            root = _envelope_root_between(
                f_rec,
                b_rec,
                prev_s,
                s,
                prev_F,
                F,
                profiler=profiler,
            )
        else:
            root = None
        if root is not None and cuts[-1] + eps < root < b - eps:
            cuts.append(root)
        prev_s = s
        prev_F = F
    cuts.append(b)
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        if c1 <= c0 + eps:
            continue
        mid = 0.5 * (c0 + c1)
        F_mid, _ = _envelope_difference(
            f_rec, b_rec, mid, with_derivative=False, profiler=profiler
        )
        if F_mid <= 0.0:
            _append_envelope_piece(
                out, PassKind.FORWARD, f_idx, f_rec, c0, c1, eps=eps
            )
        else:
            _append_envelope_piece(
                out, PassKind.BACKWARD, b_idx, b_rec, c0, c1, eps=eps
            )


def merge_lower_envelope(
    forward: ProdScalarPass,
    backward: ProdScalarPass,
    *,
    n_scan: int = 64,
    eps: float = 1e-12,
    profiler: PhaseProfiler | None = None,
) -> List[EnvelopePiece]:
    """Merge monotone forward/backward profiles into their lower envelope.

    Along absolute distance, every valid forward segment is nondecreasing and
    every backward segment is nonincreasing.  Therefore ``w_f - w_b`` is
    nondecreasing and each overlap has at most one crossing.  ``n_scan`` is
    retained only for a defensive numerical fallback.
    """
    f_refs = _sorted_segment_refs(forward)
    b_refs = _sorted_segment_refs(backward)
    out: List[EnvelopePiece] = []
    i = 0
    j = 0
    while i < len(f_refs) and j < len(b_refs):
        f_lo, f_hi, f_idx = f_refs[i]
        b_lo, b_hi, b_idx = b_refs[j]
        a = max(f_lo, b_lo)
        b = min(f_hi, b_hi)
        if a < b:
            f_rec = forward.segments[f_idx]
            b_rec = backward.segments[b_idx]
            F_a, _ = _envelope_difference(
                f_rec, b_rec, a, with_derivative=False, profiler=profiler
            )
            F_b, _ = _envelope_difference(
                f_rec, b_rec, b, with_derivative=False, profiler=profiler
            )
            if not _append_monotone_overlap(
                out,
                f_rec,
                b_rec,
                f_idx,
                b_idx,
                a,
                b,
                F_a,
                F_b,
                eps=eps,
                profiler=profiler,
            ):
                _append_overlap_uniform_fallback(
                    out,
                    f_rec,
                    b_rec,
                    f_idx,
                    b_idx,
                    a,
                    b,
                    n_scan=n_scan,
                    eps=eps,
                    profiler=profiler,
                )
        if f_hi <= b_hi + eps:
            i += 1
        if b_hi <= f_hi + eps:
            j += 1
    return out


def _candidate_difference(
    a_rec: ProdScalarSegment,
    b_rec: ProdScalarSegment,
    abs_s: float,
    *,
    with_derivative: bool,
    profiler: PhaseProfiler | None,
) -> tuple[float, float | None]:
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    try:
        value = segment_w_at_abs(a_rec, abs_s) - segment_w_at_abs(b_rec, abs_s)
        if not with_derivative:
            return value, None
        derivative = segment_abs_dw_ds(a_rec, abs_s) - segment_abs_dw_ds(b_rec, abs_s)
        return value, derivative
    finally:
        if profiler is not None and profiler.enabled:
            profiler.record(ENVELOPE_OVERLAP_EVAL, perf_counter() - started)


def _candidate_root_between(
    a_rec: ProdScalarSegment,
    b_rec: ProdScalarSegment,
    lo: float,
    hi: float,
    f_lo: float,
    f_hi: float,
    *,
    profiler: PhaseProfiler | None,
) -> float | None:
    def f_df(s: float) -> tuple[float, float | None]:
        return _candidate_difference(
            a_rec,
            b_rec,
            s,
            with_derivative=True,
            profiler=profiler,
        )

    return bracketed_newton(f_df, lo, hi, f_lo=f_lo, f_hi=f_hi)


def _append_candidate_envelope_piece(
    out: List[EnvelopePiece],
    pass_index: int,
    segment_index: int,
    rec: ProdScalarSegment,
    abs0: float,
    abs1: float,
    *,
    eps: float,
) -> None:
    if abs1 <= abs0 + eps:
        return
    local0 = abs_to_local(rec, abs0)
    local1 = abs_to_local(rec, abs1)
    if out:
        last = out[-1]
        if (
            last.pass_index == pass_index
            and last.source_index == segment_index
            and abs(last.abs1 - abs0) <= eps
        ):
            last.abs1 = abs1
            last.local1 = local1
            return
    out.append(
        EnvelopePiece(
            rec.kind,
            segment_index,
            abs0,
            abs1,
            local0,
            local1,
            pass_index,
        )
    )


def merge_candidate_envelope(
    passes: Sequence[ProdScalarPass],
    *,
    total_L: float,
    eps: float = 1e-12,
    profiler: PhaseProfiler | None = None,
) -> List[EnvelopePiece]:
    """Return the pointwise lower envelope of arbitrary candidate passes.

    Segment endpoints form atomic overlap intervals.  Within one such
    interval, scalar ODE uniqueness makes equal-direction candidates ordered,
    while an opposite-direction pair has at most one transverse crossing.
    Pairwise bracketed roots therefore partition the interval into regions
    with a fixed minimum source.
    """
    refs: list[tuple[float, float, int, int]] = []
    cuts = [0.0, float(total_L)]
    for pass_index, scalar_pass in enumerate(passes):
        for segment_index, rec in enumerate(scalar_pass.segments):
            lo, hi = interval_low_high(rec)
            if hi <= lo + eps:
                continue
            refs.append((lo, hi, pass_index, segment_index))
            cuts.extend((lo, hi))
    cuts.sort()
    unique_cuts: list[float] = []
    for value in cuts:
        value = min(max(value, 0.0), total_L)
        if not unique_cuts or value > unique_cuts[-1] + eps:
            unique_cuts.append(value)
        elif value > unique_cuts[-1]:
            unique_cuts[-1] = value

    out: List[EnvelopePiece] = []
    for base_lo, base_hi in zip(unique_cuts[:-1], unique_cuts[1:]):
        if base_hi <= base_lo + eps:
            continue
        mid = 0.5 * (base_lo + base_hi)
        active = [
            ref for ref in refs
            if ref[0] <= mid <= ref[1]
        ]
        if not active:
            raise FloatingPointError(
                "candidate speed profiles leave an uncovered interval: "
                f"[{base_lo:.17g}, {base_hi:.17g}]"
            )

        roots: list[float] = []
        for i in range(len(active)):
            _, _, pass_i, seg_i = active[i]
            rec_i = passes[pass_i].segments[seg_i]
            for j in range(i + 1, len(active)):
                _, _, pass_j, seg_j = active[j]
                rec_j = passes[pass_j].segments[seg_j]
                f_lo, _ = _candidate_difference(
                    rec_i, rec_j, base_lo,
                    with_derivative=False, profiler=profiler,
                )
                f_hi, _ = _candidate_difference(
                    rec_i, rec_j, base_hi,
                    with_derivative=False, profiler=profiler,
                )
                if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
                    continue
                if f_lo == 0.0:
                    root = base_lo
                elif f_hi == 0.0:
                    root = base_hi
                elif f_lo * f_hi < 0.0:
                    root = _candidate_root_between(
                        rec_i,
                        rec_j,
                        base_lo,
                        base_hi,
                        f_lo,
                        f_hi,
                        profiler=profiler,
                    )
                else:
                    root = None
                if root is not None and base_lo + eps < root < base_hi - eps:
                    roots.append(root)

        local_cuts = [base_lo]
        for root in sorted(roots):
            if root > local_cuts[-1] + eps:
                local_cuts.append(root)
        local_cuts.append(base_hi)

        for lo, hi in zip(local_cuts[:-1], local_cuts[1:]):
            if hi <= lo + eps:
                continue
            sample = 0.5 * (lo + hi)
            winner = min(
                active,
                key=lambda ref: segment_w_at_abs(
                    passes[ref[2]].segments[ref[3]], sample
                ),
            )
            _, _, pass_index, segment_index = winner
            rec = passes[pass_index].segments[segment_index]
            _append_candidate_envelope_piece(
                out,
                pass_index,
                segment_index,
                rec,
                lo,
                hi,
                eps=eps,
            )

    if not out:
        raise FloatingPointError("candidate speed profile envelope is empty")
    if out[0].abs0 > eps or out[-1].abs1 < total_L - eps:
        raise FloatingPointError(
            "candidate speed profiles do not cover the full path: "
            f"coverage=[{out[0].abs0:.17g}, {out[-1].abs1:.17g}], "
            f"total={total_L:.17g}"
        )
    for left, right in zip(out[:-1], out[1:]):
        if right.abs0 > left.abs1 + eps:
            raise FloatingPointError(
                "candidate speed profile envelope contains a gap: "
                f"{left.abs1:.17g} -> {right.abs0:.17g}"
            )
    return out


# =============================================================================
# High-level scalar build
# =============================================================================

def _collect_internal_cap_anchors_validated(
    raw: Sequence[float],
    *,
    initial_k: float,
    domain_margin: float,
) -> List[InternalCapAnchor]:
    """Collect cap anchors from an already validated raw parameter array."""
    n = len(raw) // 2
    anchor_g2 = max(
        INTERNAL_CAP_G2_MARGIN,
        64.0 * domain_margin,
        512.0 * math.ulp(MU_G2),
    )
    cap_numerator = math.sqrt(max(0.0, MU_G2 - anchor_g2))

    anchors: List[InternalCapAnchor] = []
    station = 0.0
    signed_k = float(initial_k)
    if not math.isfinite(signed_k):
        raise ValueError("initial_k must be finite")
    for piece_index in range(n):
        L = float(raw[2 * piece_index])
        sigma = float(raw[2 * piece_index + 1])
        station += L
        signed_k = math.fma(sigma, L, signed_k)
        knot_index = piece_index + 1
        if knot_index == n:
            break
        right_sigma = float(raw[2 * knot_index + 1])
        releases_backward = sigma * signed_k > 0.0
        releases_forward = right_sigma * signed_k < 0.0
        if not (releases_backward or releases_forward):
            # This knot is a local minimum of |k| (a local maximum of the
            # lateral speed cap), or a flat non-releasing point.  No exact
            # cap extremal can propagate away from it in either direction.
            continue
        k_abs = abs(signed_k)
        if k_abs == 0.0:
            continue
        cap_w = cap_numerator / k_abs
        if cap_w >= W_EQ:
            continue
        anchors.append(
            InternalCapAnchor(
                knot_index=knot_index,
                station=station,
                signed_k=signed_k,
                cap_w=cap_w,
                initial_w_dk=-cap_w / signed_k,
            )
        )
    return anchors


def collect_internal_cap_anchors(
    raw_params: Sequence[float],
    *,
    initial_k: float,
    domain_margin: float,
) -> List[InternalCapAnchor]:
    """Return every internal knot whose lateral cap lies below motor equilibrium."""
    raw = validate_raw_parameters(raw_params)
    return _collect_internal_cap_anchors_validated(
        raw, initial_k=initial_k, domain_margin=domain_margin
    )


def compile_internal_cap_anchor_passes(
    forward_pieces: Sequence[ProdTraversalPiece],
    backward_pieces: Sequence[ProdTraversalPiece],
    anchor: InternalCapAnchor,
    *,
    n_scan: int,
    domain_margin: float,
    domain_scan: int,
    profiler: PhaseProfiler | None,
    fused_grip_discovery: bool = True,
) -> List[ProdScalarPass]:
    """Compile both extremals launched from one internal cap anchor.

    The full forward/backward traversals are built once by the top-level
    builder.  Slicing them here avoids reconstructing station and piece metadata
    for every inserted anchor.
    """
    n = len(forward_pieces)
    knot_index = anchor.knot_index
    if not 0 <= knot_index <= n:
        raise ValueError(f"knot_index={knot_index} outside [0, {n}]")

    out: List[ProdScalarPass] = []
    forward_anchor_pieces = forward_pieces[knot_index:]
    if (
        forward_anchor_pieces
        and _cap_releases_into_piece(
            forward_anchor_pieces[0], anchor.signed_k
        )
    ):
        out.append(
            build_scalar_pass(
                forward_anchor_pieces,
                anchor.cap_w,
                anchor.signed_k,
                PassKind.FORWARD,
                n_scan=n_scan,
                domain_margin=domain_margin,
                domain_scan=domain_scan,
                profiler=profiler,
                initial_knot_index=knot_index,
                initial_w_dk=anchor.initial_w_dk,
                fused_grip_discovery=fused_grip_discovery,
            )
        )

    # backward_pieces are ordered [n-1, ..., 0].  A pass beginning at knot j
    # traverses [j-1, ..., 0], which starts at offset n-j in this array.
    backward_anchor_pieces = backward_pieces[n - knot_index :]
    if (
        backward_anchor_pieces
        and _cap_releases_into_piece(
            backward_anchor_pieces[0], anchor.signed_k
        )
    ):
        out.append(
            build_scalar_pass(
                backward_anchor_pieces,
                anchor.cap_w,
                anchor.signed_k,
                PassKind.BACKWARD,
                n_scan=n_scan,
                domain_margin=domain_margin,
                domain_scan=domain_scan,
                profiler=profiler,
                initial_knot_index=knot_index,
                initial_w_dk=anchor.initial_w_dk,
                fused_grip_discovery=fused_grip_discovery,
            )
        )
    return out


def candidate_coverage_witnesses(
    passes: Sequence[ProdScalarPass],
    *,
    total_L: float,
    eps: float = 1e-11,
    profiler: PhaseProfiler | None = None,
) -> List[AnchorWitness]:
    """Return disconnected path intervals not covered by any candidate pass."""
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    try:
        intervals: List[Tuple[float, float]] = []
        for scalar_pass in passes:
            for record in scalar_pass.segments:
                lo, hi = interval_low_high(record)
                lo = max(0.0, lo)
                hi = min(total_L, hi)
                if hi > lo + eps:
                    intervals.append((lo, hi))
        intervals.sort()
        if not intervals:
            return [AnchorWitness(0.0, total_L, "coverage")]

        merged: List[List[float]] = []
        for lo, hi in intervals:
            if not merged or lo > merged[-1][1] + eps:
                merged.append([lo, hi])
            elif hi > merged[-1][1]:
                merged[-1][1] = hi

        witnesses: List[AnchorWitness] = []
        cursor = 0.0
        for lo, hi in merged:
            if lo > cursor + eps:
                witnesses.append(AnchorWitness(cursor, lo, "coverage"))
            cursor = max(cursor, hi)
        if cursor < total_L - eps:
            witnesses.append(AnchorWitness(cursor, total_L, "coverage"))
        return witnesses
    finally:
        if profiler is not None and profiler.enabled:
            profiler.record(ANCHOR_WITNESS_SCAN, perf_counter() - started)


def envelope_continuity_witnesses(
    envelope: Sequence[EnvelopePiece],
    passes: Sequence[ProdScalarPass],
    *,
    total_L: float,
    abs_tol: float = 2e-9,
    rel_tol: float = 2e-10,
    profiler: PhaseProfiler | None = None,
) -> List[AnchorWitness]:
    """Return discontinuities in a fully covered candidate envelope.

    A speed profile may switch derivatives, but its state ``w`` must remain
    continuous.  The previous exhaustive all-anchor merge did not enforce this
    invariant and could splice a pass at its domain endpoint to another pass at
    a different speed.  Such a splice is not a physical trajectory.
    """
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    try:
        witnesses: List[AnchorWitness] = []
        scale_s = max(1.0, total_L)
        for left, right in zip(envelope[:-1], envelope[1:]):
            if right.abs0 > left.abs1 + 1e-11 * scale_s:
                witnesses.append(
                    AnchorWitness(left.abs1, right.abs0, "coverage")
                )
                continue
            station = 0.5 * (left.abs1 + right.abs0)
            left_record = passes[left.pass_index].segments[left.source_index]
            right_record = passes[right.pass_index].segments[right.source_index]
            w_left = segment_w_at_abs(left_record, left.abs1)
            w_right = segment_w_at_abs(right_record, right.abs0)
            jump = abs(w_left - w_right)
            tol = abs_tol + rel_tol * max(1.0, abs(w_left), abs(w_right))
            if jump > tol:
                witnesses.append(
                    AnchorWitness(station, station, "continuity", jump)
                )
        return witnesses
    finally:
        if profiler is not None and profiler.enabled:
            profiler.record(ANCHOR_WITNESS_SCAN, perf_counter() - started)


def require_envelope_boundary_states(
    envelope: Sequence[EnvelopePiece],
    passes: Sequence[ProdScalarPass],
    *,
    total_L: float,
    start_w: float,
    end_w: float,
    abs_tol: float = 2e-9,
    rel_tol: float = 2e-10,
) -> None:
    """Require the fixed start speed and the terminal maximum-speed cap.

    ``end_w`` is an upper bound, not a fixed terminal state.  The backward
    extremal starts at that cap, while the merged envelope may legitimately
    finish below it when the forward dynamics or an internal grip cap are more
    restrictive.
    """
    if not envelope:
        raise FloatingPointError("speed-profile envelope is empty")

    first = envelope[0]
    last = envelope[-1]
    first_record = passes[first.pass_index].segments[first.source_index]
    last_record = passes[last.pass_index].segments[last.source_index]
    actual_start = segment_w_at_abs(first_record, 0.0)
    actual_end = segment_w_at_abs(last_record, total_L)

    start_tol = abs_tol + rel_tol * max(1.0, abs(actual_start), abs(start_w))
    if abs(actual_start - start_w) > start_tol:
        raise FloatingPointError(
            "speed-profile envelope violates fixed start speed: "
            f"actual={actual_start:.17g}, expected={start_w:.17g}, tol={start_tol:.17g}"
        )
    end_tol = abs_tol + rel_tol * max(1.0, abs(actual_end), abs(end_w))
    if actual_end > end_w + end_tol:
        raise FloatingPointError(
            "speed-profile envelope violates terminal speed cap: "
            f"actual={actual_end:.17g}, maximum={end_w:.17g}, tol={end_tol:.17g}"
        )


def _anchor_distance_to_witness(
    anchor: InternalCapAnchor,
    witness: AnchorWitness,
) -> float:
    if witness.abs0 <= anchor.station <= witness.abs1:
        return 0.0
    return min(
        abs(anchor.station - witness.abs0),
        abs(anchor.station - witness.abs1),
    )


def select_anchor_batch(
    witnesses: Sequence[AnchorWitness],
    inactive: Sequence[InternalCapAnchor],
) -> List[InternalCapAnchor]:
    """Select one most restrictive inactive anchor per disconnected witness.

    Selection order affects work only.  Closure is certified independently by
    full-path coverage, continuity, and fixed-boundary validation.
    """
    selected: List[InternalCapAnchor] = []
    selected_indices: set[int] = set()
    for witness in witnesses:
        inside = [
            anchor
            for anchor in inactive
            if witness.abs0 <= anchor.station <= witness.abs1
        ]
        if inside:
            candidate = min(inside, key=lambda a: (a.cap_w, a.knot_index))
        elif inactive:
            candidate = min(
                inactive,
                key=lambda a: (
                    _anchor_distance_to_witness(a, witness),
                    a.cap_w,
                    a.knot_index,
                ),
            )
        else:
            continue
        if candidate.knot_index not in selected_indices:
            selected.append(candidate)
            selected_indices.add(candidate.knot_index)
    return selected


def build_internal_cap_candidates(
    raw: Sequence[float],
    endpoint_passes: Sequence[ProdScalarPass],
    forward_pieces: Sequence[ProdTraversalPiece],
    backward_pieces: Sequence[ProdTraversalPiece],
    *,
    initial_k: float,
    n_scan: int,
    domain_margin: float,
    domain_scan: int,
    profiler: PhaseProfiler | None,
    fused_grip_discovery: bool = True,
) -> Tuple[List[ProdScalarPass], List[EnvelopePiece], AnchorInsertionStats]:
    """Build the exact multi-bottleneck profile by batch cap insertion."""
    if len(endpoint_passes) != 2:
        raise ValueError("exactly two endpoint passes are required")
    if not endpoint_passes[0].segments or not endpoint_passes[1].segments:
        raise FloatingPointError("endpoint extremals emitted no usable segment")
    if endpoint_passes[0].kind is not PassKind.FORWARD:
        raise ValueError("first endpoint pass must be forward")
    if endpoint_passes[1].kind is not PassKind.BACKWARD:
        raise ValueError("second endpoint pass must be backward")
    if len(forward_pieces) != len(backward_pieces):
        raise ValueError("forward/backward traversal sizes differ")

    total_L = math.fsum(piece.L for piece in forward_pieces)
    anchors = _collect_internal_cap_anchors_validated(
        raw, initial_k=initial_k, domain_margin=domain_margin
    )
    stats = AnchorInsertionStats(len(anchors))
    passes = list(endpoint_passes)
    active_indices: set[int] = set()

    while True:
        coverage = candidate_coverage_witnesses(
            passes, total_L=total_L, profiler=profiler
        )
        stats.coverage_witnesses += len(coverage)
        if coverage:
            witnesses = coverage
        else:
            envelope = merge_candidate_envelope(
                passes, total_L=total_L, profiler=profiler
            )
            continuity = envelope_continuity_witnesses(
                envelope,
                passes,
                total_L=total_L,
                profiler=profiler,
            )
            stats.continuity_witnesses += len(continuity)
            if not continuity:
                require_envelope_boundary_states(
                    envelope,
                    passes,
                    total_L=total_L,
                    start_w=endpoint_passes[0].segments[0].w0,
                    end_w=endpoint_passes[1].segments[0].w0,
                )
                return passes, envelope, stats
            witnesses = continuity

        inactive = [
            anchor for anchor in anchors if anchor.knot_index not in active_indices
        ]
        if not inactive:
            worst = max(witnesses, key=lambda item: item.magnitude)
            raise FloatingPointError(
                "batch internal-cap insertion exhausted all anchors without a "
                f"continuous profile: kind={worst.kind}, "
                f"interval=[{worst.abs0:.17g}, {worst.abs1:.17g}], "
                f"magnitude={worst.magnitude:.17g}"
            )

        selected = select_anchor_batch(witnesses, inactive)
        if not selected:
            raise RuntimeError(
                "anchor witness selection made no progress despite inactive anchors"
            )

        stats.rounds += 1
        if stats.rounds > len(anchors):
            raise RuntimeError(
                "anchor insertion exceeded the finite anchor catalog size"
            )

        inserted_this_round = 0
        for anchor in selected:
            if anchor.knot_index in active_indices:
                continue
            active_indices.add(anchor.knot_index)
            stats.inserted_anchor_indices.append(anchor.knot_index)
            inserted_this_round += 1
            passes.extend(
                compile_internal_cap_anchor_passes(
                    forward_pieces,
                    backward_pieces,
                    anchor,
                    n_scan=n_scan,
                    domain_margin=domain_margin,
                    domain_scan=domain_scan,
                    profiler=profiler,
                    fused_grip_discovery=fused_grip_discovery,
                )
            )
        if inserted_this_round == 0:
            raise RuntimeError("anchor insertion round made no progress")


def _python_build_scalar_speed_profile(
    raw_params: Sequence[float],
    *,
    init_w: Optional[float] = None,
    terminal_w_max: Optional[float] = None,
    forward_init_k: float = 0.0,
    backward_init_k: Optional[float] = None,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
    fused_grip_discovery: bool = True,
) -> ProdSpeedProfileBuild:
    raw = validate_raw_parameters(raw_params)
    del envelope_scan  # retained for API compatibility with the two-pass build
    if init_w is None:
        init_w = max(1e-3, 0.05 * W_EQ)
    if terminal_w_max is None:
        terminal_w_max = init_w
    if not math.isfinite(terminal_w_max) or not 0.0 < terminal_w_max <= W_EQ:
        raise ValueError("terminal_w_max must lie in (0, V_MAX^2]")
    if backward_init_k is None:
        backward_init_k = geometry_endpoint_k(raw, initial_k=forward_init_k)

    forward_pieces = make_forward_traversal(raw)
    backward_pieces = make_backward_traversal_flip_sigma(raw)
    forward_scalar = build_scalar_pass(
        forward_pieces,
        init_w,
        forward_init_k,
        PassKind.FORWARD,
        n_scan=n_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
        initial_knot_index=None,
        fused_grip_discovery=fused_grip_discovery,
    )
    backward_scalar = build_scalar_pass(
        backward_pieces,
        terminal_w_max,
        backward_init_k,
        PassKind.BACKWARD,
        n_scan=n_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
        initial_knot_index=len(raw) // 2,
        fused_grip_discovery=fused_grip_discovery,
    )
    scalar_passes, envelope, anchor_stats = build_internal_cap_candidates(
        raw,
        [forward_scalar, backward_scalar],
        forward_pieces,
        backward_pieces,
        initial_k=forward_init_k,
        n_scan=n_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
        fused_grip_discovery=fused_grip_discovery,
    )
    return ProdSpeedProfileBuild(
        raw_params=raw,
        forward_pieces=forward_pieces,
        backward_pieces=backward_pieces,
        forward_scalar=forward_scalar,
        backward_scalar=backward_scalar,
        scalar_passes=scalar_passes,
        envelope=envelope,
        anchor_stats=anchor_stats,
    )


def _python_time_value_and_gradient_from_build(
    build: ProdSpeedProfileBuild,
    *,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
) -> tuple[float, list[float]]:
    """Upgrade an authoritative scalar topology build to value+gradient.

    The scalar pass, switching locations, envelope, and retained scalar Cflow
    segments are reused exactly.  Only differentiable replay and the reverse
    sweep are constructed.
    """
    rev = prepare_reverse_time_build(
        build,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
    )
    return reverse_time_objective(build, rev=rev)


def _python_time_value_and_gradient(
    raw_params: Sequence[float],
    *,
    init_w: float | None = None,
    terminal_w_max: float | None = None,
    initial_k: float = 0.0,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
    fused_grip_discovery: bool = True,
) -> tuple[float, list[float]]:
    """Evaluate travel time and its flat raw gradient in one call."""
    build = _python_build_scalar_speed_profile(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        forward_init_k=initial_k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
        fused_grip_discovery=fused_grip_discovery,
    )
    return _python_time_value_and_gradient_from_build(
        build, domain_margin=domain_margin, domain_scan=domain_scan, profiler=profiler
    )


@dataclass(slots=True)
class NativeProdSpeedProfileBuild:
    """Compatibility facade for an authoritative native scalar build.

    The facade intentionally exposes only metadata that can be represented
    without reconstructing Python segment objects.  Production callers that
    only need scalar time and subsequent differentiable promotion remain fully
    native end-to-end.
    """
    raw_params: list[float]
    native: Any
    init_w: float
    terminal_w_max: float
    forward_init_k: float
    backward_init_k: float | None
    n_scan: int
    domain_scan: int
    domain_margin: float
    fused_grip_discovery: bool
    _anchor_stats: AnchorInsertionStats | None = None

    @property
    def is_native(self) -> bool:
        return True

    @property
    def anchor_stats(self) -> AnchorInsertionStats:
        cached = self._anchor_stats
        if cached is None:
            stats = self.native.stats()
            cached = AnchorInsertionStats(
                possible_anchors=int(stats["possible_anchors"]),
                inserted_anchor_indices=self.native.inserted_anchor_indices(),
                rounds=int(stats["anchor_rounds"]),
            )
            self._anchor_stats = cached
        return cached

    @property
    def closed(self) -> bool:
        return self.native.closed

    def scalar_time_value(self, *, profiler: PhaseProfiler | None = None) -> float:
        started = perf_counter()
        try:
            value = float(self.native.time_value())
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        if profiler is not None and profiler.enabled:
            # Scalar time is one native envelope operation; detailed interval
            # accounting is intentionally not re-materialized across the ABI.
            profiler.record("scalar_time_integration", perf_counter() - started, count=max(1, self.envelope_count))
        return value

    def time_value_and_gradient_native(
        self, *, profiler: PhaseProfiler | None = None
    ) -> tuple[float, list[float]]:
        if profiler is None or not profiler.enabled:
            try:
                value, gradient = self.native.time_value_and_gradient()
                return float(value), [float(v) for v in gradient]
            except Exception as exc:
                raise _translate_native_error(exc) from exc
        started = perf_counter()
        try:
            rev = self.native.promote_time()
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        profiler.record(NATIVE_REVERSE_PROMOTION, perf_counter() - started)
        try:
            started = perf_counter()
            value, gradient = rev.time_value_and_gradient()
            profiler.record(NATIVE_REVERSE_SWEEP, perf_counter() - started)
            return float(value), [float(v) for v in gradient]
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        finally:
            rev.close()

    def close(self) -> None:
        self.native.close()

    def native_stats(self) -> dict[str, int]:
        return self.native.stats()

    @property
    def envelope_count(self) -> int:
        return self.native.envelope_count

    @property
    def scalar_pass_count(self) -> int:
        return self.native.pass_count

    @property
    def scalar_segment_count(self) -> int:
        return self.native.segment_count


@dataclass(slots=True)
class NativeRevBuild:
    build: NativeProdSpeedProfileBuild
    native: Any
    full: bool
    profiler: PhaseProfiler | None = None

    def close(self) -> None:
        self.native.close()


def _translate_native_error(exc: Exception) -> Exception:
    # Preserve Python-facing exception families closely enough for optimizer
    # lifecycle semantics while retaining the native diagnostic in the message.
    status = getattr(exc, "status", None)
    if status == 1:
        return ValueError(str(exc))
    if status in {2, 3, 4, 5}:
        return ValueError(str(exc))
    if status == 6:
        return MemoryError(str(exc))
    if status == 7:
        return NotImplementedError(str(exc))
    return RuntimeError(str(exc))


def _build_scalar_speed_profile_native(
    raw_params: Sequence[float],
    *,
    init_w: Optional[float] = None,
    terminal_w_max: Optional[float] = None,
    forward_init_k: float = 0.0,
    backward_init_k: Optional[float] = None,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
    fused_grip_discovery: bool = True,
) -> NativeProdSpeedProfileBuild:
    raw = validate_raw_parameters(raw_params)
    del envelope_scan
    if init_w is None:
        init_w = max(1e-3, 0.05 * W_EQ)
    if terminal_w_max is None:
        terminal_w_max = init_w
    if not math.isfinite(terminal_w_max) or not 0.0 < terminal_w_max <= W_EQ:
        raise ValueError("terminal_w_max must lie in (0, V_MAX^2]")
    native = _native_reverse_module()
    started = perf_counter()
    try:
        handle = native.build_scalar_native(
            raw,
            init_w=float(init_w),
            terminal_w_max=float(terminal_w_max),
            initial_k=float(forward_init_k),
            backward_init_k=(None if backward_init_k is None else float(backward_init_k)),
            n_scan=int(n_scan),
            domain_scan=int(domain_scan),
            domain_margin=float(domain_margin),
            fused_grip_discovery=bool(fused_grip_discovery),
        )
    except Exception as exc:
        raise _translate_native_error(exc) from exc
    if profiler is not None and profiler.enabled:
        profiler.record(NATIVE_SCALAR_BUILD, perf_counter() - started)
    return NativeProdSpeedProfileBuild(
        raw_params=raw,
        native=handle,
        init_w=float(init_w),
        terminal_w_max=float(terminal_w_max),
        forward_init_k=float(forward_init_k),
        backward_init_k=(None if backward_init_k is None else float(backward_init_k)),
        n_scan=int(n_scan),
        domain_scan=int(domain_scan),
        domain_margin=float(domain_margin),
        fused_grip_discovery=bool(fused_grip_discovery),
    )


def native_internal_cap_count(
    raw_params: Sequence[float],
    *,
    init_w: Optional[float] = None,
    terminal_w_max: Optional[float] = None,
    forward_init_k: float = 0.0,
    backward_init_k: Optional[float] = None,
    n_scan: int = 256,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    fused_grip_discovery: bool = True,
) -> int:
    """Return the native internal-cap anchor count with deterministic lifetime.

    This is the planner-facing metadata fast path.  It intentionally does not
    construct a Python compatibility build or materialize the anchor list.
    """
    build = _build_scalar_speed_profile_native(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        forward_init_k=forward_init_k,
        backward_init_k=backward_init_k,
        n_scan=n_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        fused_grip_discovery=fused_grip_discovery,
    )
    try:
        return int(build.native.inserted_anchor_count)
    finally:
        build.close()


def build_scalar_speed_profile(
    raw_params: Sequence[float],
    *,
    init_w: Optional[float] = None,
    terminal_w_max: Optional[float] = None,
    forward_init_k: float = 0.0,
    backward_init_k: Optional[float] = None,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
    fused_grip_discovery: bool = True,
):
    if reverse_backend() == "native":
        return _build_scalar_speed_profile_native(
            raw_params,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            forward_init_k=forward_init_k,
            backward_init_k=backward_init_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_margin=domain_margin,
            domain_scan=domain_scan,
            profiler=profiler,
            fused_grip_discovery=fused_grip_discovery,
        )
    return _python_build_scalar_speed_profile(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        forward_init_k=forward_init_k,
        backward_init_k=backward_init_k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
        fused_grip_discovery=fused_grip_discovery,
    )


def time_value_and_gradient_from_build(
    build,
    *,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
) -> tuple[float, list[float]]:
    if isinstance(build, NativeProdSpeedProfileBuild):
        if int(domain_scan) != build.domain_scan or float(domain_margin) != build.domain_margin:
            raise ValueError("native build promotion options must match scalar-build domain settings")
        if profiler is None or not profiler.enabled:
            try:
                value, gradient = build.native.time_value_and_gradient()
                return float(value), [float(v) for v in gradient]
            except Exception as exc:
                raise _translate_native_error(exc) from exc
        started = perf_counter()
        try:
            rev = build.native.promote_time()
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        if profiler is not None and profiler.enabled:
            profiler.record(NATIVE_REVERSE_PROMOTION, perf_counter() - started)
        try:
            started = perf_counter()
            value, gradient = rev.time_value_and_gradient()
            if profiler is not None and profiler.enabled:
                profiler.record(NATIVE_REVERSE_SWEEP, perf_counter() - started)
            return float(value), [float(v) for v in gradient]
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        finally:
            rev.close()
    return _python_time_value_and_gradient_from_build(
        build,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
    )


def time_value_and_gradient(
    raw_params: Sequence[float],
    *,
    init_w: float | None = None,
    terminal_w_max: float | None = None,
    initial_k: float = 0.0,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    profiler: PhaseProfiler | None = None,
    fused_grip_discovery: bool = True,
) -> tuple[float, list[float]]:
    if reverse_backend() == "native":
        # The common one-shot API can remain entirely within one native call
        # when no phase profiler is requested.  Staged construction is retained
        # for profiler runs and for callers that explicitly reuse scalar builds.
        if profiler is None or not profiler.enabled:
            raw = validate_raw_parameters(raw_params)
            if init_w is None:
                init_w = max(1e-3, 0.05 * W_EQ)
            if terminal_w_max is None:
                terminal_w_max = init_w
            native = _native_reverse_module()
            try:
                value, gradient = native.time_value_and_gradient_native(
                    raw,
                    init_w=float(init_w),
                    terminal_w_max=float(terminal_w_max),
                    initial_k=float(initial_k),
                    n_scan=int(n_scan),
                    domain_scan=int(domain_scan),
                    domain_margin=float(domain_margin),
                    fused_grip_discovery=bool(fused_grip_discovery),
                )
                return float(value), [float(v) for v in gradient]
            except Exception as exc:
                raise _translate_native_error(exc) from exc
        build = _build_scalar_speed_profile_native(
            raw_params,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            forward_init_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_margin=domain_margin,
            domain_scan=domain_scan,
            profiler=profiler,
            fused_grip_discovery=fused_grip_discovery,
        )
        try:
            return time_value_and_gradient_from_build(
                build,
                domain_margin=domain_margin,
                domain_scan=domain_scan,
                profiler=profiler,
            )
        finally:
            build.close()
    return _python_time_value_and_gradient(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        initial_k=initial_k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
        fused_grip_discovery=fused_grip_discovery,
    )


# =============================================================================
# Reverse tape and sweep
# =============================================================================

@dataclass(slots=True)
class RevSegment:
    scalar: ProdScalarSegment
    seg: Any
    F_cols: Optional[Tuple[float, float, float, float]]
    F_L: Optional[float]


@dataclass(slots=True)
class RevPass:
    scalar_pass: ProdScalarPass
    segments: List[RevSegment]
    n_pieces: int


@dataclass(slots=True)
class RevBuild:
    build: ProdSpeedProfileBuild
    forward: RevPass
    backward: RevPass
    passes: List[RevPass]
    profiler: PhaseProfiler | None = None


@dataclass(slots=True)
class SegmentExtraAdj:
    # Local primitive inputs: [L_used, sigma, w0, k0]
    aL: float = 0.0
    aw0: float = 0.0
    ak0: float = 0.0
    asigma: float = 0.0
    # scalar.abs0 = traversal_abs0 + direction*offset0
    aabs0: float = 0.0


def make_empty_extras(n_segments: int) -> List[SegmentExtraAdj]:
    return [SegmentExtraAdj() for _ in range(n_segments)]


def prepare_reverse_pass(
    scalar_pass: ProdScalarPass,
    *,
    n_pieces: int,
    denom_tol: float = 1e-14,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    validate_replay_domain: bool = False,
    profiler: PhaseProfiler | None = None,
    segment_limit: int | None = None,
    reverse_eta: bool = False,
) -> RevPass:
    if segment_limit is None:
        replay_scalars = scalar_pass.segments
    else:
        if not 0 <= segment_limit <= len(scalar_pass.segments):
            raise ValueError(
                f"segment_limit={segment_limit} outside [0, {len(scalar_pass.segments)}]"
            )
        replay_scalars = scalar_pass.segments[:segment_limit]

    nodes: List[RevSegment] = []
    for scalar in replay_scalars:
        seg = compile_diff(
            scalar.mode,
            scalar.L_used,
            scalar.sigma,
            scalar.w0,
            scalar.k0,
            profiler=profiler,
            boundary_start=scalar.boundary_start,
            reverse_eta=reverse_eta,
            authoritative_w1=scalar.w1,
        )
        if validate_replay_domain:
            require_segment_friction_domain(
                seg,
                scalar.L_used,
                where=f"reverse replay {scalar.kind}/{mode_name(scalar.mode)}/{scalar.event}",
                margin=domain_margin,
                n_scan=domain_scan,
            )

        w1, jac = seg.w_and_jac(scalar.L_used)
        k1 = math.fma(seg.sigma, scalar.L_used, seg.k0)
        if abs(w1 - scalar.w1) > 1e-8 or abs(k1 - scalar.k1) > 1e-8:
            raise AssertionError(
                f"reverse replay endpoint mismatch: got={(w1, k1)}, "
                f"expected={(scalar.w1, scalar.k1)}"
            )
        if len(jac) != 8:
            raise ValueError(f"bad endpoint state jacobian: {jac}")

        if scalar.event is EventKind.PIECE_END:
            F_cols = None
            F_L = None
        else:
            cols = local_event_columns(
                seg,
                scalar.L_used,
                scalar.event,
                w_and_jac=(w1, jac),
            )
            if len(cols) != 4 or not all(math.isfinite(c) for c in cols):
                raise FloatingPointError(f"bad event columns: {scalar.event}, {cols}")
            if abs(cols[0]) <= denom_tol:
                raise FloatingPointError(
                    f"singular event root: {scalar.event}, F_L={cols[0]}"
                )
            F_cols = tuple(float(c) for c in cols)
            F_L = float(cols[0])

        nodes.append(
            RevSegment(
                scalar=scalar,
                seg=seg,
                F_cols=F_cols,
                F_L=F_L,
            )
        )
    return RevPass(scalar_pass, nodes, n_pieces)


def prepare_reverse_build(
    build: ProdSpeedProfileBuild,
    *,
    denom_tol: float = 1e-14,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    validate_replay_domain: bool = False,
    profiler: PhaseProfiler | None = None,
) -> RevBuild:
    if isinstance(build, NativeProdSpeedProfileBuild):
        if validate_replay_domain:
            raise NotImplementedError("native production path does not expose validation-only full-promotion scans")
        started = perf_counter()
        try:
            native_rev = build.native.promote_full()
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        if profiler is not None and profiler.enabled:
            profiler.record(NATIVE_REVERSE_PROMOTION, perf_counter() - started)
        return NativeRevBuild(build, native_rev, True, profiler)  # type: ignore[return-value]
    n = segment_count(build.raw_params)
    passes = [
        prepare_reverse_pass(
            scalar_pass,
            n_pieces=n,
            denom_tol=denom_tol,
            domain_margin=domain_margin,
            domain_scan=domain_scan,
            validate_replay_domain=validate_replay_domain,
            profiler=profiler,
        )
        for scalar_pass in build.scalar_passes
    ]
    return RevBuild(
        build,
        passes[0],
        passes[1],
        passes,
        profiler,
    )


def _time_replay_limits(build: ProdSpeedProfileBuild) -> List[int]:
    limits = [0] * len(build.scalar_passes)
    for piece in build.envelope:
        if not 0 <= piece.pass_index < len(limits):
            raise ValueError(f"bad envelope pass index: {piece.pass_index}")
        limits[piece.pass_index] = max(
            limits[piece.pass_index], piece.source_index + 1
        )
    return limits


def prepare_reverse_time_build(
    build: ProdSpeedProfileBuild,
    *,
    denom_tol: float = 1e-14,
    domain_margin: float = FRICTION_DOMAIN_MARGIN,
    domain_scan: int = FRICTION_DOMAIN_SCAN,
    validate_replay_domain: bool = False,
    profiler: PhaseProfiler | None = None,
) -> RevBuild:
    if isinstance(build, NativeProdSpeedProfileBuild):
        if validate_replay_domain:
            raise NotImplementedError("native production path does not expose validation-only time-promotion scans")
        started = perf_counter()
        try:
            native_rev = build.native.promote_time()
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        if profiler is not None and profiler.enabled:
            profiler.record(NATIVE_REVERSE_PROMOTION, perf_counter() - started)
        return NativeRevBuild(build, native_rev, False, profiler)  # type: ignore[return-value]
    """Compile only traversal prefixes that can influence the time objective.

    An envelope seed on segment ``j`` depends only on replay segments
    ``0..j``.  Suffix segments after the last selected envelope source have
    zero adjoints and are omitted.  The returned build is valid for
    :func:`reverse_time_objective`, but not for final-state objectives.
    """
    n = segment_count(build.raw_params)
    limits = _time_replay_limits(build)
    passes = [
        prepare_reverse_pass(
            scalar_pass,
            n_pieces=n,
            denom_tol=denom_tol,
            domain_margin=domain_margin,
            domain_scan=domain_scan,
            validate_replay_domain=validate_replay_domain,
            profiler=profiler,
            segment_limit=limit,
            reverse_eta=True,
        )
        for scalar_pass, limit in zip(build.scalar_passes, limits)
    ]
    return RevBuild(
        build,
        passes[0],
        passes[1],
        passes,
        profiler,
    )


def _require_complete_reverse_pass(rev_pass: RevPass, *, objective: str) -> None:
    expected = len(rev_pass.scalar_pass.segments)
    actual = len(rev_pass.segments)
    if actual != expected:
        raise ValueError(
            f"{objective} requires a complete reverse replay; got {actual} of "
            f"{expected} segments. Use prepare_reverse_build(), not "
            "prepare_reverse_time_build()."
        )


def reverse_scalar_pass(
    rev_pass: RevPass,
    *,
    seed_final: Tuple[float, float] = (0.0, 0.0),
    extras: Optional[Sequence[SegmentExtraAdj]] = None,
    profiler: PhaseProfiler | None = None,
) -> Tuple[List[float], Tuple[float, float]]:
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    try:
        acc = RawGradientAccumulator.create(rev_pass.n_pieces)
        if extras is None:
            extras = make_empty_extras(len(rev_pass.segments))
        if len(extras) != len(rev_pass.segments):
            raise ValueError("extras length must match reverse segment count")

        aw, ak = seed_final
        current_traversal_index: Optional[int] = None
        adj_offset_after = 0.0

        for idx in range(len(rev_pass.segments) - 1, -1, -1):
            node = rev_pass.segments[idx]
            scalar = node.scalar
            extra = extras[idx]

            if current_traversal_index != scalar.traversal_index:
                current_traversal_index = scalar.traversal_index
                adj_offset_after = 0.0

            a_w0 = extra.aw0
            a_k0 = extra.ak0
            a_sigma = extra.asigma

            if extra.aabs0 != 0.0:
                acc.add_abs0(scalar, extra.aabs0)
            adj_offset_before = scalar.direction * extra.aabs0

            # Query the retained differentiable segment at the actual emitted
            # endpoint.  No endpoint Jacobian cache or replay compilation is
            # needed beyond the segment object itself.
            _w1, jac = node.seg.w_and_jac(scalar.L_used)
            if len(jac) != 8:
                raise ValueError(f"bad state jacobian during reverse sweep: {jac}")
            a0 = math.fma(aw, jac[0], ak * jac[4])
            a1 = math.fma(aw, jac[1], ak * jac[5])
            a2 = math.fma(aw, jac[2], ak * jac[6])
            a3 = math.fma(aw, jac[3], ak * jac[7])

            # offset1 = offset0 + L_used
            adj_L_used = extra.aL + a0 + adj_offset_after
            adj_offset_before += adj_offset_after

            if scalar.event is EventKind.PIECE_END:
                # L_used = raw L - offset0
                acc.add_length(scalar.piece_index, adj_L_used)
                adj_offset_before -= adj_L_used
                a_sigma += a1
                a_w0 += a2
                a_k0 += a3
            else:
                if node.F_cols is None or node.F_L is None:
                    raise AssertionError("event segment missing event columns")
                scale = adj_L_used / node.F_L
                a_sigma += a1 - scale * node.F_cols[1]
                a_w0 += a2 - scale * node.F_cols[2]
                a_k0 += a3 - scale * node.F_cols[3]

            acc.add_traversal_sigma(scalar, a_sigma)
            aw, ak = a_w0, a_k0
            adj_offset_after = adj_offset_before

        return acc.finalize(), (aw, ak)
    finally:
        if profiler is not None and profiler.enabled:
            profiler.record(REVERSE_SWEEP, perf_counter() - started)


# =============================================================================
# Final-state gradients
# =============================================================================

def reverse_final_state_rows(
    build: ProdSpeedProfileBuild,
    *,
    which: str,
    rev: Optional[RevBuild] = None,
) -> Tuple[List[float], List[float]]:
    if isinstance(build, NativeProdSpeedProfileBuild):
        own = rev is None
        native_rev = prepare_reverse_build(build) if rev is None else rev
        if not isinstance(native_rev, NativeRevBuild) or not native_rev.full:
            raise ValueError("final-state rows require full native promotion")
        try:
            w, k = native_rev.native.final_state_rows(which)
            return [float(x) for x in w], [float(x) for x in k]
        finally:
            if own:
                native_rev.close()
    if rev is None:
        rev = prepare_reverse_build(build)
    profiler = rev.profiler
    if which == "forward":
        pass_rev = rev.forward
        needs_backward_initial_k = False
    elif which == "backward":
        pass_rev = rev.backward
        needs_backward_initial_k = True
    else:
        raise ValueError(which)

    _require_complete_reverse_pass(pass_rev, objective="final-state rows")
    rows: List[List[float]] = []
    for seed in ((1.0, 0.0), (0.0, 1.0)):
        grad, init_adj = reverse_scalar_pass(
            pass_rev, seed_final=seed, profiler=profiler
        )
        if needs_backward_initial_k:
            _a_w0, a_k0 = init_adj
            add_geometry_endpoint_k_adjoint(build.raw_params, grad, adj_k_end=a_k0)
        rows.append(grad)
    return rows[0], rows[1]


def reverse_all_final_state_rows(
    build: ProdSpeedProfileBuild,
    *,
    rev: Optional[RevBuild] = None,
) -> Dict[str, Tuple[List[float], List[float]]]:
    if isinstance(build, NativeProdSpeedProfileBuild):
        own = rev is None
        native_rev = prepare_reverse_build(build) if rev is None else rev
        if not isinstance(native_rev, NativeRevBuild) or not native_rev.full:
            raise ValueError("final-state rows require full native promotion")
        try:
            return {
                "forward": reverse_final_state_rows(build, which="forward", rev=native_rev),
                "backward": reverse_final_state_rows(build, which="backward", rev=native_rev),
            }
        finally:
            if own:
                native_rev.close()
    if rev is None:
        rev = prepare_reverse_build(build)
    return {
        "forward": reverse_final_state_rows(build, which="forward", rev=rev),
        "backward": reverse_final_state_rows(build, which="backward", rev=rev),
    }


def _endpoint_kind(x: float, *, total_L: float, tol: float = 1e-10) -> str:
    if abs(x) <= tol * max(1.0, total_L):
        return "start"
    if abs(x - total_L) <= tol * max(1.0, total_L):
        return "end"
    return "internal"


def _clamp_local_to_segment(s: float, L: float, *, eps: float = 1e-12) -> float:
    if s < 0.0 and s > -eps:
        return 0.0
    if s > L and s < L + eps:
        return L
    return s


def _zero_prefix_time_jac(node: RevSegment) -> Tuple[float, Tuple[float, float, float, float]]:
    w0 = node.scalar.w0
    if w0 <= 0.0 or not math.isfinite(w0):
        raise FloatingPointError(f"bad w0 for zero prefix time: {w0}")
    return 0.0, (1.0 / math.sqrt(w0), 0.0, 0.0, 0.0)


def _prefix_time_and_jac(
    node: RevSegment,
    local_s: float,
    *,
    profiler: PhaseProfiler | None = None,
) -> Tuple[float, Tuple[float, float, float, float]]:
    scalar = node.scalar
    L = scalar.L_used
    local_s = _clamp_local_to_segment(local_s, L)
    if local_s < 0.0 or local_s > L + 1e-10:
        raise ValueError(f"local_s outside segment interval: local_s={local_s}, L={L}")
    if abs(local_s) <= 1e-14:
        return _zero_prefix_time_jac(node)

    is_interior = abs(local_s - L) > 1e-14 * max(1.0, L)
    started = (
        perf_counter()
        if is_interior and profiler is not None and profiler.enabled
        else 0.0
    )
    try:
        time, jac = node.seg.time_and_jac(local_s)
        if len(jac) != 4:
            raise ValueError(f"bad prefix time jacobian: {jac}")
        return float(time), tuple(float(x) for x in jac)
    finally:
        if is_interior and profiler is not None and profiler.enabled:
            profiler.record(INTERIOR_PREFIX_TIME_EVAL, perf_counter() - started)


def _seed_time_envelope_piece_prefixdiff(
    ep: EnvelopePiece,
    node: RevSegment,
    extra: SegmentExtraAdj,
    *,
    min_interval: float = 1e-14,
    profiler: PhaseProfiler | None = None,
) -> Tuple[float, float, float]:
    scalar = node.scalar
    l0 = _clamp_local_to_segment(ep.local0, scalar.L_used)
    l1 = _clamp_local_to_segment(ep.local1, scalar.L_used)
    if l0 <= l1:
        lo, hi, lo_is_ep0 = l0, l1, True
    else:
        lo, hi, lo_is_ep0 = l1, l0, False
    if hi - lo <= min_interval:
        return 0.0, 0.0, 0.0

    Tlo, Jlo = _prefix_time_and_jac(node, lo, profiler=profiler)
    Thi, Jhi = _prefix_time_and_jac(node, hi, profiler=profiler)
    T = Thi - Tlo

    extra.asigma += Jhi[1] - Jlo[1]
    extra.aw0 += Jhi[2] - Jlo[2]
    extra.ak0 += Jhi[3] - Jlo[3]

    a_lo = -Jlo[0]
    a_hi = +Jhi[0]
    if lo_is_ep0:
        a_local_ep0, a_local_ep1 = a_lo, a_hi
    else:
        a_local_ep0, a_local_ep1 = a_hi, a_lo

    # local = direction * (abs - segment_abs0)
    a_abs0 = scalar.direction * a_local_ep0
    a_abs1 = scalar.direction * a_local_ep1
    extra.aabs0 += -scalar.direction * (a_local_ep0 + a_local_ep1)
    return T, a_abs0, a_abs1


def reverse_time_objective(
    build: ProdSpeedProfileBuild,
    *,
    rev: Optional[RevBuild] = None,
) -> Tuple[float, List[float]]:
    if isinstance(build, NativeProdSpeedProfileBuild):
        own = rev is None
        native_rev = prepare_reverse_time_build(build) if rev is None else rev
        if not isinstance(native_rev, NativeRevBuild):
            raise TypeError("native scalar build requires native reverse promotion")
        started = perf_counter()
        try:
            value, gradient = native_rev.native.time_value_and_gradient()
            return float(value), [float(x) for x in gradient]
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        finally:
            if native_rev.profiler is not None and native_rev.profiler.enabled:
                native_rev.profiler.record(NATIVE_REVERSE_SWEEP, perf_counter() - started)
            if own:
                native_rev.close()
    if rev is None:
        rev = prepare_reverse_time_build(build)
    profiler = rev.profiler

    n = segment_count(build.raw_params)
    total_L = total_length(build.raw_params)
    value = 0.0
    boundary_acc = RawGradientAccumulator.create(n)
    all_extras = [
        make_empty_extras(len(pass_rev.segments))
        for pass_rev in rev.passes
    ]

    def node_and_extra(ep: EnvelopePiece) -> Tuple[RevSegment, SegmentExtraAdj]:
        if not 0 <= ep.pass_index < len(rev.passes):
            raise ValueError(f"bad envelope pass index: {ep.pass_index}")
        return (
            rev.passes[ep.pass_index].segments[ep.source_index],
            all_extras[ep.pass_index][ep.source_index],
        )

    for ep in build.envelope:
        node, extra = node_and_extra(ep)
        T_piece, adj_abs0, adj_abs1 = _seed_time_envelope_piece_prefixdiff(
            ep, node, extra, profiler=profiler
        )
        value += T_piece
        if _endpoint_kind(ep.abs0, total_L=total_L) == "end":
            boundary_acc.add_global_end(adj_abs0)
        if _endpoint_kind(ep.abs1, total_L=total_L) == "end":
            boundary_acc.add_global_end(adj_abs1)

    grad_boundary = boundary_acc.finalize()
    grad = list(grad_boundary)
    for scalar_pass, pass_rev, extras in zip(
        build.scalar_passes, rev.passes, all_extras
    ):
        pass_grad, init_adj = reverse_scalar_pass(
            pass_rev,
            seed_final=(0.0, 0.0),
            extras=extras,
            profiler=profiler,
        )
        for i in range(2 * n):
            grad[i] += pass_grad[i]

        a_w0, a_k0 = init_adj
        knot_index = scalar_pass.initial_knot_index
        if knot_index is not None:
            add_geometry_knot_k_adjoint(
                build.raw_params,
                grad,
                knot_index=knot_index,
                adj_k=a_k0 + a_w0 * scalar_pass.initial_w_dk,
            )
    return value, grad


def reverse_pass_total_time(
    rev_pass: RevPass,
    build: ProdSpeedProfileBuild,
    *,
    pass_kind: str,
    profiler: PhaseProfiler | None = None,
) -> Tuple[float, List[float], Tuple[float, float]]:
    _require_complete_reverse_pass(rev_pass, objective="pass-total time")
    extras = make_empty_extras(len(rev_pass.segments))
    value = 0.0
    for i, node in enumerate(rev_pass.segments):
        time, tj = node.seg.time_and_jac(node.scalar.L_used)
        value += time
        extras[i].aL += tj[0]
        extras[i].asigma += tj[1]
        extras[i].aw0 += tj[2]
        extras[i].ak0 += tj[3]
    grad, init_adj = reverse_scalar_pass(
        rev_pass,
        seed_final=(0.0, 0.0),
        extras=extras,
        profiler=profiler,
    )
    if pass_kind == "backward":
        _a_w0, a_k0 = init_adj
        add_geometry_endpoint_k_adjoint(build.raw_params, grad, adj_k_end=a_k0)
    elif pass_kind != "forward":
        raise ValueError(pass_kind)
    return value, grad, init_adj


# =============================================================================
# Convenience diagnostics
# =============================================================================

def event_counts(build: ProdSpeedProfileBuild) -> Dict[str, int]:
    out: Dict[str, int] = {e.name: 0 for e in EventKind}
    if isinstance(build, NativeProdSpeedProfileBuild):
        for seg in build.native.segments():
            out[seg.event] += 1
        return out
    for spass in build.scalar_passes:
        for s in spass.segments:
            out[s.event.name] += 1
    return out


def build_has_mode_event(build: ProdSpeedProfileBuild) -> bool:
    counts = event_counts(build)
    return sum(v for k, v in counts.items() if k != EventKind.PIECE_END.name) > 0


@dataclass(slots=True)
class ReverseCheck:
    group: str
    name: str
    ok: bool
    detail: str


def _close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(1.0, abs(a), abs(b))
