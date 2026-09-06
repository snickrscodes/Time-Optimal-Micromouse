"""Switching-event and friction-domain searches for compiled segments.

Motor and grip candidates use one shared scan: each node evaluates the state
once, then derives both the friction-domain margin and active switching
residual.  Event and domain brackets share endpoint values.  Brake candidates
use the exact quadratic form of ``w(s) * k(s)`` to locate both thresholds
without a uniform scan.

Python 3.13+.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Protocol

from segment.constants import A_BRAKE, A_MAX, B_EMF, MU_G, V_MAX
from segment.grip import GripDomainError, GripEvalSegment
from segment.motor import (
    LW_EQ, MOTOR_U0_SERIES_TOL, MotorEvalSegment, MotorEvalSegmentStable,
)

# Derived physical constants.  The five imported constants above are the only
# independent physical constants used by this module.
MU_G2 = MU_G * MU_G
K_BRAKE = math.sqrt(math.fma(-A_BRAKE, A_BRAKE, MU_G2))

# GRIP->BRAKE structural certificates deliberately keep a small binary64
# moat around analytic inequalities.  The formulas are exact in real
# arithmetic; the moat makes every hard branch fail closed when roundoff could
# change an inequality at the threshold.  Analytic event brackets are also
# checked with the authoritative Cflow residual before refinement.
_GRIP_BRAKE_CERT_ULPS = 16
_MOTOR_CERT_ULPS = 16


@dataclass(slots=True, frozen=True)
class MotorSwitchGeometry:
    """Derived geometry of the GRIP/MOTOR switching surface.

    All values are recomputed from ``(M, A, B)``.  ``orientation_certified``
    is true only when the normalized H-prime polynomial has exactly one root
    between the maximum of R and motor equilibrium; changed physical constants
    that violate that topology therefore fail closed to the legacy scanner.
    """

    M: float
    A: float
    B: float
    v_min: float
    v_eq: float
    v_R: float
    R_max: float
    v_H: float | None
    H_max: float | None
    orientation_certified: bool
    rising_barrier_certified: bool


def _poly_eval_ascending(coeffs: tuple[float, ...], x: float) -> float:
    out = 0.0
    for c in reversed(coeffs):
        out = math.fma(out, x, c)
    return out


def _bisect_poly_root(
    coeffs: tuple[float, ...], lo: float, hi: float, *, max_iter: int = 96
) -> float:
    flo = _poly_eval_ascending(coeffs, lo)
    fhi = _poly_eval_ascending(coeffs, hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if not (math.isfinite(flo) and math.isfinite(fhi)) or flo * fhi > 0.0:
        raise ValueError("polynomial root not bracketed")
    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        if m == a or m == b:
            break
        fm = _poly_eval_ascending(coeffs, m)
        if fm == 0.0:
            return m
        if flo * fm <= 0.0:
            b, fhi = m, fm
        else:
            a, flo = m, fm
    return 0.5 * (a + b)


def _poly_real_roots_interval(
    coeffs: tuple[float, ...], lo: float, hi: float
) -> tuple[float, ...]:
    """Isolate all simple/tangent real roots of a low-degree polynomial.

    Derivative roots partition the interval into monotone pieces, so every
    sign-changing root is isolated by bisection.  A root coincident with a
    derivative root is retained when its residual is within a conservative
    floating evaluation tolerance.  This is used only during module-level
    physical-geometry setup, never in a hot crossing loop.
    """
    coeffs = tuple(float(c) for c in coeffs)
    while len(coeffs) > 1 and coeffs[-1] == 0.0:
        coeffs = coeffs[:-1]
    degree = len(coeffs) - 1
    if degree <= 0 or not lo < hi:
        return ()
    if degree == 1:
        root = -coeffs[0] / coeffs[1]
        return (root,) if lo <= root <= hi and math.isfinite(root) else ()

    deriv = tuple((i + 1) * coeffs[i + 1] for i in range(degree))
    critical = _poly_real_roots_interval(deriv, lo, hi)
    points = [lo, *critical, hi]
    roots: list[float] = []
    scale = max(1.0, sum(abs(c) for c in coeffs))
    zero_tol = 256.0 * math.ulp(scale)

    for x in critical:
        fx = _poly_eval_ascending(coeffs, x)
        if math.isfinite(fx) and abs(fx) <= zero_tol:
            roots.append(x)

    for a, b in zip(points, points[1:]):
        if not a < b:
            continue
        aa = math.nextafter(a, b) if a != lo else a
        bb = math.nextafter(b, a) if b != hi else b
        if not aa <= bb:
            continue
        fa = _poly_eval_ascending(coeffs, aa)
        fb = _poly_eval_ascending(coeffs, bb)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        if fa == 0.0:
            roots.append(aa)
        if fb == 0.0:
            roots.append(bb)
        if fa * fb < 0.0:
            roots.append(_bisect_poly_root(coeffs, aa, bb))

    roots.sort()
    dedup: list[float] = []
    for root in roots:
        if not dedup or abs(root - dedup[-1]) > 64.0 * math.ulp(max(1.0, abs(root))):
            dedup.append(root)
    return tuple(dedup)


def _motor_R_for(M: float, A: float, B: float, v: float) -> float:
    if not v > 0.0:
        return math.inf
    gap = math.fma(B, v, -A)
    rad = math.fma(-gap, gap, M * M)
    if rad < 0.0:
        return math.nan
    return math.sqrt(max(0.0, rad)) / (v * v)


def _motor_R_prime_for(M: float, A: float, B: float, v: float) -> float:
    gap = math.fma(B, v, -A)
    rad = math.fma(-gap, gap, M * M)
    if not (v > 0.0 and rad > 0.0):
        return math.nan
    num = math.fma(B * B, v * v, math.fma(-3.0 * A * B, v, 2.0 * (A * A - M * M)))
    return num / (v * v * v * math.sqrt(rad))


def _motor_H_for(M: float, A: float, B: float, v: float) -> float:
    rp = _motor_R_prime_for(M, A, B, v)
    if not math.isfinite(rp) or not v > 0.0:
        return math.nan
    return -rp * math.fma(-B, v, A) / v


def _derive_motor_switch_geometry(M: float, A: float, B: float) -> MotorSwitchGeometry:
    M, A, B = map(float, (M, A, B))
    if not all(math.isfinite(x) and x > 0.0 for x in (M, A, B)) or not A > M:
        return MotorSwitchGeometry(M, A, B, math.nan, math.nan, math.nan, math.nan, None, None, False, False)

    v_min = (A - M) / B
    v_eq = A / B
    disc = math.sqrt(math.fma(8.0, M * M, A * A))
    v_R = (3.0 * A - disc) / (2.0 * B)
    if not (v_min < v_R < v_eq):
        return MotorSwitchGeometry(M, A, B, v_min, v_eq, v_R, math.nan, None, None, False, False)
    R_max = _motor_R_for(M, A, B, v_R)

    # H'(v)=0 becomes a dimensionless quintic in y=Bv/A, depending only
    # on m=M/A.  Recompute its coefficients from the supplied constants.
    m = M / A
    m2 = m * m
    m4 = m2 * m2
    coeffs = (
        -8.0 + 16.0 * m2 - 8.0 * m4,
        33.0 - 39.0 * m2 + 6.0 * m4,
        -53.0 + 32.0 * m2,
        41.0 - 9.0 * m2,
        -15.0,
        2.0,
    )
    y_R = B * v_R / A
    lo = math.nextafter(y_R, 1.0)
    hi = math.nextafter(1.0, y_R)
    roots = _poly_real_roots_interval(coeffs, lo, hi)
    if len(roots) != 1:
        return MotorSwitchGeometry(M, A, B, v_min, v_eq, v_R, R_max, None, None, False, False)
    v_H = (A / B) * roots[0]
    H_max = _motor_H_for(M, A, B, v_H)
    certified = math.isfinite(H_max) and H_max > 0.0

    y_min = (A - M) / A
    rise_lo = math.nextafter(y_min, y_R)
    rise_hi = math.nextafter(y_R, y_min)
    rise_roots = _poly_real_roots_interval(coeffs, rise_lo, rise_hi)
    rise_mid = 0.5 * (rise_lo + rise_hi)
    rising_certified = (
        not rise_roots
        and _poly_eval_ascending(coeffs, rise_mid) > 0.0
    )
    return MotorSwitchGeometry(
        M, A, B, v_min, v_eq, v_R, R_max, v_H, H_max,
        certified, rising_certified,
    )


MOTOR_SWITCH_GEOMETRY = _derive_motor_switch_geometry(MU_G, A_MAX, B_EMF)


def _shift_ulps(value: float, toward: float, count: int = _GRIP_BRAKE_CERT_ULPS) -> float:
    out = float(value)
    if not math.isfinite(out):
        return out
    for _ in range(count):
        out = math.nextafter(out, toward)
    return out


K_BRAKE_LO = _shift_ulps(K_BRAKE, -math.inf)
K_BRAKE_HI = _shift_ulps(K_BRAKE, math.inf)
D_BRAKE = 4.0 * K_BRAKE
S_BRAKE = 2.0 * A_BRAKE
W_EQ = V_MAX * V_MAX

POS_TO_NEG = 1.0
NEG_TO_POS = -1.0

CROSSING_SCAN_PHASE = "crossing_scan_evaluations"
DOMAIN_SCAN_PHASE = "domain_scan_evaluations"


class _Profiler(Protocol):
    enabled: bool

    def record(self, phase: str, elapsed: float, *, count: int = 1) -> None: ...


EventEvaluator = Callable[[Any, float, _Profiler | None], tuple[float, float | None, float]]


@dataclass(slots=True, frozen=True)
class ScanResult:
    """Result of a shared switching/domain scan.

    ``event`` is returned only when a valid directional event is found no
    later than the first friction-domain edge.  ``domain_edge`` is the first
    location at which ``G² <= domain_margin`` when such an edge is bracketed.

    ``initial_switch`` distinguishes the scanner's zero-location mode-switch
    sentinel from a genuine positive root that merely lies below the caller's
    spatial tolerance.  Conflating the two can cause MOTOR/GRIP or
    BRAKE/GRIP chatter without station progress.
    """

    event: float | None
    domain_edge: float | None
    event_bracket: tuple[float, float, float, float] | None = None
    domain_bracket: tuple[float, float, float, float] | None = None
    initial_switch: bool = False
    # Valid-side station immediately before ``domain_edge``.  Legacy callers
    # ignore this field; fused GRIP discovery uses it to emit a replay-safe
    # prefix without a second full domain scan.
    domain_safe: float | None = None
    # Structured native friction-terminal location, when the domain stop was
    # reported by Cflow rather than only by a sampled G^2 crossing.  This is
    # retained so the caller can reproduce the legacy conservative endpoint
    # fallback without doing a speculative full-horizon probe up front.
    domain_event: float | None = None
    # The switching theorem has excluded a mode event over this horizon and
    # the physical-domain bounds prove a domain stop.  The reverse solver may
    # therefore delegate endpoint selection directly to its unchanged legacy
    # domain certifier without rescanning for a switch.
    domain_switch_excluded: bool = False


def _profile_record(
    profiler: _Profiler | None,
    phase: str,
    started: float,
) -> None:
    if profiler is not None and profiler.enabled:
        profiler.record(phase, perf_counter() - started)


def bracketed_newton(
    f_df: Callable[[float], tuple[float, float | None]],
    lo: float,
    hi: float,
    f_lo: float,
    f_hi: float,
    *,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    max_iter: int = 64,
) -> float | None:
    """Safeguarded Newton/secant solve on an already validated bracket."""
    if hi < lo:
        lo, hi = hi, lo
        f_lo, f_hi = f_hi, f_lo

    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return None
    if f_lo * f_hi > 0.0:
        return None

    if f_hi != f_lo:
        x = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        if not (lo < x < hi) or not math.isfinite(x):
            x = 0.5 * (lo + hi)
    else:
        x = 0.5 * (lo + hi)

    for _ in range(max_iter):
        f_x, df_x = f_df(x)
        if not math.isfinite(f_x):
            x = 0.5 * (lo + hi)
            f_x, df_x = f_df(x)
            if not math.isfinite(f_x):
                return None
        if abs(f_x) <= f_tol:
            return x

        if f_lo * f_x <= 0.0:
            hi = x
            f_hi = f_x
        else:
            lo = x
            f_lo = f_x

        mid = 0.5 * (lo + hi)
        if hi - lo <= x_abs_tol + x_rel_tol * max(1.0, abs(mid)):
            return mid

        if df_x is not None and df_x != 0.0 and math.isfinite(df_x):
            xn = x - f_x / df_x
            if lo < xn < hi and math.isfinite(xn):
                x = xn
                continue

        if f_hi != f_lo:
            xs = hi - f_hi * (hi - lo) / (f_hi - f_lo)
            if lo < xs < hi and math.isfinite(xs):
                x = xs
                continue

        x = mid

    return 0.5 * (lo + hi)


def _crosses(fa: float, fb: float, direction: float) -> bool:
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    if direction == POS_TO_NEG:
        return fa >= 0.0 and fb <= 0.0
    if direction == NEG_TO_POS:
        return fa <= 0.0 and fb >= 0.0
    raise ValueError(f"invalid crossing direction: {direction!r}")


def _already_on_target_side(f: float, direction: float) -> bool:
    if direction == POS_TO_NEG:
        return f <= 0.0
    if direction == NEG_TO_POS:
        return f >= 0.0
    raise ValueError(f"invalid crossing direction: {direction!r}")


def _initial_switch_required(
    f: float,
    df: float | None,
    direction: float,
    *,
    residual_tol: float,
    spatial_tol: float,
) -> bool:
    """Resolve a near-start switching surface without numerical chatter.

    If the inferred root ``abs(f / df)`` lies below the event locator's
    spatial resolution, residual sign is not a meaningful mode classifier.
    The one-sided derivative determines which active region is entered after
    the boundary.  This prevents zero-scale MOTOR/GRIP and BRAKE/GRIP bounce.
    """
    if not math.isfinite(f):
        return False

    finite_df = df is not None and math.isfinite(df)
    near_boundary = abs(f) <= residual_tol
    if finite_df and df != 0.0:
        near_boundary = near_boundary or abs(f / df) <= spatial_tol

    if near_boundary:
        if not finite_df:
            return False
        if direction == POS_TO_NEG:
            return df < 0.0
        if direction == NEG_TO_POS:
            return df > 0.0
        raise ValueError(f"invalid crossing direction: {direction!r}")

    return _already_on_target_side(f, direction)


def _state_g2(
    seg: Any,
    ds: float,
    profiler: _Profiler | None,
) -> tuple[float, float, float, float]:
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0
    w = seg.w(ds)
    k = math.fma(seg.sigma, ds, seg.k0)
    q = w * k
    g2 = math.fma(-q, q, MU_G2)
    _profile_record(profiler, DOMAIN_SCAN_PHASE, started)
    return w, k, q, g2


def motor_grip_f_df_g2(
    seg: Any,
    ds: float,
    profiler: _Profiler | None = None,
) -> tuple[float, float | None, float]:
    w, k, q, g2 = _state_g2(seg, ds, profiler)
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0

    if not math.isfinite(w) or w < 0.0:
        result = (float("nan"), None, g2)
    else:
        y = math.sqrt(w)
        if g2 <= 0.0 or not math.isfinite(g2):
            result = (math.fma(B_EMF, y, -A_MAX), None, g2)
        else:
            g = math.sqrt(g2)
            residual = math.fma(B_EMF, y, g - A_MAX)
            dw = 2.0 * math.fma(B_EMF, -y, A_MAX)
            q_prime = math.fma(k, dw, seg.sigma * w)
            if y == 0.0:
                deriv = None
            else:
                deriv = math.fma(-q / g, q_prime, 0.5 * B_EMF * dw / y)
            result = (residual, deriv, g2)

    _profile_record(profiler, CROSSING_SCAN_PHASE, started)
    return result


def grip_motor_f_df_g2(
    seg: Any,
    ds: float,
    profiler: _Profiler | None = None,
) -> tuple[float, float | None, float]:
    w, k, q, g2 = _state_g2(seg, ds, profiler)
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0

    if not math.isfinite(w) or w < 0.0:
        result = (float("nan"), None, g2)
    else:
        y = math.sqrt(w)
        if g2 <= 0.0 or not math.isfinite(g2):
            result = (math.fma(B_EMF, y, -A_MAX), None, g2)
        else:
            g = math.sqrt(g2)
            residual = math.fma(B_EMF, y, g - A_MAX)
            dw = 2.0 * g
            q_prime = math.fma(k, dw, seg.sigma * w)
            if y == 0.0:
                deriv = None
            else:
                deriv = math.fma(-q / g, q_prime, 0.5 * B_EMF * dw / y)
            result = (residual, deriv, g2)

    _profile_record(profiler, CROSSING_SCAN_PHASE, started)
    return result


def grip_brake_f_df_g2(
    seg: Any,
    ds: float,
    profiler: _Profiler | None = None,
) -> tuple[float, float | None, float]:
    w, k, q, g2 = _state_g2(seg, ds, profiler)
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0

    if g2 <= 0.0 or not math.isfinite(g2):
        result = (-A_BRAKE, None, g2)
    else:
        g = math.sqrt(g2)
        residual = g - A_BRAKE
        dw = 2.0 * g
        q_prime = math.fma(k, dw, seg.sigma * w)
        result = (residual, -q * q_prime / g, g2)

    _profile_record(profiler, CROSSING_SCAN_PHASE, started)
    return result


def brake_grip_f_df_g2(
    seg: Any,
    ds: float,
    profiler: _Profiler | None = None,
) -> tuple[float, float | None, float]:
    w, k, q, g2 = _state_g2(seg, ds, profiler)
    started = perf_counter() if profiler is not None and profiler.enabled else 0.0

    if g2 <= 0.0 or not math.isfinite(g2):
        result = (-A_BRAKE, None, g2)
    else:
        g = math.sqrt(g2)
        residual = g - A_BRAKE
        dw = S_BRAKE
        q_prime = math.fma(k, dw, seg.sigma * w)
        result = (residual, -q * q_prime / g, g2)

    _profile_record(profiler, CROSSING_SCAN_PHASE, started)
    return result


def _bisect_domain_edge(
    f_df_g2: EventEvaluator,
    seg: Any,
    lo: float,
    hi: float,
    g2_lo: float,
    g2_hi: float,
    *,
    margin: float,
    max_iter: int = 80,
) -> float:
    """Refine the first ``G²-margin`` sign change inside one scan cell."""
    if g2_lo <= margin:
        return lo
    if g2_hi > margin:
        raise ValueError("domain edge not bracketed")

    a = lo
    b = hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        try:
            _, _, g2_mid = f_df_g2(seg, mid, None)
        except Exception:
            g2_mid = -math.inf
        if math.isfinite(g2_mid) and g2_mid > margin:
            a = mid
        else:
            b = mid
    return b


def scan_first_event_and_domain(
    seg: Any,
    L: float,
    *,
    f_df_g2: EventEvaluator,
    direction: float,
    n_scan: int = 256,
    domain_margin: float = 1e-12,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    max_iter: int = 64,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    """Find the first directional event and first domain edge in one scan."""
    if L < 0.0 or not math.isfinite(L):
        raise ValueError(f"scan length must be finite and nonnegative, got {L!r}")
    if n_scan <= 0:
        raise ValueError("n_scan must be positive")

    try:
        f_prev, df_prev, g2_prev = f_df_g2(seg, 0.0, profiler)
    except Exception:
        return ScanResult(None, 0.0, None, (0.0, 0.0, float("nan"), -math.inf))

    if not math.isfinite(g2_prev):
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_prev, g2_prev))
    if g2_prev <= domain_margin and not allow_initial_boundary:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_prev, g2_prev))

    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    elif not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
        raise ValueError(
            "initial_spatial_tol must be finite and nonnegative when supplied"
        )
    else:
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)
    if _initial_switch_required(
        f_prev,
        df_prev,
        direction,
        residual_tol=f_tol,
        spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(
            0.0, None, (0.0, 0.0, f_prev, f_prev), None, True
        )

    if L == 0.0:
        return ScanResult(None, None)

    s_prev = 0.0
    for j in range(1, n_scan + 1):
        s = L * (j / n_scan)
        try:
            f, _, g2 = f_df_g2(seg, s, profiler)
        except Exception:
            f = float("nan")
            g2 = -math.inf

        if not math.isfinite(g2) or g2 <= domain_margin:
            edge = _bisect_domain_edge(
                f_df_g2,
                seg,
                s_prev,
                s,
                g2_prev,
                g2 if math.isfinite(g2) else -math.inf,
                margin=domain_margin,
            )
            f_edge, _, g2_edge = f_df_g2(seg, edge, None)
            domain_bracket = (s_prev, s, g2_prev, g2)

            if _crosses(f_prev, f_edge, direction):
                event_bracket = (s_prev, edge, f_prev, f_edge)

                def f_df(x: float) -> tuple[float, float | None]:
                    fx, dfx, _ = f_df_g2(seg, x, None)
                    return fx, dfx

                root = bracketed_newton(
                    f_df,
                    s_prev,
                    edge,
                    f_lo=f_prev,
                    f_hi=f_edge,
                    x_abs_tol=x_abs_tol,
                    x_rel_tol=x_rel_tol,
                    f_tol=f_tol,
                    max_iter=max_iter,
                )
                if root is not None and root <= edge:
                    return ScanResult(root, edge, event_bracket, domain_bracket)

            return ScanResult(None, edge, None, domain_bracket)

        if _crosses(f_prev, f, direction):
            event_bracket = (s_prev, s, f_prev, f)

            def f_df(x: float) -> tuple[float, float | None]:
                fx, dfx, _ = f_df_g2(seg, x, None)
                return fx, dfx

            root = bracketed_newton(
                f_df,
                s_prev,
                s,
                f_lo=f_prev,
                f_hi=f,
                x_abs_tol=x_abs_tol,
                x_rel_tol=x_rel_tol,
                f_tol=f_tol,
                max_iter=max_iter,
            )
            return ScanResult(root, None, event_bracket, None)

        s_prev = s
        f_prev = f
        g2_prev = g2

    return ScanResult(None, None)



def _grip_point_or_domain(
    f_df_g2: EventEvaluator,
    seg: Any,
    ds: float,
    profiler: _Profiler | None,
) -> tuple[float, float | None, float, bool, float | None]:
    """Evaluate one GRIP scan point, distinguishing physical-domain exit.

    Only ``GripDomainError`` is converted into a domain sentinel.  Conditioning
    and numerical failures continue to propagate exactly as they did through
    the former full-horizon ``domain_probe``.
    """
    try:
        f, df, g2 = f_df_g2(seg, ds, profiler)
        return f, df, g2, False, None
    except GripDomainError as error:
        event = error.event_position
        event = float(event) if event is not None and math.isfinite(event) else None
        return float("nan"), None, -math.inf, True, event


def _refine_grip_domain_stop(
    f_df_g2: EventEvaluator,
    seg: Any,
    lo: float,
    hi: float,
    g2_lo: float,
    *,
    physical_margin: float,
    replay_target: float,
    max_iter: int = 80,
) -> tuple[float, float, float, float]:
    """Refine a physical friction edge and select its replay-safe side.

    ``lo`` must be an authoritative in-domain state with ``G^2`` above the
    physical margin.  ``hi`` may either evaluate at/below the physical margin
    or terminate through Cflow's structured friction event.  The physical edge
    is refined first; when the bracket contains the same replay cushion used by
    ``first_domain_clip`` we then refine that state-space target on the valid
    side.  No propagation beyond the first physical edge is requested.
    """
    if not g2_lo > physical_margin:
        return lo, lo, g2_lo, g2_lo

    a, b, ga, gb = lo, hi, g2_lo, -math.inf
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        _f, _df, gm, outside, _event = _grip_point_or_domain(f_df_g2, seg, mid, None)
        if not outside and math.isfinite(gm) and gm > physical_margin:
            a, ga = mid, gm
        else:
            b, gb = mid, gm if math.isfinite(gm) else -math.inf

    # Match first_domain_clip's state-space replay guard whenever the coarse
    # valid side lies outside that guard.  If the coarse side is already below
    # the replay target, retain the physical valid side just as the legacy
    # clipper does rather than inventing a station-distance epsilon.
    safe, g_safe = a, ga
    if g2_lo >= replay_target and ga < replay_target:
        c, d, gc = lo, a, g2_lo
        for _ in range(max_iter):
            mid = 0.5 * (c + d)
            if mid == c or mid == d:
                break
            _f, _df, gm, outside, _event = _grip_point_or_domain(f_df_g2, seg, mid, None)
            if not outside and math.isfinite(gm) and gm >= replay_target:
                c, gc = mid, gm
            else:
                d = mid
        safe, g_safe = c, gc
    return safe, b, g_safe, gb


def scan_first_event_or_domain_safe(
    seg: Any,
    L: float,
    *,
    f_df_g2: EventEvaluator,
    direction: float,
    n_scan: int = 256,
    domain_stop_margin: float = 4e-12,
    physical_domain_margin: float = 1e-12,
    domain_safe_floor: float = 0.0,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    max_iter: int = 64,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    """Find the earliest certified GRIP switch or physical domain boundary.

    The legacy path first propagated to the end of the requested GRIP horizon
    to learn whether a friction terminal existed, then rescanned the safe
    prefix for a possibly much earlier mode switch.  Here switching residuals
    and physical-domain state are inspected in one monotone traversal.  A
    replay cushion is selected only *after* a physical domain boundary is
    actually bracketed, so a finite horizon that merely approaches the cushion
    is not truncated.

    A native friction terminal is returned separately in ``domain_event``.
    The caller may use that location for the legacy few-ulp endpoint
    recertification/fallback without reinstating a speculative full-horizon
    probe before the switching search.
    """
    if L < 0.0 or not math.isfinite(L):
        raise ValueError(f"scan length must be finite and nonnegative, got {L!r}")
    if n_scan <= 0:
        raise ValueError("n_scan must be positive")
    if domain_stop_margin < physical_domain_margin:
        raise ValueError("domain_stop_margin must not be below physical_domain_margin")

    f_prev, df_prev, g2_prev, outside, event0 = _grip_point_or_domain(f_df_g2, seg, 0.0, profiler)
    if outside or not math.isfinite(g2_prev):
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_prev, g2_prev), False, 0.0, event0)
    if g2_prev <= physical_domain_margin and not allow_initial_boundary:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_prev, g2_prev), False, 0.0, None)

    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    elif not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
        raise ValueError("initial_spatial_tol must be finite and nonnegative when supplied")
    else:
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)
    if _initial_switch_required(
        f_prev, df_prev, direction,
        residual_tol=f_tol, spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(0.0, None, (0.0, 0.0, f_prev, f_prev), None, True, None, None)
    if L == 0.0:
        return ScanResult(None, None)

    s_prev = 0.0
    for j in range(1, n_scan + 1):
        s = L * (j / n_scan)
        f, _df, g2, outside, native_event = _grip_point_or_domain(f_df_g2, seg, s, profiler)
        physical_stop = outside or not math.isfinite(g2) or g2 <= physical_domain_margin
        if physical_stop:
            # First refine the authoritative physical boundary; only then move
            # back to the replay cushion.  This lets a switch inside the same
            # coarse cell win if it occurs on the certified valid side.
            safe, edge, g_safe, g_edge = _refine_grip_domain_stop(
                f_df_g2, seg, s_prev, s, g2_prev,
                physical_margin=physical_domain_margin,
                replay_target=domain_stop_margin,
            )
            if safe < domain_safe_floor and edge > domain_safe_floor:
                safe = domain_safe_floor
                _fs, _dfs, g_safe, safe_outside, _safe_event = _grip_point_or_domain(
                    f_df_g2, seg, safe, None
                )
                if safe_outside or not math.isfinite(g_safe) or g_safe <= physical_domain_margin:
                    safe = math.nextafter(edge, 0.0)

            f_safe, _df_safe, _g2_safe, safe_outside, _safe_event = _grip_point_or_domain(
                f_df_g2, seg, safe, None
            )
            if not safe_outside and safe > s_prev and _crosses(f_prev, f_safe, direction):
                event_bracket = (s_prev, safe, f_prev, f_safe)

                def f_df(x: float) -> tuple[float, float | None]:
                    fx, dfx, _g, is_outside, _ev = _grip_point_or_domain(f_df_g2, seg, x, None)
                    if is_outside:
                        return float("nan"), None
                    return fx, dfx

                root = bracketed_newton(
                    f_df, s_prev, safe, f_lo=f_prev, f_hi=f_safe,
                    x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol,
                    f_tol=f_tol, max_iter=max_iter,
                )
                if root is not None and root <= safe:
                    return ScanResult(root, None, event_bracket, None, False, None, None)
            return ScanResult(
                None, edge, None, (s_prev, s, g2_prev, g2), False, safe, native_event
            )

        if _crosses(f_prev, f, direction):
            event_bracket = (s_prev, s, f_prev, f)

            def f_df(x: float) -> tuple[float, float | None]:
                fx, dfx, _g, is_outside, _ev = _grip_point_or_domain(f_df_g2, seg, x, None)
                if is_outside:
                    return float("nan"), None
                return fx, dfx

            root = bracketed_newton(
                f_df, s_prev, s, f_lo=f_prev, f_hi=f,
                x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol,
                f_tol=f_tol, max_iter=max_iter,
            )
            return ScanResult(root, None, event_bracket, None, False, None, None)
        s_prev, f_prev, g2_prev = s, f, g2
    return ScanResult(None, None)


def _motor_R(v: float) -> float:
    g = MOTOR_SWITCH_GEOMETRY
    return _motor_R_for(g.M, g.A, g.B, v)


def _motor_H(v: float) -> float:
    g = MOTOR_SWITCH_GEOMETRY
    return _motor_H_for(g.M, g.A, g.B, v)


def _bisect_scalar_level(
    func: Callable[[float], float], lo: float, hi: float, target: float,
    *, increasing: bool, max_iter: int = 80,
) -> float:
    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        if m == a or m == b:
            break
        fm = func(m)
        if (fm < target) == increasing:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _motor_upper_w(w0: float, ds: float) -> float | None:
    """Analytic MOTOR-flow upper speed envelope before a GRIP->MOTOR switch."""
    if ds < 0.0 or not (math.isfinite(w0) and w0 >= 0.0 and math.isfinite(ds)):
        return None
    try:
        u0 = math.fma(LW_EQ, math.sqrt(w0), -1.0)
        cls = MotorEvalSegmentStable if abs(u0) <= MOTOR_U0_SERIES_TOL else MotorEvalSegment
        value = float(cls(ds, 0.0, w0, 0.0).w(ds))
        return _shift_ulps(value, math.inf, _MOTOR_CERT_ULPS)
    except Exception:
        return None


def _motor_R_interval_max(v_lo: float, v_hi: float) -> float:
    """Upper bound on R(v) over a reachable speed interval."""
    g = MOTOR_SWITCH_GEOMETRY
    if v_hi < v_lo:
        v_lo, v_hi = v_hi, v_lo
    lo = max(v_lo, g.v_min)
    hi = min(v_hi, g.v_eq)
    if not lo <= hi:
        return 0.0
    vals = [_motor_R(lo), _motor_R(hi)]
    if lo <= g.v_R <= hi:
        vals.append(g.R_max)
    value = max(v for v in vals if math.isfinite(v))
    return _shift_ulps(value, math.inf, _MOTOR_CERT_ULPS)


def _grip_motor_structural_decreasing_scan(
    seg: Any,
    L: float,
    *,
    domain_margin: float,
    profiler: _Profiler | None,
    x_abs_tol: float,
    x_rel_tol: float,
    f_tol: float,
    max_iter: int,
    allow_initial_boundary: bool,
    initial_spatial_tol: float | None,
) -> ScanResult | None:
    """Fail-closed GRIP->MOTOR classification on decreasing ``|k|``.

    Production deliberately uses the structural theorem only to prove that no
    MOTOR contact exists (or to preserve the established initial-switch
    semantics).  Every positive noninitial MOTOR event is delegated to the
    legacy scanner so its first-root partition remains the numerical authority.
    """
    _ = max_iter
    g = MOTOR_SWITCH_GEOMETRY
    if not g.orientation_certified:
        return None
    branch = _grip_brake_decreasing_horizon(seg, L)
    if branch is None:
        return None
    h, crosses_zero = branch

    f0, df0, g2_0 = grip_motor_f_df_g2(seg, 0.0, profiler)
    if not math.isfinite(g2_0):
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    if g2_0 <= domain_margin and not allow_initial_boundary:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    elif not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
        raise ValueError("initial_spatial_tol must be finite and nonnegative when supplied")
    else:
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)
    if _initial_switch_required(
        f0, df0, NEG_TO_POS, residual_tol=f_tol, spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(0.0, None, (0.0, 0.0, f0, f0), None, True)
    if L == 0.0:
        return ScanResult(None, None)

    # Once k crosses zero, |k| begins increasing again.  This production fast
    # path does not reason across that mixed branch; leave it to legacy.
    if crosses_zero and h < L:
        return None

    # Before a GRIP->MOTOR contact, G <= A-Bv, so the analytic MOTOR flow is
    # an upper envelope for speed.  If even the minimum reachable curvature
    # magnitude stays above the maximum switching curvature R(v) over that
    # entire reachable speed interval, D=|k|-R(v) remains strictly positive
    # and the switch is impossible.
    v0 = math.sqrt(max(0.0, seg.w0))
    w_up_h = _motor_upper_w(seg.w0, h)
    if w_up_h is None or not math.isfinite(w_up_h) or w_up_h < seg.w0:
        return None
    v_up_h = math.sqrt(max(0.0, w_up_h))
    r_h = abs(math.fma(seg.sigma, h, seg.k0))
    max_R = _motor_R_interval_max(v0, v_up_h)
    if _shift_ulps(r_h, -math.inf, _MOTOR_CERT_ULPS) > max_R:
        return ScanResult(None, None)
    return None


def _grip_increasing_domain_certificate(
    seg: GripEvalSegment,
    L: float,
    *,
    domain_margin: float,
) -> bool | None:
    """Classify physical domain reach for sign-stable increasing ``|k|``.

    Returns ``False`` when the full horizon is certified inside, ``True`` when
    a friction terminal is certified to occur, and ``None`` when the cheap
    envelopes overlap.  No Cflow propagation is performed.
    """
    if L < 0.0 or not math.isfinite(L) or seg.sigma == 0.0 or seg.k0 == 0.0:
        return None
    k1 = math.fma(seg.sigma, L, seg.k0)
    if k1 == 0.0 or math.copysign(1.0, k1) != math.copysign(1.0, seg.k0):
        return None
    r0 = abs(seg.k0)
    r1 = abs(k1)
    if not r1 > r0:
        return None
    q0 = r0 * seg.w0
    g2_0 = math.fma(-q0, q0, MU_G2)
    if not g2_0 > domain_margin:
        return True
    g0 = math.sqrt(g2_0)
    # z=rw/M strictly increases, hence G decreases and
    # w0 <= w(s) <= w0+2G0*s.
    q_lo = _shift_ulps(r1 * seg.w0, -math.inf, _MOTOR_CERT_ULPS)
    w_hi = math.fma(2.0 * g0, L, seg.w0)
    q_hi = _shift_ulps(r1 * w_hi, math.inf, _MOTOR_CERT_ULPS)
    q_target_lo = _shift_ulps(
        math.sqrt(max(0.0, MU_G2 - domain_margin)), -math.inf, _MOTOR_CERT_ULPS
    )
    q_target_hi = _shift_ulps(
        math.sqrt(max(0.0, MU_G2 - domain_margin)), math.inf, _MOTOR_CERT_ULPS
    )
    if q_hi < q_target_lo:
        return False
    if q_lo > q_target_hi:
        return True
    return None


def _motor_speed_integral(v: float, v0: float) -> float | None:
    """Integral of ``v/(A-Bv)`` from ``v0`` to ``v``, stably evaluated."""
    g = MOTOR_SWITCH_GEOMETRY
    if not (0.0 <= v0 <= v < g.v_eq):
        return 0.0 if v == v0 else None
    gap0 = math.fma(-g.B, v0, g.A)
    if not gap0 > 0.0:
        return None
    x = g.B * (v - v0) / gap0
    if not (0.0 <= x < 1.0):
        return None
    if x < 1.0e-4:
        # -log(1-x)-x = sum_{n>=2} x^n/n.
        term = x * x
        rem = 0.5 * term
        for n in range(3, 9):
            term *= x
            rem += term / n
    else:
        rem = -math.log1p(-x) - x
    return (g.A * rem + x * g.B * v0) / (g.B * g.B)


def _motor_J(v: float) -> float:
    return -_motor_H(v)


def _motor_rising_barrier_min_speed(v0: float, v_hi: float, slope: float) -> float | None:
    """Speed where the increasing-|k| no-switch barrier is smallest."""
    g = MOTOR_SWITCH_GEOMETRY
    if not g.rising_barrier_certified:
        return None
    lo = max(v0, g.v_min)
    hi = min(v_hi, g.v_R)
    if not lo <= hi:
        return None
    S = abs(float(slope))
    if S == 0.0 or lo == hi:
        return hi
    # J decreases strictly from +infinity at v_min to zero at v_R.
    left = max(lo, math.nextafter(g.v_min, g.v_R))
    jl = _motor_J(left)
    jh = _motor_J(hi)
    if not (math.isfinite(jl) and math.isfinite(jh)):
        return None
    if S >= jl:
        return left
    if S <= jh:
        return hi
    return _bisect_scalar_level(_motor_J, left, hi, S, increasing=False)


def _motor_increasing_low_speed_no_switch(seg: GripEvalSegment, L: float) -> bool:
    """Integrated lower-r barrier for the rising branch of R(v)."""
    g = MOTOR_SWITCH_GEOMETRY
    if not g.rising_barrier_certified:
        return False
    v0 = math.sqrt(max(0.0, seg.w0))
    if v0 >= g.v_R or v0 >= g.v_eq:
        return False
    w_up = _motor_upper_w(seg.w0, L)
    if w_up is None or not math.isfinite(w_up) or w_up < seg.w0:
        return False
    v_hi = min(math.sqrt(max(0.0, w_up)), g.v_R)
    if v_hi <= g.v_min:
        return True
    v_test = _motor_rising_barrier_min_speed(v0, v_hi, seg.sigma)
    if v_test is None:
        return False
    integral = _motor_speed_integral(v_test, v0)
    if integral is None:
        return False
    r_lower = math.fma(abs(seg.sigma), integral, abs(seg.k0))
    R_test = _motor_R(v_test)
    if not (math.isfinite(r_lower) and math.isfinite(R_test)):
        return False
    return _shift_ulps(r_lower, -math.inf, _MOTOR_CERT_ULPS) > _shift_ulps(
        R_test, math.inf, _MOTOR_CERT_ULPS
    )


def _grip_motor_structural_increasing_noevent(
    seg: Any,
    L: float,
    *,
    domain_margin: float,
    profiler: _Profiler | None,
    x_abs_tol: float,
    x_rel_tol: float,
    f_tol: float,
    allow_initial_boundary: bool,
    initial_spatial_tol: float | None,
) -> ScanResult | None:
    """Zero-propagation no-event certificate for increasing |k| at high speed."""
    g = MOTOR_SWITCH_GEOMETRY
    if not g.orientation_certified or not isinstance(seg, GripEvalSegment):
        return None
    if seg.sigma == 0.0 or seg.k0 == 0.0 or L < 0.0 or not math.isfinite(L):
        return None
    k1 = math.fma(seg.sigma, L, seg.k0)
    if k1 == 0.0 or math.copysign(1.0, k1) != math.copysign(1.0, seg.k0):
        return None
    if abs(k1) <= abs(seg.k0):
        return None

    f0, df0, g2_0 = grip_motor_f_df_g2(seg, 0.0, profiler)
    if not math.isfinite(g2_0):
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    if g2_0 <= domain_margin and not allow_initial_boundary:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    else:
        if not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
            raise ValueError("initial_spatial_tol must be finite and nonnegative when supplied")
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)
    if _initial_switch_required(
        f0, df0, NEG_TO_POS, residual_tol=f_tol, spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(0.0, None, (0.0, 0.0, f0, f0), None, True)
    if L == 0.0:
        return ScanResult(None, None)

    v0 = math.sqrt(max(0.0, seg.w0))
    high_speed = v0 >= _shift_ulps(g.v_R, math.inf, _MOTOR_CERT_ULPS)
    low_speed_cert = False
    if not high_speed:
        low_speed_cert = _motor_increasing_low_speed_no_switch(seg, L)
        if not low_speed_cert:
            return None
    # High speed: R decreases while r and v increase.  Low speed: the
    # integrated r(v) barrier stays strictly above the rising R(v) surface.
    # In either case MOTOR contact is excluded over the horizon.
    domain = _grip_increasing_domain_certificate(seg, L, domain_margin=domain_margin)
    if domain is False:
        return ScanResult(None, None)
    if domain is True and high_speed:
        # This domain-special path has been separately qualified against the
        # legacy endpoint authority.  Do not extend it merely because the
        # low-speed MOTOR barrier excludes switching: low-speed domain winners
        # can sit on the historical event/state reconciliation seam.
        return ScanResult(None, L, domain_switch_excluded=True)
    return None


def grip_motor_earliest_scan(
    seg: Any,
    L: float | None = None,
    *,
    n_scan: int = 256,
    domain_stop_margin: float = 4e-12,
    physical_domain_margin: float = 1e-12,
    domain_safe_floor: float = 0.0,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    max_iter: int = 64,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    scan_L = seg.L if L is None else L
    structural = _grip_motor_structural_decreasing_scan(
        seg, scan_L, domain_margin=physical_domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol, max_iter=max_iter,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )
    if structural is not None:
        return structural
    structural = _grip_motor_structural_increasing_noevent(
        seg, scan_L, domain_margin=physical_domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )
    if structural is not None:
        return structural
    return scan_first_event_or_domain_safe(
        seg, scan_L, f_df_g2=grip_motor_f_df_g2, direction=NEG_TO_POS,
        n_scan=n_scan, domain_stop_margin=domain_stop_margin,
        physical_domain_margin=physical_domain_margin,
        domain_safe_floor=domain_safe_floor, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol,
        max_iter=max_iter, allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )

def _grip_brake_structural_increasing_domain_scan(
    seg: Any,
    L: float,
    *,
    domain_margin: float,
    profiler: _Profiler | None,
    x_abs_tol: float,
    x_rel_tol: float,
    f_tol: float,
    allow_initial_boundary: bool,
    initial_spatial_tol: float | None,
) -> ScanResult | None:
    """Increasing-|k| BRAKE impossibility plus cheap domain classification."""
    if not isinstance(seg, GripEvalSegment) or seg.sigma == 0.0 or seg.k0 == 0.0:
        return None
    k1 = math.fma(seg.sigma, L, seg.k0)
    if k1 == 0.0 or math.copysign(1.0, k1) != math.copysign(1.0, seg.k0):
        return None
    if abs(k1) <= abs(seg.k0):
        return None
    f0, df0, g2_0 = grip_brake_f_df_g2(seg, 0.0, profiler)
    if not math.isfinite(g2_0):
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    if g2_0 <= domain_margin and not allow_initial_boundary:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    else:
        if not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
            raise ValueError("initial_spatial_tol must be finite and nonnegative when supplied")
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)
    if _initial_switch_required(
        f0, df0, NEG_TO_POS, residual_tol=f_tol, spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(0.0, None, (0.0, 0.0, f0, f0), None, True)
    # z strictly increases, so G strictly decreases: GRIP->BRAKE (which needs
    # G to rise through A_BRAKE) cannot occur after this valid start.
    domain = _grip_increasing_domain_certificate(seg, L, domain_margin=domain_margin)
    if domain is False:
        return ScanResult(None, None)
    if domain is True:
        return ScanResult(None, L, domain_switch_excluded=True)
    return None


def grip_brake_earliest_scan(
    seg: Any,
    L: float | None = None,
    *,
    n_scan: int = 256,
    domain_stop_margin: float = 4e-12,
    physical_domain_margin: float = 1e-12,
    domain_safe_floor: float = 0.0,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    max_iter: int = 64,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    """Find GRIP->BRAKE with a theorem-backed decreasing-|k| fast path.

    When ``|k|`` decreases (including a curvature-zero crossing), the
    normalized load ``z=|k|w/MU_G`` has at most one interior maximum.  A
    physical friction terminal is therefore impossible after a valid start,
    and the BRAKE surface can have at most one future recontact.  Analytic
    acceleration envelopes classify most horizons and bracket every admitted
    event.  Other curvature branches retain the legacy shared scanner.
    """
    scan_L = seg.L if L is None else L
    structural = _grip_brake_structural_scan(
        seg,
        scan_L,
        domain_margin=physical_domain_margin,
        profiler=profiler,
        x_abs_tol=x_abs_tol,
        x_rel_tol=x_rel_tol,
        f_tol=f_tol,
        max_iter=max_iter,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
        n_scan=n_scan,
    )
    if structural is not None:
        return structural
    structural = _grip_brake_structural_increasing_domain_scan(
        seg, scan_L, domain_margin=physical_domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )
    if structural is not None:
        return structural
    return scan_first_event_or_domain_safe(
        seg,
        scan_L,
        f_df_g2=grip_brake_f_df_g2,
        direction=NEG_TO_POS,
        n_scan=n_scan,
        domain_stop_margin=domain_stop_margin,
        physical_domain_margin=physical_domain_margin,
        domain_safe_floor=domain_safe_floor,
        profiler=profiler,
        x_abs_tol=x_abs_tol,
        x_rel_tol=x_rel_tol,
        f_tol=f_tol,
        max_iter=max_iter,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )

def _scan_wrapper(
    seg: Any,
    L: float | None,
    *,
    evaluator: EventEvaluator,
    direction: float,
    n_scan: int,
    domain_margin: float,
    profiler: _Profiler | None,
    x_abs_tol: float,
    x_rel_tol: float,
    f_tol: float,
    allow_initial_boundary: bool,
    initial_spatial_tol: float | None,
) -> ScanResult:
    scan_L = seg.L if L is None else L
    return scan_first_event_and_domain(
        seg,
        scan_L,
        f_df_g2=evaluator,
        direction=direction,
        n_scan=n_scan,
        domain_margin=domain_margin,
        profiler=profiler,
        x_abs_tol=x_abs_tol,
        x_rel_tol=x_rel_tol,
        f_tol=f_tol,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )


def motor_grip_scan(
    seg: Any,
    L: float | None = None,
    *,
    n_scan: int = 256,
    domain_margin: float = 1e-12,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    return _scan_wrapper(
        seg, L, evaluator=motor_grip_f_df_g2, direction=POS_TO_NEG,
        n_scan=n_scan, domain_margin=domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )


def grip_motor_scan(
    seg: Any,
    L: float | None = None,
    *,
    n_scan: int = 256,
    domain_margin: float = 1e-12,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    scan_L = seg.L if L is None else L
    structural = _grip_motor_structural_decreasing_scan(
        seg, scan_L, domain_margin=domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol, max_iter=64,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )
    if structural is not None:
        return structural
    return _scan_wrapper(
        seg, scan_L, evaluator=grip_motor_f_df_g2, direction=NEG_TO_POS,
        n_scan=n_scan, domain_margin=domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )


def grip_brake_scan(
    seg: Any,
    L: float | None = None,
    *,
    n_scan: int = 256,
    domain_margin: float = 1e-12,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    allow_initial_boundary: bool = False,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    scan_L = seg.L if L is None else L
    structural = _grip_brake_structural_scan(
        seg,
        scan_L,
        domain_margin=domain_margin,
        profiler=profiler,
        x_abs_tol=x_abs_tol,
        x_rel_tol=x_rel_tol,
        f_tol=f_tol,
        max_iter=64,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
        n_scan=n_scan,
    )
    if structural is not None:
        return structural
    return _scan_wrapper(
        seg, scan_L, evaluator=grip_brake_f_df_g2, direction=NEG_TO_POS,
        n_scan=n_scan, domain_margin=domain_margin, profiler=profiler,
        x_abs_tol=x_abs_tol, x_rel_tol=x_rel_tol, f_tol=f_tol,
        allow_initial_boundary=allow_initial_boundary,
        initial_spatial_tol=initial_spatial_tol,
    )



def _quadratic_roots_for_target(
    a: float,
    b: float,
    c0: float,
    target: float,
    L: float,
) -> list[float]:
    """Return finite roots of ``a*s²+b*s+c0 == target`` in ``[0,L]``."""
    c = c0 - target
    if a == 0.0:
        if b == 0.0:
            return []
        root = -c / b
        return [root] if 0.0 <= root <= L and math.isfinite(root) else []

    disc = math.fma(-4.0 * a, c, b * b)
    if disc < 0.0 or not math.isfinite(disc):
        return []
    sqrt_disc = math.sqrt(max(0.0, disc))
    if sqrt_disc == 0.0:
        roots = [-0.5 * b / a]
    else:
        q = -0.5 * (b + math.copysign(sqrt_disc, b))
        roots = [q / a]
        if q != 0.0:
            roots.append(c / q)

    roots = [x for x in roots if math.isfinite(x) and 0.0 <= x <= L]
    roots.sort()
    if len(roots) == 2 and roots[0] == roots[1]:
        roots.pop()
    return roots


def _grip_brake_decreasing_horizon(seg: Any, L: float) -> tuple[float, bool] | None:
    """Return the decreasing-|k| horizon and whether it reaches ``k=0``.

    The structural fast path is intentionally limited to the general Cflow
    GRIP segment.  Circular/straight GRIP and test doubles keep the legacy
    path, preserving their existing numerical authority.
    """
    if not isinstance(seg, GripEvalSegment) or seg.sigma == 0.0 or seg.k0 == 0.0:
        return None

    k1 = math.fma(seg.sigma, L, seg.k0)
    if k1 == 0.0 or math.copysign(1.0, k1) != math.copysign(1.0, seg.k0):
        zero = abs(seg.k0) / abs(seg.sigma)
        if not math.isfinite(zero) or zero <= 0.0:
            return None
        return min(L, zero), True

    if abs(k1) >= abs(seg.k0):
        return None
    return L, False


def _grip_brake_gmin_lower(seg: GripEvalSegment, g2_0: float) -> float | None:
    """Conservative lower bound for ``G`` on a decreasing-|k| GRIP branch.

    For ``r=|k|`` and ``S=|sigma|`` the nullcline of

        z' = -(S/r) z + 2 r sqrt(1-z^2)

    is ``z*=2r^2/hypot(S,2r^2)``.  Since that nullcline decreases with ``r``,
    ``z`` has at most one interior maximum.  At the nullcline
    ``G*=MU_G*S/hypot(S,2r^2)``, so ``min(G0,G*(r0))`` is a global lower bound
    for ``G`` until ``k=0``.  The returned value is shifted downward before it
    is used by a hard certificate.
    """
    if not math.isfinite(g2_0) or g2_0 < 0.0:
        return None
    r0 = abs(seg.k0)
    slope = abs(seg.sigma)
    if r0 == 0.0 or slope == 0.0:
        return None
    g0 = math.sqrt(max(0.0, g2_0))
    r2 = r0 * r0
    denom = math.hypot(slope, 2.0 * r2)
    if denom == 0.0 or not math.isfinite(denom):
        return None
    g_star = MU_G * slope / denom
    if not math.isfinite(g_star):
        return None
    return max(0.0, _shift_ulps(min(g0, g_star), -math.inf))


def _grip_brake_endpoint_q_bounds(
    seg: GripEvalSegment,
    h: float,
    g_min: float,
) -> tuple[float, float]:
    """Outward-rounded bounds for ``|k(h)|*w(h)`` before a BRAKE event."""
    r = abs(math.fma(seg.sigma, h, seg.k0))
    r_lo = max(0.0, _shift_ulps(r, -math.inf))
    r_hi = _shift_ulps(r, math.inf)

    w_lo = math.fma(2.0 * g_min, h, seg.w0)
    w_hi = math.fma(2.0 * _shift_ulps(A_BRAKE, math.inf), h, seg.w0)
    w_lo = max(0.0, _shift_ulps(w_lo, -math.inf))
    w_hi = _shift_ulps(w_hi, math.inf)

    q_lo = _shift_ulps(r_lo * w_lo, -math.inf)
    q_hi = _shift_ulps(r_hi * w_hi, math.inf)
    return q_lo, q_hi


def _grip_brake_later_envelope_root(
    seg: GripEvalSegment,
    h: float,
    *,
    acceleration: float,
    target: float,
    lower_bound: bool,
) -> float | None:
    """Return the later quadratic-envelope root, rounded out of the bracket."""
    r0 = abs(seg.k0)
    slope = abs(seg.sigma)
    a = -2.0 * slope * acceleration
    b = math.fma(2.0 * r0, acceleration, -slope * seg.w0)
    c0 = r0 * seg.w0
    roots = _quadratic_roots_for_target(a, b, c0, target, h)
    if not roots:
        return None
    root = roots[-1]
    root = _shift_ulps(root, -math.inf if lower_bound else math.inf)
    return min(h, max(0.0, root))


def _grip_brake_structural_scan(
    seg: Any,
    L: float,
    *,
    domain_margin: float,
    profiler: _Profiler | None,
    x_abs_tol: float,
    x_rel_tol: float,
    f_tol: float,
    max_iter: int,
    allow_initial_boundary: bool,
    initial_spatial_tol: float | None,
    n_scan: int,
) -> ScanResult | None:
    """Structural GRIP->BRAKE search for decreasing ``|k|``.

    ``None`` means fail closed to the legacy uniform scanner.  A ``ScanResult``
    is returned only after the decreasing-|k| theorem applies and every hard
    binary64 decision has a conservative guard.  Event brackets are validated
    by the original Cflow residual before the original safeguarded root solver
    is invoked.
    """
    if L < 0.0 or not math.isfinite(L):
        raise ValueError(f"scan length must be finite and nonnegative, got {L!r}")
    if n_scan <= 0:
        raise ValueError("n_scan must be positive")
    if not (0.0 <= domain_margin < MU_G2):
        raise ValueError(f"invalid domain margin: {domain_margin!r}")

    branch = _grip_brake_decreasing_horizon(seg, L)
    if branch is None:
        return None

    f0, df0, g2_0 = grip_brake_f_df_g2(seg, 0.0, profiler)
    if not math.isfinite(g2_0):
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    if g2_0 <= domain_margin and not allow_initial_boundary:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))

    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    elif not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
        raise ValueError(
            "initial_spatial_tol must be finite and nonnegative when supplied"
        )
    else:
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)

    if _initial_switch_required(
        f0,
        df0,
        NEG_TO_POS,
        residual_tol=f_tol,
        spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(0.0, None, (0.0, 0.0, f0, f0), None, True)
    if L == 0.0:
        return ScanResult(None, None)

    h, crosses_zero = branch
    g2_for_bound = (
        0.0
        if allow_initial_boundary and g2_0 <= domain_margin
        else g2_0
    )
    g_min = _grip_brake_gmin_lower(seg, g2_for_bound)
    if g_min is None:
        return None

    # On a decreasing-|k| branch the one-maximum theorem prevents a future
    # physical friction terminal.  At k=0 the BRAKE residual is MU_G-A_BRAKE>0,
    # so a cross-zero horizon guarantees recontact.  Otherwise the two
    # acceleration envelopes classify the horizon whenever they separate it
    # from the threshold by the outward-rounded moat.
    q_lower_h, q_upper_h = _grip_brake_endpoint_q_bounds(seg, h, g_min)
    if not crosses_zero and q_lower_h > K_BRAKE_HI:
        return ScanResult(None, None)

    event_guaranteed = crosses_zero or q_upper_h < K_BRAKE_LO
    f_h: float | None = None
    if not event_guaranteed:
        try:
            f_h, _df_h, _g2_h = grip_brake_f_df_g2(seg, h, profiler)
        except Exception:
            return None
        if f_h < 0.0:
            return ScanResult(None, None)

    lo = _grip_brake_later_envelope_root(
        seg,
        h,
        acceleration=g_min,
        target=K_BRAKE_HI,
        lower_bound=True,
    )
    hi = _grip_brake_later_envelope_root(
        seg,
        h,
        acceleration=_shift_ulps(A_BRAKE, math.inf),
        target=K_BRAKE_LO,
        lower_bound=False,
    )
    if lo is None or hi is None or hi < lo:
        return None

    try:
        f_lo, _df_lo, _g2_lo = grip_brake_f_df_g2(seg, lo, profiler)
        f_hi, _df_hi, _g2_hi = grip_brake_f_df_g2(seg, hi, profiler)
    except Exception:
        return None

    # A start exactly on the BRAKE surface can legitimately depart into GRIP
    # before recontacting later.  Never let the trivial zero at s=0 collapse
    # that second-root topology.
    surface_departure = (
        abs(f0) <= f_tol
        and df0 is not None
        and math.isfinite(df0)
        and df0 < 0.0
    )
    if lo <= initial_spatial_tol and abs(f_lo) <= f_tol and surface_departure:
        interior = min(hi, max(initial_spatial_tol, 0.125 * hi))
        if not (0.0 < interior < hi):
            return None
        try:
            f_interior, _df_i, _g2_i = grip_brake_f_df_g2(seg, interior, profiler)
        except Exception:
            return None
        if f_interior >= 0.0:
            return None
        lo, f_lo = interior, f_interior

    # The analytic roots are only certificates/brackets; Cflow remains event
    # authority.  Any sign disagreement fails closed to the legacy scanner.
    if f_lo > 0.0 or f_hi < 0.0:
        return None
    if f_lo == 0.0:
        return ScanResult(lo, None, (lo, lo, f_lo, f_lo), None)
    if f_hi == 0.0:
        return ScanResult(hi, None, (hi, hi, f_hi, f_hi), None)

    # One cache-aware interior estimate narrows the bracket without creating a
    # fixed scan grid.  It is the later root of the midpoint-acceleration
    # envelope; if rounding places it outside the certified bracket, use the
    # geometric midpoint instead.
    g_guess = 0.5 * (g_min + A_BRAKE)
    guess = _grip_brake_later_envelope_root(
        seg,
        h,
        acceleration=g_guess,
        target=K_BRAKE,
        lower_bound=False,
    )
    if guess is None or not (lo < guess < hi):
        guess = 0.5 * (lo + hi)
    try:
        f_guess, _df_guess, _g2_guess = grip_brake_f_df_g2(seg, guess, profiler)
    except Exception:
        return None

    if f_guess >= 0.0:
        root_lo, root_hi = lo, guess
        root_f_lo, root_f_hi = f_lo, f_guess
    else:
        root_lo, root_hi = guess, hi
        root_f_lo, root_f_hi = f_guess, f_hi

    def f_df(x: float) -> tuple[float, float | None]:
        fx, dfx, _g2 = grip_brake_f_df_g2(seg, x, None)
        return fx, dfx

    root = bracketed_newton(
        f_df,
        root_lo,
        root_hi,
        f_lo=root_f_lo,
        f_hi=root_f_hi,
        x_abs_tol=x_abs_tol,
        x_rel_tol=x_rel_tol,
        f_tol=f_tol,
        max_iter=max_iter,
    )
    if root is None:
        return None
    return ScanResult(
        root,
        None,
        (root_lo, root_hi, root_f_lo, root_f_hi),
        None,
    )


def _first_brake_threshold_crossing(
    seg: Any,
    L: float,
    *,
    threshold: float,
    event: bool,
    residual_tol: float,
) -> float | None:
    rate = S_BRAKE
    a = rate * seg.sigma
    b = math.fma(seg.w0, seg.sigma, rate * seg.k0)
    c0 = seg.w0 * seg.k0
    candidates: list[float] = []

    for target in (threshold, -threshold):
        for root in _quadratic_roots_for_target(a, b, c0, target, L):
            w = seg.w(root)
            k = math.fma(seg.sigma, root, seg.k0)
            q = w * k
            q_prime = math.fma(k, rate, seg.sigma * w)
            if event:
                g2 = math.fma(-q, q, MU_G2)
                if g2 <= 0.0:
                    continue
                g = math.sqrt(g2)
                residual = g - A_BRAKE
                deriv = -q * q_prime / g
                if deriv < 0.0 and abs(residual) <= residual_tol:
                    candidates.append(root)
            else:
                # G² decreases through the domain threshold only when
                # -2*q*q' < 0.
                if -2.0 * q * q_prime < 0.0:
                    candidates.append(root)

    return min(candidates) if candidates else None

def brake_grip_scan(
    seg: Any,
    L: float | None = None,
    *,
    n_scan: int = 256,
    domain_margin: float = 1e-12,
    profiler: _Profiler | None = None,
    x_abs_tol: float = 1e-13,
    x_rel_tol: float = 1e-13,
    f_tol: float = 1e-13,
    initial_spatial_tol: float | None = None,
) -> ScanResult:
    """Analytic combined brake→grip/domain search.

    For a brake segment, ``w`` and ``k`` are affine, so ``q=w*k`` is
    quadratic.  Both the switching surface ``|q|=K_BRAKE`` and friction
    edge ``|q|=sqrt(MU_G²-margin)`` are therefore solved from the same
    quadratic without a uniform scan.  ``n_scan`` and root-x tolerances are
    accepted for API compatibility.
    """
    del n_scan
    scan_L = seg.L if L is None else L
    if scan_L < 0.0 or not math.isfinite(scan_L):
        raise ValueError(f"scan length must be finite and nonnegative, got {scan_L!r}")
    if not (0.0 <= domain_margin < MU_G2):
        raise ValueError(f"invalid domain margin: {domain_margin!r}")

    residual0, deriv0, g2_0 = brake_grip_f_df_g2(seg, 0.0, profiler)
    if not math.isfinite(g2_0) or g2_0 <= domain_margin:
        return ScanResult(None, 0.0, None, (0.0, 0.0, g2_0, g2_0))
    locator_spatial_tol = x_abs_tol + x_rel_tol * max(1.0, abs(scan_L))
    if initial_spatial_tol is None:
        initial_spatial_tol = locator_spatial_tol
    elif not math.isfinite(initial_spatial_tol) or initial_spatial_tol < 0.0:
        raise ValueError(
            "initial_spatial_tol must be finite and nonnegative when supplied"
        )
    else:
        initial_spatial_tol = max(locator_spatial_tol, initial_spatial_tol)
    if _initial_switch_required(
        residual0,
        deriv0,
        POS_TO_NEG,
        residual_tol=f_tol,
        spatial_tol=initial_spatial_tol,
    ):
        return ScanResult(
            0.0, None, (0.0, 0.0, residual0, residual0), None, True
        )

    if scan_L == 0.0:
        return ScanResult(None, None)

    event_root = _first_brake_threshold_crossing(
        seg,
        scan_L,
        threshold=K_BRAKE,
        event=True,
        residual_tol=max(1e-8, 32.0 * f_tol),
    )
    domain_threshold = math.sqrt(MU_G2 - domain_margin)
    domain_edge = _first_brake_threshold_crossing(
        seg,
        scan_L,
        threshold=domain_threshold,
        event=False,
        residual_tol=0.0,
    )

    if domain_edge is not None and (
        event_root is None or domain_edge <= event_root
    ):
        return ScanResult(None, domain_edge)
    return ScanResult(event_root, domain_edge)



# Compatibility wrappers: callers that only need the event location can retain
# the old function names.  They now use the shared scan and the new segment API.
def motor_grip_cross(seg: Any, n_scan: int = 256, **kwargs: Any) -> float | None:
    return motor_grip_scan(seg, n_scan=n_scan, **kwargs).event


def grip_motor_cross(seg: Any, n_scan: int = 256, **kwargs: Any) -> float | None:
    return grip_motor_scan(seg, n_scan=n_scan, **kwargs).event


def grip_brake_cross(seg: Any, n_scan: int = 256, **kwargs: Any) -> float | None:
    return grip_brake_scan(seg, n_scan=n_scan, **kwargs).event


def brake_grip_cross(seg: Any, n_scan: int = 256, **kwargs: Any) -> float | None:
    return brake_grip_scan(seg, n_scan=n_scan, **kwargs).event


__all__ = [
    "NEG_TO_POS",
    "POS_TO_NEG",
    "ScanResult",
    "bracketed_newton",
    "scan_first_event_and_domain",
    "motor_grip_f_df_g2",
    "grip_motor_f_df_g2",
    "grip_brake_f_df_g2",
    "brake_grip_f_df_g2",
    "motor_grip_scan",
    "grip_motor_scan",
    "grip_brake_scan",
    "brake_grip_scan",
    "motor_grip_cross",
    "grip_motor_cross",
    "grip_brake_cross",
    "brake_grip_cross",
]
