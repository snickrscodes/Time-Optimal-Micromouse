from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence, TypeAlias
from .geometry import create_geo, fresnel

Float2: TypeAlias = tuple[float, float]
Float4: TypeAlias = tuple[float, float, float, float]

_PI = math.pi
_SQRT_PI = math.sqrt(math.pi)

# 16-point Gauss-Legendre rule on [-1, 1].  Used only as a guarded fallback
# when the algebraic Fresnel moment recurrences are ill-conditioned.
_GL16_X = (
    -0.989400934991649932596154173450,
    -0.944575023073232576077988415535,
    -0.865631202387831743880467897712,
    -0.755404408355003033895101194847,
    -0.617876244402643748446671764049,
    -0.458016777657227386342419442984,
    -0.281603550779258913230460501460,
    -0.095012509837637440185319335425,
     0.095012509837637440185319335425,
     0.281603550779258913230460501460,
     0.458016777657227386342419442984,
     0.617876244402643748446671764049,
     0.755404408355003033895101194847,
     0.865631202387831743880467897712,
     0.944575023073232576077988415535,
     0.989400934991649932596154173450,
)
_GL16_W = (
    0.027152459411754094851780572456,
    0.062253523938647892862843836994,
    0.095158511682492784809925107602,
    0.124628971255533872052476282192,
    0.149595988816576732081501730547,
    0.169156519395002538189312079030,
    0.182603415044923588866763667969,
    0.189450610455068496285396723208,
    0.189450610455068496285396723208,
    0.182603415044923588866763667969,
    0.169156519395002538189312079030,
    0.149595988816576732081501730547,
    0.124628971255533872052476282192,
    0.095158511682492784809925107602,
    0.062253523938647892862843836994,
    0.027152459411754094851780572456,
)

# Tuning constants.  The series branch removes all removable singularities;
# the Fresnel branch is the normal large-parameter path; quadrature is rare.
_SERIES_PHASE_BOUND = 0.25
_SERIES_MAX_TERMS = 80
_SERIES_REL_TOL = 2.0e-18
_FRESNEL_RECURRENCE_COND_MAX = 10.0
_QUAD_PHASE_PER_PANEL = 0.75
_QUAD_MAX_PANELS = 4096


class GeometryState(NamedTuple):
    x: float
    y: float
    theta: float
    k: float


@dataclass(frozen=True, slots=True)
class MomentResult:
    """Path moments represented as paired real x/y channels.

    z0 = integral exp(i theta(s)) ds
    z1 = integral s exp(i theta(s)) ds
    z2 = integral s^2 exp(i theta(s)) ds
    """

    z0_re: float
    z0_im: float
    z1_re: float
    z1_im: float
    z2_re: float
    z2_im: float
    theta1: float
    k1: float


@dataclass(slots=True)
class GeometryPath:
    """Compiled geometry tape for flat [L0, sigma0, ...] parameters.

    Raw gradient layout is [d/dL0, d/dsigma0, d/dL1, d/dsigma1, ...].
    The initial state is treated as fixed by the convenience methods, while
    VJP methods optionally return its adjoint for composition with another tape.
    """

    lengths: list[float]
    sigmas: list[float]
    xs: list[float]
    ys: list[float]
    thetas: list[float]
    curvatures: list[float]
    dx: list[float]
    dy: list[float]
    m1x: list[float]
    m1y: list[float]
    m2x: list[float]
    m2y: list[float]
    cos_end: list[float]
    sin_end: list[float]

    @property
    def n_segments(self) -> int:
        return len(self.lengths)

    @property
    def initial_state(self) -> GeometryState:
        return GeometryState(self.xs[0], self.ys[0], self.thetas[0], self.curvatures[0])

    @property
    def final_state(self) -> GeometryState:
        return GeometryState(self.xs[-1], self.ys[-1], self.thetas[-1], self.curvatures[-1])

    def state_at_fraction(self, segment_index: int, tau: float) -> GeometryState:
        i = _checked_segment_index(self, segment_index)
        tau = _checked_fraction(tau)
        if tau == 0.0:
            return GeometryState(self.xs[i], self.ys[i], self.thetas[i], self.curvatures[i])
        if tau == 1.0:
            return GeometryState(self.xs[i + 1], self.ys[i + 1], self.thetas[i + 1], self.curvatures[i + 1])
        d = tau * self.lengths[i]
        m = clothoid_moments(self.thetas[i], self.curvatures[i], self.sigmas[i], d)
        return GeometryState(
            self.xs[i] + m.z0_re,
            self.ys[i] + m.z0_im,
            m.theta1,
            m.k1,
        )

    def endpoint_vjp(
        self,
        seed: Float4,
        *,
        return_initial_adjoint: bool = False,
    ) -> list[float] | tuple[list[float], Float4]:
        """Reverse a scalar seed on final (x, y, theta, k)."""
        ax, ay, atheta, ak = map(float, seed)
        grad = [0.0] * (2 * self.n_segments)
        for i in range(self.n_segments - 1, -1, -1):
            atheta, ak = self._reverse_full_segment(i, ax, ay, atheta, ak, grad)
        if return_initial_adjoint:
            return grad, (ax, ay, atheta, ak)
        return grad

    def endpoint_jacobian(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """Return the four final-state Jacobian rows in raw (L, sigma) basis."""
        return (
            self.endpoint_vjp((1.0, 0.0, 0.0, 0.0)),
            self.endpoint_vjp((0.0, 1.0, 0.0, 0.0)),
            self.endpoint_vjp((0.0, 0.0, 1.0, 0.0)),
            self.endpoint_vjp((0.0, 0.0, 0.0, 1.0)),
        )

    def point_pose_vjp(
        self,
        segment_index: int,
        tau: float,
        seed: Float4,
        *,
        return_initial_adjoint: bool = False,
    ) -> tuple[GeometryState, list[float]] | tuple[GeometryState, list[float], Float4]:
        """Value and VJP at a fixed normalized segment location tau.

        tau is held fixed, so local distance d=tau*L moves with segment length.
        This is the correct primitive for generated/collocation constraints.
        """
        i = _checked_segment_index(self, segment_index)
        tau = _checked_fraction(tau)
        state, context = self._point_context(i, tau)
        grad, initial_adj = self._point_vjp_from_context(i, tau, context, seed)
        if return_initial_adjoint:
            return state, grad, initial_adj
        return state, grad

    def point_xy_vjp(
        self,
        segment_index: int,
        tau: float,
        seed_xy: Float2,
        *,
        return_initial_adjoint: bool = False,
    ):
        return self.point_pose_vjp(
            segment_index,
            tau,
            (float(seed_xy[0]), float(seed_xy[1]), 0.0, 0.0),
            return_initial_adjoint=return_initial_adjoint,
        )

    def point_xy_jacobian(self, segment_index: int, tau: float) -> tuple[Float2, list[float], list[float]]:
        """Return point value and the x/y Jacobian rows.

        The local Fresnel/moment evaluation is shared by both rows.
        """
        i = _checked_segment_index(self, segment_index)
        tau = _checked_fraction(tau)
        state, context = self._point_context(i, tau)
        gx, _ = self._point_vjp_from_context(i, tau, context, (1.0, 0.0, 0.0, 0.0))
        gy, _ = self._point_vjp_from_context(i, tau, context, (0.0, 1.0, 0.0, 0.0))
        return (state.x, state.y), gx, gy

    def halfspace_value_and_grad(
        self,
        segment_index: int,
        tau: float,
        normal: Float2,
        offset: float,
    ) -> tuple[float, list[float]]:
        i = _checked_segment_index(self, segment_index)
        tau = _checked_fraction(tau)
        nx = float(normal[0])
        ny = float(normal[1])
        state, context = self._point_context(i, tau)
        grad, _ = self._point_vjp_from_context(i, tau, context, (nx, ny, 0.0, 0.0))
        value = math.fma(nx, state.x, math.fma(ny, state.y, -float(offset)))
        return value, grad

    def _point_context(
        self, i: int, tau: float
    ) -> tuple[GeometryState, tuple[float, MomentResult] | None]:
        if tau == 0.0:
            return GeometryState(self.xs[i], self.ys[i], self.thetas[i], self.curvatures[i]), None
        if tau == 1.0:
            return GeometryState(self.xs[i + 1], self.ys[i + 1], self.thetas[i + 1], self.curvatures[i + 1]), None
        d = tau * self.lengths[i]
        m = clothoid_moments(self.thetas[i], self.curvatures[i], self.sigmas[i], d)
        return (
            GeometryState(self.xs[i] + m.z0_re, self.ys[i] + m.z0_im, m.theta1, m.k1),
            (d, m),
        )

    def _point_vjp_from_context(
        self,
        i: int,
        tau: float,
        context: tuple[float, MomentResult] | None,
        seed: Float4,
    ) -> tuple[list[float], Float4]:
        ax, ay, atheta, ak = map(float, seed)
        grad = [0.0] * (2 * self.n_segments)

        if tau == 0.0:
            i -= 1
        elif tau == 1.0:
            atheta, ak = self._reverse_full_segment(i, ax, ay, atheta, ak, grad)
            i -= 1
        else:
            if context is None:
                raise AssertionError("interior point missing local moment context")
            d, m = context
            sigma = self.sigmas[i]
            p_t = math.fma(ax, math.cos(m.theta1), ay * math.sin(m.theta1))
            p_r0 = math.fma(-ax, m.z0_im, ay * m.z0_re)
            p_r1 = math.fma(-ax, m.z1_im, ay * m.z1_re)
            p_r2 = math.fma(-ax, m.z2_im, ay * m.z2_re)
            incoming_atheta = atheta

            grad[2 * i] = tau * math.fma(incoming_atheta, m.k1, math.fma(ak, sigma, p_t))
            grad[2 * i + 1] = math.fma(0.5, p_r2, math.fma(0.5 * incoming_atheta, d * d, ak * d))
            atheta = incoming_atheta + p_r0
            ak = math.fma(incoming_atheta, d, ak + p_r1)
            i -= 1

        for j in range(i, -1, -1):
            atheta, ak = self._reverse_full_segment(j, ax, ay, atheta, ak, grad)
        return grad, (ax, ay, atheta, ak)

    def _reverse_full_segment(
        self,
        i: int,
        ax: float,
        ay: float,
        atheta: float,
        ak: float,
        grad: list[float],
    ) -> tuple[float, float]:
        # Position VJP terms. Multiplication by +i is a +90 degree rotation.
        p_t = math.fma(ax, self.cos_end[i], ay * self.sin_end[i])
        p_r0 = math.fma(-ax, self.dy[i], ay * self.dx[i])
        p_r1 = math.fma(-ax, self.m1y[i], ay * self.m1x[i])
        p_r2 = math.fma(-ax, self.m2y[i], ay * self.m2x[i])

        L = self.lengths[i]
        sigma = self.sigmas[i]
        incoming_atheta = atheta
        grad[2 * i] = math.fma(incoming_atheta, self.curvatures[i + 1], math.fma(ak, sigma, p_t))
        grad[2 * i + 1] = math.fma(0.5, p_r2, math.fma(0.5 * incoming_atheta, L * L, ak * L))

        atheta = incoming_atheta + p_r0
        ak = math.fma(incoming_atheta, L, ak + p_r1)
        return atheta, ak


def validate_raw_parameters(raw_params: Sequence[float]) -> list[float]:
    """Validate flat ``[L0, sigma0, L1, sigma1, ...]`` parameters."""
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


def knot_parameters_to_raw(
    knot_params: Sequence[float],
    *,
    initial_k: float,
    initial_s: float = 0.0,
) -> list[float]:
    """Convert ``[s1, k1, ..., sn, kn]`` to flat ``(L, sigma)`` parameters.

    ``s0=initial_s`` and ``k0=initial_k`` are fixed and omitted from the knot
    vector.  Knot locations must be strictly increasing.
    """
    params = [float(value) for value in knot_params]
    if len(params) % 2:
        raise ValueError("knot parameter array must have even length: [s1, k1, ...]")
    s_prev = float(initial_s)
    k_prev = float(initial_k)
    if not math.isfinite(s_prev) or not math.isfinite(k_prev):
        raise ValueError("initial_s and initial_k must be finite")
    raw = [0.0] * len(params)
    for i in range(len(params) // 2):
        s_next = params[2 * i]
        k_next = params[2 * i + 1]
        if not math.isfinite(s_next) or not math.isfinite(k_next):
            raise ValueError(f"knot {i + 1} must be finite")
        L = s_next - s_prev
        if L <= 0.0:
            raise ValueError(
                f"knot locations must be strictly increasing: s[{i}]={s_prev!r}, "
                f"s[{i + 1}]={s_next!r}"
            )
        raw[2 * i] = L
        raw[2 * i + 1] = (k_next - k_prev) / L
        s_prev = s_next
        k_prev = k_next
    return raw


def pullback_raw_gradient_from_precomputed_raw(
    raw_params: Sequence[float],
    raw_gradient: Sequence[float],
) -> list[float]:
    """Pull a raw gradient back using an already-computed knot→raw map.

    This is algebraically and operation-order identical to the historical
    pullback after ``knot_parameters_to_raw``.  Geometry caches already own the
    exact raw vector for the current optimizer point, so pose rows can reuse it
    instead of rebuilding the same transform three times per pose.
    """
    raw = raw_params
    if len(raw_gradient) != len(raw):
        raise ValueError("raw gradient length does not match raw parameters")
    n = len(raw) // 2
    grad_s = [0.0] * (n + 1)
    grad_k = [0.0] * (n + 1)
    for i in range(n):
        L = raw[2 * i]
        sigma = raw[2 * i + 1]
        gL = float(raw_gradient[2 * i])
        gsigma = float(raw_gradient[2 * i + 1])
        inv_L = 1.0 / L
        q = math.fma(-sigma * inv_L, gsigma, gL)
        r = gsigma * inv_L
        grad_s[i] -= q
        grad_s[i + 1] += q
        grad_k[i] -= r
        grad_k[i + 1] += r

    out = [0.0] * (2 * n)
    for j in range(1, n + 1):
        out[2 * (j - 1)] = grad_s[j]
        out[2 * (j - 1) + 1] = grad_k[j]
    return out


def pullback_raw_gradient_to_knot_parameters(
    knot_params: Sequence[float],
    raw_gradient: Sequence[float],
    *,
    initial_k: float,
    initial_s: float = 0.0,
) -> list[float]:
    """Pull a flat raw gradient back to ``[s1, k1, ..., sn, kn]``.

    The fixed ``s0`` and ``k0`` components are intentionally omitted.
    """
    raw = knot_parameters_to_raw(
        knot_params, initial_k=initial_k, initial_s=initial_s
    )
    return pullback_raw_gradient_from_precomputed_raw(raw, raw_gradient)


def _checked_fraction(tau: float) -> float:
    tau = float(tau)
    if not math.isfinite(tau) or tau < 0.0 or tau > 1.0:
        raise ValueError(f"tau must lie in [0,1], got {tau!r}")
    return tau


def _checked_segment_index(path: GeometryPath, i: int) -> int:
    i = int(i)
    if not 0 <= i < path.n_segments:
        raise IndexError((i, path.n_segments))
    return i


def compile_geometry_path(
    raw_params: Sequence[float],
    initial_state: GeometryState | Sequence[float] = GeometryState(0.0, 0.0, 0.0, 0.0),
) -> GeometryPath:
    """Compile a compact geometry tape from flat raw parameters.

    ``raw_params`` uses ``[L0, sigma0, L1, sigma1, ...]``.
    """
    raw = validate_raw_parameters(raw_params)
    if len(initial_state) != 4:
        raise ValueError("initial_state must be (x0, y0, theta0, k0)")
    x, y, theta, k = map(float, initial_state)
    if not all(map(math.isfinite, (x, y, theta, k))):
        raise ValueError("initial state must be finite")

    n = len(raw) // 2
    lengths = [0.0] * n
    sigmas = [0.0] * n
    xs = [0.0] * (n + 1)
    ys = [0.0] * (n + 1)
    thetas = [0.0] * (n + 1)
    curvatures = [0.0] * (n + 1)
    dx = [0.0] * n
    dy = [0.0] * n
    m1x = [0.0] * n
    m1y = [0.0] * n
    m2x = [0.0] * n
    m2y = [0.0] * n
    cos_end = [0.0] * n
    sin_end = [0.0] * n

    xs[0], ys[0], thetas[0], curvatures[0] = x, y, theta, k
    for i in range(n):
        L = raw[2 * i]
        sigma = raw[2 * i + 1]
        m = clothoid_moments(theta, k, sigma, L)
        lengths[i] = L
        sigmas[i] = sigma
        dx[i], dy[i] = m.z0_re, m.z0_im
        m1x[i], m1y[i] = m.z1_re, m.z1_im
        m2x[i], m2y[i] = m.z2_re, m.z2_im
        c1 = math.cos(m.theta1)
        s1 = math.sin(m.theta1)
        cos_end[i], sin_end[i] = c1, s1
        x += m.z0_re
        y += m.z0_im
        theta = m.theta1
        k = m.k1
        xs[i + 1], ys[i + 1], thetas[i + 1], curvatures[i + 1] = x, y, theta, k

    return GeometryPath(
        lengths, sigmas, xs, ys, thetas, curvatures, dx, dy, m1x, m1y,
        m2x, m2y, cos_end, sin_end
    )



def pullback_raw_gradient_to_knots(
    raw_params: Sequence[float],
    raw_gradient: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Pull flat raw gradients back to full knot arrays ``(s0..sn, k0..kn)``."""
    raw = validate_raw_parameters(raw_params)
    n = len(raw) // 2
    if len(raw_gradient) != len(raw):
        raise ValueError("raw gradient length does not match raw parameters")
    grad_s = [0.0] * (n + 1)
    grad_k = [0.0] * (n + 1)
    for i in range(n):
        L = raw[2 * i]
        sigma = raw[2 * i + 1]
        gL = float(raw_gradient[2 * i])
        gsigma = float(raw_gradient[2 * i + 1])
        inv_L = 1.0 / L
        q = math.fma(-sigma * inv_L, gsigma, gL)
        r = gsigma * inv_L
        grad_s[i] -= q
        grad_s[i + 1] += q
        grad_k[i] -= r
        grad_k[i + 1] += r
    return grad_s, grad_k



def clothoid_moments(theta0: float, k0: float, sigma: float, length: float) -> MomentResult:
    """Evaluate displacement and the first two path moments in real channels.

    The general Euler branch uses the production scalar ``geometry.fresnel``
    implementation.  Higher moments are reconstructed by stable real
    recurrences when well conditioned, with real-valued series, circular
    perturbation, and compensated quadrature fallbacks near removable
    singularities.
    """
    theta0 = float(theta0)
    k0 = float(k0)
    sigma = float(sigma)
    length = float(length)
    if not all(map(math.isfinite, (theta0, k0, sigma, length))):
        raise ValueError("moment inputs must be finite")
    if length < 0.0:
        raise ValueError("length must be nonnegative")
    if length == 0.0:
        return MomentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, theta0, k0)

    a = k0 * length
    b = sigma * length * length
    midpoint_k = math.fma(0.5 * sigma, length, k0)
    theta1 = math.fma(length, midpoint_k, theta0)
    k1 = math.fma(sigma, length, k0)
    c0 = math.cos(theta0)
    s0 = math.sin(theta0)

    if sigma == 0.0 and k0 == 0.0:
        L2 = length * length
        L3_over_3 = L2 * length / 3.0
        return MomentResult(
            length * c0,
            length * s0,
            0.5 * L2 * c0,
            0.5 * L2 * s0,
            L3_over_3 * c0,
            L3_over_3 * s0,
            theta1,
            k1,
        )

    phase_bound = abs(a) + 0.5 * abs(b)
    if phase_bound <= _SERIES_PHASE_BOUND:
        m = _dimensionless_series_moments_real(a, b)
        return _rotate_scale_moments(m, c0, s0, length, theta1, k1)

    if sigma == 0.0:
        m = _dimensionless_circular_moments_real(a)
        return _rotate_scale_moments(m, c0, s0, length, theta1, k1)

    # General production Fresnel path, using the scalar implementation from
    # geometry.py.  All arithmetic remains in paired real channels.
    abs_b = abs(b)
    root_pi_b = math.sqrt(_PI * abs_b)
    eps = math.copysign(1.0, b)
    u0 = eps * a / root_pi_b
    if abs(u0) <= 2.0e5:
        u1 = math.fma(1.0 / _SQRT_PI, math.sqrt(abs_b), u0)
        s_f0, c_f0 = fresnel(u0)
        s_f1, c_f1 = fresnel(u1)
        delta_x = c_f1 - c_f0
        delta_y = eps * (s_f1 - s_f0)
        phase = math.fma(-0.5 * a, a / b, theta0)
        cp = math.cos(phase)
        sp = math.sin(phase)
        scale = _SQRT_PI / math.sqrt(abs_b)
        m0x = scale * math.fma(cp, delta_x, -sp * delta_y)
        m0y = scale * math.fma(sp, delta_x, cp * delta_y)

        c1 = math.cos(theta1)
        s1 = math.sin(theta1)
        n1x = math.fma(-a, m0x, s1 - s0)
        n1y = math.fma(-a, m0y, c0 - c1)
        m1x = n1x / b
        m1y = n1y / b
        n2x = math.fma(-a, m1x, s1 - m0y)
        n2y = math.fma(-a, m1y, m0x - c1)
        m2x = n2x / b
        m2y = n2y / b

        cond1 = (
            math.hypot(c1 - c0, s1 - s0) + abs(a) * math.hypot(m0x, m0y)
        ) / max(math.hypot(n1x, n1y), 1.0e-300)
        cond2 = (
            math.hypot(c1 - m0x, s1 - m0y) + abs(a) * math.hypot(m1x, m1y)
        ) / max(math.hypot(n2x, n2y), 1.0e-300)

        if (
            all(math.isfinite(v) for v in (m0x, m0y, m1x, m1y, m2x, m2y))
            and cond1 <= _FRESNEL_RECURRENCE_COND_MAX
            and cond2 <= _FRESNEL_RECURRENCE_COND_MAX
        ):
            return _rotate_scale_moments(
                (m0x, m0y, m1x, m1y, m2x, m2y),
                1.0,
                0.0,
                length,
                theta1,
                k1,
            )

    # For very large completed-square arguments or a cancelled Fresnel
    # recurrence, recover displacement through the full production geometry
    # evaluator (including its Gauss/asymptotic branches), then retry the
    # physical integration-by-parts recurrences.  This path is exceptional and
    # avoids an unbounded oscillatory quadrature fallback.
    geo = create_geo(0.0, 0.0, theta0, k0, sigma)
    z0x, z0y, theta1_geo, k1_geo = geo.end(length)
    c1 = math.cos(theta1_geo)
    s1 = math.sin(theta1_geo)
    n1x = math.fma(-k0, z0x, s1 - s0)
    n1y = math.fma(-k0, z0y, c0 - c1)
    z1x = n1x / sigma
    z1y = n1y / sigma
    n2x = math.fma(-k0, z1x, math.fma(length, s1, -z0y))
    n2y = math.fma(-k0, z1y, math.fma(-length, c1, z0x))
    z2x = n2x / sigma
    z2y = n2y / sigma
    cond1 = (
        math.hypot(c1 - c0, s1 - s0) + abs(k0) * math.hypot(z0x, z0y)
    ) / max(math.hypot(n1x, n1y), 1.0e-300)
    cond2 = (
        math.hypot(length * c1 - z0x, length * s1 - z0y)
        + abs(k0) * math.hypot(z1x, z1y)
    ) / max(math.hypot(n2x, n2y), 1.0e-300)
    if (
        all(math.isfinite(v) for v in (z0x, z0y, z1x, z1y, z2x, z2y))
        and cond1 <= _FRESNEL_RECURRENCE_COND_MAX
        and cond2 <= _FRESNEL_RECURRENCE_COND_MAX
    ):
        return MomentResult(
            z0x, z0y, z1x, z1y, z2x, z2y, theta1_geo, k1_geo
        )

    # Stable real fallbacks.  These also ensure smooth derivatives through the
    # exact sigma=0 line/circle branches.
    if phase_bound <= 4.0:
        m = _dimensionless_series_moments_real(a, b)
        return _rotate_scale_moments(m, c0, s0, length, theta1, k1)

    perturbation = _dimensionless_b_perturbation_moments_real(a, b)
    if perturbation is not None:
        return _rotate_scale_moments(perturbation, c0, s0, length, theta1, k1)

    m = _dimensionless_quadrature_moments_real(theta0, a, b)
    L2 = length * length
    return MomentResult(
        length * m[0],
        length * m[1],
        L2 * m[2],
        L2 * m[3],
        L2 * length * m[4],
        L2 * length * m[5],
        theta1,
        k1,
    )


def _rotate_scale_moments(
    m: tuple[float, float, float, float, float, float],
    cosine: float,
    sine: float,
    length: float,
    theta1: float,
    k1: float,
) -> MomentResult:
    m0x, m0y = _rotate_pair(m[0], m[1], cosine, sine)
    m1x, m1y = _rotate_pair(m[2], m[3], cosine, sine)
    m2x, m2y = _rotate_pair(m[4], m[5], cosine, sine)
    L2 = length * length
    return MomentResult(
        length * m0x,
        length * m0y,
        L2 * m1x,
        L2 * m1y,
        L2 * length * m2x,
        L2 * length * m2y,
        theta1,
        k1,
    )


def _rotate_pair(
    x: float,
    y: float,
    cosine: float,
    sine: float,
) -> tuple[float, float]:
    return (
        math.fma(cosine, x, -sine * y),
        math.fma(sine, x, cosine * y),
    )


def _dimensionless_series_moments_real(
    a: float,
    b: float,
) -> tuple[float, float, float, float, float, float]:
    # exp(i(a t + b t^2/2)) = sum d_n t^n,
    # (n+1)d_{n+1}=i*a*d_n+i*b*d_{n-1}.
    d_prev_x = 0.0
    d_prev_y = 0.0
    d_x = 1.0
    d_y = 0.0
    m0x, m0y = 1.0, 0.0
    m1x, m1y = 0.5, 0.0
    m2x, m2y = 1.0 / 3.0, 0.0
    quiet = 0

    for n in range(_SERIES_MAX_TERMS - 1):
        qx = math.fma(a, d_x, b * d_prev_x)
        qy = math.fma(a, d_y, b * d_prev_y)
        inv = 1.0 / (n + 1.0)
        d_next_x = -qy * inv
        d_next_y = qx * inv
        j = n + 1
        inv0 = 1.0 / (j + 1.0)
        inv1 = 1.0 / (j + 2.0)
        inv2 = 1.0 / (j + 3.0)
        t0x, t0y = d_next_x * inv0, d_next_y * inv0
        t1x, t1y = d_next_x * inv1, d_next_y * inv1
        t2x, t2y = d_next_x * inv2, d_next_y * inv2
        m0x += t0x
        m0y += t0y
        m1x += t1x
        m1y += t1y
        m2x += t2x
        m2y += t2y
        scale = max(
            1.0,
            math.hypot(m0x, m0y),
            math.hypot(m1x, m1y),
            math.hypot(m2x, m2y),
        )
        term = max(
            math.hypot(t0x, t0y),
            math.hypot(t1x, t1y),
            math.hypot(t2x, t2y),
        )
        if term <= _SERIES_REL_TOL * scale:
            quiet += 1
            if quiet >= 3:
                return m0x, m0y, m1x, m1y, m2x, m2y
        else:
            quiet = 0
        d_prev_x, d_prev_y = d_x, d_y
        d_x, d_y = d_next_x, d_next_y

    return m0x, m0y, m1x, m1y, m2x, m2y


def _dimensionless_circular_moments_real(
    a: float,
) -> tuple[float, float, float, float, float, float]:
    # Called only outside the small-a series branch.
    sa = math.sin(a)
    ca = math.cos(a)
    m0x = sa / a
    m0y = (2.0 * math.sin(0.5 * a) ** 2) / a

    n1x = ca - m0x
    n1y = sa - m0y
    m1x = n1y / a
    m1y = -n1x / a

    n2x = ca - 2.0 * m1x
    n2y = sa - 2.0 * m1y
    m2x = n2y / a
    m2y = -n2x / a
    return m0x, m0y, m1x, m1y, m2x, m2y


def _dimensionless_b_perturbation_moments_real(
    a: float,
    b: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Expansion in quadratic phase around the circular solution."""
    if a == 0.0 or abs(b) > 4.0:
        return None

    coeff_x = [1.0]
    coeff_y = [0.0]
    for r in range(1, 41):
        scale = b / (2.0 * r)
        cx = -coeff_y[-1] * scale
        cy = coeff_x[-1] * scale
        coeff_x.append(cx)
        coeff_y.append(cy)
        if r >= 4 and abs(cx) + abs(cy) <= 2.0e-19:
            break

    terms = len(coeff_x) - 1
    max_order = 2 + 2 * terms
    if abs(a) < 0.25 * max_order:
        return None

    sa = math.sin(a)
    ca = math.cos(a)
    cnx = sa / a
    cny = 2.0 * math.sin(0.5 * a) ** 2 / a
    m0x = m0y = 0.0
    m1x = m1y = 0.0
    m2x = m2y = 0.0

    for n in range(max_order + 1):
        if n & 1:
            r = (n - 1) >> 1
            if r <= terms:
                cx = coeff_x[r]
                cy = coeff_y[r]
                m1x += math.fma(cx, cnx, -cy * cny)
                m1y += math.fma(cx, cny, cy * cnx)
        else:
            r0 = n >> 1
            if r0 <= terms:
                cx = coeff_x[r0]
                cy = coeff_y[r0]
                m0x += math.fma(cx, cnx, -cy * cny)
                m0y += math.fma(cx, cny, cy * cnx)
            if n >= 2:
                r2 = (n - 2) >> 1
                if r2 <= terms:
                    cx = coeff_x[r2]
                    cy = coeff_y[r2]
                    m2x += math.fma(cx, cnx, -cy * cny)
                    m2y += math.fma(cx, cny, cy * cnx)

        if n != max_order:
            nx = ca - (n + 1) * cnx
            ny = sa - (n + 1) * cny
            cnx = ny / a
            cny = -nx / a

    values = (m0x, m0y, m1x, m1y, m2x, m2y)
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _total_heading_variation(a: float, b: float) -> float:
    if b == 0.0:
        return abs(a)
    root = -a / b
    primitive_1 = a + 0.5 * b
    if 0.0 < root < 1.0:
        primitive_root = a * root + 0.5 * b * root * root
        return abs(primitive_root) + abs(primitive_1 - primitive_root)
    return abs(primitive_1)


def _dimensionless_quadrature_moments_real(
    theta0: float,
    a: float,
    b: float,
) -> tuple[float, float, float, float, float, float]:
    variation = _total_heading_variation(a, b)
    panels = max(1, int(math.ceil(variation / _QUAD_PHASE_PER_PANEL)))
    if panels > _QUAD_MAX_PANELS:
        raise ArithmeticError(
            f"geometry phase variation requires {panels} quadrature panels; "
            f"limit is {_QUAD_MAX_PANELS}"
        )

    sums = [0.0] * 6
    comps = [0.0] * 6
    inv_panels = 1.0 / panels
    for p in range(panels):
        lo = p * inv_panels
        hi = (p + 1) * inv_panels
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        for xg, wg in zip(_GL16_X, _GL16_W):
            t = math.fma(half, xg, mid)
            phase = math.fma(0.5 * b * t, t, math.fma(a, t, theta0))
            c = math.cos(phase)
            s = math.sin(phase)
            w = half * wg
            t2 = t * t
            vals = (
                w * c,
                w * s,
                w * t * c,
                w * t * s,
                w * t2 * c,
                w * t2 * s,
            )
            for j, value in enumerate(vals):
                old = sums[j]
                new = old + value
                if abs(old) >= abs(value):
                    comps[j] += (old - new) + value
                else:
                    comps[j] += (value - new) + old
                sums[j] = new
    return tuple(sums[j] + comps[j] for j in range(6))  # type: ignore[return-value]


__all__ = [
    "GeometryPath",
    "GeometryState",
    "MomentResult",
    "clothoid_moments",
    "compile_geometry_path",
    "knot_parameters_to_raw",
    "pullback_raw_gradient_to_knot_parameters",
    "pullback_raw_gradient_to_knots",
    "validate_raw_parameters",
]
