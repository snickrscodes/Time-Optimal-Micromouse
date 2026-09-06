from __future__ import annotations

import math

from .base import DiffSegment, EvalSegment, SegmentType
from .constants import A_MAX, V_MAX, B_EMF
from .lambertw import lambertw0


B_EMF_INV = 1.0 / B_EMF
LW_EQ = 1.0 / V_MAX
LW_EQ_INV = V_MAX
W_EQ = V_MAX * V_MAX

LW_EQB = -A_MAX / W_EQ

MOTOR_DT_DW0_SCALE = 0.5 * LW_EQ / A_MAX

# Stable expansion around normalized equilibrium speed v = 1.
MOTOR_U0_SERIES_TOL = 2.0e-3

# Small-distance expansion:
#
#     rho = |tau| * max(1, |u|)
#     tau = a / v^2
#
MOTOR_SMALL_STEP_TOL = 2.0e-3

# Inverse-gap solver thresholds.
_MOTOR_GAP_DIRECT_MAX = 3.0e-3
_MOTOR_GAP_ONE_HALLEY_MAX = 1.0e-1
_MOTOR_GAP_LAMBERT_MIN = 5.0e-1

# Time refinement is needed when theta is small relative to the
# quantities from which it was reconstructed.
_MOTOR_TIME_REFINE_MAX = 2.0
_MOTOR_TIME_REFINE_REL = 2.5e-1

_INV_E = 1.0 / math.e
_LOG_HALF = math.log(0.5)


# ----------------------------------------------------------------------
# Equilibrium u-series
# ----------------------------------------------------------------------

def _motor_q_series_h5(
    u: float,
    e: float,
) -> float:
    """
    Return H such that

        q = e + u*e*(e - 1)*H + O(u^6),

    where

        q = W(u*exp(u + a)) / u
        e = exp(a).

    The returned polynomial includes terms through u^5 in q.
    """

    p2 = math.fma(
        3.0,
        e,
        -1.0,
    )

    p3 = math.fma(
        e,
        16.0 * e - 11.0,
        1.0,
    )

    p4 = math.fma(
        e,
        math.fma(
            e,
            math.fma(
                125.0,
                e,
                -131.0,
            ),
            31.0,
        ),
        -1.0,
    )

    p5 = math.fma(
        e,
        math.fma(
            e,
            math.fma(
                e,
                math.fma(
                    1296.0,
                    e,
                    -1829.0,
                ),
                731.0,
            ),
            -79.0,
        ),
        1.0,
    )

    h = math.fma(
        -p5 / 120.0,
        u,
        p4 / 24.0,
    )

    h = math.fma(
        h,
        u,
        -p3 / 6.0,
    )

    h = math.fma(
        h,
        u,
        0.5 * p2,
    )

    return math.fma(
        h,
        u,
        -1.0,
    )


# ----------------------------------------------------------------------
# Small-distance expansion
# ----------------------------------------------------------------------

def _motor_small_step(
    v: float,
    tau: float,
):
    """
    Small-step expansion in

        tau = a / v^2.

    Returns

        q, z, theta, m

    where

        q     = r/u
        z     = 1 + r
        theta = B_EMF*T = -log(q)
        m     = (q - 1)/(v*z).

    The caller precomputes tau from ds, avoiding divisions in the hot
    evaluation path.
    """

    c3 = math.fma(
        -2.0,
        v,
        3.0,
    ) / 6.0

    c4 = math.fma(
        v,
        math.fma(
            6.0,
            v,
            -20.0,
        ),
        15.0,
    ) / 24.0

    c5 = math.fma(
        v,
        math.fma(
            v,
            math.fma(
                -24.0,
                v,
                130.0,
            ),
            -210.0,
        ),
        105.0,
    ) / 120.0

    c6 = math.fma(
        v,
        math.fma(
            v,
            math.fma(
                v,
                math.fma(
                    120.0,
                    v,
                    -924.0,
                ),
                2380.0,
            ),
            -2520.0,
        ),
        945.0,
    ) / 720.0

    h = math.fma(
        c6,
        tau,
        c5,
    )

    h = math.fma(
        h,
        tau,
        c4,
    )

    h = math.fma(
        h,
        tau,
        c3,
    )

    h = math.fma(
        h,
        tau,
        0.5,
    )

    # s = (q - 1)/v
    s = tau * math.fma(
        tau,
        h,
        1.0,
    )

    q = math.fma(
        v,
        s,
        1.0,
    )

    # z = v * (1 + (v - 1)*s)
    #
    # The equivalent form below preserves v when v - 1 rounds to
    # exactly -1.
    z = v * math.fma(
        v,
        s,
        1.0 - s,
    )

    # theta = v*tau*((v - 1)*tau*h - 1)
    u_tau = math.fma(
        v,
        tau,
        -tau,
    )

    theta = (
        v
        * tau
        * math.fma(
            u_tau,
            h,
            -1.0,
        )
    )

    # Exact removable-singularity form.
    m = s / z

    return q, z, theta, m


def _motor_small_step_parameters(
    v: float,
    u: float,
):
    """
    Precompute the mapping from ds to tau and the maximum ds for which
    the small-step expansion is enabled.
    """

    if v == 0.0:
        return 0.0, 0.0

    v2 = v * v

    # Extremely tiny v can underflow v^2. Such states should use the
    # branch-gap path instead.
    if v2 == 0.0:
        return 0.0, 0.0

    tau_per_ds = LW_EQB / v2

    rho_per_ds = (
        -tau_per_ds
        * max(
            1.0,
            abs(u),
        )
    )

    small_ds_max = MOTOR_SMALL_STEP_TOL / rho_per_ds

    return tau_per_ds, small_ds_max


# ----------------------------------------------------------------------
# Branch-gap representation for v < 1
# ----------------------------------------------------------------------

def _motor_gap(
    z: float,
) -> float:
    """
    F(z) = 1 - (1 - z)*exp(z).

    The degree-14 series avoids endpoint cancellation. Extending its
    crossover to 0.4 improved the inverse-gap solver from roughly
    three ulps to two ulps in the validation sweep.
    """

    if z < 0.4:
        h = 1.0 / 6706022400.0
        h = math.fma(h, z, 1.0 / 518918400.0)
        h = math.fma(h, z, 1.0 / 43545600.0)
        h = math.fma(h, z, 1.0 / 3991680.0)
        h = math.fma(h, z, 1.0 / 403200.0)
        h = math.fma(h, z, 1.0 / 45360.0)
        h = math.fma(h, z, 1.0 / 5760.0)
        h = math.fma(h, z, 1.0 / 840.0)
        h = math.fma(h, z, 1.0 / 144.0)
        h = math.fma(h, z, 1.0 / 30.0)
        h = math.fma(h, z, 1.0 / 8.0)
        h = math.fma(h, z, 1.0 / 3.0)
        h = math.fma(h, z, 0.5)

        return z * z * h

    return math.fma(
        z - 1.0,
        math.expm1(z),
        z,
    )


def _motor_gap_at_a(
    g0: float,
    a: float,
) -> float:
    """
    Evolve the branch gap using

        1 - g = (1 - g0)*exp(a).
    """

    if a > -0.5:
        em1 = math.expm1(a)
        e = 1.0 + em1
    else:
        e = math.exp(a)
        em1 = e - 1.0

    return math.fma(
        g0,
        e,
        -em1,
    )


def _motor_gap_lambert_switch_a(
    g0: float,
) -> float:
    """
    Return the a at which the evolved gap reaches 0.5.

    For more negative a, the direct Lambert representation is already
    well-conditioned and faster than continuing through the gap
    machinery.
    """

    if g0 >= _MOTOR_GAP_LAMBERT_MIN:
        return 0.0

    # Solve
    #
    #     0.5 = 1 - (1 - g0)*exp(a).
    return (
        _LOG_HALF
        - math.log1p(-g0)
    )


def _motor_z_puiseux6(
    g: float,
) -> float:
    p = math.sqrt(2.0 * g)

    h = math.fma(
        -221.0 / 8505.0,
        p,
        769.0 / 17280.0,
    )

    h = math.fma(
        h,
        p,
        -43.0 / 540.0,
    )

    h = math.fma(
        h,
        p,
        11.0 / 72.0,
    )

    h = math.fma(
        h,
        p,
        -1.0 / 3.0,
    )

    h = math.fma(
        h,
        p,
        1.0,
    )

    return p * h


def _motor_z_puiseux12(
    g: float,
) -> float:
    p = math.sqrt(2.0 * g)

    h = math.fma(
        -1118511313.0 / 709296588000.0,
        p,
        169709463197.0 / 69528040243200.0,
    )

    h = math.fma(
        h,
        p,
        -5776369.0 / 1515591000.0,
    )

    h = math.fma(
        h,
        p,
        226287557.0 / 37623398400.0,
    )

    h = math.fma(
        h,
        p,
        -1963.0 / 204120.0,
    )

    h = math.fma(
        h,
        p,
        680863.0 / 43545600.0,
    )

    h = math.fma(
        h,
        p,
        -221.0 / 8505.0,
    )

    h = math.fma(
        h,
        p,
        769.0 / 17280.0,
    )

    h = math.fma(
        h,
        p,
        -43.0 / 540.0,
    )

    h = math.fma(
        h,
        p,
        11.0 / 72.0,
    )

    h = math.fma(
        h,
        p,
        -1.0 / 3.0,
    )

    h = math.fma(
        h,
        p,
        1.0,
    )

    return p * h


def _motor_z_halley(
    z: float,
    g: float,
) -> float:
    residual = _motor_gap(z) - g
    scaled = residual * math.exp(-z)

    return z - (
        2.0 * scaled * z
        / math.fma(
            2.0 * z,
            z,
            -scaled * (z + 1.0),
        )
    )


def _motor_z_from_gap(
    g: float,
) -> float:
    if g < 0.0:
        raise ValueError(
            "motor state is outside the real Lambert-W branch"
        )

    if g == 0.0:
        return 0.0

    if g >= 1.0:
        return 1.0

    if g < _MOTOR_GAP_DIRECT_MAX:
        return _motor_z_puiseux12(g)

    if g < _MOTOR_GAP_ONE_HALLEY_MAX:
        return _motor_z_halley(
            _motor_z_puiseux12(g),
            g,
        )

    if g < _MOTOR_GAP_LAMBERT_MIN:
        z = _motor_z_halley(
            _motor_z_puiseux6(g),
            g,
        )

        return _motor_z_halley(
            z,
            g,
        )

    return (
        1.0
        + lambertw0(
            (g - 1.0) * _INV_E
        )
    )


# ----------------------------------------------------------------------
# Time reconstruction and refinement
# ----------------------------------------------------------------------

def _motor_expm1mx(
    theta: float,
) -> float:
    """
    Return

        exp(-theta) - 1 + theta

    without cancellation near theta = 0.
    """

    if abs(theta) < 0.25:
        h = 1.0 / 87178291200.0
        h = math.fma(h, theta, -1.0 / 6227020800.0)
        h = math.fma(h, theta, 1.0 / 479001600.0)
        h = math.fma(h, theta, -1.0 / 39916800.0)
        h = math.fma(h, theta, 1.0 / 3628800.0)
        h = math.fma(h, theta, -1.0 / 362880.0)
        h = math.fma(h, theta, 1.0 / 40320.0)
        h = math.fma(h, theta, -1.0 / 5040.0)
        h = math.fma(h, theta, 1.0 / 720.0)
        h = math.fma(h, theta, -1.0 / 120.0)
        h = math.fma(h, theta, 1.0 / 24.0)
        h = math.fma(h, theta, -1.0 / 6.0)
        h = math.fma(h, theta, 0.5)

        return theta * theta * h

    return math.expm1(-theta) + theta


def _motor_finalize_time_state(
    v: float,
    z: float,
    b: float,
    theta0: float,
    q_direct: float | None,
):
    """
    Refine the cancellation-sensitive time variable while preserving
    the already accurate final-speed state z.

    Parameters
    ----------
    v:
        Initial normalized speed.
    z:
        Final normalized speed from the small-step, gap, or Lambert
        state solver.
    b:
        -a, which is nonnegative for forward evaluation.
    theta0:
        Initial estimate

            theta0 = z - v + b.

    q_direct:
        r/u when that quotient is available directly from a
        well-conditioned Lambert evaluation. Pass None for gap-based
        states.

    Returns
    -------
    theta, m
    """

    refine = (
        theta0 < _MOTOR_TIME_REFINE_MAX
        and theta0
        < _MOTOR_TIME_REFINE_REL
        * max(
            v,
            z,
            b,
        )
    )

    # When q came directly from r/u, use it in well-conditioned cases
    # instead of computing another exponential.
    #
    # For v < 1 and moderate theta, exp(-theta0) gave a marginally
    # more accurate m in testing. For theta >= 2, q - 1 is already far
    # from cancellation and q_direct is preferable.
    use_q_direct = (
        not refine
        and q_direct is not None
        and (
            v > 1.0
            or theta0 >= _MOTOR_TIME_REFINE_MAX
        )
    )

    if use_q_direct:
        qm1 = q_direct - 1.0

        return (
            theta0,
            qm1 / (v * z),
        )

    if theta0 < 0.5:
        qm1 = math.expm1(-theta0)
        q = 1.0 + qm1
    else:
        q = math.exp(-theta0)
        qm1 = q - 1.0

    theta = theta0

    if refine:
        # Exact scalar equation:
        #
        #   theta
        #   + (1 - v)*(exp(-theta) - 1)
        #   - b
        #   = 0.
        #
        # Rewriting exp(-theta) - 1 as
        #
        #   -theta + expm1mx(theta)
        #
        # removes the leading cancellation.
        remainder = _motor_expm1mx(theta0)

        residual = math.fma(
            1.0 - v,
            remainder,
            math.fma(
                v,
                theta0,
                -b,
            ),
        )

        # Derivative and second derivative of the scalar residual.
        z_theta = math.fma(
            v,
            q,
            -qm1,
        )

        second = (1.0 - v) * q

        delta = (
            2.0 * residual * z_theta
            / math.fma(
                2.0 * z_theta,
                z_theta,
                -residual * second,
            )
        )

        theta = theta0 - delta

        if abs(delta) < 1.0e-4:
            # expm1(delta) through delta^3. The omitted term is below
            # fp64 significance at this threshold.
            ed = delta * math.fma(
                delta,
                math.fma(
                    delta,
                    1.0 / 6.0,
                    0.5,
                ),
                1.0,
            )
        else:
            ed = math.expm1(delta)

        # q_new - 1 = q_old*exp(delta) - 1
        qm1 = math.fma(
            q,
            ed,
            qm1,
        )

    # Critically, retain the original z. Reconstructing z from the
    # refined theta can replace a correctly rounded state with one
    # several ulps worse.
    m = qm1 / (v * z)

    return theta, m


# ----------------------------------------------------------------------
# Shared generic-state preparation
# ----------------------------------------------------------------------

def _motor_generic_initial_data(
    w0: float,
):
    y0 = math.sqrt(w0)

    v0 = LW_EQ * y0
    u0 = math.fma(
        LW_EQ,
        y0,
        -1.0,
    )

    tau_per_ds, small_ds_max = (
        _motor_small_step_parameters(
            v0,
            u0,
        )
    )

    if v0 < 1.0:
        g0 = _motor_gap(v0)

        gap_lambert_a = (
            _motor_gap_lambert_switch_a(
                g0,
            )
        )
    else:
        g0 = 0.0
        gap_lambert_a = 0.0

    return (
        v0,
        u0,
        g0,
        gap_lambert_a,
        tau_per_ds,
        small_ds_max,
    )


def _motor_generic_final_state(
    v: float,
    u: float,
    g0: float,
    gap_lambert_a: float,
    a: float,
):
    """
    Return

        z, r_or_none

    where r_or_none is available when the direct Lambert path was
    used.
    """

    if (
        v < 1.0
        and a > gap_lambert_a
    ):
        g = _motor_gap_at_a(
            g0,
            a,
        )

        return (
            _motor_z_from_gap(g),
            None,
        )

    r = lambertw0(
        u * math.exp(
            u + a
        )
    )

    return 1.0 + r, r


# ----------------------------------------------------------------------
# Stable equilibrium segments
# ----------------------------------------------------------------------

class MotorEvalSegmentStable(EvalSegment):
    def __init__(
        self,
        L,
        sigma,
        w0,
        k0,
    ):
        super().__init__(
            L,
            sigma,
            w0,
            k0,
            SegmentType.MOTOR,
        )

        self.w0 = w0

        y0 = math.sqrt(w0)

        self.v0 = LW_EQ * y0
        self.u0 = math.fma(
            LW_EQ,
            y0,
            -1.0,
        )

    def w(
        self,
        ds: float,
    ) -> float:
        if ds == 0.0:
            return self.w0

        u = self.u0
        a = LW_EQB * ds

        e = math.exp(a)
        em1 = e - 1.0

        h = _motor_q_series_h5(
            u,
            e,
        )

        ue = u * e

        q = math.fma(
            ue * em1,
            h,
            e,
        )

        z = math.fma(
            u,
            q,
            1.0,
        )

        return W_EQ * z * z

    def time(self, ds: float) -> float:
        if ds == 0.0:
            return 0.0
        v = self.v0
        u = self.u0
        a = LW_EQB * ds
        if a > -0.5:
            em1 = math.expm1(a)
            e = 1.0 + em1
        else:
            e = math.exp(a)
            em1 = e - 1.0
        h = _motor_q_series_h5(u, e)
        ue = u * e
        q = math.fma(ue * em1, h, e)
        qm1 = em1 * math.fma(ue, h, 1.0)
        theta = math.fma(u, qm1, -a)
        return theta * B_EMF_INV


class MotorSegmentStable(DiffSegment):
    def __init__(
        self,
        L,
        sigma,
        w0,
        k0,
    ):
        super().__init__(
            L,
            sigma,
            w0,
            k0,
            SegmentType.MOTOR,
        )

        self.w0 = w0

        y0 = math.sqrt(w0)

        self.v0 = LW_EQ * y0
        self.u0 = math.fma(
            LW_EQ,
            y0,
            -1.0,
        )

    def w(
        self,
        ds: float,
    ) -> float:
        if ds == 0.0:
            return self.w0

        u = self.u0
        a = LW_EQB * ds

        e = math.exp(a)
        em1 = e - 1.0

        h = _motor_q_series_h5(
            u,
            e,
        )

        ue = u * e

        q = math.fma(
            ue * em1,
            h,
            e,
        )

        z = math.fma(
            u,
            q,
            1.0,
        )

        return W_EQ * z * z

    def w_and_jac(
        self,
        ds: float,
    ):
        if ds == 0.0:
            return self.w0, (
                -2.0 * A_MAX * self.u0,
                0.0,
                1.0,
                0.0,
                self.sigma,
                ds,
                0.0,
                1.0,
            )

        u = self.u0
        a = LW_EQB * ds

        e = math.exp(a)
        em1 = e - 1.0

        h = _motor_q_series_h5(
            u,
            e,
        )

        ue = u * e

        q = math.fma(
            ue * em1,
            h,
            e,
        )

        r = u * q

        z = math.fma(
            u,
            q,
            1.0,
        )

        return W_EQ * z * z, (
            -2.0 * A_MAX * r,
            0.0,
            q,
            0.0,
            self.sigma,
            ds,
            0.0,
            1.0,
        )

    def time_and_jac(
        self,
        ds: float,
    ):
        v = self.v0

        if ds == 0.0:
            return 0.0, (
                LW_EQ / v,
                0.0,
                0.0,
                0.0,
            )

        u = self.u0
        a = LW_EQB * ds

        if a > -0.5:
            em1 = math.expm1(a)
            e = 1.0 + em1
        else:
            e = math.exp(a)
            em1 = e - 1.0

        h = _motor_q_series_h5(
            u,
            e,
        )

        ue = u * e

        q = math.fma(
            ue * em1,
            h,
            e,
        )

        qm1 = em1 * math.fma(
            ue,
            h,
            1.0,
        )

        z = math.fma(
            u,
            q,
            1.0,
        )

        theta = math.fma(
            u,
            qm1,
            -a,
        )

        m = qm1 / (v * z)

        return theta * B_EMF_INV, (
            LW_EQ / z,
            0.0,
            MOTOR_DT_DW0_SCALE * m,
            0.0,
        )


# ----------------------------------------------------------------------
# Generic segments
# ----------------------------------------------------------------------

class MotorEvalSegment(EvalSegment):
    def __init__(
        self,
        L,
        sigma,
        w0,
        k0,
    ):
        super().__init__(
            L,
            sigma,
            w0,
            k0,
            SegmentType.MOTOR,
        )

        self.w0 = w0

        (
            self.v0,
            self.u0,
            self.g0,
            self.gap_lambert_a,
            self.tau_per_ds,
            self.small_ds_max,
        ) = _motor_generic_initial_data(w0)

    def w(
        self,
        ds: float,
    ) -> float:
        if ds == 0.0:
            return self.w0

        v = self.v0

        if (
            v != 0.0
            and abs(ds) <= self.small_ds_max
        ):
            tau = self.tau_per_ds * ds

            _, z, _, _ = _motor_small_step(
                v,
                tau,
            )

            return W_EQ * z * z

        if v == 1.0:
            return W_EQ

        a = LW_EQB * ds

        z, _ = _motor_generic_final_state(
            v,
            self.u0,
            self.g0,
            self.gap_lambert_a,
            a,
        )

        return W_EQ * z * z

    def time(self, ds: float) -> float:
        if ds == 0.0:
            return 0.0
        v = self.v0
        u = self.u0
        if v != 0.0 and abs(ds) <= self.small_ds_max:
            tau = self.tau_per_ds * ds
            _, _, theta, _ = _motor_small_step(v, tau)
            return theta * B_EMF_INV
        a = LW_EQB * ds
        if v == 1.0:
            return ds * LW_EQ
        z, r_direct = _motor_generic_final_state(
            v, u, self.g0, self.gap_lambert_a, a
        )
        theta0 = math.fma(-1.0, a, z - v)
        if v == 0.0:
            return theta0 * B_EMF_INV
        q_direct = None if r_direct is None else r_direct / u
        theta, _ = _motor_finalize_time_state(v, z, -a, theta0, q_direct)
        return theta * B_EMF_INV


class MotorSegment(DiffSegment):
    def __init__(
        self,
        L,
        sigma,
        w0,
        k0,
    ):
        super().__init__(
            L,
            sigma,
            w0,
            k0,
            SegmentType.MOTOR,
        )

        self.w0 = w0

        (
            self.v0,
            self.u0,
            self.g0,
            self.gap_lambert_a,
            self.tau_per_ds,
            self.small_ds_max,
        ) = _motor_generic_initial_data(w0)

    def w(
        self,
        ds: float,
    ) -> float:
        if ds == 0.0:
            return self.w0

        v = self.v0

        if (
            v != 0.0
            and abs(ds) <= self.small_ds_max
        ):
            tau = self.tau_per_ds * ds

            _, z, _, _ = _motor_small_step(
                v,
                tau,
            )

            return W_EQ * z * z

        if v == 1.0:
            return W_EQ

        a = LW_EQB * ds

        z, _ = _motor_generic_final_state(
            v,
            self.u0,
            self.g0,
            self.gap_lambert_a,
            a,
        )

        return W_EQ * z * z

    def w_and_jac(
        self,
        ds: float,
    ):
        if ds == 0.0:
            return self.w0, (
                -2.0 * A_MAX * self.u0,
                0.0,
                1.0,
                0.0,
                self.sigma,
                ds,
                0.0,
                1.0,
            )

        v = self.v0
        u = self.u0

        if (
            v != 0.0
            and abs(ds) <= self.small_ds_max
        ):
            tau = self.tau_per_ds * ds

            q, z, _, _ = _motor_small_step(
                v,
                tau,
            )

            r = z - 1.0

        elif v == 1.0:
            a = LW_EQB * ds

            q = math.exp(a)
            z = 1.0
            r = 0.0

        else:
            a = LW_EQB * ds

            z, r_direct = _motor_generic_final_state(
                v,
                u,
                self.g0,
                self.gap_lambert_a,
                a,
            )

            if r_direct is None:
                r = z - 1.0

                theta = math.fma(
                    -1.0,
                    a,
                    z - v,
                )

                q = math.exp(-theta)
            else:
                r = r_direct
                q = r / u

        return W_EQ * z * z, (
            -2.0 * A_MAX * r,
            0.0,
            q,
            0.0,
            self.sigma,
            ds,
            0.0,
            1.0,
        )

    def time_and_jac(
        self,
        ds: float,
    ):
        v = self.v0

        if ds == 0.0:
            return 0.0, (
                math.inf
                if v == 0.0
                else LW_EQ / v,
                0.0,
                0.0,
                0.0,
            )

        u = self.u0

        if (
            v != 0.0
            and abs(ds) <= self.small_ds_max
        ):
            tau = self.tau_per_ds * ds

            _, z, theta, m = _motor_small_step(
                v,
                tau,
            )

            return theta * B_EMF_INV, (
                LW_EQ / z,
                0.0,
                MOTOR_DT_DW0_SCALE * m,
                0.0,
            )

        a = LW_EQB * ds
        b = -a

        if v == 1.0:
            return ds * LW_EQ, (
                LW_EQ,
                0.0,
                MOTOR_DT_DW0_SCALE
                * math.expm1(a),
                0.0,
            )

        z, r_direct = _motor_generic_final_state(
            v,
            u,
            self.g0,
            self.gap_lambert_a,
            a,
        )

        theta0 = math.fma(
            -1.0,
            a,
            z - v,
        )

        if v == 0.0:
            return theta0 * B_EMF_INV, (
                LW_EQ / z,
                0.0,
                -math.inf,
                0.0,
            )

        q_direct = (
            None
            if r_direct is None
            else r_direct / u
        )

        theta, m = _motor_finalize_time_state(
            v,
            z,
            b,
            theta0,
            q_direct,
        )

        return theta * B_EMF_INV, (
            LW_EQ / z,
            0.0,
            MOTOR_DT_DW0_SCALE * m,
            0.0,
        )