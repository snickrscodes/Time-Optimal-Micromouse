from __future__ import annotations

import math
import random

import pytest

from segment.base import SegmentType
from segment.differential_drive import (
    DifferentialDriveDomainError,
    DriveSide,
    acceleration_interval,
    red_comet_dd_yaw_v1_parameters,
    side_acceleration_bounds,
    side_acceleration_grid,
    side_candidate,
    side_candidate_partials,
    side_force_demand_n,
    side_force_demand_q,
    side_lower_partials,
    side_state,
    uniform_rectangle_yaw_inertia,
    yaw_acceleration,
    yaw_rate,
)
from segment.physics_profiles import RED_COMET_2017_NOMINAL
from segment.side_actuator import SideEvalSegment, SideSegment


P = red_comet_dd_yaw_v1_parameters()


def _fd(fun, x: float, rel: float = 2.0e-6) -> float:
    h = rel * max(1.0, abs(x))
    return (fun(x + h) - fun(x - h)) / (2.0 * h)


def _scaled_error(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def test_legacy_segment_enum_values_remain_stable_and_side_modes_append():
    assert SegmentType.GRIP.value == 1
    assert SegmentType.MOTOR.value == 2
    assert SegmentType.BRAKE.value == 3
    assert SegmentType.SIDE_RIGHT.value == 4
    assert SegmentType.SIDE_LEFT.value == 5


def test_frozen_nominal_parameter_derivation_matches_design():
    assert P.beta == pytest.approx(0.10833333333333334, abs=1e-15)
    assert P.eta == pytest.approx(0.18520892687559354, abs=1e-15)
    assert P.yaw_inertia_kg_m2 == pytest.approx(1.9632516666666666e-5, rel=1e-15)
    assert P.side_free_speed_mps == pytest.approx(8.128870991163591, rel=1e-15)
    assert P.side_stall_force_n == pytest.approx(0.46490785185185185, rel=1e-15)
    assert P.q0 == pytest.approx(171.0477747799308, rel=1e-15)


def test_uniform_rectangle_inertia_proxy():
    iz = uniform_rectangle_yaw_inertia(0.0302, 0.076, 0.045)
    assert iz == pytest.approx(P.yaw_inertia_kg_m2, rel=1e-15)


def test_zero_speed_algebra_is_defined_but_derivative_chart_is_not():
    st = side_state(0.0, 0.0, DriveSide.RIGHT, P)
    assert st.side_speed_grid == 0.0
    assert st.force_capacity_q == pytest.approx(P.q0)
    bounds = side_acceleration_bounds(0.0, 0.0, 0.0, DriveSide.LEFT, P)
    assert bounds.lower == pytest.approx(-P.q0)
    assert bounds.upper == pytest.approx(P.q0)
    interval = acceleration_interval(0.0, 0.0, 0.0, P)
    assert interval.feasible
    assert interval.upper == pytest.approx(RED_COMET_2017_NOMINAL.a_max)
    with pytest.raises(DifferentialDriveDomainError):
        side_candidate_partials(0.0, 0.0, 0.0, DriveSide.RIGHT, P)


def test_left_right_mirror_symmetry_of_algebra():
    rng = random.Random(20260904)
    for _ in range(120):
        v_mps = rng.uniform(0.4, 4.5)
        w = (v_mps / P.cell_pitch_m) ** 2
        kappa = rng.uniform(-2.5, 2.5)
        sigma = rng.uniform(-20.0, 20.0)
        try:
            ar = side_candidate(w, kappa, sigma, DriveSide.RIGHT, P)
            al_mirror = side_candidate(w, -kappa, -sigma, DriveSide.LEFT, P)
            lr = side_acceleration_bounds(w, kappa, sigma, DriveSide.RIGHT, P).lower
            ll_mirror = side_acceleration_bounds(w, -kappa, -sigma, DriveSide.LEFT, P).lower
        except DifferentialDriveDomainError:
            continue
        assert ar == pytest.approx(al_mirror, rel=2e-14, abs=2e-14)
        assert lr == pytest.approx(ll_mirror, rel=2e-14, abs=2e-14)


def test_force_equations_recover_translation_and_yaw_moments():
    rng = random.Random(7)
    for _ in range(100):
        v_mps = rng.uniform(0.5, 4.0)
        w = (v_mps / P.cell_pitch_m) ** 2
        kappa = rng.uniform(-2.5, 2.5)
        sigma = rng.uniform(-18.0, 18.0)
        a = rng.uniform(-40.0, 40.0)
        fr = side_force_demand_n(a, w, kappa, sigma, DriveSide.RIGHT, P)
        fl = side_force_demand_n(a, w, kappa, sigma, DriveSide.LEFT, P)
        alpha = yaw_acceleration(w, kappa, sigma, a)
        assert fr + fl == pytest.approx(P.mass_kg * P.cell_pitch_m * a, rel=5e-14, abs=5e-14)
        assert 0.5 * P.effective_track_m * (fr - fl) == pytest.approx(
            P.yaw_inertia_kg_m2 * alpha, rel=2e-13, abs=2e-13
        )


def test_side_acceleration_matches_differentiated_side_speed_formula():
    w = (2.7 / P.cell_pitch_m) ** 2
    kappa = 1.2
    sigma = -8.0
    a = 12.0
    dt = 2.0e-7
    v = math.sqrt(w)
    for side in (DriveSide.LEFT, DriveSide.RIGHT):
        eps = int(side)
        h = 1.0 + eps * P.beta * kappa
        v_side = v * h
        alpha = yaw_acceleration(w, kappa, sigma, a)
        # dv_side/dt = a +/- beta*alpha, equivalent to the implemented form.
        expected = a + eps * P.beta * alpha
        got = side_acceleration_grid(a, w, kappa, sigma, side, P)
        assert got == pytest.approx(expected, rel=2e-14, abs=2e-14)
        # Small explicit-time check of the kinematic identity.
        v2 = v + a * dt
        k2 = kappa + (sigma * v) * dt
        v_side2 = v2 * (1.0 + eps * P.beta * k2)
        assert (v_side2 - v_side) / dt == pytest.approx(got, rel=2e-5, abs=2e-5)


def test_side_bounds_are_exact_force_capacity_boundaries():
    w = (2.2 / P.cell_pitch_m) ** 2
    kappa = -1.3
    sigma = 11.0
    for side in (DriveSide.LEFT, DriveSide.RIGHT):
        bounds = side_acceleration_bounds(w, kappa, sigma, side, P)
        q = bounds.state.force_capacity_q
        assert side_force_demand_q(bounds.upper, w, kappa, sigma, side, P) == pytest.approx(q, rel=2e-14)
        assert side_force_demand_q(bounds.lower, w, kappa, sigma, side, P) == pytest.approx(-q, rel=2e-14)
        midpoint = 0.5 * (bounds.lower + bounds.upper)
        assert abs(side_force_demand_q(midpoint, w, kappa, sigma, side, P)) <= q


def test_straight_side_envelope_is_looser_than_existing_global_motor_law():
    profile = RED_COMET_2017_NOMINAL
    for v_mps in (0.0, 0.25, 1.0, 2.0, 3.0, 4.0, 5.0):
        w = (v_mps / profile.cell_pitch_m) ** 2
        side_upper = side_acceleration_bounds(w, 0.0, 0.0, DriveSide.RIGHT, P).upper
        global_motor = profile.a_max - profile.b_emf * math.sqrt(w)
        assert side_upper + 1e-12 >= global_motor
        interval = acceleration_interval(w, 0.0, 0.0, P, profile)
        assert interval.upper == pytest.approx(global_motor, rel=1e-13, abs=1e-13)


def test_complete_interval_matches_manual_max_min_construction():
    w = (2.4 / P.cell_pitch_m) ** 2
    kappa = 0.85
    sigma = 7.2
    interval = acceleration_interval(w, kappa, sigma, P)
    grip = math.sqrt(RED_COMET_2017_NOMINAL.mu_g**2 - (w * kappa) ** 2)
    motor = RED_COMET_2017_NOMINAL.a_max - RED_COMET_2017_NOMINAL.b_emf * math.sqrt(w)
    assert interval.lower == max(
        -RED_COMET_2017_NOMINAL.a_brake,
        -grip,
        interval.left.lower,
        interval.right.lower,
    )
    assert interval.upper == min(motor, grip, interval.left.upper, interval.right.upper)


def test_side_candidate_analytic_partials_match_finite_differences():
    rng = random.Random(101)
    seen = 0
    worst = 0.0
    for _ in range(300):
        v_mps = rng.uniform(0.5, 4.6)
        w = (v_mps / P.cell_pitch_m) ** 2
        kappa = rng.uniform(-2.0, 2.0)
        sigma = rng.uniform(-18.0, 18.0)
        side = rng.choice((DriveSide.LEFT, DriveSide.RIGHT))
        try:
            a, aw, ak, ass = side_candidate_partials(w, kappa, sigma, side, P)
            fdw = _fd(lambda x: side_candidate(x, kappa, sigma, side, P), w, rel=7e-7)
            fdk = _fd(lambda x: side_candidate(w, x, sigma, side, P), kappa, rel=7e-7)
            fds = _fd(lambda x: side_candidate(w, kappa, x, side, P), sigma, rel=7e-7)
        except DifferentialDriveDomainError:
            continue
        seen += 1
        worst = max(worst, _scaled_error(aw, fdw), _scaled_error(ak, fdk), _scaled_error(ass, fds))
    assert seen >= 250
    assert worst < 2.5e-7


def test_side_lower_analytic_partials_match_finite_differences():
    w = (2.0 / P.cell_pitch_m) ** 2
    kappa = -0.9
    sigma = 6.0
    for side in (DriveSide.LEFT, DriveSide.RIGHT):
        lower, lw, lk, ls = side_lower_partials(w, kappa, sigma, side, P)
        def L(ww=w, kk=kappa, ss=sigma):
            return side_acceleration_bounds(ww, kk, ss, side, P).lower
        assert lower == pytest.approx(L(), rel=1e-15)
        assert lw == pytest.approx(_fd(lambda x: L(ww=x), w), rel=2e-7, abs=2e-7)
        assert lk == pytest.approx(_fd(lambda x: L(kk=x), kappa), rel=2e-7, abs=2e-7)
        assert ls == pytest.approx(_fd(lambda x: L(ss=x), sigma), rel=2e-7, abs=2e-7)


def test_yaw_kinematics_are_algebraically_consistent():
    w = (3.1 / P.cell_pitch_m) ** 2
    kappa = -1.4
    sigma = 9.0
    a = -5.0
    omega = yaw_rate(w, kappa)
    assert omega == pytest.approx(math.sqrt(w) * kappa)
    assert yaw_acceleration(w, kappa, sigma, a) == pytest.approx(a * kappa + w * sigma)


def test_invalid_charts_fail_closed():
    with pytest.raises(DifferentialDriveDomainError):
        side_state(-1.0, 0.0, DriveSide.RIGHT, P)

    # Right-side kinematic factor h <= 0.
    kappa_h = -1.0 / P.beta - 1.0
    with pytest.raises(DifferentialDriveDomainError):
        side_state(100.0, kappa_h, DriveSide.RIGHT, P)

    # Right-side force coefficient c <= 0.
    kappa_c = -1.0 / P.eta - 1.0
    with pytest.raises(DifferentialDriveDomainError):
        side_state(100.0, kappa_c, DriveSide.RIGHT, P)

    # Rated side free speed exceeded.
    w_fast = (1.01 * P.side_free_speed_grid) ** 2
    with pytest.raises(DifferentialDriveDomainError):
        side_state(w_fast, 0.0, DriveSide.RIGHT, P)


def _segment_value(L, sigma, w0, k0, side, quantity):
    seg = SideEvalSegment(L, sigma, w0, k0, side, P)
    if quantity == "w":
        return seg.w(L)
    if quantity == "time":
        return seg.time(L)
    raise AssertionError(quantity)


def test_side_segment_eval_and_diff_implementations_agree():
    cases = [
        (0.03, 1.5, (2.0 / 0.18) ** 2, 0.2, DriveSide.RIGHT),
        (0.02, -2.0, (2.6 / 0.18) ** 2, -0.4, DriveSide.LEFT),
        (0.015, 5.0, (1.6 / 0.18) ** 2, 0.1, DriveSide.RIGHT),
    ]
    for L, sigma, w0, k0, side in cases:
        e = SideEvalSegment(L, sigma, w0, k0, side, P)
        d = SideSegment(L, sigma, w0, k0, side, P)
        for frac in (0.0, 0.2, 0.55, 1.0):
            ds = frac * L
            assert d.w(ds) == pytest.approx(e.w(ds), rel=3e-11, abs=3e-11)
            assert d.time(ds) == pytest.approx(e.time(ds), rel=3e-11, abs=3e-12)


def test_side_segment_mirror_symmetry():
    L = 0.025
    w0 = (2.2 / 0.18) ** 2
    r = SideSegment(L, 2.7, w0, 0.35, DriveSide.RIGHT, P)
    l = SideSegment(L, -2.7, w0, -0.35, DriveSide.LEFT, P)
    wr, jr, tr, tjr = r.state_time_and_jac(L)
    wl, jl, tl, tjl = l.state_time_and_jac(L)
    assert wr == pytest.approx(wl, rel=2e-12, abs=2e-12)
    assert tr == pytest.approx(tl, rel=2e-12, abs=2e-12)
    # sigma/kappa sensitivities change sign under the mirror; w0/L do not.
    assert jr[0] == pytest.approx(jl[0], rel=2e-12)
    assert jr[1] == pytest.approx(-jl[1], rel=3e-10, abs=3e-10)
    assert jr[2] == pytest.approx(jl[2], rel=2e-12)
    assert jr[3] == pytest.approx(-jl[3], rel=3e-10, abs=3e-10)
    assert tjr[0] == pytest.approx(tjl[0], rel=2e-12)
    assert tjr[1] == pytest.approx(-tjl[1], rel=3e-10, abs=3e-10)
    assert tjr[2] == pytest.approx(tjl[2], rel=3e-10, abs=3e-10)
    assert tjr[3] == pytest.approx(-tjl[3], rel=3e-10, abs=3e-10)


def test_side_segment_endpoint_state_and_time_jacobians_match_fd():
    cases = [
        (0.018, 1.8, (1.9 / 0.18) ** 2, 0.3, DriveSide.RIGHT),
        (0.015, -2.5, (2.4 / 0.18) ** 2, -0.25, DriveSide.LEFT),
        # A genuinely signed SIDE arc: the active-side acceleration begins negative.
        (0.003, 12.0, (2.0 / 0.18) ** 2, 0.8, DriveSide.RIGHT),
    ]

    for L, sigma, w0, k0, side in cases:
        seg = SideSegment(L, sigma, w0, k0, side, P)
        w, wjac, t, tjac = seg.state_time_and_jac(L)
        assert math.isfinite(w) and w > 0.0
        assert math.isfinite(t) and t > 0.0

        # Endpoint derivatives are ordered (L, sigma, w0, k0).
        params = [L, sigma, w0, k0]
        for j, name in enumerate(("L", "sigma", "w0", "k0")):
            x = params[j]
            rel = 4e-6 if name != "L" else 2e-6
            h = rel * max(1.0, abs(x))
            if name == "L":
                h = min(h, 0.15 * L)
            plus = params.copy(); plus[j] = x + h
            minus = params.copy(); minus[j] = x - h
            if name == "L" and minus[j] <= 0.0:
                pytest.skip("degenerate FD length")
            wp = _segment_value(*plus, side, "w")
            wm = _segment_value(*minus, side, "w")
            tp = _segment_value(*plus, side, "time")
            tm = _segment_value(*minus, side, "time")
            fdw = (wp - wm) / (2.0 * h)
            fdt = (tp - tm) / (2.0 * h)
            assert _scaled_error(wjac[j], fdw) < 3.5e-6, (name, wjac[j], fdw)
            assert _scaled_error(tjac[j], fdt) < 3.5e-6, (name, tjac[j], fdt)


def test_side_segment_can_integrate_a_signed_forced_deceleration_arc():
    w0 = (2.0 / P.cell_pitch_m) ** 2
    sigma = 12.0
    k0 = 0.8
    a0 = side_candidate(w0, k0, sigma, DriveSide.RIGHT, P)
    assert a0 < 0.0
    L = 0.003
    seg = SideSegment(L, sigma, w0, k0, DriveSide.RIGHT, P)
    assert seg.w(L) < w0
    assert seg.time(L) > 0.0


def test_side_segment_rejects_out_of_range_distance_and_free_speed_crossing():
    w0 = (2.0 / P.cell_pitch_m) ** 2
    seg = SideSegment(0.02, 0.0, w0, 0.0, DriveSide.RIGHT, P)
    with pytest.raises(ValueError):
        seg.w(-0.1)
    with pytest.raises(ValueError):
        seg.w(0.021)

    # Aggressive curvature evolution can leave the qualified SIDE chart.
    # The reference kernel must fail closed rather than extrapolate through it.
    domain_cross = SideSegment(0.20, -10.0, (1.0 / P.cell_pitch_m) ** 2, -4.5, DriveSide.RIGHT, P)
    with pytest.raises(DifferentialDriveDomainError):
        domain_cross.w(0.20)


def _straight_side_closed_form_endpoint(L: float, w0: float):
    """Independent straight-SIDE oracle from the separable v ODE."""
    from scipy.optimize import brentq

    v0 = math.sqrt(w0)
    vf = P.side_free_speed_grid
    q0 = P.q0
    y0 = v0 / vf
    assert 0.0 <= y0 < 1.0

    def phi(y):
        return -y - math.log1p(-y)

    target = phi(y0) + L * q0 / (vf * vf)
    y1 = brentq(lambda y: phi(y) - target, y0, 1.0 - 1e-13, xtol=2e-14, rtol=2e-14)
    v1 = vf * y1
    time = (vf / q0) * (math.log1p(-y0) - math.log1p(-y1))
    return v1 * v1, time


def test_straight_side_segment_matches_independent_separable_closed_form():
    for v0_mps in (0.25, 1.0, 2.5, 4.5, 7.0):
        w0 = (v0_mps / P.cell_pitch_m) ** 2
        for L in (1e-5, 0.01, 0.2, 1.0):
            ref_w, ref_t = _straight_side_closed_form_endpoint(L, w0)
            seg = SideSegment(L, 0.0, w0, 0.0, DriveSide.RIGHT, P)
            got_w, _wj, got_t, _tj = seg.state_time_and_jac(L)
            assert got_w == pytest.approx(ref_w, rel=2e-11, abs=2e-10)
            assert got_t == pytest.approx(ref_t, rel=2e-11, abs=2e-12)


def test_randomized_side_segment_endpoint_jacobian_stress():
    rng = random.Random(0xDD2017)
    checked = 0
    worst_w = 0.0
    worst_t = 0.0

    for _ in range(120):
        L = rng.uniform(0.0015, 0.035)
        sigma = rng.uniform(-8.0, 8.0)
        k0 = rng.uniform(-1.2, 1.2)
        v0_mps = rng.uniform(0.7, 3.8)
        w0 = (v0_mps / P.cell_pitch_m) ** 2
        side = rng.choice((DriveSide.LEFT, DriveSide.RIGHT))

        try:
            seg = SideSegment(L, sigma, w0, k0, side, P)
            _w, wjac, _t, tjac = seg.state_time_and_jac(L)
        except DifferentialDriveDomainError:
            continue

        vals = [L, sigma, w0, k0]
        ok = True
        for j, x in enumerate(vals):
            h = 2.5e-6 * max(1.0, abs(x))
            if j == 0:
                h = min(h, 0.1 * L)
            plus = vals.copy(); plus[j] += h
            minus = vals.copy(); minus[j] -= h
            if minus[0] <= 0.0:
                ok = False
                break
            try:
                wp = _segment_value(*plus, side, "w")
                wm = _segment_value(*minus, side, "w")
                tp = _segment_value(*plus, side, "time")
                tm = _segment_value(*minus, side, "time")
            except DifferentialDriveDomainError:
                ok = False
                break
            fdw = (wp - wm) / (2.0 * h)
            fdt = (tp - tm) / (2.0 * h)
            worst_w = max(worst_w, _scaled_error(wjac[j], fdw))
            worst_t = max(worst_t, _scaled_error(tjac[j], fdt))
        if ok:
            checked += 1

    assert checked >= 90
    assert worst_w < 7.5e-6
    assert worst_t < 7.5e-6
