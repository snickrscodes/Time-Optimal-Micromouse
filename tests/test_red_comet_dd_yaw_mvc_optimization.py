from __future__ import annotations

import math
import random

import pytest

from optimization.dd_yaw_anchors import (
    _mvc_point_python,
    build_complete_speed_profile,
    hard_state_cap_w,
    mvc_point,
)
from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic
from optimization.dd_yaw_mvc_direct import direct_mvc_point_candidate
from segment.physics_identity import differential_drive_parameters_for_profile
from segment.physics_profiles import RED_COMET_2017_DD_YAW_V1

P = differential_drive_parameters_for_profile(RED_COMET_2017_DD_YAW_V1)
Q = RED_COMET_2017_DD_YAW_V1


def _states(n: int, seed: int = 90605):
    r = random.Random(seed)
    out = []
    while len(out) < n:
        k = r.uniform(-8.0, 8.0)
        s = r.uniform(-50.0, 50.0)
        if hard_state_cap_w(k, P, Q) > 1e-8:
            out.append((k, s))
    return out


def test_native_bulk_mvc_scan_matches_frozen_python_reference():
    states = _states(256)
    for k, s in states:
        ref = _mvc_point_python(k, s, P, Q, n_scan=112)
        got = mvc_point(k, s, P, Q, n_scan=112, backend="native_scan")
        assert got.w == pytest.approx(ref.w, rel=3e-12, abs=3e-11)
        assert got.hard_cap_w == pytest.approx(ref.hard_cap_w, rel=0.0, abs=2e-12)
        assert (got.upper_mode, got.lower_mode) == (ref.upper_mode, ref.lower_mode)


def test_direct_active_pair_candidate_matches_scan_reference():
    # Candidate B is an independently derived algebraic oracle.  It is not the
    # production default, but it must identify the same first connected cap.
    for k, s in _states(192, seed=90606):
        hard = hard_state_cap_w(k, P, Q)
        direct = direct_mvc_point_candidate(k, s, P, Q, hard_cap_w=hard)
        assert direct is not None
        ref = mvc_point(k, s, P, Q, n_scan=160, backend="native_scan")
        assert direct[0] == pytest.approx(ref.w, rel=3e-11, abs=3e-10)


def test_native_mvc_catalog_preserves_complete_profile_time_and_gradient():
    raw = [0.5, 10.0, 0.5, -10.0] * 3
    init = (0.6 / Q.cell_pitch_m) ** 2
    kw = dict(
        init_w=init,
        terminal_w_max=init,
        pass_scan=18,
        anchor_scan=7,
        cap_scan=24,
        envelope_root_scan=6,
        segment_backend="python",
    )
    a = build_complete_speed_profile(raw, P, Q, mvc_backend="python", **kw)
    b = build_complete_speed_profile(raw, P, Q, mvc_backend="native_scan", **kw)
    va, ga = time_value_and_raw_gradient_analytic(a, clarke_ties=True)
    vb, gb = time_value_and_raw_gradient_analytic(b, clarke_ties=True)
    assert vb == pytest.approx(va, rel=0.0, abs=2e-10)
    assert len(a.anchors) == len(b.anchors)
    assert [x.source for x in a.anchors] == [x.source for x in b.anchors]
    for x, y in zip(ga, gb):
        assert y == pytest.approx(x, rel=3e-9, abs=3e-9)


def test_direct_backend_is_fail_closed_and_profile_equivalent_on_synthetic_path():
    raw = [0.5, 10.0, 0.5, -10.0] * 2
    init = (0.6 / Q.cell_pitch_m) ** 2
    kw = dict(
        init_w=init,
        terminal_w_max=init,
        pass_scan=18,
        anchor_scan=7,
        cap_scan=24,
        envelope_root_scan=6,
        segment_backend="python",
    )
    ref = build_complete_speed_profile(raw, P, Q, mvc_backend="native_scan", **kw)
    cand = build_complete_speed_profile(raw, P, Q, mvc_backend="direct", **kw)
    assert cand.total_time == pytest.approx(ref.total_time, rel=0.0, abs=3e-10)
    assert len(cand.anchors) == len(ref.anchors)
