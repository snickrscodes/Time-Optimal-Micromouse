from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from segment.physics_profiles import (
    LEGACY_GRID_V1,
    RED_COMET_2017_DD_YAW_V1,
    RED_COMET_2017_NOMINAL,
)
from segment.physics_identity import (
    DD_YAW_MODEL_REVISION,
    differential_drive_parameters_for_profile,
    physics_model_identity,
    physics_model_signature,
)


def test_phase6_dd_profile_is_explicit_and_scalar_constants_match_nominal():
    p = RED_COMET_2017_DD_YAW_V1
    q = RED_COMET_2017_NOMINAL
    assert p.time_model == "dd_yaw_v1"
    assert (p.mu_g, p.a_brake, p.a_max, p.v_max) == (q.mu_g, q.a_brake, q.a_max, q.v_max)
    assert p.dd_effective_track_m == pytest.approx(0.039)
    assert p.dd_yaw_inertia_scale == pytest.approx(1.0)


def test_phase6_physics_signature_covers_dd_parameters_and_is_deterministic():
    p = RED_COMET_2017_DD_YAW_V1
    a = physics_model_signature(p)
    b = physics_model_signature(p)
    assert a == b and len(a) == 64
    changed = replace(p, dd_effective_track_m=0.035)
    assert physics_model_signature(changed) != a
    ident = physics_model_identity(p)
    assert ident["model_revision"] == DD_YAW_MODEL_REVISION
    assert ident["dd_effective_track_m"] == pytest.approx(0.039)
    assert "optimization/dd_yaw_gradients.py" in ident["source_sha256"]


def test_phase6_dd_parameter_factory_reads_profile_not_hidden_defaults():
    p = replace(RED_COMET_2017_DD_YAW_V1, dd_effective_track_m=0.035, dd_yaw_inertia_scale=0.5)
    dd = differential_drive_parameters_for_profile(p)
    assert dd.effective_track_m == pytest.approx(0.035)
    full = differential_drive_parameters_for_profile(RED_COMET_2017_DD_YAW_V1)
    assert dd.yaw_inertia_kg_m2 == pytest.approx(0.5 * full.yaw_inertia_kg_m2)


def test_phase6_time_dispatch_legacy_is_unchanged(monkeypatch):
    monkeypatch.setenv("AME_PHYSICS_PROFILE", LEGACY_GRID_V1.name)
    monkeypatch.setenv("AME_REVERSE_BACKEND", "python")
    from optimization import scalar_reverse_solver
    from optimization.time_model import evaluate_time_scalar
    raw = [0.8, 0.0]
    kwargs = dict(init_w=0.8, terminal_w_max=0.8, initial_k=0.0, n_scan=24, envelope_scan=12, domain_scan=24)
    old = scalar_reverse_solver.evaluate_time_scalar(raw, **kwargs)
    new = evaluate_time_scalar(raw, **kwargs)
    assert new == pytest.approx(old, rel=0.0, abs=1e-14)


def test_phase6_dd_dispatch_native_python_certificate_and_visual_trace(monkeypatch):
    monkeypatch.setenv("AME_PHYSICS_PROFILE", RED_COMET_2017_DD_YAW_V1.name)
    # The legacy native reverse stack is intentionally not selected for DD/yaw;
    # only the separate runtime-parameterized smooth DD kernel is native.
    monkeypatch.setenv("AME_REVERSE_BACKEND", "python")
    monkeypatch.setenv("AME_DD_SEGMENT_BACKEND", "native")
    from optimization.time_model import evaluate_time_scalar, certify_dd_time_profile
    raw = [1.0, 0.0]
    kwargs = dict(init_w=0.8, terminal_w_max=0.8, initial_k=0.0, n_scan=24, envelope_scan=12, domain_scan=24)
    value = evaluate_time_scalar(raw, **kwargs)
    cert = certify_dd_time_profile(raw, expected_time=value, include_trace=True, trace_samples=31, **kwargs)
    assert cert["certified"] is True
    assert abs(cert["native_minus_python"]) < 2e-9
    trace = cert["visual_trace"]
    assert len(trace["s_grid"]) == 31
    assert trace["s_grid"][0] == pytest.approx(0.0)
    assert trace["s_grid"][-1] == pytest.approx(1.0)
    assert max(abs(x) for x in trace["yaw_rate_rad_s"]) < 1e-12
    assert set(trace["active_mode"]) <= {"MOTOR", "BRAKE"}


def test_phase6_active_basis_request_digest_includes_physics_signature(monkeypatch):
    from planning.active_basis_optimizer import _canonical_request
    from tools.geometry_homotopy.state_machine import HomotopyPolicy
    common = dict(
        cells=((0,0),(1,0)), open_room_spans=(), body_length=0.3, body_height=0.2,
        clearance=0.0, refinement_factor=1, init_w=0.8, terminal_w_max=0.8,
        n_scan=96, envelope_scan=48, domain_scan=96, policy=HomotopyPolicy(),
    )
    monkeypatch.setenv("AME_PHYSICS_PROFILE", LEGACY_GRID_V1.name)
    a = _canonical_request(**common)
    monkeypatch.setenv("AME_PHYSICS_PROFILE", RED_COMET_2017_DD_YAW_V1.name)
    b = _canonical_request(**common)
    assert a["request_digest"] != b["request_digest"]
    assert b["physics_model_signature"] == physics_model_signature(RED_COMET_2017_DD_YAW_V1)
    assert b["physics_model_identity"]["time_model"] == "dd_yaw_v1"


def test_phase6_final_run_default_profile_is_dd_yaw():
    from tools.red_comet import final_run, fixed_route_compare
    assert final_run.DEFAULT_PROFILE == RED_COMET_2017_DD_YAW_V1.name
    assert fixed_route_compare.DEFAULT_PROFILE == RED_COMET_2017_DD_YAW_V1.name


def test_phase6_clarke_tie_gradient_matches_centered_directional_derivative():
    from optimization.dd_yaw_anchors import build_complete_speed_profile
    from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic, replay_complete_speed_profile
    p = RED_COMET_2017_DD_YAW_V1
    dd = differential_drive_parameters_for_profile(p)
    raw = [0.5,10.0,0.5,-10.0]*3
    init=(0.6/p.cell_pitch_m)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    b=build_complete_speed_profile(raw,dd,p,segment_backend="python",**kw)
    _,g=time_value_and_raw_gradient_analytic(b,clarke_ties=True)
    d=[0.12,-0.08,-0.06,0.07,0.05,-0.04,-0.03,0.05,0.08,-0.03,-0.04,0.02]
    h=2e-6
    bp=build_complete_speed_profile([x+h*q for x,q in zip(raw,d)],dd,p,segment_backend="python",**kw)
    bm=build_complete_speed_profile([x-h*q for x,q in zip(raw,d)],dd,p,segment_backend="python",**kw)
    fd=(bp.total_time-bm.total_time)/(2*h)
    ad=sum(a*q for a,q in zip(g,d))
    assert abs(ad-fd)/max(1.0,abs(ad),abs(fd)) < 3e-5
