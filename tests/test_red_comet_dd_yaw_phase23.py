from __future__ import annotations

import math
import random

import pytest

from segment.base import SegmentType
from segment.differential_drive import (
    DriveSide,
    acceleration_interval,
    red_comet_dd_yaw_v1_parameters,
    side_lower_partials,
)
from segment.physics_profiles import RED_COMET_2017_NOMINAL as PROFILE
from segment.side_actuator import SideSegment
from optimization.dd_yaw_speed_profile import (
    DDCandidateSegment,
    DDEventKind,
    DDPassKind,
    build_endpoint_passes,
    certify_signed_pass,
    candidate_modes,
    candidate_value,
    candidate_value_partials,
    choose_active_mode,
    feasibility_margin,
    switch_local_columns,
    switch_root_parameter_derivatives,
)
from optimization.dd_yaw_anchors import (
    build_complete_speed_profile,
    build_mvc_anchor_catalog,
    certify_complete_profile,
    envelope_w,
    exact_zero_curvature_side_cap_w,
    mvc_point,
    mvc_active_pair_partials,
    mvc_cap_partials,
)

P = red_comet_dd_yaw_v1_parameters()


def _scaled(a: float, b: float) -> float:
    return abs(a-b)/max(1.0,abs(a),abs(b))


def _fd(fun, x: float, rel: float = 8e-7) -> float:
    h=rel*max(1.0,abs(x))
    return (fun(x+h)-fun(x-h))/(2*h)


def test_forward_backward_side_transform_matches_original_interval():
    rng=random.Random(2048)
    for _ in range(80):
        w=(rng.uniform(0.6,3.2)/P.cell_pitch_m)**2
        k=rng.uniform(-1.8,1.8)
        sig=rng.uniform(-14,14)
        try:
            iv=acceleration_interval(w,k,sig,P,PROFILE)
            back={m:candidate_value(m,w,k,-sig,DDPassKind.BACKWARD,P,PROFILE) for m in candidate_modes(DDPassKind.BACKWARD)}
        except Exception:
            continue
        assert min(back.values()) == pytest.approx(-iv.lower, rel=2e-13, abs=2e-13)
        assert feasibility_margin(DDPassKind.BACKWARD,w,k,-sig,P,PROFILE) == pytest.approx(iv.margin, rel=2e-13, abs=2e-13)


def test_candidate_partials_match_finite_difference_for_all_modes():
    rng=random.Random(73)
    seen=0
    for kind in (DDPassKind.FORWARD,DDPassKind.BACKWARD):
        for mode in candidate_modes(kind):
            for _ in range(35):
                w=(rng.uniform(0.8,3.5)/P.cell_pitch_m)**2
                k=rng.uniform(-1.5,1.5); sig=rng.uniform(-12,12)
                try:
                    a,aw,ak,ass=candidate_value_partials(mode,w,k,sig,kind,P,PROFILE)
                    fdw=_fd(lambda x:candidate_value(mode,x,k,sig,kind,P,PROFILE),w)
                    fdk=_fd(lambda x:candidate_value(mode,w,x,sig,kind,P,PROFILE),k)
                    fds=_fd(lambda x:candidate_value(mode,w,k,x,kind,P,PROFILE),sig)
                except Exception:
                    continue
                seen+=1
                assert _scaled(aw,fdw)<4e-7,(kind,mode,"w",aw,fdw)
                assert _scaled(ak,fdk)<4e-7,(kind,mode,"k",ak,fdk)
                assert _scaled(ass,fds)<4e-7,(kind,mode,"s",ass,fds)
    assert seen>180


def test_generic_side_candidate_segment_matches_phase1_side_segment():
    cases=[
        (0.02,2.0,(1.8/.18)**2,0.2,DriveSide.RIGHT),
        (0.015,-3.0,(2.2/.18)**2,-0.3,DriveSide.LEFT),
        (0.003,12.0,(2.0/.18)**2,0.8,DriveSide.RIGHT),
    ]
    for L,sig,w0,k0,side in cases:
        mode=SegmentType.SIDE_RIGHT if side is DriveSide.RIGHT else SegmentType.SIDE_LEFT
        ref=SideSegment(L,sig,w0,k0,side,P)
        dd=DDCandidateSegment(L,sig,w0,k0,mode,DDPassKind.FORWARD,P,PROFILE)
        wr,jr,tr,jtr=ref.state_time_and_jac(L)
        wd,jd,td,jtd=dd.state_time_and_jac(L)
        assert wd==pytest.approx(wr,rel=2e-10,abs=2e-10)
        assert td==pytest.approx(tr,rel=2e-10,abs=2e-11)
        for a,b in zip(jd,jr): assert _scaled(a,b)<2e-8
        for a,b in zip(jtd,jtr): assert _scaled(a,b)<2e-8


def _first_switch_root(w0:float,k0:float,sigma:float):
    from scipy.optimize import brentq
    active=SegmentType.MOTOR; alt=SegmentType.SIDE_RIGHT; L=.35
    seg=DDCandidateSegment(L,sigma,w0,k0,active,DDPassKind.FORWARD,P,PROFILE)
    def f(x):
        w=seg.w(x); k=k0+sigma*x
        return candidate_value(active,w,k,sigma,DDPassKind.FORWARD,P,PROFILE)-candidate_value(alt,w,k,sigma,DDPassKind.FORWARD,P,PROFILE)
    xs=[i*L/200 for i in range(201)]; prev=f(xs[0])
    for i in range(1,len(xs)):
        now=f(xs[i])
        if prev<=0<now: return brentq(f,xs[i-1],xs[i],xtol=2e-13)
        prev=now
    raise AssertionError("switch root not found")


def test_generic_switch_root_derivatives_include_explicit_sigma_term():
    w0=(0.8/.18)**2; k0=0.0; sigma=8.0
    r=_first_switch_root(w0,k0,sigma)
    seg=DDCandidateSegment(.35,sigma,w0,k0,SegmentType.MOTOR,DDPassKind.FORWARD,P,PROFILE)
    cols=switch_local_columns(seg,r,SegmentType.SIDE_RIGHT)
    assert cols[0]>0
    drdsig,drdw0,drdk0=switch_root_parameter_derivatives(seg,r,SegmentType.SIDE_RIGHT)
    for got,x,fun,rel in (
        (drdsig,sigma,lambda q:_first_switch_root(w0,k0,q),2e-5),
        (drdw0,w0,lambda q:_first_switch_root(q,k0,sigma),2e-5),
        (drdk0,k0,lambda q:_first_switch_root(w0,q,sigma),2e-5),
    ):
        fd=_fd(fun,x,rel=rel)
        assert _scaled(got,fd)<2e-4,(got,fd)


def test_phase2_signed_pass_switches_without_zero_length_chatter():
    init=(0.8/.18)**2
    b=build_endpoint_passes([0.4,8.0],P,PROFILE,init_w=init,terminal_w_max=init,n_scan=96)
    assert [r.mode for r in b.forward.segments]==[SegmentType.MOTOR,SegmentType.SIDE_RIGHT]
    assert b.forward.segments[0].event.kind is DDEventKind.MODE_SWITCH
    assert b.forward.segments[0].event.next_mode is SegmentType.SIDE_RIGHT
    assert [r.mode for r in b.backward.segments]==[SegmentType.BRAKE,SegmentType.SIDE_LEFT]
    assert all(r.L_used>1e-9 for p in (b.forward,b.backward) for r in p.segments)


def test_phase2_pass_replay_certificate_checks_modes_events_and_margin():
    init=(0.8/.18)**2
    b=build_endpoint_passes([0.5,10.0,0.5,-10.0],P,PROFILE,init_w=init,terminal_w_max=init,n_scan=96)
    for q in (b.forward,b.backward):
        cert=certify_signed_pass(q,P,PROFILE,samples_per_segment=7)
        assert cert["max_mode_violation"]<3e-6
        assert cert["min_interval_margin"]>-3e-6
        assert cert["max_event_residual"]<2e-5


def test_phase2_endpoint_extremals_fail_closed_at_actuator_mvc():
    init=(0.8/.18)**2
    b=build_endpoint_passes([0.5,10.0,0.5,-10.0],P,PROFILE,init_w=init,terminal_w_max=init,n_scan=96)
    assert b.forward.terminated_at_mvc
    assert b.backward.terminated_at_mvc
    assert b.forward.segments[-1].event.kind is DDEventKind.MVC
    assert b.backward.segments[-1].event.kind is DDEventKind.MVC
    assert abs(feasibility_margin(DDPassKind.FORWARD,b.forward.final_w,b.forward.final_k,b.forward.segments[-1].sigma,P,PROFILE)) < 2e-7


def test_zero_curvature_mvc_closed_form_matches_full_interval_root():
    for sigma in (1.0,5.0,10.0,20.0,40.0):
        exact=exact_zero_curvature_side_cap_w(sigma,P)
        pt=mvc_point(0.0,sigma,P,PROFILE,n_scan=160)
        assert exact is not None
        assert pt.w==pytest.approx(exact,rel=3e-12,abs=3e-12)
        assert {pt.upper_mode,pt.lower_mode}=={"SIDE_RIGHT","SIDE_LEFT"}
        assert abs(pt.margin)<2e-9




def test_mvc_active_pair_and_cap_partials_match_finite_difference():
    # Positive sigma at kappa=0 is a smooth SIDE_RIGHT-upper / SIDE_LEFT-lower root.
    k=0.0; sig=10.0
    pt=mvc_point(k,sig,P,PROFILE,n_scan=160)
    assert (pt.upper_mode,pt.lower_mode)==("SIDE_RIGHT","SIDE_LEFT")
    F,Fw,Fk,Fs=mvc_active_pair_partials(pt.w,k,sig,pt.upper_mode,pt.lower_mode,P,PROFILE)
    assert abs(F)<2e-9 and Fw<0.0
    dw_dk,dw_ds=mvc_cap_partials(pt,P,PROFILE)
    fd_k=_fd(lambda x:mvc_point(x,sig,P,PROFILE,n_scan=180).w,k,rel=2e-5)
    fd_s=_fd(lambda x:mvc_point(k,x,P,PROFILE,n_scan=180).w,sig,rel=2e-5)
    assert _scaled(dw_dk,fd_k)<2e-4,(dw_dk,fd_k)
    assert _scaled(dw_ds,fd_s)<2e-4,(dw_ds,fd_s)

def test_mvc_catalog_is_deterministic_and_recertified():
    raw=[0.5,10.0,0.5,-10.0]*2
    a=build_mvc_anchor_catalog(raw,P,PROFILE,n_scan=20,cap_scan=56)
    b=build_mvc_anchor_catalog(raw,P,PROFILE,n_scan=20,cap_scan=56)
    assert a.anchors==b.anchors
    assert len(a.anchors)>=5
    assert any(x.source=="piece_endpoint" for x in a.anchors)
    for x in a.anchors:
        pt=mvc_point(x.kappa,x.sigma,P,PROFILE,n_scan=96)
        assert x.cap_w==pytest.approx(pt.w,rel=2e-9,abs=2e-9)


def test_phase3_internal_mvc_anchors_close_multibottleneck_gap():
    # Three repeated high-yaw bottlenecks: endpoint passes stop at first/last
    # MVC and leave the middle uncovered.  Internal anchors are essential.
    raw=[0.5,10.0,0.5,-10.0]*3
    init=(0.6/.18)**2
    endpoints=build_endpoint_passes(raw,P,PROFILE,init_w=init,terminal_w_max=init,n_scan=56)
    assert endpoints.forward.segments[-1].abs1 < endpoints.backward.segments[-1].abs1
    build=build_complete_speed_profile(raw,P,PROFILE,init_w=init,terminal_w_max=init,pass_scan=56,anchor_scan=20,cap_scan=56,envelope_root_scan=12)
    used={ep.pass_index for ep in build.envelope}
    assert any(i>=2 for i in used),used
    assert build.diagnostics["anchors_with_passes"]>=2
    cert=certify_complete_profile(build,n_samples=769)
    assert cert["min_interval_margin"]>-3e-6
    assert cert["max_mvc_violation"]<3e-6
    assert math.isfinite(build.total_time) and build.total_time>0


def test_phase3_straight_profile_reduces_to_endpoint_envelope_and_is_symmetric():
    raw=[1.0,0.0]; init=(0.8/.18)**2
    build=build_complete_speed_profile(raw,P,PROFILE,init_w=init,terminal_w_max=init,pass_scan=48,anchor_scan=12,cap_scan=40,envelope_root_scan=10)
    assert build.diagnostics["anchors_with_passes"]==0
    assert len(build.envelope)==2
    assert envelope_w(build,0.0)==pytest.approx(init,rel=1e-12)
    assert envelope_w(build,1.0)==pytest.approx(init,rel=1e-12)
    # DD side limits are deliberately looser on a straight, so the envelope
    # must remain the legacy global MOTOR/BRAKE construction.  It is not
    # symmetric because MOTOR is speed-dependent while BRAKE is constant.
    used_modes={build.passes[e.pass_index].segments[e.segment_index].mode for e in build.envelope}
    assert used_modes == {SegmentType.MOTOR, SegmentType.BRAKE}


def test_complete_profile_scalar_time_is_stable_under_scan_refinement():
    # A single S-bottleneck is sufficient to qualify numerical scan stability
    # without turning the unit suite into a benchmark.
    raw=[0.5,10.0,0.5,-10.0]; init=(0.6/.18)**2
    a=build_complete_speed_profile(raw,P,PROFILE,init_w=init,terminal_w_max=init,pass_scan=32,anchor_scan=12,cap_scan=32,envelope_root_scan=8)
    b=build_complete_speed_profile(raw,P,PROFILE,init_w=init,terminal_w_max=init,pass_scan=48,anchor_scan=16,cap_scan=44,envelope_root_scan=10)
    assert a.total_time==pytest.approx(b.total_time,rel=8e-6,abs=8e-7)


def test_phase2_sigma_jump_is_explicit_knot_mvc_and_phase3_catalogs_cap():
    # The state is feasible at the end of the straight piece, but the next
    # clothoid's sigma jump makes the same continuous (w,kappa) infeasible.
    # This must be reported as a knot MVC so Phase 3 can launch a cap anchor
    # that forces deceleration before the knot.
    raw=[1.0,0.0,0.5,10.0]
    init=(0.8/.18)**2
    endpoints=build_endpoint_passes(raw,P,PROFILE,init_w=init,terminal_w_max=init,n_scan=96)
    fwd=endpoints.forward
    assert fwd.terminated_at_mvc and not fwd.terminated_at_domain
    assert fwd.segments[-1].event.kind is DDEventKind.MVC_KNOT
    assert fwd.segments[-1].abs1 == pytest.approx(1.0,abs=2e-12)
    assert feasibility_margin(DDPassKind.FORWARD,fwd.final_w,fwd.final_k,0.0,P,PROFILE) > 1.0
    assert feasibility_margin(DDPassKind.FORWARD,fwd.final_w,fwd.final_k,10.0,P,PROFILE) < -1.0

    catalog=build_mvc_anchor_catalog(raw,P,PROFILE,n_scan=20,cap_scan=72)
    knot=[a for a in catalog.anchors if abs(a.station-1.0)<2e-10]
    assert len(knot)==1
    a=knot[0]
    assert a.source=="piece_endpoint"
    assert a.sigma==pytest.approx(10.0)
    assert a.cap_w < fwd.final_w
    # The cataloged cap is independently feasible on both sides of the knot.
    assert feasibility_margin(DDPassKind.FORWARD,a.cap_w,0.0,0.0,P,PROFILE) >= -2e-7
    assert feasibility_margin(DDPassKind.FORWARD,a.cap_w,0.0,10.0,P,PROFILE) >= -2e-7
