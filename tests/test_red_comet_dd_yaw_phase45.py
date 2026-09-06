from __future__ import annotations
import math, random
import pytest

from segment.base import SegmentType
from segment.differential_drive import red_comet_dd_yaw_v1_parameters, acceleration_interval
from segment.physics_profiles import RED_COMET_2017_NOMINAL as PROFILE
from optimization.dd_yaw_speed_profile import DDPassKind, DDCandidateSegment, candidate_value_partials
from optimization.dd_yaw_anchors import build_complete_speed_profile
from optimization.dd_yaw_gradients import (
    time_value_and_raw_gradient, time_value_and_knot_gradient,
    replay_complete_speed_profile, profile_topology_fingerprint,
)
from cdd_yaw import candidate_native, interval_native, segment_native

P=red_comet_dd_yaw_v1_parameters()

def scaled(a,b): return abs(a-b)/max(1.0,abs(a),abs(b))

def test_phase5_native_candidate_and_interval_random_parity():
    rng=random.Random(4605); seen=0; worst=0.0
    combos=[
        (DDPassKind.FORWARD,(SegmentType.MOTOR,SegmentType.GRIP,SegmentType.SIDE_RIGHT,SegmentType.SIDE_LEFT)),
        (DDPassKind.BACKWARD,(SegmentType.BRAKE,SegmentType.GRIP,SegmentType.SIDE_RIGHT,SegmentType.SIDE_LEFT)),
    ]
    for _ in range(650):
        w=rng.uniform(8.0,520.0); k=rng.uniform(-1.4,1.4); sig=rng.uniform(-12,12)
        try:
            iv=acceleration_interval(w,k,sig,P,PROFILE)
            ni=interval_native(w,k,sig,P,PROFILE)
            pv=(iv.lower,iv.upper,iv.margin,iv.motor_upper,iv.grip_magnitude,iv.left.lower,iv.left.upper,iv.right.lower,iv.right.upper)
            worst=max(worst,max(scaled(a,b) for a,b in zip(pv,ni))); seen+=1
        except Exception:
            pass
        for kind,modes in combos:
            for mode in modes:
                try:
                    py=candidate_value_partials(mode,w,k,sig,kind,P,PROFILE)
                    na=candidate_native(mode,w,k,sig,kind,P,PROFILE)
                except Exception:
                    continue
                worst=max(worst,max(scaled(a,b) for a,b in zip(py,na))); seen+=1
    assert seen>2400
    assert worst < 2e-14


def test_phase5_native_smooth_segment_state_time_and_jacobian_parity():
    rng=random.Random(905); seen=0; worst=0.0
    combos=[
        (DDPassKind.FORWARD,(SegmentType.MOTOR,SegmentType.GRIP,SegmentType.SIDE_RIGHT,SegmentType.SIDE_LEFT)),
        (DDPassKind.BACKWARD,(SegmentType.BRAKE,SegmentType.GRIP,SegmentType.SIDE_RIGHT,SegmentType.SIDE_LEFT)),
    ]
    for _ in range(180):
        kind,modes=rng.choice(combos); mode=rng.choice(modes)
        L=rng.uniform(.001,.035); sig=rng.uniform(-7,7); w=rng.uniform(20,320); k=rng.uniform(-.65,.65)
        try:
            py=DDCandidateSegment(L,sig,w,k,mode,kind,P,PROFILE).state_time_and_jac(L)
            na=segment_native(L,sig,w,k,mode,kind,P,PROFILE)
        except Exception:
            continue
        fp=(py[0],)+tuple(py[1])+(py[2],)+tuple(py[3])
        fn=(na[0],)+tuple(na[1])+(na[2],)+tuple(na[3])
        worst=max(worst,max(scaled(a,b) for a,b in zip(fp,fn))); seen+=1
    assert seen>100
    assert worst < 3e-10


def test_phase5_native_complete_profile_matches_python_with_internal_mvc_anchors():
    raw=[.5,10.,.5,-10.]*3; init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=16,anchor_scan=6,cap_scan=16,envelope_root_scan=5)
    py=build_complete_speed_profile(raw,P,PROFILE,segment_backend="python",**kw)
    na=build_complete_speed_profile(raw,P,PROFILE,segment_backend="native",**kw)
    assert na.total_time==pytest.approx(py.total_time,rel=1e-10,abs=4e-11)
    assert na.diagnostics["anchors_with_passes"]==py.diagnostics["anchors_with_passes"]==2
    assert na.diagnostics["envelope_piece_count"]==py.diagnostics["envelope_piece_count"]


def test_phase4_topology_replay_is_exact_at_baseline_internal_anchor_profile():
    raw=[.5,10.,.5,-10.]*3; init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    b=build_complete_speed_profile(raw,P,PROFILE,**kw)
    r=replay_complete_speed_profile(raw,b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    assert r.total_time==pytest.approx(b.total_time,rel=0,abs=2e-14)
    assert profile_topology_fingerprint(r)==profile_topology_fingerprint(b)


def test_phase4_whole_profile_raw_gradient_is_richardson_stable():
    raw=[.5,10.,.5,-10.]; init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    r=time_value_and_raw_gradient(raw,P,PROFILE,relative_step=1e-5,max_shrinks=4,build_kwargs=kw)
    assert r.value>0 and all(math.isfinite(g) for g in r.gradient)
    assert r.diagnostics.max_richardson_disagreement < 2e-7
    # Independent coarser two-point stencil through topology replay.
    for j in range(len(raw)):
        h=3e-5*max(1.0,abs(raw[j])); xp=list(raw); xm=list(raw); xp[j]+=h; xm[j]-=h
        bp=replay_complete_speed_profile(xp,r.build,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
        bm=replay_complete_speed_profile(xm,r.build,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
        fd=(bp.total_time-bm.total_time)/(2*h)
        assert scaled(r.gradient[j],fd) < 2e-5


def test_phase4_knot_pullback_matches_direct_knot_difference_on_straight():
    # one knot: [s1,k1]; k1=0 -> raw=[L,0]
    init=(.8/.18)**2; knots=[1.0,0.0]
    kw=dict(build_kwargs=dict(init_w=init,terminal_w_max=init,pass_scan=22,anchor_scan=8,cap_scan=24,envelope_root_scan=6))
    r=time_value_and_knot_gradient(knots,P,PROFILE,relative_step=2e-5,max_shrinks=3,**kw)
    assert abs(r.gradient[1]) < 2e-8
    h=2e-5
    def f(s):
        b=build_complete_speed_profile([s,0.0],P,PROFILE,init_w=init,terminal_w_max=init,pass_scan=22,anchor_scan=8,cap_scan=24,envelope_root_scan=6)
        return b.total_time
    fd=(f(1+h)-f(1-h))/(2*h)
    assert scaled(r.gradient[0],fd)<2e-6


def test_phase4_analytic_reverse_matches_reference_on_smooth_piece_endpoint_anchor():
    from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic, replay_complete_speed_profile
    raw=[1.0,0.0,0.5,10.0,0.5,-10.0]; init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    b=build_complete_speed_profile(raw,P,PROFILE,**kw)
    value,g=time_value_and_raw_gradient_analytic(b)
    assert value==pytest.approx(b.total_time,rel=0,abs=2e-14)
    d=[.13,.04,-.09,.12,.07,-.08]; h=1e-5
    bp=replay_complete_speed_profile([x+h*q for x,q in zip(raw,d)],b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    bm=replay_complete_speed_profile([x-h*q for x,q in zip(raw,d)],b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    fd=(bp.total_time-bm.total_time)/(2*h); ad=sum(a*q for a,q in zip(g,d))
    assert scaled(ad,fd)<2e-9


def test_phase4_analytic_reverse_matches_reference_on_interior_zero_curvature_anchors():
    from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic, replay_complete_speed_profile
    raw=[]
    for _ in range(3): raw += [.5,8.0,.5,-10.0]
    init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    b=build_complete_speed_profile(raw,P,PROFILE,**kw)
    used={b.passes[e.pass_index].anchor_index for e in b.envelope if b.passes[e.pass_index].anchor_index is not None}
    assert used and all(a.source=="zero_curvature_exact" for a in b.anchors if a.index in used)
    _,g=time_value_and_raw_gradient_analytic(b)
    d=[.2,-.13,-.17,.11,.08,-.07,-.1,.09,.14,-.05,-.12,.04];h=1e-5
    bp=replay_complete_speed_profile([x+h*q for x,q in zip(raw,d)],b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    bm=replay_complete_speed_profile([x-h*q for x,q in zip(raw,d)],b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    fd=(bp.total_time-bm.total_time)/(2*h);ad=sum(a*q for a,q in zip(g,d))
    assert scaled(ad,fd)<2e-9


def test_phase4_analytic_reverse_rejects_nonsmooth_piece_endpoint_cap_tie():
    from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic, DDGradientTopologyError
    raw=[.5,10.,.5,-10.]*3;init=(.6/.18)**2
    b=build_complete_speed_profile(raw,P,PROFILE,init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    with pytest.raises(DDGradientTopologyError,match="nonsmooth left/right cap tie"):
        time_value_and_raw_gradient_analytic(b)


def test_phase5_native_segments_preserve_analytic_reverse_gradient():
    from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic
    raw=[1.0,0.0,0.5,10.0,0.5,-10.0];init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    py=build_complete_speed_profile(raw,P,PROFILE,segment_backend="python",**kw)
    na=build_complete_speed_profile(raw,P,PROFILE,segment_backend="native",**kw)
    vp,gp=time_value_and_raw_gradient_analytic(py);vn,gn=time_value_and_raw_gradient_analytic(na)
    assert vn==pytest.approx(vp,rel=1e-10,abs=5e-11)
    assert max(scaled(a,b) for a,b in zip(gp,gn))<2e-9


def test_phase4_analytic_reverse_matches_reference_on_smooth_local_minimum_anchor():
    from optimization.dd_yaw_gradients import time_value_and_raw_gradient_analytic, replay_complete_speed_profile
    raw=[]
    for _ in range(3): raw += [.5,10.0,.5,-12.0]
    init=(.6/.18)**2
    kw=dict(init_w=init,terminal_w_max=init,pass_scan=18,anchor_scan=7,cap_scan=18,envelope_root_scan=6)
    b=build_complete_speed_profile(raw,P,PROFILE,**kw)
    used={b.passes[e.pass_index].anchor_index for e in b.envelope if b.passes[e.pass_index].anchor_index is not None}
    assert any(a.source=="local_minimum" for a in b.anchors if a.index in used)
    _,g=time_value_and_raw_gradient_analytic(b)
    d=[.2,-.13,-.17,.11,.08,-.07,-.1,.09,.14,-.05,-.12,.04];h=1e-5
    bp=replay_complete_speed_profile([x+h*q for x,q in zip(raw,d)],b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    bm=replay_complete_speed_profile([x-h*q for x,q in zip(raw,d)],b,P,PROFILE,pass_scan=18,cap_scan=18,envelope_root_scan=6,certify=False)
    fd=(bp.total_time-bm.total_time)/(2*h);ad=sum(a*q for a,q in zip(g,d))
    assert scaled(ad,fd)<2e-9
