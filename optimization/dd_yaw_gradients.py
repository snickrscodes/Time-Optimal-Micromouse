"""Topology-qualified whole-profile gradients for Red Comet DD/yaw v1.

Phase 4 reference implementation.

The local SIDE and switch/MVC derivatives are analytic in the Phase-1/2/3
kernels.  This module supplies an independent whole-profile derivative oracle
for the complete envelope by high-order, topology-locked finite differences.
It is intentionally a *reference* gradient: expensive but fail-closed.  A
perturbation that changes the active profile/anchor topology is rejected rather
than differentiated across a kink.

This gives Phase 5/native work a trustworthy geometry->time gradient oracle
without modifying the historically qualified legacy reverse solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from scipy.optimize import minimize_scalar

from segment.differential_drive import DifferentialDriveParameters
from segment.physics_profiles import PhysicsProfile, RED_COMET_2017_NOMINAL
from .dd_yaw_anchors import (
    DDCompleteProfile, DDMVCAnchor, build_complete_speed_profile, compile_anchor_passes,
    merge_pass_envelope, envelope_time, certify_complete_profile, mvc_point,
)
from .dd_yaw_speed_profile import build_endpoint_passes, geometry_knot_curvatures, validate_raw_parameters
from .geometry_gradients import (
    knot_parameters_to_raw,
    pullback_raw_gradient_to_knot_parameters,
)


class DDGradientError(RuntimeError):
    pass


class DDGradientTopologyError(DDGradientError):
    pass


@dataclass(frozen=True, slots=True)
class DDGradientDiagnostics:
    topology_fingerprint: tuple
    function_evaluations: int
    max_richardson_disagreement: float
    min_step: float
    max_step: float


@dataclass(frozen=True, slots=True)
class DDRawGradientResult:
    value: float
    gradient: tuple[float, ...]
    build: DDCompleteProfile
    diagnostics: DDGradientDiagnostics


@dataclass(frozen=True, slots=True)
class DDKnotGradientResult:
    value: float
    gradient: tuple[float, ...]
    raw_gradient: tuple[float, ...]
    raw_params: tuple[float, ...]
    build: DDCompleteProfile
    diagnostics: DDGradientDiagnostics


def profile_topology_fingerprint(build: DDCompleteProfile) -> tuple:
    """Return a geometry-perturbation-stable active-topology fingerprint.

    Numerical stations/cap values are deliberately omitted.  The fingerprint
    retains pass origin, anchor source/piece, active mode/event sequences and
    the lower-envelope ownership sequence.
    """
    anchors = {a.index: a for a in build.anchors}
    pass_keys=[]
    for p in build.passes:
        if p.anchor_index is None:
            anchor_key=None
        else:
            a=anchors.get(p.anchor_index)
            anchor_key=(a.source,a.piece_index) if a is not None else ("missing",p.anchor_index)
        segs=tuple(
            (r.piece_index,r.mode.value,r.event.kind.name,
             None if r.event.next_mode is None else r.event.next_mode.value)
            for r in p.segments
        )
        pass_keys.append((p.kind.name,p.source,anchor_key,segs))
    env=[]
    for ep in build.envelope:
        p=build.passes[ep.pass_index]
        r=p.segments[ep.segment_index]
        if p.anchor_index is None:
            ak=None
        else:
            a=anchors.get(p.anchor_index)
            ak=(a.source,a.piece_index) if a is not None else ("missing",p.anchor_index)
        env.append((p.kind.name,p.source,ak,r.piece_index,r.mode.value))
    # Compress consecutive identical envelope owners so harmless root splitting
    # does not create a false topology change.
    envc=[]
    for x in env:
        if not envc or envc[-1]!=x:
            envc.append(x)
    return (tuple(pass_keys),tuple(envc))



def _used_anchor_objects(build: DDCompleteProfile):
    ids=[]
    for ep in build.envelope:
        aid=build.passes[ep.pass_index].anchor_index
        if aid is not None and aid not in ids:
            ids.append(aid)
    amap={a.index:a for a in build.anchors}
    return [amap[i] for i in ids if i in amap]


def _reproject_anchor(raw, base: DDCompleteProfile, a: DDMVCAnchor, params, profile, cap_scan: int):
    raw=validate_raw_parameters(raw); ks=geometry_knot_curvatures(raw)
    n=len(raw)//2
    starts=[0.0]
    for i in range(n): starts.append(starts[-1]+raw[2*i])
    braw=base.raw_params
    if a.source=="piece_endpoint":
        # Recover the geometry-knot identity from the baseline local coordinate.
        Lb=braw[2*a.piece_index]
        if abs(a.local_s)<=2e-8*max(1.0,Lb): knot=a.piece_index
        elif abs(a.local_s-Lb)<=2e-8*max(1.0,Lb): knot=a.piece_index+1
        else:
            # Station-based fallback for a previously deduplicated endpoint.
            bs=[0.0]
            for i in range(len(braw)//2): bs.append(bs[-1]+braw[2*i])
            knot=min(range(1,len(bs)-1),key=lambda j:abs(bs[j]-a.station))
        if not 0<knot<n:
            raise DDGradientTopologyError("piece-endpoint anchor moved to path endpoint")
        k=ks[knot]; sl=raw[2*(knot-1)+1]; sr=raw[2*knot+1]
        pl=mvc_point(k,sl,params,profile,n_scan=cap_scan)
        pr=mvc_point(k,sr,params,profile,n_scan=cap_scan)
        if pl.w<=pr.w:
            pt,sig,pi,lx=pl,sl,knot-1,raw[2*(knot-1)]
        else:
            pt,sig,pi,lx=pr,sr,knot,0.0
        return DDMVCAnchor(a.index,starts[knot],pi,lx,k,sig,pt.w,pt.margin,pt.upper_mode,pt.lower_mode,a.source)
    if a.source=="zero_curvature_exact":
        i=a.piece_index; L=raw[2*i]; sig=raw[2*i+1]; k0=ks[i]
        if sig==0.0: raise DDGradientTopologyError("zero-curvature anchor lost nonzero sigma")
        x=-k0/sig
        if not (0.0<x<L): raise DDGradientTopologyError("zero-curvature anchor left its piece")
        pt=mvc_point(0.0,sig,params,profile,n_scan=cap_scan)
        return DDMVCAnchor(a.index,starts[i]+x,i,x,0.0,sig,pt.w,pt.margin,pt.upper_mode,pt.lower_mode,a.source)
    if a.source=="local_minimum":
        i=a.piece_index; L=raw[2*i]; sig=raw[2*i+1]; k0=ks[i]
        # Follow the same local well rather than globally re-searching the piece.
        xb=a.local_s; frac=xb/max(1e-15,base.raw_params[2*i]); xguess=min(max(frac*L,0.0),L)
        radius=max(2e-4,min(0.22*L,0.08*L+2e-3))
        lo=max(1e-10,xguess-radius); hi=min(L-1e-10,xguess+radius)
        if not hi>lo: raise DDGradientTopologyError("local-minimum anchor lost interior bracket")
        def fun(x): return mvc_point(math.fma(sig,float(x),k0),sig,params,profile,n_scan=max(48,cap_scan)).w
        opt=minimize_scalar(fun,bounds=(lo,hi),method="bounded",options={"xatol":2e-11,"maxiter":100})
        if not opt.success: raise DDGradientTopologyError("local-minimum MVC reprojection failed")
        x=float(opt.x)
        if x<=lo+2e-7*max(1.0,L) or x>=hi-2e-7*max(1.0,L):
            raise DDGradientTopologyError("local-minimum MVC moved outside qualified local well")
        k=math.fma(sig,x,k0);pt=mvc_point(k,sig,params,profile,n_scan=cap_scan)
        return DDMVCAnchor(a.index,starts[i]+x,i,x,k,sig,pt.w,pt.margin,pt.upper_mode,pt.lower_mode,a.source)
    # Active-pair boundaries are genuine MVC kinks, not missing smooth algebra.
    raise DDGradientTopologyError(f"nondifferentiable/unqualified anchor source {a.source!r}")


def replay_complete_speed_profile(
    raw_params, baseline: DDCompleteProfile, params, profile=RED_COMET_2017_NOMINAL, *,
    pass_scan: int=48, cap_scan: int=64, envelope_root_scan: int=12,
    certify: bool=True,
):
    """Replay a known smooth complete-profile topology without global anchor discovery."""
    raw=validate_raw_parameters(raw_params)
    endpoints=build_endpoint_passes(raw,params,profile,init_w=baseline.init_w,
        terminal_w_max=baseline.terminal_w_max,n_scan=pass_scan)
    passes=[endpoints.forward,endpoints.backward]; anchors=[]
    for a0 in _used_anchor_objects(baseline):
        a=_reproject_anchor(raw,baseline,a0,params,profile,cap_scan)
        aps=compile_anchor_passes(raw,a,params,profile,n_scan=pass_scan)
        if not aps: raise DDGradientTopologyError(f"reprojected anchor {a.source}@{a.station} cannot release")
        anchors.append(a); passes.extend(aps)
    T=math.fsum(raw[0::2])
    env=merge_pass_envelope(passes,T,root_scan=envelope_root_scan)
    out=DDCompleteProfile(tuple(raw),params,profile,endpoints.forward,endpoints.backward,anchors,passes,env,0.0,T,baseline.init_w,baseline.terminal_w_max)
    out.total_time=envelope_time(out)
    if certify:
        cert=certify_complete_profile(out,n_samples=257)
        out.diagnostics.update(cert)
    return out

def _build(raw, params, profile, kwargs):
    return build_complete_speed_profile(raw,params,profile,**kwargs)


def _five_point(values: tuple[float,float,float,float], h: float) -> float:
    fm2,fm1,fp1,fp2=values
    return (fm2-8.0*fm1+8.0*fp1-fp2)/(12.0*h)


def time_value_and_raw_gradient(
    raw_params: Sequence[float],
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    relative_step: float = 3.0e-5,
    minimum_step: float = 2.0e-7,
    max_shrinks: int = 6,
    richardson_tol: float = 4.0e-4,
    build_kwargs: dict | None = None,
) -> DDRawGradientResult:
    """Return complete-profile time and a topology-qualified raw gradient.

    Uses two 5-point stencils (h and h/2) and Richardson comparison.  Every
    stencil member must preserve the baseline active topology.  If a component
    sits on a nonsmooth profile transition, the function raises rather than
    manufacturing a misleading derivative.
    """
    raw=tuple(float(x) for x in raw_params)
    if len(raw)%2:
        raise ValueError("raw_params must have [L,sigma,...] layout")
    if not (math.isfinite(relative_step) and relative_step>0 and math.isfinite(minimum_step) and minimum_step>0):
        raise ValueError("finite-difference steps must be positive and finite")
    kwargs={} if build_kwargs is None else dict(build_kwargs)
    base=_build(raw,params,profile,kwargs)
    fp0=profile_topology_fingerprint(base)
    neval=1; grad=[]; disagreements=[]; steps=[]

    cache: dict[tuple[float,...], tuple[float,tuple]]={raw:(base.total_time,fp0)}
    def ev(x):
        nonlocal neval
        key=tuple(x)
        if key in cache: return cache[key]
        # Scalar topology is discovered once. Perturbations replay the baseline
        # anchor identities/envelope, analogous to the legacy scalar-build ->
        # differentiable-replay architecture.
        replay_kwargs={k:v for k,v in kwargs.items() if k in {"pass_scan","cap_scan","envelope_root_scan"}}
        b=replay_complete_speed_profile(key,base,params,profile,certify=False,**replay_kwargs); neval+=1
        out=(float(b.total_time),profile_topology_fingerprint(b)); cache[key]=out; return out

    for j,x0 in enumerate(raw):
        scale=max(1.0,abs(x0))
        h=max(minimum_step,relative_step*scale)
        if j%2==0:
            h=min(h,0.20*x0)  # keep length perturbations strictly positive
        accepted=False; last_reason=""
        for _ in range(max_shrinks+1):
            if h<minimum_step*0.999:
                break
            def stencil(hh):
                vals=[]
                for mult in (-2.0,-1.0,1.0,2.0):
                    xx=list(raw); xx[j]=x0+mult*hh
                    if j%2==0 and xx[j]<=0:
                        return None,"nonpositive length"
                    try:
                        v,fp=ev(xx)
                    except Exception as exc:
                        return None,f"build failure: {type(exc).__name__}: {exc}"
                    if fp!=fp0:
                        return None,"topology change"
                    vals.append(v)
                return tuple(vals),""
            s1,why1=stencil(h)
            s2,why2=stencil(0.5*h)
            if s1 is None or s2 is None:
                last_reason=why1 or why2; h*=0.5; continue
            d1=_five_point(s1,h); d2=_five_point(s2,0.5*h)
            # For a 5-point central derivative the leading error is O(h^4).
            dr=d2+(d2-d1)/15.0
            dis=abs(d2-d1)/max(1.0,abs(dr),abs(d1),abs(d2))
            if not (math.isfinite(dr) and math.isfinite(dis)):
                last_reason="nonfinite stencil"; h*=0.5; continue
            if dis>richardson_tol:
                last_reason=f"Richardson disagreement {dis:.3e}"; h*=0.5; continue
            grad.append(float(dr)); disagreements.append(float(dis)); steps.append(float(h)); accepted=True; break
        if not accepted:
            raise DDGradientTopologyError(
                f"raw component {j} has no smooth topology-qualified derivative near {x0}: {last_reason}"
            )
    diag=DDGradientDiagnostics(
        fp0,neval,max(disagreements,default=0.0),min(steps,default=0.0),max(steps,default=0.0)
    )
    return DDRawGradientResult(float(base.total_time),tuple(grad),base,diag)


def time_value_and_knot_gradient(
    knot_params: Sequence[float],
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    initial_k: float = 0.0,
    initial_s: float = 0.0,
    **kwargs,
) -> DDKnotGradientResult:
    raw=tuple(knot_parameters_to_raw(knot_params,initial_k=initial_k,initial_s=initial_s))
    rr=time_value_and_raw_gradient(raw,params,profile,**kwargs)
    kg=tuple(pullback_raw_gradient_to_knot_parameters(
        knot_params,rr.gradient,initial_k=initial_k,initial_s=initial_s
    ))
    return DDKnotGradientResult(rr.value,kg,rr.gradient,raw,rr.build,rr.diagnostics)


__all__=[
    "DDGradientError","DDGradientTopologyError","DDGradientDiagnostics",
    "DDRawGradientResult","DDKnotGradientResult","profile_topology_fingerprint","replay_complete_speed_profile",
    "time_value_and_raw_gradient","time_value_and_knot_gradient",
]

# ---------------------------------------------------------------------------
# Analytic reverse replay for the currently qualified smooth DD topology.
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from .dd_yaw_speed_profile import DDPassKind as _PK, DDEventKind as _EK, record_local as _record_local
from .dd_yaw_anchors import mvc_active_pair_partials as _mvc_pair_partials, mvc_cap_partials as _mvc_cap_partials, DDMVCPoint as _MVCPoint
from segment.differential_drive import acceleration_interval as _accel_interval

@_dataclass(slots=True)
class _Extra:
    aL: float=0.0
    aw0: float=0.0
    ak0: float=0.0
    asigma: float=0.0
    aabs0: float=0.0

class _RawAcc:
    def __init__(self,n): self.n=n; self.g=[0.0]*(2*n); self.diff=[0.0]*(n+1)
    def length(self,i,a): self.g[2*i]+=a
    def sigma(self,i,a): self.g[2*i+1]+=a
    def traversal_sigma(self,rec,a): self.sigma(rec.piece_index,a if rec.kind is _PK.FORWARD else -a)
    def prefix_ex(self,i,a):
        if a and i>0:self.diff[0]+=a;self.diff[i]-=a
    def prefix_in(self,i,a):
        if a:self.diff[0]+=a;self.diff[i+1]-=a
    def abs0(self,rec,a):
        if rec.kind is _PK.FORWARD:self.prefix_ex(rec.piece_index,a)
        else:self.prefix_in(rec.piece_index,a)
    def end(self,a):
        if a:self.diff[0]+=a;self.diff[self.n]-=a
    def finish(self):
        z=self.g[:]; r=0.0
        for i in range(self.n): r+=self.diff[i];z[2*i]+=r
        return z


def _add_knot_k_adjoint(raw, grad, knot_index: int, adj: float):
    if not adj:return
    for i in range(knot_index):
        L=raw[2*i]; sig=raw[2*i+1]
        grad[2*i]+=adj*sig; grad[2*i+1]+=adj*L


def _active_pair_at_state(w,k,sigma,params,profile):
    iv=_accel_interval(w,k,sigma,params,profile)
    up={"MOTOR":iv.motor_upper,"GRIP":iv.grip_magnitude,"SIDE_RIGHT":iv.right.upper,"SIDE_LEFT":iv.left.upper}
    lo={"BRAKE":-profile.a_brake,"GRIP":-iv.grip_magnitude,"SIDE_RIGHT":iv.right.lower,"SIDE_LEFT":iv.left.lower}
    uo=("MOTOR","GRIP","SIDE_RIGHT","SIDE_LEFT"); lo_order=("BRAKE","GRIP","SIDE_RIGHT","SIDE_LEFT")
    um=min(uo,key=lambda x:(up[x],uo.index(x))); lm=max(lo_order,key=lambda x:(lo[x],-lo_order.index(x)))
    return um,lm,iv.margin


def _mvc_local_columns(rec, params, profile):
    phys_sigma=rec.direction*rec.sigma
    um,lm,margin=_active_pair_at_state(rec.w1,rec.k1,phys_sigma,params,profile)
    F,Fw,Fk,Fs=_mvc_pair_partials(rec.w1,rec.k1,phys_sigma,um,lm,params,profile)
    if abs(F)>3e-5*max(1.0,abs(rec.w1)):
        raise DDGradientTopologyError(f"analytic MVC replay requires smooth H=0 root, got margin={margin}, F={F}")
    _w,jw=rec.seg.w_and_jac(rec.L_used)
    jk=(rec.sigma,rec.L_used,0.0,1.0)
    cols=[]
    for i in range(4):
        explicit=Fs*rec.direction if i==1 else 0.0
        cols.append(Fw*jw[i]+Fk*jk[i]+explicit)
    if not math.isfinite(cols[0]) or abs(cols[0])<1e-12:
        raise DDGradientTopologyError("analytic MVC root is nontransverse")
    return tuple(cols)


def _reverse_pass_analytic(build: DDCompleteProfile, pass_index: int, extras: list[_Extra]):
    p=build.passes[pass_index]; n=len(build.raw_params)//2; acc=_RawAcc(n)
    amap={a.index:a for a in build.anchors}; anchor=amap.get(p.anchor_index) if p.anchor_index is not None else None
    interior_anchor=anchor is not None and anchor.source in {"zero_curvature_exact","local_minimum"}
    anchor_x_adj=0.0
    aw=ak=0.0; current=None; adj_offset_after=0.0
    for idx in range(len(p.segments)-1,-1,-1):
        rec=p.segments[idx]; ex=extras[idx]
        if current!=rec.traversal_index: current=rec.traversal_index; adj_offset_after=0.0
        aw0,ak0,asig=ex.aw0,ex.ak0,ex.asigma
        if ex.aabs0:
            if interior_anchor and rec.traversal_index==0:
                # First partial traversal begins at S_anchor = prefix(piece)+x.
                acc.prefix_ex(rec.piece_index,ex.aabs0); anchor_x_adj+=ex.aabs0
            else:
                acc.abs0(rec,ex.aabs0)
        adj_offset_before=rec.direction*ex.aabs0
        _w,jac=rec.seg.w_and_jac(rec.L_used)
        a0=aw*jac[0]+ak*jac[4];a1=aw*jac[1]+ak*jac[5];a2=aw*jac[2]+ak*jac[6];a3=aw*jac[3]+ak*jac[7]
        adj_L=ex.aL+a0+adj_offset_after;adj_offset_before+=adj_offset_after
        if rec.event.kind in (_EK.PIECE_END,_EK.MVC_KNOT):
            if interior_anchor and rec.traversal_index==0:
                # Forward partial length = raw_L-x; backward partial length=x.
                if rec.kind is _PK.FORWARD:
                    acc.length(rec.piece_index,adj_L); anchor_x_adj-=adj_L
                else:
                    anchor_x_adj+=adj_L
            else:
                acc.length(rec.piece_index,adj_L)
            adj_offset_before-=adj_L
            asig+=a1;aw0+=a2;ak0+=a3
        elif rec.event.kind is _EK.MODE_SWITCH:
            cols=rec.event.local_columns
            if cols is None: raise DDGradientTopologyError("mode-switch replay missing event columns")
            if abs(cols[0])<1e-12: raise DDGradientTopologyError("mode-switch replay is nontransverse")
            sc=adj_L/cols[0];asig+=a1-sc*cols[1];aw0+=a2-sc*cols[2];ak0+=a3-sc*cols[3]
        elif rec.event.kind is _EK.MVC:
            cols=_mvc_local_columns(rec,build.params,build.profile);sc=adj_L/cols[0]
            asig+=a1-sc*cols[1];aw0+=a2-sc*cols[2];ak0+=a3-sc*cols[3]
        else:
            raise DDGradientTopologyError(f"analytic reverse does not accept terminal event {rec.event.kind.name}")
        acc.traversal_sigma(rec,asig);aw,ak=aw0,ak0;adj_offset_after=adj_offset_before
    return acc.finish(),(aw,ak),anchor_x_adj

def _seed_envelope_piece(build, ep, ex: _Extra):
    rec=build.passes[ep.pass_index].segments[ep.segment_index]
    l0=_record_local(rec,ep.abs0);l1=_record_local(rec,ep.abs1)
    if l0<=l1:lo,hi,lo0=l0,l1,True
    else:lo,hi,lo0=l1,l0,False
    if hi-lo<=1e-14:return 0.0,0.0,0.0
    Tlo,Jlo=rec.seg.time_and_jac(lo);Thi,Jhi=rec.seg.time_and_jac(hi)
    ex.asigma+=Jhi[1]-Jlo[1];ex.aw0+=Jhi[2]-Jlo[2];ex.ak0+=Jhi[3]-Jlo[3]
    alo=-Jlo[0];ahi=Jhi[0]
    if lo0:a0,a1=alo,ahi
    else:a0,a1=ahi,alo
    ex.aabs0+=-rec.direction*(a0+a1)
    return Thi-Tlo,rec.direction*a0,rec.direction*a1


def _piece_endpoint_anchor_adjoint(build, anchor, aw: float, ak: float, grad: list[float], *, clarke_ties: bool=False):
    raw=build.raw_params; n=len(raw)//2; Lb=raw[2*anchor.piece_index]
    if abs(anchor.local_s)<=2e-8*max(1.0,Lb): knot=anchor.piece_index; sig_index=anchor.piece_index
    elif abs(anchor.local_s-Lb)<=2e-8*max(1.0,Lb): knot=anchor.piece_index+1; sig_index=anchor.piece_index
    else: raise DDGradientTopologyError("piece_endpoint anchor is not at a geometry knot")
    if not 0<knot<n: raise DDGradientTopologyError("internal anchor unexpectedly at global endpoint")
    # Determine the side of the knot whose sigma defines this recertified cap.
    # A left/right cap tie is a genuine nonsmooth minimum and must fail closed.
    from .dd_yaw_anchors import mvc_point as _mvc_point_local
    pl=_mvc_point_local(anchor.kappa,raw[2*(knot-1)+1],build.params,build.profile,n_scan=96)
    pr=_mvc_point_local(anchor.kappa,raw[2*knot+1],build.params,build.profile,n_scan=96)
    tie_scale=max(1.0,abs(pl.w),abs(pr.w))
    tied = abs(pl.w-pr.w)<=2e-8*tie_scale and abs(raw[2*(knot-1)+1]-raw[2*knot+1])>1e-10
    if tied:
        if not clarke_ties:
            raise DDGradientTopologyError("piece-endpoint MVC has a nonsmooth left/right cap tie")
        # w_cap = min(w_left, w_right).  At an exact tie the Clarke
        # subdifferential is the convex hull of the two active gradients.
        # Use the symmetric midpoint so the production objective preserves
        # left/right mirror symmetry and matches centered directional FD.
        ptl=_MVCPoint(pl.w,anchor.kappa,raw[2*(knot-1)+1],pl.margin,pl.upper_mode,pl.lower_mode,0.0)
        ptr=_MVCPoint(pr.w,anchor.kappa,raw[2*knot+1],pr.margin,pr.upper_mode,pr.lower_mode,0.0)
        wkl,wsl=_mvc_cap_partials(ptl,build.params,build.profile)
        wkr,wsr=_mvc_cap_partials(ptr,build.params,build.profile)
        _add_knot_k_adjoint(raw,grad,knot,ak+0.5*aw*(wkl+wkr))
        grad[2*(knot-1)+1]+=0.5*aw*wsl
        grad[2*knot+1]+=0.5*aw*wsr
        return
    sig_index=knot-1 if pl.w<pr.w else knot
    if abs(raw[2*sig_index+1]-anchor.sigma)>2e-6*max(1.0,abs(anchor.sigma)):
        raise DDGradientTopologyError("piece-endpoint active sigma side changed since scalar build")
    pt=_MVCPoint(anchor.cap_w,anchor.kappa,anchor.sigma,anchor.margin,anchor.upper_mode,anchor.lower_mode,0.0)
    wk,ws=_mvc_cap_partials(pt,build.params,build.profile)
    _add_knot_k_adjoint(raw,grad,knot,ak+aw*wk)
    grad[2*sig_index+1]+=aw*ws



def _zero_curvature_anchor_adjoint(build, anchor, aw: float, ak: float, ax: float, grad: list[float]):
    # The reprojected anchor is defined by k_i + sigma_i*x = 0.  Therefore
    # k(anchor) is identically zero and the pass-initial k adjoint has no direct
    # contribution; it is absorbed by motion of x.
    raw=build.raw_params; i=anchor.piece_index; sig=raw[2*i+1]
    if sig==0.0: raise DDGradientTopologyError("zero-curvature anchor lost sigma")
    ks=geometry_knot_curvatures(raw); ki=ks[i]; x=-ki/sig
    if not (0.0<x<raw[2*i]): raise DDGradientTopologyError("zero-curvature anchor left its geometry piece")
    pt=_MVCPoint(anchor.cap_w,0.0,sig,anchor.margin,anchor.upper_mode,anchor.lower_mode,0.0)
    _wk,ws=_mvc_cap_partials(pt,build.params,build.profile)
    # x = -ki/sig.
    aki=-ax/sig
    _add_knot_k_adjoint(raw,grad,i,aki)
    grad[2*i+1]+=ax*ki/(sig*sig) + aw*ws
    # ``ak`` intentionally multiplies d k(anchor)/dp = 0 for this anchor class.



def _local_minimum_anchor_state(k0: float, sig: float, L: float, x_hint: float, params, profile):
    radius=max(2e-4,min(0.22*L,0.08*L+2e-3));lo=max(1e-10,x_hint-radius);hi=min(L-1e-10,x_hint+radius)
    if not hi>lo: raise DDGradientTopologyError("local-minimum anchor has no local bracket")
    def fun(x): return mvc_point(math.fma(sig,float(x),k0),sig,params,profile,n_scan=64).w
    opt=minimize_scalar(fun,bounds=(lo,hi),method="bounded",options={"xatol":1e-11,"maxiter":120})
    if not opt.success: raise DDGradientTopologyError("local-minimum local solve failed")
    x=float(opt.x);k=math.fma(sig,x,k0);pt=mvc_point(k,sig,params,profile,n_scan=80)
    if x<=lo+1e-7*max(1.0,L) or x>=hi-1e-7*max(1.0,L):raise DDGradientTopologyError("local-minimum anchor left local well")
    return x,k,pt.w


def _local_minimum_anchor_adjoint(build, anchor, aw: float, ak: float, ax: float, grad: list[float]):
    raw=build.raw_params;i=anchor.piece_index;L=raw[2*i];sig=raw[2*i+1];ks=geometry_knot_curvatures(raw);k0=ks[i];x0=anchor.local_s
    # Local 5-point derivatives are cheap: no complete profile or anchor catalog
    # rebuild, only the one scalar MVC well is followed.
    def deriv(which: str):
        xbase=k0 if which=="k" else sig; h=2e-5*max(1.0,abs(xbase))
        vals=[]
        for m in (-2,-1,1,2):
            kk=k0+m*h if which=="k" else k0; ss=sig+m*h if which=="s" else sig
            vals.append(_local_minimum_anchor_state(kk,ss,L,x0,build.params,build.profile))
        return tuple(( -vals[3][j]+8*vals[2][j]-8*vals[1][j]+vals[0][j])/(12*h) for j in range(3))
    dxk,dkk,dwk=deriv("k");dxs,dks,dws=deriv("s")
    coeff_k=ax*dxk+ak*dkk+aw*dwk; coeff_s=ax*dxs+ak*dks+aw*dws
    _add_knot_k_adjoint(raw,grad,i,coeff_k);grad[2*i+1]+=coeff_s


def time_value_and_raw_gradient_analytic(build: DDCompleteProfile, *, clarke_ties: bool=False) -> tuple[float, tuple[float,...]]:
    """Analytic reverse of a qualified complete DD profile.

    Current Phase-4 promotion supports endpoint passes and smooth H=0 / knot
    MVCs whose active internal anchors are geometry ``piece_endpoint`` anchors.
    Other anchor families fail closed and remain covered by the independent
    topology-locked finite-difference oracle above.
    """
    n=len(build.raw_params)//2; extras=[[_Extra() for _ in p.segments] for p in build.passes];bound=_RawAcc(n);value=0.0;T=build.total_length
    for ep in build.envelope:
        ex=extras[ep.pass_index][ep.segment_index];v,a0,a1=_seed_envelope_piece(build,ep,ex);value+=v
        if abs(ep.abs0-T)<=1e-10*max(1.0,T):bound.end(a0)
        if abs(ep.abs1-T)<=1e-10*max(1.0,T):bound.end(a1)
    grad=bound.finish(); amap={a.index:a for a in build.anchors}
    for pi,p in enumerate(build.passes):
        pg,(aw,ak),ax=_reverse_pass_analytic(build,pi,extras[pi])
        for j in range(2*n):grad[j]+=pg[j]
        if p.anchor_index is None:
            if p.kind is _PK.BACKWARD:_add_knot_k_adjoint(build.raw_params,grad,n,ak)
            # forward initial k and both endpoint speeds are fixed.
        else:
            a=amap.get(p.anchor_index)
            if a is None:raise DDGradientTopologyError("anchor pass missing anchor record")
            if a.source=="piece_endpoint":
                _piece_endpoint_anchor_adjoint(build,a,aw,ak,grad,clarke_ties=clarke_ties)
            elif a.source=="zero_curvature_exact":
                _zero_curvature_anchor_adjoint(build,a,aw,ak,ax,grad)
            elif a.source=="local_minimum":
                _local_minimum_anchor_adjoint(build,a,aw,ak,ax,grad)
            else:
                raise DDGradientTopologyError(f"analytic anchor adjoint not yet promoted for {a.source!r}")
    return float(value),tuple(float(x) for x in grad)


try:
    __all__ += ["time_value_and_raw_gradient_analytic"]
except NameError:
    pass
