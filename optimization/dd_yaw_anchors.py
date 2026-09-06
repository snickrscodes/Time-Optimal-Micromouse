"""Actuator-MVC anchors and complete Python DD/yaw speed profiles (Phase 3).

The signed endpoint extremals in :mod:`optimization.dd_yaw_speed_profile` can
terminate when the admissible acceleration interval collapses.  This module
builds the missing internal maximum-velocity-curve (MVC) anchors, launches
forward/backward extremals from those bottlenecks, and forms the lower envelope
of all candidate passes.

This remains a scalar/reference implementation.  It is deliberately numerical
and independently recertifies every anchor against the full acceleration
interval; native parity and the reverse geometry gradient are later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from segment.base import SegmentType
from segment.differential_drive import (
    DifferentialDriveDomainError,
    DifferentialDriveParameters,
    DriveSide,
    acceleration_interval,
    side_acceleration_bounds,
    side_candidate_partials,
    side_lower_partials,
)
from segment.physics_profiles import PhysicsProfile, RED_COMET_2017_NOMINAL

from .dd_yaw_speed_profile import (
    DDPassKind,
    DDProfileError,
    DDScalarPass,
    DDTraversalPiece,
    build_endpoint_passes,
    build_signed_pass,
    geometry_knot_curvatures,
    pass_record_at,
    record_interval,
    record_time_between,
    record_w,
    state_domain_margins,
    total_length,
    validate_raw_parameters,
)


@dataclass(frozen=True, slots=True)
class DDMVCPoint:
    w: float
    kappa: float
    sigma: float
    margin: float
    upper_mode: str
    lower_mode: str
    hard_cap_w: float


@dataclass(frozen=True, slots=True)
class DDMVCAnchor:
    index: int
    station: float
    piece_index: int
    local_s: float
    kappa: float
    sigma: float
    cap_w: float
    margin: float
    upper_mode: str
    lower_mode: str
    source: str


@dataclass(slots=True)
class DDAnchorCatalog:
    anchors: list[DDMVCAnchor]
    scan_points: int
    exact_zero_curvature_anchors: int = 0
    local_minimum_anchors: int = 0
    active_pair_boundary_anchors: int = 0
    endpoint_anchors: int = 0


@dataclass(frozen=True, slots=True)
class DDEnvelopePiece:
    pass_index: int
    segment_index: int
    abs0: float
    abs1: float


@dataclass(frozen=True, slots=True)
class DDEnvelopeWitness:
    abs0: float
    abs1: float
    kind: str
    magnitude: float = 0.0


@dataclass(slots=True)
class DDCompleteProfile:
    raw_params: tuple[float, ...]
    params: DifferentialDriveParameters
    profile: PhysicsProfile
    endpoint_forward: DDScalarPass
    endpoint_backward: DDScalarPass
    anchors: list[DDMVCAnchor]
    passes: list[DDScalarPass]
    envelope: list[DDEnvelopePiece]
    total_time: float
    total_length: float
    init_w: float
    terminal_w_max: float
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)



def _grip_partials(w: float, kappa: float, profile: PhysicsProfile):
    q=w*kappa; g2=profile.mu_g*profile.mu_g-q*q
    if not g2>0.0:
        raise DDProfileError("MVC derivative requested at friction-domain edge")
    g=math.sqrt(g2)
    return g,-q*kappa/g,-q*w/g,0.0


def mvc_bound_partials(
    label: str,
    *,
    upper: bool,
    w: float,
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> tuple[float,float,float,float]:
    """Return value and ``(w,kappa,sigma)`` partials for one MVC bound."""
    if upper and label=="MOTOR":
        root=math.sqrt(w)
        return profile.a_max-profile.b_emf*root,-profile.b_emf/(2*root),0.0,0.0
    if (not upper) and label=="BRAKE":
        return -profile.a_brake,0.0,0.0,0.0
    if label=="GRIP":
        g,gw,gk,gs=_grip_partials(w,kappa,profile)
        return (g,gw,gk,gs) if upper else (-g,-gw,-gk,-gs)
    if label=="SIDE_RIGHT":
        return side_candidate_partials(w,kappa,sigma,DriveSide.RIGHT,params) if upper else side_lower_partials(w,kappa,sigma,DriveSide.RIGHT,params)
    if label=="SIDE_LEFT":
        return side_candidate_partials(w,kappa,sigma,DriveSide.LEFT,params) if upper else side_lower_partials(w,kappa,sigma,DriveSide.LEFT,params)
    raise ValueError(f"invalid MVC {'upper' if upper else 'lower'} label {label!r}")


def mvc_active_pair_partials(
    w: float,
    kappa: float,
    sigma: float,
    upper_mode: str,
    lower_mode: str,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> tuple[float,float,float,float]:
    """Return ``F=U-L`` and its smooth active-pair partial derivatives."""
    u=mvc_bound_partials(upper_mode,upper=True,w=w,kappa=kappa,sigma=sigma,params=params,profile=profile)
    l=mvc_bound_partials(lower_mode,upper=False,w=w,kappa=kappa,sigma=sigma,params=params,profile=profile)
    return tuple(u[i]-l[i] for i in range(4))


def mvc_cap_partials(
    point: DDMVCPoint,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> tuple[float,float]:
    """Return smooth ``(dw_cap/dkappa, dw_cap/dsigma)`` by implicit differentiation."""
    if point.upper_mode in {"DOMAIN","INFEASIBLE"} or point.lower_mode in {"DOMAIN","INFEASIBLE"}:
        raise DDProfileError("hard-domain MVC does not have active-pair derivatives")
    F,Fw,Fk,Fs=mvc_active_pair_partials(point.w,point.kappa,point.sigma,point.upper_mode,point.lower_mode,params,profile)
    if abs(F)>2e-6*max(1.0,abs(point.w)) or not math.isfinite(Fw) or abs(Fw)<1e-12:
        raise DDProfileError(f"MVC active-pair root not qualified for differentiation: F={F}, Fw={Fw}")
    return -Fk/Fw,-Fs/Fw


def _interval_components(
    w: float,
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile,
):
    iv=acceleration_interval(w,kappa,sigma,params,profile)
    upper={
        "MOTOR": iv.motor_upper,
        "GRIP": iv.grip_magnitude,
        "SIDE_RIGHT": iv.right.upper,
        "SIDE_LEFT": iv.left.upper,
    }
    lower={
        "BRAKE": -profile.a_brake,
        "GRIP": -iv.grip_magnitude,
        "SIDE_RIGHT": iv.right.lower,
        "SIDE_LEFT": iv.left.lower,
    }
    # Deterministic stable ties.
    uorder=("MOTOR","GRIP","SIDE_RIGHT","SIDE_LEFT")
    lorder=("BRAKE","GRIP","SIDE_RIGHT","SIDE_LEFT")
    um=min(uorder,key=lambda x:(upper[x],uorder.index(x)))
    lm=max(lorder,key=lambda x:(lower[x],-lorder.index(x)))
    return iv,um,lm


def hard_state_cap_w(
    kappa: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> float:
    """Largest squared speed inside the smooth friction/side-speed chart."""
    caps=[]
    ak=abs(kappa)
    if ak>0.0:
        caps.append(profile.mu_g/ak)
    for side in (DriveSide.RIGHT,DriveSide.LEFT):
        eps=int(side)
        h=1.0+eps*params.beta*kappa
        c=1.0+eps*params.eta*kappa
        if h<=params.h_floor or c<=params.c_floor:
            return 0.0
        caps.append((params.side_free_speed_grid/h)**2)
    # On kappa=0 this is the side free-speed chart cap.  Else friction often wins.
    cap=min(caps) if caps else params.side_free_speed_grid**2
    return max(0.0,float(cap))


def exact_zero_curvature_side_cap_w(
    sigma: float,
    params: DifferentialDriveParameters,
) -> float | None:
    """Closed-form side-vs-side yaw-transient MVC at kappa=0."""
    z=abs(float(sigma))
    if z==0.0 or params.eta==0.0:
        return None
    a=params.eta*z
    b=params.q0/params.side_free_speed_grid
    c=-params.q0
    disc=b*b-4*a*c
    if disc<=0.0: return None
    y=(-b+math.sqrt(disc))/(2*a)
    return y*y if y>0.0 else None


def _mvc_point_python(
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    n_scan: int = 160,
) -> DDMVCPoint:
    """Return the highest speed in the connected feasible interval from w=0.

    The full interval is piecewise smooth because its active upper/lower labels
    can switch.  A deterministic scan brackets the first ``H=0`` crossing and
    Brent refinement certifies the root.  The returned state is independently
    replayed through the complete interval, never trusted from an active pair.
    """
    kappa=float(kappa); sigma=float(sigma)
    hard=hard_state_cap_w(kappa,params,profile)
    if not hard>0.0:
        return DDMVCPoint(0.0,kappa,sigma,-math.inf,"DOMAIN","DOMAIN",hard)
    # Stay just inside strict side-free/friction charts during scalar evaluation.
    hi=math.nextafter(hard,0.0)
    if not hi>0.0: hi=0.5*hard
    lo=max(1e-12,1e-12*max(1.0,hi))

    def H(w:float)->float:
        try: return acceleration_interval(w,kappa,sigma,params,profile).margin
        except DifferentialDriveDomainError: return -math.inf

    h0=H(lo)
    if not math.isfinite(h0) or h0<0.0:
        return DDMVCPoint(lo,kappa,sigma,h0,"INFEASIBLE","INFEASIBLE",hard)

    # Hybrid quadratic/linear scan gives resolution near both low-speed yaw caps
    # and high-speed motor/free-speed caps.
    ys=np.linspace(math.sqrt(lo),math.sqrt(hi),max(24,int(n_scan))+1)
    ws=ys*ys
    prev_w=lo; prev_h=h0; bracket=None
    for w in ws[1:]:
        w=float(w); h=H(w)
        if prev_h>=0.0 and (not math.isfinite(h) or h<=0.0):
            bracket=(prev_w,w); break
        prev_w,prev_h=w,h

    if bracket is None:
        cap=hi
    else:
        a,b=bracket
        def finite_h(x):
            h=H(x)
            return h if math.isfinite(h) else -1e300
        try: cap=float(brentq(finite_h,a,b,xtol=4e-12,rtol=8*np.finfo(float).eps,maxiter=120))
        except ValueError: cap=float(a)

    # Classify from an infinitesimal feasible interior state; exact active pair
    # at H=0 can be numerically tied across several candidates.
    wc=max(lo,math.nextafter(cap,0.0))
    try:
        iv,um,lm=_interval_components(wc,kappa,sigma,params,profile)
        margin=acceleration_interval(cap,kappa,sigma,params,profile).margin
    except DifferentialDriveDomainError:
        iv,um,lm=_interval_components(wc,kappa,sigma,params,profile)
        margin=iv.margin
    return DDMVCPoint(cap,kappa,sigma,float(margin),um,lm,hard)


def mvc_points(
    kappas: Sequence[float],
    sigmas: Sequence[float],
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    n_scan: int = 160,
    backend: str = "python",
) -> list[DDMVCPoint]:
    """Evaluate one or more connected-component MVC points.

    ``python`` is the frozen Phase-3 reference. ``native_scan`` performs the
    identical sqrt(w)-scan/refinement policy inside the independently qualified
    DD native library and returns the same public point semantics.
    """
    ks=[float(x) for x in kappas]; ss=[float(x) for x in sigmas]
    if len(ks)!=len(ss): raise ValueError("kappas and sigmas must have equal length")
    b=str(backend).strip().lower()
    if b=="python":
        return [_mvc_point_python(k,s,params,profile,n_scan=n_scan) for k,s in zip(ks,ss)]
    if b=="native_scan":
        from cdd_yaw._binding import mvc_scan_bulk_native
        rows=mvc_scan_bulk_native(ks,ss,params,profile,n_scan=n_scan)
        return [DDMVCPoint(w,k,s,m,u,l,h) for (k,s),(w,m,h,u,l,_status) in zip(zip(ks,ss),rows)]
    if b=="direct":
        from .dd_yaw_mvc_direct import direct_mvc_point_candidate
        out=[]
        for k,sig in zip(ks,ss):
            hard=hard_state_cap_w(k,params,profile)
            cand=direct_mvc_point_candidate(k,sig,params,profile,hard_cap_w=hard)
            if cand is None:
                # Candidate B is fail-closed: ambiguous algebraic cases fall back
                # to Candidate A's independently qualified scan, never to a guess.
                out.extend(mvc_points([k],[sig],params,profile,n_scan=n_scan,backend="native_scan")); continue
            cap,_roots=cand
            hi=math.nextafter(hard,0.0) if hard>0.0 else 0.0
            lo=max(1e-12,1e-12*max(1.0,hi)) if hi>0.0 else 0.0
            wc=max(lo,math.nextafter(cap,0.0))
            try:
                _iv,um,lm=_interval_components(wc,k,sig,params,profile)
                margin=acceleration_interval(cap,k,sig,params,profile).margin
            except DifferentialDriveDomainError:
                _iv,um,lm=_interval_components(wc,k,sig,params,profile)
                margin=_iv.margin
            out.append(DDMVCPoint(float(cap),k,sig,float(margin),um,lm,float(hard)))
        return out
    raise ValueError(f"unknown DD MVC backend {backend!r}")


def mvc_point(
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    n_scan: int = 160,
    backend: str = "python",
) -> DDMVCPoint:
    return mvc_points([kappa],[sigma],params,profile,n_scan=n_scan,backend=backend)[0]


def _piece_starts(raw: Sequence[float]) -> list[float]:
    raw=validate_raw_parameters(raw)
    out=[]; s=0.0
    for i in range(len(raw)//2):
        out.append(s); s+=raw[2*i]
    return out


def _anchor_candidate(
    station:float,piece_index:int,local_s:float,kappa:float,sigma:float,source:str,
    params:DifferentialDriveParameters,profile:PhysicsProfile,n_scan:int,mvc_backend:str="python",
):
    pt=mvc_point(kappa,sigma,params,profile,n_scan=n_scan,backend=mvc_backend)
    return (station,piece_index,local_s,kappa,sigma,pt,source)


def build_mvc_anchor_catalog(
    raw_params: Sequence[float],
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    initial_k: float = 0.0,
    n_scan: int = 96,
    cap_scan: int = 144,
    dedup_tol: float = 2.0e-8,
    mvc_backend: str = "python",
) -> DDAnchorCatalog:
    raw=validate_raw_parameters(raw_params)
    ks=geometry_knot_curvatures(raw,initial_k=initial_k)
    starts=_piece_starts(raw)
    candidates=[]
    counts={"zero":0,"min":0,"tie":0,"endpoint":0}

    for i in range(len(raw)//2):
        L=raw[2*i]; sigma=raw[2*i+1]; k0=ks[i]; start=starts[i]
        xs=np.linspace(0.0,L,max(16,int(n_scan))+1)
        kvals=[math.fma(sigma,float(x),k0) for x in xs]
        pts=mvc_points(kvals,[sigma]*len(kvals),params,profile,n_scan=cap_scan,backend=mvc_backend)
        caps=[pt.w for pt in pts]; pairs=[(pt.upper_mode,pt.lower_mode) for pt in pts]

        # Exact/closed-form zero-curvature candidate, always independently recertified.
        if sigma!=0.0:
            xz=-k0/sigma
            if dedup_tol < xz < L-dedup_tol:
                wz=exact_zero_curvature_side_cap_w(sigma,params)
                if wz is not None:
                    c=_anchor_candidate(start+xz,i,xz,0.0,sigma,"zero_curvature_exact",params,profile,cap_scan,mvc_backend)
                    # The exact side-side root can be looser than another global pair;
                    # mvc_point is authoritative.  Record source only if close enough,
                    # otherwise this remains a useful generic zero-curvature witness.
                    candidates.append(c); counts["zero"]+=1

        # Interior local minima of the complete cap.
        for j in range(1,len(xs)-1):
            scale=max(1.0,abs(caps[j-1]),abs(caps[j]),abs(caps[j+1]))
            strict_tol=5.0e-8*scale
            if caps[j] < caps[j-1]-strict_tol and caps[j] < caps[j+1]-strict_tol:
                a=float(xs[j-1]); b=float(xs[j+1])
                def fun(x):
                    k=math.fma(sigma,float(x),k0)
                    return mvc_point(k,sigma,params,profile,n_scan=max(64,cap_scan//2),backend=mvc_backend).w
                opt=minimize_scalar(fun,bounds=(a,b),method="bounded",options={"xatol":2e-10,"maxiter":80})
                x=float(opt.x if opt.success else xs[j]); k=math.fma(sigma,x,k0)
                candidates.append(_anchor_candidate(start+x,i,x,k,sigma,"local_minimum",params,profile,cap_scan,mvc_backend)); counts["min"]+=1

        # Active-pair changes are nondifferentiable cap points and must be retained
        # even when the sampled cap is monotone through them.
        for j in range(1,len(xs)):
            if pairs[j]!=pairs[j-1]:
                lo=float(xs[j-1]); hi=float(xs[j]); pair_lo=pairs[j-1]
                # Deterministic bisection on the discrete active-pair classifier.
                for _ in range(48):
                    mid=0.5*(lo+hi); k=math.fma(sigma,mid,k0)
                    pt=mvc_point(k,sigma,params,profile,n_scan=max(64,cap_scan//2),backend=mvc_backend)
                    if (pt.upper_mode,pt.lower_mode)==pair_lo: lo=mid
                    else: hi=mid
                x=0.5*(lo+hi); k=math.fma(sigma,x,k0)
                candidates.append(_anchor_candidate(start+x,i,x,k,sigma,"active_pair_boundary",params,profile,cap_scan,mvc_backend)); counts["tie"]+=1

    # Internal geometry knots are MVC candidates only when the sigma jump makes
    # the knot a genuine local bottleneck.  Since H depends explicitly on
    # sigma, the cap can be discontinuous across a knot; the admissible knot
    # speed is the minimum of the left- and right-piece limits.
    for j in range(1, len(raw)//2):
        station=starts[j]; k=ks[j]
        sig_l=raw[2*(j-1)+1]; sig_r=raw[2*j+1]
        pt_l=mvc_point(k,sig_l,params,profile,n_scan=cap_scan,backend=mvc_backend)
        pt_r=mvc_point(k,sig_r,params,profile,n_scan=cap_scan,backend=mvc_backend)
        chosen=(pt_l,sig_l,j-1,raw[2*(j-1)]) if pt_l.w <= pt_r.w else (pt_r,sig_r,j,0.0)
        pt,sig,piece_i,local_x=chosen
        # Compare against small one-sided interior witnesses.
        L_l=raw[2*(j-1)]; L_r=raw[2*j]
        dl=min(1e-4*max(1.0,L_l),0.02*L_l)
        dr=min(1e-4*max(1.0,L_r),0.02*L_r)
        kl=math.fma(-sig_l,dl,k); kr=math.fma(sig_r,dr,k)
        wl=mvc_point(kl,sig_l,params,profile,n_scan=max(48,cap_scan//2),backend=mvc_backend).w
        wr=mvc_point(kr,sig_r,params,profile,n_scan=max(48,cap_scan//2),backend=mvc_backend).w
        scale=max(1.0,abs(pt.w),abs(wl),abs(wr))
        if pt.w <= wl + 2e-6*scale and pt.w <= wr + 2e-6*scale:
            candidates.append((station,piece_i,local_x,k,sig,pt,"piece_endpoint"))
            counts["endpoint"]+=1

    # Deduplicate by global station, retaining the most restrictive recertified cap.
    candidates.sort(key=lambda z:(z[0],z[5].w,z[6]))
    dedup=[]
    for c in candidates:
        if dedup and abs(c[0]-dedup[-1][0])<=dedup_tol:
            if c[5].w < dedup[-1][5].w: dedup[-1]=c
        else: dedup.append(c)

    anchors=[]
    for idx,c in enumerate(dedup):
        station,piece_i,x,k,sigma,pt,source=c
        if not (pt.w>0.0 and math.isfinite(pt.w)): continue
        # At hard chart-only caps margin may be small positive rather than exactly 0;
        # keep them because they are still first-class state-domain speed limits.
        anchors.append(DDMVCAnchor(idx,float(station),piece_i,float(x),float(k),float(sigma),float(pt.w),float(pt.margin),pt.upper_mode,pt.lower_mode,source))

    return DDAnchorCatalog(
        anchors=anchors,scan_points=int(n_scan),
        exact_zero_curvature_anchors=counts["zero"],
        local_minimum_anchors=counts["min"],
        active_pair_boundary_anchors=counts["tie"],
        endpoint_anchors=counts["endpoint"],
    )


def _split_forward_from_station(raw:Sequence[float],station:float,*,initial_k:float=0.0):
    raw=validate_raw_parameters(raw); starts=_piece_starts(raw); ks=geometry_knot_curvatures(raw,initial_k=initial_k)
    T=math.fsum(raw[0::2]); tol=2e-10*max(1.0,T)
    if station< -tol or station>T+tol: raise ValueError("anchor station outside path")
    station=min(max(station,0.0),T); pieces=[]; k_at=None
    for i in range(len(raw)//2):
        a=starts[i]; L=raw[2*i]; b=a+L; sig=raw[2*i+1]
        if station>b+tol: continue
        if k_at is None:
            x=min(max(station-a,0.0),L); k_at=math.fma(sig,x,ks[i]); rem=L-x
            if rem>tol: pieces.append(DDTraversalPiece(rem,sig,i,station,1.0))
        else:
            pieces.append(DDTraversalPiece(L,sig,i,a,1.0))
    return pieces,k_at


def _split_backward_from_station(raw:Sequence[float],station:float,*,initial_k:float=0.0):
    raw=validate_raw_parameters(raw); starts=_piece_starts(raw); ks=geometry_knot_curvatures(raw,initial_k=initial_k)
    T=math.fsum(raw[0::2]); tol=2e-10*max(1.0,T)
    station=min(max(float(station),0.0),T); pieces=[]; k_at=None
    for i in range(len(raw)//2-1,-1,-1):
        a=starts[i]; L=raw[2*i]; b=a+L; sig=raw[2*i+1]
        if station<a-tol: continue
        if k_at is None:
            x=min(max(station-a,0.0),L); k_at=math.fma(sig,x,ks[i]); rem=x
            if rem>tol: pieces.append(DDTraversalPiece(rem,-sig,i,station,-1.0))
        else:
            pieces.append(DDTraversalPiece(L,-sig,i,b,-1.0))
    return pieces,k_at


def compile_anchor_passes(
    raw_params:Sequence[float],anchor:DDMVCAnchor,params:DifferentialDriveParameters,
    profile:PhysicsProfile=RED_COMET_2017_NOMINAL,*,initial_k:float=0.0,n_scan:int=96,
    segment_backend:str="python",
)->list[DDScalarPass]:
    out=[]
    fp,kf=_split_forward_from_station(raw_params,anchor.station,initial_k=initial_k)
    if fp:
        try:
            p=build_signed_pass(fp,anchor.cap_w,kf,DDPassKind.FORWARD,params,profile,n_scan=n_scan,initial_station=anchor.station,source="mvc_anchor",anchor_index=anchor.index,allow_initial_mvc_release=True,segment_backend=segment_backend)
            if p.segments: out.append(p)
        except (DDProfileError,DifferentialDriveDomainError,ValueError): pass
    bp,kb=_split_backward_from_station(raw_params,anchor.station,initial_k=initial_k)
    if bp:
        try:
            p=build_signed_pass(bp,anchor.cap_w,kb,DDPassKind.BACKWARD,params,profile,n_scan=n_scan,initial_station=anchor.station,source="mvc_anchor",anchor_index=anchor.index,allow_initial_mvc_release=True,segment_backend=segment_backend)
            if p.segments: out.append(p)
        except (DDProfileError,DifferentialDriveDomainError,ValueError): pass
    return out


def _covering_records(passes:list[DDScalarPass],station:float,tol:float=2e-10):
    out=[]
    for pi,p in enumerate(passes):
        rec=pass_record_at(p,station,tol=tol)
        if rec is not None: out.append((pi,p.segments.index(rec),rec))
    return out


def _pair_roots(rec_a,rec_b,a:float,b:float,*,n_scan:int=24):
    if b<=a: return []
    def f(x): return record_w(rec_a,x)-record_w(rec_b,x)
    xs=np.linspace(a,b,max(8,int(n_scan))+1); vals=[]
    for x in xs:
        try: vals.append(f(float(x)))
        except Exception: vals.append(math.nan)
    roots=[]
    for i in range(1,len(xs)):
        x0=float(xs[i-1]); x1=float(xs[i]); y0=vals[i-1]; y1=vals[i]
        if not (math.isfinite(y0) and math.isfinite(y1)): continue
        if y0==0.0: roots.append(x0)
        if y0*y1<0.0:
            try: roots.append(float(brentq(f,x0,x1,xtol=3e-12,rtol=8*np.finfo(float).eps)))
            except ValueError: pass
    return roots


def coverage_witnesses(
    passes:list[DDScalarPass],total_L:float,*,tol:float=3e-9,
)->list[DDEnvelopeWitness]:
    intervals=[]
    for p in passes:
        for r in p.segments:
            lo,hi=record_interval(r)
            if hi-lo>tol: intervals.append((max(0.0,lo),min(total_L,hi)))
    if not intervals:
        return [DDEnvelopeWitness(0.0,total_L,"coverage",total_L)]
    intervals.sort(); merged=[]
    for lo,hi in intervals:
        if not merged or lo>merged[-1][1]+tol: merged.append([lo,hi])
        else: merged[-1][1]=max(merged[-1][1],hi)
    out=[]; x=0.0
    for lo,hi in merged:
        if lo>x+tol: out.append(DDEnvelopeWitness(x,lo,"coverage",lo-x))
        x=max(x,hi)
    if x<total_L-tol: out.append(DDEnvelopeWitness(x,total_L,"coverage",total_L-x))
    return out


def envelope_continuity_witnesses(
    envelope:list[DDEnvelopePiece],passes:list[DDScalarPass],*,tol_w:float=2e-6,
)->list[DDEnvelopeWitness]:
    out=[]
    for i in range(1,len(envelope)):
        left=envelope[i-1]; right=envelope[i]
        if abs(left.abs1-right.abs0)>2e-9: continue
        s=0.5*(left.abs1+right.abs0)
        lr=passes[left.pass_index].segments[left.segment_index]
        rr=passes[right.pass_index].segments[right.segment_index]
        try:
            wl=record_w(lr,s); wr=record_w(rr,s)
        except Exception:
            continue
        scale=max(1.0,abs(wl),abs(wr)); jump=abs(wl-wr)
        if jump>tol_w*scale:
            eps=max(2e-8,2e-7*max(1.0,abs(s)))
            out.append(DDEnvelopeWitness(max(0.0,s-eps),s+eps,"continuity",jump))
    return out


def select_anchor_batch(
    witnesses:list[DDEnvelopeWitness],anchors:list[DDMVCAnchor],active:set[int],
)->list[DDMVCAnchor]:
    inactive=[a for a in anchors if a.index not in active]
    selected=[]; ids=set()
    for w in witnesses:
        inside=[a for a in inactive if w.abs0-1e-10<=a.station<=w.abs1+1e-10]
        if inside:
            a=min(inside,key=lambda q:(q.cap_w,q.station,q.index))
        elif inactive:
            mid=0.5*(w.abs0+w.abs1)
            a=min(inactive,key=lambda q:(abs(q.station-mid),q.cap_w,q.index))
        else:
            continue
        if a.index not in ids:
            selected.append(a); ids.add(a.index)
    return selected


def merge_pass_envelope(
    passes:list[DDScalarPass],total_L:float,*,root_scan:int=24,tol:float=3e-10,
)->list[DDEnvelopePiece]:
    bounds={0.0,float(total_L)}
    for p in passes:
        for r in p.segments:
            lo,hi=record_interval(r); bounds.add(max(0.0,min(total_L,lo))); bounds.add(max(0.0,min(total_L,hi)))
    coarse=sorted(bounds); refined=set(coarse)
    for bi in range(1,len(coarse)):
        a,b=coarse[bi-1],coarse[bi]
        if b-a<=tol: continue
        mid=0.5*(a+b); cov=_covering_records(passes,mid,tol=tol)
        for i in range(len(cov)):
            for j in range(i+1,len(cov)):
                refined.update(_pair_roots(cov[i][2],cov[j][2],a,b,n_scan=root_scan))
    xs=sorted(x for x in refined if -tol<=x<=total_L+tol)
    pieces=[]
    for i in range(1,len(xs)):
        a=max(0.0,xs[i-1]); b=min(total_L,xs[i])
        if b-a<=tol: continue
        mid=0.5*(a+b); cov=_covering_records(passes,mid,tol=tol)
        if not cov:
            raise DDProfileError(f"DD MVC candidates leave uncovered interval [{a},{b}]")
        best=min(cov,key=lambda q:(record_w(q[2],mid),q[0],q[1]))
        ep=DDEnvelopePiece(best[0],best[1],a,b)
        if pieces and pieces[-1].pass_index==ep.pass_index and pieces[-1].segment_index==ep.segment_index and abs(pieces[-1].abs1-ep.abs0)<=tol:
            prev=pieces[-1]; pieces[-1]=DDEnvelopePiece(prev.pass_index,prev.segment_index,prev.abs0,ep.abs1)
        else: pieces.append(ep)
    return pieces


def envelope_w(profile_build:DDCompleteProfile,station:float)->float:
    T=profile_build.total_length; s=min(max(float(station),0.0),T); tol=3e-10*max(1.0,T)
    for ep in profile_build.envelope:
        if ep.abs0-tol<=s<=ep.abs1+tol:
            return record_w(profile_build.passes[ep.pass_index].segments[ep.segment_index],s)
    raise DDProfileError(f"envelope does not cover station {station}")


def envelope_time(profile_build:DDCompleteProfile)->float:
    t=0.0
    for ep in profile_build.envelope:
        rec=profile_build.passes[ep.pass_index].segments[ep.segment_index]
        t+=record_time_between(rec,ep.abs0,ep.abs1)
    return t


def certify_complete_profile(build:DDCompleteProfile,*,n_samples:int=1025,margin_tol:float=2e-6)->dict[str,float]:
    min_h=math.inf; max_cap_violation=0.0; min_w=math.inf
    for s in np.linspace(0.0,build.total_length,max(65,int(n_samples))):
        s=float(s); w=envelope_w(build,s); min_w=min(min_w,w)
        # Determine geometry piece / kappa / physical sigma.
        accum=0.0
        for i in range(len(build.raw_params)//2):
            L=build.raw_params[2*i]; sig=build.raw_params[2*i+1]
            if s<=accum+L+2e-12:
                ks=geometry_knot_curvatures(build.raw_params)[i]
                x=min(max(s-accum,0.0),L); k=math.fma(sig,x,ks); sigma=sig; break
            accum+=L
        try:
            iv=acceleration_interval(w,k,sigma,build.params,build.profile); h=iv.margin
        except DifferentialDriveDomainError:
            h=-math.inf
        min_h=min(min_h,h)
        cap=mvc_point(k,sigma,build.params,build.profile,n_scan=96).w
        max_cap_violation=max(max_cap_violation,w-cap)
    if not min_w>0.0: raise DDProfileError("complete DD envelope reaches nonpositive speed")
    if min_h < -margin_tol: raise DDProfileError(f"complete DD envelope violates acceleration interval: min H={min_h}")
    if max_cap_violation > margin_tol*max(1.0,max(envelope_w(build,s) for s in (0.0,build.total_length))):
        raise DDProfileError(f"complete DD envelope exceeds recertified MVC by {max_cap_violation}")
    return {"min_w":min_w,"min_interval_margin":min_h,"max_mvc_violation":max_cap_violation}


def build_complete_speed_profile(
    raw_params:Sequence[float],params:DifferentialDriveParameters,
    profile:PhysicsProfile=RED_COMET_2017_NOMINAL,*,init_w:float|None=None,
    terminal_w_max:float|None=None,initial_k:float=0.0,pass_scan:int=96,
    anchor_scan:int=72,cap_scan:int=112,envelope_root_scan:int=20,
    segment_backend:str="python",
    mvc_backend:str="python",
)->DDCompleteProfile:
    raw=validate_raw_parameters(raw_params)
    if init_w is None: init_w=max(1e-3,(0.05*profile.v_max)**2)
    if terminal_w_max is None: terminal_w_max=init_w
    endpoints=build_endpoint_passes(raw,params,profile,init_w=init_w,terminal_w_max=terminal_w_max,initial_k=initial_k,n_scan=pass_scan,segment_backend=segment_backend)
    catalog=build_mvc_anchor_catalog(raw,params,profile,initial_k=initial_k,n_scan=anchor_scan,cap_scan=cap_scan,mvc_backend=mvc_backend)
    passes=[endpoints.forward,endpoints.backward]
    inserted=0; active_anchor_ids:set[int]=set(); insertion_rounds=0
    T=math.fsum(raw[0::2]); endpoint_tol=2e-9*max(1.0,T)
    usable=[a for a in catalog.anchors if endpoint_tol<a.station<T-endpoint_tol]
    while True:
        witnesses=coverage_witnesses(passes,T)
        envelope=None
        if not witnesses:
            envelope=merge_pass_envelope(passes,T,root_scan=envelope_root_scan)
            witnesses=envelope_continuity_witnesses(envelope,passes)
        if not witnesses:
            break
        batch=select_anchor_batch(witnesses,usable,active_anchor_ids)
        if not batch:
            worst=max(witnesses,key=lambda w:w.magnitude)
            raise DDProfileError(
                f"DD MVC anchor catalog exhausted before closure: {worst.kind} "
                f"[{worst.abs0},{worst.abs1}] magnitude={worst.magnitude}"
            )
        insertion_rounds+=1
        if insertion_rounds>len(usable)+1:
            raise DDProfileError("DD MVC anchor insertion exceeded finite catalog")
        progressed=False
        for a in batch:
            if a.index in active_anchor_ids: continue
            active_anchor_ids.add(a.index)
            aps=compile_anchor_passes(raw,a,params,profile,initial_k=initial_k,n_scan=pass_scan,segment_backend=segment_backend)
            if aps:
                passes.extend(aps); inserted+=1; progressed=True
        if not progressed:
            # Marked anchors that cannot release are still exhausted; continue
            # selecting around the same witness from the remaining catalog.
            if len(active_anchor_ids)>=len(usable):
                raise DDProfileError("all MVC anchors exhausted without candidate coverage")
    assert envelope is not None
    out=DDCompleteProfile(raw,params,profile,endpoints.forward,endpoints.backward,catalog.anchors,passes,envelope,0.0,T,float(init_w),float(terminal_w_max))
    out.total_time=envelope_time(out)
    cert=certify_complete_profile(out,n_samples=513)
    # Boundary semantics: exact start, terminal is an upper bound.
    sw=envelope_w(out,0.0); ew=envelope_w(out,T)
    start_tol=2e-6*max(1.0,abs(init_w)); end_tol=2e-6*max(1.0,abs(terminal_w_max))
    if abs(sw-init_w)>start_tol:
        raise DDProfileError(f"complete DD profile violates fixed start speed: {sw} vs {init_w}")
    if ew>terminal_w_max+end_tol:
        raise DDProfileError(f"complete DD profile violates terminal speed cap: {ew} vs {terminal_w_max}")
    out.diagnostics.update(cert)
    out.diagnostics.update({
        "anchor_catalog_size":len(catalog.anchors),"anchors_with_passes":inserted,
        "anchor_insertion_rounds":insertion_rounds,"active_anchor_count":len(active_anchor_ids),
        "pass_count":len(passes),"envelope_piece_count":len(envelope),
        "forward_terminated_at_mvc":int(endpoints.forward.terminated_at_mvc),
        "backward_terminated_at_mvc":int(endpoints.backward.terminated_at_mvc),
    })
    return out


__all__=[
    "DDMVCPoint","DDMVCAnchor","DDAnchorCatalog","DDEnvelopePiece","DDEnvelopeWitness","DDCompleteProfile",
    "mvc_bound_partials","mvc_active_pair_partials","mvc_cap_partials",
    "hard_state_cap_w","exact_zero_curvature_side_cap_w","mvc_point","mvc_points",
    "build_mvc_anchor_catalog","compile_anchor_passes","coverage_witnesses",
    "envelope_continuity_witnesses","select_anchor_batch","merge_pass_envelope",
    "envelope_w","envelope_time","certify_complete_profile","build_complete_speed_profile",
]
