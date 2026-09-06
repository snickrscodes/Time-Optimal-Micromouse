"""Python reference signed speed-profile backend for Red Comet DD/yaw v1.

This module is deliberately profile-gated and independent of the qualified
legacy ``optimization.reverse_solver`` implementation.  It implements Phase 2
of the DD/yaw plan:

* one-state traversal dynamics ``dw/ds = 2 a``;
* signed forward/backward extremals;
* MOTOR / GRIP / SIDE_RIGHT / SIDE_LEFT forward candidates;
* BRAKE / GRIP / SIDE_RIGHT / SIDE_LEFT backward candidates;
* generic candidate-switch location with explicit sigma derivatives;
* fail-closed termination at the actuator maximum-velocity boundary ``H=0``.

Phase 3 (``optimization.dd_yaw_anchors``) supplies internal MVC anchors and
merges their extremals with the endpoint passes.  This module intentionally
contains no native dispatch and does not modify legacy solver state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from segment.base import SegmentType
from segment.differential_drive import (
    DifferentialDriveDomainError,
    DifferentialDriveParameters,
    DriveSide,
    acceleration_interval,
    side_candidate_partials,
    side_state,
)
from segment.physics_profiles import PhysicsProfile, RED_COMET_2017_NOMINAL


class DDProfileError(RuntimeError):
    """Base class for DD/yaw reference speed-profile failures."""


class DDStateDomainError(DDProfileError):
    """The speed state left the qualified physical chart."""


class DDPassKind(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class DDEventKind(Enum):
    PIECE_END = auto()
    MODE_SWITCH = auto()
    MVC = auto()
    MVC_KNOT = auto()
    STATE_DOMAIN = auto()


# Stable deterministic tie order.  GRIP precedes SIDE at exact ties so the
# legacy straight/curvature limit remains the preferred semantic label.
_FORWARD_MODES = (
    SegmentType.MOTOR,
    SegmentType.GRIP,
    SegmentType.SIDE_RIGHT,
    SegmentType.SIDE_LEFT,
)
_BACKWARD_MODES = (
    SegmentType.BRAKE,
    SegmentType.GRIP,
    SegmentType.SIDE_RIGHT,
    SegmentType.SIDE_LEFT,
)
_MODE_RANK = {mode: i for i, mode in enumerate((*_FORWARD_MODES, SegmentType.BRAKE))}
_MODE_RANK[SegmentType.BRAKE] = 0


@dataclass(frozen=True, slots=True)
class DDTraversalPiece:
    L: float
    sigma: float
    piece_index: int
    abs0: float
    direction: float


@dataclass(slots=True)
class DDSwitchEvent:
    kind: DDEventKind
    station: float
    local_s: float
    active_mode: SegmentType
    next_mode: SegmentType | None = None
    residual: float = 0.0
    spatial_derivative: float | None = None
    local_columns: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class DDSegmentRecord:
    kind: DDPassKind
    mode: SegmentType
    event: DDSwitchEvent
    traversal_index: int
    piece_index: int
    offset0: float
    L_used: float
    sigma: float
    abs0: float
    abs1: float
    direction: float
    w0: float
    k0: float
    w1: float
    k1: float
    seg: "DDCandidateSegment"


@dataclass(slots=True)
class DDScalarPass:
    kind: DDPassKind
    pieces: list[DDTraversalPiece]
    segments: list[DDSegmentRecord]
    final_w: float
    final_k: float
    final_mode: SegmentType
    terminated_at_mvc: bool = False
    terminated_at_domain: bool = False
    initial_station: float = 0.0
    initial_w: float = 0.0
    initial_k: float = 0.0
    source: str = "endpoint"
    anchor_index: int | None = None


@dataclass(frozen=True, slots=True)
class DDEndpointBuild:
    raw_params: tuple[float, ...]
    forward: DDScalarPass
    backward: DDScalarPass
    total_length: float


@dataclass(frozen=True, slots=True)
class _ScanHit:
    local_s: float
    kind: DDEventKind
    next_mode: SegmentType | None = None
    residual: float = 0.0


_RTL = 3.0e-11
_ATOL = np.array([2.0e-12, 2.0e-12, 2.0e-11, 2.0e-11, 2.0e-11, 2.0e-11, 2.0e-11, 2.0e-11])


def validate_raw_parameters(raw_params: Sequence[float]) -> tuple[float, ...]:
    raw = tuple(float(x) for x in raw_params)
    if len(raw) % 2:
        raise ValueError("raw parameter array must have even length")
    for i in range(0, len(raw), 2):
        if not math.isfinite(raw[i]) or raw[i] <= 0.0:
            raise ValueError(f"segment {i//2} length must be finite and positive")
        if not math.isfinite(raw[i + 1]):
            raise ValueError(f"segment {i//2} sigma must be finite")
    return raw


def geometry_knot_curvatures(raw_params: Sequence[float], *, initial_k: float = 0.0) -> list[float]:
    raw = validate_raw_parameters(raw_params)
    k = float(initial_k)
    if not math.isfinite(k):
        raise ValueError("initial_k must be finite")
    out = [k]
    for i in range(0, len(raw), 2):
        k = math.fma(raw[i + 1], raw[i], k)
        out.append(k)
    return out


def total_length(raw_params: Sequence[float]) -> float:
    raw = validate_raw_parameters(raw_params)
    return math.fsum(raw[0::2])


def make_forward_traversal(raw_params: Sequence[float]) -> list[DDTraversalPiece]:
    raw = validate_raw_parameters(raw_params)
    out: list[DDTraversalPiece] = []
    station = 0.0
    for i in range(len(raw) // 2):
        L, sigma = raw[2*i], raw[2*i + 1]
        out.append(DDTraversalPiece(L, sigma, i, station, 1.0))
        station += L
    return out


def make_backward_traversal(raw_params: Sequence[float]) -> list[DDTraversalPiece]:
    raw = validate_raw_parameters(raw_params)
    starts: list[float] = []
    station = 0.0
    for i in range(len(raw) // 2):
        starts.append(station)
        station += raw[2*i]
    out: list[DDTraversalPiece] = []
    for i in range(len(raw)//2 - 1, -1, -1):
        L, sigma = raw[2*i], raw[2*i + 1]
        out.append(DDTraversalPiece(L, -sigma, i, starts[i] + L, -1.0))
    return out


def candidate_modes(kind: DDPassKind) -> tuple[SegmentType, ...]:
    return _FORWARD_MODES if kind is DDPassKind.FORWARD else _BACKWARD_MODES


def _grip_partials(w: float, kappa: float, profile: PhysicsProfile) -> tuple[float, float, float, float]:
    q = w * kappa
    g2 = profile.mu_g * profile.mu_g - q*q
    if not g2 > 0.0:
        raise DDStateDomainError("GRIP candidate requested at/outside friction domain")
    g = math.sqrt(g2)
    return g, -(q*kappa)/g, -(q*w)/g, 0.0


def candidate_value_partials(
    mode: SegmentType,
    w: float,
    kappa: float,
    sigma: float,
    kind: DDPassKind,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> tuple[float, float, float, float]:
    """Return traversal acceleration candidate and ``(w,kappa,sigma)`` partials."""

    w = float(w); kappa = float(kappa); sigma = float(sigma)
    if not (math.isfinite(w) and math.isfinite(kappa) and math.isfinite(sigma)) or w <= 0.0:
        raise DDStateDomainError("candidate chart requires finite w>0, kappa and sigma")

    if mode is SegmentType.MOTOR:
        if kind is not DDPassKind.FORWARD:
            raise ValueError("MOTOR is only a forward DD candidate")
        root = math.sqrt(w)
        return (
            profile.a_max - profile.b_emf * root,
            -profile.b_emf / (2.0 * root),
            0.0,
            0.0,
        )
    if mode is SegmentType.BRAKE:
        if kind is not DDPassKind.BACKWARD:
            raise ValueError("BRAKE is only a backward DD candidate")
        return profile.a_brake, 0.0, 0.0, 0.0
    if mode is SegmentType.GRIP:
        return _grip_partials(w, kappa, profile)
    if mode is SegmentType.SIDE_RIGHT:
        return side_candidate_partials(w, kappa, sigma, DriveSide.RIGHT, params)
    if mode is SegmentType.SIDE_LEFT:
        return side_candidate_partials(w, kappa, sigma, DriveSide.LEFT, params)
    raise ValueError(f"unsupported DD candidate mode {mode!r}")


def candidate_value(*args, **kwargs) -> float:
    return candidate_value_partials(*args, **kwargs)[0]


def candidate_vector(
    kind: DDPassKind,
    w: float,
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> dict[SegmentType, float]:
    return {
        mode: candidate_value(mode, w, kappa, sigma, kind, params, profile)
        for mode in candidate_modes(kind)
    }


def choose_active_mode(
    kind: DDPassKind,
    w: float,
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    tie_tol: float = 2.0e-12,
) -> SegmentType:
    values = candidate_vector(kind, w, kappa, sigma, params, profile)
    best = min(values.values())
    tied = [m for m, v in values.items() if v <= best + tie_tol * max(1.0, abs(best), abs(v))]
    order = candidate_modes(kind)
    return min(tied, key=order.index)


def _physical_sigma(kind: DDPassKind, traversal_sigma: float) -> float:
    return traversal_sigma if kind is DDPassKind.FORWARD else -traversal_sigma


def feasibility_margin(
    kind: DDPassKind,
    w: float,
    kappa: float,
    traversal_sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> float:
    """Return the original-coordinate acceleration-interval width ``H``."""
    try:
        interval = acceleration_interval(
            w, kappa, _physical_sigma(kind, traversal_sigma), params, profile
        )
    except DifferentialDriveDomainError as exc:
        raise DDStateDomainError(str(exc)) from exc
    return interval.margin


def state_domain_margins(
    w: float,
    kappa: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> dict[str, float]:
    """Positive margins for the smooth DD chart, excluding ``H`` itself."""
    if not math.isfinite(w) or w <= 0.0 or not math.isfinite(kappa):
        return {"w": float(w), "friction": -math.inf}
    root = math.sqrt(w)
    out = {
        "w": w,
        "friction": profile.mu_g * profile.mu_g - (w*kappa)**2,
    }
    for side, label in ((DriveSide.RIGHT, "right"), (DriveSide.LEFT, "left")):
        eps = int(side)
        h = 1.0 + eps * params.beta*kappa
        c = 1.0 + eps * params.eta*kappa
        out[f"{label}_h"] = h - params.h_floor
        out[f"{label}_c"] = c - params.c_floor
        out[f"{label}_free"] = params.side_free_speed_grid - root*h - params.speed_margin
    return out


def state_domain_ok(
    w: float,
    kappa: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    tol: float = 0.0,
) -> bool:
    margins = state_domain_margins(w, kappa, params, profile)
    return bool(margins) and all(math.isfinite(v) and v > tol for v in margins.values())


class DDCandidateSegment:
    """Smooth numerical reference segment for one active DD pass candidate.

    The differentiable state is ``(w,T,S_sigma,S_w0,S_k0,R_sigma,R_w0,R_k0)``.
    The local mode may be signed; only positivity of ``w`` and the smooth chart
    are required.  Candidate switches and MVC/domain events are owned by the
    pass builder, not by this segment object.
    """

    def __init__(
        self,
        L: float,
        sigma: float,
        w0: float,
        k0: float,
        mode: SegmentType,
        kind: DDPassKind,
        params: DifferentialDriveParameters,
        profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    ):
        self.L=float(L); self.sigma=float(sigma); self.w0=float(w0); self.k0=float(k0)
        self.mode=mode; self.kind=kind; self.params=params; self.profile=profile
        if not all(math.isfinite(x) for x in (self.L,self.sigma,self.w0,self.k0)):
            raise ValueError("DD candidate segment inputs must be finite")
        if self.L < 0.0 or self.w0 <= 0.0:
            raise ValueError("DD candidate segment requires L>=0 and w0>0")
        if mode not in candidate_modes(kind):
            raise ValueError(f"mode {mode.name} invalid for {kind.name} pass")
        if not state_domain_ok(self.w0,self.k0,params,profile):
            raise DDStateDomainError("DD candidate initial state outside smooth chart")
        # Force candidate-domain validation now.
        candidate_value(mode,self.w0,self.k0,self.sigma,kind,params,profile)
        self._cache: dict[float, tuple[float,...]] = {}

    def _integrate(self, ds: float) -> tuple[float,...]:
        ds=float(ds)
        tol=max(32*math.ulp(max(1.0,abs(self.L))), 1.0e-9*max(1.0,abs(self.L)))
        if ds < -tol or ds > self.L+tol:
            raise ValueError("DD segment prefix outside [0,L]")
        ds=min(max(ds,0.0),self.L)
        if ds==0.0:
            a=candidate_value(self.mode,self.w0,self.k0,self.sigma,self.kind,self.params,self.profile)
            return self.w0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,2*a
        if ds in self._cache:
            return self._cache[ds]

        def rhs(s: float, y: np.ndarray) -> np.ndarray:
            w=float(y[0])
            if w <= 0.0:
                w=np.finfo(float).tiny
            k=math.fma(self.sigma,s,self.k0)
            a,aw,ak,asig=candidate_value_partials(
                self.mode,w,k,self.sigma,self.kind,self.params,self.profile
            )
            f=2*a; fw=2*aw; fk=2*ak; fs=2*asig
            ss,sw,sk=map(float,y[2:5])
            rt=math.sqrt(w)
            q=-0.5/(w*rt)
            return np.array((
                f,1/rt,
                fw*ss + fk*s + fs,
                fw*sw,
                fw*sk + fk,
                q*ss,q*sw,q*sk,
            ),dtype=float)

        def w_event(_s,y): return float(y[0])
        w_event.terminal=True; w_event.direction=-1.0
        sol=solve_ivp(
            rhs,(0.0,ds),np.array((self.w0,0,0,1,0,0,0,0),dtype=float),
            method="DOP853",rtol=_RTL,atol=_ATOL,
            max_step=min(0.02,max(ds/16.0,1e-5)),events=(w_event,),
        )
        if not sol.success or sol.status==1 or abs(float(sol.t[-1])-ds)>5e-10*max(1.0,ds):
            raise DDStateDomainError(f"DD candidate integration failed before ds={ds}: {sol.message}")
        w=float(sol.y[0,-1]); k=math.fma(self.sigma,ds,self.k0)
        if not state_domain_ok(w,k,self.params,self.profile,tol=-1e-11):
            raise DDStateDomainError("DD candidate endpoint left smooth state chart")
        a=candidate_value(self.mode,w,k,self.sigma,self.kind,self.params,self.profile)
        out=tuple(float(x) for x in sol.y[:,-1])+(2*a,)
        self._cache[ds]=out
        return out

    def w(self,ds:float)->float: return self._integrate(ds)[0]
    def time(self,ds:float)->float: return self._integrate(ds)[1]
    def w_and_jac(self,ds:float):
        z=self._integrate(ds); w=z[0]; ss,sw,sk=z[2:5]; f=z[8]
        return w,(f,ss,sw,sk, self.sigma,ds,0.0,1.0)
    def time_and_jac(self,ds:float):
        z=self._integrate(ds); w=z[0]; t=z[1]; rs,rw,rk=z[5:8]
        return t,(1.0/math.sqrt(w),rs,rw,rk)
    def state_time_and_jac(self,ds:float):
        w,jw=self.w_and_jac(ds); t,jt=self.time_and_jac(ds); return w,jw,t,jt



class DDNativeCandidateSegment(DDCandidateSegment):
    """Phase-5 native mirror of one smooth DD candidate segment.

    The public interface intentionally matches ``DDCandidateSegment`` so the
    Python topology/event machinery can be A/B qualified with identical logic.
    Native calls are cached per prefix station.
    """
    def _integrate(self, ds: float) -> tuple[float,...]:
        ds=float(ds)
        tol=max(32*math.ulp(max(1.0,abs(self.L))), 1.0e-9*max(1.0,abs(self.L)))
        if ds < -tol or ds > self.L+tol:
            raise ValueError("DD native segment prefix outside [0,L]")
        ds=min(max(ds,0.0),self.L)
        if ds in self._cache:
            return self._cache[ds]
        from cdd_yaw import segment_native
        try:
            w,jw,t,jt=segment_native(
                self.L,self.sigma,self.w0,self.k0,self.mode,self.kind,
                self.params,self.profile,ds=ds,
            )
        except RuntimeError as exc:
            raise DDStateDomainError(str(exc)) from exc
        # Repack to the authoritative Python internal cache convention:
        # (w,T,Ssigma,Sw0,Sk0,Rsigma,Rw0,Rk0,f_endpoint).
        out=(float(w),float(t),float(jw[1]),float(jw[2]),float(jw[3]),
             float(jt[1]),float(jt[2]),float(jt[3]),float(jw[0]))
        self._cache[ds]=out
        return out


def candidate_segment_class(segment_backend: str):
    if segment_backend=="python": return DDCandidateSegment
    if segment_backend=="native": return DDNativeCandidateSegment
    raise ValueError("segment_backend must be 'python' or 'native'")

def switch_residual_partials(
    active: SegmentType,
    alternative: SegmentType,
    w: float,
    kappa: float,
    sigma: float,
    kind: DDPassKind,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> tuple[float,float,float,float]:
    ai,iw,ik,isg=candidate_value_partials(active,w,kappa,sigma,kind,params,profile)
    aj,jw,jk,jsg=candidate_value_partials(alternative,w,kappa,sigma,kind,params,profile)
    return ai-aj, iw-jw, ik-jk, isg-jsg


def switch_local_columns(
    seg: DDCandidateSegment,
    ds: float,
    alternative: SegmentType,
) -> tuple[float,float,float,float]:
    w,jac=seg.w_and_jac(ds)
    k=math.fma(seg.sigma,ds,seg.k0)
    _,fw,fk,fs=switch_residual_partials(
        seg.mode,alternative,w,k,seg.sigma,seg.kind,seg.params,seg.profile
    )
    jw=jac[:4]; jk=jac[4:8]
    # Explicit sigma dependence is additional to state-map dependence.
    return tuple(fw*jw[i]+fk*jk[i]+(fs if i==1 else 0.0) for i in range(4))


def switch_root_parameter_derivatives(
    seg: DDCandidateSegment,
    ds: float,
    alternative: SegmentType,
) -> tuple[float,float,float]:
    cols=switch_local_columns(seg,ds,alternative)
    if not math.isfinite(cols[0]) or abs(cols[0]) < 1e-12:
        raise DDProfileError("switch root is not transverse")
    return tuple(-cols[i]/cols[0] for i in (1,2,3))


def _safe_state_at(seg: DDCandidateSegment, ds: float):
    try:
        w=seg.w(ds); k=math.fma(seg.sigma,ds,seg.k0)
        if not state_domain_ok(w,k,seg.params,seg.profile,tol=-1e-11):
            return None
        return w,k
    except (DDProfileError,DifferentialDriveDomainError,ValueError,FloatingPointError):
        return None


def _find_first_event(
    seg: DDCandidateSegment,
    horizon: float,
    *,
    n_scan: int = 96,
    root_tol: float = 2.0e-11,
    ignore_mvc_at_start: bool = False,
) -> _ScanHit | None:
    """Locate earliest candidate switch, MVC hit, or state-chart edge."""
    if horizon <= 0.0: return None
    modes=[m for m in candidate_modes(seg.kind) if m is not seg.mode]
    xs=np.linspace(0.0,horizon,max(8,int(n_scan))+1)

    st0=_safe_state_at(seg,0.0)
    if st0 is None:
        return _ScanHit(0.0,DDEventKind.STATE_DOMAIN)
    w0,k0=st0
    try:
        fprev={m:switch_residual_partials(seg.mode,m,w0,k0,seg.sigma,seg.kind,seg.params,seg.profile)[0] for m in modes}
        hprev=feasibility_margin(seg.kind,w0,k0,seg.sigma,seg.params,seg.profile)
    except Exception:
        return _ScanHit(0.0,DDEventKind.STATE_DOMAIN)

    for idx in range(1,len(xs)):
        x=float(xs[idx]); xp=float(xs[idx-1])
        st=_safe_state_at(seg,x)
        if st is None:
            # Refine the first smooth-chart edge.  Friction and rated side
            # free-speed edges are legitimate maximum-speed boundaries, while
            # h/c sign loss (or w->0) leaves the v1 coordinate chart and is
            # fatal.  This distinction is important when H approaches zero
            # *at* the hard boundary without becoming negative beforehand.
            lo,hi=xp,x
            for _ in range(70):
                mid=0.5*(lo+hi)
                if _safe_state_at(seg,mid) is None: hi=mid
                else: lo=mid
            edge=_safe_state_at(seg,lo)
            if edge is not None:
                margins=state_domain_margins(edge[0],edge[1],seg.params,seg.profile)
                hard_keys=("friction","right_free","left_free")
                hard_scale=max(1.0,seg.profile.mu_g*seg.profile.mu_g,seg.params.side_free_speed_grid)
                hard_near=any(margins.get(key,math.inf) <= 2e-8*hard_scale for key in hard_keys)
                chart_near=any(margins.get(key,math.inf) <= 2e-8 for key in ("w","right_h","left_h","right_c","left_c"))
                if hard_near and not chart_near:
                    return _ScanHit(lo,DDEventKind.MVC)
            return _ScanHit(lo,DDEventKind.STATE_DOMAIN)
        w,k=st
        try:
            h=feasibility_margin(seg.kind,w,k,seg.sigma,seg.params,seg.profile)
        except Exception:
            h=-math.inf
        if (not ignore_mvc_at_start or xp>0.0 or hprev>root_tol) and hprev >= -root_tol and h < -root_tol:
            def hf(s):
                stx=_safe_state_at(seg,s)
                if stx is None: return -1.0
                return feasibility_margin(seg.kind,stx[0],stx[1],seg.sigma,seg.params,seg.profile)
            try:
                r=brentq(hf,xp,x,xtol=3e-13,rtol=4*np.finfo(float).eps,maxiter=100)
            except ValueError:
                r=xp
            return _ScanHit(float(r),DDEventKind.MVC)

        hits=[]
        for m in modes:
            try:
                f=switch_residual_partials(seg.mode,m,w,k,seg.sigma,seg.kind,seg.params,seg.profile)[0]
            except Exception:
                continue
            fp=fprev[m]
            # active-alt is <=0 while active remains minimal; we want the first
            # transverse upward crossing into alternative-lower territory.
            if fp <= root_tol and f > root_tol:
                def ff(s, alt=m):
                    stx=_safe_state_at(seg,s)
                    if stx is None: return math.nan
                    return switch_residual_partials(seg.mode,alt,stx[0],stx[1],seg.sigma,seg.kind,seg.params,seg.profile)[0]
                try:
                    r=brentq(ff,xp,x,xtol=3e-13,rtol=4*np.finfo(float).eps,maxiter=100)
                    # Reject tangencies / wrong-oriented numerical roots.
                    cols=switch_local_columns(seg,float(r),m)
                    if cols[0] > 1e-10:
                        hits.append((float(r),candidate_modes(seg.kind).index(m),m,ff(float(r))))
                except (ValueError,DDProfileError):
                    pass
            fprev[m]=f
        if hits:
            r,_,m,res=min(hits,key=lambda z:(z[0],z[1]))
            return _ScanHit(r,DDEventKind.MODE_SWITCH,m,res)
        hprev=h
    return None


def _mode_after_switch(
    kind: DDPassKind,
    seg: DDCandidateSegment,
    root: float,
    nominal_next: SegmentType,
) -> SegmentType:
    # Probe an ulp/short spatial step after the switch along the authoritative
    # active segment.  If unavailable, deterministic event target wins.
    eps=max(1e-9,2e-7*max(1.0,seg.L))
    q=min(seg.L,root+eps)
    if q>root:
        st=_safe_state_at(seg,q)
        if st is not None:
            try: return choose_active_mode(kind,st[0],st[1],seg.sigma,seg.params,seg.profile)
            except Exception: pass
    return nominal_next


def build_signed_pass(
    pieces: Sequence[DDTraversalPiece],
    init_w: float,
    init_k: float,
    kind: DDPassKind,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    n_scan: int = 96,
    piece_eps: float = 2e-12,
    max_subsegments_per_piece: int = 96,
    initial_station: float | None = None,
    source: str = "endpoint",
    anchor_index: int | None = None,
    allow_initial_mvc_release: bool = False,
    segment_backend: str = "python",
) -> DDScalarPass:
    """Build one signed endpoint/anchor extremal until piece/domain/MVC stop."""
    if not pieces: raise ValueError("DD pass requires at least one traversal piece")
    w=float(init_w); k=float(init_k)
    if w<=0 or not math.isfinite(w) or not math.isfinite(k):
        raise ValueError("DD pass initial state must have finite w>0 and k")
    station = pieces[0].abs0 if initial_station is None else float(initial_station)
    mode=choose_active_mode(kind,w,k,pieces[0].sigma,params,profile)
    records: list[DDSegmentRecord]=[]
    terminated_mvc=False; terminated_domain=False
    first_arc=True

    for ti,piece in enumerate(pieces):
        offset=0.0
        # Re-select only at a geometry-piece boundary.  Within a smooth piece,
        # an accepted transverse switch owns the next mode.  Re-selecting at
        # the exact root would deterministically choose the old tie-ranked mode
        # again and create zero-length chatter.
        # Sigma is piecewise constant but can jump at a knot.  Because the DD
        # interval depends explicitly on sigma, an otherwise continuous (w,k)
        # state can become infeasible instantaneously on the new piece.  That
        # is a knot MVC, not a numerical domain failure.
        h_piece=feasibility_margin(kind,w,k,piece.sigma,params,profile)
        if h_piece < -3e-8:
            if records:
                last=records[-1]
                last.event=DDSwitchEvent(
                    DDEventKind.MVC_KNOT,last.abs1,last.L_used,last.mode,None,
                    float(h_piece),None,None,
                )
            return DDScalarPass(kind,list(pieces),records,w,k,mode,True,False,station,init_w,init_k,source,anchor_index)
        mode=choose_active_mode(kind,w,k,piece.sigma,params,profile)
        for _sub in range(max_subsegments_per_piece):
            remaining=piece.L-offset
            if remaining<=piece_eps: break
            seg_cls=candidate_segment_class(segment_backend)
            seg=seg_cls(remaining,piece.sigma,w,k,mode,kind,params,profile)
            hit=_find_first_event(
                seg,remaining,n_scan=n_scan,
                ignore_mvc_at_start=allow_initial_mvc_release and first_arc,
            )
            if hit is None:
                L_used=remaining; event_kind=DDEventKind.PIECE_END; next_mode=None
            else:
                L_used=max(0.0,min(remaining,hit.local_s)); event_kind=hit.kind; next_mode=hit.next_mode
            if L_used<=piece_eps and event_kind in (DDEventKind.MVC,DDEventKind.STATE_DOMAIN):
                if event_kind is DDEventKind.MVC: terminated_mvc=True
                else: terminated_domain=True
                return DDScalarPass(kind,list(pieces),records,w,k,mode,terminated_mvc,terminated_domain,station,init_w,init_k,source,anchor_index)
            if L_used<=piece_eps and event_kind is DDEventKind.MODE_SWITCH and next_mode is not None:
                # A simultaneous/tied switch at the current state: advance the
                # mode graph without emitting a fake segment.  The next loop
                # iteration starts from the same authoritative state.
                mode=next_mode
                continue

            w1=seg.w(L_used); k1=math.fma(piece.sigma,L_used,k)
            abs0=piece.abs0+piece.direction*offset
            abs1=piece.abs0+piece.direction*(offset+L_used)
            cols=None; dF=None; residual=hit.residual if hit else 0.0
            if event_kind is DDEventKind.MODE_SWITCH and next_mode is not None:
                try:
                    cols=switch_local_columns(seg,L_used,next_mode); dF=cols[0]
                except Exception:
                    cols=None; dF=None
            event=DDSwitchEvent(event_kind,abs1,L_used,mode,next_mode,residual,dF,cols)
            records.append(DDSegmentRecord(kind,mode,event,ti,piece.piece_index,offset,L_used,piece.sigma,abs0,abs1,piece.direction,w,k,w1,k1,seg))
            offset += L_used; w,k=w1,k1; first_arc=False
            if event_kind is DDEventKind.MVC:
                terminated_mvc=True
                return DDScalarPass(kind,list(pieces),records,w,k,mode,True,False,station,init_w,init_k,source,anchor_index)
            if event_kind is DDEventKind.STATE_DOMAIN:
                terminated_domain=True
                return DDScalarPass(kind,list(pieces),records,w,k,mode,False,True,station,init_w,init_k,source,anchor_index)
            if event_kind is DDEventKind.PIECE_END:
                if offset>=piece.L-piece_eps: break
            elif next_mode is not None:
                mode=_mode_after_switch(kind,seg,L_used,next_mode)
        else:
            raise DDProfileError(f"too many DD subsegments in piece {piece.piece_index}; possible switch chatter")

    return DDScalarPass(kind,list(pieces),records,w,k,mode,terminated_mvc,terminated_domain,station,init_w,init_k,source,anchor_index)


def build_endpoint_passes(
    raw_params: Sequence[float],
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    init_w: float | None = None,
    terminal_w_max: float | None = None,
    initial_k: float = 0.0,
    n_scan: int = 96,
    segment_backend: str = "python",
) -> DDEndpointBuild:
    raw=validate_raw_parameters(raw_params)
    if init_w is None: init_w=max(1e-3,(0.05*profile.v_max)**2)
    if terminal_w_max is None: terminal_w_max=init_w
    ks=geometry_knot_curvatures(raw,initial_k=initial_k)
    fpieces=make_forward_traversal(raw); bpieces=make_backward_traversal(raw)
    f=build_signed_pass(fpieces,init_w,initial_k,DDPassKind.FORWARD,params,profile,n_scan=n_scan,initial_station=0.0,segment_backend=segment_backend)
    b=build_signed_pass(bpieces,terminal_w_max,ks[-1],DDPassKind.BACKWARD,params,profile,n_scan=n_scan,initial_station=math.fsum(raw[0::2]),segment_backend=segment_backend)
    return DDEndpointBuild(raw,f,b,math.fsum(raw[0::2]))



def certify_signed_pass(
    pass_: DDScalarPass,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
    *,
    samples_per_segment: int = 9,
    mode_tol: float = 3e-6,
    margin_tol: float = 3e-6,
) -> dict[str, float]:
    """Independently replay one scalar pass against the complete candidate set."""
    max_mode_violation=0.0; min_margin=math.inf; max_event_residual=0.0; min_w=math.inf
    prev=None
    for rec in pass_.segments:
        if prev is not None:
            state_scale=max(1.0,abs(prev.w1),abs(rec.w0),abs(prev.k1),abs(rec.k0))
            if abs(prev.w1-rec.w0)>2e-8*state_scale or abs(prev.k1-rec.k0)>2e-8*state_scale:
                raise DDProfileError("DD scalar pass is state-discontinuous")
        prev=rec
        # Do not sample the exact hard-cap endpoint through candidate partials;
        # acceleration_interval itself remains well-defined there.
        end_frac=1.0 if rec.event.kind not in (DDEventKind.MVC,DDEventKind.MVC_KNOT,DDEventKind.STATE_DOMAIN) else 1.0-2e-9
        for frac in np.linspace(0.0,end_frac,max(3,int(samples_per_segment))):
            ds=float(frac)*rec.L_used
            w=rec.seg.w(ds); k=math.fma(rec.sigma,ds,rec.k0); min_w=min(min_w,w)
            vals=candidate_vector(rec.kind,w,k,rec.sigma,params,profile)
            active=vals[rec.mode]; best=min(vals.values())
            max_mode_violation=max(max_mode_violation,active-best)
            h=feasibility_margin(rec.kind,w,k,rec.sigma,params,profile); min_margin=min(min_margin,h)
        if rec.event.kind is DDEventKind.MODE_SWITCH and rec.event.next_mode is not None:
            w=rec.w1; k=rec.k1
            f=switch_residual_partials(rec.mode,rec.event.next_mode,w,k,rec.sigma,rec.kind,params,profile)[0]
            max_event_residual=max(max_event_residual,abs(f))
            cols=switch_local_columns(rec.seg,rec.L_used,rec.event.next_mode)
            if cols[0] <= 0.0:
                raise DDProfileError("DD mode switch is not transverse/upward")
        elif rec.event.kind is DDEventKind.MVC:
            try:
                h=feasibility_margin(rec.kind,rec.w1,rec.k1,rec.sigma,params,profile)
                max_event_residual=max(max_event_residual,abs(h))
            except Exception:
                # Hard friction/free-speed caps can be strict-chart boundaries;
                # the pre-endpoint samples above certify the inward approach.
                pass
    if min_w<=0.0 or not math.isfinite(min_w):
        raise DDProfileError("DD scalar pass reaches nonpositive speed")
    if max_mode_violation>mode_tol:
        raise DDProfileError(f"DD pass active-mode violation {max_mode_violation}")
    if min_margin < -margin_tol:
        raise DDProfileError(f"DD pass acceleration-interval violation {min_margin}")
    return {
        "min_w":min_w,
        "min_interval_margin":min_margin,
        "max_mode_violation":max_mode_violation,
        "max_event_residual":max_event_residual,
    }

def record_interval(rec: DDSegmentRecord) -> tuple[float,float]:
    return min(rec.abs0,rec.abs1),max(rec.abs0,rec.abs1)


def record_local(rec: DDSegmentRecord, station: float) -> float:
    ds=rec.direction*(station-rec.abs0)
    tol=64*math.ulp(max(1.0,abs(station),abs(rec.abs0),abs(rec.abs1)))
    if -tol<=ds<0: ds=0.0
    if rec.L_used<ds<=rec.L_used+tol: ds=rec.L_used
    return ds


def record_w(rec: DDSegmentRecord,station:float)->float:
    return rec.seg.w(record_local(rec,station))


def record_time_between(rec:DDSegmentRecord,s0:float,s1:float)->float:
    d0=record_local(rec,s0); d1=record_local(rec,s1)
    return abs(rec.seg.time(d1)-rec.seg.time(d0))


def pass_record_at(pass_: DDScalarPass, station: float, *, tol: float=2e-11) -> DDSegmentRecord | None:
    for rec in pass_.segments:
        lo,hi=record_interval(rec)
        if lo-tol<=station<=hi+tol:
            return rec
    return None


def pass_w(pass_:DDScalarPass,station:float)->float:
    rec=pass_record_at(pass_,station)
    if rec is None: raise DDProfileError(f"pass does not cover station {station}")
    return record_w(rec,station)


__all__=[
    "DDProfileError","DDStateDomainError","DDPassKind","DDEventKind",
    "DDTraversalPiece","DDSwitchEvent","DDSegmentRecord","DDScalarPass","DDEndpointBuild",
    "DDCandidateSegment","validate_raw_parameters","geometry_knot_curvatures","total_length",
    "make_forward_traversal","make_backward_traversal","candidate_modes","candidate_value",
    "candidate_value_partials","candidate_vector","choose_active_mode","feasibility_margin",
    "state_domain_margins","state_domain_ok","switch_residual_partials","switch_local_columns",
    "switch_root_parameter_derivatives","build_signed_pass","build_endpoint_passes","certify_signed_pass",
    "record_interval","record_local","record_w","record_time_between","pass_record_at","pass_w",
]
