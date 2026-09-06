"""Direct active-pair MVC root candidate solver (performance research candidate).

Let ``y = sqrt(w)``.  The DD/yaw upper/lower acceleration bounds are
piecewise algebraic in ``y``.  Every non-GRIP upper/lower equality is at most
quadratic.  GRIP equalities become quartics after one squaring; each resulting
root is sign-checked against the original unsquared equation and the complete
acceleration interval.

This module is intentionally independent of the Phase-3 reference scan.  It is
used as a candidate generator and always admits a fail-closed scan fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np

from segment.differential_drive import (
    DifferentialDriveDomainError, DifferentialDriveParameters,
    acceleration_interval,
)
from segment.physics_profiles import PhysicsProfile, RED_COMET_2017_NOMINAL


@dataclass(frozen=True, slots=True)
class DirectMVCRoot:
    y: float
    upper_mode: str
    lower_mode: str
    family: str


def _real_positive_roots(coeff_asc: Iterable[float], y_hi: float) -> list[float]:
    c=np.asarray(tuple(float(x) for x in coeff_asc),dtype=float)
    if c.size==0 or not np.all(np.isfinite(c)): return []
    scale=float(np.max(np.abs(c)))
    if scale==0.0: return []
    # Drop structurally vanished high-order coefficients without erasing a
    # genuinely small physical coefficient after normalization.
    while c.size>1 and abs(c[-1]) <= 2e-14*scale:
        c=c[:-1]
    c=c/scale
    deg=c.size-1
    roots=[]
    if deg==1:
        if c[1]!=0.0: roots=[-c[0]/c[1]]
    elif deg==2:
        a,b,d=c[2],c[1],c[0]
        disc=b*b-4*a*d
        tol=4e-14*max(1.0,b*b,abs(4*a*d))
        if disc>=-tol:
            disc=max(0.0,disc); sd=math.sqrt(disc)
            # Stable quadratic roots.
            q=-0.5*(b+math.copysign(sd,b)) if b!=0.0 else -0.5*sd
            if a!=0.0:
                roots.append(q/a)
                if q!=0.0: roots.append(d/q)
    else:
        rr=np.roots(c[::-1])
        for z in rr:
            if abs(float(z.imag)) <= 5e-9*max(1.0,abs(float(z.real))):
                roots.append(float(z.real))
    out=[]
    for y in roots:
        if math.isfinite(y) and y>0.0 and y <= y_hi*(1.0+2e-10):
            out.append(min(y,y_hi))
    out.sort()
    ded=[]
    for y in out:
        if not ded or abs(y-ded[-1])>2e-8*max(1.0,abs(y)):
            ded.append(y)
    return ded


def _side_terms(eps:int,k:float,sigma:float,p:DifferentialDriveParameters):
    h=1.0+eps*p.beta*k; c=1.0+eps*p.eta*k
    r=p.q0*h/p.side_free_speed_grid
    d=eps*p.eta*sigma
    # U numerator: q0-r*y-d*y^2; L numerator: -q0+r*y-d*y^2
    return h,c,r,d


def direct_pair_roots(
    kappa:float,sigma:float,params:DifferentialDriveParameters,
    profile:PhysicsProfile=RED_COMET_2017_NOMINAL,*,hard_cap_w:float,
)->list[DirectMVCRoot]:
    """Generate all real active-pair equality candidates inside the hard chart."""
    k=float(kappa); sig=float(sigma); p=params; q=profile
    if not hard_cap_w>0.0: return []
    yh=math.sqrt(hard_cap_w)
    side={eps:_side_terms(eps,k,sig,p) for eps in (-1,1)}
    roots:list[DirectMVCRoot]=[]
    def add(coeff,u,l,fam,signcheck=None):
        for y in _real_positive_roots(coeff,yh):
            if signcheck is None or signcheck(y):
                roots.append(DirectMVCRoot(y,u,l,fam))
    # M = -BRAKE.
    add([q.a_max+q.a_brake,-q.b_emf],"MOTOR","BRAKE","linear")
    # M = L_e, U_e = -BRAKE, and U_e = L_d.
    for eps,ulab,llab in ((1,"SIDE_RIGHT","SIDE_RIGHT"),(-1,"SIDE_LEFT","SIDE_LEFT")):
        h,c,r,d=side[eps]
        add([q.a_max*c+p.q0,-q.b_emf*c-r,d],"MOTOR",llab,"quadratic")
        add([p.q0+q.a_brake*c,-r,-d],ulab,"BRAKE","quadratic")
    for eps,ulab in ((1,"SIDE_RIGHT"),(-1,"SIDE_LEFT")):
        he,ce,re,de=side[eps]
        for delt,llab in ((1,"SIDE_RIGHT"),(-1,"SIDE_LEFT")):
            hd,cd,rd,dd=side[delt]
            add([p.q0*(cd+ce),-(re*cd+rd*ce),-de*cd+dd*ce],ulab,llab,"quadratic")
    # Friction hard edge: G = -G.
    if abs(k)>0.0:
        yf=math.sqrt(q.mu_g/abs(k))
        if yf<=yh*(1+2e-10): roots.append(DirectMVCRoot(yf,"GRIP","GRIP","grip_edge"))
    # M = -G.  Squared root needs M<=0.
    add([q.a_max*q.a_max-q.mu_g*q.mu_g,-2*q.a_max*q.b_emf,q.b_emf*q.b_emf,0.0,k*k],
        "MOTOR","GRIP","quartic",lambda y:q.a_max-q.b_emf*y<=1e-9)
    # G = L_e and U_e = -G.
    for eps,ulab,llab in ((1,"SIDE_RIGHT","SIDE_RIGHT"),(-1,"SIDE_LEFT","SIDE_LEFT")):
        h,c,r,d=side[eps]
        # N_L=-q0+r*y-d*y^2
        nl=np.array([-p.q0,r,-d],float)
        # N_U= q0-r*y-d*y^2
        nu=np.array([ p.q0,-r,-d],float)
        def square_plus_grip(poly):
            sq=np.convolve(poly,poly)
            out=np.zeros(5,float);out[:sq.size]+=sq;out[0]-=c*c*q.mu_g*q.mu_g;out[4]+=c*c*k*k
            return out
        add(square_plus_grip(nl),"GRIP",llab,"quartic",lambda y, nl=nl,c=c: np.polynomial.polynomial.polyval(y,nl)/c>=-1e-8)
        add(square_plus_grip(nu),ulab,"GRIP","quartic",lambda y, nu=nu,c=c: np.polynomial.polynomial.polyval(y,nu)/c<=1e-8)
    roots.sort(key=lambda r:(r.y,r.upper_mode,r.lower_mode,r.family))
    return roots


def _interval_margin(w,k,sig,p,q):
    try:return acceleration_interval(w,k,sig,p,q).margin
    except DifferentialDriveDomainError:return -math.inf


def direct_mvc_point_candidate(
    kappa:float,sigma:float,params:DifferentialDriveParameters,
    profile:PhysicsProfile=RED_COMET_2017_NOMINAL,*,hard_cap_w:float,
):
    """Return a direct first-boundary candidate or ``None`` if not certified.

    This routine does not itself replace the reference/native scan.  It proves a
    candidate by checking the complete interval on both adjacent root cells; on
    any numerical ambiguity the caller must fall back to the scan solver.
    """
    k=float(kappa);sig=float(sigma);hard=float(hard_cap_w)
    if not hard>0:return None
    hi=math.nextafter(hard,0.0); lo=max(1e-12,1e-12*max(1.0,hi)); yl=math.sqrt(lo); yh=math.sqrt(hi)
    h0=_interval_margin(lo,k,sig,params,profile)
    if not math.isfinite(h0) or h0<0:return None
    rr=direct_pair_roots(k,sig,params,profile,hard_cap_w=hard)
    ys=[yl]+[r.y for r in rr if yl<r.y<yh]+[yh]
    # Candidate roots may duplicate across inactive pairs; cluster them.
    groups=[]
    for r in rr:
        if not (yl<r.y<yh):continue
        if groups and abs(r.y-groups[-1][0].y)<=3e-7*max(1.0,r.y):groups[-1].append(r)
        else:groups.append([r])
    bounds=[yl]+[sum(x.y for x in g)/len(g) for g in groups]+[yh]
    feas=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        ym=0.5*(a+b); h=_interval_margin(ym*ym,k,sig,params,profile);feas.append(math.isfinite(h) and h>=0.0)
    if not feas or not feas[0]: return None
    for i,g in enumerate(groups):
        if feas[i] and i+1<len(feas) and not feas[i+1]:
            y=sum(x.y for x in g)/len(g); return y*y,g
    # If all open intervals remain feasible, connected interval reaches hard cap.
    if all(feas): return hi,[]
    return None

__all__=["DirectMVCRoot","direct_pair_roots","direct_mvc_point_candidate"]
