#include "cflow_internal.h"

int cflow_terminal_candidate(double x,double q,double b,double remaining,double *hcap){
    if(x==0.0||remaining<=0.0) return 0;
    double z=fabs(q*x); if(z>=1.0) return 0;
    double sd=sqrt(fmax(0.0,(1-z)*(1+z))),delta=atan2(sd,z);if(delta>CFLOW_TERM_DELTA)return 0;
    double hc=0.18*fabs(x)*delta/2.0; double bx=fabs(b*x); if(bx>0.0)hc=fmin(hc,0.10*delta*delta/bx);
    hc*=0.985;if(hc>remaining)hc=remaining;if(hc<=0.0)return 0;*hcap=hc;return 1;
}
void cflow_terminal_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o){
    /* Canonical terminal state authority.

       The value-only physical RK path is the authoritative endpoint map for
       every public API.  Sensitivity-controlled adaptivity can legitimately
       choose a different RK step sequence and perturb the final state by a few
       ulps; immediately beside a first event those ulps can change structural
       event classification.  When derivatives are requested, integrate the
       variational channels with their stricter controller but overwrite only
       the endpoint state with the canonical value path.

       This deliberately costs one extra scalar terminal RK integration on
       Jacobian calls. Terminal calls are already a minority/specialized path,
       and cross-API state/status consistency is a hard production invariant. */
    if(!want_jac){
        cflow_rk_physical_local(x,q,b,h,0,o);
        return;
    }
    cflow_local_jac jac,value;
    cflow_rk_physical_local(x,q,b,h,1,&jac);
    if(!isfinite(jac.x)){*o=jac;return;}
    cflow_rk_physical_local(x,q,b,h,0,&value);
    if(!isfinite(value.x)){*o=value;return;}
    o->x=value.x;
    o->dx=jac.dx;
    o->dq=jac.dq;
    o->db=jac.db;
}

/* -------------------------------------------------------------------------
 * Separatrix conditioning analysis.
 *
 * Runtime separatrix snapping was removed in Phase VIIA.  The asymptotic
 * separatrix remains useful only as a conditioning diagnostic in the regime
 * |v| >= 3.  All ordinary endpoints are propagated by the canonical Cflow
 * maps, preserving the numerical semigroup.
 * ------------------------------------------------------------------------- */

#define CFLOW_SEP_V_MIN 3.0
#define CFLOW_SEP_RHO_CONDITION 1.0
#define CFLOW_SEP_X_UNCERTAINTY_LIMIT 1.0e-12
#define CFLOW_SEP_CONDITION_SAFETY 4.0

typedef struct {
    int valid;
    int expanding;
    double v0, v1;
    double eta_sep0, eta_sep1;
    double rho;
    double dphi;
    double log_x_uncertainty;
} cflow_sep_analysis;

static double cflow_sep_residual_u(double v){
    /* u_sep(v) + 1/v, evaluated without subtracting the common -1/v term. */
    double w=1.0/v, w2=w*w, w4=w2*w2;
    return w*w4*(
        1.0/8.0 + w4*(
        -19.0/128.0 + w4*(
        373.0/1024.0 + w4*(
        -43779.0/32768.0 + w4*(1682975.0/262144.0)
    ))));
}

static double cflow_sep_eta(double v){
    /* eta = 1 + v*u.  On the separatrix eta ~ 1/(8 v^4) and is positive. */
    return v*cflow_sep_residual_u(v);
}

static double cflow_sep_eta_dv(double v){
    /* d/dv [1 + v*u_sep(v)] from the same residual series. */
    double w=1.0/v, w2=w*w, w4=w2*w2;
    double w5=w*w4;
    return w5*(
        -4.0/8.0 + w4*(
        8.0*19.0/128.0 + w4*(
        -12.0*373.0/1024.0 + w4*(
        16.0*43779.0/32768.0 + w4*(-20.0*1682975.0/262144.0)
    ))));
}

static double cflow_sep_phi(double v){
    /* Phi(r), r=v^2, through O(r^-4).  Only differences are used. */
    double av=fabs(v), iv=1.0/av, iv2=iv*iv, iv4=iv2*iv2, iv8=iv4*iv4;
    double v2=av*av, v4=v2*v2;
    return v4 + 5.0*log(av) + 1.5*iv4 - (87.0/32.0)*iv8;
}

static double cflow_half_ulp(double x){
    if(!isfinite(x)) return INFINITY;
    double up=fabs(nextafter(x,INFINITY)-x);
    double dn=fabs(x-nextafter(x,-INFINITY));
    double span=fmax(up,dn);
    return 0.5*span;
}

static int cflow_sep_analyze(double x,double q,double b,double remaining,cflow_sep_analysis *a){
    *a=(cflow_sep_analysis){0};
    if(b==0.0 || !(remaining>0.0)) return 0;
    double ab=fabs(b), g=sqrt(ab), sb=copysign(1.0,b);
    double q1=fma(b,remaining,q);
    double v0=q*sb/g, v1=q1*sb/g;
    if(!isfinite(v0)||!isfinite(v1)||fabs(v0)<CFLOW_SEP_V_MIN||fabs(v1)<CFLOW_SEP_V_MIN||v0*v1<=0.0) return 0;

    double eta0=cflow_sep_eta(v0), eta1=cflow_sep_eta(v1);
    if(!(eta0>0.0) || !(eta1>0.0) || !isfinite(eta0) || !isfinite(eta1)) return 0;

    /* z = sign(b)*q*x = u*v.  Using an FMA preserves the small 1+z
       residual far better than forming u and the boundary state separately. */
    double eta=fma(sb*q,x,1.0);
    double delta_eta=eta-eta0;
    double rho=fabs(delta_eta)/eta0;
    double dphi=cflow_sep_phi(v1)-cflow_sep_phi(v0);

    a->valid=1;
    a->expanding=dphi>0.0;
    a->v0=v0; a->v1=v1;
    a->eta_sep0=eta0; a->eta_sep1=eta1;
    a->rho=rho; a->dphi=dphi;
    a->log_x_uncertainty=-INFINITY;

    if(!a->expanding) return 1;

    /* Binary64 quantization uncertainty in the transverse coordinate
         Delta eta = (1 + sign(b) q x) - eta_sep(v).
       Include x, q and b input ulps, and the shift of the asymptotic
       separatrix itself through v(q,b).  This is a conditioning estimate,
       not an approximation-error estimate. */
    double hx=cflow_half_ulp(x), hq=cflow_half_ulp(q), hb=cflow_half_ulp(b);
    double deta_dv=cflow_sep_eta_dv(v0);
    double dv_unc=hq/g + 0.5*fabs(v0/b)*hb;
    double eta_unc=fabs(q)*hx + fabs(x)*hq + fabs(deta_dv)*dv_unc;
    eta_unc += 4.0*DBL_EPSILON*fmax(1.0,fabs(q*x));
    if(!(eta_unc>0.0) || !isfinite(eta_unc)) return 1;

    /* The linear transverse model is only used inside the actual physical
       separatrix-to-boundary tube, enlarged by the input quantization itself.
       This avoids declaring generic high-friction states "unresolved". */
    if(fabs(delta_eta) > CFLOW_SEP_RHO_CONDITION*eta0 + eta_unc) return 1;

    double c0=sqrt(fmax(DBL_MIN,eta0*fmax(0.0,2.0-eta0)));
    double c1=sqrt(fmax(DBL_MIN,eta1*fmax(0.0,2.0-eta1)));
    if(!(c0>0.0) || !(c1>0.0) || q1==0.0) return 1;
    a->log_x_uncertainty=log(eta_unc)-log(c0)+dphi+log(c1)-log(fabs(q1));
    return 1;
}

double cflow_conditioning_logamp(double x,double q,double b,double h){
    cflow_sep_analysis a;
    if(!cflow_sep_analyze(x,q,b,h,&a)) return 0.0;
    return a.dphi;
}

int cflow_sep_unresolved(double x,double q,double b,double remaining){
    cflow_sep_analysis a;
    if(!cflow_sep_analyze(x,q,b,remaining,&a) || !a.expanding) return 0;
    if(!isfinite(a.log_x_uncertainty)) return 0;
    return a.log_x_uncertainty+log(CFLOW_SEP_CONDITION_SAFETY)>=log(CFLOW_SEP_X_UNCERTAINTY_LIMIT);
}
