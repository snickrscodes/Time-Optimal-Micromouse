#include "cflow_internal.h"

typedef struct { double v,dr,dk; } d2;
static d2 D(double v,double dr,double dk){d2 x={v,dr,dk};return x;}
static d2 da(d2 a,d2 b){return D(a.v+b.v,a.dr+b.dr,a.dk+b.dk);} static d2 ds(d2 a,d2 b){return D(a.v-b.v,a.dr-b.dr,a.dk-b.dk);}
static d2 dm(d2 a,d2 b){return D(a.v*b.v,a.dr*b.v+a.v*b.dr,a.dk*b.v+a.v*b.dk);}
static d2 dd(d2 a,d2 b){double iv=1.0/b.v,iv2=iv*iv;return D(a.v*iv,(a.dr*b.v-a.v*b.dr)*iv2,(a.dk*b.v-a.v*b.dk)*iv2);}
static d2 dscale(d2 a,double s){return D(a.v*s,a.dr*s,a.dk*s);} static d2 dneg(d2 a){return D(-a.v,-a.dr,-a.dk);}
static d2 dsqrt2(d2 a){double r=sqrt(fmax(a.v,0.0)),g=r>0?0.5/r:0.0;return D(r,g*a.dr,g*a.dk);}

static void rho_funcs(double rho,d2 *c,d2 *y0){
    double cv,yv,dc,dy;
    if(fabs(rho)<1e-7){
        double r=rho,r2=r*r,r3=r2*r,r4=r3*r;
        cv=1-r/2+r2/24-r3/720+r4/40320;
        dc=-0.5+r/12-r2/240+r3/10080-r4/725760;
        yv=1-r/6+r2/120-r3/5040+r4/362880;
        dy=-1.0/6+r/60-r2/1680+r3/90720-r4/7983360;
    }else{
        double d=sqrt(rho),sn=sin(d);cv=cos(d);yv=sn/d;dc=-0.5*yv;dy=(d*cv-sn)/(2*d*d*d);
    }
    *c=D(cv,dc,0);*y0=D(yv,dy,0);
}

static d2 event_rhs(double eta,d2 tau,d2 rho,d2 kap,int *ok){
    d2 c,y0;rho_funcs(rho.v,&c,&y0); /* overwrite rho derivatives from analytic functions */
    c.dr*=rho.dr;c.dk=0;y0.dr*=rho.dr;y0.dk=0;
    d2 y=dscale(y0,eta);
    d2 one=D(1,0,0); d2 Sy=dsqrt2(ds(one,dm(rho,dm(y,y))));
    d2 a=da(c,dm(kap,dm(rho,tau)));
    d2 inner=da(dm(kap,Sy),dm(dm(a,a),y));
    d2 den=dm(Sy,inner);
    if(!isfinite(den.v) || fabs(den.v)<1e-15){*ok=0;return D(0,0,0);}
    d2 F=dd(dneg(dm(a,y)),den);
    return dm(y0,F);
}

static d2 addd(d2 a,double h,d2 k){return D(a.v+h*k.v,a.dr+h*k.dr,a.dk+h*k.dk);}
static d2 combd(d2 y,double h,d2 *k,const double *a,int n){for(int i=0;i<n;i++)y=addd(y,h*a[i],k[i]);return y;}

int cflow_event_lambda(double rho_v,double kap_v,cflow_event_local *ev){
    if(kap_v==0.0){
        double d=sqrt(rho_v),c=cos(d); ev->lambda=1.0/c;
        /* d sec(sqrt(rho))/d rho = sec*tan/(2 sqrt(rho)); regular limit 1/2. */
        ev->dl_drho=(rho_v<1e-12)?0.5:(sin(d)/(2*d*c*c)); ev->dl_dkappa=-INFINITY; ev->ok=1;ev->conditioning=1;ev->no_event=0;return 1;
    }
    d2 rho=D(rho_v,1,0),kap=D(kap_v,0,1),tau=D(0,0,0);
    double eta=1.0,dt=-0.05;int it=0;d2 k1cache=D(0,0,0);int have_k1=0;
    while(eta>0.0 && it++<100000){
        if(eta+dt<0.0)dt=-eta;
        d2 k[7],yt;int ok=1;
        if(CFLOW_E0_FSAL_ENABLED && have_k1)k[0]=k1cache;else k[0]=event_rhs(eta,tau,rho,kap,&ok);
        const double a21[]={1.0/5};yt=combd(tau,dt,k,a21,1);k[1]=event_rhs(eta+dt/5,yt,rho,kap,&ok);
        const double a31[]={3.0/40,9.0/40};yt=combd(tau,dt,k,a31,2);k[2]=event_rhs(eta+3*dt/10,yt,rho,kap,&ok);
        const double a41[]={44.0/45,-56.0/15,32.0/9};yt=combd(tau,dt,k,a41,3);k[3]=event_rhs(eta+4*dt/5,yt,rho,kap,&ok);
        const double a51[]={19372.0/6561,-25360.0/2187,64448.0/6561,-212.0/729};yt=combd(tau,dt,k,a51,4);k[4]=event_rhs(eta+8*dt/9,yt,rho,kap,&ok);
        const double a61[]={9017.0/3168,-355.0/33,46732.0/5247,49.0/176,-5103.0/18656};yt=combd(tau,dt,k,a61,5);k[5]=event_rhs(eta+dt,yt,rho,kap,&ok);
        const double a71[]={35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84};yt=combd(tau,dt,k,a71,6);k[6]=event_rhs(eta+dt,yt,rho,kap,&ok);
        if(!ok){have_k1=0;ev->ok=0;ev->no_event=1;return 0;}
        const double b5[]={35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0};
        const double b4[]={5179.0/57600,0,7571.0/16695,393.0/640,-92097.0/339200,187.0/2100,1.0/40};
        d2 y5=tau,y4=tau;for(int i=0;i<7;i++){y5=addd(y5,dt*b5[i],k[i]);y4=addd(y4,dt*b4[i],k[i]);}
        double e0=fabs(y5.v-y4.v)/(CFLOW_EVENT_RK_ATOL+CFLOW_EVENT_RK_RTOL*fmax(1.0,fabs(y5.v)));
        double e1=fabs(y5.dr-y4.dr)/(2e-13+1e-12*fmax(1.0,fabs(y5.dr)));
        double e2=fabs(y5.dk-y4.dk)/(2e-13+1e-12*fmax(1.0,fabs(y5.dk)));
        double err=fmax(e0,fmax(e1,e2));
        if(err<=1.0){tau=y5;eta+=dt;if(CFLOW_E0_FSAL_ENABLED){k1cache=k[6];have_k1=1;}}else if(CFLOW_E0_FSAL_ENABLED){k1cache=k[0];have_k1=1;}
        double fac=err==0?5.0:0.9*pow(err,-0.2);if(fac<0.15)fac=0.15;if(fac>4.0)fac=4.0;dt*=fac;
        if(fabs(dt)<4*DBL_EPSILON*fmax(1.0,eta)){ev->ok=0;ev->conditioning=1;return 0;}
    }
    if(it>=100000 || !isfinite(tau.v)){ev->ok=0;return 0;}
    ev->lambda=tau.v;ev->dl_drho=tau.dr;ev->dl_dkappa=tau.dk;ev->ok=1;ev->no_event=0;
    ev->conditioning=(!isfinite(tau.dk)||fabs(tau.dk)>1e12);return 1;
}

int cflow_terminal_event_local(double x,double q,double b,cflow_event_local *ev,double *te,double *dte_dx,double *dte_dq,double *dte_db){
    ev->ok=ev->no_event=ev->conditioning=0; *te=INFINITY; if(dte_dx)*dte_dx=NAN;if(dte_dq)*dte_dq=NAN;if(dte_db)*dte_db=NAN;
    double z=q*x,az=fabs(z); if(az>=1.0){*te=0.0;return 1;} if(az<cos(CFLOW_TERM_DELTA) || x==0.0) return 0;
    double sig=z>=0?1.0:-1.0, sd=sqrt(fmax(0.0,(1.0-az)*(1.0+az))), delta=atan2(sd,az), rho=delta*delta;
    double zdot=b*x+2.0*q*sd; if(sig*zdot<=0.0) return 0;
    double kap=sig*b*x*x/(2.0*delta);
    if(!cflow_event_lambda(rho,kap,ev)) return 0;
    double T=0.5*x*delta*ev->lambda; if(!(T>0.0) || !isfinite(T)){ev->no_event=1;return 0;} *te=T;
    if(dte_dx&&dte_dq&&dte_db){
        double ddx=-sig*q/sd, ddq=-sig*x/sd;
        double drx=2.0*delta*ddx, drq=2.0*delta*ddq;
        double kx=sig*b*x/delta-kap*ddx/delta;
        double kq=-kap*ddq/delta;
        double kb=sig*x*x/(2.0*delta);
        double lx=ev->dl_drho*drx+ev->dl_dkappa*kx;
        double lq=ev->dl_drho*drq+ev->dl_dkappa*kq;
        double lb=ev->dl_dkappa*kb;
        *dte_dx=0.5*(delta*ev->lambda+x*ddx*ev->lambda+x*delta*lx);
        *dte_dq=0.5*x*(ddq*ev->lambda+delta*lq);
        *dte_db=0.5*x*delta*lb;
    }
    return 1;
}


/* Return the existing authoritative terminal event only when it could lie in
 * the requested forward interval.  In the real domain the physical equation
 * is x' = 2*sqrt(1-(q*x)^2), hence |x'| <= 2, while q(t)=q+b*t exactly.
 * Therefore for 0 <= t <= h,
 *
 *   |q(t)*x(t)| <= (|q|+|b|h)(|x|+2h).
 *
 * Each arithmetic stage below is rounded one representable value toward
 * +infinity.  A finite bound strictly below one is consequently a certificate
 * that |q*x|=1 is impossible on this interval.  Any non-finite/ambiguous case
 * falls through to the unchanged event solver. */
static int terminal_event_impossible_on_forward_interval(double x,double q,double b,double h){
    if(!(h > 0.0) || !isfinite(h)) return 0;
    double qmax = nextafter(fma(fabs(b), h, fabs(q)), INFINITY);
    double xmax = nextafter(fma(2.0, h, fabs(x)), INFINITY);
    double zmax = nextafter(qmax * xmax, INFINITY);
    return isfinite(zmax) && zmax < 1.0;
}

/* A sign-stable contracting-|q| trajectory starting strictly inside the
 * real domain cannot newly hit |q*x|=1.  With x>0 and q*b<0, |q| decreases;
 * at a hypothetical first terminal contact x'=0 while d|q|/dt=-|b|, so the
 * boundary derivative of |q|*x is strictly inward.  The only remaining
 * requirement is proving that q cannot cross zero over this public horizon.
 * fma gives the correctly rounded endpoint q; one nextafter step toward zero
 * is therefore a conservative sign bound.  Ambiguous/non-finite cases fail
 * closed to the existing per-map event query.
 *
 * CFLOW_E0_DISABLE_NO_EVENT_HOIST is an internal test-build switch only. */
int cflow_contracting_no_terminal_event(double x,double q,double b,double h){
#ifdef CFLOW_E0_DISABLE_NO_EVENT_HOIST
    (void)x;(void)q;(void)b;(void)h;
    return 0;
#else
    if(!isfinite(x)||!isfinite(q)||!isfinite(b)||!isfinite(h))return 0;
    if(!(x>0.0) || !(h>=0.0) || q==0.0 || b==0.0 || !(q*b<0.0))return 0;
    double qe=fma(b,h,q);
    if(!isfinite(qe))return 0;
    if(q>0.0){
        double lower=nextafter(qe,-INFINITY);
        return lower>0.0;
    }
    double upper=nextafter(qe,INFINITY);
    return upper<0.0;
#endif
}

int cflow_terminal_event_local_with_horizon(double x,double q,double b,double h,cflow_event_local *ev,double *te,double *dte_dx,double *dte_dq,double *dte_db){
    if(terminal_event_impossible_on_forward_interval(x,q,b,h)){
        ev->ok=ev->no_event=ev->conditioning=0;
        *te=INFINITY;
        if(dte_dx)*dte_dx=NAN;
        if(dte_dq)*dte_dq=NAN;
        if(dte_db)*dte_db=NAN;
        return 0;
    }
    return cflow_terminal_event_local(x,q,b,ev,te,dte_dx,dte_dq,dte_db);
}
