#include "cflow_internal.h"

typedef struct { double x,sx,sq,sb; } rkstate;
static rkstate add_scaled(rkstate a,double h,rkstate b){ a.x+=h*b.x;a.sx+=h*b.sx;a.sq+=h*b.sq;a.sb+=h*b.sb;return a; }
static rkstate lincomb(rkstate y,double h,const rkstate *k,const double *a,int n){ for(int i=0;i<n;i++) y=add_scaled(y,h*a[i],k[i]); return y; }

static int deriv(double t,rkstate y,double q0,double b,int jac,rkstate *d){
    double q=q0+b*t, z=q*y.x, rr=1.0-z*z; if(rr<=0.0) return 0; double s=sqrt(rr); double f=2.0*s;
    d->x=f;
    if(!jac){d->sx=d->sq=d->sb=0;return 1;}
    double fx=-2.0*q*z/s, fq=-2.0*y.x*z/s;
    d->sx=fx*y.sx;
    d->sq=fx*y.sq+fq;
    d->sb=fx*y.sb+fq*t;
    return 1;
}

void cflow_rk_physical_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o){
    rkstate y={x,1.0,0.0,0.0}; if(h==0.0){o->x=x;o->dx=1;o->dq=o->db=0;return;}
    double t=0.0, dt=h/8.0; if(dt<=0)dt=h; int it=0; rkstate k1cache={0}; int have_k1=0;
    const double rtol=4e-14, atol_rel=4e-16;
    double S=hypot(x,2.0*h); if(S==0.0)S=fabs(x); if(S==0.0)S=DBL_MIN;
    double scx=S, scsx=1.0, scsq=pow(S,2.0), scsb=pow(S,3.0);
    if(scx==0.0)scx=DBL_MIN;
    if(!isfinite(scx))scx=DBL_MAX;
    if(scsq==0.0)scsq=DBL_MIN;
    if(!isfinite(scsq))scsq=DBL_MAX;
    if(scsb==0.0)scsb=DBL_MIN;
    if(!isfinite(scsb))scsb=DBL_MAX;
    while(t<h && it++<20000){
        if(t+dt>h) dt=h-t;
        rkstate k[7]={{0}},yt; int ok=1;
        if(CFLOW_E0_FSAL_ENABLED && have_k1)k[0]=k1cache;else ok&=deriv(t,y,q,b,want_jac,&k[0]);
        const double a21[]={1.0/5}; yt=lincomb(y,dt,k,a21,1); ok&=deriv(t+dt/5,yt,q,b,want_jac,&k[1]);
        const double a31[]={3.0/40,9.0/40}; yt=lincomb(y,dt,k,a31,2); ok&=deriv(t+3*dt/10,yt,q,b,want_jac,&k[2]);
        const double a41[]={44.0/45,-56.0/15,32.0/9}; yt=lincomb(y,dt,k,a41,3); ok&=deriv(t+4*dt/5,yt,q,b,want_jac,&k[3]);
        const double a51[]={19372.0/6561,-25360.0/2187,64448.0/6561,-212.0/729}; yt=lincomb(y,dt,k,a51,4); ok&=deriv(t+8*dt/9,yt,q,b,want_jac,&k[4]);
        const double a61[]={9017.0/3168,-355.0/33,46732.0/5247,49.0/176,-5103.0/18656}; yt=lincomb(y,dt,k,a61,5); ok&=deriv(t+dt,yt,q,b,want_jac,&k[5]);
        const double a71[]={35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84}; yt=lincomb(y,dt,k,a71,6); ok&=deriv(t+dt,yt,q,b,want_jac,&k[6]);
        if(!ok){ have_k1=0; dt*=0.25; if(dt<=DBL_MIN){o->x=NAN;o->dx=o->dq=o->db=NAN;return;} continue; }
        const double b5[]={35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0};
        const double b4[]={5179.0/57600,0,7571.0/16695,393.0/640,-92097.0/339200,187.0/2100,1.0/40};
        rkstate y5=y,y4=y;
        for(int i=0;i<7;i++){ y5=add_scaled(y5,dt*b5[i],k[i]); y4=add_scaled(y4,dt*b4[i],k[i]); }
        double err=fabs(y5.x-y4.x)/(atol_rel*scx+rtol*fmax(scx,fmax(fabs(y.x),fabs(y5.x))));
        if(want_jac){
            double e1=fabs(y5.sx-y4.sx)/(atol_rel*scsx+rtol*fmax(scsx,fmax(fabs(y.sx),fabs(y5.sx))));
            double e2=fabs(y5.sq-y4.sq)/(atol_rel*scsq+rtol*fmax(scsq,fmax(fabs(y.sq),fabs(y5.sq))));
            double e3=fabs(y5.sb-y4.sb)/(atol_rel*scsb+rtol*fmax(scsb,fmax(fabs(y.sb),fabs(y5.sb))));
            err=fmax(err,fmax(e1,fmax(e2,e3)));
        }
        if(err<=1.0){ y=y5; t+=dt; if(CFLOW_E0_FSAL_ENABLED){k1cache=k[6];have_k1=1;} } else if(CFLOW_E0_FSAL_ENABLED){k1cache=k[0];have_k1=1;}
        double fac=err==0.0?5.0:0.9*pow(err,-0.2); if(fac<0.2)fac=0.2;if(fac>5.0)fac=5.0;dt*=fac;
    }
    if(it>=20000){o->x=NAN;o->dx=o->dq=o->db=NAN;return;}
    o->x=y.x;o->dx=want_jac?y.sx:NAN;o->dq=want_jac?y.sq:NAN;o->db=want_jac?y.sb:NAN;
}
