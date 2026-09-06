#include "cflow_internal.h"
#include <string.h>

typedef struct {
    double x;
    double xq;
    double xb;
    double j;
    double jq;
    double jb;
} bstate;

static bstate badd(bstate a, double h, bstate k){
    double *pa=&a.x, *pk=&k.x;
    for(int i=0;i<6;i++) pa[i]+=h*pk[i];
    return a;
}
static bstate blin(bstate y,double h,const bstate *k,const double *a,int n){
    for(int i=0;i<n;i++) y=badd(y,h*a[i],k[i]);
    return y;
}

/* Exact inward-boundary continuation in y=sqrt(t).  The q0 derivative is
   constrained to remain on x0=1/|q0|, matching the historical boundary-start
   segment semantics. */
static int bderiv(double y,bstate st,double q0,double b,bstate *d){
    double t=y*y, q=fma(b,t,q0), z=q*st.x;
    double rr=fma(-z,z,1.0);
    if(rr < -256.0*DBL_EPSILON) return 0;
    double g=sqrt(fmax(0.0,rr));
    d->x=4.0*y*g;
    if(y==0.0 || g <= 16.0*DBL_MIN){
        /* 1-z^2 = (-2 b/q0) y^2 + O(y^3). */
        double c=sqrt(-2.0*b/q0);
        if(!(c>0.0) || !isfinite(c)) return 0;
        double fx=-4.0*fabs(q0)/c;
        double fq=-4.0/(q0*c);
        d->xq=fx*st.xq+fq;
        d->xb=fx*st.xb; /* direct q_b=y^2 vanishes at y=0 */
    }else{
        double fx=-4.0*y*q*z/g;
        double fq=-4.0*y*st.x*z/g;
        d->xq=fx*st.xq+fq;
        d->xb=fx*st.xb+fq*t;
    }
    if(!(st.x>0.0)) return 0;
    double invroot=1.0/sqrt(st.x), invx3=invroot/st.x;
    d->j=2.0*y*invroot;
    d->jq=-y*invx3*st.xq;
    d->jb=-y*invx3*st.xb;
    return 1;
}

static int boundary_integrate(double x0,double q0,double b,double h,bstate *out){
    if(!(h>=0.0) || !(x0>0.0) || q0==0.0 || !(b/q0<0.0)) return 0;
    double expect=1.0/fabs(q0);
    double tol=256.0*DBL_EPSILON*fmax(1.0,expect);
    if(fabs(x0-expect)>tol) return 0;
    double sg=copysign(1.0,q0);
    bstate st={expect,-sg/(q0*q0),0.0,0.0,0.0,0.0};
    if(h==0.0){*out=st;return 1;}
    double ymax=sqrt(h), y=0.0, dy=ymax/8.0;
    if(!(dy>0.0))dy=ymax;
    const double rtol=8e-14, atol=2e-15;
    int it=0;
    while(y<ymax && it++<30000){
        if(y+dy>ymax)dy=ymax-y;
        bstate k[7]={{0}},yt;int ok=1;
        ok&=bderiv(y,st,q0,b,&k[0]);
        const double a21[]={1.0/5};yt=blin(st,dy,k,a21,1);ok&=bderiv(y+dy/5,yt,q0,b,&k[1]);
        const double a31[]={3.0/40,9.0/40};yt=blin(st,dy,k,a31,2);ok&=bderiv(y+3*dy/10,yt,q0,b,&k[2]);
        const double a41[]={44.0/45,-56.0/15,32.0/9};yt=blin(st,dy,k,a41,3);ok&=bderiv(y+4*dy/5,yt,q0,b,&k[3]);
        const double a51[]={19372.0/6561,-25360.0/2187,64448.0/6561,-212.0/729};yt=blin(st,dy,k,a51,4);ok&=bderiv(y+8*dy/9,yt,q0,b,&k[4]);
        const double a61[]={9017.0/3168,-355.0/33,46732.0/5247,49.0/176,-5103.0/18656};yt=blin(st,dy,k,a61,5);ok&=bderiv(y+dy,yt,q0,b,&k[5]);
        const double a71[]={35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84};yt=blin(st,dy,k,a71,6);ok&=bderiv(y+dy,yt,q0,b,&k[6]);
        if(!ok){dy*=0.25;if(!(dy>DBL_MIN))return 0;continue;}
        const double c5[]={35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0};
        const double c4[]={5179.0/57600,0,7571.0/16695,393.0/640,-92097.0/339200,187.0/2100,1.0/40};
        bstate s5=st,s4=st;
        for(int i=0;i<7;i++){s5=badd(s5,dy*c5[i],k[i]);s4=badd(s4,dy*c4[i],k[i]);}
        double *p=&st.x,*a=&s5.x,*c=&s4.x;double err=0.0;
        for(int i=0;i<6;i++){
            double sc=atol+rtol*fmax(1.0,fmax(fabs(p[i]),fabs(a[i])));
            err=fmax(err,fabs(a[i]-c[i])/sc);
        }
        if(err<=1.0){st=s5;y+=dy;}
        double fac=err==0.0?5.0:0.9*pow(err,-0.2);if(fac<0.2)fac=0.2;if(fac>5.0)fac=5.0;dy*=fac;
    }
    if(y<ymax || !isfinite(st.x)||!isfinite(st.xq)||!isfinite(st.xb)||!isfinite(st.j)||!isfinite(st.jq)||!isfinite(st.jb))return 0;
    *out=st;return 1;
}

static cflow_status boundary_status(double x0,double q0,double b,double h){
    if(!isfinite(x0)||!isfinite(q0)||!isfinite(b)||!isfinite(h)||h<0.0)return CFLOW_INVALID_ARGUMENT;
    if(!(x0>0.0)||q0==0.0||!(b/q0<0.0))return CFLOW_OUTSIDE_REAL_DOMAIN;
    double expect=1.0/fabs(q0),tol=256.0*DBL_EPSILON*fmax(1.0,expect);
    if(fabs(x0-expect)>tol)return CFLOW_OUTSIDE_REAL_DOMAIN;
    return CFLOW_OK;
}

/* Scale an exact boundary start to x(0)=1 and q(0)=sign(q0).

       x = xscale * U,   t = xscale * tau,
       beta = b*xscale^2, xscale=1/|q0|.

   The canonical IVP depends only on sign(q0), beta and tau, so the one-sided
   extension inherits the same physical scaling invariance as the main flow.
   boundary_integrate supplies derivatives with respect to canonical beta in
   its xb/jb channels; the constrained physical q0 derivative is reconstructed
   analytically below. */
static int boundary_scaled(
    double x0,double q0,double b,double h,
    bstate *s,double *xscale,double *tau,double *beta
){
    (void)x0;
    *xscale=1.0/fabs(q0);
    *tau=h/(*xscale);
    *beta=(b*(*xscale))*(*xscale);
    double sg=copysign(1.0,q0);
    return boundary_integrate(1.0,sg,*beta,*tau,s);
}

static void boundary_chain(
    const bstate *s,double q0,double xscale,double tau,double beta,
    double *x,double *xq,double *xb,double *xh,
    double *j,double *jq,double *jb,double *jh
){
    double sg=copysign(1.0,q0);
    double qhat=fma(beta,tau,sg);
    double z=qhat*s->x;
    double ut=2.0*sqrt(fmax(0.0,fma(-z,z,1.0)));
    double it=1.0/sqrt(s->x);
    double xs_q=-sg/(q0*q0);
    double tau_q=tau/q0;
    double beta_q=-2.0*beta/q0;
    double xs2=xscale*xscale;
    double rootxs=sqrt(xscale);

    *x=xscale*s->x;
    *xq=xs_q*s->x + xscale*(ut*tau_q + s->xb*beta_q);
    *xb=xscale*s->xb*xs2;
    *xh=ut;

    *j=rootxs*s->j;
    *jq=rootxs*((-0.5/q0)*s->j + it*tau_q + s->jb*beta_q);
    *jb=rootxs*s->jb*xs2;
    *jh=it/rootxs;
}

void cflow_boundary_eval(double x0,double q0,double b,double h,cflow_eval_result *o){
    if(!o) return;
    memset(o,0,sizeof(*o));
    o->x=x0; o->event_time=NAN; o->status=boundary_status(x0,q0,b,h);
    if(o->status!=CFLOW_OK) return;
    bstate s; double xs,tau,beta;
    if(!boundary_scaled(x0,q0,b,h,&s,&xs,&tau,&beta)){o->status=CFLOW_NUMERICAL_FAILURE;return;}
    o->x=xs*s.x; o->steps=1;
}
void cflow_boundary_eval_jacobian(double x0,double q0,double b,double h,cflow_eval_jac_result *o){
    if(!o) return;
    memset(o,0,sizeof(*o));
    o->x=x0; o->event_time=NAN; o->status=boundary_status(x0,q0,b,h);
    if(o->status!=CFLOW_OK) return;
    bstate s; double xs,tau,beta,x,xq,xb,xh,j,jq,jb,jh;
    if(!boundary_scaled(x0,q0,b,h,&s,&xs,&tau,&beta)){o->status=CFLOW_NUMERICAL_FAILURE;return;}
    boundary_chain(&s,q0,xs,tau,beta,&x,&xq,&xb,&xh,&j,&jq,&jb,&jh);
    o->x=x; o->dx_dx0=0.0; o->dx_dq0=xq; o->dx_db=xb; o->dx_dh=xh; o->steps=1;
}
void cflow_boundary_integral_jacobian(double x0,double q0,double b,double h,cflow_integral_jac_result *o){
    if(!o) return;
    memset(o,0,sizeof(*o));
    o->integral=o->dI_dx0=o->dI_dq0=o->dI_db=o->dI_dh=o->regularized_dI_dx0=NAN;
    o->event_time=NAN; o->status=boundary_status(x0,q0,b,h);
    if(o->status!=CFLOW_OK) return;
    bstate s; double xs,tau,beta,x,xq,xb,xh,j,jq,jb,jh;
    if(!boundary_scaled(x0,q0,b,h,&s,&xs,&tau,&beta)){o->status=CFLOW_NUMERICAL_FAILURE;return;}
    boundary_chain(&s,q0,xs,tau,beta,&x,&xq,&xb,&xh,&j,&jq,&jb,&jh);
    o->integral=j; o->dI_dx0=0.0; o->dI_dq0=jq; o->dI_db=jb;
    o->dI_dh=jh; o->regularized_dI_dx0=0.0; o->steps=1;
}

void cflow_boundary_integral_value(double x0,double q0,double b,double h,cflow_integral_value_result *o){
    if(!o) return;
    memset(o,0,sizeof(*o));
    o->integral=NAN; o->event_time=NAN; o->status=boundary_status(x0,q0,b,h);
    if(o->status!=CFLOW_OK) return;
    bstate s; double xs,tau,beta;
    if(!boundary_scaled(x0,q0,b,h,&s,&xs,&tau,&beta)){o->status=CFLOW_NUMERICAL_FAILURE;return;}
    o->integral=sqrt(xs)*s.j; o->steps=1;
}

void cflow_boundary_eval_all(double x0,double q0,double b,double h,cflow_all_result *o){
    if(!o) return;
    memset(o,0,sizeof(*o));
    o->x=x0; o->event_time=NAN; o->status=boundary_status(x0,q0,b,h);
    if(o->status!=CFLOW_OK) return;
    bstate s; double xs,tau,beta,x,xq,xb,xh,j,jq,jb,jh;
    if(!boundary_scaled(x0,q0,b,h,&s,&xs,&tau,&beta)){o->status=CFLOW_NUMERICAL_FAILURE;return;}
    boundary_chain(&s,q0,xs,tau,beta,&x,&xq,&xb,&xh,&j,&jq,&jb,&jh);
    o->x=x; o->dx_dx0=0.0; o->dx_dq0=xq; o->dx_db=xb; o->dx_dh=xh;
    o->integral=j; o->dI_dx0=0.0; o->dI_dq0=jq; o->dI_db=jb;
    o->dI_dh=jh; o->regularized_dI_dx0=0.0; o->steps=1;
}
