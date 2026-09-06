#include "cflow_internal.h"

typedef struct { double x,sx,sq,sb,j,jx,jq,jb; } irkst;
static irkst iadd(irkst a,double h,irkst k){double *p=&a.x,*q=&k.x;for(int i=0;i<8;i++)p[i]+=h*q[i];return a;}
static irkst ilin(irkst y,double h,const irkst *k,const double *a,int n){for(int i=0;i<n;i++)y=iadd(y,h*a[i],k[i]);return y;}
static int ider(double t,irkst y,double q0,double b,irkst *d){
    double q=q0+b*t,z=q*y.x,rr=1-z*z;
    if(rr<=0.0||y.x<=0.0)return 0;
    double sr=sqrt(rr),rootx=sqrt(y.x),invx3=1.0/(y.x*rootx);
    d->x=2*sr;
    double fx=-2*q*z/sr,fq=-2*y.x*z/sr;
    d->sx=fx*y.sx;d->sq=fx*y.sq+fq;d->sb=fx*y.sb+fq*t;
    d->j=1.0/rootx;d->jx=-.5*invx3*y.sx;d->jq=-.5*invx3*y.sq;d->jb=-.5*invx3*y.sb;
    return 1;
}

static int terminal_all_unit(double x,double q,double b,double h,cflow_local_jac *flow,cflow_integral_local *integ){
    irkst y={x,1,0,0,0,0,0,0};double t=0,dt=h/8.0;int it=0;irkst k1cache={0};int have_k1=0;const double rtol=1.2e-13,atol_rel=8e-16;
    double S=hypot(x,2.0*h); if(S==0.0)S=fabs(x); if(S==0.0)S=DBL_MIN; double rs=sqrt(S);
    double scale[8]={S,1.0,pow(S,2.0),pow(S,3.0),rs,1.0/rs,S*rs,S*S*rs};
    for(int ii=0;ii<8;ii++){if(scale[ii]==0.0)scale[ii]=DBL_MIN;else if(!isfinite(scale[ii]))scale[ii]=DBL_MAX;}
    if(h==0.0){
        if(flow){flow->x=x;flow->dx=1;flow->dq=flow->db=0;}
        if(integ){integ->j=integ->jq=integ->jb=0;integ->jx=x>0?0:-INFINITY;integ->regx=x>0?.5/sqrt(x):0;}
        return 1;
    }
    while(t<h&&it++<30000){
        if(t+dt>h)dt=h-t;
        irkst k[7]={{0}},yt;int ok=1;if(CFLOW_E0_FSAL_ENABLED && have_k1)k[0]=k1cache;else ok&=ider(t,y,q,b,&k[0]);
        const double a21[]={1./5};yt=ilin(y,dt,k,a21,1);ok&=ider(t+dt/5,yt,q,b,&k[1]);
        const double a31[]={3./40,9./40};yt=ilin(y,dt,k,a31,2);ok&=ider(t+3*dt/10,yt,q,b,&k[2]);
        const double a41[]={44./45,-56./15,32./9};yt=ilin(y,dt,k,a41,3);ok&=ider(t+4*dt/5,yt,q,b,&k[3]);
        const double a51[]={19372./6561,-25360./2187,64448./6561,-212./729};yt=ilin(y,dt,k,a51,4);ok&=ider(t+8*dt/9,yt,q,b,&k[4]);
        const double a61[]={9017./3168,-355./33,46732./5247,49./176,-5103./18656};yt=ilin(y,dt,k,a61,5);ok&=ider(t+dt,yt,q,b,&k[5]);
        const double a71[]={35./384,0,500./1113,125./192,-2187./6784,11./84};yt=ilin(y,dt,k,a71,6);ok&=ider(t+dt,yt,q,b,&k[6]);
        if(!ok){have_k1=0;dt*=.25;if(dt<=DBL_MIN)break;continue;}
        const double b5[]={35./384,0,500./1113,125./192,-2187./6784,11./84,0};
        const double b4[]={5179./57600,0,7571./16695,393./640,-92097./339200,187./2100,1./40};
        irkst y5=y,y4=y;for(int i=0;i<7;i++){y5=iadd(y5,dt*b5[i],k[i]);y4=iadd(y4,dt*b4[i],k[i]);}
        double *p=&y.x,*a=&y5.x,*c=&y4.x;double err=0;
        for(int i=0;i<8;i++){double sc=atol_rel*scale[i]+rtol*fmax(scale[i],fmax(fabs(p[i]),fabs(a[i])));err=fmax(err,fabs(a[i]-c[i])/sc);}
        if(err<=1){y=y5;t+=dt;if(CFLOW_E0_FSAL_ENABLED){k1cache=k[6];have_k1=1;}}else if(CFLOW_E0_FSAL_ENABLED){k1cache=k[0];have_k1=1;}
        double fac=err==0?5:.9*pow(err,-.2);if(fac<.2)fac=.2;if(fac>5)fac=5;dt*=fac;
    }
    if(t<h){
        if(flow)flow->x=flow->dx=flow->dq=flow->db=NAN;
        if(integ)integ->j=integ->jx=integ->jq=integ->jb=integ->regx=NAN;
        return 0;
    }
    if(flow){flow->x=y.x;flow->dx=y.sx;flow->dq=y.sq;flow->db=y.sb;}
    if(integ){integ->j=y.j;integ->jx=y.jx;integ->jq=y.jq;integ->jb=y.jb;integ->regx=x>0?y.jx+.5/sqrt(x):0.0;}
    return 1;
}

int cflow_integral_terminal_all_local(double x,double q,double b,double h,cflow_local_jac *flow,cflow_integral_local *integ){
    if(h==0.0)return terminal_all_unit(x,q,b,h,flow,integ);
    /* Normalize the entire augmented local IVP before integrating.  The ODE
       scaling symmetry makes this exact, while keeping state and sensitivity
       channels near natural O(1) magnitudes.  This is essential for history
       segments at extreme binary64 scales. */
    double S=hypot(x,2.0*h);
    if(!(S>0.0)||!isfinite(S))return 0;
    double xn=x/S, hn=h/S;
    double qn=q*S;
    double bn=(b*S)*S;
    if(!isfinite(xn)||!isfinite(hn)||!isfinite(qn)||!isfinite(bn))return 0;
    cflow_local_jac fn; cflow_integral_local in;
    if(!terminal_all_unit(xn,qn,bn,hn,&fn,&in))return 0;
    double S2=S*S, S3=S2*S, rs=sqrt(S), S32=S*rs, S52=S2*rs;
    if(flow){
        flow->x=S*fn.x;
        flow->dx=fn.dx;
        flow->dq=S2*fn.dq;
        flow->db=S3*fn.db;
    }
    if(integ){
        integ->j=rs*in.j;
        integ->jx=in.jx/rs;
        integ->jq=S32*in.jq;
        integ->jb=S52*in.jb;
        integ->regx=in.regx/rs;
    }
    return (!flow || (isfinite(flow->x)&&isfinite(flow->dx)&&isfinite(flow->dq)&&isfinite(flow->db))) &&
           (!integ || (isfinite(integ->j)&&isfinite(integ->jq)&&isfinite(integ->jb)&&isfinite(integ->regx)));
}

void cflow_integral_terminal_local(double x,double q,double b,double h,cflow_integral_local *o){
    cflow_local_jac f;(void)cflow_integral_terminal_all_local(x,q,b,h,&f,o);
}


typedef struct { double x,j; } vrkst;
static vrkst vadd(vrkst a,double h,vrkst k){a.x+=h*k.x;a.j+=h*k.j;return a;}
static vrkst vlin(vrkst y,double h,const vrkst*k,const double*a,int n){for(int i=0;i<n;i++)y=vadd(y,h*a[i],k[i]);return y;}
static int vder(double t,vrkst y,double q0,double b,vrkst*d){double q=q0+b*t,z=q*y.x,rr=1-z*z;if(rr<=0.0||y.x<=0.0)return 0;d->x=2*sqrt(rr);d->j=1.0/sqrt(y.x);return 1;}
static int terminal_value_unit(double x,double q,double b,double h,double*jout){
    vrkst y={x,0.0};double t=0,dt=h/8.0;int it=0;vrkst k1cache={0};int have_k1=0;const double rtol=1.2e-13,atol_rel=8e-16;double S=hypot(x,2.0*h);if(S==0.0)S=fabs(x);if(S==0.0)S=DBL_MIN;double rs=sqrt(S);double sx=S,sj=rs;
    if(h==0.0){*jout=0.0;return 1;}
    while(t<h&&it++<30000){if(t+dt>h)dt=h-t;vrkst k[7]={{0}},yt;int ok=1;if(CFLOW_E0_FSAL_ENABLED && have_k1)k[0]=k1cache;else ok&=vder(t,y,q,b,&k[0]);const double a21[]={1./5};yt=vlin(y,dt,k,a21,1);ok&=vder(t+dt/5,yt,q,b,&k[1]);const double a31[]={3./40,9./40};yt=vlin(y,dt,k,a31,2);ok&=vder(t+3*dt/10,yt,q,b,&k[2]);const double a41[]={44./45,-56./15,32./9};yt=vlin(y,dt,k,a41,3);ok&=vder(t+4*dt/5,yt,q,b,&k[3]);const double a51[]={19372./6561,-25360./2187,64448./6561,-212./729};yt=vlin(y,dt,k,a51,4);ok&=vder(t+8*dt/9,yt,q,b,&k[4]);const double a61[]={9017./3168,-355./33,46732./5247,49./176,-5103./18656};yt=vlin(y,dt,k,a61,5);ok&=vder(t+dt,yt,q,b,&k[5]);const double a71[]={35./384,0,500./1113,125./192,-2187./6784,11./84};yt=vlin(y,dt,k,a71,6);ok&=vder(t+dt,yt,q,b,&k[6]);if(!ok){have_k1=0;dt*=.25;if(dt<=DBL_MIN)break;continue;}const double b5[]={35./384,0,500./1113,125./192,-2187./6784,11./84,0};const double b4[]={5179./57600,0,7571./16695,393./640,-92097./339200,187./2100,1./40};vrkst y5=y,y4=y;for(int i=0;i<7;i++){y5=vadd(y5,dt*b5[i],k[i]);y4=vadd(y4,dt*b4[i],k[i]);}double ex=fabs(y5.x-y4.x)/(atol_rel*sx+rtol*fmax(sx,fmax(fabs(y.x),fabs(y5.x))));double ej=fabs(y5.j-y4.j)/(atol_rel*sj+rtol*fmax(sj,fmax(fabs(y.j),fabs(y5.j))));double err=fmax(ex,ej);if(err<=1){y=y5;t+=dt;if(CFLOW_E0_FSAL_ENABLED){k1cache=k[6];have_k1=1;}}else if(CFLOW_E0_FSAL_ENABLED){k1cache=k[0];have_k1=1;}double fac=err==0?5:.9*pow(err,-.2);if(fac<.2)fac=.2;if(fac>5)fac=5;dt*=fac;
    }
    if(t<h||!isfinite(y.j))return 0;
    *jout=y.j;return 1;
}
int cflow_integral_terminal_value_local(double x,double q,double b,double h,double*j){if(h==0.0){*j=0;return 1;}double S=hypot(x,2.0*h);if(!(S>0.0)||!isfinite(S))return 0;double jn;if(!terminal_value_unit(x/S,q*S,(b*S)*S,h/S,&jn))return 0;*j=sqrt(S)*jn;return isfinite(*j);}
