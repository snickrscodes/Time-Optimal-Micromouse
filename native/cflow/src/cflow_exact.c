#include "cflow_internal.h"

static double sinc1(double x){
    double ax=fabs(x);
    if(ax<1e-4){ double x2=x*x; return 1.0-x2/6.0+x2*x2/120.0-x2*x2*x2/5040.0+x2*x2*x2*x2/362880.0; }
    return sin(x)/x;
}
static double dsinc1(double x){
    double ax=fabs(x);
    if(ax<1e-4){ double x2=x*x; return -x/3.0+x*x2/30.0-x*x2*x2/840.0+x*x2*x2*x2/45360.0; }
    return (x*cos(x)-sin(x))/(x*x);
}

int cflow_real_domain(double x,double q){
    if(!isfinite(x)||!isfinite(q)) return 0;
    return fabs(x*q)<=1.0+32.0*DBL_EPSILON;
}

void cflow_exact_b0(double x,double q,double h,int want_jac,cflow_local_jac *o){
    if(q==0.0){ o->x=x+2.0*h; o->dx=1.0; o->dq=0.0; o->db=0.0; return; }
    double z=q*x; double rad=fmax(0.0,1.0-z*z), c0=sqrt(rad); double w=2.0*q*h;
    double sw=sinc1(w), dsw=dsinc1(w), cw=cos(w), sn=sin(w);
    double X=x*cw+2.0*h*c0*sw; o->x=X;
    if(!want_jac){o->dx=o->dq=o->db=NAN;return;}
    o->dx=cw-2.0*h*q*q*x*sw/c0;
    o->dq=-2.0*h*x*sn+2.0*h*((-q*x*x/c0)*sw+c0*dsw*(2.0*h));
    /* Exact first b-variation. */
    double theta0=asin(z), theta1=theta0+w; double c1=cos(theta1);
    double scale=fabs(q)*(fabs(x)+2.0*fabs(h));
    if(scale<1e-4){
        /* Small-q expansion: S_b = -q[x^2 h^2 + (8/3)x h^3 + 2 h^4] + O(q^3). */
        o->db=-q*(x*x*h*h+(8.0/3.0)*x*h*h*h+2.0*h*h*h*h);
    }else if(c1<=0.0){
        o->db=NAN;
    }else{
        double lr=log1p((c1-c0)/c0);
        double eta=h*h-lr/(2.0*q*q);
        o->db=(c1/q)*eta-h*sin(theta1)/(q*q);
    }
}

double cflow_exact_b0_event(double x,double q){
    if(q==0.0) return INFINITY;
    double z=q*x; if(fabs(z)>=1.0) return 0.0;
    double th=asin(z), target=copysign(CFLOW_PI/2.0,q);
    double t=(target-th)/(2.0*q);
    return t>=0.0?t:INFINITY;
}
