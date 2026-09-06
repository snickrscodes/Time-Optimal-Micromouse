#include "segment_internal.h"

#define B_INV (1.0/AME_B_EMF)
#define LW_EQ (1.0/AME_V_MAX)
#define W_EQ (AME_V_MAX*AME_V_MAX)
#define LW_EQB (-AME_A_MAX/W_EQ)
#define DT_DW0_SCALE (0.5*LW_EQ/AME_A_MAX)
#define SMALL_STEP_TOL 2.0e-3
#define GAP_DIRECT_MAX 3.0e-3
#define GAP_ONE_HALLEY_MAX 1.0e-1
#define GAP_LAMBERT_MIN 5.0e-1
#define TIME_REFINE_MAX 2.0
#define TIME_REFINE_REL 2.5e-1
#define INV_E 0x1.78b56362cef38p-2
#define INV_E_LO (-1.2428753672788363e-17)
#define NEG_INV_E (-INV_E)
#define LOG_HALF (-0.693147180559945309417232121458176568)

static double lambertw0_c(double x){
    const double E=2.718281828459045235360287471352662498;
    if(isnan(x)||isinf(x)||x==0.0)return x;
    if(x==NEG_INV_E)return -1.0;
    if(x<NEG_INV_E)return NAN;
    double w;
    if(x < -0.2){
        double u=(x+INV_E)+INV_E_LO,p=sqrt(2.0*E*u);
        if(u<1e-5){w=fma(fma(fma(fma(fma(fma(-221.0/8505.0,p,769.0/17280.0),p,-43.0/540.0),p,11.0/72.0),p,-1.0/3.0),p,1.0),p,-1.0);return w;}
        if(u<1e-3){w=fma(-1118511313.0/709296588000.0,p,169709463197.0/69528040243200.0);w=fma(w,p,-5776369.0/1515591000.0);w=fma(w,p,226287557.0/37623398400.0);w=fma(w,p,-1963.0/204120.0);w=fma(w,p,680863.0/43545600.0);w=fma(w,p,-221.0/8505.0);w=fma(w,p,769.0/17280.0);w=fma(w,p,-43.0/540.0);w=fma(w,p,11.0/72.0);w=fma(w,p,-1.0/3.0);w=fma(w,p,1.0);return fma(w,p,-1.0);}
        w=fma(fma(fma(fma(fma(fma(-221.0/8505.0,p,769.0/17280.0),p,-43.0/540.0),p,11.0/72.0),p,-1.0/3.0),p,1.0),p,-1.0);
        double eu=E*u,q=w+1.0;
        for(int i=0;i<2;++i){double em1=expm1(q),res=fma(q-1.0,em1,q)-eu,sr=res*exp(-q);q-=2*sr*q/fma(2*q,q,-sr*(q+1.0));}
        return q-1.0;
    } else if(x<0.5){double num=1+x*(2.2380488687194635+x*(0.5323229596042623-0.08326761577458537*x));double den=1+x*(3.238004480223941+2.2706502298872793*x);w=x*num/den;}
    else if(x<4.0){double y=log(x);w=(((((0.000254691122488741*y-0.001604936436645819)*y-0.0013647878217518169)*y+0.07367221971761982)*y+0.36189954940301794)*y+0.5671435067846463);}
    else if(x<64.0){double y=log(x);w=((((0.0003229652230849584*y-0.007594247957917449)*y+0.08435395036268026)*y+0.35232059530135407)*y+0.5706766967072727);}
    else {double l1=log(x),l2=log(l1),il=1/l1,l2l=l2*il;w=l1-l2+l2l*(1+0.5*l2l-il);if(x<1099511627776.0)w+=(((((-18.044416677004946*il+12.960214166179945)*il-2.0337318250793284)*il-0.1835022973519072)*il+0.011947745371139771)*il-0.00017964131708255804);else w+=l2*il*il*il*(l2*(2*l2-9)+6)/6;}
    double t=fma(-x,exp(-w),w),p=w+1;return w-t/fma(-0.5*(w+2),t/p,p);
}

static double qseries(double u,double e){double p2=fma(3,e,-1),p3=fma(e,16*e-11,1),p4=fma(e,fma(e,fma(125,e,-131),31),-1),p5=fma(e,fma(e,fma(e,fma(1296,e,-1829),731),-79),1);double h=fma(-p5/120.0,u,p4/24.0);h=fma(h,u,-p3/6.0);h=fma(h,u,0.5*p2);return fma(h,u,-1.0);}
static void small_step(double v,double tau,double*q,double*z,double*theta,double*m){double c3=fma(-2,v,3)/6.0,c4=fma(v,fma(6,v,-20),15)/24.0,c5=fma(v,fma(v,fma(-24,v,130),-210),105)/120.0,c6=fma(v,fma(v,fma(v,fma(120,v,-924),2380),-2520),945)/720.0;double h=fma(c6,tau,c5);h=fma(h,tau,c4);h=fma(h,tau,c3);h=fma(h,tau,0.5);double ss=tau*fma(tau,h,1);*q=fma(v,ss,1);*z=v*fma(v,ss,1-ss);double ut=fma(v,tau,-tau);*theta=v*tau*fma(ut,h,-1);*m=ss/(*z);}
static void small_params(double v,double u,double*tp,double*mx){if(v==0){*tp=*mx=0;return;}double v2=v*v;if(v2==0){*tp=*mx=0;return;}*tp=LW_EQB/v2;double rho=-(*tp)*fmax(1.0,fabs(u));*mx=SMALL_STEP_TOL/rho;}
static double gap(double z){if(z<0.4){double h=1.0/6706022400.0;h=fma(h,z,1.0/518918400.0);h=fma(h,z,1.0/43545600.0);h=fma(h,z,1.0/3991680.0);h=fma(h,z,1.0/403200.0);h=fma(h,z,1.0/45360.0);h=fma(h,z,1.0/5760.0);h=fma(h,z,1.0/840.0);h=fma(h,z,1.0/144.0);h=fma(h,z,1.0/30.0);h=fma(h,z,1.0/8.0);h=fma(h,z,1.0/3.0);h=fma(h,z,0.5);return z*z*h;}return fma(z-1,expm1(z),z);}
static double gap_at(double g0,double a){double e,em1;if(a>-0.5){em1=expm1(a);e=1+em1;}else{e=exp(a);em1=e-1;}return fma(g0,e,-em1);}
static double gap_switch(double g0){return g0>=GAP_LAMBERT_MIN?0.0:LOG_HALF-log1p(-g0);}
static double z_p6(double g){double p=sqrt(2*g),h=fma(-221.0/8505.0,p,769.0/17280.0);h=fma(h,p,-43.0/540.0);h=fma(h,p,11.0/72.0);h=fma(h,p,-1.0/3.0);h=fma(h,p,1);return p*h;}
static double z_p12(double g){double p=sqrt(2*g),h=fma(-1118511313.0/709296588000.0,p,169709463197.0/69528040243200.0);h=fma(h,p,-5776369.0/1515591000.0);h=fma(h,p,226287557.0/37623398400.0);h=fma(h,p,-1963.0/204120.0);h=fma(h,p,680863.0/43545600.0);h=fma(h,p,-221.0/8505.0);h=fma(h,p,769.0/17280.0);h=fma(h,p,-43.0/540.0);h=fma(h,p,11.0/72.0);h=fma(h,p,-1.0/3.0);h=fma(h,p,1);return p*h;}
static double z_halley(double z,double g){double res=gap(z)-g,sc=res*exp(-z);return z-2*sc*z/fma(2*z,z,-sc*(z+1));}
static double z_from_gap(double g){if(g<0)return NAN;if(g==0)return 0;if(g>=1)return 1;if(g<GAP_DIRECT_MAX)return z_p12(g);if(g<GAP_ONE_HALLEY_MAX)return z_halley(z_p12(g),g);if(g<GAP_LAMBERT_MIN){double z=z_halley(z_p6(g),g);return z_halley(z,g);}return 1+lambertw0_c((g-1)*INV_E);}
static double expm1mx(double th){if(fabs(th)<0.25){double h=1.0/87178291200.0;h=fma(h,th,-1.0/6227020800.0);h=fma(h,th,1.0/479001600.0);h=fma(h,th,-1.0/39916800.0);h=fma(h,th,1.0/3628800.0);h=fma(h,th,-1.0/362880.0);h=fma(h,th,1.0/40320.0);h=fma(h,th,-1.0/5040.0);h=fma(h,th,1.0/720.0);h=fma(h,th,-1.0/120.0);h=fma(h,th,1.0/24.0);h=fma(h,th,-1.0/6.0);h=fma(h,th,0.5);return th*th*h;}return expm1(-th)+th;}
static void finalize(double v,double z,double b,double th0,int hasq,double qd,double*th,double*m){int refine=th0<TIME_REFINE_MAX&&th0<TIME_REFINE_REL*fmax(v,fmax(z,b));int use=!refine&&hasq&&(v>1||th0>=TIME_REFINE_MAX);if(use){double qm1=qd-1;*th=th0;*m=qm1/(v*z);return;}double q,qm1;if(th0<0.5){qm1=expm1(-th0);q=1+qm1;}else{q=exp(-th0);qm1=q-1;}*th=th0;if(refine){double rem=expm1mx(th0),res=fma(1-v,rem,fma(v,th0,-b)),zt=fma(v,q,-qm1),second=(1-v)*q,delta=2*res*zt/fma(2*zt,zt,-res*second);*th=th0-delta;double ed=fabs(delta)<1e-4?delta*fma(delta,fma(delta,1.0/6.0,0.5),1.0):expm1(delta);qm1=fma(q,ed,qm1);}*m=qm1/(v*z);}
static void final_state(ame_motor_ws*w,double a,double*z,double*r,int*hasr){if(w->v0<1&&a>w->gap_lambert_a){*z=z_from_gap(gap_at(w->g0,a));*hasr=0;*r=0;return;}*r=lambertw0_c(w->u0*exp(w->u0+a));*z=1+*r;*hasr=1;}

ame_segment_status ame_motor_init(ame_segment*s,int stable){if(!ame_finite4(s->L,s->sigma,s->w0,s->k0)||s->w0<0)return AME_SEGMENT_INVALID_ARGUMENT;ame_motor_ws*w=&s->u.motor;memset(w,0,sizeof(*w));double y=sqrt(s->w0);w->v0=LW_EQ*y;w->u0=fma(LW_EQ,y,-1);if(!stable){small_params(w->v0,w->u0,&w->tau_per_ds,&w->small_ds_max);if(w->v0<1){w->g0=gap(w->v0);w->gap_lambert_a=gap_switch(w->g0);}}return AME_SEGMENT_OK;}
static void stable_state(ame_segment*s,double ds,double*z,double*q,double*qm1){ame_motor_ws*w=&s->u.motor;double a=LW_EQB*ds,e,em1;if(a>-0.5){em1=expm1(a);e=1+em1;}else{e=exp(a);em1=e-1;}double h=qseries(w->u0,e),ue=w->u0*e;*q=fma(ue*em1,h,e);*qm1=em1*fma(ue,h,1);*z=fma(w->u0,*q,1);}

ame_segment_status ame_motor_w(ame_segment*s,double ds,double*out){ame_motor_ws*w=&s->u.motor;if(ds==0){*out=s->w0;return AME_SEGMENT_OK;}double z,q,qm1;if(s->impl==AME_SEGMENT_IMPL_MOTOR_STABLE){stable_state(s,ds,&z,&q,&qm1);*out=W_EQ*z*z;return AME_SEGMENT_OK;}if(w->v0!=0&&fabs(ds)<=w->small_ds_max){double th,m;small_step(w->v0,w->tau_per_ds*ds,&q,&z,&th,&m);*out=W_EQ*z*z;return AME_SEGMENT_OK;}if(w->v0==1){*out=W_EQ;return AME_SEGMENT_OK;}double r;int hr;final_state(w,LW_EQB*ds,&z,&r,&hr);if(!isfinite(z))return AME_SEGMENT_DOMAIN;*out=W_EQ*z*z;return AME_SEGMENT_OK;}
ame_segment_status ame_motor_time(ame_segment*s,double ds,double*out){ame_motor_ws*w=&s->u.motor;if(ds==0){*out=0;return AME_SEGMENT_OK;}if(s->impl==AME_SEGMENT_IMPL_MOTOR_STABLE){double z,q,qm1;stable_state(s,ds,&z,&q,&qm1);double a=LW_EQB*ds,th=fma(w->u0,qm1,-a);*out=th*B_INV;return AME_SEGMENT_OK;}if(w->v0!=0&&fabs(ds)<=w->small_ds_max){double q,z,th,m;small_step(w->v0,w->tau_per_ds*ds,&q,&z,&th,&m);*out=th*B_INV;return AME_SEGMENT_OK;}double a=LW_EQB*ds;if(w->v0==1){*out=ds*LW_EQ;return AME_SEGMENT_OK;}double z,r;int hr;final_state(w,a,&z,&r,&hr);if(!isfinite(z))return AME_SEGMENT_DOMAIN;double th0=fma(-1.0,a,z-w->v0);if(w->v0==0){*out=th0*B_INV;return AME_SEGMENT_OK;}double qd=hr?r/w->u0:0,th,m;finalize(w->v0,z,-a,th0,hr,qd,&th,&m);*out=th*B_INV;return AME_SEGMENT_OK;}
ame_segment_status ame_motor_wjac(ame_segment*s,double ds,ame_segment_state_jac*out){if(!s->grad)return AME_SEGMENT_UNSUPPORTED;ame_motor_ws*w=&s->u.motor;if(ds==0){out->w=s->w0;out->jac[0]=-2*AME_A_MAX*w->u0;out->jac[1]=0;out->jac[2]=1;out->jac[3]=0;out->jac[4]=s->sigma;out->jac[5]=ds;out->jac[6]=0;out->jac[7]=1;return AME_SEGMENT_OK;}double z,q,r,qm1;if(s->impl==AME_SEGMENT_IMPL_MOTOR_STABLE){stable_state(s,ds,&z,&q,&qm1);r=w->u0*q;}else if(w->v0!=0&&fabs(ds)<=w->small_ds_max){double th,m;small_step(w->v0,w->tau_per_ds*ds,&q,&z,&th,&m);r=z-1;}else if(w->v0==1){double a=LW_EQB*ds;q=exp(a);z=1;r=0;}else{double rd;int hr;double a=LW_EQB*ds;final_state(w,a,&z,&rd,&hr);if(!isfinite(z))return AME_SEGMENT_DOMAIN;if(hr){r=rd;q=r/w->u0;}else{r=z-1;double th=fma(-1.0,a,z-w->v0);q=exp(-th);}}out->w=W_EQ*z*z;out->jac[0]=-2*AME_A_MAX*r;out->jac[1]=0;out->jac[2]=q;out->jac[3]=0;out->jac[4]=s->sigma;out->jac[5]=ds;out->jac[6]=0;out->jac[7]=1;return AME_SEGMENT_OK;}
ame_segment_status ame_motor_tjac(ame_segment*s,double ds,ame_segment_time_jac*out){if(!s->grad)return AME_SEGMENT_UNSUPPORTED;ame_motor_ws*w=&s->u.motor;double v=w->v0;if(ds==0){out->time=0;out->jac[0]=v==0?INFINITY:LW_EQ/v;out->jac[1]=out->jac[2]=out->jac[3]=0;return AME_SEGMENT_OK;}double z,q,qm1,th,m;if(s->impl==AME_SEGMENT_IMPL_MOTOR_STABLE){stable_state(s,ds,&z,&q,&qm1);double a=LW_EQB*ds;th=fma(w->u0,qm1,-a);m=qm1/(v*z);}else if(v!=0&&fabs(ds)<=w->small_ds_max){small_step(v,w->tau_per_ds*ds,&q,&z,&th,&m);}else{double a=LW_EQB*ds,b=-a;if(v==1){out->time=ds*LW_EQ;out->jac[0]=LW_EQ;out->jac[1]=out->jac[3]=0;out->jac[2]=DT_DW0_SCALE*expm1(a);return AME_SEGMENT_OK;}double r;int hr;final_state(w,a,&z,&r,&hr);if(!isfinite(z))return AME_SEGMENT_DOMAIN;double th0=fma(-1.0,a,z-v);if(v==0){out->time=th0*B_INV;out->jac[0]=LW_EQ/z;out->jac[1]=out->jac[3]=0;out->jac[2]=-INFINITY;return AME_SEGMENT_OK;}finalize(v,z,b,th0,hr,hr?r/w->u0:0,&th,&m);}out->time=th*B_INV;out->jac[0]=LW_EQ/z;out->jac[1]=out->jac[3]=0;out->jac[2]=DT_DW0_SCALE*m;return AME_SEGMENT_OK;}
