#include "cflow_internal.h"
#define FAR_EMAX 0.05
#define FAR_TMAX 1.05
#define FAR_SMAX 0.18

static double signed_s_for_h(double q,double b,double h){return h*(2.0*q+b*h);}
int cflow_farfield_candidate(double x,double q,double b,double remaining,double *hcap){
    if(remaining<=0.0||q==0.0||b==0.0)return 0;
    double z=q*x;if(fabs(z)>=sin(FAR_TMAX))return 0;
    double eps=b/(q*q);if(fabs(eps)>0.048)return 0;
    double qend=q+b*remaining;if(q*qend>0.0){double sv=signed_s_for_h(q,b,remaining);if(fabs(sv)<=FAR_SMAX&&fabs(asin(z)+sv)<=1.20){*hcap=remaining;return 1;}}
    double st=copysign(FAR_SMAX,q), rad=q*q+b*st;if(!(rad>0.0))return 0;double q1=copysign(sqrt(rad),q);double den=q+q1;if(den==0.0)return 0;double h=st/den;
    if(!(h>0.0)||q*(q+b*h)<=0.0) return 0;
    double th=asin(z); if(fabs(th+st)>1.20) return 0;
    h*=0.985; if(h>remaining)h=remaining; *hcap=h; return h>0.0;
}
void cflow_farfield_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o){
    double z=q*x,ct=sqrt(fmax(0.0,1.0-z*z)),th=asin(z),eps=b/(q*q),sv=signed_s_for_h(q,b,h),q1=q+b*h;
    CFLOW_ASSERT(fabs(eps)<=FAR_EMAX+64*DBL_EPSILON && fabs(th)<=FAR_TMAX+64*DBL_EPSILON && fabs(sv)<=FAR_SMAX+64*DBL_EPSILON);
    cflow_t3_eval e;cflow_cheb3(cflow_far_g,CFLOW_FAR_NE,CFLOW_FAR_NT,CFLOW_FAR_NS,eps/FAR_EMAX,th/FAR_TMAX,sv/FAR_SMAX,&e);
    double Ge=e.da/FAR_EMAX,Gt=e.db/FAR_TMAX,Gs=e.dc/FAR_SMAX;
    double T=th+sv+eps*e.k,sn=sin(T),co=cos(T);o->x=sn/q1;
    if(!want_jac){o->dx=o->dq=o->db=NAN;return;}
    double thx=q/ct,thq=x/ct;
    double eq=-2.0*eps/q, eb=1.0/(q*q), sq=2.0*h, sb=h*h;
    double Tx=(1.0+eps*Gt)*thx;
    double Tq=thq+sq+eq*e.k+eps*(Ge*eq+Gt*thq+Gs*sq);
    double Tb=sb+eb*e.k+eps*(Ge*eb+Gs*sb);
    o->dx=co*Tx/q1;
    o->dq=co*Tq/q1-sn/(q1*q1);
    o->db=co*Tb/q1-sn*h/(q1*q1);
}

double cflow_farfield_dx_defect(double x,double q,double b,double h){
    double z=q*x,ct=sqrt(fmax(0.0,1.0-z*z)),th=asin(z),eps=b/(q*q),sv=signed_s_for_h(q,b,h),q1=q+b*h;
    cflow_t3_eval e;cflow_cheb3(cflow_far_g,CFLOW_FAR_NE,CFLOW_FAR_NT,CFLOW_FAR_NS,eps/FAR_EMAX,th/FAR_TMAX,sv/FAR_SMAX,&e);
    double Gt=e.db/FAR_TMAX; double T=th+sv+eps*e.k;
    double aq=-b*h/q1; /* q/q1 - 1 */
    double cdiff=-2.0*sin(0.5*(T+th))*sin(0.5*(T-th));
    double ac=cdiff/ct; /* cos(T)/ct - 1 */
    double ag=eps*Gt;
    return aq+ac+ag+aq*ac+aq*ag+ac*ag+aq*ac*ag;
}
