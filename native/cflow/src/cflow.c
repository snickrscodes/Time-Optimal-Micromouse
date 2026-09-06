#include "cflow_internal.h"
#include <string.h>

static double end_speed(double x,double q){double z=x*q,r=1-z*z;return r<=0?0.0:2.0*sqrt(r);}

int cflow_choose_local(double x,double q,double b,double rem,int *kind,int *panel,cflow_highz_face *face,double *hs){
    double best=0;int bk=-1,bp=-1;double h;cflow_highz_face bf=CFLOW_HZ_FACE_NONE;
    h=cflow_core_cap(x,q,b,rem);if(h>best){best=h;bk=0;bf=CFLOW_HZ_FACE_NONE;}
    int p;cflow_highz_face hf=CFLOW_HZ_FACE_NONE;
    if(cflow_highz_candidate(x,q,b,rem,&p,&hf,&h)&&h>best){best=h;bk=1;bp=p;bf=hf;}
    if(cflow_terminal_candidate(x,q,b,rem,&h)&&h>best){best=h;bk=2;bp=-1;bf=CFLOW_HZ_FACE_NONE;}
    if(cflow_farfield_candidate(x,q,b,rem,&h) && h>best*(1.000000000001)){best=h;bk=3;bp=-1;bf=CFLOW_HZ_FACE_NONE;}
    if(best<=0) return 0;
    *kind=bk;*panel=bp;*face=bf;*hs=best;return 1;
}

void cflow_eval_local(int kind,int panel,cflow_highz_face face,double x,double q,double b,double h,int jac,cflow_local_jac *o){
    if(kind==0)cflow_core_local(x,q,b,h,jac,o);else if(kind==1)cflow_highz_local(x,q,b,h,panel,face,jac,o);else if(kind==2)cflow_terminal_local(x,q,b,h,jac,o);else cflow_farfield_local(x,q,b,h,jac,o);
}

static void advance_forward(double x0,double q0,double b,double h,int jac,cflow_eval_jac_result *out){
    memset(out,0,sizeof(*out));out->x=x0;out->event_time=NAN;out->status=CFLOW_OK;
    if(!isfinite(x0)||!isfinite(q0)||!isfinite(b)||!isfinite(h)||h<0){out->status=CFLOW_INVALID_ARGUMENT;return;}
    if(!cflow_real_domain(x0,q0)){out->status=CFLOW_OUTSIDE_REAL_DOMAIN;return;}
    if(h==0){out->dx_dx0=1;out->dx_dq0=out->dx_db=0;out->dx_dh=end_speed(x0,q0);return;}
    if(b==0.0){
        double te=cflow_exact_b0_event(x0,q0);if(te<=h){double tol=64*DBL_EPSILON*fmax(1.0,fabs(te));cflow_local_jac e;cflow_exact_b0(x0,q0,te,jac,&e);out->x=e.x;out->event_time=te;out->status=fabs(h-te)<=tol?CFLOW_EVENT:CFLOW_BEYOND_EVENT;out->steps=1;out->dx_dx0=out->dx_dq0=out->dx_db=out->dx_dh=NAN;return;}
        cflow_local_jac e;cflow_exact_b0(x0,q0,h,jac,&e);out->x=e.x;out->steps=1;out->dx_dx0=jac?e.dx:NAN;out->dx_dq0=jac?e.dq:NAN;out->dx_db=jac?e.db:NAN;out->dx_dh=end_speed(e.x,q0);return;
    }
    double x=x0,q=q0,t=0,rem=h,Sx=1,Sq=0,Sb=0;
    int no_terminal_event=cflow_contracting_no_terminal_event(x0,q0,b,h);
    for(size_t step=0;step<CFLOW_MAX_STEPS;step++){
        if(rem==0.0){out->x=x;out->steps=step;out->status=CFLOW_OK;out->dx_dx0=jac?Sx:NAN;out->dx_dq0=jac?Sq:NAN;out->dx_db=jac?Sb:NAN;out->dx_dh=end_speed(x,q);return;}
        if(fabs(q*x)>=1.0){out->x=x;out->event_time=t;out->steps=step;out->status=CFLOW_EVENT;out->dx_dx0=out->dx_dq0=out->dx_db=out->dx_dh=NAN;return;}
        if(cflow_sep_unresolved(x,q,b,rem)){out->x=x;out->steps=step;out->status=CFLOW_CONDITIONING_LIMIT;return;}
        /* First-event query becomes reliable in the terminal chart. */
        cflow_event_local ev;double te,dtx,dtq,dtb;
        double event_horizon=rem*(1.0+128.0*DBL_EPSILON);
        if(!no_terminal_event && cflow_terminal_event_local_with_horizon(x,q,b,event_horizon,&ev,&te,&dtx,&dtq,&dtb) && te<=event_horizon){
            double qe=q+b*te,zsign=(q*x>=0)?1.0:-1.0,xe=zsign/qe;double total=t+te,tol=128*DBL_EPSILON*fmax(1.0,fabs(total));
            out->x=xe;out->event_time=total;out->steps=step+1;out->status=fabs(h-total)<=tol?CFLOW_EVENT:CFLOW_BEYOND_EVENT;out->dx_dx0=out->dx_dq0=out->dx_db=out->dx_dh=NAN;return;
        }
        int kind,panel;cflow_highz_face face;double hs;if(!cflow_choose_local(x,q,b,rem,&kind,&panel,&face,&hs)||hs<=0){out->x=x;out->steps=step;out->status=CFLOW_CONDITIONING_LIMIT;return;}
        cflow_local_jac e;cflow_eval_local(kind,panel,face,x,q,b,hs,jac,&e);if(!isfinite(e.x)){out->status=CFLOW_NUMERICAL_FAILURE;out->x=x;out->steps=step;return;}
        if(jac){double nSx=e.dx*Sx;double nSq=e.dx*Sq+e.dq;double nSb=e.dx*Sb+e.dq*t+e.db;Sx=nSx;Sq=nSq;Sb=nSb;}
        x=e.x;q=fma(b,hs,q);t+=hs;double old=rem;rem=h-t;if(rem<0&&fabs(rem)<=64*DBL_EPSILON*fmax(h,old))rem=0;if(hs==old)rem=0;
    }
    out->x=x;out->steps=CFLOW_MAX_STEPS;out->status=CFLOW_CONDITIONING_LIMIT;
}


void cflow_first_event(double x0,double q0,double b,double max_time,cflow_event_result *out){
    memset(out,0,sizeof(*out));out->time=NAN;out->x=NAN;out->dt_dx0=out->dt_dq0=out->dt_db=NAN;out->conditioning_log_amp=0;out->status=CFLOW_NO_EVENT_WITHIN_HORIZON;
    if(!isfinite(x0)||!isfinite(q0)||!isfinite(b)||!(max_time>0)||!cflow_real_domain(x0,q0)){out->status=CFLOW_INVALID_ARGUMENT;return;}
    if(fabs(q0*x0)>=1.0){out->time=0;out->x=x0;out->dt_dx0=out->dt_dq0=out->dt_db=0;out->status=CFLOW_EVENT;return;}
    if(b==0.0){double te=cflow_exact_b0_event(x0,q0);if(te>max_time||!isfinite(te))return;double z=q0*x0,c0=sqrt(1-z*z);out->time=te;out->x=copysign(1.0,q0)/q0;out->dt_dx0=-1.0/(2*c0);double th=asin(z),target=copysign(CFLOW_PI/2,q0),n=target-th;out->dt_dq0=-x0/(2*q0*c0)-n/(2*q0*q0);out->dt_db=NAN;out->status=CFLOW_CONDITIONING_LIMIT;out->steps=1;return;}
    double x=x0,q=q0,t=0,Sx=1,Sq=0,Sb=0;
    int no_terminal_event=cflow_contracting_no_terminal_event(x0,q0,b,max_time);
    for(size_t step=0;step<CFLOW_MAX_STEPS && t<max_time;step++){
        if(cflow_sep_unresolved(x,q,b,max_time-t)){out->status=CFLOW_CONDITIONING_LIMIT;out->steps=step;out->conditioning_log_amp=cflow_conditioning_logamp(x0,q0,b,max_time);return;}
        cflow_event_local ev;double te,ex,eq,eb;
        double event_horizon=max_time-t;
        if(!no_terminal_event && cflow_terminal_event_local_with_horizon(x,q,b,event_horizon,&ev,&te,&ex,&eq,&eb) && te<=event_horizon){
            double T=t+te,qe=q+b*te,sig=(q*x>=0)?1.0:-1.0;out->time=T;out->x=sig/qe;
            out->dt_dx0=ex*Sx;out->dt_dq0=ex*Sq+eq;out->dt_db=ex*Sb+eq*t+eb;out->steps=step+1;out->conditioning_log_amp=cflow_conditioning_logamp(x0,q0,b,T);out->status=ev.conditioning?CFLOW_CONDITIONING_LIMIT:CFLOW_EVENT;return;
        }
        double rem=max_time-t;int kind,panel;cflow_highz_face face;double hs;if(!cflow_choose_local(x,q,b,rem,&kind,&panel,&face,&hs)||hs<=0){out->status=CFLOW_CONDITIONING_LIMIT;out->steps=step;out->conditioning_log_amp=cflow_conditioning_logamp(x0,q0,b,t);return;}
        cflow_local_jac e;cflow_eval_local(kind,panel,face,x,q,b,hs,1,&e);if(!isfinite(e.x)){out->status=CFLOW_NUMERICAL_FAILURE;return;}
        double nSx=e.dx*Sx,nSq=e.dx*Sq+e.dq,nSb=e.dx*Sb+e.dq*t+e.db;Sx=nSx;Sq=nSq;Sb=nSb;x=e.x;q=fma(b,hs,q);t+=hs;
    }
    out->steps=CFLOW_MAX_STEPS; if(t>=max_time)out->status=CFLOW_NO_EVENT_WITHIN_HORIZON;else out->status=CFLOW_CONDITIONING_LIMIT;
}


void cflow_eval(double x0,double q0,double b,double h,cflow_eval_result *out){
    if(!out)return;
    cflow_eval_jac_result j;
    if(h>=0.0)advance_forward(x0,q0,b,h,0,&j);
    else{advance_forward(-x0,q0,-b,-h,0,&j);j.x=-j.x;if(isfinite(j.event_time))j.event_time=-j.event_time;}
    out->x=j.x;out->event_time=j.event_time;out->steps=j.steps;out->status=j.status;
}

void cflow_eval_jacobian(double x0,double q0,double b,double h,cflow_eval_jac_result *out){
    if(!out)return;
    if(h>=0.0){advance_forward(x0,q0,b,h,1,out);return;}
    cflow_eval_jac_result g;advance_forward(-x0,q0,-b,-h,1,&g);*out=g;out->x=-g.x;
    if(isfinite(g.event_time))out->event_time=-g.event_time;
    if(g.status==CFLOW_OK){out->dx_dx0=g.dx_dx0;out->dx_dq0=-g.dx_dq0;out->dx_db=g.dx_db;out->dx_dh=g.dx_dh;}
}
