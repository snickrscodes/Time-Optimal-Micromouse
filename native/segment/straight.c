#include "segment_internal.h"

static double rate2(const ame_segment *s) {
    return s->impl == AME_SEGMENT_IMPL_BRAKE ? 2.0 * AME_A_BRAKE : 2.0 * AME_MU_G;
}

static ame_segment_status time_terms(double root0,double w1,double ds,double *t,double *ts,double *tw0) {
    if (w1 < 0.0) return AME_SEGMENT_DOMAIN;
    double root1=sqrt(w1);
    if (ds==0.0) {
        *t=0.0; *ts=root1==0.0?INFINITY:1.0/root1; *tw0=0.0; return AME_SEGMENT_OK;
    }
    double sum=root0+root1;
    if (sum==0.0) { *t=INFINITY; *ts=INFINITY; *tw0=-INFINITY; return AME_SEGMENT_OK; }
    double d=ds/sum;
    *t=2.0*d;
    *ts=root1==0.0?INFINITY:1.0/root1;
    *tw0=(root0==0.0||root1==0.0)?-INFINITY:-d/(root0*root1);
    return AME_SEGMENT_OK;
}

ame_segment_status ame_straight_init(ame_segment *s) {
    if (!isfinite(s->L)||!isfinite(s->w0)||s->L<0.0||s->w0<0.0) return AME_SEGMENT_INVALID_ARGUMENT;
    double wend=fma(rate2(s),s->L,s->w0);
    if (wend<0.0) return AME_SEGMENT_DOMAIN;
    s->u.sqrt_w0=sqrt(s->w0);
    return AME_SEGMENT_OK;
}

ame_segment_status ame_straight_w(ame_segment*s,double ds,double*out){*out=fma(rate2(s),ds,s->w0);return *out<0.0?AME_SEGMENT_DOMAIN:AME_SEGMENT_OK;}
ame_segment_status ame_straight_time(ame_segment*s,double ds,double*out){double w=fma(rate2(s),ds,s->w0),ts,tw;return time_terms(s->u.sqrt_w0,w,ds,out,&ts,&tw);}
ame_segment_status ame_straight_wjac(ame_segment*s,double ds,ame_segment_state_jac*out){double r=rate2(s);out->w=fma(r,ds,s->w0);if(out->w<0.0)return AME_SEGMENT_DOMAIN;out->jac[0]=r;out->jac[1]=0;out->jac[2]=1;out->jac[3]=0;out->jac[4]=s->impl==AME_SEGMENT_IMPL_BRAKE?s->sigma:0.0;out->jac[5]=ds;out->jac[6]=0;out->jac[7]=1;return AME_SEGMENT_OK;}
ame_segment_status ame_straight_tjac(ame_segment*s,double ds,ame_segment_time_jac*out){double w=fma(rate2(s),ds,s->w0),ts=0.0,tw=0.0;ame_segment_status st=time_terms(s->u.sqrt_w0,w,ds,&out->time,&ts,&tw);if(st!=AME_SEGMENT_OK)return st;out->jac[0]=ts;out->jac[1]=0;out->jac[2]=tw;out->jac[3]=0;return AME_SEGMENT_OK;}
ame_segment_status ame_straight_all(ame_segment*s,double ds,ame_segment_all_jac*out){ame_segment_state_jac w;ame_segment_time_jac t;ame_segment_status st=ame_straight_wjac(s,ds,&w);if(st)return st;st=ame_straight_tjac(s,ds,&t);if(st)return st;out->w=w.w;memcpy(out->state_jac,w.jac,sizeof(w.jac));out->time=t.time;memcpy(out->time_jac,t.jac,sizeof(t.jac));return AME_SEGMENT_OK;}
