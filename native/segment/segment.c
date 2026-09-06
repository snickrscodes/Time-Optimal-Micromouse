#include "segment_internal.h"

ame_segment_options ame_segment_default_options(void){ame_segment_options o={0};o.time_w0_policy=AME_TIME_W0_NEGATIVE_INFINITY;return o;}
static int opts_default(const ame_segment_options*o){if(!o)return 1;return !o->boundary_start&&!o->reverse_eta&&!o->has_authoritative_w1&&o->time_w0_policy==AME_TIME_W0_NEGATIVE_INFINITY;}
static int circular_stable(double L,double w0,double k0){if(k0==0)return 1;double phase=fabs(k0)*fma(2*AME_MU_G,L,w0)*AME_MU_G_INV;return phase<=AME_CIRCULAR_STABLE_PHASE_MAX;}
static int motor_stable(double w0){if(w0<0)return 0;double u=fma(1.0/AME_V_MAX,sqrt(w0),-1);return fabs(u)<=AME_MOTOR_U0_SERIES_TOL;}
ame_segment *ame_segment_compile(double L,double sigma,double w0,double k0,ame_segment_mode mode,int grad,const ame_segment_options*options,ame_segment_status*status){ame_segment_status st=AME_SEGMENT_OK;if(status)*status=AME_SEGMENT_OK;if(grad!=0&&grad!=1){if(status)*status=AME_SEGMENT_INVALID_ARGUMENT;return NULL;}ame_segment*s=calloc(1,sizeof(*s));if(!s){if(status)*status=AME_SEGMENT_ALLOCATION_FAILURE;return NULL;}s->L=L;s->sigma=sigma;s->w0=w0;s->k0=k0;s->mode=mode;s->grad=grad;
    if(mode==AME_SEGMENT_GRIP){if(sigma==0.0){if(!opts_default(options)){st=AME_SEGMENT_INVALID_ARGUMENT;goto fail;}if(k0==0.0){s->impl=AME_SEGMENT_IMPL_STRAIGHT;st=ame_straight_init(s);}else{int stable=circular_stable(L,w0,k0);s->impl=stable?AME_SEGMENT_IMPL_CIRCULAR_STABLE:AME_SEGMENT_IMPL_CIRCULAR;st=ame_circular_init(s,stable);}}else{s->impl=AME_SEGMENT_IMPL_GRIP_CFLOW;st=ame_grip_init(s,options);}}
    else if(mode==AME_SEGMENT_MOTOR){if(!opts_default(options)){st=AME_SEGMENT_INVALID_ARGUMENT;goto fail;}int stable=motor_stable(w0);s->impl=stable?AME_SEGMENT_IMPL_MOTOR_STABLE:AME_SEGMENT_IMPL_MOTOR;st=ame_motor_init(s,stable);}
    else if(mode==AME_SEGMENT_BRAKE){if(!opts_default(options)){st=AME_SEGMENT_INVALID_ARGUMENT;goto fail;}s->impl=AME_SEGMENT_IMPL_BRAKE;st=ame_straight_init(s);}else st=AME_SEGMENT_INVALID_ARGUMENT;
    if (st != AME_SEGMENT_OK) goto fail;
    return s;
fail:
    if (s->impl == AME_SEGMENT_IMPL_GRIP_CFLOW) ame_grip_destroy(s);
    free(s);
    if (status) *status = st;
    return NULL;
}
void ame_segment_destroy(ame_segment*s){if(!s)return;if(s->impl==AME_SEGMENT_IMPL_GRIP_CFLOW)ame_grip_destroy(s);free(s);}
ame_segment_impl ame_segment_implementation(const ame_segment*s){return s?s->impl:0;}
ame_segment_mode ame_segment_mode_of(const ame_segment*s){return s?s->mode:0;}int ame_segment_is_differentiable(const ame_segment*s){return s?s->grad:0;}double ame_segment_length(const ame_segment*s){return s?s->L:NAN;}double ame_segment_sigma(const ame_segment*s){return s?s->sigma:NAN;}double ame_segment_w0(const ame_segment*s){return s?s->w0:NAN;}double ame_segment_k0(const ame_segment*s){return s?s->k0:NAN;}
#define DISPATCH(name,s,...) do{if(!(s))return AME_SEGMENT_INVALID_ARGUMENT;switch((s)->impl){case AME_SEGMENT_IMPL_STRAIGHT:case AME_SEGMENT_IMPL_BRAKE:return ame_straight_##name((s),__VA_ARGS__);case AME_SEGMENT_IMPL_CIRCULAR_STABLE:case AME_SEGMENT_IMPL_CIRCULAR:return ame_circular_##name((s),__VA_ARGS__);case AME_SEGMENT_IMPL_MOTOR_STABLE:case AME_SEGMENT_IMPL_MOTOR:return ame_motor_##name((s),__VA_ARGS__);case AME_SEGMENT_IMPL_GRIP_CFLOW:return ame_grip_##name((s),__VA_ARGS__);default:return AME_SEGMENT_INVALID_ARGUMENT;}}while(0)
ame_segment_status ame_segment_w(ame_segment*s,double ds,double*out){if(!out)return AME_SEGMENT_INVALID_ARGUMENT;DISPATCH(w,s,ds,out);}
ame_segment_status ame_segment_time(ame_segment*s,double ds,double*out){if(!out)return AME_SEGMENT_INVALID_ARGUMENT;DISPATCH(time,s,ds,out);}
ame_segment_status ame_segment_w_and_jac(ame_segment*s,double ds,ame_segment_state_jac*out){if(!out)return AME_SEGMENT_INVALID_ARGUMENT;DISPATCH(wjac,s,ds,out);}
ame_segment_status ame_segment_time_and_jac(ame_segment*s,double ds,ame_segment_time_jac*out){if(!out)return AME_SEGMENT_INVALID_ARGUMENT;DISPATCH(tjac,s,ds,out);}
ame_segment_status ame_segment_state_time_and_jac(ame_segment*s,double ds,ame_segment_all_jac*out){if(!out||!s)return AME_SEGMENT_INVALID_ARGUMENT;switch(s->impl){case AME_SEGMENT_IMPL_STRAIGHT:case AME_SEGMENT_IMPL_BRAKE:return ame_straight_all(s,ds,out);case AME_SEGMENT_IMPL_CIRCULAR_STABLE:case AME_SEGMENT_IMPL_CIRCULAR:return ame_circular_all(s,ds,out);case AME_SEGMENT_IMPL_GRIP_CFLOW:return ame_grip_all(s,ds,out);default:return AME_SEGMENT_UNSUPPORTED;}}
ame_segment_status ame_segment_domain_probe_at(ame_segment*s,double ds,ame_segment_domain_probe*out){if(!out||!s)return AME_SEGMENT_INVALID_ARGUMENT;if(s->impl!=AME_SEGMENT_IMPL_GRIP_CFLOW)return AME_SEGMENT_UNSUPPORTED;return ame_grip_probe(s,ds,out);}
ame_segment_status ame_segment_renormalized_time_w0(ame_segment*s,double ds,double*out){if(!out||!s)return AME_SEGMENT_INVALID_ARGUMENT;if(s->impl!=AME_SEGMENT_IMPL_GRIP_CFLOW)return AME_SEGMENT_UNSUPPORTED;return ame_grip_renorm(s,ds,out);}
ame_segment_status ame_segment_crossing_state_at(ame_segment*s,double ds,ame_segment_crossing_state*out){
    if(!s||!out)return AME_SEGMENT_INVALID_ARGUMENT;
    if(s->impl==AME_SEGMENT_IMPL_GRIP_CFLOW)return ame_grip_cross_state(s,ds,out);
    double w=NAN; ame_segment_status st=ame_segment_w(s,ds,&w); if(st)return st;
    double k=fma(s->sigma,ds,s->k0),q=w*k,g2=fma(-q,q,AME_MU_G*AME_MU_G);
    out->w=w;out->k=k;out->q=q;out->g2=g2;out->outside_domain=0;out->event_position=NAN;out->cflow_status=CFLOW_OK;
    return AME_SEGMENT_OK;
}
ame_segment_status ame_segment_cache_stats_get(const ame_segment*s,ame_segment_cache_stats*out){if(!s||!out)return AME_SEGMENT_INVALID_ARGUMENT;if(s->impl!=AME_SEGMENT_IMPL_GRIP_CFLOW)return AME_SEGMENT_UNSUPPORTED;const ame_grip_ws*g=&s->u.grip;out->stations=g->n_anchors;out->cache_hits=g->cache_hits;out->cflow_calls=g->cflow_calls;out->local_steps=g->local_steps;return AME_SEGMENT_OK;}
const char *ame_segment_status_name(ame_segment_status s){switch(s){case AME_SEGMENT_OK:return"ok";case AME_SEGMENT_INVALID_ARGUMENT:return"invalid_argument";case AME_SEGMENT_OUTSIDE_PREFIX:return"outside_prefix";case AME_SEGMENT_DOMAIN:return"domain";case AME_SEGMENT_CONDITIONING:return"conditioning";case AME_SEGMENT_NUMERICAL_FAILURE:return"numerical_failure";case AME_SEGMENT_UNSUPPORTED:return"unsupported";case AME_SEGMENT_ALLOCATION_FAILURE:return"allocation_failure";default:return"unknown";}}
