#ifndef AME_SEGMENT_INTERNAL_H
#define AME_SEGMENT_INTERNAL_H

#include "include/ame_segment.h"
#include "include/ame_segment_constants.h"
#include "../cflow/include/cflow.h"
#include "../reverse_eta/reverse_eta.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AME_MU_G AME_SEGMENT_MU_G
#define AME_A_BRAKE AME_SEGMENT_A_BRAKE
#define AME_A_MAX AME_SEGMENT_A_MAX
#define AME_V_MAX AME_SEGMENT_V_MAX
#define AME_B_EMF AME_SEGMENT_B_EMF
#define AME_MU_G_INV (1.0 / AME_MU_G)
#define AME_SQRT_MU_G (sqrt(AME_MU_G))
#define AME_INV_SQRT_MU_G (1.0 / AME_SQRT_MU_G)
#define AME_INV_MU_G_SQRT_MU_G (AME_MU_G_INV * AME_INV_SQRT_MU_G)

#define AME_CIRCULAR_STABLE_PHASE_MAX 3.0e-2
#define AME_MOTOR_U0_SERIES_TOL 2.0e-3

typedef struct ame_circular_ws {
    double k_abs, eps, sqrt_w0;
    double s0, c0, x0, domain_end;
    double small_ds_max, small_time_ds_max, small_jac_ds_max, small_sigma_ds_max;
    int has_local;
    int stable_order;
    double local_time[10];
    double local_tk[10];
    double local_ws[11];
    double local_ts[9];
    double local_ws_ext[15];
    double local_ts_ext[11];
    double J0, Q0, Ht0, logm0, logp0, time_scale, sigma_time_scale;
} ame_circular_ws;

typedef struct ame_motor_ws {
    double v0, u0, g0, gap_lambert_a, tau_per_ds, small_ds_max;
} ame_motor_ws;

typedef struct ame_grip_anchor {
    double s, x;
} ame_grip_anchor;

typedef struct ame_grip_ws {
    double x0;
    int boundary_start;
    ame_time_w0_policy time_w0_policy;
    int reverse_eta;
    int has_authoritative_w1;
    double authoritative_w1;
    int has_endpoint_value;
    double endpoint_value;
    int has_endpoint_all;
    ame_segment_all_jac endpoint_all;
    int has_eta_endpoint;
    double eta_endpoint_ds;
    ame_reverse_eta_result eta_endpoint;
    ame_grip_anchor inline_anchors[64];
    ame_grip_anchor *anchors;
    size_t n_anchors, cap_anchors;
    uint64_t cache_hits, cflow_calls, local_steps;
} ame_grip_ws;

struct ame_segment {
    double L, sigma, w0, k0;
    ame_segment_mode mode;
    ame_segment_impl impl;
    int grad;
    union {
        double sqrt_w0;
        ame_circular_ws circular;
        ame_motor_ws motor;
        ame_grip_ws grip;
    } u;
};

static inline int ame_finite4(double a,double b,double c,double d) {
    return isfinite(a)&&isfinite(b)&&isfinite(c)&&isfinite(d);
}

ame_segment_status ame_straight_init(ame_segment *s);
ame_segment_status ame_circular_init(ame_segment *s, int stable);
ame_segment_status ame_motor_init(ame_segment *s, int stable);
ame_segment_status ame_grip_init(ame_segment *s, const ame_segment_options *o);
void ame_grip_destroy(ame_segment *s);

ame_segment_status ame_straight_w(ame_segment*,double,double*);
ame_segment_status ame_straight_time(ame_segment*,double,double*);
ame_segment_status ame_straight_wjac(ame_segment*,double,ame_segment_state_jac*);
ame_segment_status ame_straight_tjac(ame_segment*,double,ame_segment_time_jac*);
ame_segment_status ame_straight_all(ame_segment*,double,ame_segment_all_jac*);

ame_segment_status ame_circular_w(ame_segment*,double,double*);
ame_segment_status ame_circular_time(ame_segment*,double,double*);
ame_segment_status ame_circular_wjac(ame_segment*,double,ame_segment_state_jac*);
ame_segment_status ame_circular_tjac(ame_segment*,double,ame_segment_time_jac*);
ame_segment_status ame_circular_all(ame_segment*,double,ame_segment_all_jac*);

ame_segment_status ame_motor_w(ame_segment*,double,double*);
ame_segment_status ame_motor_time(ame_segment*,double,double*);
ame_segment_status ame_motor_wjac(ame_segment*,double,ame_segment_state_jac*);
ame_segment_status ame_motor_tjac(ame_segment*,double,ame_segment_time_jac*);

ame_segment_status ame_grip_w(ame_segment*,double,double*);
ame_segment_status ame_grip_time(ame_segment*,double,double*);
ame_segment_status ame_grip_wjac(ame_segment*,double,ame_segment_state_jac*);
ame_segment_status ame_grip_tjac(ame_segment*,double,ame_segment_time_jac*);
ame_segment_status ame_grip_all(ame_segment*,double,ame_segment_all_jac*);
ame_segment_status ame_grip_probe(ame_segment*,double,ame_segment_domain_probe*);
ame_segment_status ame_grip_cross_state(ame_segment*,double,ame_segment_crossing_state*);
ame_segment_status ame_grip_renorm(ame_segment*,double,double*);

#endif
