#ifndef AME_SEGMENT_H
#define AME_SEGMENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
# define AME_SEGMENT_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
# define AME_SEGMENT_API __attribute__((visibility("default")))
#else
# define AME_SEGMENT_API
#endif

typedef enum {
    AME_SEGMENT_GRIP = 1,
    AME_SEGMENT_MOTOR = 2,
    AME_SEGMENT_BRAKE = 3
} ame_segment_mode;

typedef enum {
    AME_SEGMENT_IMPL_STRAIGHT = 1,
    AME_SEGMENT_IMPL_CIRCULAR_STABLE = 2,
    AME_SEGMENT_IMPL_CIRCULAR = 3,
    AME_SEGMENT_IMPL_GRIP_CFLOW = 4,
    AME_SEGMENT_IMPL_MOTOR_STABLE = 5,
    AME_SEGMENT_IMPL_MOTOR = 6,
    AME_SEGMENT_IMPL_BRAKE = 7
} ame_segment_impl;

typedef enum {
    AME_SEGMENT_OK = 0,
    AME_SEGMENT_INVALID_ARGUMENT = 1,
    AME_SEGMENT_OUTSIDE_PREFIX = 2,
    AME_SEGMENT_DOMAIN = 3,
    AME_SEGMENT_CONDITIONING = 4,
    AME_SEGMENT_NUMERICAL_FAILURE = 5,
    AME_SEGMENT_UNSUPPORTED = 6,
    AME_SEGMENT_ALLOCATION_FAILURE = 7
} ame_segment_status;

typedef enum {
    AME_TIME_W0_NEGATIVE_INFINITY = 0,
    AME_TIME_W0_RAISE = 1,
    AME_TIME_W0_RENORMALIZED = 2
} ame_time_w0_policy;

typedef struct ame_segment_options {
    int boundary_start;
    int reverse_eta;
    int has_authoritative_w1;
    double authoritative_w1;
    ame_time_w0_policy time_w0_policy;
} ame_segment_options;

typedef struct ame_segment_state_jac {
    double w;
    /* d(w,k)/d(ds,sigma,w0,k0), flattened row-major exactly as Python. */
    double jac[8];
} ame_segment_state_jac;

typedef struct ame_segment_time_jac {
    double time;
    double jac[4];
} ame_segment_time_jac;

typedef struct ame_segment_all_jac {
    double w;
    double state_jac[8];
    double time;
    double time_jac[4];
} ame_segment_all_jac;

typedef struct ame_segment_domain_probe {
    int pre_event;
    double event_position;
    int cflow_status;
} ame_segment_domain_probe;

typedef struct ame_segment_cache_stats {
    size_t stations;
    uint64_t cache_hits;
    uint64_t cflow_calls;
    uint64_t local_steps;
} ame_segment_cache_stats;

/* Research/native crossing view.  This keeps ordered crossing probes inside
 * the segment object and preserves structured Cflow domain metadata without a
 * second public probe. */
typedef struct ame_segment_crossing_state {
    double w;
    double k;
    double q;
    double g2;
    int outside_domain;
    double event_position;
    int cflow_status;
} ame_segment_crossing_state;

typedef struct ame_segment ame_segment;

AME_SEGMENT_API ame_segment_options ame_segment_default_options(void);
AME_SEGMENT_API ame_segment *ame_segment_compile(
    double L, double sigma, double w0, double k0,
    ame_segment_mode mode, int grad,
    const ame_segment_options *options,
    ame_segment_status *status
);
AME_SEGMENT_API void ame_segment_destroy(ame_segment *segment);

AME_SEGMENT_API ame_segment_impl ame_segment_implementation(const ame_segment *segment);
AME_SEGMENT_API ame_segment_mode ame_segment_mode_of(const ame_segment *segment);
AME_SEGMENT_API int ame_segment_is_differentiable(const ame_segment *segment);
AME_SEGMENT_API double ame_segment_length(const ame_segment *segment);
AME_SEGMENT_API double ame_segment_sigma(const ame_segment *segment);
AME_SEGMENT_API double ame_segment_w0(const ame_segment *segment);
AME_SEGMENT_API double ame_segment_k0(const ame_segment *segment);

AME_SEGMENT_API ame_segment_status ame_segment_w(ame_segment *segment, double ds, double *out_w);
AME_SEGMENT_API ame_segment_status ame_segment_time(ame_segment *segment, double ds, double *out_time);
AME_SEGMENT_API ame_segment_status ame_segment_w_and_jac(ame_segment *segment, double ds, ame_segment_state_jac *out);
AME_SEGMENT_API ame_segment_status ame_segment_time_and_jac(ame_segment *segment, double ds, ame_segment_time_jac *out);
AME_SEGMENT_API ame_segment_status ame_segment_state_time_and_jac(ame_segment *segment, double ds, ame_segment_all_jac *out);
AME_SEGMENT_API ame_segment_status ame_segment_domain_probe_at(ame_segment *segment, double ds, ame_segment_domain_probe *out);
AME_SEGMENT_API ame_segment_status ame_segment_renormalized_time_w0(ame_segment *segment, double ds, double *out);
AME_SEGMENT_API ame_segment_status ame_segment_cache_stats_get(const ame_segment *segment, ame_segment_cache_stats *out);
AME_SEGMENT_API ame_segment_status ame_segment_crossing_state_at(ame_segment *segment, double ds, ame_segment_crossing_state *out);
AME_SEGMENT_API const char *ame_segment_status_name(ame_segment_status status);

#ifdef __cplusplus
}
#endif
#endif
