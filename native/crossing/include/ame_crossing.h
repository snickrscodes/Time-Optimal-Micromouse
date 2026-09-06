#ifndef AME_CROSSING_H
#define AME_CROSSING_H

#include <stdint.h>
#include "../../segment/include/ame_segment.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
# define AME_CROSSING_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
# define AME_CROSSING_API __attribute__((visibility("default")))
#else
# define AME_CROSSING_API
#endif

typedef enum {
    AME_CROSSING_MOTOR_GRIP = 1,
    AME_CROSSING_GRIP_MOTOR = 2,
    AME_CROSSING_GRIP_BRAKE = 3,
    AME_CROSSING_BRAKE_GRIP = 4
} ame_crossing_kind;

typedef enum {
    AME_CROSSING_OK = 0,
    AME_CROSSING_INVALID_ARGUMENT = 1,
    AME_CROSSING_SEGMENT_ERROR = 2,
    AME_CROSSING_NUMERICAL_FAILURE = 3,
    AME_CROSSING_UNSUPPORTED = 4
} ame_crossing_status;

typedef struct ame_crossing_options {
    int n_scan;
    double domain_margin;
    double domain_stop_margin;
    double physical_domain_margin;
    double domain_safe_floor;
    double x_abs_tol;
    double x_rel_tol;
    double f_tol;
    int max_iter;
    int allow_initial_boundary;
    int has_initial_spatial_tol;
    double initial_spatial_tol;
} ame_crossing_options;

typedef struct ame_crossing_bracket {
    int present;
    double lo;
    double hi;
    double f_lo;
    double f_hi;
} ame_crossing_bracket;

typedef struct ame_crossing_result {
    int has_event;
    double event;
    int has_domain_edge;
    double domain_edge;
    ame_crossing_bracket event_bracket;
    ame_crossing_bracket domain_bracket;
    int initial_switch;
    int has_domain_safe;
    double domain_safe;
    int has_domain_event;
    double domain_event;
    int domain_switch_excluded;
    uint64_t state_evaluations;
} ame_crossing_result;

typedef struct ame_motor_switch_geometry_native {
    double M, A, B;
    double v_min, v_eq, v_R, R_max;
    double v_H, H_max;
    int has_v_H;
    int orientation_certified;
    int rising_barrier_certified;
} ame_motor_switch_geometry_native;

AME_CROSSING_API ame_crossing_options ame_crossing_default_options(void);
AME_CROSSING_API ame_crossing_status ame_crossing_scan(
    ame_segment *segment,
    double L,
    ame_crossing_kind kind,
    int earliest_safe,
    const ame_crossing_options *options,
    ame_crossing_result *out
);
AME_CROSSING_API ame_crossing_status ame_crossing_motor_grip(
    ame_segment *segment, double L, const ame_crossing_options *options, ame_crossing_result *out);
AME_CROSSING_API ame_crossing_status ame_crossing_grip_motor(
    ame_segment *segment, double L, int earliest_safe, const ame_crossing_options *options, ame_crossing_result *out);
AME_CROSSING_API ame_crossing_status ame_crossing_grip_brake(
    ame_segment *segment, double L, int earliest_safe, const ame_crossing_options *options, ame_crossing_result *out);
AME_CROSSING_API ame_crossing_status ame_crossing_brake_grip(
    ame_segment *segment, double L, const ame_crossing_options *options, ame_crossing_result *out);
AME_CROSSING_API ame_crossing_status ame_crossing_motor_geometry(ame_motor_switch_geometry_native *out);
AME_CROSSING_API const char *ame_crossing_status_name(ame_crossing_status status);

#ifdef __cplusplus
}
#endif
#endif
