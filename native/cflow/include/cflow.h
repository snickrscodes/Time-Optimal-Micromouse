#ifndef CFLOW_H
#define CFLOW_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define CFLOW_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define CFLOW_API __attribute__((visibility("default")))
#else
#  define CFLOW_API
#endif

typedef enum {
    CFLOW_OK = 0,
    CFLOW_EVENT = 1,
    CFLOW_BEYOND_EVENT = 2,
    CFLOW_OUTSIDE_REAL_DOMAIN = 3,
    CFLOW_CONDITIONING_LIMIT = 4,
    CFLOW_NUMERICAL_FAILURE = 5,
    CFLOW_NO_EVENT_WITHIN_HORIZON = 6,
    CFLOW_INVALID_ARGUMENT = 7,
    CFLOW_OUTSIDE_INTEGRAL_DOMAIN = 8
} cflow_status;

typedef struct { double x,event_time; size_t steps; cflow_status status; } cflow_eval_result;
typedef struct {
    double x,dx_dx0,dx_dq0,dx_db,dx_dh,event_time;
    size_t steps; cflow_status status;
} cflow_eval_jac_result;
typedef struct {
    double integral,dI_dx0,dI_dq0,dI_db,dI_dh,regularized_dI_dx0,event_time;
    size_t steps; cflow_status status;
} cflow_integral_jac_result;
typedef struct {
    double integral,event_time;
    size_t steps; cflow_status status;
} cflow_integral_value_result;
typedef struct {
    double x,dx_dx0,dx_dq0,dx_db,dx_dh;
    double integral,dI_dx0,dI_dq0,dI_db,dI_dh,regularized_dI_dx0,event_time;
    size_t steps; cflow_status status;
} cflow_all_result;
typedef struct {
    double time,x,dt_dx0,dt_dq0,dt_db,conditioning_log_amp;
    size_t steps; cflow_status status;
} cflow_event_result;

/* Application-facing normalized APIs. */
CFLOW_API void cflow_eval(double x0,double q0,double b,double h,cflow_eval_result *out);
CFLOW_API void cflow_eval_jacobian(double x0,double q0,double b,double h,cflow_eval_jac_result *out);
CFLOW_API void cflow_integral_jacobian(double x0,double q0,double b,double h,cflow_integral_jac_result *out);
CFLOW_API void cflow_integral_value(double x0,double q0,double b,double h,cflow_integral_value_result *out);
CFLOW_API void cflow_eval_all(double x0,double q0,double b,double h,cflow_all_result *out);
CFLOW_API void cflow_first_event(double x0,double q0,double b,double max_time,cflow_event_result *out);

/* One-sided inward continuation from the exact friction cap. */
CFLOW_API void cflow_boundary_eval(double x0,double q0,double b,double h,cflow_eval_result *out);
CFLOW_API void cflow_boundary_eval_jacobian(double x0,double q0,double b,double h,cflow_eval_jac_result *out);
CFLOW_API void cflow_boundary_integral_jacobian(double x0,double q0,double b,double h,cflow_integral_jac_result *out);
CFLOW_API void cflow_boundary_integral_value(double x0,double q0,double b,double h,cflow_integral_value_result *out);
CFLOW_API void cflow_boundary_eval_all(double x0,double q0,double b,double h,cflow_all_result *out);

#ifdef __cplusplus
}
#endif
#endif
