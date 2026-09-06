#ifndef AME_DD_YAW_H
#define AME_DD_YAW_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
# define AME_DD_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
# define AME_DD_API __attribute__((visibility("default")))
#else
# define AME_DD_API
#endif

typedef struct {
    double beta, eta, q0, v_free;
    double h_floor, c_floor, speed_margin;
} ame_dd_params;
typedef struct {
    double a_max, b_emf, a_brake, mu_g;
} ame_dd_profile;
typedef struct { double value,dw,dk,dsigma; } ame_dd_candidate;
typedef struct {
    double lower, upper, margin, motor_upper, grip;
    double left_lower,left_upper,right_lower,right_upper;
} ame_dd_interval;
typedef struct {
    double w; double state_jac[8];
    double time; double time_jac[4];
} ame_dd_segment_all;
typedef struct {
    double w, margin, hard_cap_w;
    int upper_mode, lower_mode, status;
} ame_dd_mvc_point;

enum { AME_DD_FORWARD=1, AME_DD_BACKWARD=-1 };
enum { AME_DD_GRIP=1, AME_DD_MOTOR=2, AME_DD_BRAKE=3, AME_DD_SIDE_RIGHT=4, AME_DD_SIDE_LEFT=5 };
enum { AME_DD_OK=0, AME_DD_INVALID=1, AME_DD_DOMAIN=2, AME_DD_NUMERICAL=3 };

AME_DD_API int ame_dd_candidate_eval(const ame_dd_params*,const ame_dd_profile*,int mode,int kind,double w,double k,double sigma,ame_dd_candidate*);
AME_DD_API int ame_dd_interval_eval(const ame_dd_params*,const ame_dd_profile*,double w,double k,double sigma,ame_dd_interval*);
AME_DD_API int ame_dd_segment_eval(const ame_dd_params*,const ame_dd_profile*,int mode,int kind,double L,double sigma,double w0,double k0,double ds,ame_dd_segment_all*);
AME_DD_API int ame_dd_mvc_scan_bulk(const ame_dd_params*,const ame_dd_profile*,const double *kappa,const double *sigma,size_t count,int n_scan,ame_dd_mvc_point *out);
AME_DD_API const char *ame_dd_status_name(int);
#ifdef __cplusplus
}
#endif
#endif
