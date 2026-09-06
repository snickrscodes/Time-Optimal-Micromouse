#ifndef CFLOW_INTERNAL_H
#define CFLOW_INTERNAL_H
#include <math.h>
#include <float.h>
#include <stddef.h>
#ifdef CFLOW_DEBUG
#include <assert.h>
#define CFLOW_ASSERT(x) assert(x)
#else
#define CFLOW_ASSERT(x) ((void)0)
#endif
#include "cflow.h"
#include "core_coeffs.h"
#include "highz_coeffs.h"
#include "farfield_coeffs.h"

#define CFLOW_PI 3.141592653589793238462643383279502884
#define CFLOW_CORE_SAFETY 0.985
#define CFLOW_MAX_STEPS 20000u
#define CFLOW_Z_CORE_SWITCH 0.195
#define CFLOW_TERM_DELTA 0.20
#define CFLOW_EVENT_RK_RTOL 2e-13
#define CFLOW_EVENT_RK_ATOL 2e-15

/* Internal differential-test switch. Production builds leave FSAL enabled. */
#ifdef CFLOW_E0_DISABLE_FSAL
#define CFLOW_E0_FSAL_ENABLED 0
#else
#define CFLOW_E0_FSAL_ENABLED 1
#endif

typedef struct { double k,kp,kr,kc,ks; } cflow_core_eval;
typedef struct { double k, da, db, dc; } cflow_t3_eval;
typedef struct { double x, dx, dq, db; } cflow_local_jac;

typedef enum {
    CFLOW_HZ_FACE_NONE = 0,
    CFLOW_HZ_FACE_C_POS = 1,
    CFLOW_HZ_FACE_C_NEG = 2,
    CFLOW_HZ_FACE_S_POS = 3,
    CFLOW_HZ_FACE_S_NEG = 4
} cflow_highz_face;

int cflow_choose_local(double x,double q,double b,double rem,int *kind,int *panel,cflow_highz_face *face,double *hs);
void cflow_eval_local(int kind,int panel,cflow_highz_face face,double x,double q,double b,double h,int jac,cflow_local_jac *o);
double cflow_core_dx_defect(double x,double q,double b,double h);
double cflow_highz_dx_defect(double x,double q,double b,double h,int panel);
typedef struct { double lambda, dl_drho, dl_dkappa; int ok; int no_event; int conditioning; } cflow_event_local;

int cflow_real_domain(double x,double q);
double cflow_core_eval_compact_value(double p,double r,double c,double s);
double cflow_core_eval_partial_value(double p,double r,double c,double s);
void cflow_core_eval_compact(double p,double r,double c,double s,cflow_core_eval *o);
void cflow_core_eval_partial(double p,double r,double c,double s,cflow_core_eval *o);
void cflow_core_eval_unrolled(double p,double r,double c,double s,cflow_core_eval *o);
double cflow_core_eval_unrolled_value(double p,double r,double c,double s);
void cflow_core_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o);
int cflow_core_legal(double x,double q,double b,double h);
double cflow_core_cap(double x,double q,double b,double remaining);

int cflow_highz_candidate(double x,double q,double b,double remaining,int *panel,cflow_highz_face *face,double *hcap);
void cflow_highz_local(double x,double q,double b,double h,int panel,cflow_highz_face face,int want_jac,cflow_local_jac *o);

int cflow_farfield_candidate(double x,double q,double b,double remaining,double *hcap);
void cflow_farfield_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o);

int cflow_terminal_candidate(double x,double q,double b,double remaining,double *hcap);
void cflow_terminal_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o);
int cflow_event_lambda(double rho,double kappa,cflow_event_local *ev);
int cflow_terminal_event_local(double x,double q,double b,cflow_event_local *ev,double *te,double *dte_dx,double *dte_dq,double *dte_db);
int cflow_contracting_no_terminal_event(double x,double q,double b,double h);
int cflow_terminal_event_local_with_horizon(double x,double q,double b,double h,cflow_event_local *ev,double *te,double *dte_dx,double *dte_dq,double *dte_db);

void cflow_exact_b0(double x,double q,double h,int want_jac,cflow_local_jac *o);
double cflow_exact_b0_event(double x,double q);

void cflow_rk_physical_local(double x,double q,double b,double h,int want_jac,cflow_local_jac *o);

int cflow_sep_unresolved(double x,double q,double b,double remaining);
double cflow_conditioning_logamp(double x,double q,double b,double h);

/* Tensor Chebyshev evaluation with coordinate derivatives. */
void cflow_cheb3(const double *coef,int n0,int n1,int n2,double x0,double x1,double x2,cflow_t3_eval *o);
/* Scalar-only tensor evaluation.  This intentionally omits coordinate
 * derivative recurrences; cflow_eval never consumes them.  The Jacobian
 * path continues to use cflow_cheb3 unchanged. */
double cflow_cheb3_value(const double *coef,int n0,int n1,int n2,double x0,double x1,double x2);
double cflow_cheb2_value(const double *coef,int n0,int n1,double x0,double x1);
void cflow_cheb2(const double *coef,int n0,int n1,double x0,double x1,double *v,double *d0,double *d1);
void cflow_cheb2_with_normal(const double *coef,const double *normal,int n0,int n1,double x0,double x1,double *v,double *d0,double *d1,double *dn);

typedef struct { double j,jx,jq,jb,regx; } cflow_integral_local;
void cflow_integral_core_local(double x,double q,double b,double h,cflow_integral_local *o);
void cflow_integral_highz_local(double x,double q,double b,double h,int panel,cflow_highz_face face,cflow_integral_local *o);
void cflow_integral_terminal_local(double x,double q,double b,double h,cflow_integral_local *o);
int cflow_integral_terminal_value_local(double x,double q,double b,double h,double *j);
int cflow_integral_terminal_all_local(double x,double q,double b,double h,cflow_local_jac *flow,cflow_integral_local *integ);
int cflow_integral_farfield_local(double x,double q,double b,double h,cflow_integral_local *o);
double cflow_farfield_dx_defect(double x,double q,double b,double h);

#endif
