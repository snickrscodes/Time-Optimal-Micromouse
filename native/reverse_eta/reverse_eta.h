#ifndef AME_REVERSE_ETA_H
#define AME_REVERSE_ETA_H

#include <stdint.h>

#if defined(_WIN32)
  #define AME_REVERSE_ETA_API __declspec(dllexport)
#else
  #define AME_REVERSE_ETA_API __attribute__((visibility("default")))
#endif

typedef struct ame_reverse_eta_result {
    double x;
    double dx_dx0;
    double dx_dq0;
    double dx_db;
    double dx_dh;
    double integral;
    double dI_dx0;
    double dI_dq0;
    double dI_db;
    double dI_dh;
    double regularized_dI_dx0;
    int64_t accepted;
    int64_t rejected;
    int64_t one_steps;
    int64_t newton_iters;
    int status;
} ame_reverse_eta_result;

enum {
    AME_REVERSE_ETA_OK = 0,
    AME_REVERSE_ETA_INVALID_ARGUMENT = 1,
    AME_REVERSE_ETA_Q_CROSSING = 2,
    AME_REVERSE_ETA_OUTSIDE_DOMAIN = 3,
    AME_REVERSE_ETA_STEP_FAILURE = 4,
    AME_REVERSE_ETA_WORK_LIMIT = 5,
    AME_REVERSE_ETA_NONFINITE_OUTPUT = 6
};

AME_REVERSE_ETA_API int ame_reverse_eta_all(
    double x0,
    double q0,
    double b,
    double h,
    double rtol,
    ame_reverse_eta_result *out
);

#endif
