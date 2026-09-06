#ifndef CFLOW_HIGHZ_TUCKER_COEFFS_H
#define CFLOW_HIGHZ_TUCKER_COEFFS_H

#define CFLOW_HZT_MID_N0 25
#define CFLOW_HZT_MID_N1 9
#define CFLOW_HZT_MID_N2 9
#define CFLOW_HZT_MID_R0 12
#define CFLOW_HZT_MID_R1 7
#define CFLOW_HZT_MID_R2 8
#define CFLOW_HZT_UPPER_N0 27
#define CFLOW_HZT_UPPER_N1 9
#define CFLOW_HZT_UPPER_N2 9
#define CFLOW_HZT_UPPER_R0 11
#define CFLOW_HZT_UPPER_R1 7
#define CFLOW_HZT_UPPER_R2 8
#define CFLOW_HZT_NEAR_N0 31
#define CFLOW_HZT_NEAR_N1 9
#define CFLOW_HZT_NEAR_N2 9
#define CFLOW_HZT_NEAR_R0 10
#define CFLOW_HZT_NEAR_R1 6
#define CFLOW_HZT_NEAR_R2 7

extern const double cflow_hzt_mid_g[672];
extern const double cflow_hzt_mid_u0[300];
extern const double cflow_hzt_mid_u1[63];
extern const double cflow_hzt_mid_u2[72];
extern const double cflow_hzt_upper_g[616];
extern const double cflow_hzt_upper_u0[297];
extern const double cflow_hzt_upper_u1[63];
extern const double cflow_hzt_upper_u2[72];
extern const double cflow_hzt_near_g[420];
extern const double cflow_hzt_near_u0[310];
extern const double cflow_hzt_near_u1[54];
extern const double cflow_hzt_near_u2[63];

#endif
