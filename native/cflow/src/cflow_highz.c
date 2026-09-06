#include "cflow_internal.h"
#include "highz_tucker_coeffs.h"
#include "highz_face_tables.h"

typedef struct {
    double value;
    double d0;
    double d1;
    double d2;
} cflow_tucker_eval;

typedef struct {
    double zlo, zhi, cmax, smax, tlo, thi;
    int n0, n1, n2;
    int r0, r1, r2;
    const double *core;
    const double *u0;
    const double *u1;
    const double *u2;
    const double *cface_p;
    const double *cface_m;
    const double *cface_d_p;
    const double *cface_d_m;
    const double *sface_p;
    const double *sface_m;
    const double *sface_d_p;
    const double *sface_d_m;
    const double *dense;
    int nt, nc, ns;
} cflow_highz_panel;

static const cflow_highz_panel P[3] = {
    {
        0.15, 0.80, 0.010, 0.12,
        0.15056827277668602, 0.9272952180016123,
        CFLOW_HZT_MID_N0, CFLOW_HZT_MID_N1, CFLOW_HZT_MID_N2,
        CFLOW_HZT_MID_R0, CFLOW_HZT_MID_R1, CFLOW_HZT_MID_R2,
        cflow_hzt_mid_g, cflow_hzt_mid_u0, cflow_hzt_mid_u1, cflow_hzt_mid_u2,
        cflow_hzt_mid_cface_p, cflow_hzt_mid_cface_m,
        cflow_hzt_mid_cface_d_p, cflow_hzt_mid_cface_d_m,
        cflow_hzt_mid_sface_p, cflow_hzt_mid_sface_m,
        cflow_hzt_mid_sface_d_p, cflow_hzt_mid_sface_d_m,
        cflow_hz_mid, CFLOW_HZ_MID_NT, CFLOW_HZ_MID_NC, CFLOW_HZ_MID_NS
    },
    {
        0.75, 0.94, 0.004, 0.045,
        0.848062078981481, 1.2226303055219356,
        CFLOW_HZT_UPPER_N0, CFLOW_HZT_UPPER_N1, CFLOW_HZT_UPPER_N2,
        CFLOW_HZT_UPPER_R0, CFLOW_HZT_UPPER_R1, CFLOW_HZT_UPPER_R2,
        cflow_hzt_upper_g, cflow_hzt_upper_u0, cflow_hzt_upper_u1, cflow_hzt_upper_u2,
        cflow_hzt_upper_cface_p, cflow_hzt_upper_cface_m,
        cflow_hzt_upper_cface_d_p, cflow_hzt_upper_cface_d_m,
        cflow_hzt_upper_sface_p, cflow_hzt_upper_sface_m,
        cflow_hzt_upper_sface_d_p, cflow_hzt_upper_sface_d_m,
        cflow_hz_upper, CFLOW_HZ_UPPER_NT, CFLOW_HZ_UPPER_NC, CFLOW_HZ_UPPER_NS
    },
    {
        0.92, 0.9800665778412416, 0.001, 0.010,
        1.1680804852142352, 1.3707963267948966,
        CFLOW_HZT_NEAR_N0, CFLOW_HZT_NEAR_N1, CFLOW_HZT_NEAR_N2,
        CFLOW_HZT_NEAR_R0, CFLOW_HZT_NEAR_R1, CFLOW_HZT_NEAR_R2,
        cflow_hzt_near_g, cflow_hzt_near_u0, cflow_hzt_near_u1, cflow_hzt_near_u2,
        cflow_hzt_near_cface_p, cflow_hzt_near_cface_m,
        cflow_hzt_near_cface_d_p, cflow_hzt_near_cface_d_m,
        cflow_hzt_near_sface_p, cflow_hzt_near_sface_m,
        cflow_hzt_near_sface_d_p, cflow_hzt_near_sface_d_m,
        cflow_hz_near, CFLOW_HZ_NEAR_NT, CFLOW_HZ_NEAR_NC, CFLOW_HZ_NEAR_NS
    }
};

#define CFLOW_HZT_MAX_N 31
#define CFLOW_HZT_MAX_R0 12
#define CFLOW_HZT_MAX_R1 7
#define CFLOW_HZT_MAX_R2 8

static void cflow_tucker_basis_value(
    const double *u, int n, int rank, double x, double *a
) {
    double t[CFLOW_HZT_MAX_N];
    t[0] = 1.0;
    if (n > 1) t[1] = x;
    for (int k = 2; k < n; ++k) t[k] = 2.0 * x * t[k - 1] - t[k - 2];
    for (int j = 0; j < rank; ++j) {
        double v = 0.0;
        for (int k = 0; k < n; ++k)
            v = fma(u[(size_t)k * (size_t)rank + (size_t)j], t[k], v);
        a[j] = v;
    }
}

static void cflow_tucker_basis(
    const double *u, int n, int rank, double x, double *a, double *da
) {
    double t[CFLOW_HZT_MAX_N], dt[CFLOW_HZT_MAX_N];
    t[0] = 1.0; dt[0] = 0.0;
    if (n > 1) { t[1] = x; dt[1] = 1.0; }
    for (int k = 2; k < n; ++k) {
        t[k] = 2.0 * x * t[k - 1] - t[k - 2];
        dt[k] = 2.0 * t[k - 1] + 2.0 * x * dt[k - 1] - dt[k - 2];
    }
    for (int j = 0; j < rank; ++j) {
        double v = 0.0, d = 0.0;
        for (int k = 0; k < n; ++k) {
            const double ukj = u[(size_t)k * (size_t)rank + (size_t)j];
            v = fma(ukj, t[k], v);
            d = fma(ukj, dt[k], d);
        }
        a[j] = v; da[j] = d;
    }
}

static double cflow_tucker3_value(
    const cflow_highz_panel *p, double x0, double x1, double x2
) {
    double a[CFLOW_HZT_MAX_R0], b[CFLOW_HZT_MAX_R1], c[CFLOW_HZT_MAX_R2];
    double z[CFLOW_HZT_MAX_R0 * CFLOW_HZT_MAX_R1], y[CFLOW_HZT_MAX_R0];
    cflow_tucker_basis_value(p->u0, p->n0, p->r0, x0, a);
    cflow_tucker_basis_value(p->u1, p->n1, p->r1, x1, b);
    cflow_tucker_basis_value(p->u2, p->n2, p->r2, x2, c);
    for (int i = 0; i < p->r0; ++i) {
        for (int j = 0; j < p->r1; ++j) {
            const double *g = p->core + ((size_t)i * (size_t)p->r1 + (size_t)j) * (size_t)p->r2;
            double v = 0.0;
            for (int k = 0; k < p->r2; ++k) v = fma(g[k], c[k], v);
            z[i * p->r1 + j] = v;
        }
    }
    for (int i = 0; i < p->r0; ++i) {
        double v = 0.0;
        for (int j = 0; j < p->r1; ++j) v = fma(z[i * p->r1 + j], b[j], v);
        y[i] = v;
    }
    double value = 0.0;
    for (int i = 0; i < p->r0; ++i) value = fma(y[i], a[i], value);
    return value;
}

static void cflow_tucker3(
    const cflow_highz_panel *p, double x0, double x1, double x2,
    cflow_tucker_eval *o
) {
    double a[CFLOW_HZT_MAX_R0], da[CFLOW_HZT_MAX_R0];
    double b[CFLOW_HZT_MAX_R1], db[CFLOW_HZT_MAX_R1];
    double c[CFLOW_HZT_MAX_R2], dc[CFLOW_HZT_MAX_R2];
    double z[CFLOW_HZT_MAX_R0 * CFLOW_HZT_MAX_R1];
    double zc[CFLOW_HZT_MAX_R0 * CFLOW_HZT_MAX_R1];
    double y[CFLOW_HZT_MAX_R0], yb[CFLOW_HZT_MAX_R0], yc[CFLOW_HZT_MAX_R0];
    cflow_tucker_basis(p->u0, p->n0, p->r0, x0, a, da);
    cflow_tucker_basis(p->u1, p->n1, p->r1, x1, b, db);
    cflow_tucker_basis(p->u2, p->n2, p->r2, x2, c, dc);
    for (int i = 0; i < p->r0; ++i) {
        for (int j = 0; j < p->r1; ++j) {
            const double *g = p->core + ((size_t)i * (size_t)p->r1 + (size_t)j) * (size_t)p->r2;
            double v = 0.0, d = 0.0;
            for (int k = 0; k < p->r2; ++k) {
                v = fma(g[k], c[k], v);
                d = fma(g[k], dc[k], d);
            }
            z[i * p->r1 + j] = v;
            zc[i * p->r1 + j] = d;
        }
    }
    for (int i = 0; i < p->r0; ++i) {
        double v = 0.0, d1 = 0.0, d2 = 0.0;
        for (int j = 0; j < p->r1; ++j) {
            v = fma(z[i * p->r1 + j], b[j], v);
            d1 = fma(z[i * p->r1 + j], db[j], d1);
            d2 = fma(zc[i * p->r1 + j], b[j], d2);
        }
        y[i] = v; yb[i] = d1; yc[i] = d2;
    }
    o->value = o->d0 = o->d1 = o->d2 = 0.0;
    for (int i = 0; i < p->r0; ++i) {
        o->value = fma(y[i], a[i], o->value);
        o->d0 = fma(y[i], da[i], o->d0);
        o->d1 = fma(yb[i], a[i], o->d1);
        o->d2 = fma(yc[i], a[i], o->d2);
    }
}

static cflow_highz_face cflow_s_face(double x) {
    return x >= 0.0 ? CFLOW_HZ_FACE_S_POS : CFLOW_HZ_FACE_S_NEG;
}

static cflow_highz_face cflow_c_face(double z, double bx_signed) {
    const double sign = (z >= 0.0 ? 1.0 : -1.0) * bx_signed;
    return sign >= 0.0 ? CFLOW_HZ_FACE_C_POS : CFLOW_HZ_FACE_C_NEG;
}

int cflow_highz_candidate(
    double x, double q, double b, double remaining,
    int *panel, cflow_highz_face *face, double *hcap
) {
    const double signed_z = q * x;
    const double z = fabs(signed_z);
    if (x == 0.0 || z < 0.15 || z > 0.9800665778412416) return 0;

    double best = 0.0;
    int best_panel = -1;
    cflow_highz_face best_face = CFLOW_HZ_FACE_NONE;
    for (int i = 0; i < 3; ++i) {
        if (z < P[i].zlo || z > P[i].zhi) continue;

        double hc = P[i].smax * fabs(x) / 2.0;
        cflow_highz_face candidate_face = cflow_s_face(x);
        const double bx_signed = b * x;
        const double bx = fabs(bx_signed);
        if (bx > 0.0) {
            const double cc = P[i].cmax / bx;
            if (cc <= hc) {
                hc = cc;
                candidate_face = cflow_c_face(signed_z, bx_signed);
            }
        }
        hc *= CFLOW_CORE_SAFETY;
        if (hc > remaining) {
            hc = remaining;
            candidate_face = CFLOW_HZ_FACE_NONE;
        }
#ifdef CFLOW_DISABLE_HIGHZ_FACE_SPECIALIZATION
        candidate_face = CFLOW_HZ_FACE_NONE;
#endif
        if (hc > best) {
            best = hc;
            best_panel = i;
            best_face = candidate_face;
        }
    }
    if (best_panel < 0 || best <= 0.0) return 0;
    *panel = best_panel;
    *face = best_face;
    *hcap = best;
    return 1;
}

static const double *cflow_face_value_table(const cflow_highz_panel *p, cflow_highz_face face) {
    switch (face) {
        case CFLOW_HZ_FACE_C_POS: return p->cface_p;
        case CFLOW_HZ_FACE_C_NEG: return p->cface_m;
        case CFLOW_HZ_FACE_S_POS: return p->sface_p;
        case CFLOW_HZ_FACE_S_NEG: return p->sface_m;
        default: return NULL;
    }
}

static const double *cflow_face_normal_table(const cflow_highz_panel *p, cflow_highz_face face) {
    switch (face) {
        case CFLOW_HZ_FACE_C_POS: return p->cface_d_p;
        case CFLOW_HZ_FACE_C_NEG: return p->cface_d_m;
        case CFLOW_HZ_FACE_S_POS: return p->sface_d_p;
        case CFLOW_HZ_FACE_S_NEG: return p->sface_d_m;
        default: return NULL;
    }
}

static int cflow_is_c_face(cflow_highz_face face) {
    return face == CFLOW_HZ_FACE_C_POS || face == CFLOW_HZ_FACE_C_NEG;
}

static int cflow_is_s_face(cflow_highz_face face) {
    return face == CFLOW_HZ_FACE_S_POS || face == CFLOW_HZ_FACE_S_NEG;
}

void cflow_highz_local(
    double x, double q, double b, double h, int panel, cflow_highz_face face,
    int want_jac, cflow_local_jac *o
) {
    CFLOW_ASSERT(panel >= 0 && panel < 3);
    const cflow_highz_panel *p = &P[panel];
    const double z = q * x;
    const double sig = z >= 0.0 ? 1.0 : -1.0;
    const double az = fabs(z);
    const double th = asin(az);
    const double xt = (2.0 * th - p->tlo - p->thi) / (p->thi - p->tlo);
    CFLOW_ASSERT(fabs(xt) <= 1.0 + 64.0 * DBL_EPSILON);

    if (!want_jac) {
        double kval;
        if (cflow_is_c_face(face)) {
            const double xs = (2.0 * h / x) / p->smax;
            CFLOW_ASSERT(fabs(xs) <= 1.0 + 64.0 * DBL_EPSILON);
            kval = cflow_cheb2_value(cflow_face_value_table(p, face), p->n0, p->n2, xt, xs);
        } else if (cflow_is_s_face(face)) {
            const double xc = (sig * b * h * x) / p->cmax;
            CFLOW_ASSERT(fabs(xc) <= 1.0 + 64.0 * DBL_EPSILON);
            kval = cflow_cheb2_value(cflow_face_value_table(p, face), p->n0, p->n1, xt, xc);
        } else {
            const double C = sig * b * h * x;
            const double sv = 2.0 * h / x;
            const double xc = C / p->cmax;
            const double xs = sv / p->smax;
            CFLOW_ASSERT(fabs(xc) <= 1.0 + 64.0 * DBL_EPSILON);
            CFLOW_ASSERT(fabs(xs) <= 1.0 + 64.0 * DBL_EPSILON);
            kval = cflow_tucker3_value(p, xt, xc, xs);
        }
        o->x = x + 2.0 * h * kval;
        o->dx = o->dq = o->db = NAN;
        return;
    }

    const double sv = 2.0 * h / x;
    const double xs = sv / p->smax;
    const double C = sig * b * h * x;
    const double xc = C / p->cmax;
    cflow_tucker_eval e;
    if (cflow_is_c_face(face)) {
        cflow_cheb2_with_normal(cflow_face_value_table(p, face), cflow_face_normal_table(p, face), p->n0, p->n2, xt, xs,
                    &e.value, &e.d0, &e.d2, &e.d1);
    } else if (cflow_is_s_face(face)) {
        cflow_cheb2_with_normal(cflow_face_value_table(p, face), cflow_face_normal_table(p, face), p->n0, p->n1, xt, xc,
                    &e.value, &e.d0, &e.d1, &e.d2);
    } else {
        cflow_tucker3(p, xt, xc, xs, &e);
    }
    CFLOW_ASSERT(fabs(xc) <= 1.0 + 64.0 * DBL_EPSILON);
    CFLOW_ASSERT(fabs(xs) <= 1.0 + 64.0 * DBL_EPSILON);

    const double ct = sqrt(fmax(0.0, 1.0 - az * az));
    const double kth = e.d0 * 2.0 / (p->thi - p->tlo);
    const double kC = e.d1 / p->cmax;
    const double ks = e.d2 / p->smax;
    o->x = x + 2.0 * h * e.value;

    const double thx = sig * q / ct;
    const double thq = sig * x / ct;
    const double Cx = sig * b * h;
    const double Cb = sig * h * x;
    const double sx = -sv / x;
    const double kx = kth * thx + kC * Cx + ks * sx;
    const double kq = kth * thq;
    const double kb = kC * Cb;
    o->dx = 1.0 + 2.0 * h * kx;
    o->dq = 2.0 * h * kq;
    o->db = 2.0 * h * kb;
}

double cflow_highz_dx_defect(double x, double q, double b, double h, int panel) {
    /* Preserve the dense polynomial derivative for the integral correction path.
     * Tucker compression changes only the high-z endpoint map. */
    CFLOW_ASSERT(panel >= 0 && panel < 3);
    const cflow_highz_panel *p = &P[panel];
    const double z = q * x;
    const double sig = z >= 0.0 ? 1.0 : -1.0;
    const double az = fabs(z);
    const double ct = sqrt(fmax(0.0, 1.0 - az * az));
    const double th = asin(az);
    const double C = sig * b * h * x;
    const double sv = 2.0 * h / x;
    const double xt = (2.0 * th - p->tlo - p->thi) / (p->thi - p->tlo);
    const double xc = C / p->cmax;
    const double xs = sv / p->smax;

    cflow_t3_eval e;
    cflow_cheb3(p->dense, p->nt, p->nc, p->ns, xt, xc, xs, &e);
    const double kth = e.da * 2.0 / (p->thi - p->tlo);
    const double kC = e.db / p->cmax;
    const double ks = e.dc / p->smax;
    const double thx = sig * q / ct;
    const double Cx = sig * b * h;
    const double sx = -sv / x;
    const double kx = kth * thx + kC * Cx + ks * sx;
    return 2.0 * h * kx;
}
