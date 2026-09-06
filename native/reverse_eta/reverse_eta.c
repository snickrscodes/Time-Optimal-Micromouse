#include "reverse_eta.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/*
 * Reverse-only regularized tangent cocycle for the contracting, sign-stable
 * Cflow branch.  It is deliberately not a public Cflow authority: the Python
 * reverse solver invokes it only after the scalar topology/status build has
 * succeeded and only behind a conservative high-work selector.
 *
 * Coordinate:
 *   Q = |q|, B = |b|, eta = tan(delta) = sqrt(1-(Qx)^2)/(Qx)
 *
 * Integrator:
 *   3-stage Radau IIA in long double, full-vs-two-half adaptive estimator.
 *   The exponentially contracting homogeneous multiplier is stored as log Phi.
 */

typedef struct eta_y {
    long double eta, SQ, SB, I, ID, IQ, IB, L;
} eta_y;

typedef struct eta_coef {
    long double f, a, fQ, fB, G, Ge, GQ;
} eta_coef;

#define ETA_SQ6 2.4494897427831780981972840747058913919659474806567L
static const long double ETA_C[3] = {
    (4.0L - ETA_SQ6) / 10.0L,
    (4.0L + ETA_SQ6) / 10.0L,
    1.0L
};
static const long double ETA_A[3][3] = {
    {(88.0L - 7.0L * ETA_SQ6) / 360.0L,
     (296.0L - 169.0L * ETA_SQ6) / 1800.0L,
     (-2.0L + 3.0L * ETA_SQ6) / 225.0L},
    {(296.0L + 169.0L * ETA_SQ6) / 1800.0L,
     (88.0L + 7.0L * ETA_SQ6) / 360.0L,
     (-2.0L - 3.0L * ETA_SQ6) / 225.0L},
    {(16.0L - ETA_SQ6) / 36.0L,
     (16.0L + ETA_SQ6) / 36.0L,
     1.0L / 9.0L}
};
static const long double ETA_BW[3] = {
    (16.0L - ETA_SQ6) / 36.0L,
    (16.0L + ETA_SQ6) / 36.0L,
    1.0L / 9.0L
};

static int eta_coef_eval(long double eta, long double Q, long double B, eta_coef *o) {
    long double T, D, k, rootT;
    if (o == NULL || !(eta > 0.0L) || !(Q > 0.0L) || !(B > 0.0L) ||
        !isfinite(eta) || !isfinite(Q) || !isfinite(B)) {
        return 0;
    }
    T = 1.0L + eta * eta;
    D = B / (Q * eta);
    k = D - 2.0L * Q;
    o->f = T * k;
    o->a = 2.0L * eta * k - T * D / eta;
    o->fQ = T * (-D / Q - 2.0L);
    o->fB = T / (Q * eta);
    rootT = sqrtl(T);
    o->G = sqrtl(Q * rootT);
    o->Ge = o->G * eta / (2.0L * T);
    o->GQ = o->G / (2.0L * Q);
    return isfinite(o->f) && isfinite(o->a) && isfinite(o->fQ) &&
           isfinite(o->fB) && isfinite(o->G) && isfinite(o->Ge) &&
           isfinite(o->GQ);
}

static void swap_ld(long double *a, long double *b) {
    long double t = *a;
    *a = *b;
    *b = t;
}

static int solve3(long double M[3][3], const long double rhs[3], long double x[3]) {
    long double a[3][4];
    int i, j, k;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) a[i][j] = M[i][j];
        a[i][3] = rhs[i];
        x[i] = 0.0L;
    }
    for (k = 0; k < 3; ++k) {
        int p = k;
        long double best = fabsl(a[k][k]);
        for (i = k + 1; i < 3; ++i) {
            long double v = fabsl(a[i][k]);
            if (v > best) { best = v; p = i; }
        }
        if (!(best > 1e-30L) || !isfinite(best)) return 0;
        if (p != k) {
            for (j = k; j < 4; ++j) swap_ld(&a[p][j], &a[k][j]);
        }
        {
            long double inv = 1.0L / a[k][k];
            for (i = k + 1; i < 3; ++i) {
                long double m = a[i][k] * inv;
                for (j = k; j < 4; ++j) a[i][j] -= m * a[k][j];
            }
        }
    }
    for (i = 2; i >= 0; --i) {
        long double s = a[i][3];
        for (j = i + 1; j < 3; ++j) s -= a[i][j] * x[j];
        x[i] = s / a[i][i];
    }
    return 1;
}

/* Solve the same 3x3 matrix for two RHS vectors with one elimination. */
static int solve3_two_rhs(
    long double M[3][3],
    const long double rhs0[3],
    const long double rhs1[3],
    long double x0[3],
    long double x1[3]
) {
    long double a[3][5];
    int i, j, k;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) a[i][j] = M[i][j];
        a[i][3] = rhs0[i];
        a[i][4] = rhs1[i];
        x0[i] = x1[i] = 0.0L;
    }
    for (k = 0; k < 3; ++k) {
        int p = k;
        long double best = fabsl(a[k][k]);
        for (i = k + 1; i < 3; ++i) {
            long double v = fabsl(a[i][k]);
            if (v > best) { best = v; p = i; }
        }
        if (!(best > 1e-30L) || !isfinite(best)) return 0;
        if (p != k) {
            for (j = k; j < 5; ++j) swap_ld(&a[p][j], &a[k][j]);
        }
        {
            long double inv = 1.0L / a[k][k];
            for (i = k + 1; i < 3; ++i) {
                long double m = a[i][k] * inv;
                for (j = k; j < 5; ++j) a[i][j] -= m * a[k][j];
            }
        }
    }
    for (i = 2; i >= 0; --i) {
        long double s0 = a[i][3], s1 = a[i][4];
        for (j = i + 1; j < 3; ++j) {
            s0 -= a[i][j] * x0[j];
            s1 -= a[i][j] * x1[j];
        }
        x0[i] = s0 / a[i][i];
        x1[i] = s1 / a[i][i];
    }
    return 1;
}

static int one_step(
    const eta_y *y,
    long double t,
    long double h,
    long double Q0,
    long double B,
    eta_y *out,
    int64_t *newton_iters
) {
    long double z[3];
    eta_coef c0, cf[3];
    long double M[3][3];
    int i, j, it;

    if (!eta_coef_eval(y->eta, Q0 - B * t, B, &c0)) return 0;
    for (i = 0; i < 3; ++i) {
        z[i] = y->eta + ETA_C[i] * h * c0.f;
        if (!(z[i] > 0.0L)) z[i] = fmaxl(y->eta * 0.25L, 1e-24L);
    }

    for (it = 0; it < 20; ++it) {
        long double R[3], norm = 0.0L;
        long double scale;
        int ok = 1;
        ++(*newton_iters);
        for (j = 0; j < 3; ++j) {
            long double Q = Q0 - B * (t + ETA_C[j] * h);
            ok = ok && eta_coef_eval(z[j], Q, B, &cf[j]);
        }
        if (!ok) return 0;
        for (i = 0; i < 3; ++i) {
            long double s = y->eta;
            for (j = 0; j < 3; ++j) s += h * ETA_A[i][j] * cf[j].f;
            R[i] = z[i] - s;
            norm = fmaxl(norm, fabsl(R[i]));
            for (j = 0; j < 3; ++j) {
                M[i][j] = (i == j ? 1.0L : 0.0L) - h * ETA_A[i][j] * cf[j].a;
            }
        }
        scale = fmaxl(1.0L, fabsl(y->eta));
        if (norm < 2e-18L * scale) break;
        {
            const long double rhs[3] = {-R[0], -R[1], -R[2]};
            long double dz[3], Mc[3][3], alpha = 1.0L, dnorm = 0.0L;
            int ls;
            memcpy(Mc, M, sizeof(Mc));
            if (!solve3(Mc, rhs, dz)) return 0;
            for (ls = 0; ls < 20; ++ls) {
                int positive = 1;
                for (i = 0; i < 3; ++i) {
                    if (!(z[i] + alpha * dz[i] > 0.0L)) positive = 0;
                }
                if (positive) break;
                alpha *= 0.5L;
            }
            if (alpha < 1e-6L) return 0;
            for (i = 0; i < 3; ++i) {
                long double d = alpha * dz[i];
                z[i] += d;
                dnorm = fmaxl(dnorm, fabsl(d));
            }
            if (dnorm < 2e-18L * scale) break;
        }
        if (it == 19) return 0;
    }

    for (j = 0; j < 3; ++j) {
        long double Q = Q0 - B * (t + ETA_C[j] * h);
        if (!eta_coef_eval(z[j], Q, B, &cf[j])) return 0;
    }
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            M[i][j] = (i == j ? 1.0L : 0.0L) - h * ETA_A[i][j] * cf[j].a;
        }
    }

    {
        long double rhsQ[3], rhsB[3], SQs[3], SBs[3], Ls[3];
        long double dI[3], dID[3], dIQ[3], dIB[3];
        long double fend = 0.0L, aend = 0.0L, qend = 0.0L, bend = 0.0L;
        long double iend = 0.0L, idend = 0.0L, iqend = 0.0L, ibend = 0.0L;

        for (i = 0; i < 3; ++i) {
            long double fq = 0.0L, fb = 0.0L;
            for (j = 0; j < 3; ++j) {
                long double tj = t + ETA_C[j] * h;
                fq += ETA_A[i][j] * cf[j].fQ;
                fb += ETA_A[i][j] * (cf[j].fB - tj * cf[j].fQ);
            }
            rhsQ[i] = y->SQ + h * fq;
            rhsB[i] = y->SB + h * fb;
        }
        if (!solve3_two_rhs(M, rhsQ, rhsB, SQs, SBs)) return 0;

        for (i = 0; i < 3; ++i) {
            long double s = y->L;
            for (j = 0; j < 3; ++j) s += h * ETA_A[i][j] * cf[j].a;
            Ls[i] = s;
        }
        for (j = 0; j < 3; ++j) {
            long double tj = t + ETA_C[j] * h;
            long double ph = expl(Ls[j]);
            dI[j] = cf[j].G;
            dID[j] = cf[j].Ge * ph;
            dIQ[j] = cf[j].Ge * SQs[j] + cf[j].GQ;
            dIB[j] = cf[j].Ge * SBs[j] - tj * cf[j].GQ;
        }
        for (j = 0; j < 3; ++j) {
            long double tj = t + ETA_C[j] * h;
            fend += ETA_BW[j] * cf[j].f;
            aend += ETA_BW[j] * cf[j].a;
            qend += ETA_BW[j] * (cf[j].a * SQs[j] + cf[j].fQ);
            bend += ETA_BW[j] * (cf[j].a * SBs[j] + cf[j].fB - tj * cf[j].fQ);
            iend += ETA_BW[j] * dI[j];
            idend += ETA_BW[j] * dID[j];
            iqend += ETA_BW[j] * dIQ[j];
            ibend += ETA_BW[j] * dIB[j];
        }

        *out = *y;
        out->eta = y->eta + h * fend;
        out->SQ = y->SQ + h * qend;
        out->SB = y->SB + h * bend;
        out->I = y->I + h * iend;
        out->ID = y->ID + h * idend;
        out->IQ = y->IQ + h * iqend;
        out->IB = y->IB + h * ibend;
        out->L = y->L + h * aend;
    }
    return out->eta > 0.0L && isfinite(out->eta) && isfinite(out->SQ) &&
           isfinite(out->SB) && isfinite(out->I) && isfinite(out->ID) &&
           isfinite(out->IQ) && isfinite(out->IB) && isfinite(out->L);
}

static long double errnorm(const eta_y *a, const eta_y *b, long double tol) {
    const long double *pa = &a->eta;
    const long double *pb = &b->eta;
    long double e = 0.0L;
    int i;
    for (i = 0; i < 8; ++i) {
        long double sc = 1e-18L + tol * fmaxl(1.0L, fmaxl(fabsl(pa[i]), fabsl(pb[i])));
        e = fmaxl(e, fabsl(pa[i] - pb[i]) / sc);
    }
    return e;
}

static int all_double_outputs_finite(const ame_reverse_eta_result *r) {
    return isfinite(r->x) && isfinite(r->dx_dx0) && isfinite(r->dx_dq0) &&
           isfinite(r->dx_db) && isfinite(r->dx_dh) && isfinite(r->integral) &&
           isfinite(r->dI_dx0) && isfinite(r->dI_dq0) && isfinite(r->dI_db) &&
           isfinite(r->dI_dh) && isfinite(r->regularized_dI_dx0);
}

int ame_reverse_eta_all(
    double x0d,
    double q0d,
    double bd,
    double hd,
    double told,
    ame_reverse_eta_result *r
) {
    long double x0 = (long double)x0d;
    long double q0 = (long double)q0d;
    long double b = (long double)bd;
    long double H = (long double)hd;
    long double tol = (long double)told;
    long double sig, Q0, B, Qh, z, s, eta0, t, dt;
    eta_y y;
    int64_t acc = 0, rej = 0, ones = 0, nit = 0;

    if (r == NULL) return 0;
    memset(r, 0, sizeof(*r));
    r->status = AME_REVERSE_ETA_INVALID_ARGUMENT;

    if (!(x0 > 0.0L) || !(H > 0.0L) || !(tol > 0.0L) ||
        q0 == 0.0L || b == 0.0L || q0 * b >= 0.0L ||
        !isfinite(x0) || !isfinite(q0) || !isfinite(b) ||
        !isfinite(H) || !isfinite(tol)) {
        return 0;
    }
    sig = q0 > 0.0L ? 1.0L : -1.0L;
    Q0 = sig * q0;
    B = -sig * b;
    Qh = Q0 - B * H;
    if (!(Qh > 0.0L)) { r->status = AME_REVERSE_ETA_Q_CROSSING; return 0; }
    z = Q0 * x0;
    if (!(z > 0.0L && z < 1.0L)) { r->status = AME_REVERSE_ETA_OUTSIDE_DOMAIN; return 0; }
    s = sqrtl((1.0L - z) * (1.0L + z));
    eta0 = s / z;
    if (!(eta0 > 0.0L) || !isfinite(eta0)) {
        r->status = AME_REVERSE_ETA_OUTSIDE_DOMAIN;
        return 0;
    }

    y.eta = eta0;
    y.SQ = y.SB = y.I = y.ID = y.IQ = y.IB = y.L = 0.0L;
    t = 0.0L;
    dt = H / 16.0L;

    while (t < H && acc + rej < 200000) {
        eta_y full, h1, h2;
        int64_t ni0 = 0, ni1 = 0, ni2 = 0;
        int ok0, ok1, ok2;
        long double e, fac;
        if (t + dt > H) dt = H - t;
        ok0 = one_step(&y, t, dt, Q0, B, &full, &ni0); ++ones;
        ok1 = one_step(&y, t, dt / 2.0L, Q0, B, &h1, &ni1); ++ones;
        ok2 = ok1 && one_step(&h1, t + dt / 2.0L, dt / 2.0L, Q0, B, &h2, &ni2); ++ones;
        nit += ni0 + ni1 + ni2;
        if (!(ok0 && ok2)) {
            ++rej;
            dt *= 0.25L;
            if (dt < 1e-18L * fmaxl(1.0L, H)) {
                r->status = AME_REVERSE_ETA_STEP_FAILURE;
                return 0;
            }
            continue;
        }
        e = errnorm(&full, &h2, tol);
        if (e <= 1.0L) {
            y = h2;
            t += dt;
            ++acc;
        } else {
            ++rej;
        }
        fac = e == 0.0L ? 2.5L : 0.9L * powl(1.0L / e, 1.0L / 6.0L);
        if (fac < 0.2L) fac = 0.2L;
        if (fac > 2.5L) fac = 2.5L;
        dt *= fac;
    }
    if (t < H) { r->status = AME_REVERSE_ETA_WORK_LIMIT; return 0; }

    {
        long double eta = y.eta;
        long double T = 1.0L + eta * eta;
        long double Q = Qh;
        long double x = 1.0L / (Q * sqrtl(T));
        long double xe = -x * eta / T;
        long double xQ = -x / Q;
        long double e0x = -(1.0L + eta0 * eta0) / (eta0 * x0);
        long double e0Q = -(1.0L + eta0 * eta0) / (eta0 * Q0);
        long double ph = expl(y.L);
        long double xx = xe * ph * e0x;
        long double xq = sig * (xe * (ph * e0Q + y.SQ) + xQ);
        long double xb = sig * (-xe * y.SB + H * xQ);
        long double xh = 2.0L * eta / sqrtl(T);
        long double Ix = y.ID * e0x;
        long double Iq = sig * (y.ID * e0Q + y.IQ);
        long double Ib = -sig * y.IB;
        eta_coef ce;
        if (!eta_coef_eval(eta, Q, B, &ce)) {
            r->status = AME_REVERSE_ETA_NONFINITE_OUTPUT;
            return 0;
        }
        r->x = (double)x;
        r->dx_dx0 = (double)xx;
        r->dx_dq0 = (double)xq;
        r->dx_db = (double)xb;
        r->dx_dh = (double)xh;
        r->integral = (double)y.I;
        r->dI_dx0 = (double)Ix;
        r->dI_dq0 = (double)Iq;
        r->dI_db = (double)Ib;
        r->dI_dh = (double)ce.G;
        r->regularized_dI_dx0 = (double)(Ix + 1.0L / (2.0L * sqrtl(x0)));
    }
    r->accepted = acc;
    r->rejected = rej;
    r->one_steps = ones;
    r->newton_iters = nit;
    if (!all_double_outputs_finite(r)) {
        r->status = AME_REVERSE_ETA_NONFINITE_OUTPUT;
        return 0;
    }
    r->status = AME_REVERSE_ETA_OK;
    return 1;
}
