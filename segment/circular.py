from __future__ import annotations

import math

from .base import DiffSegment, EvalSegment, SegmentType
from .constants import MU_G


MU_G_INV = 1.0 / MU_G

CIRCULAR_STABLE_PHASE_MAX = 3.0e-2
CIRCULAR_SMALL_STEP_TOL = 3.0e-2
CIRCULAR_SMALL_TIME_TOL = 5.0e-2
CIRCULAR_SMALL_JAC_TOL = 5.0e-2
CIRCULAR_SMALL_SIGMA_TOL = 1.7e-1

_PHASE_ORDER3_MAX = 3.0e-3
_PHASE_ORDER4_MAX = 1.5e-2

PI = math.pi
HALF_PI = 0.5 * math.pi
PI_LO = 1.2246467991473532e-16
J_PI = 5.244115108584239


# ---------------------------------------------------------------------------
# Generic polynomial helpers
# ---------------------------------------------------------------------------


def _poly_val_asc(c: tuple[float, ...], x: float) -> float:
    p = c[-1]
    for a in c[-2::-1]:
        p = math.fma(p, x, a)
    return p


def _poly_add(a: list[float], b: list[float]) -> list[float]:
    n = max(len(a), len(b))
    out = [0.0] * n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0)
    return out


def _poly_mul(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _time_tables(
    coefficient_sets: tuple[tuple[float, ...], ...],
    denominators: tuple[float, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Normalize T polynomials and precombine the dT/dw0 polynomials."""
    f_tables: list[tuple[float, ...]] = []
    tw_tables: list[tuple[float, ...]] = []

    for order, (coef, den) in enumerate(zip(coefficient_sets, denominators), 1):
        f = [v / den for v in coef]
        derivative = [(i + 1.0) * f[i + 1] for i in range(len(f) - 1)]

        # g = (1-x) f'(x) - f(x)
        g = _poly_add(_poly_mul([1.0, -1.0], derivative), [-v for v in f])

        # The complete coefficient multiplying z**order in the dT/dw0
        # bracket.  This removes Q and every runtime polynomial derivative.
        term = _poly_add(
            _poly_mul([0.5, 0.5], g),
            _poly_mul([-0.5, 2.0 * order], f),
        )
        tw = _poly_mul([1.0, -1.0], term)

        f_tables.append(tuple(f))
        tw_tables.append(tuple(tw))

    return tuple(f_tables), tuple(tw_tables)


# ---------------------------------------------------------------------------
# Small-angle joint trig kernel
# ---------------------------------------------------------------------------


def _circular_trig_small(h: float) -> tuple[float, float, float, float, float]:
    """Return sin, cos, sinc, sinc', and cos-1 for |h| <= 0.1."""
    y = h * h

    sinc_tail = math.fma(
        y,
        math.fma(
            y,
            math.fma(
                y,
                math.fma(y, -1.0 / 39916800.0, 1.0 / 362880.0),
                -1.0 / 5040.0,
            ),
            1.0 / 120.0,
        ),
        -1.0 / 6.0,
    )
    sinc_h = math.fma(y, sinc_tail, 1.0)

    cos_tail = math.fma(
        y,
        math.fma(
            y,
            math.fma(
                y,
                math.fma(y, -1.0 / 3628800.0, 1.0 / 40320.0),
                -1.0 / 720.0,
            ),
            1.0 / 24.0,
        ),
        -0.5,
    )
    cosm1_h = y * cos_tail
    cos_h = math.fma(y, cos_tail, 1.0)

    sinc_prime_tail = math.fma(
        y,
        math.fma(
            y,
            math.fma(
                y,
                math.fma(y, 1.0 / 518918400.0, -1.0 / 3991680.0),
                1.0 / 45360.0,
            ),
            -1.0 / 840.0,
        ),
        1.0 / 30.0,
    )
    sinc_prime_h = h * math.fma(y, sinc_prime_tail, -1.0 / 3.0)

    return h * sinc_h, cos_h, sinc_h, sinc_prime_h, cosm1_h


# ---------------------------------------------------------------------------
# Paired J / Q evaluator and exact H-tilde construction
# ---------------------------------------------------------------------------

# Two extra high-order terms beyond the original endpoint series keep both
# integrals within a few ulps on [0, pi/2].  Reflection uses a split pi so the
# endpoint distance remains accurate near pi.
J_HALF_PI = 2.6220575542921196
Q_HALF_PI = 1.4441419676288028
Q_PI = 8.237436749853744

_J_COEF = (
    5.415061430367538e-24, 5.747962014242548e-23,
    6.124041419251274e-22, 6.55163840757515e-21,
    7.041300404354642e-20, 7.606546176774606e-19,
    8.26496149318051e-18, 9.039807986355853e-17,
    9.962444876166615e-16, 1.1076078857213077e-14,
    1.2441763326112309e-13, 1.4148369695947575e-12,
    1.632992892830863e-11, 1.9197487480703726e-10,
    2.3101703945245365e-09, 2.8666683707606327e-08,
    3.710843554593555e-07, 5.1102616972715015e-06,
    7.758445258445259e-05, 0.001388888888888889,
    0.03333333333333333, 2.0,
)

_Q_COEF = (
    5.290577259554491e-24, 5.609456905465618e-23,
    5.96900239597909e-22, 6.376928050039812e-21,
    6.842953914091131e-20, 7.379485096870887e-19,
    8.002581763238272e-18, 8.73337381732684e-17,
    9.60017415339692e-16, 1.0641722823596878e-14,
    1.1912326588830934e-13, 1.3490305989159316e-12,
    1.5492496675574852e-11, 1.8100488196092084e-10,
    2.161127143264889e-09, 2.6543225655191043e-08,
    3.3881615063680283e-07, 4.5723394133481856e-06,
    6.723985890652557e-05, 0.0011363636363636363,
    0.023809523809523808, 0.6666666666666666,
)


def _J_Q_left(x: float) -> tuple[float, float]:
    if x == 0.0:
        return 0.0, 0.0

    z = x * x
    pj = _J_COEF[0]
    pq = _Q_COEF[0]
    for aj, aq in zip(_J_COEF[1:], _Q_COEF[1:]):
        pj = math.fma(pj, z, aj)
        pq = math.fma(pq, z, aq)

    root = math.sqrt(x)
    return root * pj, x * root * pq


def _J_left(x: float) -> float:
    if x == 0.0:
        return 0.0

    z = x * x
    p = _J_COEF[0]
    for a in _J_COEF[1:]:
        p = math.fma(p, z, a)
    return math.sqrt(x) * p


def _J_Q(x: float) -> tuple[float, float]:
    if x == 0.0:
        return 0.0, 0.0
    if x == HALF_PI:
        return J_HALF_PI, Q_HALF_PI
    if x == PI:
        return J_PI, Q_PI
    if x <= HALF_PI:
        return _J_Q_left(x)

    r = (PI - x) + PI_LO
    jr, qr = _J_Q_left(r)
    reflected_q = math.fma(-PI, jr, Q_PI)
    reflected_q = math.fma(-PI_LO, jr, reflected_q)
    reflected_q += qr
    return J_PI - jr, reflected_q


def _J(x: float) -> float:
    if x == 0.0:
        return 0.0
    if x == HALF_PI:
        return J_HALF_PI
    if x == PI:
        return J_PI
    if x <= HALF_PI:
        return _J_left(x)

    r = (PI - x) + PI_LO
    return J_PI - _J_left(r)


def _H_tilde_from_JQ(x: float, sin_x: float, j: float, q: float) -> float:
    """Nonsingular H-tilde built from the more accurate J/Q evaluator."""
    y = math.sqrt(sin_x)
    elementary = 2.0 * (math.log1p(y) - math.atan(y))
    return elementary + math.fma(-x, j, q)


# ---------------------------------------------------------------------------
# Stable small-curvature series tables
# ---------------------------------------------------------------------------

_T2 = (1.0, 2.0, 3.0, 4.0, 5.0)
_T4 = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 17.0, 28.0, 39.0, -10.0, -5.0)
_T6 = (61.0, 122.0, 183.0, 244.0, 305.0, 366.0, 1246.0, 2126.0, 3006.0, 3886.0, 9680.0, 15474.0, 22633.0, -8008.0, -3549.0, 910.0, 455.0)
_T8 = (1261.0, 2522.0, 3783.0, 5044.0, 6305.0, 7566.0, 29567.0, 51568.0, 73569.0, 95570.0, 233239.0, 370908.0, 583547.0, 796186.0, 1926825.0, 3057464.0, 4573663.0, -1848138.0, -761039.0, 326060.0, 148155.0, -29750.0, -14875.0)
_T10 = (711.0, 1422.0, 2133.0, 2844.0, 3555.0, 4266.0, 18848.0, 33430.0, 48012.0, 62594.0, 149644.0, 236694.0, 397554.0, 558414.0, 1253874.0, 1949334.0, 3168702.0, 4388070.0, 10539288.0, 16690506.0, 25297320.0, -11002266.0, -4311912.0, 2378442.0, 1008271.0, -361900.0, -167475.0, 26950.0, 13475.0)

_S1 = (15.0, 45.0, 62.0, 66.0, 57.0, 35.0)
_S3 = (35.0, 105.0, 166.0, 218.0, 261.0, 295.0, 397.0, 567.0, 530.0, 286.0, -165.0, -55.0)
_S5 = (4459.0, 13377.0, 23094.0, 33610.0, 44925.0, 57039.0, 94522.0, 157374.0, 208740.0, 248620.0, 311412.0, 397116.0, 354217.0, 182715.0, -158340.0, -34580.0, 20475.0, 6825.0)
_S7 = (2612883.0, 7838649.0, 14431430.0, 22391226.0, 31718037.0, 42411863.0, 73712199.0, 125619045.0, 181483366.0, 241305162.0, 347939427.0, 501386161.0, 654822315.0, 808247889.0, 994730008.0, 1214268672.0, 1059854811.0, 531488425.0, -545865801.0, -82019067.0, 108754100.0, 26453700.0, -11022375.0, -3674125.0)
_S9 = (303833673.0, 911501019.0, 1750721778.0, 2821495950.0, 4123823535.0, 5657704533.0, 9934658812.0, 16954686372.0, 25519180232.0, 35628140392.0, 52815250656.0, 77080511024.0, 109082017506.0, 148819770102.0, 207345532392.0, 284659304376.0, 365217719358.0, 449020777338.0, 542906861868.0, 646875972948.0, 557256397476.0, 274048135452.0, -309179208624.0, -32603093952.0, 74929067213.0, 13417274871.0, -12661498850.0, -3307253950.0, 1027401375.0, 342467125.0)

_F_TABLES, _TW_TABLES = _time_tables(
    (_T2, _T4, _T6, _T8, _T10),
    (60.0, 1440.0, 1572480.0, 493516800.0, 3832012800.0),
)
_SIGMA_TABLES = tuple(
    tuple(v / scale for v in coef)
    for coef, scale in zip(
        (_S1, _S3, _S5, _S7, _S9),
        (7.0 * 240.0, 132.0 * 240.0, 131040.0 * 240.0, 507911040.0 * 240.0, 324635351040.0 * 240.0),
    )
)


def _stable_time_series(x: float, z: float, count: int):
    """Evaluate adaptive time/Jacobian series without temporary lists."""
    f0 = _poly_val_asc(_F_TABLES[0], x)
    f1 = _poly_val_asc(_F_TABLES[1], x)
    f2 = _poly_val_asc(_F_TABLES[2], x)
    tw0 = _poly_val_asc(_TW_TABLES[0], x)
    tw1 = _poly_val_asc(_TW_TABLES[1], x)
    tw2 = _poly_val_asc(_TW_TABLES[2], x)
    s0 = _poly_val_asc(_SIGMA_TABLES[0], x)
    s1 = _poly_val_asc(_SIGMA_TABLES[1], x)
    s2 = _poly_val_asc(_SIGMA_TABLES[2], x)

    if count == 3:
        F = math.fma(z, math.fma(z, f2, f1), f0)
        G = math.fma(z, math.fma(z, 3.0 * f2, 2.0 * f1), f0)
        TW = math.fma(z, math.fma(z, tw2, tw1), tw0)
        HS = math.fma(z, math.fma(z, s2, s1), s0)
        return F, G, TW, HS

    f3 = _poly_val_asc(_F_TABLES[3], x)
    tw3 = _poly_val_asc(_TW_TABLES[3], x)
    s3 = _poly_val_asc(_SIGMA_TABLES[3], x)

    if count == 4:
        F = math.fma(z, math.fma(z, math.fma(z, f3, f2), f1), f0)
        G = math.fma(z, math.fma(z, math.fma(z, 4.0 * f3, 3.0 * f2), 2.0 * f1), f0)
        TW = math.fma(z, math.fma(z, math.fma(z, tw3, tw2), tw1), tw0)
        HS = math.fma(z, math.fma(z, math.fma(z, s3, s2), s1), s0)
        return F, G, TW, HS

    f4 = _poly_val_asc(_F_TABLES[4], x)
    tw4 = _poly_val_asc(_TW_TABLES[4], x)
    s4 = _poly_val_asc(_SIGMA_TABLES[4], x)
    F = math.fma(z, math.fma(z, math.fma(z, math.fma(z, f4, f3), f2), f1), f0)
    G = math.fma(z, math.fma(z, math.fma(z, math.fma(z, 5.0 * f4, 4.0 * f3), 3.0 * f2), 2.0 * f1), f0)
    TW = math.fma(z, math.fma(z, math.fma(z, math.fma(z, tw4, tw3), tw2), tw1), tw0)
    HS = math.fma(z, math.fma(z, math.fma(z, math.fma(z, s4, s3), s2), s1), s0)
    return F, G, TW, HS


# Homogeneous w_sigma polynomials through phase order 12.
_WS_P0 = (3.0, 8.0, 6.0)
_WS_P1 = (45.0, 60.0, -30.0, -72.0, -40.0)
_WS_P2 = (315.0, 560.0, 630.0, 1120.0, 1344.0, 832.0, 224.0)
_WS_P3 = (70875.0, 151200.0, 255150.0, 483840.0, 672840.0, 665280.0, 468000.0, 206720.0, 41088.0)
_WS_P4 = (16372125.0, 39916800.0, 82640250.0, 175633920.0, 296049600.0, 399168000.0, 431006400.0, 355660800.0, 207981312.0, 75644928.0, 12610048.0)

_WS_P0_N = tuple(-v / 3.0 for v in _WS_P0)
_WS_P1_N = tuple(-v / 90.0 for v in _WS_P1)
_WS_P2_N = tuple(-v / 840.0 for v in _WS_P2)
_WS_P3_N = tuple(-v / 226800.0 for v in _WS_P3)
_WS_P4_N = tuple(-v / 59875200.0 for v in _WS_P4)
_WS_P0_NR = _WS_P0_N[::-1]
_WS_P1_NR = _WS_P1_N[::-1]
_WS_P2_NR = _WS_P2_N[::-1]
_WS_P3_NR = _WS_P3_N[::-1]
_WS_P4_NR = _WS_P4_N[::-1]

def _stable_order(phase: float) -> int:
    if phase < _PHASE_ORDER3_MAX:
        return 3
    if phase < _PHASE_ORDER4_MAX:
        return 4
    return 5


def _stable_w_sigma(w0: float, k0: float, ds: float, order: int) -> float:
    k_abs = abs(k0)
    a = k_abs * w0 * MU_G_INV
    d = k_abs * ds

    if a >= d:
        x = 0.0 if a == 0.0 else d / a
        beta = a * a
        p0 = _poly_val_asc(_WS_P0_N, x)
        p1 = _poly_val_asc(_WS_P1_N, x)
        p2 = _poly_val_asc(_WS_P2_N, x)
        prefactor = (k0 * MU_G_INV) * ds * ds * w0 * w0
    else:
        x = 0.0 if d == 0.0 else a / d
        beta = d * d
        p0 = _poly_val_asc(_WS_P0_NR, x)
        p1 = _poly_val_asc(_WS_P1_NR, x)
        p2 = _poly_val_asc(_WS_P2_NR, x)
        q = MU_G * ds
        prefactor = (k0 * MU_G_INV) * ds * ds * q * q

    if order == 3:
        series = math.fma(beta, math.fma(beta, p2, p1), p0)
    elif order == 4:
        p3_coef = _WS_P3_N if a >= d else _WS_P3_NR
        p3 = _poly_val_asc(p3_coef, x)
        series = math.fma(beta, math.fma(beta, math.fma(beta, p3, p2), p1), p0)
    else:
        p3_coef = _WS_P3_N if a >= d else _WS_P3_NR
        p4_coef = _WS_P4_N if a >= d else _WS_P4_NR
        p3 = _poly_val_asc(p3_coef, x)
        p4 = _poly_val_asc(p4_coef, x)
        series = math.fma(beta, math.fma(beta, math.fma(beta, math.fma(beta, p4, p3), p2), p1), p0)

    return prefactor * series


# ---------------------------------------------------------------------------
# Generic short-segment expansion
# ---------------------------------------------------------------------------


def _local_coefficients(s: float, c: float):
    """Build factored coefficients through H^9/H^10 once per segment."""
    p = s / c
    q = c / s

    p2 = p * p
    p4 = p2 * p2
    p6 = p4 * p2
    p8 = p4 * p4
    p10 = p8 * p2
    p12 = p6 * p6
    p14 = p12 * p2
    p16 = p8 * p8

    q2 = q * q
    q3 = q2 * q
    q4 = q2 * q2
    q5 = q4 * q
    q6 = q3 * q3
    q7 = q6 * q
    q8 = q4 * q4
    q9 = q8 * q

    time = (
        1.0,
        -q / 4.0,
        (2.0 * p2 + 3.0) * q2 / 24.0,
        -(14.0 * p2 + 15.0) * q3 / 192.0,
        (28.0 * p4 + 132.0 * p2 + 105.0) * q4 / 1920.0,
        -(556.0 * p4 + 1500.0 * p2 + 945.0) * q5 / 23040.0,
        (1112.0 * p6 + 10668.0 * p4 + 19950.0 * p2 + 10395.0) * q6 / 322560.0,
        -(43784.0 * p6 + 212940.0 * p4 + 304290.0 * p2 + 135135.0) * q7 / 5160960.0,
        (87568.0 * p8 + 1408992.0 * p6 + 4533480.0 * p4 + 5239080.0 * p2 + 2027025.0) * q8 / 92897280.0,
        -(5723536.0 * p8 + 43312800.0 * p6 + 103670280.0 * p4 + 100540440.0 * p2 + 34459425.0) * q9 / 1857945600.0,
    )

    t_k = (
        0.0,
        p / 4.0,
        -1.0 / 12.0,
        (14.0 * p2 + 17.0) * q / 192.0,
        -(38.0 * p2 + 39.0) * q2 / 480.0,
        (556.0 * p4 + 2276.0 * p2 + 1725.0) * q3 / 23040.0,
        -(2444.0 * p4 + 6188.0 * p2 + 3745.0) * q4 / 53760.0,
        (43784.0 * p6 + 376116.0 * p4 + 669690.0 * p2 + 337365.0) * q5 / 5160960.0,
        -(264680.0 * p6 + 1209996.0 * p4 + 1662570.0 * p2 + 717255.0) * q6 / 11612160.0,
        (5723536.0 * p8 + 84150112.0 * p6 + 258474600.0 * p4 + 289101960.0 * p2 + 109053945.0) * q7 / 1857945600.0,
    )

    w_sigma = (
        0.0,
        0.0,
        -p / 4.0,
        -(p2 + 4.0) / 12.0,
        -(p4 + p2 + 3.0) * q / 24.0,
        -(3.0 * p4 + 5.0 * p2 - 3.0) / 120.0,
        -(24.0 * p6 + 50.0 * p4 + 31.0 * p2 - 10.0) * q / 1440.0,
        -(40.0 * p6 + 98.0 * p4 + 77.0 * p2 + 26.0) / 3360.0,
        -(180.0 * p8 + 504.0 * p6 + 483.0 * p4 + 173.0 * p2 + 21.0) * q / 20160.0,
        -(1260.0 * p8 + 3960.0 * p6 + 4473.0 * p4 + 2105.0 * p2 + 323.0) / 181440.0,
        -(40320.0 * p10 + 140400.0 * p8 + 182448.0 * p6 + 106300.0 * p4 + 25261.0 * p2 + 1284.0) * q / 7257600.0,
    )

    w_sigma_extended = w_sigma + (
        -(362880.0 * p10 + 1386000.0 * p8 + 2035440.0 * p6 + 1413588.0 * p4 + 450461.0 * p2 + 49248.0) / 79833600.0,
        -(1814400.0 * p12 + 7539840.0 * p10 + 12343320.0 * p8 + 9951216.0 * p6 + 3984431.0 * p4 + 675691.0 * p2 + 24629.0) * q / 479001600.0,
        -(6652800.0 * p12 + 29877120.0 * p10 + 53933880.0 * p8 + 49500880.0 * p6 + 23886863.0 * p4 + 5537805.0 * p2 + 442249.0) / 2075673600.0,
        -(479001600.0 * p14 + 2311545600.0 * p12 + 4560716160.0 * p10 + 4694461200.0 * p8 + 2652321672.0 * p6 + 783059550.0 * p4 + 99680491.0 * p2 + 2653482.0) * q / 174356582400.0,
    )

    t_sigma = (
        p / 6.0,
        (2.0 * p2 - 1.0) / 48.0,
        (4.0 * p4 + 10.0 * p2 + 9.0) * q / 240.0,
        (24.0 * p6 + 40.0 * p4 - 84.0 * p2 - 105.0) * q2 / 2880.0,
        (384.0 * p8 + 656.0 * p6 + 796.0 * p4 + 3284.0 * p2 + 2775.0) * q3 / 80640.0,
        (1280.0 * p10 + 2464.0 * p8 + 1624.0 * p6 - 6812.0 * p4 - 21210.0 * p2 - 13965.0) * q4 / 430080.0,
        (11520.0 * p12 + 25344.0 * p10 + 17808.0 * p8 + 14600.0 * p6 + 162768.0 * p4 + 330750.0 * p2 + 178605.0) * q5 / 5806080.0,
        (161280.0 * p14 + 403200.0 * p12 + 334944.0 * p10 + 118240.0 * p8 - 840296.0 * p6 - 4892796.0 * p4 - 7426440.0 * p2 - 3399165.0) * q6 / 116121600.0,
        (10321920.0 * p16 + 29030400.0 * p14 + 28459008.0 * p12 + 11270272.0 * p10 + 7142416.0 * p8 + 165796032.0 * p6 + 593976600.0 * p4 + 719613720.0 * p2 + 285810525.0) * q7 / 10218700800.0,
    )

    t_sigma_extended = t_sigma + (
        (185794560.0 * p**18 + 581898240.0 * p16 + 661985280.0 * p14 + 323752704.0 * p12 + 70636192.0 * p10 - 764073744.0 * p8 - 7204438032.0 * p6 - 18602254440.0 * p4 - 18752351310.0 * p2 - 6577696125.0) * q8 / 245248819200.0,
        (1857945600.0 * p**20 + 6420234240.0 * p**18 + 8336148480.0 * p16 + 4935929856.0 * p14 + 1274406848.0 * p12 + 665772640.0 * p10 + 27349293776.0 * p8 + 150588364080.0 * p6 + 303441650820.0 * p4 + 261903792150.0 * p2 + 82254647475.0) * q9 / 3188234649600.0,
    )

    return time, t_k, w_sigma, t_sigma, w_sigma_extended, t_sigma_extended


# ---------------------------------------------------------------------------
# Common initial data and dispatch
# ---------------------------------------------------------------------------


def circular_use_stable(L: float, w0: float, k0: float) -> bool:
    if k0 == 0.0:
        return True
    phase = abs(k0) * math.fma(2.0 * MU_G, L, w0) * MU_G_INV
    return phase <= CIRCULAR_STABLE_PHASE_MAX


def make_circular_eval_segment(L: float, w0: float, k0: float):
    """Construct the numerically appropriate evaluation-only segment."""
    cls = (
        CircularEvalSegmentStable
        if circular_use_stable(L, w0, k0)
        else CircularEvalSegment
    )
    return cls(L, w0, k0)


def make_circular_segment(L: float, w0: float, k0: float):
    """Construct the numerically appropriate differentiable segment."""
    cls = (
        CircularSegmentStable
        if circular_use_stable(L, w0, k0)
        else CircularSegment
    )
    return cls(L, w0, k0)


class _CircularCommon:
    def _init_circular(self, L: float, w0: float, k0: float) -> None:
        if w0 < 0.0:
            raise ValueError("circular segment requires w0 >= 0")

        self.w0 = w0
        self.k0 = k0
        self.k_abs = abs(k0)
        self.eps = math.copysign(1.0, k0)
        self.sqrt_w0 = math.sqrt(w0)

        if k0 == 0.0:
            self.s0 = 0.0
            self.c0 = 1.0
            self.x0 = 0.0
            self.small_ds_max = math.inf
            self.small_time_ds_max = math.inf
            self.small_jac_ds_max = math.inf
            self.small_sigma_ds_max = math.inf
            self.local_coefficients = None
            return

        s0 = self.k_abs * w0 * MU_G_INV
        if s0 > 1.0:
            raise ValueError("initial circular state exceeds the grip boundary")

        self.s0 = s0
        self.c0 = math.sqrt(max(0.0, math.fma(-s0, s0, 1.0)))
        self.x0 = math.atan2(s0, self.c0)

        # The closed form w=(MU_G/|k|) sin(x0+2|k|s) represents the
        # positive-square-root GRIP branch only until its first maximum.
        # Continuing the sine beyond x=pi/2 silently switches to the negative
        # square-root branch.  Because G² has a tangential zero there, a finite
        # uniform domain scan can miss the endpoint entirely and later observe
        # an apparently in-domain but physically invalid continuation.
        self.domain_end = (0.5 * math.pi - self.x0) / (2.0 * self.k_abs)
        endpoint_tolerance = max(
            128.0 * math.ulp(max(abs(self.domain_end), abs(L), 1.0)),
            1.0e-14 * max(1.0, abs(L)),
        )
        if L > self.domain_end + endpoint_tolerance:
            raise ArithmeticError(
                "circular grip segment reaches its physical friction endpoint "
                f"before the requested end: t={self.domain_end:.17g}, "
                f"t_end={L:.17g}"
            )

        if 0.0 < s0 < 1.0:
            rho_per_ds = 2.0 * self.k_abs * max(1.0, self.c0 / s0, s0 / self.c0)
            self.small_ds_max = CIRCULAR_SMALL_STEP_TOL / rho_per_ds
            self.small_time_ds_max = CIRCULAR_SMALL_TIME_TOL / rho_per_ds
            self.small_jac_ds_max = CIRCULAR_SMALL_JAC_TOL / rho_per_ds
            self.small_sigma_ds_max = CIRCULAR_SMALL_SIGMA_TOL / rho_per_ds
            self.local_coefficients = _local_coefficients(s0, self.c0)
        else:
            self.small_ds_max = 0.0
            self.small_time_ds_max = 0.0
            self.small_jac_ds_max = 0.0
            self.small_sigma_ds_max = 0.0
            self.local_coefficients = None


class _CircularStableCommon(_CircularCommon):
    def _init_stable(self, L: float, w0: float, k0: float) -> None:
        self._init_circular(L, w0, k0)
        phase_max = 0.0 if k0 == 0.0 else self.k_abs * math.fma(2.0 * MU_G, L, w0) * MU_G_INV
        if phase_max > CIRCULAR_STABLE_PHASE_MAX:
            raise ValueError("stable circular segment exceeds CIRCULAR_STABLE_PHASE_MAX")
        self.series_order = _stable_order(phase_max)

    def _stable_state(self, ds: float):
        h = 2.0 * self.k_abs * ds
        sin_h, cos_h, sinc_h, sinc_prime_h, _ = _circular_trig_small(h)
        w = math.fma(2.0 * MU_G * self.c0 * ds, sinc_h, self.w0 * cos_h)
        return h, sin_h, cos_h, sinc_h, sinc_prime_h, w


# ---------------------------------------------------------------------------
# Stable classes
# ---------------------------------------------------------------------------


class CircularEvalSegmentStable(_CircularStableCommon, EvalSegment):
    def __init__(self, L, w0, k0):
        super().__init__(L, 0.0, w0, k0, SegmentType.GRIP)
        self._init_stable(L, w0, k0)

    def w(self, ds: float) -> float:
        if ds == 0.0:
            return self.w0
        if self.k0 == 0.0:
            return math.fma(2.0 * MU_G, ds, self.w0)
        return self._stable_state(ds)[-1]


class CircularSegmentStable(_CircularStableCommon, DiffSegment):
    def __init__(self, L, w0, k0):
        super().__init__(L, 0.0, w0, k0, SegmentType.GRIP)
        self._init_stable(L, w0, k0)

    def w(self, ds: float) -> float:
        if ds == 0.0:
            return self.w0
        if self.k0 == 0.0:
            return math.fma(2.0 * MU_G, ds, self.w0)
        return self._stable_state(ds)[-1]

    def _w_jac_from_state(self, ds: float, state):
        h, sin_h, cos_h, _, sinc_prime_h, w = state
        w_s = 2.0 * MU_G * math.fma(self.c0, cos_h, -self.s0 * sin_h)
        w_w0 = math.fma(-self.s0 / self.c0, sin_h, cos_h)
        w_k0 = self.eps * math.fma(
            4.0 * MU_G * self.c0 * ds * ds,
            sinc_prime_h,
            -self.w0 * sin_h * (self.w0 / (MU_G * self.c0) + 2.0 * ds),
        )
        w_sigma = _stable_w_sigma(
            self.w0,
            self.k0,
            ds,
            _stable_order(self.s0 + h),
        )
        return w, (w_s, w_sigma, w_w0, w_k0, 0.0, ds, 0.0, 1.0)

    def _time_jac_from_w1(self, ds: float, w1: float):
        r2 = math.fma(2.0 * MU_G, ds, self.w0)
        root = math.sqrt(r2)
        d_invM = (2.0 * ds) / (root + self.sqrt_w0)
        d = MU_G * d_invM
        x = 0.0 if root == 0.0 else self.sqrt_w0 / root
        delta = 0.0 if root == 0.0 else d / root
        eta = self.k_abs * r2 * MU_G_INV
        z = eta * eta

        F, G, TW, HS = _stable_time_series(x, z, _stable_order(eta))
        base = math.fma(delta * z, F, 1.0)
        time = d_invM * base
        t_s = 1.0 / math.sqrt(w1)
        t_k0 = (2.0 / self.k0) * d_invM * delta * z * G
        t_sigma = MU_G * d_invM * d_invM * d_invM * z * HS / self.k0

        if self.sqrt_w0 == 0.0:
            t_w0 = -math.inf
        else:
            tw_bracket = math.fma(z, TW, -0.5)
            t_w0 = delta * tw_bracket / (MU_G * self.sqrt_w0)

        return time, (t_s, t_sigma, t_w0, t_k0)

    def w_and_jac(self, ds: float):
        if ds == 0.0:
            return self.w0, (
                2.0 * MU_G * self.c0,
                0.0,
                1.0,
                0.0,
                0.0,
                ds,
                0.0,
                1.0,
            )

        if self.k0 == 0.0:
            w = math.fma(2.0 * MU_G, ds, self.w0)
            return w, (2.0 * MU_G, 0.0, 1.0, 0.0, 0.0, ds, 0.0, 1.0)

        return self._w_jac_from_state(ds, self._stable_state(ds))

    def time_and_jac(self, ds: float):
        if ds == 0.0:
            return 0.0, (
                math.inf if self.w0 == 0.0 else 1.0 / self.sqrt_w0,
                0.0,
                0.0,
                0.0,
            )

        if self.k0 == 0.0:
            r2 = math.fma(2.0 * MU_G, ds, self.w0)
            root = math.sqrt(r2)
            d_invM = (2.0 * ds) / (root + self.sqrt_w0)
            t_w0 = (
                -math.inf
                if self.w0 == 0.0
                else 0.5 * MU_G_INV * (1.0 / root - 1.0 / self.sqrt_w0)
            )
            return d_invM, (1.0 / root, 0.0, t_w0, 0.0)

        state = self._stable_state(ds)
        return self._time_jac_from_w1(ds, state[-1])

    def state_time_and_jac(self, ds: float):
        """Combined stable evaluator; the joint trig/state kernel runs once."""
        if ds == 0.0:
            return (
                self.w0,
                (2.0 * MU_G * self.c0, 0.0, 1.0, 0.0, 0.0, ds, 0.0, 1.0),
                0.0,
                (
                    math.inf if self.w0 == 0.0 else 1.0 / self.sqrt_w0,
                    0.0,
                    0.0,
                    0.0,
                ),
            )

        if self.k0 == 0.0:
            w = math.fma(2.0 * MU_G, ds, self.w0)
            root = math.sqrt(w)
            time = (2.0 * ds) / (root + self.sqrt_w0)
            t_w0 = (
                -math.inf
                if self.w0 == 0.0
                else 0.5 * MU_G_INV * (1.0 / root - 1.0 / self.sqrt_w0)
            )
            return (
                w,
                (2.0 * MU_G, 0.0, 1.0, 0.0, 0.0, ds, 0.0, 1.0),
                time,
                (1.0 / root, 0.0, t_w0, 0.0),
            )

        state = self._stable_state(ds)
        w, wjac = self._w_jac_from_state(ds, state)
        time, tjac = self._time_jac_from_w1(ds, w)
        return w, wjac, time, tjac


# ---------------------------------------------------------------------------
# Generic exact classes with local-ds overlap
# ---------------------------------------------------------------------------


class CircularEvalSegment(_CircularCommon, EvalSegment):
    def __init__(self, L, w0, k0):
        super().__init__(L, 0.0, w0, k0, SegmentType.GRIP)
        self._init_circular(L, w0, k0)

    def w(self, ds: float) -> float:
        if ds == 0.0:
            return self.w0
        if self.k0 == 0.0:
            return math.fma(2.0 * MU_G, ds, self.w0)

        h = 2.0 * self.k_abs * ds
        if self.local_coefficients is not None and abs(ds) <= self.small_ds_max:
            sin_h, cos_h, _, _, _ = _circular_trig_small(h)
            g = math.fma(self.c0 / self.s0, sin_h, cos_h)
            return self.w0 * g

        x1 = self.x0 + h
        return (MU_G / self.k_abs) * math.sin(x1)


class CircularSegment(_CircularCommon, DiffSegment):
    def __init__(self, L, w0, k0):
        super().__init__(L, 0.0, w0, k0, SegmentType.GRIP)
        self._init_circular(L, w0, k0)

        if k0 != 0.0 and self.c0 == 0.0:
            raise ValueError(
                "circular Jacobians are singular at the exact grip boundary; "
                "use CircularEvalSegment for state-only evaluation"
            )

        if k0 != 0.0:
            self.J0, self.Q0 = _J_Q(self.x0)
            self.Ht0 = _H_tilde_from_JQ(self.x0, self.s0, self.J0, self.Q0)
            self.logm0 = math.log1p(-self.s0)
            self.logp0 = math.log1p(self.s0)
            self.time_scale = 0.5 / math.sqrt(MU_G * self.k_abs)
            self.sigma_time_scale = (
                0.125
                * self.eps
                / (self.k_abs * self.k_abs * math.sqrt(MU_G * self.k_abs))
            )

    def w(self, ds: float) -> float:
        if ds == 0.0:
            return self.w0
        if self.k0 == 0.0:
            return math.fma(2.0 * MU_G, ds, self.w0)

        h = 2.0 * self.k_abs * ds
        if self.local_coefficients is not None and abs(ds) <= self.small_ds_max:
            sin_h, cos_h, _, _, _ = _circular_trig_small(h)
            return self.w0 * math.fma(self.c0 / self.s0, sin_h, cos_h)

        return (MU_G / self.k_abs) * math.sin(self.x0 + h)

    def _phase_state(self, ds: float):
        """Positive-phase state without re-evaluating sin(x0+h)."""
        h = 2.0 * self.k_abs * ds
        sin_h = math.sin(h)
        cos_h = math.cos(h)
        sin1 = math.fma(self.c0, sin_h, self.s0 * cos_h)
        cos1 = math.fma(self.c0, cos_h, -self.s0 * sin_h)
        return h, self.x0 + h, sin1, cos1

    def _local_all(self, ds: float):
        time_coef, tk_coef, ws_coef, ts_coef, _, _ = self.local_coefficients
        h = 2.0 * self.k_abs * ds
        sin_h, cos_h, _, sinc_prime_h, cosm1_h = _circular_trig_small(h)

        cot0 = self.c0 / self.s0
        tan0 = self.s0 / self.c0
        g = math.fma(cot0, sin_h, cos_h)
        w = self.w0 * g

        w_s = 2.0 * MU_G * math.fma(self.c0, cos_h, -self.s0 * sin_h)
        w_w0 = math.fma(-tan0, sin_h, cos_h)
        w_k0 = self.eps * math.fma(
            4.0 * MU_G * self.c0 * ds * ds,
            sinc_prime_h,
            -self.w0 * sin_h * (self.w0 / (MU_G * self.c0) + 2.0 * ds),
        )
        w_sigma = (
            self.eps
            * self.w0
            / (self.k_abs * self.k_abs)
            * _poly_val_asc(ws_coef, h)
        )

        sqrt_g = math.sqrt(g)
        r1 = 1.0 / (self.sqrt_w0 * sqrt_g)
        time = (ds / self.sqrt_w0) * _poly_val_asc(time_coef, h)

        gm1 = math.fma(cot0, sin_h, cosm1_h)
        invsqrt_gm1 = -gm1 / (sqrt_g * (1.0 + sqrt_g))
        t_w0 = 0.5 * invsqrt_gm1 / (MU_G * self.c0 * self.sqrt_w0)
        t_k0 = (ds / (self.k0 * self.sqrt_w0)) * _poly_val_asc(tk_coef, h)
        t_sigma = (
            self.eps
            * ds
            * ds
            * ds
            / self.sqrt_w0
            * _poly_val_asc(ts_coef, h)
        )

        return w, (w_s, w_sigma, w_w0, w_k0), time, (r1, t_sigma, t_w0, t_k0)

    def _local_sigma(self, ds: float) -> tuple[float, float]:
        """Only the cancellation-sensitive sigma Jacobians."""
        _, _, _, _, ws_coef, ts_coef = self.local_coefficients
        h = 2.0 * self.k_abs * ds
        w_sigma = (
            self.eps
            * self.w0
            / (self.k_abs * self.k_abs)
            * _poly_val_asc(ws_coef, h)
        )
        t_sigma = (
            self.eps
            * ds
            * ds
            * ds
            / self.sqrt_w0
            * _poly_val_asc(ts_coef, h)
        )
        return w_sigma, t_sigma

    def _exact_state_jac(
        self,
        ds: float,
        h: float,
        sin1: float,
        cos1: float,
        local_sigma: float | None,
    ):
        w = (MU_G / self.k_abs) * sin1
        w_s = 2.0 * MU_G * cos1
        w_w0 = cos1 / self.c0
        w_k0 = self.eps * (MU_G / self.k_abs) * math.fma(
            cos1,
            self.w0 / (MU_G * self.c0) + 2.0 * ds,
            -sin1 / self.k_abs,
        )

        if local_sigma is None:
            logm1 = math.log1p(-sin1)
            logp1 = math.log1p(sin1)
            log_cos_ratio = 0.5 * (
                (logm1 - self.logm0) + (logp1 - self.logp0)
            )
            c1 = math.fma(-0.5, h * h, log_cos_ratio)
            w_sigma = (
                -0.5
                * self.eps
                * MU_G
                / (self.k_abs ** 3)
                * math.fma(cos1, c1, h * sin1)
            )
        else:
            w_sigma = local_sigma

        return w, (w_s, w_sigma, w_w0, w_k0)

    def w_and_jac(self, ds: float):
        if ds == 0.0:
            return self.w0, (
                2.0 * MU_G * self.c0,
                0.0,
                1.0,
                0.0,
                0.0,
                ds,
                0.0,
                1.0,
            )

        if self.k0 == 0.0:
            w = math.fma(2.0 * MU_G, ds, self.w0)
            return w, (2.0 * MU_G, 0.0, 1.0, 0.0, 0.0, ds, 0.0, 1.0)

        if self.local_coefficients is not None and abs(ds) <= self.small_jac_ds_max:
            w, jac, _, _ = self._local_all(ds)
            return w, (*jac, 0.0, ds, 0.0, 1.0)

        h, _, sin1, cos1 = self._phase_state(ds)
        local_sigma = None
        if self.local_coefficients is not None and abs(ds) <= self.small_sigma_ds_max:
            local_sigma, _ = self._local_sigma(ds)

        w, jac = self._exact_state_jac(ds, h, sin1, cos1, local_sigma)
        return w, (*jac, 0.0, ds, 0.0, 1.0)

    def _exact_time_jac(
        self,
        ds: float,
        h: float,
        x1: float,
        sin1: float,
        w1: float,
        local_jac: tuple[float, float, float, float] | None,
        local_sigma: float | None,
    ):
        r1 = 1.0 / math.sqrt(w1)

        if local_sigma is None:
            j1, q1 = _J_Q(x1)
            ht1 = _H_tilde_from_JQ(x1, sin1, j1, q1)
            logm1 = math.log1p(-sin1)
            logp1 = math.log1p(sin1)
            log_cos_ratio = 0.5 * (
                (logm1 - self.logm0) + (logp1 - self.logp0)
            )
            c1 = math.fma(-0.5, h * h, log_cos_ratio)
            hd = math.fma(
                -h,
                j1,
                (self.Ht0 - ht1) + (logm1 - self.logm0),
            )
            t_sigma = self.sigma_time_scale * math.fma(
                -2.0 / math.sqrt(sin1),
                c1,
                hd,
            )
        else:
            j1 = _J(x1)
            t_sigma = local_sigma

        time = self.time_scale * (j1 - self.J0)

        if self.w0 == 0.0:
            t_w0 = -math.inf
            t_k0 = math.fma(ds, r1, -0.5 * time) / self.k0
        else:
            r0 = 1.0 / self.sqrt_w0
            t_w0 = 0.5 * (r1 - r0) / (MU_G * self.c0)
            t_k0 = (
                math.fma(self.w0, t_w0, math.fma(ds, r1, -0.5 * time))
                / self.k0
            )

        if local_jac is not None:
            t_sigma = local_jac[1]
            t_w0 = local_jac[2]
            t_k0 = local_jac[3]

        return time, (r1, t_sigma, t_w0, t_k0)

    def time_and_jac(self, ds: float):
        if ds == 0.0:
            return 0.0, (
                math.inf if self.w0 == 0.0 else 1.0 / self.sqrt_w0,
                0.0,
                0.0,
                0.0,
            )

        if self.k0 == 0.0:
            r2 = math.fma(2.0 * MU_G, ds, self.w0)
            root = math.sqrt(r2)
            time = (2.0 * ds) / (root + self.sqrt_w0)
            t_w0 = (
                -math.inf
                if self.w0 == 0.0
                else 0.5 * MU_G_INV * (1.0 / root - 1.0 / self.sqrt_w0)
            )
            return time, (1.0 / root, 0.0, t_w0, 0.0)

        if self.local_coefficients is not None and abs(ds) <= self.small_ds_max:
            _, _, time, jac = self._local_all(ds)
            return time, jac

        local_jac = None
        if self.local_coefficients is not None and abs(ds) <= self.small_jac_ds_max:
            _, _, local_time, local_jac = self._local_all(ds)
            if abs(ds) <= self.small_time_ds_max:
                return local_time, local_jac

        local_sigma = None
        if self.local_coefficients is not None and abs(ds) <= self.small_sigma_ds_max:
            _, local_sigma = self._local_sigma(ds)

        h, x1, sin1, _ = self._phase_state(ds)
        w1 = (MU_G / self.k_abs) * sin1
        return self._exact_time_jac(
            ds,
            h,
            x1,
            sin1,
            w1,
            local_jac,
            local_sigma,
        )

    def state_time_and_jac(self, ds: float):
        """Combined evaluator that shares phase, trig, and exact integrals."""
        if ds == 0.0:
            wjac = (
                2.0 * MU_G * self.c0,
                0.0,
                1.0,
                0.0,
                0.0,
                ds,
                0.0,
                1.0,
            )
            tjac = (
                math.inf if self.w0 == 0.0 else 1.0 / self.sqrt_w0,
                0.0,
                0.0,
                0.0,
            )
            return self.w0, wjac, 0.0, tjac

        if self.k0 == 0.0:
            w = math.fma(2.0 * MU_G, ds, self.w0)
            root = math.sqrt(w)
            time = (2.0 * ds) / (root + self.sqrt_w0)
            t_w0 = (
                -math.inf
                if self.w0 == 0.0
                else 0.5 * MU_G_INV * (1.0 / root - 1.0 / self.sqrt_w0)
            )
            return (
                w,
                (2.0 * MU_G, 0.0, 1.0, 0.0, 0.0, ds, 0.0, 1.0),
                time,
                (1.0 / root, 0.0, t_w0, 0.0),
            )

        if self.local_coefficients is not None and abs(ds) <= self.small_time_ds_max:
            w, wjac4, time, tjac = self._local_all(ds)
            return w, (*wjac4, 0.0, ds, 0.0, 1.0), time, tjac

        local_jac = None
        local_time = None
        if self.local_coefficients is not None and abs(ds) <= self.small_jac_ds_max:
            _, wjac4, local_time, local_jac = self._local_all(ds)

        local_w_sigma = None
        local_t_sigma = None
        if self.local_coefficients is not None and abs(ds) <= self.small_sigma_ds_max:
            local_w_sigma, local_t_sigma = self._local_sigma(ds)

        h, x1, sin1, cos1 = self._phase_state(ds)
        w, exact_wjac4 = self._exact_state_jac(
            ds,
            h,
            sin1,
            cos1,
            local_w_sigma,
        )
        if local_jac is None:
            wjac4 = exact_wjac4

        time, tjac = self._exact_time_jac(
            ds,
            h,
            x1,
            sin1,
            w,
            local_jac,
            local_t_sigma,
        )
        if local_time is not None and abs(ds) <= self.small_time_ds_max:
            time = local_time

        return w, (*wjac4, 0.0, ds, 0.0, 1.0), time, tjac