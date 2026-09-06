import math
from typing import Final, Tuple

Point = Tuple[float, float]
EndState = Tuple[float, float, float, float]

_SQRT_PI: Final = math.sqrt(math.pi)
_PI_OVER_2_HI: Final = math.pi / 2.0
# Exact pi/2 minus its nearest binary64 number, rounded to binary64.
_PI_OVER_2_LO: Final = 6.12323399573676603587e-17

# The Fresnel tail is below half an ulp of 0.5 beyond this value.
_FRESNEL_ROUND_TO_HALF: Final = float(2**53)
# Compensated phase multiplication becomes worthwhile above this point.
_FRESNEL_DD_PHASE_THRESHOLD: Final = 32.0

# Hybrid Euler evaluator thresholds.
_GL12_BETA_MAX: Final = 0.25
_GL14_BETA_MAX: Final = 0.5
_GL16_BETA_MAX: Final = 1.0
_ASYM_MIN_ENDPOINT_PHASE: Final = 8.0
_ASYM_MAX_RATIO: Final = 0.01
_ASYM_TERM_TOL: Final = 1.0e-19

# Cephes double-precision Fresnel coefficients.
_SN: Final = (
    -2.99181919401019853726e3,
    7.08840045257738576863e5,
    -6.29741486205862506537e7,
    2.54890880573376359104e9,
    -4.42979518059697779103e10,
    3.18016297876567817986e11,
)
_SD: Final = (
    2.81376268889994315696e2,
    4.55847810806532581675e4,
    5.17343888770096400730e6,
    4.19320245898111231129e8,
    2.24411795645340920940e10,
    6.07366389490084639049e11,
)
_CN: Final = (
    -4.98843114573573548651e-8,
    9.50428062829859605134e-6,
    -6.45191435683965050962e-4,
    1.88843319396703850064e-2,
    -2.05525900955013891793e-1,
    9.99999999999999998822e-1,
)
_CD: Final = (
    3.99982968972495980367e-12,
    9.15439215774657478799e-10,
    1.25001862479598821474e-7,
    1.22262789024179030997e-5,
    8.68029542941784300606e-4,
    4.12142090722199792936e-2,
    1.00000000000000000118,
)
_FN: Final = (
    4.21543555043677546506e-1,
    1.43407919780758885261e-1,
    1.15220955073585758835e-2,
    3.45017939782574027900e-4,
    4.63613749287867322088e-6,
    3.05568983790257605827e-8,
    1.02304514164907233465e-10,
    1.72010743268161828879e-13,
    1.34283276233062758925e-16,
    3.76329711269987889006e-20,
)
_FD: Final = (
    7.51586398353378947175e-1,
    1.16888925859191382142e-1,
    6.44051526508858611005e-3,
    1.55934409164153020873e-4,
    1.84627567348930545870e-6,
    1.12699224763999035261e-8,
    3.60140029589371370404e-11,
    5.88754533621578410010e-14,
    4.52001434074129701496e-17,
    1.25443237090011264384e-20,
)
_GN: Final = (
    5.04442073643383265887e-1,
    1.97102833525523411709e-1,
    1.87648584092575249293e-2,
    6.84079380915393090172e-4,
    1.15138826111884280931e-5,
    9.82852443688422223854e-8,
    4.45344415861750144738e-10,
    1.08268041139020870318e-12,
    1.37555460633261799868e-15,
    8.36354435630677421531e-19,
    1.86958710162783235106e-22,
)
_GD: Final = (
    1.47495759925128324529,
    3.37748989120019970451e-1,
    2.53603741420338795122e-2,
    8.14679107184306179049e-4,
    1.27545075667729118702e-5,
    1.04314589657571990585e-7,
    4.60680728146520428211e-10,
    1.10273215066240270757e-12,
    1.38796531259578871258e-15,
    8.39158816283118707363e-19,
    1.86958710162783236342e-22,
)

# Positive midpoint offsets and [0, 1] weights. Symmetry halves the nodes.
_GL12: Final = (
    (6.261670425573440e-2, 1.2457352290670134e-1),
    (1.8391574949909006e-1, 1.1674626826917730e-1),
    (2.9365897714330870e-1, 1.0158371336153287e-1),
    (3.8495133709715235e-1, 8.003916427167321e-2),
    (4.5205862818523745e-1, 5.346966299765953e-2),
    (4.9078031712335957e-1, 2.3587668193255706e-2),
)
_GL14: Final = (
    (5.4027474353671834e-2, 1.0763192673157897e-1),
    (1.5955618446394482e-1, 1.0259923186064786e-1),
    (2.5762431817907705e-1, 9.276919873896894e-2),
    (3.4364645240584270e-1, 7.860158357909681e-2),
    (4.1360065753488250e-1, 6.0759285343951655e-2),
    (4.6421744183178680e-1, 4.007904357987999e-2),
    (4.9314190434840620e-1, 1.7559730165875875e-2),
)
_GL16: Final = (
    (4.750625491881877e-2, 9.472530522753432e-2),
    (1.4080177538962946e-1, 9.130170752246182e-2),
    (2.2900838882861363e-1, 8.457825969750132e-2),
    (3.0893812220132190e-1, 7.479799440828835e-2),
    (3.7770220417750155e-1, 6.231448562776704e-2),
    (4.3281560119391593e-1, 4.757925584124630e-2),
    (4.7228751153661630e-1, 3.112676196932373e-2),
    (4.9470046749582497e-1, 1.3576229705877088e-2),
)


def _horner6(x: float, c: tuple[float, ...]) -> float:
    return (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5])


def _horner7(x: float, c: tuple[float, ...]) -> float:
    return ((((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) * x + c[6])


def _p1_horner6(x: float, c: tuple[float, ...]) -> float:
    return ((((((x + c[0]) * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5])


def _horner10(x: float, c: tuple[float, ...]) -> float:
    return (((((((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) * x + c[6]) * x + c[7]) * x + c[8]) * x + c[9])


def _p1_horner10(x: float, c: tuple[float, ...]) -> float:
    return ((((((((((x + c[0]) * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) * x + c[6]) * x + c[7]) * x + c[8]) * x + c[9])


def _horner11(x: float, c: tuple[float, ...]) -> float:
    return ((((((((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) * x + c[6]) * x + c[7]) * x + c[8]) * x + c[9]) * x + c[10])


def _p1_horner11(x: float, c: tuple[float, ...]) -> float:
    return (((((((((((x + c[0]) * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) * x + c[6]) * x + c[7]) * x + c[8]) * x + c[9]) * x + c[10])


def _sinc(x: float) -> float:
    ax = abs(x)
    if ax < 1.0e-4:
        x2 = x * x
        return 1.0 + x2 * (
            -1.0 / 6.0
            + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 / 362880.0))
        )
    return math.sin(x) / x


def _phase_sum_cs(a: float, b: float) -> tuple[float, float]:
    """Return cosine and sine of ``a + b`` while preserving a small ``b``."""
    ca = math.cos(a)
    sa = math.sin(a)
    cb = math.cos(b)
    sb = math.sin(b)
    return (
        math.fma(ca, cb, -sa * sb),
        math.fma(sa, cb, ca * sb),
    )


def _phase_ratio_square_cs(
    x: float,
    denominator: float,
    factor: float,
) -> tuple[float, float]:
    """Compensated cosine and sine of ``factor*x**2/denominator``."""
    q_hi = x / denominator
    if not math.isfinite(q_hi):
        raise OverflowError("completed-square phase is not representable")
    q_lo = math.fma(-q_hi, denominator, x) / denominator
    product_hi = x * q_hi
    product_lo = math.fma(x, q_hi, -product_hi) + x * q_lo
    return _phase_sum_cs(factor * product_hi, factor * product_lo)


def _rotate_xy(
    x: float,
    y: float,
    cosine: float,
    sine: float,
) -> tuple[float, float]:
    """Rotate an x/y vector by an angle represented by cosine and sine."""
    return (
        math.fma(cosine, x, -sine * y),
        math.fma(sine, x, cosine * y),
    )

def fresnel(xxa: float) -> Tuple[float, float]:
    """Return the scalar Fresnel integrals ``(S(x), C(x))``."""
    if math.isnan(xxa):
        return math.nan, math.nan
    if xxa == 0.0:
        return xxa, xxa

    x = abs(xxa)
    if math.isinf(x) or x >= _FRESNEL_ROUND_TO_HALF:
        ss = 0.5
        cc = 0.5
    else:
        x2 = x * x
        if x2 < 2.5625:
            t = x2 * x2
            ss = x * x2 * _horner6(t, _SN) / _p1_horner6(t, _SD)
            cc = x * _horner6(t, _CN) / _horner7(t, _CD)
        else:
            t = math.pi * x2
            u = 1.0 / (t * t)
            f = 1.0 - u * _horner10(u, _FN) / _p1_horner10(u, _FD)
            g = (1.0 / t) * _horner11(u, _GN) / _p1_horner11(u, _GD)

            if x < _FRESNEL_DD_PHASE_THRESHOLD:
                phase = 0.5 * math.pi * x2
                c = math.cos(phase)
                s = math.sin(phase)
            else:
                # Recover the low product bits and the low bits of pi/2.
                square_hi = x * x
                square_lo = math.fma(x, x, -square_hi)
                phase_hi = _PI_OVER_2_HI * square_hi
                phase_lo = (
                    math.fma(_PI_OVER_2_HI, square_hi, -phase_hi)
                    + _PI_OVER_2_HI * square_lo
                    + _PI_OVER_2_LO * square_hi
                )
                c, s = _phase_sum_cs(phase_hi, phase_lo)

            inv = 1.0 / (math.pi * x)
            cc = 0.5 + (f * s - g * c) * inv
            ss = 0.5 - (f * c + g * s) * inv

    if xxa < 0.0:
        return -ss, -cc
    return ss, cc


def _quadratic_phase_gauss(
    alpha: float,
    beta: float,
    pairs: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    """Return longitudinal and lateral normalized displacements."""
    center = 0.5 * alpha + 0.25 * beta
    midpoint_slope = alpha + beta
    along = 0.0
    lateral = 0.0

    for d, weight in pairs:
        phase = center + beta * d * d
        pair_weight = 2.0 * weight * math.cos(midpoint_slope * d)
        along += pair_weight * math.cos(phase)
        lateral += pair_weight * math.sin(phase)

    return along, lateral


def _asymptotic_antiderivative(
    curvature: float,
    sigma: float,
) -> tuple[float, float]:
    """Return endpoint antiderivative coefficients in x/y components."""
    term_x = 0.0
    term_y = -1.0 / curvature
    total_x = term_x
    total_y = term_y
    best_x = total_x
    best_y = total_y
    best_magnitude = abs(term_y)
    previous_magnitude = best_magnitude
    curvature2 = curvature * curvature

    for n in range(1, 64):
        factor = (2 * n - 1) * sigma / curvature2

        # Apply a quarter-turn and scale by ``factor``.
        new_x = factor * term_y
        new_y = -factor * term_x
        term_x = new_x
        term_y = new_y

        magnitude = math.hypot(term_x, term_y)
        candidate_x = total_x + term_x
        candidate_y = total_y + term_y

        if magnitude < best_magnitude:
            best_x = candidate_x
            best_y = candidate_y
            best_magnitude = magnitude

        total_x = candidate_x
        total_y = candidate_y

        if magnitude <= _ASYM_TERM_TOL:
            return total_x, total_y
        if magnitude > previous_magnitude:
            return best_x, best_y

        previous_magnitude = magnitude

    return best_x, best_y


class Geometry:
    __slots__ = ("x0", "y0", "th0", "k0", "sigma")

    def __init__(self, x0: float, y0: float, th0: float, k0: float, sigma: float):
        if not all(math.isfinite(v) for v in (x0, y0, th0, k0, sigma)):
            raise ValueError("geometry parameters must be finite")
        self.x0 = x0
        self.y0 = y0
        self.th0 = th0
        self.k0 = k0
        self.sigma = sigma

    def xy(self, ds: float) -> Point:
        raise NotImplementedError

    def end(self, length: float) -> EndState:
        raise NotImplementedError


class Line(Geometry):
    __slots__ = ("_sin0", "_cos0")

    def __init__(self, x0: float, y0: float, th0: float):
        super().__init__(x0, y0, th0, 0.0, 0.0)
        self._sin0 = math.sin(th0)
        self._cos0 = math.cos(th0)

    def xy(self, ds: float) -> Point:
        return (
            math.fma(ds, self._cos0, self.x0),
            math.fma(ds, self._sin0, self.y0),
        )

    def end(self, length: float) -> EndState:
        x1, y1 = self.xy(length)
        return x1, y1, self.th0, 0.0


class CircularArc(Geometry):
    __slots__ = ("_sin0", "_cos0")

    def __init__(self, x0: float, y0: float, th0: float, k0: float):
        super().__init__(x0, y0, th0, k0, 0.0)
        self._sin0 = math.sin(th0)
        self._cos0 = math.cos(th0)

    def xy(self, ds: float) -> Point:
        half_turn = 0.5 * self.k0 * ds
        sh = math.sin(half_turn)
        ch = math.cos(half_turn)
        magnitude = ds * _sinc(half_turn)

        midpoint_cos = math.fma(self._cos0, ch, -self._sin0 * sh)
        midpoint_sin = math.fma(self._sin0, ch, self._cos0 * sh)

        return (
            math.fma(magnitude, midpoint_cos, self.x0),
            math.fma(magnitude, midpoint_sin, self.y0),
        )

    def end(self, length: float) -> EndState:
        x1, y1 = self.xy(length)
        th1 = math.fma(self.k0, length, self.th0)
        return x1, y1, th1, self.k0


class EulerSpiral(Geometry):
    __slots__ = (
        "_sin0",
        "_cos0",
        "_eps",
        "_root_rho",
        "_amp",
        "_u0",
        "_fresnel_s0",
        "_fresnel_c0",
        "_phase_global_cos",
        "_phase_global_sin",
        "_stationary_dx",
        "_stationary_dy",
        "_asym_a0_x",
        "_asym_a0_y",
    )

    def __init__(
        self,
        x0: float,
        y0: float,
        th0: float,
        k0: float,
        sigma: float,
    ):
        if sigma == 0.0:
            raise ValueError("EulerSpiral requires nonzero sigma")
        super().__init__(x0, y0, th0, k0, sigma)

        self._sin0 = math.sin(th0)
        self._cos0 = math.cos(th0)

        rho = abs(sigma)
        root_rho = math.sqrt(rho)
        eps = math.copysign(1.0, sigma)

        self._eps = eps
        self._root_rho = root_rho
        self._amp = _SQRT_PI / root_rho

        # Optional caches use None until the corresponding scalar values are available.
        self._u0: float | None = None
        self._fresnel_s0: float | None = None
        self._fresnel_c0: float | None = None
        self._phase_global_cos: float | None = None
        self._phase_global_sin: float | None = None
        self._stationary_dx: float | None = None
        self._stationary_dy: float | None = None
        self._asym_a0_x: float | None = None
        self._asym_a0_y: float | None = None

        try:
            phase_local_cos, phase_local_sin = _phase_ratio_square_cs(
                k0, sigma, -0.5
            )
        except OverflowError:
            pass
        else:
            phase_global_cos, phase_global_sin = _rotate_xy(
                phase_local_cos,
                phase_local_sin,
                self._cos0,
                self._sin0,
            )
            self._phase_global_cos = phase_global_cos
            self._phase_global_sin = phase_global_sin

            scale = _SQRT_PI / root_rho
            self._stationary_dx = scale * (
                phase_global_cos - eps * phase_global_sin
            )
            self._stationary_dy = scale * (
                phase_global_sin + eps * phase_global_cos
            )

        u0 = eps * k0 / (_SQRT_PI * root_rho)
        if math.isfinite(u0):
            self._u0 = u0
            self._fresnel_s0, self._fresnel_c0 = fresnel(u0)

        if k0 != 0.0 and abs(sigma) / (k0 * k0) <= _ASYM_MAX_RATIO:
            self._asym_a0_x, self._asym_a0_y = (
                _asymptotic_antiderivative(k0, sigma)
            )

    def _ensure_phase(self) -> tuple[float, float, float, float]:
        phase_cos = self._phase_global_cos
        phase_sin = self._phase_global_sin
        stationary_dx = self._stationary_dx
        stationary_dy = self._stationary_dy

        if (
            phase_cos is None
            or phase_sin is None
            or stationary_dx is None
            or stationary_dy is None
        ):
            local_x, local_y = _phase_ratio_square_cs(
                self.k0, self.sigma, -0.5
            )
            phase_cos, phase_sin = _rotate_xy(
                local_x, local_y, self._cos0, self._sin0
            )

            scale = _SQRT_PI / self._root_rho
            stationary_dx = scale * (
                phase_cos - self._eps * phase_sin
            )
            stationary_dy = scale * (
                phase_sin + self._eps * phase_cos
            )

            self._phase_global_cos = phase_cos
            self._phase_global_sin = phase_sin
            self._stationary_dx = stationary_dx
            self._stationary_dy = stationary_dy

        return phase_cos, phase_sin, stationary_dx, stationary_dy

    def _ensure_fresnel(self) -> tuple[float, float, float, float, float]:
        u0 = self._u0
        fresnel_s0 = self._fresnel_s0
        fresnel_c0 = self._fresnel_c0
        phase_cos = self._phase_global_cos
        phase_sin = self._phase_global_sin

        if phase_cos is None or phase_sin is None:
            phase_cos, phase_sin, _, _ = self._ensure_phase()

        if u0 is None or fresnel_s0 is None or fresnel_c0 is None:
            u0 = self._eps * self.k0 / (_SQRT_PI * self._root_rho)
            if not math.isfinite(u0):
                raise OverflowError(
                    "Fresnel transform endpoint is not representable"
                )
            fresnel_s0, fresnel_c0 = fresnel(u0)
            self._u0 = u0
            self._fresnel_s0 = fresnel_s0
            self._fresnel_c0 = fresnel_c0

        return u0, fresnel_s0, fresnel_c0, phase_cos, phase_sin

    def _global_displacement(self, ds: float) -> tuple[float, float]:
        if ds == 0.0:
            return 0.0, 0.0
        if not math.isfinite(ds):
            raise ValueError("ds must be finite")

        k1 = math.fma(self.sigma, ds, self.k0)
        alpha = self.k0 * ds
        beta = (0.5 * self.sigma * ds) * ds
        q0 = alpha
        q1 = k1 * ds

        # Strongly oscillatory endpoint/stationary-phase expansion.
        if self.k0 != 0.0 and k1 != 0.0:
            if min(abs(q0), abs(q1)) >= _ASYM_MIN_ENDPOINT_PHASE:
                ratio = max(
                    abs(self.sigma) / (self.k0 * self.k0),
                    abs(self.sigma) / (k1 * k1),
                )
                if ratio <= _ASYM_MAX_RATIO:
                    a0_x = self._asym_a0_x
                    a0_y = self._asym_a0_y
                    if a0_x is None or a0_y is None:
                        a0_x, a0_y = _asymptotic_antiderivative(
                            self.k0, self.sigma
                        )
                        self._asym_a0_x = a0_x
                        self._asym_a0_y = a0_y

                    a1_x, a1_y = _asymptotic_antiderivative(
                        k1, self.sigma
                    )
                    end_cos, end_sin = _phase_sum_cs(alpha, beta)
                    local_x, local_y = _rotate_xy(
                        a1_x, a1_y, end_cos, end_sin
                    )
                    local_x -= a0_x
                    local_y -= a0_y

                    displacement_x, displacement_y = _rotate_xy(
                        local_x,
                        local_y,
                        self._cos0,
                        self._sin0,
                    )

                    if self.k0 * k1 < 0.0:
                        _, _, stationary_dx, stationary_dy = (
                            self._ensure_phase()
                        )
                        direction = math.copysign(1.0, ds)
                        displacement_x = math.fma(
                            direction, stationary_dx, displacement_x
                        )
                        displacement_y = math.fma(
                            direction, stationary_dy, displacement_y
                        )

                    return displacement_x, displacement_y

        abs_beta = abs(beta)
        if abs_beta <= _GL16_BETA_MAX:
            if abs_beta <= _GL12_BETA_MAX:
                integral_x, integral_y = _quadratic_phase_gauss(
                    alpha, beta, _GL12
                )
            elif abs_beta <= _GL14_BETA_MAX:
                integral_x, integral_y = _quadratic_phase_gauss(
                    alpha, beta, _GL14
                )
            else:
                integral_x, integral_y = _quadratic_phase_gauss(
                    alpha, beta, _GL16
                )

            local_x = ds * integral_x
            local_y = ds * integral_y
            return _rotate_xy(
                local_x, local_y, self._cos0, self._sin0
            )

        # Regular cached Fresnel path.
        fresnel_s0 = self._fresnel_s0
        fresnel_c0 = self._fresnel_c0
        phase_cos = self._phase_global_cos
        phase_sin = self._phase_global_sin
        if (
            fresnel_s0 is None
            or fresnel_c0 is None
            or phase_cos is None
            or phase_sin is None
        ):
            _, fresnel_s0, fresnel_c0, phase_cos, phase_sin = (
                self._ensure_fresnel()
            )

        u1 = self._eps * k1 / (_SQRT_PI * self._root_rho)
        fresnel_s1, fresnel_c1 = fresnel(u1)
        delta_x = fresnel_c1 - fresnel_c0
        delta_y = self._eps * (fresnel_s1 - fresnel_s0)
        product_x, product_y = _rotate_xy(
            delta_x, delta_y, phase_cos, phase_sin
        )
        return self._amp * product_x, self._amp * product_y

    def xy(self, ds: float) -> Point:
        displacement_x, displacement_y = self._global_displacement(ds)
        return self.x0 + displacement_x, self.y0 + displacement_y

    def end(self, length: float) -> EndState:
        x1, y1 = self.xy(length)
        midpoint_curvature = math.fma(
            0.5 * self.sigma, length, self.k0
        )
        th1 = math.fma(length, midpoint_curvature, self.th0)
        k1 = math.fma(self.sigma, length, self.k0)
        return x1, y1, th1, k1


def create_geo(
    x0: float,
    y0: float,
    th0: float,
    k0: float,
    sigma: float,
) -> Geometry:
    """Create the fastest exact specialization for the supplied parameters."""
    if sigma == 0.0:
        if k0 == 0.0:
            return Line(x0, y0, th0)
        return CircularArc(x0, y0, th0, k0)
    return EulerSpiral(x0, y0, th0, k0, sigma)


__all__ = [
    "Geometry",
    "Line",
    "CircularArc",
    "EulerSpiral",
    "create_geo",
    "fresnel",
]
