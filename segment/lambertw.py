import math

# Local bindings reduce attribute-lookup overhead in CPython.
_FMA = math.fma
_EXP = math.exp
_EXPM1 = math.expm1
_LOG = math.log
_SQRT = math.sqrt

_E = math.e
_TWO_E = 2.0 * _E

# Binary64 approximation to 1/e and its exact-value residual:
#
#     1/e = _INV_E + _INV_E_LO
#
_INV_E = 1.0 / _E
_INV_E_LO = -1.2428753672788363e-17
_NEG_INV_E = -_INV_E

# Seed-region boundaries.
_PUISEUX6_MAX_U = 1.0e-5
_PUISEUX12_MAX_U = 1.0e-3

_NEGATIVE_SEED_MAX_X = -0.2
_LOW_RATIONAL_MAX_X = 0.5
_MID_POLY1_MAX_X = 4.0
_MID_POLY2_MAX_X = 64.0

# Switch from the fitted asymptotic correction to the exact next
# asymptotic term.
_ASYMPTOTIC_CORRECTION_MAX_X = 1099511627776.0  # 2**40


def lambertw0(x: float) -> float:
    """
    Real principal branch W_0(x) of the Lambert W function.

    Solves

        w * exp(w) = x

    for

        x >= -1/e

    and returns a binary64 result.

    Special cases
    -------------
    NaN:
        Returned unchanged.
    +inf:
        Returns +inf.
    +/-0:
        Returned unchanged, preserving the sign of zero.
    x < -1/e:
        Raises ValueError.

    Notes
    -----
    The binary64 value produced by ``-1.0 / math.e`` lies microscopically
    below the exact real branch point. This implementation treats that
    conventional rounded representation as the branch point and returns
    exactly -1.0.
    """

    # Preserve NaN.
    if x != x:
        return x

    if x == math.inf:
        return x

    # Preserve signed zero.
    if x == 0.0:
        return x

    # Accommodate the conventional binary64 representation of -1/e.
    if x == _NEG_INV_E:
        return -1.0

    if x < _NEG_INV_E:
        raise ValueError("lambertw0 is real only for x >= -1/e")

    # ------------------------------------------------------------------
    # Negative branch-point region
    # ------------------------------------------------------------------
    if x < _NEGATIVE_SEED_MAX_X:
        # Compensated evaluation of x + 1/e.
        u = (x + _INV_E) + _INV_E_LO
        p = _SQRT(_TWO_E * u)

        # Degree-6 Puiseux expansion:
        #
        # W_0(x) =
        #     -1 + p - p²/3 + 11p³/72 - 43p⁴/540
        #        + 769p⁵/17280 - 221p⁶/8505 + O(p⁷)
        #
        if u < _PUISEUX6_MAX_U:
            w = _FMA(
                _FMA(
                    _FMA(
                        _FMA(
                            _FMA(
                                _FMA(
                                    -221.0 / 8505.0,
                                    p,
                                    769.0 / 17280.0,
                                ),
                                p,
                                -43.0 / 540.0,
                            ),
                            p,
                            11.0 / 72.0,
                        ),
                        p,
                        -1.0 / 3.0,
                    ),
                    p,
                    1.0,
                ),
                p,
                -1.0,
            )
            return w

        # Degree-12 Puiseux expansion.
        if u < _PUISEUX12_MAX_U:
            w = _FMA(
                -1118511313.0 / 709296588000.0,
                p,
                169709463197.0 / 69528040243200.0,
            )
            w = _FMA(w, p, -5776369.0 / 1515591000.0)
            w = _FMA(w, p, 226287557.0 / 37623398400.0)
            w = _FMA(w, p, -1963.0 / 204120.0)
            w = _FMA(w, p, 680863.0 / 43545600.0)
            w = _FMA(w, p, -221.0 / 8505.0)
            w = _FMA(w, p, 769.0 / 17280.0)
            w = _FMA(w, p, -43.0 / 540.0)
            w = _FMA(w, p, 11.0 / 72.0)
            w = _FMA(w, p, -1.0 / 3.0)
            w = _FMA(w, p, 1.0)
            return _FMA(w, p, -1.0)

        # Degree-6 seed for the remainder of the negative interval.
        w = _FMA(
            _FMA(
                _FMA(
                    _FMA(
                        _FMA(
                            _FMA(
                                -221.0 / 8505.0,
                                p,
                                769.0 / 17280.0,
                            ),
                            p,
                            -43.0 / 540.0,
                        ),
                        p,
                        11.0 / 72.0,
                    ),
                    p,
                    -1.0 / 3.0,
                ),
                p,
                1.0,
            ),
            p,
            -1.0,
        )

        # Use q = 1 + w and the branch-point-scaled equation
        #
        #     F(q) = 1 - (1-q)e^q - e*u = 0.
        #
        # The first term is evaluated as
        #
        #     q + (q-1)expm1(q),
        #
        # whose leading cancellation occurs inside the FMA. This is much
        # more accurate near q = 0 than forming w - x*exp(-w).
        eu = _E * u
        q = w + 1.0

        # Scaled Halley iteration 1.
        em1 = _EXPM1(q)
        residual = _FMA(q - 1.0, em1, q) - eu
        scaled_residual = residual * _EXP(-q)

        q -= (
            2.0 * scaled_residual * q
            / _FMA(
                2.0 * q,
                q,
                -scaled_residual * (q + 1.0),
            )
        )

        # Scaled Halley iteration 2.
        em1 = _EXPM1(q)
        residual = _FMA(q - 1.0, em1, q) - eu
        scaled_residual = residual * _EXP(-q)

        q -= (
            2.0 * scaled_residual * q
            / _FMA(
                2.0 * q,
                q,
                -scaled_residual * (q + 1.0),
            )
        )

        return q - 1.0

    # ------------------------------------------------------------------
    # Low range: rational seed, one Halley iteration
    # ------------------------------------------------------------------
    if x < _LOW_RATIONAL_MAX_X:
        numerator = 1.0 + x * (
            2.2380488687194635
            + x * (
                0.5323229596042623
                - 0.08326761577458537 * x
            )
        )

        denominator = 1.0 + x * (
            3.238004480223941
            + 2.2706502298872793 * x
        )

        w = x * numerator / denominator

    # ------------------------------------------------------------------
    # First midrange polynomial: 0.5 <= x < 4
    # ------------------------------------------------------------------
    elif x < _MID_POLY1_MAX_X:
        y = _LOG(x)

        # Ordinary Horner is intentional here. The seed needs only enough
        # accuracy for the final Halley iteration, and Python-level FMA
        # calls cost more than ordinary multiply-add expressions.
        w = (
            (
                (
                    (
                        (
                            0.000254691122488741 * y
                            - 0.001604936436645819
                        ) * y
                        - 0.0013647878217518169
                    ) * y
                    + 0.07367221971761982
                ) * y
                + 0.36189954940301794
            ) * y
            + 0.5671435067846463
        )

    # ------------------------------------------------------------------
    # Second midrange polynomial: 4 <= x < 64
    # ------------------------------------------------------------------
    elif x < _MID_POLY2_MAX_X:
        y = _LOG(x)

        w = (
            (
                (
                    (
                        0.0003229652230849584 * y
                        - 0.007594247957917449
                    ) * y
                    + 0.08435395036268026
                ) * y
                + 0.35232059530135407
            ) * y
            + 0.5706766967072727
        )

    # ------------------------------------------------------------------
    # Large-x asymptotic seeds
    # ------------------------------------------------------------------
    else:
        l1 = _LOG(x)
        l2 = _LOG(l1)

        inv_l1 = 1.0 / l1
        l2_over_l1 = l2 * inv_l1

        # Expansion through 1/l1².
        w = (
            l1
            - l2
            + l2_over_l1
            * (
                1.0
                + 0.5 * l2_over_l1
                - inv_l1
            )
        )

        if x < _ASYMPTOTIC_CORRECTION_MAX_X:
            # Fitted correction over [64, 2**40).
            w += (
                (
                    (
                        (
                            (
                                -18.044416677004946 * inv_l1
                                + 12.960214166179945
                            ) * inv_l1
                            - 2.0337318250793284
                        ) * inv_l1
                        - 0.1835022973519072
                    ) * inv_l1
                    + 0.011947745371139771
                ) * inv_l1
                - 0.00017964131708255804
            )

        else:
            w += (l2 * inv_l1 * inv_l1 * inv_l1 * (l2 * (2.0 * l2 - 9.0) + 6.0) / 6.0)

    # ------------------------------------------------------------------
    # One generic Halley iteration
    # ------------------------------------------------------------------
    #
    # The residual is formed as
    #
    #     w - x*exp(-w)
    #
    # with an FMA. All non-endpoint seeds above are designed to reach
    # binary64 precision in this single iteration.
    t = _FMA(-x, _EXP(-w), w)
    p = w + 1.0
    return w - t / _FMA(-0.5 * (w + 2.0), t / p, p)