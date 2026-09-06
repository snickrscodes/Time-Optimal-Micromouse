"""Constant-rate straight grip and braking segments."""

from __future__ import annotations

import math

from .base import DiffSegment, EvalSegment, SegmentType
from .constants import A_BRAKE, MU_G


_TWO_MU_G = 2.0 * MU_G
_TWO_A_BRAKE = 2.0 * A_BRAKE


def _validate_constant_rate_segment(
    L: float,
    w0: float,
    two_rate: float,
    family: str,
) -> tuple[float, float]:
    L = float(L)
    w0 = float(w0)

    if not math.isfinite(L) or not math.isfinite(w0):
        raise ValueError(f"{family} inputs must be finite")
    if L < 0.0:
        raise ValueError(f"{family} length must be nonnegative")
    if w0 < 0.0:
        raise ValueError(f"{family} requires w0 >= 0")

    w_end = math.fma(two_rate, L, w0)
    if w_end < 0.0:
        raise ValueError(
            f"{family} reaches negative squared speed over the segment: "
            f"w(L)={w_end!r}"
        )

    return L, w0


def _constant_rate_time_terms(
    w0_root: float,
    w1: float,
    ds: float,
) -> tuple[float, float, float]:
    """Return T, dT/dds, and dT/dw0.

    For

        w(s) = w0 + 2*a*s,

    the time integral is evaluated as

        T = 2*ds / (sqrt(w0) + sqrt(w1)),

    avoiding subtraction when ds is small.
    """

    if w1 < 0.0:
        raise ValueError("constant-rate segment evaluated beyond zero speed")

    w1_root = math.sqrt(w1)

    if ds == 0.0:
        return (
            0.0,
            math.inf if w1_root == 0.0 else 1.0 / w1_root,
            0.0,
        )

    root_sum = w0_root + w1_root

    # This can occur only for an identically zero-speed interval.
    if root_sum == 0.0:
        return math.inf, math.inf, -math.inf

    d = ds / root_sum
    time = 2.0 * d

    time_s = (
        math.inf
        if w1_root == 0.0
        else 1.0 / w1_root
    )

    time_w0 = (
        -math.inf
        if w0_root == 0.0 or w1_root == 0.0
        else -d / (w0_root * w1_root)
    )

    return time, time_s, time_w0


# ---------------------------------------------------------------------------
# Straight grip: k0 = sigma = 0
# ---------------------------------------------------------------------------


class StraightEvalSegment(EvalSegment):
    def __init__(self, L, w0):
        L, w0 = _validate_constant_rate_segment(
            L,
            w0,
            _TWO_MU_G,
            "straight grip segment",
        )
        super().__init__(
            L,
            0.0,
            w0,
            0.0,
            SegmentType.GRIP,
        )

    def w(self, ds: float) -> float:
        return math.fma(
            _TWO_MU_G,
            ds,
            self.w0,
        )

    def time(self, ds: float) -> float:
        w1 = math.fma(_TWO_MU_G, ds, self.w0)
        return _constant_rate_time_terms(math.sqrt(self.w0), w1, ds)[0]


class StraightSegment(DiffSegment):
    def __init__(self, L, w0):
        L, w0 = _validate_constant_rate_segment(
            L,
            w0,
            _TWO_MU_G,
            "straight grip segment",
        )
        super().__init__(
            L,
            0.0,
            w0,
            0.0,
            SegmentType.GRIP,
        )

        self.sqrt_w0 = math.sqrt(w0)

    def w(self, ds: float) -> float:
        return math.fma(
            _TWO_MU_G,
            ds,
            self.w0,
        )

    def w_and_jac(self, ds: float):
        w1 = math.fma(
            _TWO_MU_G,
            ds,
            self.w0,
        )

        return w1, (
            _TWO_MU_G,  # dw/ds
            0.0,        # dw/dsigma
            1.0,        # dw/dw0
            0.0,        # dw/dk0
            0.0,        # sigma
            ds,
            0.0,
            1.0,
        )

    def time_and_jac(self, ds: float):
        w1 = math.fma(
            _TWO_MU_G,
            ds,
            self.w0,
        )

        time, time_s, time_w0 = _constant_rate_time_terms(
            self.sqrt_w0,
            w1,
            ds,
        )

        return time, (
            time_s,
            0.0,      # dT/dsigma
            time_w0,
            0.0,      # dT/dk0
        )

    def state_time_and_jac(self, ds: float):
        w1 = math.fma(
            _TWO_MU_G,
            ds,
            self.w0,
        )

        time, time_s, time_w0 = _constant_rate_time_terms(
            self.sqrt_w0,
            w1,
            ds,
        )

        return (
            w1,
            (
                _TWO_MU_G,
                0.0,
                1.0,
                0.0,
                0.0,
                ds,
                0.0,
                1.0,
            ),
            time,
            (
                time_s,
                0.0,
                time_w0,
                0.0,
            ),
        )


# ---------------------------------------------------------------------------
# Constant braking-rate segment
# ---------------------------------------------------------------------------


class BrakeEvalSegment(EvalSegment):
    def __init__(self, L, sigma, w0, k0):
        L, w0 = _validate_constant_rate_segment(
            L,
            w0,
            _TWO_A_BRAKE,
            "brake segment",
        )
        super().__init__(
            L,
            sigma,
            w0,
            k0,
            SegmentType.BRAKE,
        )

    def w(self, ds: float) -> float:
        return math.fma(
            _TWO_A_BRAKE,
            ds,
            self.w0,
        )

    def time(self, ds: float) -> float:
        w1 = math.fma(_TWO_A_BRAKE, ds, self.w0)
        return _constant_rate_time_terms(math.sqrt(self.w0), w1, ds)[0]


class BrakeSegment(DiffSegment):
    def __init__(self, L, sigma, w0, k0):
        L, w0 = _validate_constant_rate_segment(
            L,
            w0,
            _TWO_A_BRAKE,
            "brake segment",
        )
        super().__init__(
            L,
            sigma,
            w0,
            k0,
            SegmentType.BRAKE,
        )

        self.sqrt_w0 = math.sqrt(w0)

    def w(self, ds: float) -> float:
        return math.fma(
            _TWO_A_BRAKE,
            ds,
            self.w0,
        )

    def w_and_jac(self, ds: float):
        w1 = math.fma(
            _TWO_A_BRAKE,
            ds,
            self.w0,
        )

        return w1, (
            _TWO_A_BRAKE,  # dw/ds
            0.0,           # dw/dsigma
            1.0,           # dw/dw0
            0.0,           # dw/dk0
            self.sigma,
            ds,
            0.0,
            1.0,
        )

    def time_and_jac(self, ds: float):
        w1 = math.fma(
            _TWO_A_BRAKE,
            ds,
            self.w0,
        )

        time, time_s, time_w0 = _constant_rate_time_terms(
            self.sqrt_w0,
            w1,
            ds,
        )

        return time, (
            time_s,
            0.0,      # dT/dsigma
            time_w0,
            0.0,      # dT/dk0
        )

    def state_time_and_jac(self, ds: float):
        w1 = math.fma(
            _TWO_A_BRAKE,
            ds,
            self.w0,
        )

        time, time_s, time_w0 = _constant_rate_time_terms(
            self.sqrt_w0,
            w1,
            ds,
        )

        return (
            w1,
            (
                _TWO_A_BRAKE,
                0.0,
                1.0,
                0.0,
                self.sigma,
                ds,
                0.0,
                1.0,
            ),
            time,
            (
                time_s,
                0.0,
                time_w0,
                0.0,
            ),
        )


__all__ = [
    "StraightEvalSegment",
    "StraightSegment",
    "BrakeEvalSegment",
    "BrakeSegment",
]