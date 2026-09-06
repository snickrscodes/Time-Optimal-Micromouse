"""Public segment package and centralized segment dispatcher."""

from __future__ import annotations

import math
from typing import Any

from .base import DiffSegment, EvalSegment, SegmentType

from .straight import (
    BrakeEvalSegment,
    BrakeSegment,
    StraightEvalSegment,
    StraightSegment,
)

from .circular import (
    CircularEvalSegment,
    CircularEvalSegmentStable,
    CircularSegment,
    CircularSegmentStable,
    circular_use_stable,
    make_circular_eval_segment,
    make_circular_segment,
)

from .grip import (
    GripEvalSegment,
    GripSegment,
)

from .motor import (
    LW_EQ,
    MOTOR_U0_SERIES_TOL,
    MotorEvalSegment,
    MotorEvalSegmentStable,
    MotorSegment,
    MotorSegmentStable,
)


def motor_use_stable(w0: float) -> bool:
    """Return whether the equilibrium-centered motor series is preferred."""
    if w0 < 0.0:
        raise ValueError("motor segment requires w0 >= 0")

    u0 = math.fma(
        LW_EQ,
        math.sqrt(w0),
        -1.0,
    )

    return abs(u0) <= MOTOR_U0_SERIES_TOL


def make_motor_eval_segment(
    L: float,
    sigma: float,
    w0: float,
    k0: float,
):
    cls = (
        MotorEvalSegmentStable
        if motor_use_stable(w0)
        else MotorEvalSegment
    )

    return cls(
        L,
        sigma,
        w0,
        k0,
    )


def make_motor_segment(
    L: float,
    sigma: float,
    w0: float,
    k0: float,
):
    cls = (
        MotorSegmentStable
        if motor_use_stable(w0)
        else MotorSegment
    )

    return cls(
        L,
        sigma,
        w0,
        k0,
    )


def _reject_options(
    family: str,
    options: dict[str, Any],
) -> None:
    if not options:
        return

    names = ", ".join(sorted(options))
    raise TypeError(
        f"{family} does not accept compile options: {names}"
    )


def compile_segment(
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    segment_type: SegmentType,
    grad: bool = False,
    **options: Any,
) -> EvalSegment | DiffSegment:
    """Compile the numerically appropriate segment implementation.

    Routing
    -------
    GRIP:
        sigma == 0 and k0 == 0:
            Straight segment.

        sigma == 0 and k0 != 0:
            Circular segment, with automatic regular/stable dispatch.

        sigma != 0:
            General linear-curvature native Cflow segment.

    MOTOR:
        Automatic regular/equilibrium-stable dispatch.

    BRAKE:
        Constant-rate brake segment.

    Parameters
    ----------
    L, sigma, w0, k0:
        Segment parameters.

    segment_type:
        Member of ``SegmentType``.

    grad:
        ``False`` selects an evaluation-only segment.
        ``True`` selects a differentiable segment.

    **options:
        Forwarded only to the general Cflow-backed Grip segment. The supported
        options are boundary_start, x0_sensitivity_policy, reverse_eta, and
        authoritative_w1.  The latter two are internal reverse-time replay
        controls and are not used by scalar/public Cflow paths.
    """

    if not isinstance(grad, bool):
        raise TypeError("grad must be a bool")

    # ------------------------------------------------------------------
    # Grip-limited families
    # ------------------------------------------------------------------
    if segment_type == SegmentType.GRIP:
        if sigma == 0.0:
            _reject_options(
                "straight/circular grip segments",
                options,
            )

            if k0 == 0.0:
                cls = (
                    StraightSegment
                    if grad
                    else StraightEvalSegment
                )

                return cls(
                    L,
                    w0,
                )

            if grad:
                return make_circular_segment(
                    L,
                    w0,
                    k0,
                )

            return make_circular_eval_segment(
                L,
                w0,
                k0,
            )

        cls = (
            GripSegment
            if grad
            else GripEvalSegment
        )

        return cls(
            L,
            sigma,
            w0,
            k0,
            **options,
        )

    # ------------------------------------------------------------------
    # Motor-limited family
    # ------------------------------------------------------------------
    if segment_type == SegmentType.MOTOR:
        _reject_options(
            "motor segments",
            options,
        )

        if grad:
            return make_motor_segment(
                L,
                sigma,
                w0,
                k0,
            )

        return make_motor_eval_segment(
            L,
            sigma,
            w0,
            k0,
        )

    # ------------------------------------------------------------------
    # Braking family
    # ------------------------------------------------------------------
    if segment_type == SegmentType.BRAKE:
        _reject_options(
            "brake segments",
            options,
        )

        cls = (
            BrakeSegment
            if grad
            else BrakeEvalSegment
        )

        return cls(
            L,
            sigma,
            w0,
            k0,
        )

    raise ValueError(
        f"unsupported segment type: {segment_type!r}"
    )


__all__ = [
    # Core protocol
    "EvalSegment",
    "DiffSegment",
    "SegmentType",

    # Straight/brake
    "StraightEvalSegment",
    "StraightSegment",
    "BrakeEvalSegment",
    "BrakeSegment",

    # Circular grip
    "CircularEvalSegment",
    "CircularEvalSegmentStable",
    "CircularSegment",
    "CircularSegmentStable",
    "circular_use_stable",
    "make_circular_eval_segment",
    "make_circular_segment",

    # General linear-curvature grip
    "GripEvalSegment",
    "GripSegment",

    # Motor
    "MotorEvalSegment",
    "MotorEvalSegmentStable",
    "MotorSegment",
    "MotorSegmentStable",
    "motor_use_stable",
    "make_motor_eval_segment",
    "make_motor_segment",

    # Unified factory
    "compile_segment",
]