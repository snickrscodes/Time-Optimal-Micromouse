"""Reference Python SIDE actuator segment for DD/yaw Phase 1.

The segment integrates the smooth active-side extremal

    dw/ds = 2 A_eps(w, kappa_0 + sigma s, sigma)

and, for the differentiable implementation, the complete local variational
system required by the existing segment protocol.  This is intentionally a
reference implementation: correctness and derivative transparency are more
important than hot-path performance.  The future Phase-5 native mirror can be
optimized only after this kernel is qualified.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .base import DiffSegment, EvalSegment, SegmentType
from .differential_drive import (
    DifferentialDriveDomainError,
    DifferentialDriveParameters,
    DriveSide,
    side_candidate,
    side_candidate_partials,
    side_state,
)


class SideActuatorIntegrationError(ArithmeticError):
    """Raised when the numerical reference segment cannot integrate reliably."""


_RTOL = 2.0e-11
_ATOL_EVAL = np.array([2.0e-12, 2.0e-12], dtype=float)
_ATOL_DIFF = np.array(
    [
        2.0e-12,  # w
        2.0e-12,  # time
        2.0e-11,  # S_sigma
        2.0e-11,  # S_w0
        2.0e-11,  # S_k0
        2.0e-11,  # R_sigma
        2.0e-11,  # R_w0
        2.0e-11,  # R_k0
    ],
    dtype=float,
)


def _validate_segment_inputs(
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> tuple[float, float, float, float, DriveSide]:
    L = float(L)
    sigma = float(sigma)
    w0 = float(w0)
    k0 = float(k0)
    if not all(math.isfinite(v) for v in (L, sigma, w0, k0)):
        raise ValueError("SIDE segment inputs must be finite")
    if L < 0.0:
        raise ValueError("SIDE segment length must be nonnegative")
    if w0 <= 0.0:
        raise ValueError("SIDE segment requires w0 > 0")
    try:
        side = DriveSide(int(side))
    except (TypeError, ValueError) as exc:
        raise ValueError("SIDE segment side must be LEFT/-1 or RIGHT/+1") from exc

    # Fail immediately if the initial state is outside the qualified chart.
    side_state(w0, k0, side, params)
    side_candidate(w0, k0, sigma, side, params)
    return L, sigma, w0, k0, side


def _validate_ds(ds: float, L: float) -> float:
    ds = float(ds)
    if not math.isfinite(ds):
        raise ValueError("SIDE segment distance must be finite")
    tol = 8.0 * math.ulp(max(1.0, abs(L)))
    if ds < -tol or ds > L + tol:
        raise ValueError(f"SIDE segment distance {ds!r} is outside [0, {L!r}]")
    if ds < 0.0:
        return 0.0
    if ds > L:
        return L
    return ds


def _event_w_positive(_s: float, y: np.ndarray) -> float:
    return float(y[0])


_event_w_positive.terminal = True
_event_w_positive.direction = -1.0


class _SideCommon:
    def _init_side_common(
        self,
        L: float,
        sigma: float,
        w0: float,
        k0: float,
        side: DriveSide,
        params: DifferentialDriveParameters,
    ) -> None:
        self.side = side
        self.params = params
        self._cache: dict[tuple[bool, float], tuple[float, ...]] = {}

    def _domain_event(self, kind: str) -> Callable[[float, np.ndarray], float]:
        eps = int(self.side)
        beta = self.params.beta
        eta = self.params.eta
        v_free = self.params.side_free_speed_grid
        h_floor = self.params.h_floor
        c_floor = self.params.c_floor
        speed_margin = self.params.speed_margin
        k0 = self.k0
        sigma = self.sigma

        if kind == "h":
            def event(s: float, _y: np.ndarray) -> float:
                kappa = k0 + sigma * s
                return 1.0 + eps * beta * kappa - h_floor
        elif kind == "c":
            def event(s: float, _y: np.ndarray) -> float:
                kappa = k0 + sigma * s
                return 1.0 + eps * eta * kappa - c_floor
        elif kind == "free":
            def event(s: float, y: np.ndarray) -> float:
                w = max(float(y[0]), 0.0)
                kappa = k0 + sigma * s
                h = 1.0 + eps * beta * kappa
                return v_free - math.sqrt(w) * h - speed_margin
        else:  # pragma: no cover - internal programming error
            raise AssertionError(kind)

        event.terminal = True
        event.direction = -1.0
        return event

    def _events(self):
        return (
            _event_w_positive,
            self._domain_event("h"),
            self._domain_event("c"),
            self._domain_event("free"),
        )

    def _max_step(self, ds: float) -> float:
        if ds <= 0.0:
            return math.inf
        # Reference-kernel policy: force at least 12 accepted intervals over a
        # queried prefix and cap individual steps to keep domain-event location
        # predictable on aggressive clothoids.
        return min(0.02, max(ds / 12.0, 1.0e-5))

    def _check_solution(self, sol, ds: float) -> None:
        if not sol.success:
            raise SideActuatorIntegrationError(
                f"SIDE integration failed at s={sol.t[-1] if sol.t.size else 0.0!r}: {sol.message}"
            )
        if sol.status == 1:
            names = ("w=0", "h floor", "c floor", "rated side free speed")
            hits = [
                (names[i], float(ev[0]))
                for i, ev in enumerate(sol.t_events)
                if len(ev)
            ]
            raise DifferentialDriveDomainError(
                f"SIDE segment leaves qualified chart before s={ds!r}: {hits!r}"
            )
        if not sol.t.size or abs(float(sol.t[-1]) - ds) > 5.0e-11 * max(1.0, ds):
            raise SideActuatorIntegrationError("SIDE integration did not reach requested endpoint")


class SideEvalSegment(_SideCommon, EvalSegment):
    """Evaluation-only reference SIDE segment."""

    def __init__(
        self,
        L: float,
        sigma: float,
        w0: float,
        k0: float,
        side: DriveSide | int,
        params: DifferentialDriveParameters,
    ):
        L, sigma, w0, k0, side = _validate_segment_inputs(
            L, sigma, w0, k0, side, params
        )
        mode = SegmentType.SIDE_RIGHT if side is DriveSide.RIGHT else SegmentType.SIDE_LEFT
        super().__init__(L, sigma, w0, k0, mode)
        self._init_side_common(L, sigma, w0, k0, side, params)

    def _integrate(self, ds: float) -> tuple[float, float]:
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            return self.w0, 0.0
        key = (False, ds)
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0], cached[1]

        def rhs(s: float, y: np.ndarray) -> np.ndarray:
            w = float(y[0])
            if w <= 0.0:
                # Let the terminal event own the user-facing diagnostic.
                w = max(w, np.finfo(float).tiny)
            kappa = self.k0 + self.sigma * s
            a = side_candidate(w, kappa, self.sigma, self.side, self.params)
            return np.array((2.0 * a, 1.0 / math.sqrt(w)), dtype=float)

        sol = solve_ivp(
            rhs,
            (0.0, ds),
            np.array((self.w0, 0.0), dtype=float),
            method="DOP853",
            rtol=_RTOL,
            atol=_ATOL_EVAL,
            max_step=self._max_step(ds),
            events=self._events(),
        )
        self._check_solution(sol, ds)
        out = (float(sol.y[0, -1]), float(sol.y[1, -1]))
        self._cache[key] = out
        return out

    def w(self, ds: float) -> float:
        return self._integrate(ds)[0]

    def time(self, ds: float) -> float:
        return self._integrate(ds)[1]


class SideSegment(_SideCommon, DiffSegment):
    """Differentiable reference SIDE segment with integrated sensitivities."""

    def __init__(
        self,
        L: float,
        sigma: float,
        w0: float,
        k0: float,
        side: DriveSide | int,
        params: DifferentialDriveParameters,
    ):
        L, sigma, w0, k0, side = _validate_segment_inputs(
            L, sigma, w0, k0, side, params
        )
        mode = SegmentType.SIDE_RIGHT if side is DriveSide.RIGHT else SegmentType.SIDE_LEFT
        super().__init__(L, sigma, w0, k0, mode)
        self._init_side_common(L, sigma, w0, k0, side, params)

    def _integrate_all(self, ds: float) -> tuple[float, ...]:
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            a, _, _, _ = side_candidate_partials(
                self.w0, self.k0, self.sigma, self.side, self.params
            )
            return (
                self.w0,
                0.0,
                0.0,  # S_sigma
                1.0,  # S_w0
                0.0,  # S_k0
                0.0,  # R_sigma
                0.0,  # R_w0
                0.0,  # R_k0
                2.0 * a,
            )

        key = (True, ds)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def rhs(s: float, y: np.ndarray) -> np.ndarray:
            w = float(y[0])
            if w <= 0.0:
                w = max(w, np.finfo(float).tiny)
            kappa = self.k0 + self.sigma * s
            a, a_w, a_kappa, a_sigma = side_candidate_partials(
                w, kappa, self.sigma, self.side, self.params
            )
            f = 2.0 * a
            f_w = 2.0 * a_w
            f_kappa = 2.0 * a_kappa
            f_sigma = 2.0 * a_sigma

            s_sigma, s_w0, s_k0 = map(float, y[2:5])
            inv_root = 1.0 / math.sqrt(w)
            time_sensitivity_scale = -0.5 / (w * math.sqrt(w))

            return np.array(
                (
                    f,
                    inv_root,
                    f_w * s_sigma + f_kappa * s + f_sigma,
                    f_w * s_w0,
                    f_w * s_k0 + f_kappa,
                    time_sensitivity_scale * s_sigma,
                    time_sensitivity_scale * s_w0,
                    time_sensitivity_scale * s_k0,
                ),
                dtype=float,
            )

        y0 = np.array((self.w0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=float)
        sol = solve_ivp(
            rhs,
            (0.0, ds),
            y0,
            method="DOP853",
            rtol=_RTOL,
            atol=_ATOL_DIFF,
            max_step=self._max_step(ds),
            events=self._events(),
        )
        self._check_solution(sol, ds)
        y = tuple(float(v) for v in sol.y[:, -1])
        kappa = self.k0 + self.sigma * ds
        a = side_candidate(y[0], kappa, self.sigma, self.side, self.params)
        out = (*y, 2.0 * a)
        self._cache[key] = out
        return out

    def w(self, ds: float) -> float:
        return self._integrate_all(ds)[0]

    def time(self, ds: float) -> float:
        return self._integrate_all(ds)[1]

    def w_and_jac(self, ds: float):
        ds = _validate_ds(ds, self.L)
        w, _time, s_sigma, s_w0, s_k0, *_rest, f_endpoint = self._integrate_all(ds)
        return w, (
            f_endpoint,
            s_sigma,
            s_w0,
            s_k0,
            self.sigma,
            ds,
            0.0,
            1.0,
        )

    def time_and_jac(self, ds: float):
        ds = _validate_ds(ds, self.L)
        w, time, _s_sigma, _s_w0, _s_k0, r_sigma, r_w0, r_k0, _f = self._integrate_all(ds)
        return time, (
            1.0 / math.sqrt(w),
            r_sigma,
            r_w0,
            r_k0,
        )

    def state_time_and_jac(self, ds: float):
        w, w_jac = self.w_and_jac(ds)
        time, time_jac = self.time_and_jac(ds)
        return w, w_jac, time, time_jac


__all__ = [
    "SideActuatorIntegrationError",
    "SideEvalSegment",
    "SideSegment",
]
