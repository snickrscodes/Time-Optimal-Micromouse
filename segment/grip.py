"""General linear-curvature GRIP segments backed by the vendored Cflow kernel.

The application state is ``w = MU_G * X`` and curvature is
``k(s) = k0 + sigma*s``.  General ``sigma != 0`` GRIP segments map directly to
Cflow inputs ``(X0, q0, b, h) = (w0/MU_G, k0, sigma, ds)``.

The module is intentionally thin: physical scaling, exact zero-prefix
identities, native status translation, boundary-start semantics, and lazy
endpoint caches.  All numerical flow/integral work lives in ``native/cflow``.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Literal

from cflow._binding import (
    Status,
    all_raw,
    eval_jac_raw,
    eval_raw,
    first_event_raw,
    integral_jac_raw, integral_value_raw,
    status_name,
)

from cflow._reverse_eta import (
    all_raw as reverse_eta_all_raw,
    should_use_reverse_eta,
)

from .base import DiffSegment, EvalSegment, SegmentType
from .constants import MU_G


MU_G_INV = 1.0 / MU_G
SQRT_MU_G = math.sqrt(MU_G)
INV_SQRT_MU_G = 1.0 / SQRT_MU_G
INV_MU_G_SQRT_MU_G = MU_G_INV * INV_SQRT_MU_G
TWO_MU_G = 2.0 * MU_G

TimeX0Policy = Literal["raise", "negative_infinity", "renormalized"]


class GripCflowError(FloatingPointError):
    """Base class for structured Cflow failures at the segment boundary."""


class GripDomainError(ValueError):
    """A requested GRIP prefix is outside the real flow domain."""

    def __init__(self, message: str, *, status: int, event_position: float | None = None):
        super().__init__(message)
        self.status = int(status)
        self.event_position = event_position


class GripConditioningError(GripCflowError):
    """Cflow reports that binary64 no longer resolves the requested trajectory."""


@dataclass(frozen=True, slots=True)
class GripDomainProbe:
    pre_event: bool
    event_position: float | None
    status: int


def _validate_grip_inputs(
    L: float,
    sigma: float,
    w0: float,
    k0: float,
    *,
    boundary_start: bool,
) -> tuple[float, float, float, float]:
    try:
        L, sigma, w0, k0 = map(float, (L, sigma, w0, k0))
    except (TypeError, ValueError) as error:
        raise ValueError("grip-segment inputs must be real numbers") from error
    if not all(map(math.isfinite, (L, sigma, w0, k0))):
        raise ValueError("grip-segment inputs must be finite")
    if L < 0.0:
        raise ValueError("grip-segment length must be nonnegative")
    if w0 < 0.0:
        raise ValueError("grip segment requires w0 >= 0")

    x0 = w0 * MU_G_INV
    z0 = k0 * x0
    margin = math.fma(-z0, z0, 1.0)
    if boundary_start:
        if k0 == 0.0 or sigma == 0.0 or not sigma / k0 < 0.0:
            raise ValueError(
                "boundary grip startup requires k0 != 0, sigma != 0, and sigma/k0 < 0"
            )
        expected = MU_G / abs(k0)
        tol = 64.0 * math.ulp(max(expected, abs(w0), 1.0))
        if abs(w0 - expected) > tol:
            raise ValueError(
                "boundary grip initial speed must equal the friction cap: "
                f"w0={w0!r}, expected={expected!r}"
            )
        w0 = expected
    elif not margin > 0.0:
        raise ValueError(
            "initial grip state is outside or on the real-domain boundary: "
            f"1 - (k0*w0/MU_G)^2 = {margin!r}"
        )
    return L, sigma, w0, k0


def _validate_ds(ds: float, L: float) -> float:
    ds = float(ds)
    if not math.isfinite(ds):
        raise ValueError("segment prefix must be finite")
    tol = 64.0 * math.ulp(max(1.0, abs(L)))
    if ds < -tol or ds > L + tol:
        raise ValueError(f"segment prefix {ds!r} outside [0, {L!r}]")
    if ds <= 0.0:
        return 0.0
    if ds >= L:
        return L
    return ds


def _state_rate(x: float, ds: float, k0: float, sigma: float) -> float:
    q = math.fma(sigma, ds, k0)
    z = q * x
    return TWO_MU_G * math.sqrt(max(0.0, math.fma(-z, z, 1.0)))


def _status_error(status: int, ds: float, event_time: float) -> Exception:
    event = event_time if math.isfinite(event_time) else None
    if status in (Status.EVENT, Status.BEYOND_EVENT, Status.OUTSIDE_REAL_DOMAIN):
        return GripDomainError(
            f"GRIP prefix ds={ds!r} is not a differentiable pre-event Cflow request "
            f"({status_name(status)})",
            status=status,
            event_position=event,
        )
    if status == Status.CONDITIONING_LIMIT:
        return GripConditioningError(
            f"GRIP prefix ds={ds!r} exceeds Cflow's binary64 conditioning limit"
        )
    return GripCflowError(
        f"Cflow failed for GRIP prefix ds={ds!r}: {status_name(status)}"
    )


def _boundary_next_event(x0: float, q0: float, b: float, horizon: float) -> float | None:
    """Return the first event strictly after an inward exact-boundary start.

    Cflow's ordinary event API correctly regards the initial boundary as an
    event.  For the application's repelling/inward boundary semantics, move a
    tiny deterministic fraction along the dedicated boundary continuation and
    then query the ordinary event operator from that interior state.
    """
    if horizon <= 0.0:
        return None
    scale = abs(q0 / b) if b != 0.0 else horizon
    eps = min(horizon * 1.0e-8, scale * 1.0e-8)
    if not eps > 0.0:
        eps = math.nextafter(0.0, 1.0)
    if eps >= horizon:
        return None
    r0 = eval_raw(x0, q0, b, eps, boundary=True)
    if r0.status != Status.OK:
        return None
    q1 = math.fma(b, eps, q0)
    ev = first_event_raw(r0.x, q1, b, horizon - eps)
    if ev.status in (Status.EVENT, Status.CONDITIONING_LIMIT) and math.isfinite(ev.time):
        return eps + ev.time
    return None


class _GripCommon:
    def _init_grip_common(
        self,
        L: float,
        sigma: float,
        w0: float,
        k0: float,
        *,
        boundary_start: bool,
    ) -> None:
        L, sigma, w0, k0 = _validate_grip_inputs(
            L, sigma, w0, k0, boundary_start=boundary_start
        )
        self.L = L
        self.sigma = sigma
        self.w0 = w0
        self.k0 = k0
        self.x0 = w0 * MU_G_INV
        self._boundary_start = bool(boundary_start)

    def _probe(self, ds: float) -> GripDomainProbe:
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            return GripDomainProbe(True, None, Status.OK)
        r = eval_raw(self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start)
        if r.status == Status.OK:
            return GripDomainProbe(True, None, Status.OK)
        if r.status == Status.EVENT:
            return GripDomainProbe(False, ds if not math.isfinite(r.event_time) else r.event_time, r.status)
        if r.status == Status.BEYOND_EVENT:
            return GripDomainProbe(False, r.event_time if math.isfinite(r.event_time) else None, r.status)
        if self._boundary_start and r.status == Status.NUMERICAL_FAILURE:
            event = _boundary_next_event(self.x0, self.k0, self.sigma, ds)
            if event is not None:
                return GripDomainProbe(False, event, Status.BEYOND_EVENT)
        if r.status == Status.CONDITIONING_LIMIT:
            raise GripConditioningError("Cflow conditioning limit while probing GRIP domain")
        raise _status_error(r.status, ds, r.event_time)

    def domain_probe(self, ds: float | None = None) -> GripDomainProbe:
        return self._probe(self.L if ds is None else ds)


class GripEvalSegment(_GripCommon, EvalSegment):
    """Evaluation-only GRIP segment with semigroup continuation anchors.

    Prefix states are exact numerical Cflow endpoints, not interpolants.  An
    uncached query continues from the nearest cached station on its left, so
    repeated scans traverse each portion of a trajectory only when necessary.
    The cache is instance-local acceleration state; numerical correctness never
    depends on its contents.  Concurrent mutation of one instance is therefore
    not supported, while the underlying native Cflow library remains thread-safe.
    """

    def __init__(self, L, sigma, w0, k0, *, boundary_start: bool = False):
        super().__init__(L, sigma, w0, k0, SegmentType.GRIP)
        self._init_grip_common(L, sigma, w0, k0, boundary_start=boundary_start)
        self._prefix_s = [0.0]
        self._prefix_x = [self.x0]
        self._cflow_calls = 0
        self._cflow_steps = 0
        self._cache_hits = 0

    def _eval_from_anchor(self, ds: float):
        i = bisect_right(self._prefix_s, ds) - 1
        si = self._prefix_s[i]
        xi = self._prefix_x[i]
        if si == ds:
            self._cache_hits += 1
            return xi, Status.OK, math.nan

        delta = ds - si
        qi = math.fma(self.sigma, si, self.k0)
        boundary = self._boundary_start and si == 0.0
        r = eval_raw(xi, qi, self.sigma, delta, boundary=boundary)
        self._cflow_calls += 1
        self._cflow_steps += int(r.steps)
        event_time = (si + r.event_time) if math.isfinite(r.event_time) else math.nan
        if r.status == Status.OK:
            j = bisect_right(self._prefix_s, ds)
            self._prefix_s.insert(j, ds)
            self._prefix_x.insert(j, r.x)
        return r.x, r.status, event_time

    def w(self, ds: float) -> float:
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            self._cache_hits += 1
            return self.w0
        x, status, event_time = self._eval_from_anchor(ds)
        if status not in (Status.OK, Status.EVENT):
            raise _status_error(status, ds, event_time)
        return MU_G * x

    def time(self, ds: float) -> float:
        """Return scalar travel time for one prefix using Cflow's additive map.

        Scalar evaluation uses Cflow's dedicated value-only additive map.
        Endpoint and integral sensitivities remain exclusive to the
        differentiable/Jacobian path; this call also leaves the authoritative
        state-prefix cache unchanged.
        """
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            return 0.0
        r = integral_value_raw(
            self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start
        )
        if r.status not in (Status.OK, Status.EVENT):
            raise _status_error(r.status, ds, r.event_time)
        return INV_SQRT_MU_G * r.integral

    def _prefix_cache_stats(self) -> dict[str, int]:
        """Development/test instrumentation; not part of the segment protocol."""
        return {
            "stations": len(self._prefix_s),
            "cache_hits": self._cache_hits,
            "cflow_calls": self._cflow_calls,
            "local_steps": self._cflow_steps,
        }


class GripSegment(_GripCommon, DiffSegment):
    """Differentiable general GRIP segment using the four native Cflow hot paths."""

    def __init__(
        self,
        L,
        sigma,
        w0,
        k0,
        *,
        x0_sensitivity_policy: TimeX0Policy = "negative_infinity",
        boundary_start: bool = False,
        reverse_eta: bool = False,
        authoritative_w1: float | None = None,
    ):
        super().__init__(L, sigma, w0, k0, SegmentType.GRIP)
        if x0_sensitivity_policy not in ("negative_infinity", "raise", "renormalized"):
            raise ValueError(f"unknown x0_sensitivity_policy: {x0_sensitivity_policy!r}")
        self._init_grip_common(L, sigma, w0, k0, boundary_start=boundary_start)
        self._x0_sensitivity_policy = x0_sensitivity_policy
        self._reverse_eta = bool(reverse_eta) and not self._boundary_start
        self._authoritative_w1 = (
            None if authoritative_w1 is None else float(authoritative_w1)
        )
        self._endpoint_value: float | None = None
        self._endpoint_all = None
        self._eta_prefix_cache: dict[float, object] = {}

    @property
    def time_w0_channel_kind(self) -> str:
        return "renormalized" if self.w0 == 0.0 and self._x0_sensitivity_policy == "renormalized" else "raw"

    def w(self, ds: float) -> float:
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            return self.w0
        if ds == self.L and self._endpoint_value is not None:
            return self._endpoint_value
        r = eval_raw(self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start)
        if r.status not in (Status.OK, Status.EVENT):
            raise _status_error(r.status, ds, r.event_time)
        w = MU_G * r.x
        if ds == self.L and r.status == Status.OK:
            self._endpoint_value = w
        return w

    def _reverse_eta_result(self, ds: float):
        """Return a qualified reverse-only eta result, or ``None`` to fall back.

        This path is enabled only for differentiable time replay compiled from
        an already-successful scalar topology build.  It never owns public
        Cflow status/event semantics.
        """
        if not self._reverse_eta:
            return None
        if not should_use_reverse_eta(
            self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start
        ):
            return None
        cached = self._eta_prefix_cache.get(ds)
        if cached is not None:
            return cached
        out = reverse_eta_all_raw(self.x0, self.k0, self.sigma, ds)
        if out is None:
            return None
        if not all(
            math.isfinite(v)
            for v in (
                out.x, out.dx_dx0, out.dx_dq0, out.dx_db, out.dx_dh,
                out.integral, out.dI_dx0, out.dI_dq0, out.dI_db, out.dI_dh,
                out.regularized_dI_dx0,
            )
        ):
            return None
        if ds == self.L and self._authoritative_w1 is not None:
            w_eta = MU_G * out.x
            # The scalar build remains endpoint authority.  This check catches
            # implementation errors or use outside the qualified chart without
            # requiring a second production Cflow traversal.
            if abs(w_eta - self._authoritative_w1) > 1.0e-8:
                return None
        self._eta_prefix_cache[ds] = out
        return out

    def _eta_full_endpoint(self):
        r = self._reverse_eta_result(self.L)
        if r is None:
            return None
        time_w0 = self._time_w0(r.dI_dx0, r.regularized_dI_dx0)
        state = (
            MU_G * r.dx_dh, MU_G * r.dx_db, r.dx_dx0, MU_G * r.dx_dq0,
            self.sigma, self.L, 0.0, 1.0,
        )
        time_jac = (
            r.dI_dh * INV_SQRT_MU_G, r.dI_db * INV_SQRT_MU_G,
            time_w0, r.dI_dq0 * INV_SQRT_MU_G,
        )
        return (MU_G * r.x, state, r.integral * INV_SQRT_MU_G, time_jac)

    def _full_endpoint(self):
        if self._endpoint_all is not None:
            return self._endpoint_all
        eta_full = self._eta_full_endpoint()
        if eta_full is not None:
            self._endpoint_all = eta_full
            self._endpoint_value = eta_full[0]
            return eta_full
        ds = self.L
        r = all_raw(self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start)
        if r.status != Status.OK:
            raise _status_error(r.status, ds, r.event_time)
        time_w0 = self._time_w0(r.dI_dx0, r.regularized_dI_dx0)
        state = (
            MU_G * r.dx_dh, MU_G * r.dx_db, r.dx_dx0, MU_G * r.dx_dq0,
            self.sigma, ds, 0.0, 1.0,
        )
        time_jac = (
            r.dI_dh * INV_SQRT_MU_G, r.dI_db * INV_SQRT_MU_G,
            time_w0, r.dI_dq0 * INV_SQRT_MU_G,
        )
        self._endpoint_all = (MU_G * r.x, state, r.integral * INV_SQRT_MU_G, time_jac)
        self._endpoint_value = self._endpoint_all[0]
        return self._endpoint_all

    def w_and_jac(self, ds: float):
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            return self.w0, (
                _state_rate(self.x0, 0.0, self.k0, self.sigma),
                0.0,
                1.0,
                0.0,
                self.sigma,
                0.0,
                0.0,
                1.0,
            )
        if ds == self.L:
            full = self._full_endpoint()
            return full[0], full[1]
        r = eval_jac_raw(self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start)
        if r.status != Status.OK:
            raise _status_error(r.status, ds, r.event_time)
        out = (
            MU_G * r.x,
            (
                MU_G * r.dx_dh,
                MU_G * r.dx_db,
                r.dx_dx0,
                MU_G * r.dx_dq0,
                self.sigma,
                ds,
                0.0,
                1.0,
            ),
        )
        return out

    def _time_w0(self, raw: float, regularized: float) -> float:
        if self._boundary_start:
            return 0.0
        if self.w0 != 0.0:
            return raw * INV_MU_G_SQRT_MU_G
        if self._x0_sensitivity_policy == "raise":
            raise ArithmeticError(
                "T_w0 is singular at w0=0; use negative_infinity or renormalized"
            )
        if self._x0_sensitivity_policy == "renormalized":
            return regularized * INV_MU_G_SQRT_MU_G
        return -math.inf

    def time_and_jac(self, ds: float):
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            return 0.0, (
                math.inf if self.w0 == 0.0 else 1.0 / math.sqrt(self.w0),
                0.0,
                0.0,
                0.0,
            )
        if ds == self.L:
            full = self._full_endpoint()
            return full[2], full[3]
        eta = self._reverse_eta_result(ds)
        if eta is not None:
            time_w0 = self._time_w0(eta.dI_dx0, eta.regularized_dI_dx0)
            return (
                eta.integral * INV_SQRT_MU_G,
                (
                    eta.dI_dh * INV_SQRT_MU_G,
                    eta.dI_db * INV_SQRT_MU_G,
                    time_w0,
                    eta.dI_dq0 * INV_SQRT_MU_G,
                ),
            )
        r = integral_jac_raw(self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start)
        if r.status != Status.OK:
            raise _status_error(r.status, ds, r.event_time)
        time_w0 = self._time_w0(r.dI_dx0, r.regularized_dI_dx0)
        out = (
            r.integral * INV_SQRT_MU_G,
            (
                r.dI_dh * INV_SQRT_MU_G,
                r.dI_db * INV_SQRT_MU_G,
                time_w0,
                r.dI_dq0 * INV_SQRT_MU_G,
            ),
        )
        return out

    def renormalized_time_w0_sensitivity(self, ds: float) -> float:
        ds = _validate_ds(ds, self.L)
        if ds == 0.0 or self._boundary_start:
            return 0.0
        eta = self._reverse_eta_result(ds)
        if eta is not None:
            return eta.regularized_dI_dx0 * INV_MU_G_SQRT_MU_G
        r = integral_jac_raw(self.x0, self.k0, self.sigma, ds, boundary=False)
        if r.status != Status.OK:
            raise _status_error(r.status, ds, r.event_time)
        return r.regularized_dI_dx0 * INV_MU_G_SQRT_MU_G

    def state_time_and_jac(self, ds: float):
        ds = _validate_ds(ds, self.L)
        if ds == 0.0:
            w0_jac = 1.0
            k0_w = 0.0
            return (
                self.w0,
                (
                    _state_rate(self.x0, 0.0, self.k0, self.sigma),
                    0.0,
                    w0_jac,
                    k0_w,
                    self.sigma,
                    0.0,
                    0.0,
                    1.0,
                ),
                0.0,
                (
                    math.inf if self.w0 == 0.0 else 1.0 / math.sqrt(self.w0),
                    0.0,
                    0.0,
                    0.0,
                ),
            )
        if ds == self.L:
            return self._full_endpoint()
        r = all_raw(self.x0, self.k0, self.sigma, ds, boundary=self._boundary_start)
        if r.status != Status.OK:
            raise _status_error(r.status, ds, r.event_time)
        time_w0 = self._time_w0(r.dI_dx0, r.regularized_dI_dx0)
        state = (
            MU_G * r.dx_dh,
            MU_G * r.dx_db,
            r.dx_dx0,
            MU_G * r.dx_dq0,
            self.sigma,
            ds,
            0.0,
            1.0,
        )
        time_jac = (
            r.dI_dh * INV_SQRT_MU_G,
            r.dI_db * INV_SQRT_MU_G,
            time_w0,
            r.dI_dq0 * INV_SQRT_MU_G,
        )
        out = (MU_G * r.x, state, r.integral * INV_SQRT_MU_G, time_jac)
        return out


__all__ = [
    "MU_G",
    "GripCflowError",
    "GripConditioningError",
    "GripDomainError",
    "GripDomainProbe",
    "GripEvalSegment",
    "GripSegment",
]
