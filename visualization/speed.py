"""Backend-independent presentation trace for the production speed solution.

The trace intentionally uses the Python scalar profile build because it exposes
its retained envelope/pass topology.  Numerical timing samples are only for
visualization/time-parameterization; the authoritative travel time remains the
production objective and is cross-checked before a trace is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import reverse_solver
from segment.base import SegmentType
from segment.constants import A_BRAKE, A_MAX, B_EMF, MU_G

Array = NDArray[np.float64]
ObjectArray = NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class SpeedModeInterval:
    station0: float
    station1: float
    mode: str
    pass_index: int
    source_index: int


@dataclass(frozen=True, slots=True)
class SpeedEvent:
    station: float
    kind: str
    from_mode: str
    to_mode: str
    from_pass: int
    to_pass: int


@dataclass(frozen=True, slots=True)
class SpeedProfileTrace:
    s: Array
    t: Array
    w: Array
    v: Array
    acceleration: Array
    kappa: Array
    mode: tuple[str, ...]
    intervals: tuple[SpeedModeInterval, ...]
    events: tuple[SpeedEvent, ...]
    total_length: float
    exact_total_time: float
    sampled_total_time: float
    integration_error: float

    def __post_init__(self) -> None:
        n = self.s.size
        for name in ("t", "w", "v", "acceleration", "kappa"):
            if getattr(self, name).size != n:
                raise ValueError(f"SpeedProfileTrace.{name} length mismatch")
        if len(self.mode) != n:
            raise ValueError("SpeedProfileTrace.mode length mismatch")
        if np.any(np.diff(self.s) < 0.0) or np.any(np.diff(self.t) < -1e-12):
            raise ValueError("speed trace station/time must be monotone")

    def station_at_time(self, times: Sequence[float]) -> Array:
        """Monotone linear interpolation sufficient for animation playback."""
        query = np.asarray(times, dtype=float)
        if np.any(query < -1e-12) or np.any(query > self.exact_total_time + 1e-12):
            raise ValueError("time query lies outside the speed trace")
        query = np.clip(query, 0.0, self.exact_total_time)
        # Scale sampled time to the authoritative exact endpoint.  The error is
        # already required to be tiny; this makes the final animation frame
        # land exactly on the path endpoint.
        scale = 1.0 if self.sampled_total_time == 0.0 else self.exact_total_time / self.sampled_total_time
        return np.interp(query, self.t * scale, self.s)


def _piece_samples(abs0: float, abs1: float, density: float, minimum: int) -> Array:
    length = max(0.0, abs1 - abs0)
    count = max(minimum, int(math.ceil(density * length)) + 1)
    return np.linspace(abs0, abs1, count)


def _safe_physical_dw_ds(rec: reverse_solver.ProdScalarSegment, station: float, w: float, kappa: float) -> float:
    """Physical dw/ds, tolerant of ulp-scale friction-boundary contact."""

    if rec.mode is SegmentType.MOTOR:
        local = 2.0 * (A_MAX - B_EMF * math.sqrt(w))
    elif rec.mode is SegmentType.BRAKE:
        local = 2.0 * A_BRAKE
    elif rec.mode is SegmentType.GRIP:
        g2 = MU_G * MU_G - (w * kappa) ** 2
        scale = max(MU_G * MU_G, (w * kappa) ** 2, 1.0)
        tol = 128.0 * math.ulp(scale)
        if g2 < -tol:
            raise FloatingPointError(
                f"GRIP visualization sample outside friction domain at s={station}: G2={g2}"
            )
        local = 2.0 * math.sqrt(max(0.0, g2))
    else:
        raise ValueError(rec.mode)
    return rec.direction * local


def _sample_build(build: reverse_solver.ProdSpeedProfileBuild, *, samples_per_unit: float, minimum_samples_per_piece: int):
    stations: list[float] = []
    values: list[float] = []
    accelerations: list[float] = []
    kappas: list[float] = []
    modes: list[str] = []
    events: list[SpeedEvent] = []
    intervals: list[SpeedModeInterval] = []

    previous_ep = None
    for envelope_index, ep in enumerate(build.envelope):
        rec = build.scalar_passes[ep.pass_index].segments[ep.source_index]
        local_stations = _piece_samples(ep.abs0, ep.abs1, samples_per_unit, minimum_samples_per_piece)
        if envelope_index:
            local_stations = local_stations[1:]
        intervals.append(
            SpeedModeInterval(
                station0=float(ep.abs0),
                station1=float(ep.abs1),
                mode=reverse_solver.mode_name(rec.mode),
                pass_index=int(ep.pass_index),
                source_index=int(ep.source_index),
            )
        )
        if previous_ep is not None:
            prev_rec = build.scalar_passes[previous_ep.pass_index].segments[previous_ep.source_index]
            if prev_rec.mode is not rec.mode or previous_ep.pass_index != ep.pass_index:
                events.append(
                    SpeedEvent(
                        station=float(ep.abs0),
                        kind=("mode_switch" if prev_rec.mode is not rec.mode else "envelope_switch"),
                        from_mode=reverse_solver.mode_name(prev_rec.mode),
                        to_mode=reverse_solver.mode_name(rec.mode),
                        from_pass=int(previous_ep.pass_index),
                        to_pass=int(ep.pass_index),
                    )
                )
        for station in local_stations:
            ds = reverse_solver.abs_to_local(rec, float(station))
            w = float(rec.seg.w(ds))
            if not math.isfinite(w) or w <= 0.0:
                raise FloatingPointError(f"nonpositive speed-squared in visualization trace at s={station}: {w}")
            kappa = math.fma(rec.sigma, ds, rec.k0)
            dw_abs = _safe_physical_dw_ds(rec, float(station), w, kappa)
            stations.append(float(station))
            values.append(w)
            accelerations.append(0.5 * float(dw_abs))
            kappas.append(float(kappa))
            modes.append(reverse_solver.mode_name(rec.mode))
        previous_ep = ep

    return (
        np.asarray(stations, dtype=float),
        np.asarray(values, dtype=float),
        np.asarray(accelerations, dtype=float),
        np.asarray(kappas, dtype=float),
        tuple(modes),
        tuple(intervals),
        tuple(events),
    )


def _cumulative_time(stations: Array, w: Array) -> Array:
    inv_v = 1.0 / np.sqrt(w)
    ds = np.diff(stations)
    increments = 0.5 * (inv_v[:-1] + inv_v[1:]) * ds
    return np.concatenate(([0.0], np.cumsum(increments)))


def build_speed_profile_trace(
    raw_parameters: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None = None,
    initial_k: float = 0.0,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_scan: int = 64,
    samples_per_unit: float = 200.0,
    minimum_samples_per_piece: int = 16,
    time_tolerance: float = 1e-6,
    maximum_refinements: int = 6,
) -> SpeedProfileTrace:
    """Build an introspectable trace and cross-check its timing quadrature.

    The authoritative scalar topology is generated with the Python reference
    backend regardless of the caller's active production backend, and the
    caller's backend is restored by the context manager.
    """

    if terminal_w_max is None:
        terminal_w_max = init_w
    if samples_per_unit <= 0.0 or minimum_samples_per_piece < 2:
        raise ValueError("invalid visualization sampling density")
    if time_tolerance <= 0.0:
        raise ValueError("time_tolerance must be positive")

    with reverse_solver.using_reverse_backend("python"):
        build = reverse_solver.build_scalar_speed_profile(
            raw_parameters,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            forward_init_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
        )
        exact_time = float(reverse_solver.time_value_and_gradient_from_build(build)[0])
        density = float(samples_per_unit)
        last = None
        for _ in range(maximum_refinements + 1):
            sampled = _sample_build(
                build,
                samples_per_unit=density,
                minimum_samples_per_piece=minimum_samples_per_piece,
            )
            s, w, acceleration, kappa, modes, intervals, events = sampled
            t = _cumulative_time(s, w)
            error = abs(float(t[-1]) - exact_time)
            last = (s, w, acceleration, kappa, modes, intervals, events, t, error)
            if error <= time_tolerance:
                break
            density *= 2.0
        assert last is not None
        s, w, acceleration, kappa, modes, intervals, events, t, error = last
        if error > time_tolerance:
            raise RuntimeError(
                "speed-profile visualization quadrature failed to converge: "
                f"sampled={float(t[-1]):.12g}, exact={exact_time:.12g}, error={error:.3g}"
            )
        return SpeedProfileTrace(
            s=s,
            t=t,
            w=w,
            v=np.sqrt(w),
            acceleration=acceleration,
            kappa=kappa,
            mode=modes,
            intervals=intervals,
            events=events,
            total_length=float(s[-1]),
            exact_total_time=exact_time,
            sampled_total_time=float(t[-1]),
            integration_error=float(error),
        )


def speed_profile_trace_from_dd_record(
    trace_record: dict,
    *,
    exact_total_time: float,
    cell_pitch_m: float,
) -> SpeedProfileTrace:
    """Convert a persisted Phase-6 DD/yaw visual trace into presentation form.

    DD/yaw production runs persist the independently replayed Python trace in
    the parent physics certificate.  Postprocessing therefore never rebuilds
    the expensive MVC catalog merely to draw V3/V4.
    """
    s = np.asarray(trace_record["s_grid"], dtype=float)
    w = np.asarray(trace_record["w_grid2_s2"], dtype=float)
    kappa = np.asarray(trace_record["kappa_per_grid"], dtype=float)
    acceleration = np.asarray(trace_record["acceleration_mps2"], dtype=float) / float(cell_pitch_m)
    modes = tuple(str(x) for x in trace_record["active_mode"])
    if not (s.ndim == w.ndim == kappa.ndim == acceleration.ndim == 1):
        raise ValueError("DD visual trace arrays must be one-dimensional")
    if not (len(s) == len(w) == len(kappa) == len(acceleration) == len(modes)):
        raise ValueError("DD visual trace array length mismatch")
    if len(s) < 2 or np.any(w <= 0.0):
        raise ValueError("DD visual trace requires positive speed samples")
    t = _cumulative_time(s, w)
    sampled = float(t[-1])
    intervals: list[SpeedModeInterval] = []
    events: list[SpeedEvent] = []
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            i1 = max(start, i - 1)
            intervals.append(SpeedModeInterval(float(s[start]), float(s[i1]), modes[start], -1, -1))
            if i < len(modes):
                events.append(SpeedEvent(float(s[i]), "mode_switch", modes[i-1], modes[i], -1, -1))
            start = i
    return SpeedProfileTrace(
        s=s,
        t=t,
        w=w,
        v=np.sqrt(w),
        acceleration=acceleration,
        kappa=kappa,
        mode=modes,
        intervals=tuple(intervals),
        events=tuple(events),
        total_length=float(s[-1]),
        exact_total_time=float(exact_total_time),
        sampled_total_time=sampled,
        integration_error=abs(sampled - float(exact_total_time)),
    )


try:
    __all__ += ["speed_profile_trace_from_dd_record"]
except NameError:
    pass
