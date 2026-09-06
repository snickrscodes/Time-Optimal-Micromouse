"""Lean scalar-only speed-profile travel-time evaluation.

The differentiable reverse solver already separates scalar topology discovery
from differentiable replay.  This module reuses the authoritative scalar
builder, then integrates ``1 / sqrt(w)`` over the selected lower-envelope
intervals without compiling differentiable segments, constructing Jacobians,
or running a reverse sweep.

The public :func:`evaluate_time_scalar` API mirrors the numerical and domain
arguments of :func:`optimization.reverse_solver.time_value_and_gradient` but
returns only the scalar travel time.  A disabled-by-default cross-check mode is
provided for tests and rollout validation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Sequence

from . import reverse_solver

SCALAR_TIME_INTEGRATION = "scalar_time_integration"

# The differentiable time compiler targets approximately 1e-12 local channel
# accuracy by default.  The scalar quadrature is intentionally tighter while
# remaining above QUADPACK's roundoff-sensitive floor on ordinary intervals.
DEFAULT_QUADRATURE_ABS_TOL = 1.0e-13
DEFAULT_QUADRATURE_REL_TOL = 1.0e-13
DEFAULT_QUADRATURE_LIMIT = 160

# Debug cross-check tolerance.  Release campaigns report the observed error;
# this guard is deliberately strict enough to catch ranking-relevant changes.
DEFAULT_CROSS_CHECK_ABS_TOL = 5.0e-9
DEFAULT_CROSS_CHECK_REL_TOL = 5.0e-10


@dataclass(frozen=True, slots=True)
class ScalarTimeDiagnostics:
    """Non-gradient diagnostics for one scalar travel-time evaluation."""

    value: float
    topology_seconds: float
    integration_seconds: float
    envelope_intervals: int
    integrand_evaluations: int
    maximum_reported_quadrature_error: float
    scalar_passes: int
    scalar_segments: int


def _finite_positive_speed(segment: Any, station: float) -> float:
    try:
        value = float(segment.w(station))
    except Exception as error:  # preserve fail-closed public classification
        raise FloatingPointError(
            f"scalar speed evaluation failed at s={station:.17g}"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError(
            f"scalar speed must be finite and positive at s={station:.17g}, "
            f"got {value!r}"
        )
    return value


def _segment_breakpoints(segment: Any, lo: float, hi: float) -> tuple[float, ...]:
    """Return optional deterministic integration breakpoints when exposed.

    Cflow evaluation segments do not require internal breakpoints; the hook is
    retained generically for analytic segment families that may expose them.
    """
    ends: Iterable[Any] = getattr(segment, "integration_breakpoints", ())
    points: list[float] = []
    for raw in ends:
        try:
            point = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(point) and lo < point < hi:
            points.append(point)
    if not points:
        return ()
    return tuple(sorted(set(points)))


def _integrate_segment_interval(
    segment: Any,
    local0: float,
    local1: float,
    *,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float, int]:
    # Preserve optimization package lazy-import behavior: SciPy is loaded only
    # when the scalar evaluator is actually invoked.
    from scipy.integrate import IntegrationWarning, quad

    lo, hi = sorted((float(local0), float(local1)))
    if hi - lo <= 1.0e-14 * max(1.0, abs(lo), abs(hi)):
        return 0.0, 0.0, 0

    scalar_time = getattr(segment, "time", None)
    if callable(scalar_time):
        value = float(scalar_time(hi)) - float(scalar_time(lo))
        if not math.isfinite(value) or value < -64.0 * math.ulp(max(1.0, abs(value))):
            raise FloatingPointError(f"scalar segment time returned {value!r}")
        return max(0.0, value), 0.0, 2

    def integrand(station: float) -> float:
        return 1.0 / math.sqrt(_finite_positive_speed(segment, station))

    points = _segment_breakpoints(segment, lo, hi)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        result = quad(
            integrand,
            lo,
            hi,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            points=points if points else None,
            full_output=1,
        )

    value = float(result[0])
    error = float(result[1])
    info = result[2]
    message = result[3] if len(result) > 3 else None
    integration_warnings = [
        item for item in caught if issubclass(item.category, IntegrationWarning)
    ]
    if message is not None or integration_warnings:
        detail = str(message) if message is not None else str(integration_warnings[0].message)
        raise FloatingPointError(f"scalar time quadrature failed: {detail}")
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError(f"scalar time quadrature returned {value!r}")
    if not math.isfinite(error) or error < 0.0:
        raise FloatingPointError(
            f"scalar time quadrature reported invalid error {error!r}"
        )
    return value, error, int(info.get("neval", 0))


def scalar_time_from_build(
    build: reverse_solver.ProdSpeedProfileBuild,
    *,
    quadrature_abs_tolerance: float = DEFAULT_QUADRATURE_ABS_TOL,
    quadrature_rel_tolerance: float = DEFAULT_QUADRATURE_REL_TOL,
    quadrature_limit: int = DEFAULT_QUADRATURE_LIMIT,
    profiler: reverse_solver.PhaseProfiler | None = None,
) -> tuple[float, ScalarTimeDiagnostics]:
    """Evaluate only the scalar time of an already-built speed profile."""
    epsabs = float(quadrature_abs_tolerance)
    epsrel = float(quadrature_rel_tolerance)
    if not math.isfinite(epsabs) or epsabs <= 0.0:
        raise ValueError("quadrature_abs_tolerance must be finite and positive")
    if not math.isfinite(epsrel) or epsrel <= 0.0:
        raise ValueError("quadrature_rel_tolerance must be finite and positive")
    if isinstance(quadrature_limit, bool) or int(quadrature_limit) != quadrature_limit:
        raise ValueError("quadrature_limit must be an integer")
    limit = int(quadrature_limit)
    if limit < 16:
        raise ValueError("quadrature_limit must be at least 16")

    if isinstance(build, reverse_solver.NativeProdSpeedProfileBuild):
        # The native build owns exact segment-prefix time evaluation.  Each
        # envelope interval performs two native prefix-time evaluations rather
        # than adaptive Python quadrature, so diagnostics report that work
        # explicitly while preserving the existing nonzero-evaluation contract.
        started = perf_counter()
        try:
            value = float(build.native.time_value())
        except Exception as exc:
            raise reverse_solver._translate_native_error(exc) from exc
        elapsed = perf_counter() - started
        stats = build.native.stats()
        intervals = int(build.envelope_count)
        if profiler is not None and profiler.enabled:
            profiler.record(SCALAR_TIME_INTEGRATION, elapsed, count=max(1, intervals))
        return value, ScalarTimeDiagnostics(
            value=value,
            topology_seconds=0.0,
            integration_seconds=float(elapsed),
            envelope_intervals=intervals,
            integrand_evaluations=2 * intervals,
            maximum_reported_quadrature_error=0.0,
            scalar_passes=int(stats["scalar_passes"]),
            scalar_segments=int(stats["scalar_segments"]),
        )

    started = perf_counter()
    value = 0.0
    interval_count = 0
    integrand_evaluations = 0
    maximum_error = 0.0

    for envelope_piece in build.envelope:
        pass_index = int(envelope_piece.pass_index)
        source_index = int(envelope_piece.source_index)
        if not 0 <= pass_index < len(build.scalar_passes):
            raise ValueError(f"bad envelope pass index: {pass_index}")
        scalar_pass = build.scalar_passes[pass_index]
        if not 0 <= source_index < len(scalar_pass.segments):
            raise ValueError(f"bad envelope source index: {source_index}")
        record = scalar_pass.segments[source_index]
        if record.seg is None:
            raise ValueError("scalar envelope segment is missing its evaluator")
        interval_value, error, evaluations = _integrate_segment_interval(
            record.seg,
            envelope_piece.local0,
            envelope_piece.local1,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        value += interval_value
        if evaluations:
            interval_count += 1
            integrand_evaluations += evaluations
            maximum_error = max(maximum_error, error)

    elapsed = perf_counter() - started
    if profiler is not None and profiler.enabled:
        profiler.record(SCALAR_TIME_INTEGRATION, elapsed, count=interval_count)
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError(f"scalar travel time is invalid: {value!r}")

    diagnostics = ScalarTimeDiagnostics(
        value=float(value),
        topology_seconds=0.0,
        integration_seconds=float(elapsed),
        envelope_intervals=interval_count,
        integrand_evaluations=integrand_evaluations,
        maximum_reported_quadrature_error=float(maximum_error),
        scalar_passes=len(build.scalar_passes),
        scalar_segments=sum(len(item.segments) for item in build.scalar_passes),
    )
    return float(value), diagnostics


def evaluate_time_scalar_result(
    raw_params: Sequence[float],
    *,
    init_w: float | None = None,
    terminal_w_max: float | None = None,
    initial_k: float = 0.0,
    n_scan: int = 256,
    envelope_scan: int = 64,
    domain_margin: float = reverse_solver.FRICTION_DOMAIN_MARGIN,
    domain_scan: int = reverse_solver.FRICTION_DOMAIN_SCAN,
    quadrature_abs_tolerance: float = DEFAULT_QUADRATURE_ABS_TOL,
    quadrature_rel_tolerance: float = DEFAULT_QUADRATURE_REL_TOL,
    quadrature_limit: int = DEFAULT_QUADRATURE_LIMIT,
    profiler: reverse_solver.PhaseProfiler | None = None,
    cross_check: bool = False,
    cross_check_abs_tolerance: float = DEFAULT_CROSS_CHECK_ABS_TOL,
    cross_check_rel_tolerance: float = DEFAULT_CROSS_CHECK_REL_TOL,
) -> ScalarTimeDiagnostics:
    """Return scalar travel time plus non-gradient diagnostics.

    ``cross_check`` is intended only for tests and staged rollout validation.
    Normal value-only evaluation never invokes the differentiable solver.
    """
    topology_started = perf_counter()
    build = reverse_solver.build_scalar_speed_profile(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        forward_init_k=initial_k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_margin=domain_margin,
        domain_scan=domain_scan,
        profiler=profiler,
    )
    topology_seconds = perf_counter() - topology_started
    try:
        value, diagnostics = scalar_time_from_build(
            build,
            quadrature_abs_tolerance=quadrature_abs_tolerance,
            quadrature_rel_tolerance=quadrature_rel_tolerance,
            quadrature_limit=quadrature_limit,
            profiler=profiler,
        )
    finally:
        if isinstance(build, reverse_solver.NativeProdSpeedProfileBuild):
            build.close()
    diagnostics = ScalarTimeDiagnostics(
        value=diagnostics.value,
        topology_seconds=float(topology_seconds),
        integration_seconds=diagnostics.integration_seconds,
        envelope_intervals=diagnostics.envelope_intervals,
        integrand_evaluations=diagnostics.integrand_evaluations,
        maximum_reported_quadrature_error=diagnostics.maximum_reported_quadrature_error,
        scalar_passes=diagnostics.scalar_passes,
        scalar_segments=diagnostics.scalar_segments,
    )

    if cross_check:
        reference, _gradient = reverse_solver.time_value_and_gradient(
            raw_params,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_margin=domain_margin,
            domain_scan=domain_scan,
        )
        atol = float(cross_check_abs_tolerance)
        rtol = float(cross_check_rel_tolerance)
        if not math.isclose(value, float(reference), abs_tol=atol, rel_tol=rtol):
            raise AssertionError(
                "scalar and differentiable travel times disagree: "
                f"scalar={value:.17g}, differentiable={reference:.17g}, "
                f"abs_diff={abs(value - reference):.17g}, atol={atol:.17g}, "
                f"rtol={rtol:.17g}"
            )
    return diagnostics


def evaluate_time_scalar(
    raw_params: Sequence[float],
    **kwargs: Any,
) -> float:
    """Evaluate authoritative travel time without constructing gradients."""
    return evaluate_time_scalar_result(raw_params, **kwargs).value


# A concise compatibility spelling for callers that already use ``scalar_time``.
scalar_time = evaluate_time_scalar


__all__ = [
    "DEFAULT_CROSS_CHECK_ABS_TOL",
    "DEFAULT_CROSS_CHECK_REL_TOL",
    "DEFAULT_QUADRATURE_ABS_TOL",
    "DEFAULT_QUADRATURE_LIMIT",
    "DEFAULT_QUADRATURE_REL_TOL",
    "SCALAR_TIME_INTEGRATION",
    "ScalarTimeDiagnostics",
    "evaluate_time_scalar",
    "evaluate_time_scalar_result",
    "scalar_time",
    "scalar_time_from_build",
]
