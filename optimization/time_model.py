"""Profile-dispatched production time functional.

Legacy physics remains byte/behavior compatible with the historically qualified
reverse solver.  ``red_comet_2017_dd_yaw_v1`` routes through the isolated DD/yaw
one-state speed backend developed in Phases 1--5.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

from segment.physics_profiles import PhysicsProfile, get_physics_profile
from segment.physics_identity import (
    differential_drive_parameters_for_profile,
    physics_model_identity,
    physics_model_signature,
)

from . import reverse_solver, scalar_reverse_solver
from .dd_yaw_anchors import (
    DDCompleteProfile,
    build_complete_speed_profile,
    certify_complete_profile,
    envelope_w,
)
from .dd_yaw_gradients import (
    DDGradientTopologyError,
    time_value_and_raw_gradient_analytic,
)


class TimeModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TimeModelEvaluation:
    value: float
    raw_gradient: tuple[float, ...] | None
    model: str
    diagnostics: dict[str, Any]
    build: DDCompleteProfile | None = None


def current_physics_profile() -> PhysicsProfile:
    return get_physics_profile(os.environ.get("AME_PHYSICS_PROFILE", "legacy_grid_v1"))


def is_dd_yaw_profile(profile: PhysicsProfile | None = None) -> bool:
    p = current_physics_profile() if profile is None else profile
    return p.time_model == "dd_yaw_v1"


def _dd_segment_backend(*, independent: bool = False) -> str:
    if independent:
        return "python"
    explicit = os.environ.get("AME_DD_SEGMENT_BACKEND")
    if explicit:
        value = explicit.strip().lower()
        if value not in {"python", "native"}:
            raise TimeModelError("AME_DD_SEGMENT_BACKEND must be python or native")
        return value
    return "native" if os.environ.get("AME_REVERSE_BACKEND", "native").strip().lower() == "native" else "python"



def _dd_mvc_backend(*, independent: bool = False) -> str:
    if independent:
        return "python"
    explicit=os.environ.get("AME_DD_MVC_BACKEND")
    if explicit:
        value=explicit.strip().lower()
        if value not in {"python","native_scan"}:
            raise TimeModelError("AME_DD_MVC_BACKEND must be python or native_scan")
        return value
    # Production DD builds use the semantics-preserving bulk native scanner.
    # Independent parent certification deliberately remains on the Python
    # Phase-3 reference implementation.
    return "native_scan"

def _dd_build_kwargs(
    *,
    init_w: float,
    terminal_w_max: float | None,
    initial_k: float,
    n_scan: int,
    envelope_scan: int,
    domain_scan: int,
    independent: bool = False,
) -> dict[str, Any]:
    # DD/yaw qualification established 96/72/112/20 as the production scalar
    # discovery resolution.  Never silently inherit a cheaper caller preview.
    return {
        "init_w": float(init_w),
        "terminal_w_max": terminal_w_max,
        "initial_k": float(initial_k),
        "pass_scan": max(96, int(n_scan)),
        "anchor_scan": max(72, int(envelope_scan)),
        "cap_scan": max(112, int(domain_scan)),
        "envelope_root_scan": 20,
        "segment_backend": _dd_segment_backend(independent=independent),
        "mvc_backend": _dd_mvc_backend(independent=independent),
    }


def build_dd_profile(
    raw_params: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None,
    initial_k: float,
    n_scan: int = 96,
    envelope_scan: int = 48,
    domain_scan: int = 96,
    profile: PhysicsProfile | None = None,
    independent: bool = False,
) -> DDCompleteProfile:
    profile = current_physics_profile() if profile is None else profile
    if profile.time_model != "dd_yaw_v1":
        raise TimeModelError(f"profile {profile.name!r} does not select DD/yaw")
    params = differential_drive_parameters_for_profile(profile)
    return build_complete_speed_profile(
        raw_params,
        params,
        profile,
        **_dd_build_kwargs(
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
            independent=independent,
        ),
    )


def evaluate_time_scalar(
    raw_params: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None,
    initial_k: float,
    n_scan: int = 96,
    envelope_scan: int = 48,
    domain_scan: int = 96,
    profile: PhysicsProfile | None = None,
    independent: bool = False,
) -> float:
    profile = current_physics_profile() if profile is None else profile
    if profile.time_model != "dd_yaw_v1":
        return float(
            scalar_reverse_solver.evaluate_time_scalar(
                raw_params,
                init_w=init_w,
                terminal_w_max=terminal_w_max,
                initial_k=initial_k,
                n_scan=n_scan,
                envelope_scan=envelope_scan,
                domain_scan=domain_scan,
            )
        )
    return float(
        build_dd_profile(
            raw_params,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
            profile=profile,
            independent=independent,
        ).total_time
    )


def time_value_and_gradient(
    raw_params: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None,
    initial_k: float,
    n_scan: int = 96,
    envelope_scan: int = 48,
    domain_scan: int = 96,
    profile: PhysicsProfile | None = None,
) -> TimeModelEvaluation:
    profile = current_physics_profile() if profile is None else profile
    if profile.time_model != "dd_yaw_v1":
        value, grad = reverse_solver.time_value_and_gradient(
            raw_params,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
        )
        return TimeModelEvaluation(float(value), tuple(float(x) for x in grad), profile.time_model, {})

    build = build_dd_profile(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        initial_k=initial_k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_scan=domain_scan,
        profile=profile,
        independent=False,
    )
    try:
        value, grad = time_value_and_raw_gradient_analytic(build, clarke_ties=True)
    except DDGradientTopologyError as exc:
        # A genuine active-pair tie/kink has no unique classical derivative.
        # Production fails closed rather than silently substituting the legacy
        # time functional or an arbitrary one-sided derivative.
        raise TimeModelError(f"DD/yaw time objective is nondifferentiable at current topology: {exc}") from exc
    diagnostics = dict(build.diagnostics)
    diagnostics.update(
        physics_model_signature=physics_model_signature(profile),
        segment_backend=_dd_segment_backend(independent=False),
        mvc_backend=_dd_mvc_backend(independent=False),
    )
    return TimeModelEvaluation(float(value), tuple(float(x) for x in grad), profile.time_model, diagnostics, build)


def certify_dd_time_profile(
    raw_params: Sequence[float],
    *,
    init_w: float,
    terminal_w_max: float | None,
    initial_k: float,
    expected_time: float | None = None,
    n_scan: int = 96,
    envelope_scan: int = 48,
    domain_scan: int = 96,
    profile: PhysicsProfile | None = None,
    compare_native: bool = True,
    time_tolerance: float = 2.0e-8,
    include_trace: bool = False,
    trace_samples: int = 1601,
) -> dict[str, Any]:
    """Independently replay a DD/yaw result through the Python reference backend."""
    profile = current_physics_profile() if profile is None else profile
    if profile.time_model != "dd_yaw_v1":
        return {
            "certified": True,
            "model": profile.time_model,
            "physics_model_signature": physics_model_signature(profile),
            "note": "legacy profile: DD/yaw certification not applicable",
        }
    py = build_dd_profile(
        raw_params,
        init_w=init_w,
        terminal_w_max=terminal_w_max,
        initial_k=initial_k,
        n_scan=n_scan,
        envelope_scan=envelope_scan,
        domain_scan=domain_scan,
        profile=profile,
        independent=True,
    )
    cert = certify_complete_profile(py, n_samples=1025, margin_tol=2e-6)
    native_time = None
    native_python_delta = None
    if compare_native:
        native = build_dd_profile(
            raw_params,
            init_w=init_w,
            terminal_w_max=terminal_w_max,
            initial_k=initial_k,
            n_scan=n_scan,
            envelope_scan=envelope_scan,
            domain_scan=domain_scan,
            profile=profile,
            independent=False,
        )
        native_time = float(native.total_time)
        native_python_delta = native_time - float(py.total_time)
    expected_delta = None if expected_time is None else float(py.total_time) - float(expected_time)
    scale = max(1.0, abs(float(py.total_time)), 0.0 if expected_time is None else abs(float(expected_time)))
    time_ok = expected_time is None or abs(expected_delta) <= time_tolerance * scale
    parity_ok = native_python_delta is None or abs(native_python_delta) <= time_tolerance * scale
    min_margin = float(cert.get("min_interval_margin", math.inf))
    mvc = float(cert.get("max_mvc_violation", 0.0))
    certified = bool(time_ok and parity_ok and min_margin >= -2e-6 and mvc <= 2e-6 * scale)
    payload = {
        "certified": certified,
        "model": profile.time_model,
        "profile": profile.name,
        "physics_model_signature": physics_model_signature(profile),
        "physics_model_identity": physics_model_identity(profile),
        "python_reference_time": float(py.total_time),
        "native_time": native_time,
        "native_minus_python": native_python_delta,
        "expected_time": expected_time,
        "python_minus_expected": expected_delta,
        "time_tolerance": time_tolerance,
        "profile_certificate": cert,
        "diagnostics": dict(py.diagnostics),
    }
    if include_trace:
        payload["visual_trace"] = sample_dd_profile(py, samples=trace_samples)
    return payload


def sample_dd_profile(build: DDCompleteProfile, *, samples: int = 1601) -> dict[str, list[float] | list[str]]:
    """Dense, postprocessing-ready DD trace from an already certified profile."""
    from .dd_yaw_anchors import envelope_w
    from .dd_yaw_speed_profile import pass_record_at, record_local
    from segment.differential_drive import (
        acceleration_interval,
        side_acceleration_mps2,
        side_force_demand_n,
        side_state,
        yaw_acceleration,
        yaw_rate,
        DriveSide,
    )
    from .dd_yaw_speed_profile import candidate_value

    n = max(3, int(samples))
    T = build.total_length
    starts = [0.0]
    for i in range(len(build.raw_params)//2):
        starts.append(starts[-1] + build.raw_params[2*i])

    out: dict[str, list] = {k: [] for k in (
        "s_grid", "w_grid2_s2", "speed_mps", "kappa_per_grid", "sigma_per_grid2",
        "acceleration_mps2", "yaw_rate_rad_s", "yaw_acceleration_rad_s2",
        "left_speed_mps", "right_speed_mps", "left_accel_mps2", "right_accel_mps2",
        "left_force_n", "right_force_n", "interval_margin_grid_s2", "active_mode",
    )}
    for q in range(n):
        s = T * q / (n - 1)
        # Locate owning envelope piece/pass and geometry piece.
        ep = next((e for e in build.envelope if e.abs0-1e-12 <= s <= e.abs1+1e-12), build.envelope[-1])
        p = build.passes[ep.pass_index]
        rec = pass_record_at(p, s)
        if rec is None:
            rec = p.segments[-1]
        w = envelope_w(build, s)
        i = min(len(starts)-2, max(0, next((j for j in range(len(starts)-1) if starts[j]-1e-12 <= s <= starts[j+1]+1e-12), len(starts)-2)))
        local = s - starts[i]
        k0 = 0.0
        for j in range(i):
            k0 = math.fma(build.raw_params[2*j+1], build.raw_params[2*j], k0)
        sigma = build.raw_params[2*i+1]
        kappa = math.fma(sigma, local, k0)
        # The envelope owner gives the time-optimal signed acceleration.
        a = candidate_value(rec.mode, w, kappa, sigma, rec.kind, build.params, build.profile)
        ls = side_state(w, kappa, DriveSide.LEFT, build.params)
        rs = side_state(w, kappa, DriveSide.RIGHT, build.params)
        iv = acceleration_interval(w, kappa, sigma, build.params, build.profile)
        out["s_grid"].append(float(s)); out["w_grid2_s2"].append(float(w))
        out["speed_mps"].append(build.profile.cell_pitch_m * math.sqrt(w))
        out["kappa_per_grid"].append(float(kappa)); out["sigma_per_grid2"].append(float(sigma))
        out["acceleration_mps2"].append(build.profile.cell_pitch_m * a)
        out["yaw_rate_rad_s"].append(yaw_rate(w, kappa))
        out["yaw_acceleration_rad_s2"].append(yaw_acceleration(w, kappa, sigma, a))
        out["left_speed_mps"].append(ls.side_speed_mps); out["right_speed_mps"].append(rs.side_speed_mps)
        out["left_accel_mps2"].append(side_acceleration_mps2(a,w,kappa,sigma,DriveSide.LEFT,build.params))
        out["right_accel_mps2"].append(side_acceleration_mps2(a,w,kappa,sigma,DriveSide.RIGHT,build.params))
        out["left_force_n"].append(side_force_demand_n(a,w,kappa,sigma,DriveSide.LEFT,build.params))
        out["right_force_n"].append(side_force_demand_n(a,w,kappa,sigma,DriveSide.RIGHT,build.params))
        out["interval_margin_grid_s2"].append(iv.margin); out["active_mode"].append(rec.mode.name)
    return out


__all__ = [
    "TimeModelError",
    "TimeModelEvaluation",
    "current_physics_profile",
    "is_dd_yaw_profile",
    "build_dd_profile",
    "evaluate_time_scalar",
    "time_value_and_gradient",
    "certify_dd_time_profile",
    "sample_dd_profile",
]
