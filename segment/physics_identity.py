"""Deterministic production identity for physical time models.

The active-basis cache must never reuse a route result across a change in the
physical objective implementation.  DD/yaw v1 therefore hashes both numerical
parameters and the exact source files that define its Python/native kernels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .differential_drive import red_comet_dd_yaw_v1_parameters
from .physics_profiles import PhysicsProfile

DD_YAW_MODEL_REVISION = "red_comet_dd_yaw_v1_phase6_mvc_native_scan"
DD_YAW_NATIVE_KERNEL_REVISION = "ame_dd_yaw_native_v1_mvc_scan"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def differential_drive_parameters_for_profile(profile: PhysicsProfile):
    if profile.time_model != "dd_yaw_v1":
        raise ValueError(f"profile {profile.name!r} is not a DD/yaw profile")
    if profile.dd_effective_track_m is None or profile.dd_yaw_inertia_scale is None:
        raise ValueError("DD/yaw profile omits frozen track/inertia parameters")
    return red_comet_dd_yaw_v1_parameters(
        effective_track_m=profile.dd_effective_track_m,
        yaw_inertia_scale=profile.dd_yaw_inertia_scale,
        gear_efficiency=1.0 if profile.dd_gear_efficiency is None else profile.dd_gear_efficiency,
        wheel_inertia_kg_m2=(
            0.0 if profile.dd_wheel_inertia_kg_m2 is None else profile.dd_wheel_inertia_kg_m2
        ),
        profile=profile,
    )


def physics_model_identity(profile: PhysicsProfile, *, include_source_hashes: bool = True) -> dict[str, Any]:
    base: dict[str, Any] = {
        "profile_name": profile.name,
        "time_model": profile.time_model,
        "cell_pitch_m": profile.cell_pitch_m,
        "mu_g": profile.mu_g,
        "a_brake": profile.a_brake,
        "a_max": profile.a_max,
        "v_max": profile.v_max,
        "body_length_m": profile.body_length_m,
        "body_width_m": profile.body_width_m,
        "mass_kg": profile.mass_kg,
    }
    if profile.time_model == "dd_yaw_v1":
        p = differential_drive_parameters_for_profile(profile)
        base.update(
            model_revision=DD_YAW_MODEL_REVISION,
            native_kernel_revision=DD_YAW_NATIVE_KERNEL_REVISION,
            dd_effective_track_m=p.effective_track_m,
            dd_yaw_inertia_kg_m2=p.yaw_inertia_kg_m2,
            dd_yaw_inertia_scale=profile.dd_yaw_inertia_scale,
            dd_wheel_radius_m=p.wheel_radius_m,
            dd_gear_ratio=p.gear_ratio,
            dd_motor_no_load_rpm=p.motor_no_load_rpm,
            dd_motor_stall_torque_nm=p.motor_stall_torque_nm,
            dd_gear_efficiency=p.gear_efficiency,
            dd_wheel_inertia_kg_m2=p.wheel_inertia_kg_m2,
            dd_side_free_speed_mps=p.side_free_speed_mps,
            dd_side_stall_force_n=p.side_stall_force_n,
        )
        if include_source_hashes:
            root = Path(__file__).resolve().parents[1]
            rels = (
                "segment/physics_profiles.py",
                "segment/differential_drive.py",
                "segment/side_actuator.py",
                "optimization/dd_yaw_speed_profile.py",
                "optimization/dd_yaw_anchors.py",
                "optimization/dd_yaw_gradients.py",
                "native/dd_yaw/ame_dd_yaw.h",
                "native/dd_yaw/dd_yaw.c",
                "cdd_yaw/_binding.py",
            )
            base["source_sha256"] = {
                rel: _sha256_file(root / rel) for rel in rels
            }
    return base


def physics_model_signature(profile: PhysicsProfile) -> str:
    blob = json.dumps(
        physics_model_identity(profile, include_source_hashes=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "DD_YAW_MODEL_REVISION",
    "DD_YAW_NATIVE_KERNEL_REVISION",
    "differential_drive_parameters_for_profile",
    "physics_model_identity",
    "physics_model_signature",
]
