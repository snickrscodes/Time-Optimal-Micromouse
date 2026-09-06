"""Shared physical constants selected at fresh-process import time.

The historical qualification corpus remains the default ``legacy_grid_v1``
profile. Red Comet and other physical case studies may opt into a named
profile through ``AME_PHYSICS_PROFILE``. A non-legacy profile may use the
native reverse backend only from an isolated native build stamped with the
same profile name; this prevents silently mixing calibrated Python constants
with legacy-qualified compiled kernels.
"""

from __future__ import annotations

import os
from pathlib import Path

from .physics_profiles import get_physics_profile

PHYSICS_PROFILE_NAME = (
    os.environ.get("AME_PHYSICS_PROFILE", "legacy_grid_v1").strip().lower()
    or "legacy_grid_v1"
)
PHYSICS_PROFILE = get_physics_profile(PHYSICS_PROFILE_NAME)

MU_G = PHYSICS_PROFILE.mu_g
A_BRAKE = PHYSICS_PROFILE.a_brake
A_MAX = PHYSICS_PROFILE.a_max
V_MAX = PHYSICS_PROFILE.v_max
B_EMF = PHYSICS_PROFILE.b_emf

def _validate_native_profile_binding() -> None:
    backend = os.environ.get("AME_REVERSE_BACKEND", "native").strip().lower() or "native"
    if backend != "native" or PHYSICS_PROFILE.native_qualified:
        return
    stamp = Path(__file__).resolve().parents[1] / "native" / ".physics_profile"
    try:
        compiled_profile = stamp.read_text(encoding="utf-8").strip().lower()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"physics profile {PHYSICS_PROFILE_NAME!r} is not qualified against the "
            "checked-in native kernels; use AME_REVERSE_BACKEND=python or an "
            "isolated native build stamped with the same profile"
        ) from exc
    if compiled_profile != PHYSICS_PROFILE_NAME:
        raise RuntimeError(
            f"native physics-profile mismatch: Python selected {PHYSICS_PROFILE_NAME!r} "
            f"but native build is stamped {compiled_profile!r}"
        )

_validate_native_profile_binding()

__all__ = [
    "PHYSICS_PROFILE_NAME", "PHYSICS_PROFILE",
    "MU_G", "A_BRAKE", "A_MAX", "V_MAX", "B_EMF",
]
