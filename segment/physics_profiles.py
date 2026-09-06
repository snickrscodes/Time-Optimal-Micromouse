"""Named physical parameter sets for maze speed-profile experiments.

The production numerical qualification corpus was built with the historical
``legacy_grid_v1`` constants.  Physical case studies can select a different
profile in a *fresh Python process* through ``AME_PHYSICS_PROFILE``.  The
selection deliberately happens before modules derive switching geometry.

Native Segment/Crossing/Reverse libraries remain compile-time qualified against
``legacy_grid_v1``.  Non-legacy profiles therefore use the Python reverse
backend unless a separately rebuilt/qualified native stack is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class PhysicsProfile:
    """Base physical constants expressed in planner grid units."""

    name: str
    cell_pitch_m: float
    mu_g: float
    a_brake: float
    a_max: float
    v_max: float
    native_qualified: bool
    description: str
    body_length_m: float | None = None
    body_width_m: float | None = None
    mass_kg: float | None = None
    # Time-model dispatcher.  Legacy profiles continue through the historically
    # qualified MOTOR/GRIP/BRAKE reverse solver.  DD/yaw profiles select the
    # separate one-state aggregate left/right actuator backend.
    time_model: str = "legacy_v1"
    dd_effective_track_m: float | None = None
    dd_yaw_inertia_scale: float | None = None
    dd_gear_efficiency: float | None = None
    dd_wheel_inertia_kg_m2: float | None = None

    @property
    def body_length_grid(self) -> float | None:
        return None if self.body_length_m is None else self.body_length_m / self.cell_pitch_m

    @property
    def body_width_grid(self) -> float | None:
        return None if self.body_width_m is None else self.body_width_m / self.cell_pitch_m

    @property
    def b_emf(self) -> float:
        return self.a_max / self.v_max

    @property
    def mu_g_mps2(self) -> float:
        return self.mu_g * self.cell_pitch_m

    @property
    def a_brake_mps2(self) -> float:
        return self.a_brake * self.cell_pitch_m

    @property
    def a_max_mps2(self) -> float:
        return self.a_max * self.cell_pitch_m

    @property
    def v_max_mps(self) -> float:
        return self.v_max * self.cell_pitch_m


LEGACY_GRID_V1 = PhysicsProfile(
    name="legacy_grid_v1",
    cell_pitch_m=0.18,
    mu_g=1.2 * 9.81,
    a_brake=0.9 * (1.2 * 9.81),
    a_max=15.0,
    v_max=4.0,
    native_qualified=True,
    description="Original project benchmark constants in grid-cell units.",
    body_length_m=0.100,
    body_width_m=0.080,
)

# Red Comet calibration notes
# ---------------------------
# Published vehicle data gives 5.0 m/s straight speed and 1.6--2.1 m/s turn
# speed.  The current one-cell 90-degree Euler primitive has peak curvature
# 3.7401916933 / cell.  Taking the midpoint (1.85 m/s) as the nominal reference
# gives a model friction-circle acceleration v^2*kappa ~= 71.12 m/s^2.  This is
# a model inference, not a directly published grip measurement.
#
# The 2018 program does not list longitudinal acceleration.  A 2019 NTF
# technical-data record for the same original RedComet (same 76x45x30 mm body,
# 30.2 g mass, MCU and drive motor) lists 15.5 m/s^2 acceleration.  We use that
# as the nearest located same-vehicle longitudinal proxy and still keep it
# explicit as cross-year evidence rather than claiming it was the exact 2017
# race-day tuning.
_RED_COMET_CELL_PITCH_M = 0.18
_RED_COMET_REFERENCE_PEAK_CURVATURE_PER_CELL = 3.7401916933
_RED_COMET_REFERENCE_TURN_SPEED_MPS = 1.85
_RED_COMET_GRIP_MPS2 = (
    _RED_COMET_REFERENCE_TURN_SPEED_MPS
    * _RED_COMET_REFERENCE_TURN_SPEED_MPS
    * _RED_COMET_REFERENCE_PEAK_CURVATURE_PER_CELL
    / _RED_COMET_CELL_PITCH_M
)
_RED_COMET_LONGITUDINAL_MPS2 = 15.5

RED_COMET_2017_NOMINAL = PhysicsProfile(
    name="red_comet_2017_nominal",
    cell_pitch_m=_RED_COMET_CELL_PITCH_M,
    mu_g=_RED_COMET_GRIP_MPS2 / _RED_COMET_CELL_PITCH_M,
    a_brake=_RED_COMET_LONGITUDINAL_MPS2 / _RED_COMET_CELL_PITCH_M,
    a_max=_RED_COMET_LONGITUDINAL_MPS2 / _RED_COMET_CELL_PITCH_M,
    v_max=5.0 / _RED_COMET_CELL_PITCH_M,
    native_qualified=False,
    description=(
        "Red Comet case-study nominal: published 5 m/s straight speed, "
        "model-inferred grip from the midpoint of the published 1.6--2.1 m/s "
        "turn-speed range, and a 15.5 m/s^2 same-vehicle 2019 longitudinal proxy."
    ),
    body_length_m=0.076,
    body_width_m=0.045,
    mass_kg=0.0302,
)


# Calibration-only envelope point used after the Red Comet native timing fix.
# This is intentionally green-favorable while remaining grounded in located
# same-vehicle evidence: the published lower end (1.6 m/s) of the 2017 turn
# range, the 15.5 m/s^2 2019 longitudinal proxy, and the 5.2 m/s 2019 top speed.
# It is a sensitivity profile, not a claim about exact 2017 race-day tuning.
_RED_COMET_GREEN_FAVORABLE_TURN_SPEED_MPS = 1.60
_RED_COMET_GREEN_FAVORABLE_GRIP_MPS2 = (
    _RED_COMET_GREEN_FAVORABLE_TURN_SPEED_MPS
    * _RED_COMET_GREEN_FAVORABLE_TURN_SPEED_MPS
    * _RED_COMET_REFERENCE_PEAK_CURVATURE_PER_CELL
    / _RED_COMET_CELL_PITCH_M
)

RED_COMET_2017_GREEN_FAVORABLE_ENVELOPE = PhysicsProfile(
    name="red_comet_2017_green_favorable_envelope",
    cell_pitch_m=_RED_COMET_CELL_PITCH_M,
    mu_g=_RED_COMET_GREEN_FAVORABLE_GRIP_MPS2 / _RED_COMET_CELL_PITCH_M,
    a_brake=_RED_COMET_LONGITUDINAL_MPS2 / _RED_COMET_CELL_PITCH_M,
    a_max=_RED_COMET_LONGITUDINAL_MPS2 / _RED_COMET_CELL_PITCH_M,
    v_max=5.2 / _RED_COMET_CELL_PITCH_M,
    native_qualified=False,
    description=(
        "Calibration-only Red Comet evidence-envelope stress: published 1.6 m/s "
        "lower turn-speed endpoint, 15.5 m/s^2 same-vehicle 2019 longitudinal "
        "proxy, and 5.2 m/s same-vehicle 2019 top speed."
    ),
    body_length_m=0.076,
    body_width_m=0.045,
    mass_kg=0.0302,
)


# Production Red Comet v1 dynamics profile.  Scalar grip/longitudinal constants
# are identical to the nominal calibration above; only the time functional is
# extended with aggregate left/right drivetrain/yaw authority.
#
# The 39 mm effective track is the frozen design inference 45 mm overall width
# minus one 6 mm tire width.  The yaw inertia is the 1.0x uniform-rectangle
# proxy.  Wheel/drivetrain inertia remains zero in v1 because no independently
# defensible Red Comet value was located.
RED_COMET_2017_DD_YAW_V1 = PhysicsProfile(
    name="red_comet_2017_dd_yaw_v1",
    cell_pitch_m=_RED_COMET_CELL_PITCH_M,
    mu_g=_RED_COMET_GRIP_MPS2 / _RED_COMET_CELL_PITCH_M,
    a_brake=_RED_COMET_LONGITUDINAL_MPS2 / _RED_COMET_CELL_PITCH_M,
    a_max=_RED_COMET_LONGITUDINAL_MPS2 / _RED_COMET_CELL_PITCH_M,
    v_max=5.0 / _RED_COMET_CELL_PITCH_M,
    native_qualified=False,
    description=(
        "Red Comet production DD/yaw v1: nominal scalar grip/longitudinal "
        "profile plus aggregate left/right four-wheel drivetrain authority, "
        "motor torque-speed limits and chassis yaw inertia."
    ),
    body_length_m=0.076,
    body_width_m=0.045,
    mass_kg=0.0302,
    time_model="dd_yaw_v1",
    dd_effective_track_m=0.039,
    dd_yaw_inertia_scale=1.0,
    dd_gear_efficiency=1.0,
    dd_wheel_inertia_kg_m2=0.0,
)


_PROFILES = {
    LEGACY_GRID_V1.name: LEGACY_GRID_V1,
    RED_COMET_2017_NOMINAL.name: RED_COMET_2017_NOMINAL,
    RED_COMET_2017_GREEN_FAVORABLE_ENVELOPE.name: RED_COMET_2017_GREEN_FAVORABLE_ENVELOPE,
    RED_COMET_2017_DD_YAW_V1.name: RED_COMET_2017_DD_YAW_V1,
}


def get_physics_profile(name: str) -> PhysicsProfile:
    key = str(name).strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown AME physics profile {name!r}; choose one of: {choices}") from exc


def physics_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


__all__ = [
    "PhysicsProfile",
    "LEGACY_GRID_V1",
    "RED_COMET_2017_NOMINAL",
    "RED_COMET_2017_GREEN_FAVORABLE_ENVELOPE",
    "RED_COMET_2017_DD_YAW_V1",
    "get_physics_profile",
    "physics_profile_names",
]
