"""Reduced-order aggregate left/right drivetrain algebra for DD/yaw profiles.

This module is intentionally independent of the production speed-profile state
machine.  It provides the Phase-1 reference physics used by the future
``red_comet_2017_dd_yaw_v1`` backend.

Planner units
-------------
``s`` and curvature use grid cells, while ``w = v_grid**2``.  Physical speed
is ``cell_pitch_m * sqrt(w)``.  Centerline acceleration ``a`` is therefore in
grid-cells/s^2.

The four-driven-wheel Red Comet chassis is modeled as two aggregate longitudinal
left/right drivetrains.  Four-wheel scrub/slip is deliberately not modeled in
v1; this is an actuator/yaw authority model, not an ideal no-slip claim for all
four contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

from .physics_profiles import PhysicsProfile, RED_COMET_2017_NOMINAL


_G_CM_TO_NM = 1.0e-5 * 9.80665
_TWO_PI = 2.0 * math.pi


class DriveSide(IntEnum):
    """Aggregate drivetrain side sign used by the DD/yaw equations."""

    LEFT = -1
    RIGHT = 1


class DifferentialDriveDomainError(ValueError):
    """Raised when a state lies outside the qualified v1 DD/yaw chart."""


@dataclass(frozen=True, slots=True)
class DifferentialDriveParameters:
    """Physical and derived constants for the aggregate DD/yaw model.

    ``effective_track_m`` and ``yaw_inertia_kg_m2`` are explicit model inputs,
    because they are not independently measured for Red Comet.  The nominal
    constructor below uses the frozen design's 39 mm inferred track and
    uniform-rectangle inertia proxy.
    """

    cell_pitch_m: float
    mass_kg: float
    effective_track_m: float
    yaw_inertia_kg_m2: float
    wheel_radius_m: float
    gear_ratio: float
    motor_no_load_rpm: float
    motor_stall_torque_nm: float
    gear_efficiency: float = 1.0
    wheel_inertia_kg_m2: float = 0.0
    h_floor: float = 1.0e-10
    c_floor: float = 1.0e-10
    speed_margin: float = 1.0e-10

    def __post_init__(self) -> None:
        positive = {
            "cell_pitch_m": self.cell_pitch_m,
            "mass_kg": self.mass_kg,
            "effective_track_m": self.effective_track_m,
            "wheel_radius_m": self.wheel_radius_m,
            "gear_ratio": self.gear_ratio,
            "motor_no_load_rpm": self.motor_no_load_rpm,
            "motor_stall_torque_nm": self.motor_stall_torque_nm,
            "gear_efficiency": self.gear_efficiency,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.yaw_inertia_kg_m2) or self.yaw_inertia_kg_m2 < 0.0:
            raise ValueError("yaw_inertia_kg_m2 must be finite and nonnegative")
        if not math.isfinite(self.wheel_inertia_kg_m2) or self.wheel_inertia_kg_m2 < 0.0:
            raise ValueError("wheel_inertia_kg_m2 must be finite and nonnegative")
        for name in ("h_floor", "c_floor", "speed_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def beta(self) -> float:
        """Half-track in planner-cell units: ``b / (2 p)``."""

        return self.effective_track_m / (2.0 * self.cell_pitch_m)

    @property
    def eta(self) -> float:
        """Normalized yaw-inertia length ``2 Iz / (m p b)`` (grid cells)."""

        return (
            2.0
            * self.yaw_inertia_kg_m2
            / (self.mass_kg * self.cell_pitch_m * self.effective_track_m)
        )

    @property
    def motor_no_load_rad_s(self) -> float:
        return _TWO_PI * self.motor_no_load_rpm / 60.0

    @property
    def side_free_speed_mps(self) -> float:
        return self.wheel_radius_m * self.motor_no_load_rad_s / self.gear_ratio

    @property
    def side_free_speed_grid(self) -> float:
        return self.side_free_speed_mps / self.cell_pitch_m

    @property
    def side_stall_force_n(self) -> float:
        return (
            self.gear_efficiency
            * self.gear_ratio
            * self.motor_stall_torque_nm
            / self.wheel_radius_m
        )

    @property
    def q0(self) -> float:
        """Normalized per-side stall force ``2 F_stall / (m p)``."""

        return 2.0 * self.side_stall_force_n / (self.mass_kg * self.cell_pitch_m)

    @property
    def reflected_wheel_mass_kg(self) -> float:
        """Reserved wheel-inertia hook ``J/r^2``; zero in frozen v1 nominal."""

        return self.wheel_inertia_kg_m2 / (self.wheel_radius_m * self.wheel_radius_m)


@dataclass(frozen=True, slots=True)
class SideState:
    side: DriveSide
    h: float
    c: float
    side_speed_grid: float
    side_speed_mps: float
    force_capacity_q: float


@dataclass(frozen=True, slots=True)
class SideBounds:
    side: DriveSide
    lower: float
    upper: float
    state: SideState


@dataclass(frozen=True, slots=True)
class AccelerationInterval:
    lower: float
    upper: float
    motor_upper: float
    grip_magnitude: float
    left: SideBounds
    right: SideBounds

    @property
    def margin(self) -> float:
        return self.upper - self.lower

    @property
    def feasible(self) -> bool:
        return self.margin >= 0.0


def uniform_rectangle_yaw_inertia(mass_kg: float, length_m: float, width_m: float) -> float:
    """Uniform rectangular-lamina yaw-inertia proxy about the vertical axis."""

    values = (mass_kg, length_m, width_m)
    if not all(math.isfinite(v) and v > 0.0 for v in values):
        raise ValueError("mass and rectangle dimensions must be finite and positive")
    return mass_kg * (length_m * length_m + width_m * width_m) / 12.0


def red_comet_dd_yaw_v1_parameters(
    *,
    effective_track_m: float = 0.039,
    yaw_inertia_scale: float = 1.0,
    gear_efficiency: float = 1.0,
    wheel_inertia_kg_m2: float = 0.0,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> DifferentialDriveParameters:
    """Return the frozen Phase-1 Red Comet DD/yaw parameter set.

    Historical/hardware inputs frozen by the design:
    13.5 mm wheel diameter, 4:1 reduction, CL-0614-10250-7 at 46 krpm
    no-load and 8 g*cm stall torque.  ``effective_track_m`` and the yaw-inertia
    scale remain explicit sensitivity parameters rather than hidden calibration.
    """

    if profile.mass_kg is None or profile.body_length_m is None or profile.body_width_m is None:
        raise ValueError("DD/yaw profile requires mass and body dimensions")
    if not math.isfinite(yaw_inertia_scale) or yaw_inertia_scale < 0.0:
        raise ValueError("yaw_inertia_scale must be finite and nonnegative")

    iz_box = uniform_rectangle_yaw_inertia(
        profile.mass_kg,
        profile.body_length_m,
        profile.body_width_m,
    )

    return DifferentialDriveParameters(
        cell_pitch_m=profile.cell_pitch_m,
        mass_kg=profile.mass_kg,
        effective_track_m=float(effective_track_m),
        yaw_inertia_kg_m2=yaw_inertia_scale * iz_box,
        wheel_radius_m=0.0135 / 2.0,
        gear_ratio=36.0 / 9.0,
        motor_no_load_rpm=46_000.0,
        motor_stall_torque_nm=8.0 * _G_CM_TO_NM,
        gear_efficiency=float(gear_efficiency),
        wheel_inertia_kg_m2=float(wheel_inertia_kg_m2),
    )


def _coerce_side(side: DriveSide | int) -> DriveSide:
    try:
        return DriveSide(int(side))
    except (TypeError, ValueError) as exc:
        raise ValueError("side must be DriveSide.LEFT/-1 or DriveSide.RIGHT/+1") from exc


def _require_finite_state(w: float, kappa: float, sigma: float = 0.0) -> tuple[float, float, float]:
    w = float(w)
    kappa = float(kappa)
    sigma = float(sigma)
    if not all(math.isfinite(v) for v in (w, kappa, sigma)):
        raise DifferentialDriveDomainError("DD/yaw state must be finite")
    if w < 0.0:
        raise DifferentialDriveDomainError("DD/yaw requires w >= 0")
    return w, kappa, sigma


def _require_positive_w(w: float) -> None:
    if w <= 0.0:
        raise DifferentialDriveDomainError(
            "DD/yaw derivative/SIDE segment chart requires w > 0"
        )


def side_state(
    w: float,
    kappa: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> SideState:
    """Evaluate aggregate side speed, coefficient and motor-force capacity."""

    w, kappa, _ = _require_finite_state(w, kappa)
    eps = int(_coerce_side(side))
    root_w = math.sqrt(w)

    h = 1.0 + eps * params.beta * kappa
    c = 1.0 + eps * params.eta * kappa
    if h <= params.h_floor:
        raise DifferentialDriveDomainError(
            f"side kinematic factor h={h!r} is outside qualified positive-speed chart"
        )
    if c <= params.c_floor:
        raise DifferentialDriveDomainError(
            f"side force coefficient c={c!r} is outside qualified positive chart"
        )

    side_speed_grid = root_w * h
    v_free = params.side_free_speed_grid
    if side_speed_grid >= v_free - params.speed_margin:
        raise DifferentialDriveDomainError(
            "side speed reaches/exceeds rated free-speed envelope: "
            f"v_side_grid={side_speed_grid!r}, V_free_grid={v_free!r}"
        )

    capacity = params.q0 * (1.0 - side_speed_grid / v_free)
    if capacity <= 0.0 or not math.isfinite(capacity):
        raise DifferentialDriveDomainError("side motor-force capacity is nonpositive")

    return SideState(
        side=DriveSide(eps),
        h=h,
        c=c,
        side_speed_grid=side_speed_grid,
        side_speed_mps=params.cell_pitch_m * side_speed_grid,
        force_capacity_q=capacity,
    )


def yaw_rate(w: float, kappa: float) -> float:
    w, kappa, _ = _require_finite_state(w, kappa)
    return math.sqrt(w) * kappa


def yaw_acceleration(w: float, kappa: float, sigma: float, acceleration: float) -> float:
    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    acceleration = float(acceleration)
    if not math.isfinite(acceleration):
        raise DifferentialDriveDomainError("centerline acceleration must be finite")
    return acceleration * kappa + w * sigma


def side_acceleration_grid(
    acceleration: float,
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> float:
    """Aggregate side tangential acceleration in planner grid units."""

    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    acceleration = float(acceleration)
    if not math.isfinite(acceleration):
        raise DifferentialDriveDomainError("centerline acceleration must be finite")
    eps = int(_coerce_side(side))
    h = 1.0 + eps * params.beta * kappa
    if h <= params.h_floor:
        raise DifferentialDriveDomainError("side kinematic factor leaves qualified chart")
    return acceleration * h + eps * params.beta * w * sigma


def side_acceleration_mps2(
    acceleration: float,
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> float:
    return params.cell_pitch_m * side_acceleration_grid(
        acceleration, w, kappa, sigma, side, params
    )


def side_force_demand_q(
    acceleration: float,
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> float:
    """Normalized aggregate side-force demand ``2 F_side / (m p)``."""

    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    acceleration = float(acceleration)
    if not math.isfinite(acceleration):
        raise DifferentialDriveDomainError("centerline acceleration must be finite")
    eps = int(_coerce_side(side))
    c = 1.0 + eps * params.eta * kappa
    if c <= params.c_floor:
        raise DifferentialDriveDomainError("side force coefficient leaves qualified chart")
    return c * acceleration + eps * params.eta * w * sigma


def side_force_demand_n(
    acceleration: float,
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> float:
    q = side_force_demand_q(acceleration, w, kappa, sigma, side, params)
    return 0.5 * params.mass_kg * params.cell_pitch_m * q


def side_acceleration_bounds(
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> SideBounds:
    """Return the centerline-acceleration interval permitted by one side."""

    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    side = _coerce_side(side)
    eps = int(side)
    state = side_state(w, kappa, side, params)
    d = eps * params.eta * w * sigma
    q = state.force_capacity_q
    lower = (-q - d) / state.c
    upper = (q - d) / state.c
    return SideBounds(side=side, lower=lower, upper=upper, state=state)


def side_candidate(
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> float:
    """Upper SIDE extremal candidate used by a traversal pass."""

    return side_acceleration_bounds(w, kappa, sigma, side, params).upper


def side_candidate_partials(
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> tuple[float, float, float, float]:
    """Return ``(a, da/dw, da/dkappa, da/dsigma)`` for the SIDE upper bound."""

    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    _require_positive_w(w)
    side = _coerce_side(side)
    eps = int(side)
    state = side_state(w, kappa, side, params)
    root_w = math.sqrt(w)
    q_w = -params.q0 * state.h / (2.0 * params.side_free_speed_grid * root_w)
    q_kappa = (
        -params.q0
        * root_w
        * eps
        * params.beta
        / params.side_free_speed_grid
    )
    a = (state.force_capacity_q - eps * params.eta * w * sigma) / state.c
    a_w = (q_w - eps * params.eta * sigma) / state.c
    a_kappa = (q_kappa - eps * params.eta * a) / state.c
    a_sigma = -eps * params.eta * w / state.c
    return a, a_w, a_kappa, a_sigma


def side_lower_partials(
    w: float,
    kappa: float,
    sigma: float,
    side: DriveSide | int,
    params: DifferentialDriveParameters,
) -> tuple[float, float, float, float]:
    """Return ``(L, dL/dw, dL/dkappa, dL/dsigma)`` for one side's lower bound."""

    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    _require_positive_w(w)
    side = _coerce_side(side)
    eps = int(side)
    state = side_state(w, kappa, side, params)
    root_w = math.sqrt(w)
    q_w = -params.q0 * state.h / (2.0 * params.side_free_speed_grid * root_w)
    q_kappa = (
        -params.q0
        * root_w
        * eps
        * params.beta
        / params.side_free_speed_grid
    )
    lower = (-state.force_capacity_q - eps * params.eta * w * sigma) / state.c
    lower_w = (-q_w - eps * params.eta * sigma) / state.c
    lower_kappa = (-q_kappa - eps * params.eta * lower) / state.c
    lower_sigma = -eps * params.eta * w / state.c
    return lower, lower_w, lower_kappa, lower_sigma


def acceleration_interval(
    w: float,
    kappa: float,
    sigma: float,
    params: DifferentialDriveParameters,
    profile: PhysicsProfile = RED_COMET_2017_NOMINAL,
) -> AccelerationInterval:
    """Return the complete signed local acceleration interval for DD/yaw v1."""

    w, kappa, sigma = _require_finite_state(w, kappa, sigma)
    root_w = math.sqrt(w)
    lateral = w * kappa
    radicand = profile.mu_g * profile.mu_g - lateral * lateral
    if radicand < 0.0:
        grip = -math.inf
    else:
        grip = math.sqrt(max(0.0, radicand))

    motor_upper = profile.a_max - profile.b_emf * root_w
    left = side_acceleration_bounds(w, kappa, sigma, DriveSide.LEFT, params)
    right = side_acceleration_bounds(w, kappa, sigma, DriveSide.RIGHT, params)

    if grip == -math.inf:
        lower = math.inf
        upper = -math.inf
    else:
        lower = max(-profile.a_brake, -grip, left.lower, right.lower)
        upper = min(motor_upper, grip, left.upper, right.upper)

    return AccelerationInterval(
        lower=lower,
        upper=upper,
        motor_upper=motor_upper,
        grip_magnitude=grip,
        left=left,
        right=right,
    )


__all__ = [
    "AccelerationInterval",
    "DifferentialDriveDomainError",
    "DifferentialDriveParameters",
    "DriveSide",
    "SideBounds",
    "SideState",
    "acceleration_interval",
    "red_comet_dd_yaw_v1_parameters",
    "side_acceleration_bounds",
    "side_acceleration_grid",
    "side_acceleration_mps2",
    "side_candidate",
    "side_candidate_partials",
    "side_force_demand_n",
    "side_force_demand_q",
    "side_lower_partials",
    "side_state",
    "uniform_rectangle_yaw_inertia",
    "yaw_acceleration",
    "yaw_rate",
]
