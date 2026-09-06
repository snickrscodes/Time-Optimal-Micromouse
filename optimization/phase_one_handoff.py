"""Deterministic Phase-I-to-sparse handoff diagnostics.

The functions in this module deliberately separate two concepts:

``canonicalize_phase_one_handoff``
    A lossless normalization used by production orchestration.  It removes
    signed zero, sorts and exactly deduplicates cut rows, validates the fixed
    log-length map, and computes stable hashes.  It never rounds a certified
    path or cut location.

``equivalence_digest``
    A diagnostic-only coarse fingerprint for grouping states that differ by a
    documented numerical tolerance.  It is never fed back into a solver and
    therefore cannot weaken feasibility or derivative consistency.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .constraint_generation import (
    ConstraintFamily,
    ConstraintPool,
    KnotGeometryCache,
)
from .geometry_gradients import GeometryState
from .knot_parameterization import LogLengthKnotMap

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HandoffCanonicalizationSettings:
    """Settings for lossless normalization and diagnostic grouping."""

    equivalence_absolute_tolerance: float = 1.0e-8
    equivalence_relative_tolerance: float = 5.0e-12

    def __post_init__(self) -> None:
        for name, value in (
            ("equivalence_absolute_tolerance", self.equivalence_absolute_tolerance),
            ("equivalence_relative_tolerance", self.equivalence_relative_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")




@dataclass(frozen=True, slots=True)
class EndpointBranchAlignment:
    """Deterministic periodic-to-local endpoint branch handoff."""

    target: tuple[float | None, float | None, float | None, float | None] | None
    aligned_target: tuple[float | None, float | None, float | None, float | None] | None
    endpoint_heading: float | None
    heading_turns: int
    periodic_heading_residual: float
    local_heading_residual: float


def align_endpoint_target_to_path(
    physical_x: Sequence[float],
    initial_state: GeometryState | Sequence[float],
    endpoint_target: tuple[float | None, float | None, float | None, float | None] | None,
    *,
    initial_s: float = 0.0,
) -> EndpointBranchAlignment:
    """Align a periodic Phase-I heading target to the path's local branch.

    Phase I and planner certification identify headings modulo ``2*pi``.  The
    later sparse finite NLP intentionally uses an unwrapped local residual.
    This function changes only the *target representative*, by an integer
    multiple of ``2*pi``, so that the certified Phase-I path is also a valid
    local sparse starting point.  Geometry and all non-heading targets remain
    unchanged.
    """
    if endpoint_target is None:
        return EndpointBranchAlignment(None, None, None, 0, 0.0, 0.0)
    if len(endpoint_target) != 4:
        raise ValueError("endpoint target must contain four components")
    target = tuple(
        None if value is None else (0.0 if float(value) == 0.0 else float(value))
        for value in endpoint_target
    )
    heading_target = target[2]
    if heading_target is None:
        return EndpointBranchAlignment(target, target, None, 0, 0.0, 0.0)
    cache = KnotGeometryCache(initial_state, float(initial_s))
    cache.update(physical_x)
    endpoint_heading = float(cache.endpoint_pose_jacobian()[0].theta)
    turns = int(math.floor((endpoint_heading - heading_target) / (2.0 * math.pi) + 0.5))
    aligned_heading = heading_target + turns * (2.0 * math.pi)
    aligned = list(target)
    aligned[2] = 0.0 if aligned_heading == 0.0 else aligned_heading
    periodic = canonical_angle(endpoint_heading - heading_target)
    local = endpoint_heading - aligned_heading
    # The nearest representative must be local.  Keep a small tolerance for a
    # half-turn floating-point tie while remaining deterministic.
    if abs(local) > math.pi + 8.0 * np.finfo(float).eps * max(1.0, abs(endpoint_heading)):
        raise RuntimeError("failed to align endpoint heading to the local branch")
    return EndpointBranchAlignment(
        target,
        tuple(aligned),  # type: ignore[arg-type]
        endpoint_heading,
        turns,
        periodic,
        local,
    )


@dataclass(frozen=True, slots=True)
class CanonicalHandoffState:
    """Losslessly normalized handoff state and reproducibility diagnostics."""

    physical_x: Array
    optimizer_x: Array
    pool: ConstraintPool
    endpoint_target: tuple[float | None, float | None, float | None, float | None] | None
    periodic_endpoint_target: tuple[float | None, float | None, float | None, float | None] | None
    state_digest: str
    equivalence_digest: str
    physical_digest: str
    optimizer_digest: str
    cut_pool_digest: str
    cut_structure_digest: str
    roundtrip_inf: float
    signed_zero_repairs: int
    duplicate_cut_repairs: int


def canonical_angle(value: float) -> float:
    """Return the unique periodic representative in ``[-pi, pi)``."""
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    wrapped = math.fmod(value + math.pi, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    wrapped -= math.pi
    # The half-open interval maps +pi to -pi.  Normalize signed zero too.
    return 0.0 if wrapped == 0.0 else wrapped


def _normalize_signed_zero(values: Sequence[float]) -> tuple[Array, int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("handoff vectors must be finite and one-dimensional")
    out = array.copy()
    repairs = int(np.count_nonzero((out == 0.0) & np.signbit(out)))
    out[out == 0.0] = 0.0
    return out, repairs


def _float_bytes(value: float) -> bytes:
    normalized = 0.0 if value == 0.0 else float(value)
    return struct.pack("<d", normalized)


def _array_digest(array: Array) -> str:
    normalized = np.asarray(array, dtype="<f8").copy()
    normalized[normalized == 0.0] = 0.0
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _canonical_pool(pool: ConstraintPool) -> tuple[ConstraintPool, int]:
    canonical: dict[ConstraintFamily, list[float]] = {}
    duplicates = 0
    for family in sorted(pool.taus, key=lambda f: (f.segment, f.wall, f.corner)):
        values: list[float] = []
        for raw in sorted(float(v) for v in pool.taus[family]):
            if not math.isfinite(raw) or raw < 0.0 or raw > 1.0:
                raise ValueError("constraint tau must be finite and lie in [0, 1]")
            tau = 0.0 if raw == 0.0 else raw
            if values and tau == values[-1]:
                duplicates += 1
                continue
            values.append(tau)
        canonical[family] = values
    return ConstraintPool(pool.corridor, canonical), duplicates


def _pool_digests(pool: ConstraintPool) -> tuple[str, str]:
    exact = hashlib.sha256()
    structure = hashlib.sha256()
    exact.update(b"sparse-sqp-cut-pool-v1\0")
    structure.update(b"sparse-sqp-cut-structure-v1\0")
    for entry in pool.entries():
        key = struct.pack("<iii", entry.segment, entry.wall, entry.corner)
        exact.update(key)
        exact.update(_float_bytes(entry.tau))
        structure.update(key)
    return exact.hexdigest(), structure.hexdigest()


def _quantized_token(value: float, settings: HandoffCanonicalizationSettings) -> int:
    scale = max(
        settings.equivalence_absolute_tolerance,
        settings.equivalence_relative_tolerance * max(1.0, abs(value)),
    )
    return int(round(float(value) / scale))


def _equivalence_digest(
    physical_x: Array,
    optimizer_x: Array,
    pool: ConstraintPool,
    endpoint_target: tuple[float | None, float | None, float | None, float | None] | None,
    settings: HandoffCanonicalizationSettings,
) -> str:
    # Diagnostic only.  The header makes the tolerance policy explicit in the
    # digest namespace and prevents accidental comparison with exact hashes.
    payload: dict[str, object] = {
        "schema": "sparse-sqp-handoff-equivalence-v1",
        "absolute_tolerance": settings.equivalence_absolute_tolerance,
        "relative_tolerance": settings.equivalence_relative_tolerance,
        "physical": [_quantized_token(v, settings) for v in physical_x],
        "optimizer": [_quantized_token(v, settings) for v in optimizer_x],
        "cuts": [
            [
                e.segment,
                e.wall,
                e.corner,
                _quantized_token(e.tau, settings),
            ]
            for e in pool.entries()
        ],
        "periodic_endpoint_target": None
        if endpoint_target is None
        else [
            None
            if value is None
            else _quantized_token(
                canonical_angle(value) if index == 2 else float(value), settings
            )
            for index, value in enumerate(endpoint_target)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonicalize_phase_one_handoff(
    physical_x: Sequence[float],
    pool: ConstraintPool,
    coordinate_map: LogLengthKnotMap | None,
    *,
    endpoint_target: tuple[float | None, float | None, float | None, float | None] | None = None,
    settings: HandoffCanonicalizationSettings = HandoffCanonicalizationSettings(),
) -> CanonicalHandoffState:
    """Validate and losslessly normalize a strict Phase-I handoff.

    No tolerance-based snapping is performed.  The returned physical point is
    mathematically identical to the input apart from signed-zero normalization.
    The exact pool locations are retained apart from exact duplicate removal.
    """
    physical, repairs = _normalize_signed_zero(physical_x)
    if coordinate_map is None:
        optimizer = physical.copy()
        optimizer_repairs = 0
        roundtrip = optimizer.copy()
        coordinate_mode = "stations"
        reference_lengths_digest = None
        coordinate_initial_s = None
    else:
        if physical.shape != (coordinate_map.width,):
            raise ValueError("coordinate map width does not match handoff vector")
        optimizer, optimizer_repairs = _normalize_signed_zero(
            coordinate_map.to_optimizer(physical)
        )
        roundtrip = coordinate_map.to_physical(optimizer)
        coordinate_mode = "log_lengths"
        reference_lengths_digest = _array_digest(coordinate_map.reference_lengths)
        coordinate_initial_s = _float_bytes(coordinate_map.initial_s).hex()
    repairs += optimizer_repairs
    roundtrip_inf = float(np.linalg.norm(roundtrip - physical, ord=np.inf))
    canonical_pool, duplicate_repairs = _canonical_pool(pool)

    periodic_target = None
    target_normalized = None
    if endpoint_target is not None:
        if len(endpoint_target) != 4:
            raise ValueError("endpoint target must contain four components")
        target_values: list[float | None] = []
        periodic_values: list[float | None] = []
        for index, value in enumerate(endpoint_target):
            if value is None:
                target_values.append(None)
                periodic_values.append(None)
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("endpoint target values must be finite or None")
            number = 0.0 if number == 0.0 else number
            target_values.append(number)
            periodic_values.append(canonical_angle(number) if index == 2 else number)
        target_normalized = tuple(target_values)  # type: ignore[assignment]
        periodic_target = tuple(periodic_values)  # type: ignore[assignment]

    physical_digest = _array_digest(physical)
    optimizer_digest = _array_digest(optimizer)
    cut_digest, structure_digest = _pool_digests(canonical_pool)
    exact_payload = {
        "schema": "sparse-sqp-phase-one-handoff-v1",
        "physical": physical_digest,
        "optimizer": optimizer_digest,
        "cut_pool": cut_digest,
        "cut_structure": structure_digest,
        "coordinate_mode": coordinate_mode,
        "reference_lengths": reference_lengths_digest,
        "initial_s": coordinate_initial_s,
        "endpoint_target": target_normalized,
        "periodic_endpoint_target": periodic_target,
    }
    state_digest = hashlib.sha256(
        json.dumps(exact_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    equivalence = _equivalence_digest(
        physical, optimizer, canonical_pool, target_normalized, settings
    )
    return CanonicalHandoffState(
        physical_x=physical,
        optimizer_x=optimizer,
        pool=canonical_pool,
        endpoint_target=target_normalized,
        periodic_endpoint_target=periodic_target,
        state_digest=state_digest,
        equivalence_digest=equivalence,
        physical_digest=physical_digest,
        optimizer_digest=optimizer_digest,
        cut_pool_digest=cut_digest,
        cut_structure_digest=structure_digest,
        roundtrip_inf=roundtrip_inf,
        signed_zero_repairs=repairs,
        duplicate_cut_repairs=duplicate_repairs,
    )
