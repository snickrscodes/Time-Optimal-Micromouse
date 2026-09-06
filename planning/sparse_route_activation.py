"""Deterministic eligibility for a conditional sparse route-polishing rollout.

The policy is intentionally conservative.  It never establishes safety itself;
it only permits shadow or production sparse polishing after a strict Phase-I
certificate and a planner-owned fallback already exist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SparseRouteFeatures:
    n_cells: int
    n_variables: int
    finite_pool_size: int
    turn_count: int
    body_length: float
    body_height: float
    phase_one_seconds: float
    phase_one_strictly_certified: bool
    planner_fallback_certified: bool
    supervisor_healthy: bool
    canonicalization_succeeded: bool
    canonical_diagnostics_valid: bool
    estimated_sparse_seconds: float
    public_highspy_validated: bool
    deployment_watchdog_validated: bool
    sufficient_shadow_evidence: bool
    state_sensitivity_acceptable: bool
    strict_handoff_certified: bool | None = None
    handoff_seconds: float | None = None
    handoff_source: str = "phase_one"

    @property
    def structural_work(self) -> int:
        return self.n_variables * self.finite_pool_size


@dataclass(frozen=True, slots=True)
class SparseRouteEligibilitySettings:
    minimum_cells: int = 5
    maximum_cells: int = 35
    minimum_variables: int = 24
    maximum_variables: int = 140
    maximum_pool_size: int = 2_500
    maximum_turns: int = 14
    maximum_phase_one_seconds: float = 30.0
    maximum_handoff_seconds: float = 300.0
    maximum_estimated_sparse_seconds: float = 180.0
    production_body_length: float = 5.0 / 9.0
    production_body_height: float = 4.0 / 9.0
    body_tolerance: float = 1.0e-12
    early_certification_work_threshold: int = 150_000
    maximum_validated_work: int = 320_000
    require_public_highspy: bool = True


@dataclass(frozen=True, slots=True)
class SparseRouteEligibilityDecision:
    eligible: bool
    reasons: tuple[str, ...]
    structural_work: int
    use_cut_aware_phase_recenter: bool
    use_high_work_early_certification: bool
    mode: str


def count_route_turns(cells: Sequence[Sequence[int]]) -> int:
    if len(cells) < 2:
        return 0
    directions = [
        (int(b[0]) - int(a[0]), int(b[1]) - int(a[1]))
        for a, b in zip(cells[:-1], cells[1:], strict=True)
    ]
    return sum(first != second for first, second in zip(directions[:-1], directions[1:]))


def evaluate_sparse_route_eligibility(
    features: SparseRouteFeatures,
    settings: SparseRouteEligibilitySettings = SparseRouteEligibilitySettings(),
    *,
    shadow_mode: bool = False,
) -> SparseRouteEligibilityDecision:
    reasons: list[str] = []
    handoff_strict = (
        features.phase_one_strictly_certified
        if features.strict_handoff_certified is None
        else features.strict_handoff_certified
    )
    if not handoff_strict:
        reasons.append(
            "phase_one_not_strictly_certified"
            if features.strict_handoff_certified is None
            else "strict_handoff_not_certified"
        )
    if not features.planner_fallback_certified:
        reasons.append("planner_fallback_missing")
    if not features.supervisor_healthy:
        reasons.append("supervisor_unhealthy")
    if not features.canonicalization_succeeded:
        reasons.append("phase_one_handoff_canonicalization_failed")
    if not features.canonical_diagnostics_valid:
        reasons.append("canonical_state_diagnostics_invalid")
    if settings.require_public_highspy and not features.public_highspy_validated:
        reasons.append("standalone_public_highspy_not_validated")
    if not features.deployment_watchdog_validated:
        reasons.append("deployment_watchdog_not_validated")
    if not features.sufficient_shadow_evidence:
        reasons.append("insufficient_shadow_evidence")
    if not features.state_sensitivity_acceptable:
        reasons.append("state_sensitivity_not_qualified")
    if not settings.minimum_cells <= features.n_cells <= settings.maximum_cells:
        reasons.append("cell_count_outside_validated_range")
    if not settings.minimum_variables <= features.n_variables <= settings.maximum_variables:
        reasons.append("variable_count_outside_validated_range")
    if not 0 < features.finite_pool_size <= settings.maximum_pool_size:
        reasons.append("finite_pool_size_outside_validated_range")
    if not 0 <= features.turn_count <= settings.maximum_turns:
        reasons.append("turn_count_outside_validated_range")
    handoff_seconds = (
        features.phase_one_seconds
        if features.handoff_seconds is None
        else features.handoff_seconds
    )
    maximum_handoff_seconds = (
        settings.maximum_phase_one_seconds
        if features.handoff_source == "phase_one"
        else settings.maximum_handoff_seconds
    )
    if not math.isfinite(handoff_seconds) or not (
        0.0 <= handoff_seconds <= maximum_handoff_seconds
    ):
        reasons.append(
            "phase_one_latency_outside_budget"
            if features.handoff_source == "phase_one"
            else "handoff_latency_outside_budget"
        )
    if not math.isfinite(features.estimated_sparse_seconds) or not (
        0.0 <= features.estimated_sparse_seconds <= settings.maximum_estimated_sparse_seconds
    ):
        reasons.append("estimated_sparse_latency_outside_validated_range")
    if not math.isclose(
        features.body_length,
        settings.production_body_length,
        rel_tol=0.0,
        abs_tol=settings.body_tolerance,
    ) or not math.isclose(
        features.body_height,
        settings.production_body_height,
        rel_tol=0.0,
        abs_tol=settings.body_tolerance,
    ):
        reasons.append("body_dimensions_outside_validated_class")
    work = features.structural_work
    if work > settings.maximum_validated_work:
        reasons.append("structural_work_above_validated_range")

    # Shadow mode is the mechanism used to gather deployment and quality
    # evidence.  It may bypass only production-qualification gates; it may
    # never bypass a missing strict certificate, planner fallback, healthy
    # supervisor, canonical handoff, or structural/latency guard.
    shadow_deferred = {
        "standalone_public_highspy_not_validated",
        "deployment_watchdog_not_validated",
        "insufficient_shadow_evidence",
        "state_sensitivity_not_qualified",
    }
    blocking = [
        reason for reason in reasons if not (shadow_mode and reason in shadow_deferred)
    ]
    eligible = not blocking
    high_work = bool(
        eligible
        and settings.early_certification_work_threshold
        <= work
        <= settings.maximum_validated_work
    )
    return SparseRouteEligibilityDecision(
        eligible,
        tuple(reasons),
        work,
        high_work,
        high_work,
        "shadow" if shadow_mode else "production",
    )


__all__ = [
    "SparseRouteEligibilityDecision",
    "SparseRouteEligibilitySettings",
    "SparseRouteFeatures",
    "count_route_turns",
    "evaluate_sparse_route_eligibility",
]
