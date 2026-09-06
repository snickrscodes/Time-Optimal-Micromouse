"""Central planner optimization architecture policy.

V10 makes the modern planner architecture explicit instead of relying on
scattered default arguments in ``main.py`` and helper modules.  The low-level
optimizer still defaults to SLSQP; this module controls route-level scheduling,
strict-feasibility fallback, and optional sparse specialist polishing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .warm_start_schedule import (
    PrimaryFilterSQPPolicySettings,
    PrimaryTimeBackendMode,
    WarmStartDeadlineSettings,
    WarmStartInitializerMode,
    WarmStartScheduleSettings,
    WarmStartSchedulingMode,
)


class PlannerArchitectureMode(str, Enum):
    LEGACY_V9 = "legacy_v9"
    INTEGRATED_V10 = "integrated_v10"
    ACTIVE_BASIS_V11 = "active_basis_v11"


class SparseSpecialistMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    SHADOW = "shadow"
    EXPERIMENTAL_PROMOTE = "experimental_promote"




@dataclass(frozen=True, slots=True)
class ActiveBasisPlannerSettings:
    """Orchestration-only settings for the qualified V11 optimizer.

    Numerical policy remains frozen in ``HomotopyPolicy``; these fields control
    only persistent work placement and whether a non-converged certified
    fallback may be returned to a caller.
    """

    work_root: str | None = None
    require_convergence: bool = True

@dataclass(frozen=True, slots=True)
class SparseSpecialistPolicySettings:
    mode: SparseSpecialistMode = SparseSpecialistMode.AUTO
    minimum_slsqp_seconds: float = 20.0
    maximum_sparse_seconds: float = 120.0
    maximum_batches: int = 5
    accepted_steps_per_batch: int = 5
    public_highspy_validated: bool = False
    deployment_watchdog_validated: bool = False
    sufficient_shadow_evidence: bool = False
    state_sensitivity_acceptable: bool = True
    telemetry_jsonl: str | None = None

    def __post_init__(self) -> None:
        if self.minimum_slsqp_seconds < 0.0:
            raise ValueError("minimum_slsqp_seconds must be nonnegative")
        if self.maximum_sparse_seconds <= 0.0:
            raise ValueError("maximum_sparse_seconds must be positive")
        if self.maximum_batches <= 0:
            raise ValueError("maximum_batches must be positive")
        if self.accepted_steps_per_batch <= 0:
            raise ValueError("accepted_steps_per_batch must be positive")


@dataclass(frozen=True, slots=True)
class PlannerOptimizationPolicy:
    architecture: PlannerArchitectureMode = PlannerArchitectureMode.INTEGRATED_V10
    active_basis: ActiveBasisPlannerSettings = ActiveBasisPlannerSettings()
    warm_start: WarmStartScheduleSettings = WarmStartScheduleSettings(
        mode=WarmStartSchedulingMode.BOUNDED_BEST_OF_BOTH,
        initializer_mode=WarmStartInitializerMode.ANALYTIC_WITH_STRICT_PHASE_ONE_FALLBACK,
        deadlines=WarmStartDeadlineSettings(),
        primary_filter_sqp=PrimaryFilterSQPPolicySettings(
            mode=PrimaryTimeBackendMode.AUTO_QUALIFIED_FILTER_SQP
        ),
    )
    sparse_specialist: SparseSpecialistPolicySettings = SparseSpecialistPolicySettings()

    @classmethod
    def legacy_v9(cls) -> "PlannerOptimizationPolicy":
        return cls(
            architecture=PlannerArchitectureMode.LEGACY_V9,
            warm_start=WarmStartScheduleSettings(
                mode=WarmStartSchedulingMode.LEGACY_ALWAYS_BOTH,
                initializer_mode=WarmStartInitializerMode.ANALYTIC_INITIALIZER,
            ),
            sparse_specialist=SparseSpecialistPolicySettings(
                mode=SparseSpecialistMode.OFF
            ),
        )

    @classmethod
    def integrated_v10(cls) -> "PlannerOptimizationPolicy":
        return cls()

    @classmethod
    def active_basis_v11(
        cls, *, work_root: str | None = None, require_convergence: bool = True
    ) -> "PlannerOptimizationPolicy":
        return cls(
            architecture=PlannerArchitectureMode.ACTIVE_BASIS_V11,
            active_basis=ActiveBasisPlannerSettings(
                work_root=work_root, require_convergence=require_convergence
            ),
            sparse_specialist=SparseSpecialistPolicySettings(mode=SparseSpecialistMode.OFF),
        )


__all__ = [
    "ActiveBasisPlannerSettings",
    "PlannerArchitectureMode",
    "PlannerOptimizationPolicy",
    "SparseSpecialistMode",
    "SparseSpecialistPolicySettings",
]
