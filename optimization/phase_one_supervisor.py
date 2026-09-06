"""Externally bounded execution for elastic Phase I.

Phase I may call native SLSQP code that cannot observe the cooperative wall-clock
check in :func:`optimization.path_optimizer.run_phase_one`.  This module runs the
complete call behind the same fresh-interpreter supervisor used by adaptive sparse
polishing.  Diagnostic checkpoints are never planner-facing unless their strict
finite and continuous certificate is complete.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .constraint_generation import ConstraintPool, SeparationReport
from .kkt_lsq import BoxBounds
from .knot_parameterization import KnotCoordinateSettings, LogLengthKnotMap
from .path_optimizer import PhaseOneResult, PhaseOneSettings, run_phase_one
from .sqp_supervisor import (
    SupervisedBatchOutcome,
    SupervisedBatchRunner,
    SupervisorSettings,
    WorkerContext,
)
from .sqp_supervisor_service import (
    ExternalSupervisorServiceClient,
    SupervisorServiceClientSettings,
)


@dataclass(frozen=True, slots=True)
class PhaseOneDiagnosticCheckpoint:
    """Checkpoint-safe Phase-I state retained after a failed or timed-out call."""

    x: np.ndarray
    slack: float
    pool: ConstraintPool
    exchange_rounds: int
    final_separation: SeparationReport
    final_finite_inequality: float
    final_equality_inf: float
    final_solver_success: bool
    final_solver_message: str
    certification_retries: int
    strictly_certified: bool
    elapsed_seconds: float

    @classmethod
    def from_result(
        cls, result: PhaseOneResult, *, elapsed_seconds: float
    ) -> "PhaseOneDiagnosticCheckpoint":
        return cls(
            np.asarray(result.x, dtype=float).copy(),
            float(result.slack),
            result.pool.copy(),
            int(result.exchange_rounds),
            result.final_separation,
            float(result.final_finite_inequality),
            float(result.final_equality_inf),
            bool(result.final_solver_success),
            str(result.final_solver_message),
            int(result.certification_retries),
            bool(result.success),
            float(elapsed_seconds),
        )

    def checkpoint_safe_copy(self) -> "PhaseOneDiagnosticCheckpoint":
        return PhaseOneDiagnosticCheckpoint(
            self.x.copy(),
            self.slack,
            self.pool.copy(),
            self.exchange_rounds,
            self.final_separation,
            self.final_finite_inequality,
            self.final_equality_inf,
            self.final_solver_success,
            self.final_solver_message,
            self.certification_retries,
            self.strictly_certified,
            self.elapsed_seconds,
        )

    def to_result(self) -> PhaseOneResult | None:
        if not self.strictly_certified:
            return None
        return PhaseOneResult(
            self.x.copy(),
            self.slack,
            True,
            "continuously feasible",
            self.pool.copy(),
            self.exchange_rounds,
            self.final_separation,
            self.certification_retries,
            self.final_finite_inequality,
            self.final_equality_inf,
            self.final_solver_success,
            self.final_solver_message,
        )


@dataclass(frozen=True, slots=True)
class SupervisedPhaseOneResult:
    outcome: SupervisedBatchOutcome
    completed: PhaseOneResult | None
    last_checkpoint: PhaseOneDiagnosticCheckpoint | None
    certified_result: PhaseOneResult | None


class _CheckpointingPhaseOne:
    """Worker-local wrapper that emits the terminal result as a safe checkpoint."""

    def __init__(self, context: WorkerContext) -> None:
        self.context = context
        self.started = time.perf_counter()

    def run(self, arguments: dict[str, Any]) -> PhaseOneResult:
        self.context.set_stage("phase_one")

        def emit(result: PhaseOneResult) -> None:
            self.context.set_stage("phase_one_checkpoint")
            self.context.emit_checkpoint(
                PhaseOneDiagnosticCheckpoint.from_result(
                    result, elapsed_seconds=time.perf_counter() - self.started
                )
            )
            self.context.set_stage("phase_one")

        arguments["checkpoint_callback"] = emit
        result = run_phase_one(**arguments)
        self.context.set_stage("phase_one_complete")
        return result


def _phase_one_worker(payload: Any, context: WorkerContext) -> PhaseOneResult:
    if not isinstance(payload, dict):
        raise TypeError("Phase-I worker payload must be a dictionary")
    injection = payload.get("_test_injection", "none")
    if injection == "runtime_hang":
        context.set_stage("phase_one")
        while True:
            time.sleep(1.0)
    if injection == "worker_crash":
        import os
        os._exit(23)
    arguments = dict(payload)
    arguments.pop("_test_injection", None)
    return _CheckpointingPhaseOne(context).run(arguments)


def _valid_phase_one_result(value: Any) -> bool:
    return isinstance(value, PhaseOneResult)


def run_supervised_phase_one(
    x0: Sequence[float],
    corridor: Any,
    initial_state: Any,
    *,
    initial_s: float = 0.0,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    equalities: Any | None = None,
    endpoint_target: Any | None = None,
    additional_inequalities: Any | None = None,
    pool: ConstraintPool | None = None,
    settings: PhaseOneSettings = PhaseOneSettings(),
    coordinate_map: LogLengthKnotMap | None = None,
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
    supervisor_settings: SupervisorSettings = SupervisorSettings(),
    supervisor_service_settings: SupervisorServiceClientSettings | None = None,
    initial_safe_checkpoint: PhaseOneDiagnosticCheckpoint | None = None,
    test_injection: str = "none",
) -> SupervisedPhaseOneResult:
    """Run Phase I behind a killable process boundary.

    A non-certified diagnostic checkpoint is returned for analysis only.  The
    ``certified_result`` field is populated exclusively when a completed or
    retained checkpoint passed the unchanged strict Phase-I certificate.
    """
    payload = {
        "x0": np.asarray(x0, dtype=float).copy(),
        "corridor": corridor,
        "initial_state": initial_state,
        "initial_s": initial_s,
        "bounds": bounds,
        "equalities": equalities,
        "endpoint_target": endpoint_target,
        "additional_inequalities": additional_inequalities,
        "pool": pool,
        "settings": settings,
        "coordinate_map": coordinate_map,
        "coordinate_settings": coordinate_settings,
        "_test_injection": test_injection,
    }
    runner_settings = replace(
        supervisor_settings, deadline_refresh_on_checkpoint=False
    )
    if supervisor_service_settings is None:
        with SupervisedBatchRunner(
            _phase_one_worker,
            settings=runner_settings,
            result_validator=_valid_phase_one_result,
        ) as runner:
            outcome = runner.run(
                payload,
                initial_checkpoint=initial_safe_checkpoint,
            )
    else:
        client = ExternalSupervisorServiceClient(supervisor_service_settings)
        outcome = client.run(
            _phase_one_worker,
            payload,
            worker_settings=runner_settings,
            initial_checkpoint=initial_safe_checkpoint,
            result_validator=_valid_phase_one_result,
        )
    completed = (
        outcome.result if outcome.success and isinstance(outcome.result, PhaseOneResult) else None
    )
    checkpoint = (
        outcome.last_checkpoint
        if isinstance(outcome.last_checkpoint, PhaseOneDiagnosticCheckpoint)
        else None
    )
    certified = completed if completed is not None and completed.success else None
    if certified is None and checkpoint is not None:
        certified = checkpoint.to_result()
    return SupervisedPhaseOneResult(outcome, completed, checkpoint, certified)


__all__ = [
    "PhaseOneDiagnosticCheckpoint",
    "SupervisedPhaseOneResult",
    "run_supervised_phase_one",
]
