"""Adaptive certified short-batch schedule for the sparse SQP candidate.

This module is intentionally experimental.  It orchestrates repeated short
finite-SQP batches with exact continuous separation between batches.  The
scheduler never replaces a continuously certified incumbent by an infeasible
candidate and stops using objective progress, scale-normalized criticality,
cut activity, and explicit wall/evaluation budgets.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .path_optimizer import (
    BoxBounds,
    ConstraintPool,
    ExchangeResult,
    ExchangeSettings,
    KnotCoordinateSettings,
    LogLengthKnotMap,
    run_constraint_generation,
)

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AdaptiveBatchSettings:
    batch_iterations: int = 5
    minimum_batches: int = 2
    maximum_batches: int = 16
    maximum_wall_seconds: float = 180.0
    maximum_objective_evaluations: int = 160
    relative_objective_tolerance: float = 2.0e-4
    absolute_objective_tolerance: float = 2.0e-4
    required_stalled_certified_batches: int = 2
    criticality_target: float = 1.0e-5
    continue_after_new_cuts: bool = True
    continue_until_first_certified: bool = True

    def __post_init__(self) -> None:
        if self.batch_iterations <= 0:
            raise ValueError("batch_iterations must be positive")
        if self.minimum_batches <= 0:
            raise ValueError("minimum_batches must be positive")
        if self.maximum_batches < self.minimum_batches:
            raise ValueError("maximum_batches must be >= minimum_batches")
        if self.maximum_wall_seconds <= 0.0:
            raise ValueError("maximum_wall_seconds must be positive")
        if self.maximum_objective_evaluations <= 0:
            raise ValueError("maximum_objective_evaluations must be positive")
        if self.required_stalled_certified_batches <= 0:
            raise ValueError("required_stalled_certified_batches must be positive")
        for name, value in (
            ("relative_objective_tolerance", self.relative_objective_tolerance),
            ("absolute_objective_tolerance", self.absolute_objective_tolerance),
            ("criticality_target", self.criticality_target),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class AdaptiveBatchRecord:
    batch: int
    wall_seconds: float
    cumulative_seconds: float
    objective: float
    continuously_certified: bool
    cuts_added: int
    pool_size: int
    objective_evaluations: int
    accepted_steps: int
    normalized_stationarity: float
    normalized_complementarity: float
    criticality: float
    criticality_success: bool
    objective_improvement: float
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class AdaptivePolishResult:
    result: ExchangeResult
    best_certified_result: ExchangeResult | None
    batches: tuple[AdaptiveBatchRecord, ...]
    stop_reason: str
    total_objective_evaluations: int
    wall_seconds: float


def _diagnostics(exchange: ExchangeResult) -> Any | None:
    if not exchange.rounds:
        return None
    return exchange.rounds[-1].finite.solver_result.diagnostics


def run_adaptive_sparse_polish(
    x0: Sequence[float],
    objective: Any,
    corridor: Any,
    initial_state: Any,
    *,
    bounds: BoxBounds | Sequence[tuple[float | None, float | None]] | None = None,
    endpoint_target: Any | None = None,
    equalities: Any | None = None,
    additional_inequalities: Any | None = None,
    pool: ConstraintPool | None = None,
    exchange_settings: ExchangeSettings,
    batch_settings: AdaptiveBatchSettings = AdaptiveBatchSettings(),
    coordinate_map: LogLengthKnotMap | None = None,
    coordinate_settings: KnotCoordinateSettings = KnotCoordinateSettings(),
) -> AdaptivePolishResult:
    if exchange_settings.finite_solver != "sparse_sqp":
        raise ValueError("adaptive polish requires finite_solver='sparse_sqp'")

    started = time.perf_counter()
    x = np.asarray(x0, dtype=float).copy()
    active_pool = pool
    best_certified: ExchangeResult | None = None
    last_result: ExchangeResult | None = None
    records: list[AdaptiveBatchRecord] = []
    total_objective_evaluations = 0
    stalled_certified = 0
    previous_certified_objective = math.inf
    stop_reason = "maximum batches reached"

    finite_settings = replace(
        exchange_settings.sparse_sqp,
        max_iterations=batch_settings.batch_iterations,
    )
    one_round = replace(
        exchange_settings,
        maximum_rounds=1,
        certified_polish_rounds=0,
        sparse_sqp=finite_settings,
    )

    for batch in range(1, batch_settings.maximum_batches + 1):
        batch_started = time.perf_counter()
        result = run_constraint_generation(
            x,
            objective,
            corridor,
            initial_state,
            bounds=bounds,
            endpoint_target=endpoint_target,
            equalities=equalities,
            additional_inequalities=additional_inequalities,
            pool=active_pool,
            settings=one_round,
            coordinate_map=coordinate_map,
            coordinate_settings=coordinate_settings,
        )
        batch_wall = time.perf_counter() - batch_started
        last_result = result
        x = result.x.copy()
        active_pool = result.pool
        round_result = result.rounds[-1]
        finite = round_result.finite
        diagnostics = _diagnostics(result)
        objective_calls = int(finite.objective_calls)
        total_objective_evaluations += objective_calls
        certified = bool(
            result.final_separation.certified(
                exchange_settings.separation.certificate_tolerance
            )
            and result.final_finite_inequality
            <= exchange_settings.finite_inequality_tolerance
            and result.final_equality_inf
            <= exchange_settings.finite_equality_tolerance
        )
        improvement = 0.0
        if certified:
            if best_certified is None or result.objective < best_certified.objective:
                best_certified = result
            if math.isfinite(previous_certified_objective):
                improvement = previous_certified_objective - result.objective
            previous_certified_objective = min(
                previous_certified_objective, result.objective
            )

        normalized_stationarity = float(
            getattr(diagnostics, "kkt_stationarity_scaled", math.inf)
        )
        normalized_complementarity = float(
            getattr(diagnostics, "kkt_complementarity_scaled", math.inf)
        )
        criticality = float(getattr(diagnostics, "kkt_criticality", math.inf))
        criticality_success = bool(
            getattr(diagnostics, "kkt_criticality_success", False)
        )
        accepted = int(getattr(diagnostics, "accepted_steps", 0))
        cuts = int(round_result.cuts_added)

        current_stop: str | None = None
        elapsed = time.perf_counter() - started
        if batch >= batch_settings.minimum_batches:
            objective_threshold = max(
                batch_settings.absolute_objective_tolerance,
                batch_settings.relative_objective_tolerance
                * max(1.0, abs(result.objective)),
            )
            if certified and cuts == 0:
                if improvement <= objective_threshold:
                    stalled_certified += 1
                else:
                    stalled_certified = 0
                first_order_small = (
                    criticality_success
                    and criticality <= batch_settings.criticality_target
                ) or normalized_stationarity <= batch_settings.criticality_target
                if first_order_small and stalled_certified >= 1:
                    current_stop = "certified scale-normalized stationarity reached"
                elif (
                    stalled_certified
                    >= batch_settings.required_stalled_certified_batches
                ):
                    current_stop = "certified objective progress stalled"
                elif accepted == 0:
                    current_stop = "certified batch accepted no steps"
            elif cuts > 0 and batch_settings.continue_after_new_cuts:
                stalled_certified = 0
            elif (
                not certified
                and not batch_settings.continue_until_first_certified
            ):
                current_stop = "batch was not continuously certified"

        if elapsed >= batch_settings.maximum_wall_seconds:
            current_stop = "wall-clock budget reached"
        if total_objective_evaluations >= batch_settings.maximum_objective_evaluations:
            current_stop = "objective-evaluation budget reached"

        records.append(
            AdaptiveBatchRecord(
                batch=batch,
                wall_seconds=batch_wall,
                cumulative_seconds=elapsed,
                objective=float(result.objective),
                continuously_certified=certified,
                cuts_added=cuts,
                pool_size=result.pool.size,
                objective_evaluations=objective_calls,
                accepted_steps=accepted,
                normalized_stationarity=normalized_stationarity,
                normalized_complementarity=normalized_complementarity,
                criticality=criticality,
                criticality_success=criticality_success,
                objective_improvement=float(improvement),
                stop_reason=current_stop,
            )
        )
        if current_stop is not None:
            stop_reason = current_stop
            break

    if last_result is None:
        raise AssertionError("adaptive polish did not execute")
    returned = best_certified if best_certified is not None else last_result
    return AdaptivePolishResult(
        result=returned,
        best_certified_result=best_certified,
        batches=tuple(records),
        stop_reason=stop_reason,
        total_objective_evaluations=total_objective_evaluations,
        wall_seconds=time.perf_counter() - started,
    )


# V4 keeps this import path backward compatible while moving the persistent
# state machine into a separately auditable module.
from .sparse_sqp_continuation import (  # noqa: E402,F811
    AdaptiveBatchRecord,
    AdaptiveBatchSettings,
    AdaptivePolishResult,
    AdaptiveSafeCheckpoint,
    CutProvenance,
    RestorationAttempt,
    RestorationRecord,
    SupervisedAdaptivePolishResult,
    run_adaptive_sparse_polish,
    run_supervised_adaptive_sparse_polish,
)

__all__ = [
    "AdaptiveBatchRecord",
    "AdaptiveBatchSettings",
    "AdaptiveSafeCheckpoint",
    "AdaptivePolishResult",
    "CutProvenance",
    "SupervisedAdaptivePolishResult",
    "RestorationAttempt",
    "RestorationRecord",
    "run_supervised_adaptive_sparse_polish",
    "run_adaptive_sparse_polish",
]
