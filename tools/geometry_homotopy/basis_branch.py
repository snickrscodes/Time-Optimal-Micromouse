"""Certification-gated selective basis branch after reduced convergence.

The reduced trajectory is always retained as an incumbent.  Extra turn-cell
children are activated only for *coherent turn pairs* whose exact turn-cell
support can initialize both children above a conditioning threshold.  The
newly activated basis is optimized behind an absolute child-length floor; that
floor is then relaxed by continuation.  If a collapsed turn later develops
sufficient support, its pair can be activated in a subsequent round.

Every incumbent replacement and every basis change is independently certified.
The algorithm can therefore abandon any numerically troublesome branch without
losing the already-certified reduced solution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from benchmarks.common.certification import certify_parameters_on_problem
from optimization import knot_parameters_to_raw, scalar_reverse_solver
from optimization import time_model as time_model_dispatch
from planning.maze_routes import RouteOptimizationProblem

from .basis_activation import (
    SegmentLengthFloorConstraint,
    SelectiveBasisResult,
    activate_newly_supported_children,
    build_selective_basis,
    reanalyze_inactive_support,
)
from .research_common import geometry_metrics, run_time

Array = NDArray[np.float64]
Cell = tuple[int, int]
EventCallback = Callable[[str, dict[str, Any]], None]
IncumbentCallback = Callable[[RouteOptimizationProblem, Array, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class BasisBranchPolicy:
    activation_child_threshold: float = 7.5e-4
    support_safety_fraction: float = 0.8
    activation_granularity: str = "turn"
    floor_schedule: tuple[float, ...] = (7.5e-4, 7.0e-4, 6.5e-4, 6.0e-4)
    initial_stage_steps: int = 16
    relaxed_stage_steps: int = 8
    initial_batch_schedule: tuple[int, ...] = (12, 8, 4, 2, 1)
    relaxed_batch_schedule: tuple[int, ...] = (8, 4, 2, 1)
    initial_trust_schedule: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005)
    relaxed_trust_schedule: tuple[float, ...] = (0.02, 0.01, 0.005)
    maximum_raw_iterations_initial: int = 12
    maximum_raw_iterations_relaxed: int = 8
    stage_wall_seconds_initial: float = 60.0
    stage_wall_seconds_relaxed: float = 30.0
    maximum_epochs_per_floor: int = 4
    maximum_activation_rounds: int = 4
    feasibility_tolerance: float = 2.0e-7
    curvature_slope_limit: float = 50.0
    highs_threads: int = 1
    # Production parameterization.  Defaults exactly preserve the historical-five
    # qualification dynamics and scan resolution.
    init_w: float = 0.8
    terminal_w_max: float | None = None
    n_scan: int = 96
    envelope_scan: int = 48
    domain_scan: int = 96

    def __post_init__(self) -> None:
        if self.activation_granularity != "turn":
            raise ValueError("production-candidate branch currently requires turn-pair activation")
        if not math.isfinite(self.activation_child_threshold) or self.activation_child_threshold <= 0.0:
            raise ValueError("activation threshold must be finite and positive")
        if not math.isfinite(self.support_safety_fraction) or not 0.0 < self.support_safety_fraction < 1.0:
            raise ValueError("support safety fraction must lie in (0,1)")
        if not self.floor_schedule or any(v <= 0.0 for v in self.floor_schedule):
            raise ValueError("floor schedule must contain positive values")
        if self.floor_schedule[0] != self.activation_child_threshold:
            raise ValueError("first floor must equal activation threshold")
        if any(b > a for a, b in zip(self.floor_schedule, self.floor_schedule[1:])):
            raise ValueError("floor schedule must be nonincreasing")
        if self.highs_threads <= 0:
            raise ValueError("highs_threads must be positive")
        if not math.isfinite(self.init_w) or self.init_w <= 0.0:
            raise ValueError("init_w must be finite and positive")
        if self.terminal_w_max is not None and (
            not math.isfinite(self.terminal_w_max) or self.terminal_w_max <= 0.0
        ):
            raise ValueError("terminal_w_max must be finite and positive when provided")
        if min(self.n_scan, self.envelope_scan, self.domain_scan) <= 0:
            raise ValueError("scan resolutions must be positive")


@dataclass(slots=True)
class BasisBranchResult:
    status: str
    converged: bool
    termination_reason: str
    reduced_time: float
    final_time: float
    improved_over_reduced: bool
    final_problem: RouteOptimizationProblem
    final_parameters: Array
    final_basis: SelectiveBasisResult | None
    final_certification: dict[str, Any]
    activation_rounds: list[dict[str, Any]] = field(default_factory=list)
    incumbent_history: list[dict[str, Any]] = field(default_factory=list)
    wall_seconds: float = 0.0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "reduced_time": self.reduced_time,
            "final_time": self.final_time,
            "improved_over_reduced": self.improved_over_reduced,
            "final_segments": self.final_problem.corridor.n_segments,
            "final_metrics": geometry_metrics(self.final_problem, self.final_parameters),
            "final_certification": self.final_certification,
            "activation_rounds": self.activation_rounds,
            "incumbent_history": self.incumbent_history,
            "wall_seconds": self.wall_seconds,
        }


def _scalar_time(
    problem: RouteOptimizationProblem, x: Sequence[float],
    policy: BasisBranchPolicy = BasisBranchPolicy(),
) -> float:
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    return float(
        time_model_dispatch.evaluate_time_scalar(
            raw,
            init_w=policy.init_w,
            terminal_w_max=policy.terminal_w_max,
            initial_k=problem.initial_state.k,
            n_scan=policy.n_scan,
            envelope_scan=policy.envelope_scan,
            domain_scan=policy.domain_scan,
        )
    )


def _certify(problem, x, policy: BasisBranchPolicy, extra=None):
    cert = certify_parameters_on_problem(
        problem,
        x,
        tolerance=policy.feasibility_tolerance,
        maximum_abs_sigma=policy.curvature_slope_limit,
    )
    if extra is not None:
        values, _ = extra(x)
        upper = float(np.max(values)) if len(values) else -math.inf
        cert["auxiliary_inequality_upper_bound"] = upper
        cert["certified"] = bool(cert["certified"] and upper <= policy.feasibility_tolerance)
    return cert


def run_basis_activation_branch(
    cells: Sequence[Cell],
    reduced_problem: RouteOptimizationProblem,
    full_problem: RouteOptimizationProblem,
    reduced_parameters: Sequence[float],
    *,
    policy: BasisBranchPolicy = BasisBranchPolicy(),
    event_callback: EventCallback | None = None,
    incumbent_callback: IncumbentCallback | None = None,
) -> BasisBranchResult:
    started = time.perf_counter()

    def emit(event: str, **payload: Any) -> None:
        if event_callback is not None:
            event_callback(event, payload)
    reduced_x = np.asarray(reduced_parameters, dtype=float).copy()
    reduced_cert = _certify(reduced_problem, reduced_x, policy)
    if not reduced_cert["certified"]:
        raise ValueError("basis branch requires a certified reduced incumbent")
    reduced_time = _scalar_time(reduced_problem, reduced_x, policy)
    emit(
        "branch_start",
        reduced_time=reduced_time,
        reduced_segments=reduced_problem.corridor.n_segments,
        reduced_certification=reduced_cert,
    )

    incumbent_problem = reduced_problem
    incumbent_x = reduced_x.copy()
    incumbent_time = reduced_time
    incumbent_cert = reduced_cert
    history = [{
        "source": "reduced",
        "time": reduced_time,
        "segments": reduced_problem.corridor.n_segments,
        "certification": reduced_cert,
    }]

    basis = build_selective_basis(
        cells,
        reduced_problem,
        full_problem,
        reduced_x,
        minimum_active_child_length=policy.activation_child_threshold,
        support_safety_fraction=policy.support_safety_fraction,
        activation_granularity=policy.activation_granularity,
    )
    branch_x = basis.parameters.copy()
    initial_cert = _certify(basis.problem, branch_x, policy)
    emit(
        "activation_initialized",
        segments=basis.problem.corridor.n_segments,
        activated_children=len(basis.activated_children),
        decisions=[asdict(d) for d in basis.decisions],
        certification=initial_cert,
    )
    if not initial_cert["certified"]:
        emit(
            "branch_incomplete",
            reason="selective basis initialization failed independent certification",
        )
        return BasisBranchResult(
            "activation_initialization_failed",
            False,
            "selective basis initialization failed independent certification",
            reduced_time,
            incumbent_time,
            False,
            incumbent_problem,
            incumbent_x,
            None,
            incumbent_cert,
            [],
            history,
            time.perf_counter() - started,
        )
    if not basis.activated_children:
        emit(
            "branch_complete",
            reason="no inactive turn pair has enough two-sided support to activate",
            final_time=reduced_time,
            final_segments=reduced_problem.corridor.n_segments,
        )
        return BasisBranchResult(
            "no_supported_turn_pair",
            True,
            "no inactive turn pair has enough two-sided support to activate",
            reduced_time,
            incumbent_time,
            False,
            incumbent_problem,
            incumbent_x,
            basis,
            incumbent_cert,
            [],
            history,
            time.perf_counter() - started,
        )

    rounds: list[dict[str, Any]] = []
    converged = False
    termination_reason = "basis branch did not establish transaction exhaustion"
    status = "incomplete"
    for activation_round in range(policy.maximum_activation_rounds):
        round_row: dict[str, Any] = {
            "activation_round": activation_round,
            "segments": basis.problem.corridor.n_segments,
            "activated_children": len(basis.activated_children),
            "decisions": [asdict(d) for d in basis.decisions],
            "floor_stages": [],
        }
        final_floor_exhausted = False
        branch_abort_reason: str | None = None
        for floor_index, floor in enumerate(policy.floor_schedule):
            guard = SegmentLengthFloorConstraint(basis.activated_children, floor)
            # A newly expanded basis can legitimately start below an *intermediate*
            # conditioning floor: the support test is performed on the preceding
            # optimized basis, and reprojection of newly activated children may move
            # them slightly.  Intermediate floors are continuation stages, not
            # convergence requirements, so skip any stricter floor the expanded
            # basis cannot satisfy and resume at the first feasible scheduled floor.
            # The final floor remains mandatory: if even it is infeasible, closure
            # has not been established and the branch stays incomplete.
            initial_floor_violation = guard.maximum_violation(branch_x)
            if initial_floor_violation > policy.feasibility_tolerance:
                is_final_floor = floor_index == len(policy.floor_schedule) - 1
                round_row["floor_stages"].append({
                    "floor": floor,
                    "status": (
                        "initial_floor_violation" if is_final_floor
                        else "skipped_initial_floor_violation"
                    ),
                    "violation": initial_floor_violation,
                })
                emit(
                    "floor_skipped_initial_violation" if not is_final_floor else "floor_initial_violation",
                    activation_round=activation_round,
                    floor=floor,
                    violation=initial_floor_violation,
                    final_floor=is_final_floor,
                )
                if is_final_floor:
                    branch_abort_reason = (
                        f"activation round {activation_round}: basis violates final "
                        f"conditioning floor {floor:g}"
                    )
                    break
                continue

            floor_exhausted = False
            for epoch in range(policy.maximum_epochs_per_floor):
                first = floor_index == 0 and epoch == 0
                result = run_time(
                    basis.problem,
                    branch_x,
                    guard=guard,
                    accepted_steps=(
                        policy.initial_stage_steps if first else policy.relaxed_stage_steps
                    ),
                    batch_schedule=(
                        policy.initial_batch_schedule if first else policy.relaxed_batch_schedule
                    ),
                    trust_schedule=(
                        policy.initial_trust_schedule if first else policy.relaxed_trust_schedule
                    ),
                    max_filter_violation=1.0e-5,
                    pool=None,
                    wall=(
                        policy.stage_wall_seconds_initial if first else policy.stage_wall_seconds_relaxed
                    ),
                    maximum_raw_iterations=(
                        policy.maximum_raw_iterations_initial if first else policy.maximum_raw_iterations_relaxed
                    ),
                    pool_retention="reseed_after_promotion",
                    use_quadratic_model=False,
                    highs_threads=policy.highs_threads,
                    init_w=policy.init_w,
                    terminal_w_max=policy.terminal_w_max,
                    n_scan=policy.n_scan,
                    envelope_scan=policy.envelope_scan,
                    domain_scan=policy.domain_scan,
                )
                branch_x = result.parameters.copy()
                row = {
                    "floor": floor,
                    "epoch": epoch,
                    "time": result.scalar_time,
                    "accepted_steps": result.accepted_steps,
                    "stop_reason": result.stop_reason,
                    "wall_seconds": result.wall_seconds,
                    "floor_violation": guard.maximum_violation(branch_x),
                    "metrics": geometry_metrics(basis.problem, branch_x),
                    "certified": bool(result.certified),
                }
                round_row["floor_stages"].append(row)
                emit(
                    "floor_epoch_complete",
                    activation_round=activation_round,
                    **row,
                )
                if result.certified and result.scalar_time < incumbent_time - 1.0e-10:
                    incumbent_problem = basis.problem
                    incumbent_x = branch_x.copy()
                    incumbent_time = float(result.scalar_time)
                    incumbent_cert = result.certification
                    incumbent_row = {
                        "source": f"activation_round_{activation_round}_floor_{floor:g}_epoch_{epoch}",
                        "time": incumbent_time,
                        "segments": basis.problem.corridor.n_segments,
                        "certification": incumbent_cert,
                    }
                    history.append(incumbent_row)
                    emit("incumbent_promoted", **incumbent_row)
                    if incumbent_callback is not None:
                        incumbent_callback(
                            incumbent_problem,
                            incumbent_x.copy(),
                            incumbent_row,
                        )
                if result.stop_reason == "no improving certified trial":
                    floor_exhausted = True
                    break
                if result.stop_reason != "maximum accepted-step budget reached":
                    branch_abort_reason = (
                        f"activation round {activation_round}, floor {floor:g}: "
                        f"unexpected optimizer stop: {result.stop_reason}"
                    )
                    break
            if branch_abort_reason is not None:
                break
            # Intermediate floors are continuation stages and may be relaxed even
            # if their accepted-step budget remains productive.  The *final* floor
            # is different: production convergence requires transaction exhaustion
            # there, rather than merely hitting an epoch ceiling.
            if floor_index == len(policy.floor_schedule) - 1:
                final_floor_exhausted = floor_exhausted
                if not final_floor_exhausted:
                    branch_abort_reason = (
                        f"activation round {activation_round}: final conditioning floor "
                        f"{floor:g} hit epoch ceiling before transaction exhaustion"
                    )
                break

        inactive = reanalyze_inactive_support(
            basis,
            branch_x,
            tolerance=policy.feasibility_tolerance,
        )
        round_row["inactive_support_after"] = [asdict(row) for row in inactive]
        round_row["final_floor_exhausted"] = bool(final_floor_exhausted)
        round_row["abort_reason"] = branch_abort_reason
        rounds.append(round_row)
        emit(
            "activation_round_complete",
            activation_round=activation_round,
            final_floor_exhausted=bool(final_floor_exhausted),
            abort_reason=branch_abort_reason,
            inactive_support_after=[asdict(row) for row in inactive],
        )

        if branch_abort_reason is not None:
            termination_reason = branch_abort_reason
            status = "incomplete"
            break

        updated = activate_newly_supported_children(
            basis,
            branch_x,
            minimum_active_child_length=policy.activation_child_threshold,
            support_safety_fraction=policy.support_safety_fraction,
            support_tolerance=policy.feasibility_tolerance,
            activation_granularity=policy.activation_granularity,
        )
        if updated.problem.corridor.n_segments == basis.problem.corridor.n_segments:
            converged = True
            status = "complete"
            termination_reason = (
                "final conditioning floor transaction-exhausted and no inactive "
                "turn pair gained activation support"
            )
            emit(
                "branch_closed",
                activation_round=activation_round,
                reason=termination_reason,
                incumbent_time=incumbent_time,
            )
            break
        if activation_round + 1 >= policy.maximum_activation_rounds:
            termination_reason = (
                "activation-round ceiling reached while newly supported turn pairs remain"
            )
            status = "incomplete"
            break
        updated_cert = _certify(updated.problem, updated.parameters, policy)
        if not updated_cert["certified"]:
            termination_reason = "newly activated basis failed independent certification"
            status = "incomplete"
            break
        emit(
            "basis_expanded",
            activation_round=activation_round,
            old_segments=basis.problem.corridor.n_segments,
            new_segments=updated.problem.corridor.n_segments,
        )
        basis = updated
        branch_x = updated.parameters.copy()

    final_cert = _certify(incumbent_problem, incumbent_x, policy)
    emit(
        "branch_finish",
        status=status,
        converged=converged,
        termination_reason=termination_reason,
        final_time=incumbent_time,
        final_segments=incumbent_problem.corridor.n_segments,
        final_certification=final_cert,
    )
    return BasisBranchResult(
        status,
        converged,
        termination_reason,
        reduced_time,
        incumbent_time,
        incumbent_time < reduced_time - 1.0e-10,
        incumbent_problem,
        incumbent_x,
        basis,
        final_cert,
        rounds,
        history,
        time.perf_counter() - started,
    )


__all__ = ["BasisBranchPolicy", "BasisBranchResult", "run_basis_activation_branch"]
