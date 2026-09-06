"""Certification-gated activation of reduced turn-transition basis elements.

A ``maximal_runs`` route uses two clothoid segments per 90-degree turn.  The
nominal ``overlapping_cover`` basis splits each of those halves once more so a
child can be assigned to the explicit turn cell.  Deep reduced optimization can
make one of those children numerically negligible even though the continuous
curve remains perfectly feasible.

This module treats those extra children as *optional basis elements*.  For each
turn half we first measure, with the same exact rectangular separator used by
production certification, how much of the parent segment can be reassigned to
the turn cell while preserving the current curve.  A child is activated only
when that exact geometric support is large enough to initialize a conditioned
new degree of freedom.  Unsupported halves remain collapsed in the reduced
basis.

Every constructed hybrid point preserves the continuous curve exactly and must
still pass the ordinary authoritative corridor/endpoint certificate.  Basis
activation is therefore a numerical continuation decision, never a relaxation
of physical feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from optimization import (
    CorridorModel,
    SeparationSettings,
    compile_geometry_path,
    knot_parameters_to_raw,
    separate_rectangle_path,
)
from planning.maze_routes import RouteOptimizationProblem

from .homotopy import coarse_segment_route_indices, turn_indices

Array = NDArray[np.float64]
Cell = tuple[int, int]
HalfKind = Literal["incoming", "outgoing"]


@dataclass(frozen=True, slots=True)
class TransitionHalfSupport:
    route_index: int
    half: HalfKind
    reduced_segment: int
    parent_length: float
    support_tau: float
    support_length: float
    turn_cell_index: int


@dataclass(frozen=True, slots=True)
class BasisActivationDecision:
    route_index: int
    half: HalfKind
    reduced_segment: int
    parent_length: float
    support_length: float
    activated: bool
    child_length: float
    hybrid_segments: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelectiveBasisResult:
    problem: RouteOptimizationProblem
    parameters: Array
    supports: tuple[TransitionHalfSupport, ...]
    decisions: tuple[BasisActivationDecision, ...]
    activated_children: tuple[int, ...]
    minimum_activated_child_length: float
    minimum_support_length: float


@dataclass(frozen=True, slots=True)
class SegmentLengthFloorConstraint:
    """Conditioning inequality for selected hybrid/full-basis children.

    Returned values use the project's ``c(x) <= 0`` convention:
    ``minimum_length - L_i <= 0``.
    """

    segment_indices: tuple[int, ...]
    minimum_length: float

    def __post_init__(self) -> None:
        if not self.segment_indices:
            raise ValueError("segment_indices must not be empty")
        if len(set(self.segment_indices)) != len(self.segment_indices):
            raise ValueError("segment_indices must be unique")
        if any(i < 0 for i in self.segment_indices):
            raise ValueError("segment indices must be nonnegative")
        if not math.isfinite(self.minimum_length) or self.minimum_length <= 0.0:
            raise ValueError("minimum_length must be finite and positive")

    @property
    def row_count(self) -> int:
        return len(self.segment_indices)

    def __call__(self, knot_params: Sequence[float]) -> tuple[Array, Array]:
        x = np.asarray(knot_params, dtype=float)
        if x.ndim != 1 or x.size == 0 or x.size % 2:
            raise ValueError("knot parameters must be a nonempty [s,k] vector")
        n = x.size // 2
        if any(i >= n for i in self.segment_indices):
            raise ValueError("length-floor segment index is out of range")
        stations = x[0::2]
        previous = np.concatenate(([0.0], stations[:-1]))
        lengths = stations - previous
        values = np.empty(len(self.segment_indices), dtype=float)
        jac = np.zeros((len(self.segment_indices), x.size), dtype=float)
        for row, segment in enumerate(self.segment_indices):
            values[row] = self.minimum_length - lengths[segment]
            jac[row, 2 * segment] = -1.0
            if segment > 0:
                jac[row, 2 * (segment - 1)] = 1.0
        return values, jac

    def maximum_violation(self, knot_params: Sequence[float]) -> float:
        values, _ = self(knot_params)
        return float(np.max(values))


def _turn_cells(full_problem: RouteOptimizationProblem) -> dict[int, tuple[int, object]]:
    result: dict[int, tuple[int, object]] = {}
    for index, cell in enumerate(full_problem.corridor.cells):
        name = cell.name
        if not name.startswith("route_turn_"):
            continue
        fields = name.split("_")
        if len(fields) < 5:
            continue
        result[int(fields[2])] = (index, cell)
    return result


def _full_cell_index_by_name(full_problem: RouteOptimizationProblem) -> dict[str, int]:
    return {cell.name: i for i, cell in enumerate(full_problem.corridor.cells)}


def _single_segment_certificate(
    initial_state,
    length: float,
    end_curvature: float,
    turn_cell,
    template_problem: RouteOptimizationProblem,
    *,
    tolerance: float,
) -> bool:
    if length <= 0.0:
        return True
    corridor = CorridorModel(
        (turn_cell,),
        (0,),
        template_problem.corridor.body,
        template_problem.corridor.clearance,
    )
    report = separate_rectangle_path(
        np.asarray((float(length), float(end_curvature)), dtype=float),
        initial_state,
        corridor,
        settings=SeparationSettings(
            add_tolerance=min(1.0e-9, float(tolerance)),
            certificate_tolerance=float(tolerance),
        ),
    )
    return bool(report.certified(float(tolerance)))


def analyze_transition_support(
    cells: Sequence[Cell],
    reduced_problem: RouteOptimizationProblem,
    full_problem: RouteOptimizationProblem,
    reduced_parameters: Sequence[float],
    *,
    tolerance: float = 2.0e-7,
    bisection_iterations: int = 44,
) -> tuple[TransitionHalfSupport, ...]:
    """Measure exact turn-cell support for each reduced turn half.

    Incoming support is the longest suffix of the incoming half that is wholly
    inside the turn cell.  Outgoing support is the longest prefix of the
    outgoing half that is wholly inside the turn cell.  Membership of a suffix
    (or prefix) is monotone as its length shrinks, so a scalar bisection around
    the exact separator gives a robust activation metric.
    """
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if bisection_iterations <= 0:
        raise ValueError("bisection_iterations must be positive")
    cells = tuple(cells)
    if reduced_problem.cells != cells or full_problem.cells != cells:
        raise ValueError("support problems do not match topology")
    x = np.asarray(reduced_parameters, dtype=float)
    if x.shape != (2 * reduced_problem.corridor.n_segments,):
        raise ValueError("reduced parameter dimension mismatch")

    raw = knot_parameters_to_raw(x, initial_k=reduced_problem.initial_state.k)
    path = compile_geometry_path(raw, reduced_problem.initial_state)
    route_indices = coarse_segment_route_indices(cells, goal_entry=True)
    turns = set(turn_indices(cells))
    turn_cells = _turn_cells(full_problem)
    if set(turn_cells) != turns:
        raise ValueError("full problem does not expose expected turn cells")
    by_route: dict[int, list[int]] = {}
    for segment, route_index in enumerate(route_indices):
        by_route.setdefault(route_index, []).append(segment)

    def suffix_ok(segment: int, tau: float, turn_cell) -> bool:
        tau = float(min(1.0, max(0.0, tau)))
        length = float(path.lengths[segment]) * (1.0 - tau)
        state = path.state_at_fraction(segment, tau)
        end_k = float(path.state_at_fraction(segment, 1.0).k)
        return _single_segment_certificate(
            state, length, end_k, turn_cell, reduced_problem, tolerance=tolerance
        )

    def prefix_ok(segment: int, tau: float, turn_cell) -> bool:
        tau = float(min(1.0, max(0.0, tau)))
        length = float(path.lengths[segment]) * tau
        state = path.state_at_fraction(segment, 0.0)
        end_k = float(path.state_at_fraction(segment, tau).k)
        return _single_segment_certificate(
            state, length, end_k, turn_cell, reduced_problem, tolerance=tolerance
        )

    rows: list[TransitionHalfSupport] = []
    for route_index in sorted(turns):
        owned = by_route.get(route_index, ())
        if len(owned) != 2:
            raise ValueError("every turn must own exactly two reduced segments")
        incoming, outgoing = owned
        turn_cell_index, turn_cell = turn_cells[route_index]

        # Earliest tau whose complete incoming suffix is inside the turn cell.
        if suffix_ok(incoming, 0.0, turn_cell):
            incoming_tau = 0.0
        else:
            if not suffix_ok(incoming, 1.0 - 1.0e-12, turn_cell):
                incoming_tau = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(bisection_iterations):
                    mid = 0.5 * (lo + hi)
                    if suffix_ok(incoming, mid, turn_cell):
                        hi = mid
                    else:
                        lo = mid
                incoming_tau = hi
        incoming_length = float(path.lengths[incoming])
        rows.append(
            TransitionHalfSupport(
                route_index,
                "incoming",
                incoming,
                incoming_length,
                incoming_tau,
                max(0.0, (1.0 - incoming_tau) * incoming_length),
                turn_cell_index,
            )
        )

        # Latest tau whose complete outgoing prefix is inside the turn cell.
        if prefix_ok(outgoing, 1.0, turn_cell):
            outgoing_tau = 1.0
        else:
            if not prefix_ok(outgoing, 1.0e-12, turn_cell):
                outgoing_tau = 0.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(bisection_iterations):
                    mid = 0.5 * (lo + hi)
                    if prefix_ok(outgoing, mid, turn_cell):
                        lo = mid
                    else:
                        hi = mid
                outgoing_tau = lo
        outgoing_length = float(path.lengths[outgoing])
        rows.append(
            TransitionHalfSupport(
                route_index,
                "outgoing",
                outgoing,
                outgoing_length,
                outgoing_tau,
                max(0.0, outgoing_tau * outgoing_length),
                turn_cell_index,
            )
        )
    return tuple(rows)


def build_selective_basis(
    cells: Sequence[Cell],
    reduced_problem: RouteOptimizationProblem,
    full_problem: RouteOptimizationProblem,
    reduced_parameters: Sequence[float],
    *,
    minimum_active_child_length: float,
    support_safety_fraction: float = 0.8,
    support_tolerance: float = 2.0e-7,
    activation_granularity: Literal["half", "turn"] = "half",
) -> SelectiveBasisResult:
    """Exactly lift only transitions with nondegenerate turn-cell support.

    ``activation_granularity="half"`` activates each incoming/outgoing child
    independently.  ``"turn"`` activates the two children of one turn as a
    pair, avoiding asymmetric one-sided turn bases.
    """
    floor = float(minimum_active_child_length)
    safety = float(support_safety_fraction)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_active_child_length must be finite and positive")
    if not math.isfinite(safety) or not 0.0 < safety < 1.0:
        raise ValueError("support_safety_fraction must lie in (0,1)")
    if activation_granularity not in {"half", "turn"}:
        raise ValueError("activation_granularity must be 'half' or 'turn'")
    cells = tuple(cells)
    x = np.asarray(reduced_parameters, dtype=float)
    supports = analyze_transition_support(
        cells,
        reduced_problem,
        full_problem,
        x,
        tolerance=support_tolerance,
    )
    support_by_segment = {row.reduced_segment: row for row in supports}
    eligible_by_segment = {
        row.reduced_segment: safety * row.support_length >= floor
        for row in supports
    }
    if activation_granularity == "turn":
        by_turn: dict[int, list[TransitionHalfSupport]] = {}
        for row in supports:
            by_turn.setdefault(row.route_index, []).append(row)
        for rows in by_turn.values():
            eligible = len(rows) == 2 and all(eligible_by_segment[row.reduced_segment] for row in rows)
            for row in rows:
                eligible_by_segment[row.reduced_segment] = eligible
    route_indices = coarse_segment_route_indices(cells, goal_entry=True)
    full_name_index = _full_cell_index_by_name(full_problem)

    stations = x[0::2]
    curvatures = x[1::2]
    previous_stations = np.concatenate(([0.0], stations[:-1]))
    previous_curvatures = np.concatenate(
        ([reduced_problem.initial_state.k], curvatures[:-1])
    )

    parameters: list[float] = []
    assignments: list[int] = []
    decisions: list[BasisActivationDecision] = []
    activated_children: list[int] = []
    station = 0.0

    for segment, route_index in enumerate(route_indices):
        s0 = float(previous_stations[segment])
        s1 = float(stations[segment])
        k0 = float(previous_curvatures[segment])
        k1 = float(curvatures[segment])
        length = s1 - s0
        if length <= 0.0:
            raise ValueError("reduced path contains nonpositive segment length")
        reduced_cell = reduced_problem.corridor.cells[
            reduced_problem.corridor.segment_cells[segment]
        ]
        try:
            run_cell_index = full_name_index[reduced_cell.name]
        except KeyError as exc:
            raise ValueError(f"full corridor is missing reduced cell {reduced_cell.name}") from exc
        support = support_by_segment.get(segment)
        if support is None:
            station += length
            parameters.extend((station, k1))
            assignments.append(run_cell_index)
            continue

        proposed_child = safety * support.support_length
        activated = bool(eligible_by_segment[support.reduced_segment])
        if not activated:
            station += length
            parameters.extend((station, k1))
            assignments.append(run_cell_index)
            decisions.append(
                BasisActivationDecision(
                    support.route_index,
                    support.half,
                    segment,
                    length,
                    support.support_length,
                    False,
                    0.0,
                    (len(assignments) - 1,),
                )
            )
            continue

        child_length = proposed_child
        if child_length >= length:
            # The normal route geometry never reaches this branch, but keep the
            # basis construction strictly nondegenerate for general inputs.
            child_length = math.nextafter(length, 0.0)
        if support.half == "incoming":
            first_length = length - child_length
            split = first_length / length
            split_k = math.fma(split, k1 - k0, k0)
            station += first_length
            parameters.extend((station, split_k))
            assignments.append(run_cell_index)
            station += child_length
            parameters.extend((station, k1))
            assignments.append(support.turn_cell_index)
            active_index = len(assignments) - 1
        else:
            first_length = child_length
            split = first_length / length
            split_k = math.fma(split, k1 - k0, k0)
            station += first_length
            parameters.extend((station, split_k))
            assignments.append(support.turn_cell_index)
            active_index = len(assignments) - 1
            station += length - first_length
            parameters.extend((station, k1))
            assignments.append(run_cell_index)
        activated_children.append(active_index)
        decisions.append(
            BasisActivationDecision(
                support.route_index,
                support.half,
                segment,
                length,
                support.support_length,
                True,
                child_length,
                (active_index - 1, active_index) if support.half == "incoming" else (active_index, active_index + 1),
            )
        )

    corridor = CorridorModel(
        full_problem.corridor.cells,
        tuple(assignments),
        full_problem.corridor.body,
        full_problem.corridor.clearance,
    )
    hybrid = replace(
        full_problem,
        initial_parameters=np.asarray(parameters, dtype=float),
        corridor=corridor,
    )
    active_lengths = [d.child_length for d in decisions if d.activated]
    support_lengths = [d.support_length for d in decisions]
    return SelectiveBasisResult(
        problem=hybrid,
        parameters=np.asarray(parameters, dtype=float),
        supports=supports,
        decisions=tuple(decisions),
        activated_children=tuple(activated_children),
        minimum_activated_child_length=(
            float(min(active_lengths)) if active_lengths else math.inf
        ),
        minimum_support_length=(
            float(min(support_lengths)) if support_lengths else math.inf
        ),
    )


def _hybrid_half_support(
    problem: RouteOptimizationProblem,
    parameters: Sequence[float],
    *,
    segment: int,
    turn_cell_index: int,
    half: HalfKind,
    tolerance: float = 2.0e-7,
    bisection_iterations: int = 44,
) -> tuple[float, float, float]:
    """Return ``(parent_length, support_tau, support_length)`` for one segment."""
    x = np.asarray(parameters, dtype=float)
    if x.shape != (2 * problem.corridor.n_segments,):
        raise ValueError("parameter dimension does not match hybrid problem")
    if not 0 <= segment < problem.corridor.n_segments:
        raise ValueError("segment index out of range")
    raw = knot_parameters_to_raw(x, initial_k=problem.initial_state.k)
    path = compile_geometry_path(raw, problem.initial_state)
    turn_cell = problem.corridor.cells[int(turn_cell_index)]
    length = float(path.lengths[segment])

    def suffix_ok(tau: float) -> bool:
        tau = float(min(1.0, max(0.0, tau)))
        state = path.state_at_fraction(segment, tau)
        end_k = float(path.state_at_fraction(segment, 1.0).k)
        return _single_segment_certificate(
            state,
            length * (1.0 - tau),
            end_k,
            turn_cell,
            problem,
            tolerance=tolerance,
        )

    def prefix_ok(tau: float) -> bool:
        tau = float(min(1.0, max(0.0, tau)))
        state = path.state_at_fraction(segment, 0.0)
        end_k = float(path.state_at_fraction(segment, tau).k)
        return _single_segment_certificate(
            state,
            length * tau,
            end_k,
            turn_cell,
            problem,
            tolerance=tolerance,
        )

    if half == "incoming":
        if suffix_ok(0.0):
            tau = 0.0
        elif not suffix_ok(1.0 - 1.0e-12):
            tau = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(int(bisection_iterations)):
                mid = 0.5 * (lo + hi)
                if suffix_ok(mid):
                    hi = mid
                else:
                    lo = mid
            tau = hi
        support = max(0.0, (1.0 - tau) * length)
    elif half == "outgoing":
        if prefix_ok(1.0):
            tau = 1.0
        elif not prefix_ok(1.0e-12):
            tau = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(int(bisection_iterations)):
                mid = 0.5 * (lo + hi)
                if prefix_ok(mid):
                    lo = mid
                else:
                    hi = mid
            tau = lo
        support = max(0.0, tau * length)
    else:
        raise ValueError("unknown half kind")
    return length, float(tau), float(support)


def reanalyze_inactive_support(
    basis: SelectiveBasisResult,
    parameters: Sequence[float],
    *,
    tolerance: float = 2.0e-7,
) -> tuple[TransitionHalfSupport, ...]:
    """Recompute turn-cell support for currently collapsed transition halves."""
    rows: list[TransitionHalfSupport] = []
    turn_cells = _turn_cells(basis.problem)
    for decision in basis.decisions:
        if decision.activated:
            continue
        if len(decision.hybrid_segments) != 1:
            raise ValueError("collapsed half must own exactly one hybrid segment")
        segment = decision.hybrid_segments[0]
        turn_cell_index, _ = turn_cells[decision.route_index]
        length, tau, support = _hybrid_half_support(
            basis.problem,
            parameters,
            segment=segment,
            turn_cell_index=turn_cell_index,
            half=decision.half,
            tolerance=tolerance,
        )
        rows.append(
            TransitionHalfSupport(
                decision.route_index,
                decision.half,
                decision.reduced_segment,
                length,
                tau,
                support,
                turn_cell_index,
            )
        )
    return tuple(rows)


def activate_newly_supported_children(
    basis: SelectiveBasisResult,
    parameters: Sequence[float],
    *,
    minimum_active_child_length: float,
    support_safety_fraction: float = 0.8,
    support_tolerance: float = 2.0e-7,
    activation_granularity: Literal["half", "turn"] = "half",
) -> SelectiveBasisResult:
    """Activate newly supported collapsed halves without changing the curve.

    Existing active children are preserved exactly.  Only currently collapsed
    halves whose freshly measured turn-cell support can initialize a child at
    or above ``minimum_active_child_length`` are split.
    """
    floor = float(minimum_active_child_length)
    safety = float(support_safety_fraction)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_active_child_length must be finite and positive")
    if not math.isfinite(safety) or not 0.0 < safety < 1.0:
        raise ValueError("support_safety_fraction must lie in (0,1)")
    if activation_granularity not in {"half", "turn"}:
        raise ValueError("activation_granularity must be 'half' or 'turn'")
    x = np.asarray(parameters, dtype=float)
    if x.shape != (2 * basis.problem.corridor.n_segments,):
        raise ValueError("parameter dimension does not match basis")

    refreshed = {
        (row.route_index, row.half): row
        for row in reanalyze_inactive_support(
            basis, x, tolerance=support_tolerance
        )
    }
    candidates: dict[tuple[int, HalfKind], tuple[BasisActivationDecision, TransitionHalfSupport, float]] = {}
    for decision in basis.decisions:
        if decision.activated:
            continue
        support = refreshed[(decision.route_index, decision.half)]
        child = safety * support.support_length
        candidates[(decision.route_index, decision.half)] = (decision, support, child)

    eligible_keys = {key for key, (_d, _s, child) in candidates.items() if child >= floor}
    if activation_granularity == "turn":
        for route_index in {key[0] for key in candidates}:
            pair = {(route_index, "incoming"), (route_index, "outgoing")}
            # Existing one-sided active turns are allowed to activate the remaining
            # half independently; fully collapsed turns activate as a coherent pair.
            existing_active = any(
                d.route_index == route_index and d.activated for d in basis.decisions
            )
            if not existing_active and not pair.issubset(eligible_keys):
                eligible_keys.difference_update(pair)

    split_by_segment: dict[int, tuple[BasisActivationDecision, TransitionHalfSupport, float]] = {}
    for key in eligible_keys:
        decision, support, child = candidates[key]
        split_by_segment[decision.hybrid_segments[0]] = (decision, support, child)

    # No newly supported degree of freedom: refresh metadata and return a copy.
    if not split_by_segment:
        decisions = []
        for decision in basis.decisions:
            if decision.activated:
                decisions.append(decision)
            else:
                support = refreshed[(decision.route_index, decision.half)]
                decisions.append(
                    BasisActivationDecision(
                        decision.route_index,
                        decision.half,
                        decision.reduced_segment,
                        support.parent_length,
                        support.support_length,
                        False,
                        0.0,
                        decision.hybrid_segments,
                    )
                )
        active_lengths = []
        stations = x[0::2]
        prev = np.concatenate(([0.0], stations[:-1]))
        lengths = stations - prev
        for decision in decisions:
            if decision.activated:
                turn_child = (
                    decision.hybrid_segments[-1]
                    if decision.half == "incoming"
                    else decision.hybrid_segments[0]
                )
                active_lengths.append(float(lengths[turn_child]))
        return SelectiveBasisResult(
            basis.problem,
            x.copy(),
            basis.supports,
            tuple(decisions),
            basis.activated_children,
            float(min(active_lengths)) if active_lengths else math.inf,
            float(min((d.support_length for d in decisions), default=math.inf)),
        )

    raw = knot_parameters_to_raw(x, initial_k=basis.problem.initial_state.k)
    path = compile_geometry_path(raw, basis.problem.initial_state)
    old_assignments = tuple(basis.problem.corridor.segment_cells)
    new_params: list[float] = []
    new_assignments: list[int] = []
    old_to_new: dict[int, tuple[int, ...]] = {}
    new_station = 0.0

    for segment in range(basis.problem.corridor.n_segments):
        length = float(path.lengths[segment])
        k0 = float(path.curvatures[segment])
        k1 = float(path.curvatures[segment + 1])
        split = split_by_segment.get(segment)
        if split is None:
            new_station += length
            new_params.extend((new_station, k1))
            new_assignments.append(old_assignments[segment])
            old_to_new[segment] = (len(new_assignments) - 1,)
            continue
        decision, support, child_length = split
        turn_cell_index = support.turn_cell_index
        run_cell_index = old_assignments[segment]
        if child_length >= length:
            child_length = math.nextafter(length, 0.0)
        if decision.half == "incoming":
            first = length - child_length
            tau = first / length
            ks = math.fma(tau, k1 - k0, k0)
            new_station += first
            new_params.extend((new_station, ks))
            new_assignments.append(run_cell_index)
            first_index = len(new_assignments) - 1
            new_station += child_length
            new_params.extend((new_station, k1))
            new_assignments.append(turn_cell_index)
            second_index = len(new_assignments) - 1
        else:
            first = child_length
            tau = first / length
            ks = math.fma(tau, k1 - k0, k0)
            new_station += first
            new_params.extend((new_station, ks))
            new_assignments.append(turn_cell_index)
            first_index = len(new_assignments) - 1
            new_station += length - first
            new_params.extend((new_station, k1))
            new_assignments.append(run_cell_index)
            second_index = len(new_assignments) - 1
        old_to_new[segment] = (first_index, second_index)

    new_decisions: list[BasisActivationDecision] = []
    activated_children: list[int] = []
    for decision in basis.decisions:
        mapped = tuple(
            new_index
            for old_index in decision.hybrid_segments
            for new_index in old_to_new[old_index]
        )
        split = (
            split_by_segment.get(decision.hybrid_segments[0])
            if not decision.activated
            else None
        )
        if split is not None:
            _old_decision, support, child_length = split
            activated = True
            support_length = support.support_length
            parent_length = support.parent_length
            child = child_length
        else:
            activated = decision.activated
            support_length = (
                refreshed[(decision.route_index, decision.half)].support_length
                if not decision.activated
                else decision.support_length
            )
            parent_length = (
                refreshed[(decision.route_index, decision.half)].parent_length
                if not decision.activated
                else decision.parent_length
            )
            child = decision.child_length if decision.activated else 0.0
        new_decision = BasisActivationDecision(
            decision.route_index,
            decision.half,
            decision.reduced_segment,
            parent_length,
            support_length,
            activated,
            child,
            mapped,
        )
        new_decisions.append(new_decision)
        if activated:
            turn_child = mapped[-1] if decision.half == "incoming" else mapped[0]
            activated_children.append(turn_child)

    corridor = CorridorModel(
        basis.problem.corridor.cells,
        tuple(new_assignments),
        basis.problem.corridor.body,
        basis.problem.corridor.clearance,
    )
    problem = replace(
        basis.problem,
        initial_parameters=np.asarray(new_params, dtype=float),
        corridor=corridor,
    )
    params = np.asarray(new_params, dtype=float)
    stations = params[0::2]
    prev = np.concatenate(([0.0], stations[:-1]))
    lengths = stations - prev
    active_lengths = [float(lengths[i]) for i in activated_children]
    return SelectiveBasisResult(
        problem,
        params,
        basis.supports,
        tuple(new_decisions),
        tuple(activated_children),
        float(min(active_lengths)) if active_lengths else math.inf,
        float(min((d.support_length for d in new_decisions), default=math.inf)),
    )


__all__ = [
    "BasisActivationDecision",
    "SegmentLengthFloorConstraint",
    "SelectiveBasisResult",
    "TransitionHalfSupport",
    "activate_newly_supported_children",
    "analyze_transition_support",
    "build_selective_basis",
    "reanalyze_inactive_support",
]
