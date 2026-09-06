from __future__ import annotations

from types import SimpleNamespace
import math

import numpy as np

from planning.active_basis_optimizer import _canonical_boundary_scalar
from tools.geometry_homotopy.basis_branch import BasisBranchPolicy
from tools.geometry_homotopy.research_common import build_historical_pair


def test_terminal_w_cache_identity_collapses_cli_roundoff() -> None:
    direct = 0.8
    via_speed = math.sqrt(direct) ** 2
    assert via_speed == 0.7999999999999999
    assert _canonical_boundary_scalar(via_speed) == direct
    assert _canonical_boundary_scalar(direct) == direct


def test_expanded_basis_skips_only_infeasible_intermediate_floor(monkeypatch) -> None:
    import tools.geometry_homotopy.basis_branch as branch_module

    cells, reduced, full = build_historical_pair(0)
    calls: list[float] = []
    policy = BasisBranchPolicy(
        activation_child_threshold=1.0e-4,
        floor_schedule=(1.0e-4, 5.0e-5),
        initial_stage_steps=1,
        relaxed_stage_steps=1,
        initial_batch_schedule=(1,),
        relaxed_batch_schedule=(1,),
        initial_trust_schedule=(0.005,),
        relaxed_trust_schedule=(0.005,),
        maximum_epochs_per_floor=1,
        maximum_activation_rounds=1,
    )

    class FakeGuard:
        def __init__(self, _activated_children, minimum_length):
            self.minimum_length = float(minimum_length)

        def maximum_violation(self, _x):
            return 1.0e-3 if self.minimum_length == 1.0e-4 else 0.0

    monkeypatch.setattr(branch_module, "SegmentLengthFloorConstraint", FakeGuard)

    def fake_run_time(problem, x, *, guard, **kwargs):
        calls.append(float(guard.minimum_length))
        return SimpleNamespace(
            parameters=np.asarray(x, dtype=float).copy(),
            scalar_time=branch_module._scalar_time(problem, x),
            accepted_steps=0,
            stop_reason="no improving certified trial",
            wall_seconds=0.0,
            certified=True,
            certification={"certified": True},
        )

    monkeypatch.setattr(branch_module, "run_time", fake_run_time)
    result = branch_module.run_basis_activation_branch(
        cells, reduced, full, reduced.initial_parameters, policy=policy
    )
    assert calls == [5.0e-5]
    assert result.activation_rounds
    first = result.activation_rounds[0]["floor_stages"][0]
    assert first["floor"] == 1.0e-4
    assert first["status"] == "skipped_initial_floor_violation"
    # The final floor still ran and exhausted; any incompleteness can only be
    # the one-round activation ceiling, not the skipped continuation floor.
    assert result.activation_rounds[0]["final_floor_exhausted"] is True
    assert "violates" not in result.termination_reason.lower()
