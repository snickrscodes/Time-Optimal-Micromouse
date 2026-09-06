from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np

from tools.geometry_homotopy.basis_branch import BasisBranchPolicy, run_basis_activation_branch
from tools.geometry_homotopy.research_common import build_historical_pair
from tools.geometry_homotopy.serial_runtime import (
    THREAD_ENVIRONMENT_VARIABLES,
    force_single_thread_environment,
    run_serial_worker,
)
from tools.geometry_homotopy.state_machine import (
    HomotopyPolicy,
    JsonlLogger,
    _run_supervised_basis_branch,
)


def test_single_thread_environment_is_explicit() -> None:
    env: dict[str, str] = {}
    force_single_thread_environment(env)
    assert all(env[name] == "1" for name in THREAD_ENVIRONMENT_VARIABLES)
    assert env["PYTHONHASHSEED"] == "0"


def test_serial_worker_timeout_is_killable(tmp_path: Path) -> None:
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    result = run_serial_worker(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout_seconds=0.15,
        teardown_grace_seconds=0.05,
    )
    assert result.status == "timeout"
    assert result.timed_out
    assert result.wall_seconds < 2.0


def test_completion_marker_can_finish_before_teardown(tmp_path: Path) -> None:
    marker = tmp_path / "complete.marker"
    script = tmp_path / "marker_then_sleep.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('done\\n')\n"
        "import time\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    result = run_serial_worker(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout_seconds=2.0,
        completion_marker=marker,
        teardown_grace_seconds=0.05,
    )
    assert result.status == "complete"
    assert result.completion_marker_seen
    assert result.forced_teardown
    assert not result.timed_out


def test_no_supported_turn_pair_is_a_closed_basis_branch() -> None:
    cells, reduced, full = build_historical_pair(3)
    policy = BasisBranchPolicy(
        activation_child_threshold=0.1,
        floor_schedule=(0.1,),
    )
    result = run_basis_activation_branch(
        cells,
        reduced,
        full,
        reduced.initial_parameters,
        policy=policy,
    )
    assert result.status == "no_supported_turn_pair"
    assert result.converged
    assert result.final_problem is reduced
    np.testing.assert_array_equal(result.final_parameters, reduced.initial_parameters)


def test_parent_supervisor_runs_one_serial_worker_and_recertifies(tmp_path: Path) -> None:
    cells, reduced, full = build_historical_pair(0)
    branch_policy = BasisBranchPolicy(
        activation_child_threshold=0.1,
        floor_schedule=(0.1,),
    )
    policy = HomotopyPolicy(
        basis_branch_policy=branch_policy,
        basis_branch_worker_timeout_seconds=20.0,
    )
    route_dir = tmp_path / "seed_0"
    checkpoint_dir = route_dir / "checkpoints"
    logger = JsonlLogger(route_dir / "events.jsonl")
    problem, x, kind, payload = _run_supervised_basis_branch(
        0,
        cells,
        reduced,
        full,
        reduced.initial_parameters,
        route_dir,
        checkpoint_dir,
        logger,
        policy,
    )
    assert payload["execution_mode"] == "serial_single_worker"
    assert payload["worker_status"] == "complete"
    assert payload["branch_converged"] is True
    assert payload["branch_status"] == "no_supported_turn_pair"
    assert payload["thread_environment"]["OMP_NUM_THREADS"] == "1"
    assert kind == "reduced"
    assert problem is reduced
    np.testing.assert_array_equal(x, reduced.initial_parameters)
    events = [json.loads(line) for line in (route_dir / "events.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "basis_branch_worker_start" for row in events) == 1


def test_final_floor_epoch_ceiling_is_not_called_converged(monkeypatch) -> None:
    from types import SimpleNamespace
    import tools.geometry_homotopy.basis_branch as branch_module

    cells, reduced, full = build_historical_pair(0)
    policy = BasisBranchPolicy(
        activation_child_threshold=1.0e-4,
        floor_schedule=(1.0e-4,),
        initial_stage_steps=1,
        initial_batch_schedule=(1,),
        initial_trust_schedule=(0.005,),
        maximum_epochs_per_floor=1,
        maximum_activation_rounds=1,
    )

    def fake_run_time(problem, x, **kwargs):
        # A productive-but-budget-limited final-floor epoch is *not* evidence of
        # stationarity/transaction exhaustion.
        return SimpleNamespace(
            parameters=np.asarray(x, dtype=float).copy(),
            scalar_time=branch_module._scalar_time(problem, x),
            accepted_steps=1,
            stop_reason="maximum accepted-step budget reached",
            wall_seconds=0.0,
            certified=True,
            certification={"certified": True},
        )

    monkeypatch.setattr(branch_module, "run_time", fake_run_time)
    result = branch_module.run_basis_activation_branch(
        cells, reduced, full, reduced.initial_parameters, policy=policy
    )
    assert result.status == "incomplete"
    assert not result.converged
    assert "final conditioning floor" in result.termination_reason
