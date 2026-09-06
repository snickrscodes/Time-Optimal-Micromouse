from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.geometry_homotopy.research_common import build_historical_pair
from tools.geometry_homotopy.serial_runtime import (
    SerialWorkerResult,
    THREAD_ENVIRONMENT_VARIABLES,
    force_single_thread_environment,
)
import tools.geometry_homotopy.state_machine as state_machine
from tools.geometry_homotopy.state_machine import (
    HomotopyPolicy,
    JsonlLogger,
    _run_supervised_basis_branch,
)


def test_single_thread_environment_overrides_existing_values() -> None:
    env = {name: "8" for name in THREAD_ENVIRONMENT_VARIABLES}
    env["PYTHONHASHSEED"] = "123"
    returned = force_single_thread_environment(env)
    assert returned is env
    assert all(env[name] == "1" for name in THREAD_ENVIRONMENT_VARIABLES)
    assert env["PYTHONHASHSEED"] == "0"


def test_supervised_branch_timeout_is_serial_and_falls_back_to_reduced(
    tmp_path: Path, monkeypatch
) -> None:
    cells, reduced, full = build_historical_pair(0)
    x = reduced.initial_parameters.copy()
    seen: dict[str, object] = {}

    def fake_worker(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return SerialWorkerResult(
            "timeout", -9, 0.25, False, False, True,
            str(kwargs.get("stdout_path")), str(kwargs.get("stderr_path")),
        )

    monkeypatch.setattr(state_machine, "run_serial_worker", fake_worker)
    policy = HomotopyPolicy(basis_branch_worker_timeout_seconds=10.0)
    logger = JsonlLogger(tmp_path / "events.jsonl")
    final_problem, final_x, kind, payload = _run_supervised_basis_branch(
        0,
        cells,
        reduced,
        full,
        x,
        tmp_path,
        tmp_path / "checkpoints",
        logger,
        policy,
        timeout_seconds=0.25,
    )

    assert final_problem is reduced
    np.testing.assert_array_equal(final_x, x)
    assert kind == "reduced"
    assert payload["worker_status"] == "timeout"
    assert payload["fallback_to_reduced_incumbent"] is True
    assert payload["execution_mode"] == "serial_single_worker"
    assert payload["worker_timeout_seconds"] == 0.25
    assert seen["timeout_seconds"] == 0.25
    worker_env = seen["env"]
    assert isinstance(worker_env, dict)
    assert all(worker_env[name] == "1" for name in THREAD_ENVIRONMENT_VARIABLES)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    starts = [row for row in events if row["event"] == "basis_branch_worker_start"]
    assert len(starts) == 1
    assert starts[0]["execution_mode"] == "serial_single_worker"


def test_supervised_branch_parent_recertifies_successful_reduced_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    cells, reduced, full = build_historical_pair(0)
    x = reduced.initial_parameters.copy()

    def fake_worker(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        candidate_path = output.with_suffix(".fake.npy")
        np.save(candidate_path, x)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "complete",
                    "seed": 0,
                    "branch_converged": True,
                    "branch_status": "complete",
                    "branch_termination_reason": "test",
                    "final_kind": "reduced",
                    "final_parameters": str(candidate_path),
                    "final_segment_cells": None,
                    "result": {"status": "complete", "converged": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        marker = kwargs.get("completion_marker")
        if marker is not None:
            Path(marker).write_text("complete\n", encoding="utf-8")
        return SerialWorkerResult(
            "complete", 0, 0.01, True, False, False,
            str(kwargs.get("stdout_path")), str(kwargs.get("stderr_path")),
        )

    monkeypatch.setattr(state_machine, "run_serial_worker", fake_worker)
    policy = HomotopyPolicy(basis_branch_worker_timeout_seconds=10.0)
    logger = JsonlLogger(tmp_path / "events.jsonl")
    final_problem, final_x, kind, payload = _run_supervised_basis_branch(
        0,
        cells,
        reduced,
        full,
        x,
        tmp_path,
        tmp_path / "checkpoints",
        logger,
        policy,
        timeout_seconds=2.0,
    )

    assert final_problem is reduced
    np.testing.assert_array_equal(final_x, x)
    assert kind == "reduced"
    assert payload["worker_status"] == "complete"
    assert payload["branch_converged"] is True
    assert payload["parent_promoted"] is True
    assert payload["parent_certification"]["certified"] is True
    assert payload["parent_candidate_time"] <= payload["parent_reduced_time"] + 1.0e-10


def test_historical_campaign_dispatch_is_strictly_serial(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_route(seed, output_dir, *, policy):
        # If dispatch ever became concurrent, this simple ordered trace would no
        # longer be a sufficient architecture assertion.  The production
        # candidate intentionally performs one complete route call at a time.
        calls.append(int(seed))
        return state_machine.RouteRunResult(
            seed=int(seed),
            cells=((0, 0), (0, 1)),
            status="complete",
            stage="complete",
            converged=True,
            initial_time=1.0,
            final_time=0.9,
            reduced_converged=True,
            basis_branch_complete=True,
            basis_branch_converged=True,
            final_basis_kind="reduced",
            final_segments=1,
        )

    monkeypatch.setattr(state_machine, "run_route_to_convergence", fake_route)
    payload = state_machine.run_historical_five(
        tmp_path / "campaign", seeds=(2, 0, 4), policy=HomotopyPolicy()
    )
    assert calls == [2, 0, 4]
    assert payload["execution_mode"] == "serial_single_worker"
    assert payload["complete"] is True
    assert payload["numerical_thread_environment"] == {
        name: "1" for name in THREAD_ENVIRONMENT_VARIABLES
    }
    on_disk = json.loads((tmp_path / "campaign" / "campaign.json").read_text())
    assert [row["seed"] for row in on_disk["rows"]] == [2, 0, 4]
