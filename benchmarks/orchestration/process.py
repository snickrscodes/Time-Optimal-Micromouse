from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    wall_seconds: float
    completion_marker_seen: bool = False
    forced_teardown: bool = False


class BenchmarkProcessError(RuntimeError):
    pass


class BenchmarkTimeoutError(BenchmarkProcessError):
    pass


def _terminate_group(proc: subprocess.Popen, *, grace: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
        return
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=grace)
    except Exception:
        pass


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    completion_marker: Path | None = None,
    teardown_grace: float = 3.0,
) -> ProcessResult:
    """Run one benchmark component in an isolated process group.

    If ``completion_marker`` is supplied, the child writes it only after its
    result file is fully closed. The orchestrator then allows a short normal
    teardown window. A worker stuck only in extension/plugin destruction is
    terminated as a complete process group after that window; its completed
    numerical result remains valid and no grandchildren leak into the next run.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if completion_marker is not None:
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.unlink(missing_ok=True)

    started = time.perf_counter()
    proc = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=merged_env,
        start_new_session=True,
    )
    marker_seen = False
    forced = False
    deadline = started + timeout

    while True:
        returncode = proc.poll()
        if returncode is not None:
            break
        if completion_marker is not None and completion_marker.exists():
            marker_seen = True
            try:
                returncode = proc.wait(timeout=teardown_grace)
            except subprocess.TimeoutExpired:
                forced = True
                _terminate_group(proc, grace=teardown_grace)
                returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
            break
        if time.perf_counter() >= deadline:
            _terminate_group(proc, grace=teardown_grace)
            raise BenchmarkTimeoutError(
                f"benchmark component timed out after {timeout:.1f}s: {' '.join(command)}"
            )
        time.sleep(0.05)

    elapsed = time.perf_counter() - started
    if completion_marker is not None and completion_marker.exists():
        marker_seen = True

    # A forced teardown after a completion marker is a successful numerical run;
    # the process was killed only after the result had been durably written.
    completed_successfully = marker_seen if completion_marker is not None else returncode == 0
    if check and not completed_successfully:
        raise BenchmarkProcessError(
            f"benchmark component exited {returncode} before completion: {' '.join(command)}"
        )
    if check and completion_marker is None and returncode != 0:
        raise BenchmarkProcessError(
            f"benchmark component exited {returncode}: {' '.join(command)}"
        )
    return ProcessResult(
        tuple(command), int(returncode), float(elapsed), marker_seen, forced
    )


def run_python_module(
    module: str,
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    completion_marker: Path | None = None,
    teardown_grace: float = 3.0,
) -> ProcessResult:
    return run_command(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        timeout=timeout,
        env=env,
        check=check,
        completion_marker=completion_marker,
        teardown_grace=teardown_grace,
    )
