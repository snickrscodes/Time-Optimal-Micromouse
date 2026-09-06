"""Strictly serial runtime controls for optimizer research campaigns.

This module deliberately separates *process isolation* from *parallelism*.
The current publication-candidate research baseline executes one route, one
basis branch, and one numerical worker at a time.  A child process exists only
so a pathological native/HiGHS call can be terminated without sacrificing the
already-certified parent incumbent.

Parallel route/branch scheduling is intentionally out of scope until the
mathematical optimizer architecture is frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class SerialWorkerResult:
    """Outcome of one synchronously supervised worker process."""

    status: str
    returncode: int | None
    wall_seconds: float
    completion_marker_seen: bool
    forced_teardown: bool
    timed_out: bool
    stdout_path: str | None
    stderr_path: str | None


def force_single_thread_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return/configure an environment with common numerical thread pools at 1.

    When ``env`` is omitted, ``os.environ`` itself is updated.  Call this before
    importing NumPy/SciPy in a new process for the setting to be authoritative.
    ``PYTHONHASHSEED`` is also fixed for reproducible dictionary/set iteration in
    worker subprocesses; it is not a numerical-performance setting.
    """
    target = os.environ if env is None else env
    for name in THREAD_ENVIRONMENT_VARIABLES:
        target[name] = "1"
    target["PYTHONHASHSEED"] = "0"
    return target


def thread_environment_snapshot(env: Mapping[str, str] | None = None) -> dict[str, str | None]:
    source = os.environ if env is None else env
    return {
        name: source.get(name)
        for name in (*THREAD_ENVIRONMENT_VARIABLES, "PYTHONHASHSEED")
    }


def _linux_parent_death_signal() -> None:
    """Ask Linux to terminate a worker if its supervising parent disappears.

    The worker otherwise starts a new session so the parent can kill its whole
    process group without touching itself.  ``PR_SET_PDEATHSIG`` closes the
    complementary failure mode: an externally killed supervisor must not leave
    an orphan optimizer consuming compute in the supposedly serial baseline.
    """
    if os.name != "posix" or not Path("/proc").exists():
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1
        parent_before = os.getppid()
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            return
        # Parent may have died between fork and prctl.  In that race Linux has
        # already re-parented us; terminate immediately rather than orphaning.
        if os.getppid() != parent_before:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        # Process supervision remains functional through explicit killpg even on
        # platforms without prctl; parent-death cleanup is an extra safeguard.
        return


def _terminate_process_group(proc: subprocess.Popen, *, grace_seconds: float) -> None:
    """Terminate one isolated worker process group; never touch the parent."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=max(0.01, float(grace_seconds)))
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
        proc.wait(timeout=max(0.01, float(grace_seconds)))
    except Exception:
        pass


def run_serial_worker(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    completion_marker: Path | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    teardown_grace_seconds: float = 1.0,
    poll_seconds: float = 0.05,
) -> SerialWorkerResult:
    """Run exactly one killable worker and synchronously wait for it.

    A completion marker is stronger than process exit: the worker writes it only
    after its durable numerical result has been closed.  If extension/plugin
    teardown hangs *after* that point, the parent may terminate the process group
    without invalidating the completed result.

    This function never starts a second worker and contains no thread pool.
    """
    timeout = float(timeout_seconds)
    if not timeout > 0.0:
        raise ValueError("timeout_seconds must be positive")
    grace = max(0.01, float(teardown_grace_seconds))
    poll = max(0.005, float(poll_seconds))
    merged_env = force_single_thread_environment(os.environ.copy())
    if env is not None:
        merged_env.update(env)
        force_single_thread_environment(merged_env)

    if completion_marker is not None:
        completion_marker = Path(completion_marker)
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.unlink(missing_ok=True)
    if stdout_path is not None:
        stdout_path = Path(stdout_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path = Path(stderr_path)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_handle = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.DEVNULL
    stderr_handle = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
    started = time.perf_counter()
    proc: subprocess.Popen | None = None
    marker_seen = False
    forced_teardown = False
    timed_out = False
    returncode: int | None = None
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
            preexec_fn=_linux_parent_death_signal if os.name == "posix" else None,
        )
        deadline = started + timeout
        while True:
            returncode = proc.poll()
            if returncode is not None:
                break
            if completion_marker is not None and completion_marker.exists():
                marker_seen = True
                try:
                    returncode = proc.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    forced_teardown = True
                    _terminate_process_group(proc, grace_seconds=grace)
                    returncode = proc.returncode
                break
            if time.perf_counter() >= deadline:
                timed_out = True
                _terminate_process_group(proc, grace_seconds=grace)
                returncode = proc.returncode
                break
            time.sleep(poll)
        if completion_marker is not None and completion_marker.exists():
            marker_seen = True
    finally:
        if stdout_path is not None:
            stdout_handle.close()
        if stderr_path is not None:
            stderr_handle.close()

    if timed_out:
        status = "timeout"
    elif completion_marker is not None:
        status = "complete" if marker_seen else "failed"
    else:
        status = "complete" if returncode == 0 else "failed"
    return SerialWorkerResult(
        status=status,
        returncode=returncode,
        wall_seconds=float(time.perf_counter() - started),
        completion_marker_seen=bool(marker_seen),
        forced_teardown=bool(forced_teardown),
        timed_out=bool(timed_out),
        stdout_path=None if stdout_path is None else str(stdout_path),
        stderr_path=None if stderr_path is None else str(stderr_path),
    )


__all__ = [
    "SerialWorkerResult",
    "THREAD_ENVIRONMENT_VARIABLES",
    "force_single_thread_environment",
    "run_serial_worker",
    "thread_environment_snapshot",
]
