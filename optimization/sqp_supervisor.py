"""Externally bounded persistent subprocess supervision for sparse-SQP work."""
from __future__ import annotations

import importlib
import inspect
import math
import os
import pickle
import select
import shutil
import signal
import socket
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


FailureReason = Literal[
    "none",
    "startup_timeout",
    "objective_timeout",
    "solver_timeout",
    "hard_timeout",
    "worker_error",
    "worker_crash",
    "serialization_error",
    "malformed_result",
    "shutdown_timeout",
]

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 1 << 31


class _MalformedStartupHandshake(RuntimeError):
    """Worker connected but did not provide the required ready frame."""



def _send_message(connection: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("supervisor IPC payload is too large")
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("supervisor socket closed")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _recv_message(connection: socket.socket) -> Any:
    size = _HEADER.unpack(_recv_exact(connection, _HEADER.size))[0]
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError("supervisor IPC frame is too large")
    return pickle.loads(_recv_exact(connection, int(size)))


@dataclass(frozen=True, slots=True)
class SupervisorSettings:
    hard_timeout_seconds: float = 30.0
    worker_startup_timeout_seconds: float = 10.0
    poll_interval_seconds: float = 0.02
    terminate_grace_seconds: float = 0.25
    deadline_refresh_on_checkpoint: bool = False
    start_method: Literal["auto", "fork", "forkserver", "spawn", "posix_spawn"] = "posix_spawn"
    test_injection: Literal[
        "none", "startup_hang", "malformed_ready", "shutdown_hang"
    ] = "none"

    def __post_init__(self) -> None:
        values = (
            self.hard_timeout_seconds,
            self.poll_interval_seconds,
            self.worker_startup_timeout_seconds,
            self.terminate_grace_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("supervisor time settings must be finite and positive")
        if self.start_method not in {
            "auto", "fork", "forkserver", "spawn", "posix_spawn"
        }:
            raise ValueError("unsupported supervisor start method")


@dataclass(frozen=True, slots=True)
class SupervisedBatchOutcome:
    success: bool
    result: Any | None
    last_checkpoint: Any | None
    failure_reason: FailureReason
    message: str
    last_stage: str | None
    worker_pid: int | None
    worker_restarted: bool
    wall_seconds: float
    checkpoints_received: int


class WorkerContext:
    """Child-only event interface passed to a supervised handler."""

    def __init__(self, connection: socket.socket, request_id: int) -> None:
        self._connection = connection
        self._request_id = request_id

    def set_stage(self, stage: str) -> None:
        _send_message(
            self._connection,
            {"kind": "stage", "request_id": self._request_id, "stage": str(stage)},
        )

    def emit_checkpoint(self, checkpoint: Any) -> None:
        safe = (
            checkpoint.checkpoint_safe_copy()
            if hasattr(checkpoint, "checkpoint_safe_copy")
            else checkpoint
        )
        pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)
        _send_message(
            self._connection,
            {"kind": "checkpoint", "request_id": self._request_id, "payload": safe},
        )


class SupervisedBatchRunner:
    """Persistent worker launched without Python-after-fork bootstrap.

    Linux/Unix launches use ``os.posix_spawn`` to exec a fresh interpreter in a
    new session.  The parent owns the Unix-domain listener before launch and
    bounds the ready handshake, every request, checkpoint delivery, and process
    group teardown.  Logical recovery depends only on checkpoint-safe payloads.
    """

    def __init__(
        self,
        handler: Callable[[Any, WorkerContext], Any],
        *,
        settings: SupervisorSettings = SupervisorSettings(),
        result_validator: Callable[[Any], bool] | None = None,
    ) -> None:
        self._handler = handler
        self._settings = settings
        self._result_validator = result_validator
        self._pid: int | None = None
        self._connection: socket.socket | None = None
        self._server: socket.socket | None = None
        self._temporary_directory: str | None = None
        self._request_id = 0
        self._ever_started = False
        self._last_checkpoint: Any | None = None
        self._last_shutdown_forced = False

    @property
    def last_checkpoint(self) -> Any | None:
        return self._last_checkpoint

    @property
    def last_shutdown_forced(self) -> bool:
        return self._last_shutdown_forced

    @staticmethod
    def _handler_reference(handler: Callable[..., Any]) -> tuple[str, str, str | None]:
        module = inspect.getmodule(handler)
        module_name = getattr(handler, "__module__", None)
        qualname = getattr(handler, "__qualname__", None)
        if not module_name or not qualname or "<locals>" in qualname:
            raise TypeError("supervised handlers must be importable top-level callables")
        module_file = None if module is None else getattr(module, "__file__", None)
        return str(module_name), str(qualname), module_file

    def _alive(self) -> bool:
        if self._pid is None:
            return False
        try:
            waited, _ = os.waitpid(self._pid, os.WNOHANG)
        except ChildProcessError:
            return False
        return waited == 0

    def _spawn(self, socket_path: str) -> int:
        if os.name != "posix":
            raise RuntimeError("posix_spawn supervisor requires a POSIX platform")
        module_name, qualname, module_file = self._handler_reference(self._handler)
        worker_argv = [
            sys.executable,
            "-m",
            "optimization.sqp_supervisor_worker",
            "--socket",
            socket_path,
            "--handler-module",
            module_name,
            "--handler-qualname",
            qualname,
            "--injection",
            self._settings.test_injection,
        ]
        # Asking libc posix_spawn to create a new session may select a
        # fork-like implementation, reintroducing the native-library bootstrap
        # hazard this supervisor is intended to avoid.  The small util-linux
        # launcher performs setsid and then execs the fresh interpreter while
        # the parent retains the fast posix_spawn path.
        setsid_executable = shutil.which("setsid")
        if setsid_executable is not None:
            executable = setsid_executable
            argv = [setsid_executable, *worker_argv]
        else:
            executable = sys.executable
            argv = worker_argv
        environment = dict(os.environ)
        roots = [str(Path(__file__).resolve().parents[1])]
        if module_file:
            roots.append(str(Path(module_file).resolve().parent))
        # Payloads may contain importable classes defined outside the handler's
        # module (for example application objectives or pytest fixtures).  A
        # fresh interpreter needs the parent's explicit import roots to unpickle
        # them without relying on inherited process state.
        roots.extend(str(Path(entry).resolve()) for entry in sys.path if entry)
        existing = environment.get("PYTHONPATH")
        if existing:
            roots.extend(part for part in existing.split(os.pathsep) if part)
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(roots))
        if setsid_executable is not None:
            return int(os.posix_spawn(executable, argv, environment))
        try:
            return int(os.posix_spawn(executable, argv, environment, setpgroup=0))
        except TypeError:
            return int(os.posix_spawn(executable, argv, environment))

    def start(self) -> bool:
        if self._pid is not None and self._alive() and self._connection is not None:
            return False
        self._cleanup_resources()
        temporary = tempfile.mkdtemp(prefix="sparse-sqp-supervisor-")
        socket_path = os.path.join(temporary, "worker.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        server.settimeout(self._settings.worker_startup_timeout_seconds)
        self._temporary_directory = temporary
        self._server = server
        pid: int | None = None
        try:
            pid = self._spawn(socket_path)
            self._pid = pid
            connection, _ = server.accept()
            connection.settimeout(self._settings.worker_startup_timeout_seconds)
            ready = _recv_message(connection)
            if not isinstance(ready, dict) or ready.get("kind") != "ready":
                connection.close()
                raise _MalformedStartupHandshake("supervised worker sent an invalid startup handshake")
            self._connection = connection
        except socket.timeout as exc:
            self._terminate_tree()
            raise TimeoutError("supervised worker startup timed out") from exc
        except BaseException:
            self._terminate_tree()
            raise
        restarted = self._ever_started
        self._ever_started = True
        return restarted

    def _cleanup_resources(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        self._connection = None
        self._server = None
        if self._temporary_directory is not None:
            shutil.rmtree(self._temporary_directory, ignore_errors=True)
        self._temporary_directory = None

    def _terminate_tree(self, *, force_required: bool = False) -> None:
        pid = self._pid
        forced = bool(force_required)
        if pid is not None:
            if self._alive():
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + self._settings.terminate_grace_seconds
                while self._alive() and time.monotonic() < deadline:
                    time.sleep(min(0.01, self._settings.poll_interval_seconds))
            if self._alive():
                forced = True
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        self._last_shutdown_forced = forced
        self._pid = None
        self._cleanup_resources()

    @staticmethod
    def _timeout_reason(stage: str | None) -> FailureReason:
        normalized = "" if stage is None else stage.lower()
        if any(token in normalized for token in ("objective", "reverse", "gradient")):
            return "objective_timeout"
        if any(token in normalized for token in ("highs", "solver", "qp", "lp")):
            return "solver_timeout"
        return "hard_timeout"

    def run(
        self,
        payload: Any,
        *,
        hard_timeout_seconds: float | None = None,
        initial_checkpoint: Any | None = None,
        checkpoint_callback: Callable[[Any], None] | None = None,
    ) -> SupervisedBatchOutcome:
        started = time.perf_counter()
        if initial_checkpoint is not None:
            safe = (
                initial_checkpoint.checkpoint_safe_copy()
                if hasattr(initial_checkpoint, "checkpoint_safe_copy")
                else initial_checkpoint
            )
            pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)
            self._last_checkpoint = safe
        try:
            restarted = self.start()
        except TimeoutError as exc:
            return SupervisedBatchOutcome(
                False, None, self._last_checkpoint, "startup_timeout", str(exc),
                "worker_startup", self._pid, self._ever_started,
                time.perf_counter() - started, 0,
            )
        except _MalformedStartupHandshake as exc:
            return SupervisedBatchOutcome(
                False, None, self._last_checkpoint, "malformed_result", str(exc),
                "worker_startup", self._pid, self._ever_started,
                time.perf_counter() - started, 0,
            )
        except BaseException as exc:
            return SupervisedBatchOutcome(
                False, None, self._last_checkpoint, "worker_crash", str(exc),
                "worker_startup", self._pid, self._ever_started,
                time.perf_counter() - started, 0,
            )
        assert self._pid is not None and self._connection is not None
        connection = self._connection
        self._request_id += 1
        request_id = self._request_id
        worker_pid = self._pid
        timeout = self._settings.hard_timeout_seconds if hard_timeout_seconds is None else float(hard_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("hard_timeout_seconds must be finite and positive")
        try:
            _send_message(
                connection,
                {"kind": "run", "request_id": request_id, "payload": payload},
            )
        except BaseException as exc:
            self._terminate_tree()
            return SupervisedBatchOutcome(
                False, None, self._last_checkpoint, "serialization_error", str(exc),
                None, worker_pid, restarted, time.perf_counter() - started, 0,
            )

        deadline = time.monotonic() + timeout
        last_stage: str | None = None
        checkpoints = 0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            wait = min(self._settings.poll_interval_seconds, max(0.0, remaining))
            try:
                readable, _, _ = select.select([connection], [], [], wait)
            except (OSError, ValueError) as exc:
                self._terminate_tree()
                return SupervisedBatchOutcome(
                    False, None, self._last_checkpoint, "worker_crash",
                    f"worker connection became unavailable: {exc}", last_stage,
                    worker_pid, restarted, time.perf_counter() - started,
                    checkpoints,
                )
            if readable:
                try:
                    connection.settimeout(max(0.001, remaining))
                    message = _recv_message(connection)
                except (EOFError, OSError) as exc:
                    # Runtime socket closure means the worker disappeared before
                    # completing its framed response.  Treat it as a crash even
                    # when waitpid has not observed the exit yet.
                    self._terminate_tree()
                    return SupervisedBatchOutcome(
                        False, None, self._last_checkpoint, "worker_crash",
                        f"worker connection closed: {exc}", last_stage, worker_pid,
                        restarted, time.perf_counter() - started, checkpoints,
                    )
                except (ValueError, pickle.PickleError) as exc:
                    self._terminate_tree()
                    return SupervisedBatchOutcome(
                        False, None, self._last_checkpoint, "malformed_result",
                        f"worker IPC failed: {exc}", last_stage, worker_pid,
                        restarted, time.perf_counter() - started, checkpoints,
                    )
                if not isinstance(message, dict) or message.get("request_id") != request_id:
                    self._terminate_tree()
                    return SupervisedBatchOutcome(
                        False, None, self._last_checkpoint, "malformed_result",
                        "worker returned a malformed or mismatched response",
                        last_stage, worker_pid, restarted,
                        time.perf_counter() - started, checkpoints,
                    )
                kind = message.get("kind")
                if kind == "stage":
                    last_stage = str(message.get("stage"))
                    continue
                if kind == "checkpoint":
                    candidate = message.get("payload")
                    try:
                        pickle.dumps(candidate, protocol=pickle.HIGHEST_PROTOCOL)
                    except BaseException:
                        self._terminate_tree()
                        return SupervisedBatchOutcome(
                            False, None, self._last_checkpoint, "serialization_error",
                            "worker emitted an unserializable checkpoint", last_stage,
                            worker_pid, restarted, time.perf_counter() - started,
                            checkpoints,
                        )
                    self._last_checkpoint = candidate
                    checkpoints += 1
                    if checkpoint_callback is not None:
                        try:
                            checkpoint_callback(candidate)
                        except BaseException as exc:
                            self._terminate_tree()
                            return SupervisedBatchOutcome(
                                False, None, self._last_checkpoint, "worker_error",
                                f"checkpoint callback failed: {exc}", last_stage,
                                worker_pid, restarted, time.perf_counter() - started,
                                checkpoints,
                            )
                    if self._settings.deadline_refresh_on_checkpoint:
                        deadline = time.monotonic() + timeout
                    continue
                if kind == "error":
                    error_type = str(message.get("error_type"))
                    reason: FailureReason = (
                        "serialization_error"
                        if message.get("failure_reason") == "serialization_error"
                        else self._timeout_reason(last_stage)
                        if error_type.endswith("Timeout")
                        else "worker_error"
                    )
                    return SupervisedBatchOutcome(
                        False, None, self._last_checkpoint, reason,
                        f"{error_type}: {message.get('message')}", last_stage,
                        worker_pid, restarted, time.perf_counter() - started,
                        checkpoints,
                    )
                if kind == "result":
                    result = message.get("payload")
                    valid = True if self._result_validator is None else bool(self._result_validator(result))
                    if not valid:
                        return SupervisedBatchOutcome(
                            False, None, self._last_checkpoint, "malformed_result",
                            "worker result validation failed", last_stage, worker_pid,
                            restarted, time.perf_counter() - started, checkpoints,
                        )
                    return SupervisedBatchOutcome(
                        True, result, self._last_checkpoint, "none", "completed",
                        last_stage, worker_pid, restarted,
                        time.perf_counter() - started, checkpoints,
                    )
                self._terminate_tree()
                return SupervisedBatchOutcome(
                    False, None, self._last_checkpoint, "malformed_result",
                    "worker response omitted a recognized kind", last_stage,
                    worker_pid, restarted, time.perf_counter() - started,
                    checkpoints,
                )
            if not self._alive():
                self._terminate_tree()
                return SupervisedBatchOutcome(
                    False, None, self._last_checkpoint, "worker_crash",
                    "worker process crashed", last_stage, worker_pid, restarted,
                    time.perf_counter() - started, checkpoints,
                )

        reason = self._timeout_reason(last_stage)
        self._terminate_tree()
        return SupervisedBatchOutcome(
            False, None, self._last_checkpoint, reason,
            f"hard worker deadline exceeded during {last_stage or 'unknown stage'}",
            last_stage, worker_pid, restarted, time.perf_counter() - started,
            checkpoints,
        )

    def abort_now(self) -> None:
        """Issue an immediate process-group kill without blocking on reap.

        This is the last-resort parent-side deadline path for a worker trapped
        in an uninterruptible native call.  The process group receives SIGKILL,
        IPC resources are detached immediately, and a daemon reaper collects
        the direct child when the kernel makes it waitable.
        """
        pid = self._pid
        self._last_shutdown_forced = pid is not None
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            def reap() -> None:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
            import threading
            threading.Thread(target=reap, name=f"supervisor-reaper-{pid}", daemon=True).start()
        self._pid = None
        self._cleanup_resources()

    def close(self) -> None:
        self._last_shutdown_forced = False
        force_required = False
        if self._pid is not None and self._alive() and self._connection is not None:
            try:
                _send_message(self._connection, {"kind": "stop"})
                deadline = time.monotonic() + self._settings.terminate_grace_seconds
                while self._alive() and time.monotonic() < deadline:
                    time.sleep(min(0.01, self._settings.poll_interval_seconds))
                force_required = self._alive()
            except BaseException:
                force_required = True
        self._terminate_tree(force_required=force_required)

    def __enter__(self) -> "SupervisedBatchRunner":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "FailureReason",
    "SupervisedBatchOutcome",
    "SupervisedBatchRunner",
    "SupervisorSettings",
    "WorkerContext",
    "_recv_message",
    "_send_message",
]
