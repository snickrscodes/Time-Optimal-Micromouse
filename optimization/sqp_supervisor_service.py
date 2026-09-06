"""External containment service for sparse-SQP worker creation.

The route-planner process never calls :func:`os.posix_spawn` in service mode.
It connects to a prestarted Unix-domain service with bounded socket deadlines.
The service owns worker creation and may be independently restarted by an
application watchdog.  A service or spawn hang therefore cannot block the
planner beyond the client deadline; the planner retains its parent-owned
continuously certified checkpoint.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import pickle
import signal
import socket
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .sqp_supervisor import (
    SupervisedBatchOutcome,
    SupervisedBatchRunner,
    SupervisorSettings,
    _recv_message,
    _send_message,
)



class _PlannerDeadlineExpired(TimeoutError):
    """Raised by the planner-side POSIX wall-clock guard."""


class _PlannerWallClockDeadline:
    """Best-effort in-process guard around the entire client call.

    Socket timeouts already bound connect and receive operations.  On POSIX,
    the main thread additionally uses ``setitimer`` so request serialization
    and a blocked send cannot extend the planner-facing deadline.  Existing
    application alarms are never replaced; deployments that already own
    SIGALRM must retain an outer watchdog or service boundary.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self._armed = False
        self._previous_handler: Any = None

    @staticmethod
    def _expire(_signum: int, _frame: Any) -> None:
        raise _PlannerDeadlineExpired("external supervisor client deadline exceeded")

    def __enter__(self) -> "_PlannerWallClockDeadline":
        if (
            hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        ):
            current_delay, _ = signal.getitimer(signal.ITIMER_REAL)
            if current_delay <= 0.0:
                self._previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, self._expire)
                signal.setitimer(signal.ITIMER_REAL, self.seconds)
                self._armed = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self._previous_handler)


def _resolve(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def _handler_reference(handler: Callable[..., Any]) -> tuple[str, str]:
    module_name = getattr(handler, "__module__", None)
    qualname = getattr(handler, "__qualname__", None)
    if not module_name or not qualname or "<locals>" in qualname:
        raise TypeError("service handlers must be importable top-level callables")
    return str(module_name), str(qualname)


@dataclass(frozen=True, slots=True)
class SupervisorServiceClientSettings:
    socket_path: str
    connect_timeout_seconds: float = 1.0
    request_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        if not self.socket_path:
            raise ValueError("socket_path must be nonempty")
        values = (
            self.connect_timeout_seconds,
            self.request_timeout_seconds,
            self.poll_interval_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("service client time settings must be finite and positive")


class ExternalSupervisorServiceClient:
    """Bounded planner-side client for a prestarted supervisor service."""

    def __init__(self, settings: SupervisorServiceClientSettings) -> None:
        self.settings = settings

    @staticmethod
    def _safe_checkpoint(value: Any | None) -> Any | None:
        if value is None:
            return None
        safe = value.checkpoint_safe_copy() if hasattr(value, "checkpoint_safe_copy") else value
        pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)
        return safe

    def run(
        self,
        handler: Callable[[Any, Any], Any],
        payload: Any,
        *,
        worker_settings: SupervisorSettings,
        initial_checkpoint: Any | None = None,
        result_validator: Callable[[Any], bool] | None = None,
    ) -> SupervisedBatchOutcome:
        started = time.perf_counter()
        last_checkpoint: Any | None = None
        checkpoints = 0
        connection: socket.socket | None = None
        try:
            with _PlannerWallClockDeadline(self.settings.request_timeout_seconds):
                last_checkpoint = self._safe_checkpoint(initial_checkpoint)
                module_name, qualname = _handler_reference(handler)
                request = {
                    "kind": "run",
                    "handler_module": module_name,
                    "handler_qualname": qualname,
                    "payload": payload,
                    "worker_settings": worker_settings,
                    "initial_checkpoint": last_checkpoint,
                    "request_timeout_seconds": self.settings.request_timeout_seconds,
                }
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.settimeout(
                    min(self.settings.connect_timeout_seconds, self.settings.request_timeout_seconds)
                )
                connection.connect(self.settings.socket_path)
                _send_message(connection, request)
                deadline = time.monotonic() + self.settings.request_timeout_seconds
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    connection.settimeout(
                        max(0.001, min(remaining, self.settings.poll_interval_seconds))
                    )
                    try:
                        message = _recv_message(connection)
                    except socket.timeout:
                        continue
                    if not isinstance(message, dict):
                        return SupervisedBatchOutcome(
                            False, None, last_checkpoint, "malformed_result",
                            "supervisor service returned a non-dictionary frame", None,
                            None, False, time.perf_counter() - started, checkpoints,
                        )
                    kind = message.get("kind")
                    if kind == "checkpoint":
                        candidate = message.get("payload")
                        pickle.dumps(candidate, protocol=pickle.HIGHEST_PROTOCOL)
                        last_checkpoint = candidate
                        checkpoints += 1
                        continue
                    if kind == "outcome":
                        outcome = message.get("payload")
                        if not isinstance(outcome, SupervisedBatchOutcome):
                            return SupervisedBatchOutcome(
                                False, None, last_checkpoint, "malformed_result",
                                "supervisor service returned an invalid outcome", None,
                                None, False, time.perf_counter() - started, checkpoints,
                            )
                        result = outcome.result
                        if (
                            outcome.success
                            and result_validator is not None
                            and not result_validator(result)
                        ):
                            return SupervisedBatchOutcome(
                                False, None, last_checkpoint, "malformed_result",
                                "service result validation failed", outcome.last_stage,
                                outcome.worker_pid, outcome.worker_restarted,
                                time.perf_counter() - started, checkpoints,
                            )
                        return SupervisedBatchOutcome(
                            outcome.success,
                            result,
                            last_checkpoint
                            if last_checkpoint is not None
                            else outcome.last_checkpoint,
                            outcome.failure_reason,
                            outcome.message,
                            outcome.last_stage,
                            outcome.worker_pid,
                            outcome.worker_restarted,
                            time.perf_counter() - started,
                            max(checkpoints, outcome.checkpoints_received),
                        )
                    if kind == "service_error":
                        return SupervisedBatchOutcome(
                            False, None, last_checkpoint, "worker_error",
                            str(message.get("message", "supervisor service failed")),
                            "service", None, False, time.perf_counter() - started,
                            checkpoints,
                        )
                    return SupervisedBatchOutcome(
                        False, None, last_checkpoint, "malformed_result",
                        "supervisor service returned an unknown frame", None, None,
                        False, time.perf_counter() - started, checkpoints,
                    )
                return SupervisedBatchOutcome(
                    False, None, last_checkpoint, "startup_timeout",
                    "external supervisor service request deadline exceeded", "service",
                    None, False, time.perf_counter() - started, checkpoints,
                )
        except _PlannerDeadlineExpired as exc:
            return SupervisedBatchOutcome(
                False, None, last_checkpoint, "startup_timeout", str(exc),
                "service_client_deadline", None, False,
                time.perf_counter() - started, checkpoints,
            )
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as exc:
            return SupervisedBatchOutcome(
                False, None, last_checkpoint, "startup_timeout",
                f"external supervisor service unavailable: {exc}", "service_connect",
                None, False, time.perf_counter() - started, checkpoints,
            )
        except (pickle.PickleError, TypeError, ValueError) as exc:
            return SupervisedBatchOutcome(
                False, None, last_checkpoint, "serialization_error", str(exc),
                "service_request", None, False, time.perf_counter() - started,
                checkpoints,
            )
        finally:
            if connection is not None:
                connection.close()


def _serve_connection(connection: socket.socket) -> None:
    request = _recv_message(connection)
    if not isinstance(request, dict) or request.get("kind") != "run":
        _send_message(connection, {"kind": "service_error", "message": "invalid request"})
        return
    handler = _resolve(str(request["handler_module"]), str(request["handler_qualname"]))
    settings = request.get("worker_settings")
    if not isinstance(settings, SupervisorSettings):
        raise TypeError("worker_settings must be SupervisorSettings")
    request_timeout = float(request.get("request_timeout_seconds", settings.hard_timeout_seconds))
    if not math.isfinite(request_timeout) or request_timeout <= 0.0:
        raise ValueError("request_timeout_seconds must be finite and positive")
    # The external service is a total planner-call containment boundary.  A
    # worker checkpoint is still forwarded, but it cannot extend this total
    # request deadline.  This prevents an abandoned client from leaving a
    # separate-session worker alive indefinitely.
    settings = replace(
        settings,
        hard_timeout_seconds=request_timeout,
        deadline_refresh_on_checkpoint=False,
    )
    with SupervisedBatchRunner(handler, settings=settings) as runner:
        outcome = runner.run(
            request.get("payload"),
            initial_checkpoint=request.get("initial_checkpoint"),
            checkpoint_callback=lambda value: _send_message(
                connection, {"kind": "checkpoint", "payload": value}
            ),
        )
    _send_message(connection, {"kind": "outcome", "payload": outcome})


def serve(socket_path: str, *, once: bool = False) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(8)
    os.chmod(path, 0o600)
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        server.close()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while not stop:
            try:
                connection, _ = server.accept()
            except OSError:
                if stop:
                    break
                raise
            with connection:
                try:
                    _serve_connection(connection)
                except BaseException as exc:
                    try:
                        _send_message(connection, {"kind": "service_error", "message": f"{type(exc).__name__}: {exc}"})
                    except BaseException:
                        pass
            if once:
                break
    finally:
        try:
            server.close()
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    serve(args.socket, once=args.once)


if __name__ == "__main__":
    main()
