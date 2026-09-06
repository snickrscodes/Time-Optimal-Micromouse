"""Fresh-interpreter worker for :mod:`optimization.sqp_supervisor`."""
from __future__ import annotations

import argparse
import importlib
import os
import pickle
import socket
import time
import traceback
from typing import Any

from .sqp_supervisor import WorkerContext, _recv_message, _send_message


def _resolve(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--handler-module", required=True)
    parser.add_argument("--handler-qualname", required=True)
    parser.add_argument(
        "--injection",
        choices=("none", "startup_hang", "malformed_ready", "shutdown_hang"),
        default="none",
    )
    args = parser.parse_args()
    if args.injection == "startup_hang":
        time.sleep(3600.0)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(args.socket)
    if args.injection == "malformed_ready":
        _send_message(connection, {"kind": "not_ready", "pid": os.getpid()})
        return
    _send_message(connection, {"kind": "ready", "pid": os.getpid()})
    handler = _resolve(args.handler_module, args.handler_qualname)
    while True:
        try:
            request = _recv_message(connection)
        except EOFError:
            return
        if not isinstance(request, dict):
            return
        kind = request.get("kind")
        if kind == "stop":
            if args.injection == "shutdown_hang":
                time.sleep(3600.0)
            return
        if kind != "run":
            continue
        request_id = int(request.get("request_id", -1))
        context = WorkerContext(connection, request_id)
        phase = "handler"
        try:
            result = handler(request.get("payload"), context)
            phase = "serialization"
            safe_result = (
                result.checkpoint_safe_copy()
                if hasattr(result, "checkpoint_safe_copy")
                else result
            )
            pickle.dumps(safe_result, protocol=pickle.HIGHEST_PROTOCOL)
            _send_message(
                connection,
                {"kind": "result", "request_id": request_id, "payload": safe_result},
            )
        except BaseException as exc:
            try:
                _send_message(
                    connection,
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "failure_reason": (
                            "serialization_error" if phase == "serialization" else "worker_error"
                        ),
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            except BaseException:
                return


if __name__ == "__main__":
    main()
