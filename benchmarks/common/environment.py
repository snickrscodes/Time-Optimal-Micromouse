from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy

from .provenance import provenance_metadata
from .native import native_artifact_metadata


def _command_first_line(args: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True, timeout=5)
        text = (proc.stdout or proc.stderr).strip().splitlines()
        return text[0] if text else None
    except Exception:
        return None


def _cpu_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _casadi_metadata() -> dict[str, Any]:
    # Query package metadata without importing CasADi. Importing the plugin stack
    # in a long-lived orchestration parent can perturb later solver lifecycles.
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            casadi_version = version("casadi")
        except PackageNotFoundError:
            casadi_version = None
        return {"casadi": casadi_version, "ipopt_plugin": "not_probed" if casadi_version else "unavailable"}
    except Exception:
        return {"casadi": None, "ipopt_plugin": "unknown"}



def casadi_ipopt_runtime_metadata() -> dict[str, Any]:
    """Probe CasADi/IPOPT from inside an isolated numerical/assembler process."""
    try:
        import casadi as ca
        package_dir = Path(ca.__file__).resolve().parent
        versions = sorted(package_dir.glob("libipopt.so.*.*.*"))
        ipopt_version = None
        if versions:
            name = versions[-1].name
            prefix = "libipopt.so."
            if name.startswith(prefix):
                ipopt_version = name[len(prefix):]
        return {
            "casadi_version": ca.__version__,
            "ipopt_version": ipopt_version,
            "casadi_compiler": ca.CasadiMeta.compiler(),
            "nlpsol_ipopt_available": "Nlpsol::ipopt" in ca.CasadiMeta.plugins(),
        }
    except Exception as exc:
        return {
            "casadi_version": None,
            "ipopt_version": None,
            "casadi_compiler": None,
            "nlpsol_ipopt_available": False,
            "casadi_probe_error": f"{type(exc).__name__}: {exc}",
        }

def environment_metadata(
    config: Any | None = None,
    *,
    root: Path | None = None,
    include_reverse_backend: bool = True,
) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    thread_env = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        if os.environ.get(key) is not None
    }
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": _cpu_name(),
            "logical_count": os.cpu_count(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "compilers": {
            "cc": _command_first_line([os.environ.get("CC", "cc"), "--version"]),
            "cxx": _command_first_line([os.environ.get("CXX", "c++"), "--version"]),
        },
        "active_reverse_backend": None,
        "thread_environment": thread_env,
        "optimization_config": None if config is None else config.to_dict(),
        "provenance": provenance_metadata(root),
        "native_artifacts": native_artifact_metadata(root),
    }
    if include_reverse_backend:
        try:
            from optimization import reverse_solver
            metadata["active_reverse_backend"] = reverse_solver.reverse_backend()
        except Exception:
            metadata["active_reverse_backend"] = "unavailable"
    metadata.update(_casadi_metadata())
    return metadata
