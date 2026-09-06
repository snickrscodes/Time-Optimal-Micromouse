from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .io import sha256_file

NATIVE_ARTIFACTS = (
    Path("native/cflow/libcflow.so"),
    Path("native/reverse_eta/libreverse_eta.so"),
    Path("native/segment/libame_segment.so"),
    Path("native/crossing/libame_crossing.so"),
    Path("native/reverse/libame_reverse.so"),
)


class NativeBuildError(RuntimeError):
    pass


def build_native(root: Path) -> None:
    """Invoke the repository's canonical native build procedure."""
    subprocess.run(["make", "native"], cwd=root, check=True)


def native_artifact_metadata(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in NATIVE_ARTIFACTS:
        path = root / relative
        rows.append(
            {
                "path": relative.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def ensure_native_available(root: Path) -> None:
    missing = [row["path"] for row in native_artifact_metadata(root) if not row["exists"]]
    if missing:
        joined = ", ".join(missing)
        raise NativeBuildError(
            "native benchmark artifacts are missing: " + joined + ". Run `make native` "
            "or pass `--build-native` to the benchmark runner."
        )
