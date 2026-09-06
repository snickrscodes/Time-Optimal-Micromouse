from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.config import IMPORTED_SOURCE_ARCHIVE_SHA256

_TREE_EXCLUDES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_TREE_SUFFIX_EXCLUDES = {".pyc", ".pyo", ".o", ".so", ".a"}


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        return value or None
    except Exception:
        return None


def source_tree_sha256(root: Path) -> str:
    """Deterministic fallback identifier when the checkout has no Git metadata."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if any(part in _TREE_EXCLUDES for part in rel.parts):
            continue
        if path.suffix in _TREE_SUFFIX_EXCLUDES:
            continue
        # Generated benchmark outputs are not part of source identity.
        if rel.as_posix() == "BENCHMARK_REPORT.md":
            continue
        if rel.parts[:2] in {("benchmarks", "results"), ("benchmarks", "plots")}:
            continue
        if rel.parts and rel.parts[0] == "benchmark_results":
            continue
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            continue
        digest.update(b"\0")
    return digest.hexdigest()


def provenance_metadata(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    if commit is not None:
        status = _git(root, "status", "--porcelain")
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        return {
            "source_kind": "git",
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": bool(status),
            "source_tree_sha256": None,
            "imported_source_archive_sha256": IMPORTED_SOURCE_ARCHIVE_SHA256,
        }
    return {
        "source_kind": "tree_sha256",
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
        "source_tree_sha256": source_tree_sha256(root),
        "imported_source_archive_sha256": IMPORTED_SOURCE_ARCHIVE_SHA256,
    }
