from __future__ import annotations

from typing import Any, Mapping

from . import SCHEMA_VERSION
from .common.status import EXECUTION_STATUSES

VALID_STATUSES = {"complete", "partial", "failed", "skipped"}
VALID_NUMERICAL_PROFILES = {"core", "smoke"}


class ResultSchemaError(ValueError):
    pass


def _validate_execution_statuses(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        if "execution_status" in value and value["execution_status"] not in EXECUTION_STATUSES:
            raise ResultSchemaError(
                f"invalid execution_status={value['execution_status']!r} at {path}"
            )
        for key, child in value.items():
            _validate_execution_statuses(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_execution_statuses(child, f"{path}[{index}]")


def validate_benchmark_result(
    payload: Mapping[str, Any],
    *,
    expected_benchmark: str | None = None,
    expected_profile: str | None = None,
) -> None:
    required = {"schema_version", "benchmark", "status", "profile", "environment", "aggregate"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ResultSchemaError(f"missing benchmark result fields: {', '.join(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ResultSchemaError(
            f"unsupported schema_version={payload['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    if not isinstance(payload["benchmark"], str) or not payload["benchmark"]:
        raise ResultSchemaError("benchmark must be a nonempty string")
    if expected_benchmark is not None and payload["benchmark"] != expected_benchmark:
        raise ResultSchemaError(
            f"benchmark={payload['benchmark']!r}; expected {expected_benchmark!r}"
        )
    if payload["status"] not in VALID_STATUSES:
        raise ResultSchemaError(f"invalid status={payload['status']!r}")
    if payload["profile"] not in VALID_NUMERICAL_PROFILES:
        raise ResultSchemaError(f"invalid numerical profile={payload['profile']!r}")
    if expected_profile is not None and payload["profile"] != expected_profile:
        raise ResultSchemaError(
            f"profile={payload['profile']!r}; expected {expected_profile!r}"
        )
    if not isinstance(payload["environment"], Mapping):
        raise ResultSchemaError("environment must be an object")
    if not isinstance(payload["aggregate"], Mapping):
        raise ResultSchemaError("aggregate must be an object")
    provenance = payload["environment"].get("provenance")
    if not isinstance(provenance, Mapping):
        raise ResultSchemaError("environment.provenance must be present")
    if not provenance.get("git_commit") and not provenance.get("source_tree_sha256"):
        raise ResultSchemaError("provenance needs git_commit or source_tree_sha256")
    _validate_execution_statuses(payload)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "benchmark", "status", "profile", "numerical_profile",
        "environment", "resolved_suites", "runs",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ResultSchemaError(f"missing manifest fields: {', '.join(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ResultSchemaError("manifest schema version mismatch")
    if payload["benchmark"] != "manifest":
        raise ResultSchemaError("manifest benchmark field must equal 'manifest'")
    if payload["status"] not in VALID_STATUSES:
        raise ResultSchemaError("manifest status invalid")
    if payload["numerical_profile"] not in VALID_NUMERICAL_PROFILES:
        raise ResultSchemaError("manifest numerical_profile invalid")
    if not isinstance(payload["runs"], list):
        raise ResultSchemaError("manifest runs must be a list")
