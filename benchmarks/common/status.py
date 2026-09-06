from __future__ import annotations

from typing import Final

SUCCESS: Final = "success"
OPTIMIZER_FAILURE: Final = "optimizer_failure"
CERTIFICATION_FAILURE: Final = "certification_failure"
TIMEOUT: Final = "timeout"
NUMERICAL_FAILURE: Final = "numerical_failure"
INTERNAL_CAP: Final = "internal_cap"
NOT_REQUESTED: Final = "not_requested"

EXECUTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        SUCCESS,
        OPTIMIZER_FAILURE,
        CERTIFICATION_FAILURE,
        TIMEOUT,
        NUMERICAL_FAILURE,
        INTERNAL_CAP,
        NOT_REQUESTED,
    }
)


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "internal cap" in text or "internal_cap" in text or "internalcap" in text:
        return INTERNAL_CAP
    if "slsqp" in text or "optimizer" in text or "optimization" in text:
        return OPTIMIZER_FAILURE
    return NUMERICAL_FAILURE
