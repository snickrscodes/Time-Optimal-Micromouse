from __future__ import annotations

import statistics
import time
from typing import Any, Callable


def median_timed(
    fn: Callable[[], Any],
    *,
    repeats: int,
    warmup: int = 1,
) -> tuple[float, Any]:
    """Warm a callable, then return median ``perf_counter`` runtime and last result."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >=1 and warmup must be >=0")
    result = None
    for _ in range(warmup):
        result = fn()
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - started)
    return float(statistics.median(timings)), result
