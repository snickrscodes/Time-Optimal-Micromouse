"""Research-only Python binding for native C Segment v1.

This package is intentionally not wired into :mod:`segment`, crossings, or the
speed-profile solver.  It exists only for numerical differential testing and
microbenchmarking of ``native/segment``.
"""
from ._binding import (
    NativeSegment,
    NativeDomainProbe,
    NativeSegmentError,
    compile_segment_native,
)
__all__ = ["NativeSegment", "NativeDomainProbe", "NativeSegmentError", "compile_segment_native"]
