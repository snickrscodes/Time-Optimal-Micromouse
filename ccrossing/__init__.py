"""Research-only Python binding for native C++ Crossing v1.

This package is intentionally not imported by :mod:`optimization.crossings` or
by the speed-profile solver.  It exists only for differential qualification and
benchmarking against the current Python crossing authority.
"""
from ._binding import (
    NativeCrossingError,
    NativeScanResult,
    motor_grip_scan_native,
    grip_motor_scan_native,
    grip_brake_scan_native,
    brake_grip_scan_native,
    motor_geometry_native,
)
__all__ = [
    "NativeCrossingError", "NativeScanResult", "motor_grip_scan_native",
    "grip_motor_scan_native", "grip_brake_scan_native", "brake_grip_scan_native",
    "motor_geometry_native",
]
