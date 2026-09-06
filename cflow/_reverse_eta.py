"""Internal binding for the reverse-only regularized Cflow tangent cocycle.

This module is intentionally *not* part of the public Cflow API.  The native
kernel is used only by the fixed-topology reverse time replay after the scalar
speed-profile build has already established topology/status authority.
"""
from __future__ import annotations

import ctypes as _ct
import math
import os
import platform
from pathlib import Path


RTOL = 1.0e-12
# Production dispatch is intentionally stricter than the exploratory selector.
# A broad independent stress found that the old lower-N/r0 tiers could admit
# unexpectedly cheap face+Tucker calls.  N_asym >= 3000 had zero measured
# slowdowns in both the original independent replay and the widened Q stress,
# with a >=3x raw-kernel margin in the latter.
_N_ASYM_MIN = 3000.0
# The numerical qualification corpus covered eta0 through 30.  This check is
# evaluated only after the cheap N_asym screen passes.
_ETA0_MAX = 30.0
_DISABLED = os.environ.get("AME_ROBOT_DISABLE_REVERSE_ETA") == "1"



class ReverseEtaResult(_ct.Structure):
    _fields_ = [
        ("x", _ct.c_double),
        ("dx_dx0", _ct.c_double),
        ("dx_dq0", _ct.c_double),
        ("dx_db", _ct.c_double),
        ("dx_dh", _ct.c_double),
        ("integral", _ct.c_double),
        ("dI_dx0", _ct.c_double),
        ("dI_dq0", _ct.c_double),
        ("dI_db", _ct.c_double),
        ("dI_dh", _ct.c_double),
        ("regularized_dI_dx0", _ct.c_double),
        ("accepted", _ct.c_int64),
        ("rejected", _ct.c_int64),
        ("one_steps", _ct.c_int64),
        ("newton_iters", _ct.c_int64),
        ("status", _ct.c_int),
    ]


def _library_path() -> Path:
    override = os.environ.get("AME_ROBOT_REVERSE_ETA_LIB")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    system = platform.system()
    name = (
        "libreverse_eta.dylib"
        if system == "Darwin"
        else ("reverse_eta.dll" if system == "Windows" else "libreverse_eta.so")
    )
    return root / "native" / "reverse_eta" / name


def _load() -> _ct.CDLL:
    path = _library_path()
    if not path.exists():
        raise ImportError(
            f"reverse-eta native library not found at {path}. Build it explicitly "
            "with `make` or `python3 build_native.py`; imports never invoke a compiler."
        )
    lib = _ct.CDLL(str(path))
    lib.ame_reverse_eta_all.argtypes = [
        _ct.c_double,
        _ct.c_double,
        _ct.c_double,
        _ct.c_double,
        _ct.c_double,
        _ct.POINTER(ReverseEtaResult),
    ]
    lib.ame_reverse_eta_all.restype = _ct.c_int
    return lib


lib = _load()


def selector_metrics(x: float, q: float, b: float, h: float) -> tuple[float, float] | None:
    """Return ``(N_asym, eta0)`` for the qualified contracting chart, else None.

    ``N_asym`` is evaluated as

        h (Q0 + Qh) (Q0^2 + Qh^2) / (0.0985 B)

    which is algebraically equal to the grazing-tail estimate based on the
    difference of fourth powers, without forming that cancellation directly.
    """
    if not all(math.isfinite(v) for v in (x, q, b, h)):
        return None
    if not (x > 0.0 and h > 0.0 and q != 0.0 and b != 0.0 and q * b < 0.0):
        return None
    qh = math.fma(b, h, q)
    if q * qh <= 0.0:
        return None
    Q0 = abs(q)
    Qh = abs(qh)
    B = abs(b)
    n_asym = h * (Q0 + Qh) * (Q0 * Q0 + Qh * Qh) / (0.0985 * B)
    if not (math.isfinite(n_asym) and n_asym >= _N_ASYM_MIN):
        return None

    z = Q0 * x
    if not (0.0 < z < 1.0):
        return None
    s = math.sqrt((1.0 - z) * (1.0 + z))
    eta0 = s / z
    if not (0.0 < eta0 <= _ETA0_MAX and math.isfinite(eta0)):
        return None
    return n_asym, eta0


def should_use_reverse_eta(x: float, q: float, b: float, h: float, *, boundary: bool = False) -> bool:
    if boundary or _DISABLED:
        return False
    return selector_metrics(x, q, b, h) is not None


def all_raw(x: float, q: float, b: float, h: float) -> ReverseEtaResult | None:
    out = ReverseEtaResult()
    ok = lib.ame_reverse_eta_all(x, q, b, h, RTOL, _ct.byref(out))
    if not ok or out.status != 0:
        return None
    return out


__all__ = [
    "RTOL",
    "ReverseEtaResult",
    "all_raw",
    "selector_metrics",
    "should_use_reverse_eta",
]
