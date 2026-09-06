"""Low-level standard-library-only ctypes binding for the vendored Cflow kernel.

The shared library is built explicitly with the repository Makefile.  Importing
this module never invokes a compiler.
"""
from __future__ import annotations

import ctypes as _ct
import os
import platform
from pathlib import Path


class Status:
    OK = 0
    EVENT = 1
    BEYOND_EVENT = 2
    OUTSIDE_REAL_DOMAIN = 3
    CONDITIONING_LIMIT = 4
    NUMERICAL_FAILURE = 5
    NO_EVENT_WITHIN_HORIZON = 6
    INVALID_ARGUMENT = 7
    OUTSIDE_INTEGRAL_DOMAIN = 8


_STATUS_NAMES = {
    0: "ok",
    1: "event",
    2: "beyond_event",
    3: "outside_real_domain",
    4: "conditioning_limit",
    5: "numerical_failure",
    6: "no_event_within_horizon",
    7: "invalid_argument",
    8: "outside_integral_domain",
}


def status_name(status: int) -> str:
    return _STATUS_NAMES.get(int(status), "unknown")


class Eval(_ct.Structure):
    _fields_ = [
        ("x", _ct.c_double),
        ("event_time", _ct.c_double),
        ("steps", _ct.c_size_t),
        ("status", _ct.c_int),
    ]


class EvalJac(_ct.Structure):
    _fields_ = [
        ("x", _ct.c_double),
        ("dx_dx0", _ct.c_double),
        ("dx_dq0", _ct.c_double),
        ("dx_db", _ct.c_double),
        ("dx_dh", _ct.c_double),
        ("event_time", _ct.c_double),
        ("steps", _ct.c_size_t),
        ("status", _ct.c_int),
    ]


class IntegralValue(_ct.Structure):
    _fields_ = [
        ("integral", _ct.c_double),
        ("event_time", _ct.c_double),
        ("steps", _ct.c_size_t),
        ("status", _ct.c_int),
    ]


class IntegralJac(_ct.Structure):
    _fields_ = [
        ("integral", _ct.c_double),
        ("dI_dx0", _ct.c_double),
        ("dI_dq0", _ct.c_double),
        ("dI_db", _ct.c_double),
        ("dI_dh", _ct.c_double),
        ("regularized_dI_dx0", _ct.c_double),
        ("event_time", _ct.c_double),
        ("steps", _ct.c_size_t),
        ("status", _ct.c_int),
    ]


class All(_ct.Structure):
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
        ("event_time", _ct.c_double),
        ("steps", _ct.c_size_t),
        ("status", _ct.c_int),
    ]


class Event(_ct.Structure):
    _fields_ = [
        ("time", _ct.c_double),
        ("x", _ct.c_double),
        ("dt_dx0", _ct.c_double),
        ("dt_dq0", _ct.c_double),
        ("dt_db", _ct.c_double),
        ("conditioning_log_amp", _ct.c_double),
        ("steps", _ct.c_size_t),
        ("status", _ct.c_int),
    ]


def _library_path() -> Path:
    override = os.environ.get("AME_ROBOT_CFLOW_LIB") or os.environ.get("CFLOW_LIB")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    system = platform.system()
    name = "libcflow.dylib" if system == "Darwin" else ("cflow.dll" if system == "Windows" else "libcflow.so")
    return root / "native" / "cflow" / name


def _load() -> _ct.CDLL:
    path = _library_path()
    if not path.exists():
        raise ImportError(
            f"Cflow native library not found at {path}. Build it explicitly with `make` "
            "or `python3 build_native.py`; ordinary imports never invoke a compiler."
        )
    lib = _ct.CDLL(str(path))
    p_eval = _ct.POINTER(Eval)
    p_jac = _ct.POINTER(EvalJac)
    p_int = _ct.POINTER(IntegralJac)
    p_int_value = _ct.POINTER(IntegralValue)
    p_all = _ct.POINTER(All)
    p_event = _ct.POINTER(Event)
    for name, last in (
        ("cflow_eval", p_eval),
        ("cflow_boundary_eval", p_eval),
        ("cflow_eval_jacobian", p_jac),
        ("cflow_boundary_eval_jacobian", p_jac),
        ("cflow_integral_jacobian", p_int),
        ("cflow_boundary_integral_jacobian", p_int),
        ("cflow_integral_value", p_int_value),
        ("cflow_boundary_integral_value", p_int_value),
        ("cflow_eval_all", p_all),
        ("cflow_boundary_eval_all", p_all),
    ):
        fn = getattr(lib, name)
        fn.argtypes = [_ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, last]
        fn.restype = None
    lib.cflow_first_event.argtypes = [_ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, p_event]
    lib.cflow_first_event.restype = None
    return lib


lib = _load()


def eval_raw(x: float, q: float, b: float, h: float, *, boundary: bool = False) -> Eval:
    out = Eval()
    (lib.cflow_boundary_eval if boundary else lib.cflow_eval)(x, q, b, h, _ct.byref(out))
    return out


def eval_jac_raw(x: float, q: float, b: float, h: float, *, boundary: bool = False) -> EvalJac:
    out = EvalJac()
    (lib.cflow_boundary_eval_jacobian if boundary else lib.cflow_eval_jacobian)(x, q, b, h, _ct.byref(out))
    return out


def integral_jac_raw(x: float, q: float, b: float, h: float, *, boundary: bool = False) -> IntegralJac:
    out = IntegralJac()
    (lib.cflow_boundary_integral_jacobian if boundary else lib.cflow_integral_jacobian)(x, q, b, h, _ct.byref(out))
    return out


def integral_value_raw(x: float, q: float, b: float, h: float, *, boundary: bool = False) -> IntegralValue:
    """Return only the additive integral value, without sensitivity evolution."""
    out = IntegralValue()
    (lib.cflow_boundary_integral_value if boundary else lib.cflow_integral_value)(x, q, b, h, _ct.byref(out))
    return out


def all_raw(x: float, q: float, b: float, h: float, *, boundary: bool = False) -> All:
    out = All()
    (lib.cflow_boundary_eval_all if boundary else lib.cflow_eval_all)(x, q, b, h, _ct.byref(out))
    return out


def first_event_raw(x: float, q: float, b: float, max_time: float) -> Event:
    out = Event()
    lib.cflow_first_event(x, q, b, max_time, _ct.byref(out))
    return out

