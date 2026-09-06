from __future__ import annotations
import ctypes as ct
import math
from dataclasses import dataclass
from pathlib import Path
from csegment import NativeSegment

class _Options(ct.Structure):
    _fields_=[
        ("n_scan",ct.c_int),("domain_margin",ct.c_double),("domain_stop_margin",ct.c_double),
        ("physical_domain_margin",ct.c_double),("domain_safe_floor",ct.c_double),
        ("x_abs_tol",ct.c_double),("x_rel_tol",ct.c_double),("f_tol",ct.c_double),
        ("max_iter",ct.c_int),("allow_initial_boundary",ct.c_int),
        ("has_initial_spatial_tol",ct.c_int),("initial_spatial_tol",ct.c_double),
    ]
class _Bracket(ct.Structure):
    _fields_=[("present",ct.c_int),("lo",ct.c_double),("hi",ct.c_double),("f_lo",ct.c_double),("f_hi",ct.c_double)]
class _Result(ct.Structure):
    _fields_=[
        ("has_event",ct.c_int),("event",ct.c_double),("has_domain_edge",ct.c_int),("domain_edge",ct.c_double),
        ("event_bracket",_Bracket),("domain_bracket",_Bracket),("initial_switch",ct.c_int),
        ("has_domain_safe",ct.c_int),("domain_safe",ct.c_double),("has_domain_event",ct.c_int),("domain_event",ct.c_double),
        ("domain_switch_excluded",ct.c_int),("state_evaluations",ct.c_uint64),
    ]
class _Geometry(ct.Structure):
    _fields_=[("M",ct.c_double),("A",ct.c_double),("B",ct.c_double),("v_min",ct.c_double),("v_eq",ct.c_double),
              ("v_R",ct.c_double),("R_max",ct.c_double),("v_H",ct.c_double),("H_max",ct.c_double),("has_v_H",ct.c_int),
              ("orientation_certified",ct.c_int),("rising_barrier_certified",ct.c_int)]

@dataclass(frozen=True, slots=True)
class NativeScanResult:
    event: float|None
    domain_edge: float|None
    event_bracket: tuple[float,float,float,float]|None
    domain_bracket: tuple[float,float,float,float]|None
    initial_switch: bool
    domain_safe: float|None
    domain_event: float|None
    domain_switch_excluded: bool
    state_evaluations: int

class NativeCrossingError(RuntimeError):
    pass

def _load():
    path=Path(__file__).resolve().parents[1]/"native"/"crossing"/"libame_crossing.so"
    if not path.exists(): raise ImportError(f"native C++ Crossing v1 library not built: {path}; run `make crossing-native-v1`")
    lib=ct.CDLL(str(path)); P=ct.c_void_p
    lib.ame_crossing_default_options.restype=_Options
    lib.ame_crossing_scan.argtypes=[P,ct.c_double,ct.c_int,ct.c_int,ct.POINTER(_Options),ct.POINTER(_Result)];lib.ame_crossing_scan.restype=ct.c_int
    lib.ame_crossing_motor_geometry.argtypes=[ct.POINTER(_Geometry)];lib.ame_crossing_motor_geometry.restype=ct.c_int
    lib.ame_crossing_status_name.argtypes=[ct.c_int];lib.ame_crossing_status_name.restype=ct.c_char_p
    return lib
lib=_load()
_KIND={"motor_grip":1,"grip_motor":2,"grip_brake":3,"brake_grip":4}

def _status_name(s):
    p=lib.ame_crossing_status_name(int(s)); return p.decode() if p else f"status_{s}"
def _bracket(b): return (b.lo,b.hi,b.f_lo,b.f_hi) if b.present else None
def _to_result(r):
    return NativeScanResult(
        r.event if r.has_event else None, r.domain_edge if r.has_domain_edge else None,
        _bracket(r.event_bracket),_bracket(r.domain_bracket),bool(r.initial_switch),
        r.domain_safe if r.has_domain_safe else None,r.domain_event if r.has_domain_event else None,
        bool(r.domain_switch_excluded),int(r.state_evaluations),
    )

def _scan(seg:NativeSegment,kind:str,L=None,*,earliest_safe=False,**kwargs):
    if not isinstance(seg,NativeSegment) or not seg._ptr: raise TypeError("native crossing requires a live NativeSegment")
    o=lib.ame_crossing_default_options()
    aliases={"domain_margin":"domain_margin","domain_stop_margin":"domain_stop_margin","physical_domain_margin":"physical_domain_margin",
             "domain_safe_floor":"domain_safe_floor","x_abs_tol":"x_abs_tol","x_rel_tol":"x_rel_tol","f_tol":"f_tol","max_iter":"max_iter",
             "n_scan":"n_scan","allow_initial_boundary":"allow_initial_boundary"}
    for k,v in list(kwargs.items()):
        if k=="initial_spatial_tol":
            if v is not None: o.has_initial_spatial_tol=1;o.initial_spatial_tol=float(v)
            kwargs.pop(k);continue
        if k not in aliases: continue
        setattr(o,aliases[k],v);kwargs.pop(k)
    if kwargs: raise TypeError(f"unsupported native crossing options: {', '.join(sorted(kwargs))}")
    r=_Result(); status=lib.ame_crossing_scan(seg._ptr,seg.L if L is None else float(L),_KIND[kind],bool(earliest_safe),ct.byref(o),ct.byref(r))
    if status: raise NativeCrossingError(f"native crossing {kind} failed: {_status_name(status)}")
    return _to_result(r)

def motor_grip_scan_native(seg,L=None,**kw): return _scan(seg,"motor_grip",L,earliest_safe=False,**kw)
def grip_motor_scan_native(seg,L=None,*,earliest_safe=False,**kw): return _scan(seg,"grip_motor",L,earliest_safe=earliest_safe,**kw)
def grip_brake_scan_native(seg,L=None,*,earliest_safe=False,**kw): return _scan(seg,"grip_brake",L,earliest_safe=earliest_safe,**kw)
def brake_grip_scan_native(seg,L=None,**kw): return _scan(seg,"brake_grip",L,earliest_safe=False,**kw)
def motor_geometry_native():
    g=_Geometry(); status=lib.ame_crossing_motor_geometry(ct.byref(g))
    if status: raise NativeCrossingError(_status_name(status))
    return {name:getattr(g,name) for name,_ in g._fields_}
