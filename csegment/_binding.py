from __future__ import annotations
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
from segment.base import SegmentType

class _Options(ct.Structure):
    _fields_=[("boundary_start",ct.c_int),("reverse_eta",ct.c_int),("has_authoritative_w1",ct.c_int),("authoritative_w1",ct.c_double),("time_w0_policy",ct.c_int)]
class _StateJac(ct.Structure):
    _fields_=[("w",ct.c_double),("jac",ct.c_double*8)]
class _TimeJac(ct.Structure):
    _fields_=[("time",ct.c_double),("jac",ct.c_double*4)]
class _AllJac(ct.Structure):
    _fields_=[("w",ct.c_double),("state_jac",ct.c_double*8),("time",ct.c_double),("time_jac",ct.c_double*4)]
class _Probe(ct.Structure):
    _fields_=[("pre_event",ct.c_int),("event_position",ct.c_double),("cflow_status",ct.c_int)]
class _Stats(ct.Structure):
    _fields_=[("stations",ct.c_size_t),("cache_hits",ct.c_uint64),("cflow_calls",ct.c_uint64),("local_steps",ct.c_uint64)]

@dataclass(frozen=True, slots=True)
class NativeDomainProbe:
    pre_event: bool
    event_position: float | None
    status: int

class NativeSegmentError(RuntimeError):
    def __init__(self,status:int,op:str):
        super().__init__(f"native segment {op} failed: {_status_name(status)}")
        self.status=int(status)

def _load():
    path=Path(__file__).resolve().parents[1]/"native"/"segment"/"libame_segment.so"
    if not path.exists():
        raise ImportError(f"native C Segment v1 library not built: {path}; run `make segment-native-v1`")
    lib=ct.CDLL(str(path))
    P=ct.c_void_p
    lib.ame_segment_default_options.restype=_Options
    lib.ame_segment_compile.argtypes=[ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_int,ct.POINTER(_Options),ct.POINTER(ct.c_int)];lib.ame_segment_compile.restype=P
    lib.ame_segment_destroy.argtypes=[P]
    for name in ("implementation","mode_of","is_differentiable"):
        fn=getattr(lib,"ame_segment_"+name);fn.argtypes=[P];fn.restype=ct.c_int
    for name in ("length","sigma","w0","k0"):
        fn=getattr(lib,"ame_segment_"+name);fn.argtypes=[P];fn.restype=ct.c_double
    lib.ame_segment_w.argtypes=[P,ct.c_double,ct.POINTER(ct.c_double)];lib.ame_segment_w.restype=ct.c_int
    lib.ame_segment_time.argtypes=[P,ct.c_double,ct.POINTER(ct.c_double)];lib.ame_segment_time.restype=ct.c_int
    lib.ame_segment_w_and_jac.argtypes=[P,ct.c_double,ct.POINTER(_StateJac)];lib.ame_segment_w_and_jac.restype=ct.c_int
    lib.ame_segment_time_and_jac.argtypes=[P,ct.c_double,ct.POINTER(_TimeJac)];lib.ame_segment_time_and_jac.restype=ct.c_int
    lib.ame_segment_state_time_and_jac.argtypes=[P,ct.c_double,ct.POINTER(_AllJac)];lib.ame_segment_state_time_and_jac.restype=ct.c_int
    lib.ame_segment_domain_probe_at.argtypes=[P,ct.c_double,ct.POINTER(_Probe)];lib.ame_segment_domain_probe_at.restype=ct.c_int
    lib.ame_segment_renormalized_time_w0.argtypes=[P,ct.c_double,ct.POINTER(ct.c_double)];lib.ame_segment_renormalized_time_w0.restype=ct.c_int
    lib.ame_segment_cache_stats_get.argtypes=[P,ct.POINTER(_Stats)];lib.ame_segment_cache_stats_get.restype=ct.c_int
    lib.ame_segment_status_name.argtypes=[ct.c_int];lib.ame_segment_status_name.restype=ct.c_char_p
    return lib
lib=_load()
def _status_name(s):
    p=lib.ame_segment_status_name(int(s));return p.decode() if p else f"status_{s}"
def _check(s,op):
    if s: raise NativeSegmentError(s,op)
_IMPL={1:"straight",2:"circular_stable",3:"circular",4:"grip_cflow",5:"motor_stable",6:"motor",7:"brake"}
_POLICY={"negative_infinity":0,"raise":1,"renormalized":2}
_MODE={SegmentType.GRIP:1,SegmentType.MOTOR:2,SegmentType.BRAKE:3}

class NativeSegment:
    __slots__=("_ptr",)
    def __init__(self,ptr): self._ptr=ptr
    def close(self):
        if self._ptr: lib.ame_segment_destroy(self._ptr);self._ptr=None
    def __del__(self):
        try:self.close()
        except Exception:pass
    @property
    def implementation(self): return _IMPL.get(lib.ame_segment_implementation(self._ptr),"unknown")
    @property
    def L(self): return lib.ame_segment_length(self._ptr)
    @property
    def sigma(self): return lib.ame_segment_sigma(self._ptr)
    @property
    def w0(self): return lib.ame_segment_w0(self._ptr)
    @property
    def k0(self): return lib.ame_segment_k0(self._ptr)
    @property
    def grad(self): return bool(lib.ame_segment_is_differentiable(self._ptr))
    def w(self,ds):
        x=ct.c_double();_check(lib.ame_segment_w(self._ptr,float(ds),ct.byref(x)),"w");return x.value
    def time(self,ds):
        x=ct.c_double();_check(lib.ame_segment_time(self._ptr,float(ds),ct.byref(x)),"time");return x.value
    def w_and_jac(self,ds):
        x=_StateJac();_check(lib.ame_segment_w_and_jac(self._ptr,float(ds),ct.byref(x)),"w_and_jac");return x.w,tuple(x.jac)
    def time_and_jac(self,ds):
        x=_TimeJac();_check(lib.ame_segment_time_and_jac(self._ptr,float(ds),ct.byref(x)),"time_and_jac");return x.time,tuple(x.jac)
    def state_time_and_jac(self,ds):
        x=_AllJac();_check(lib.ame_segment_state_time_and_jac(self._ptr,float(ds),ct.byref(x)),"state_time_and_jac");return x.w,tuple(x.state_jac),x.time,tuple(x.time_jac)
    def domain_probe(self,ds=None):
        x=_Probe();_check(lib.ame_segment_domain_probe_at(self._ptr,self.L if ds is None else float(ds),ct.byref(x)),"domain_probe")
        import math
        return NativeDomainProbe(bool(x.pre_event),x.event_position if math.isfinite(x.event_position) else None,int(x.cflow_status))
    def renormalized_time_w0_sensitivity(self,ds):
        x=ct.c_double();_check(lib.ame_segment_renormalized_time_w0(self._ptr,float(ds),ct.byref(x)),"renormalized_time_w0_sensitivity");return x.value
    def cache_stats(self):
        x=_Stats();_check(lib.ame_segment_cache_stats_get(self._ptr,ct.byref(x)),"cache_stats");return {"stations":int(x.stations),"cache_hits":int(x.cache_hits),"cflow_calls":int(x.cflow_calls),"local_steps":int(x.local_steps)}

def compile_segment_native(L,sigma,w0,k0,segment_type,grad=False,**options):
    if segment_type not in _MODE: raise ValueError(f"unsupported segment type: {segment_type!r}")
    o=lib.ame_segment_default_options()
    if "boundary_start" in options:o.boundary_start=bool(options.pop("boundary_start"))
    if "reverse_eta" in options:o.reverse_eta=bool(options.pop("reverse_eta"))
    if "authoritative_w1" in options:
        v=options.pop("authoritative_w1")
        if v is not None:o.has_authoritative_w1=1;o.authoritative_w1=float(v)
    pol=options.pop("x0_sensitivity_policy","negative_infinity")
    try:o.time_w0_policy=_POLICY[pol]
    except KeyError:raise ValueError(f"unknown x0_sensitivity_policy: {pol!r}")
    if options:raise TypeError(f"unsupported native segment options: {', '.join(sorted(options))}")
    status=ct.c_int();p=lib.ame_segment_compile(float(L),float(sigma),float(w0),float(k0),_MODE[segment_type],bool(grad),ct.byref(o),ct.byref(status))
    if not p: raise NativeSegmentError(status.value,"compile")
    return NativeSegment(p)
