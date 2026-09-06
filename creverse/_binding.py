from __future__ import annotations
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path

class _Options(ct.Structure):
    _fields_=[
        ('has_init_w',ct.c_int),('init_w',ct.c_double),('has_terminal_w_max',ct.c_int),('terminal_w_max',ct.c_double),('initial_k',ct.c_double),('has_backward_init_k',ct.c_int),('backward_init_k',ct.c_double),
        ('n_scan',ct.c_int),('domain_scan',ct.c_int),('domain_margin',ct.c_double),
        ('fused_grip_discovery',ct.c_int),('validate_replay_domain',ct.c_int),
    ]
class _Seg(ct.Structure):
    _fields_=[
        ('pass_index',ct.c_int),('pass_kind',ct.c_int),('mode',ct.c_int),('event',ct.c_int),
        ('traversal_index',ct.c_int),('piece_index',ct.c_int),('initial_knot_index',ct.c_int),('boundary_start',ct.c_int),
        ('offset0',ct.c_double),('L_used',ct.c_double),('sigma',ct.c_double),('abs0',ct.c_double),('abs1',ct.c_double),
        ('direction',ct.c_double),('w0',ct.c_double),('k0',ct.c_double),('w1',ct.c_double),('k1',ct.c_double),
    ]
class _Env(ct.Structure):
    _fields_=[('source',ct.c_int),('source_index',ct.c_int),('pass_index',ct.c_int),('abs0',ct.c_double),('abs1',ct.c_double),('local0',ct.c_double),('local1',ct.c_double)]
class _ScalarStats(ct.Structure):
    _fields_=[('raw_values',ct.c_size_t),('pieces',ct.c_size_t),('scalar_passes',ct.c_size_t),('scalar_segments',ct.c_size_t),('envelope_pieces',ct.c_size_t),('possible_anchors',ct.c_size_t),('inserted_anchors',ct.c_size_t),('anchor_rounds',ct.c_size_t),('segment_compiles',ct.c_uint64),('crossing_calls',ct.c_uint64),('build_cflow_calls',ct.c_uint64),('build_cflow_local_steps',ct.c_uint64),('scalar_cflow_calls',ct.c_uint64),('scalar_cflow_local_steps',ct.c_uint64)]
class _ReverseStats(ct.Structure):
    _fields_=[('promoted_passes',ct.c_size_t),('promoted_segments',ct.c_size_t),('promoted_grip_segments',ct.c_size_t),('replay_cflow_calls',ct.c_uint64),('replay_cflow_local_steps',ct.c_uint64)]

_MODE={1:'GRIP',2:'MOTOR',3:'BRAKE'}
_EVENT={1:'PIECE_END',2:'GRIP_MOTOR',3:'MOTOR_GRIP',4:'GRIP_BRAKE',5:'BRAKE_GRIP'}
_PASS={1:'FORWARD',2:'BACKWARD'}

@dataclass(frozen=True,slots=True)
class NativeScalarSegment:
    pass_index:int; pass_kind:str; mode:str; event:str; traversal_index:int; piece_index:int; initial_knot_index:int; boundary_start:bool
    offset0:float; L_used:float; sigma:float; abs0:float; abs1:float; direction:float; w0:float; k0:float; w1:float; k1:float
@dataclass(frozen=True,slots=True)
class NativeEnvelopePiece:
    source:str; source_index:int; pass_index:int; abs0:float; abs1:float; local0:float; local1:float

class NativeReverseError(RuntimeError):
    def __init__(self,status:int,op:str,msg:str=''):
        super().__init__(f'native reverse {op} failed: {_status_name(status)}'+(f': {msg}' if msg else ''))
        self.status=int(status)

def _load():
    p=Path(__file__).resolve().parents[1]/'native'/'reverse'/'libame_reverse.so'
    if not p.exists(): raise ImportError(f'native Reverse v1 library not built: {p}; run `make reverse-native-v1`')
    lib=ct.CDLL(str(p));P=ct.c_void_p
    lib.ame_reverse_default_options.restype=_Options
    lib.ame_reverse_last_error.restype=ct.c_char_p
    lib.ame_reverse_status_name.argtypes=[ct.c_int];lib.ame_reverse_status_name.restype=ct.c_char_p
    lib.ame_scalar_build_create.argtypes=[ct.POINTER(ct.c_double),ct.c_size_t,ct.POINTER(_Options),ct.POINTER(P)];lib.ame_scalar_build_create.restype=ct.c_int
    lib.ame_scalar_build_destroy.argtypes=[P]
    lib.ame_scalar_build_get_stats.argtypes=[P,ct.POINTER(_ScalarStats)];lib.ame_scalar_build_get_stats.restype=ct.c_int
    lib.ame_scalar_build_pass_count.argtypes=[P];lib.ame_scalar_build_pass_count.restype=ct.c_size_t
    lib.ame_scalar_build_segment_count.argtypes=[P];lib.ame_scalar_build_segment_count.restype=ct.c_size_t
    lib.ame_scalar_build_envelope_count.argtypes=[P];lib.ame_scalar_build_envelope_count.restype=ct.c_size_t
    lib.ame_scalar_build_inserted_anchor_count.argtypes=[P];lib.ame_scalar_build_inserted_anchor_count.restype=ct.c_size_t
    lib.ame_scalar_build_inserted_anchor_at.argtypes=[P,ct.c_size_t];lib.ame_scalar_build_inserted_anchor_at.restype=ct.c_int
    lib.ame_scalar_build_segment_at.argtypes=[P,ct.c_size_t,ct.POINTER(_Seg)];lib.ame_scalar_build_segment_at.restype=ct.c_int
    lib.ame_scalar_build_envelope_at.argtypes=[P,ct.c_size_t,ct.POINTER(_Env)];lib.ame_scalar_build_envelope_at.restype=ct.c_int
    lib.ame_scalar_build_time_value.argtypes=[P,ct.POINTER(ct.c_double)];lib.ame_scalar_build_time_value.restype=ct.c_int
    lib.ame_scalar_build_time_value_gradient.argtypes=[P,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_size_t];lib.ame_scalar_build_time_value_gradient.restype=ct.c_int
    lib.ame_reverse_promote_time.argtypes=[P,ct.POINTER(P)];lib.ame_reverse_promote_time.restype=ct.c_int
    lib.ame_reverse_promote_full.argtypes=[P,ct.POINTER(P)];lib.ame_reverse_promote_full.restype=ct.c_int
    lib.ame_reverse_build_destroy.argtypes=[P]
    lib.ame_reverse_build_get_stats.argtypes=[P,ct.POINTER(_ReverseStats)];lib.ame_reverse_build_get_stats.restype=ct.c_int
    lib.ame_reverse_time_value_gradient.argtypes=[P,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_size_t];lib.ame_reverse_time_value_gradient.restype=ct.c_int
    lib.ame_reverse_final_state_rows.argtypes=[P,ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_size_t];lib.ame_reverse_final_state_rows.restype=ct.c_int
    lib.ame_reverse_time_value_gradient_raw.argtypes=[ct.POINTER(ct.c_double),ct.c_size_t,ct.POINTER(_Options),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_size_t];lib.ame_reverse_time_value_gradient_raw.restype=ct.c_int
    return lib
lib=_load()
def _status_name(s):
    p=lib.ame_reverse_status_name(int(s));return p.decode() if p else f'status_{s}'
def _last():
    p=lib.ame_reverse_last_error();return p.decode() if p else ''
def _check(s,op):
    if s:raise NativeReverseError(s,op,_last())

def _options(*,init_w=0.8,terminal_w_max=None,initial_k=0.0,backward_init_k=None,n_scan=48,domain_scan=48,domain_margin=None,fused_grip_discovery=True,validate_replay_domain=False):
    o=lib.ame_reverse_default_options();o.has_init_w=1;o.init_w=float(init_w);
    if terminal_w_max is not None:o.has_terminal_w_max=1;o.terminal_w_max=float(terminal_w_max)
    o.initial_k=float(initial_k);
    if backward_init_k is not None:o.has_backward_init_k=1;o.backward_init_k=float(backward_init_k)
    o.n_scan=int(n_scan);o.domain_scan=int(domain_scan)
    if domain_margin is not None:o.domain_margin=float(domain_margin)
    o.fused_grip_discovery=bool(fused_grip_discovery);o.validate_replay_domain=bool(validate_replay_domain);return o

class NativeScalarBuild:
    __slots__=('_ptr','raw_count')
    def __init__(self,ptr,raw_count):self._ptr=ptr;self.raw_count=int(raw_count)
    def close(self):
        if self._ptr:lib.ame_scalar_build_destroy(self._ptr);self._ptr=None
    def __del__(self):
        try:self.close()
        except Exception:pass
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb): self.close(); return False
    @property
    def closed(self): return not bool(self._ptr)
    @property
    def inserted_anchor_count(self): return int(lib.ame_scalar_build_inserted_anchor_count(self._ptr))
    @property
    def pass_count(self):return int(lib.ame_scalar_build_pass_count(self._ptr))
    @property
    def segment_count(self):return int(lib.ame_scalar_build_segment_count(self._ptr))
    @property
    def envelope_count(self):return int(lib.ame_scalar_build_envelope_count(self._ptr))
    def inserted_anchor_indices(self):
        n=int(lib.ame_scalar_build_inserted_anchor_count(self._ptr));return [int(lib.ame_scalar_build_inserted_anchor_at(self._ptr,i)) for i in range(n)]
    def stats(self):
        x=_ScalarStats();_check(lib.ame_scalar_build_get_stats(self._ptr,ct.byref(x)),'scalar_stats');return {k:int(getattr(x,k)) for k,_ in x._fields_}
    def segments(self):
        out=[]
        for i in range(self.segment_count):
            x=_Seg();_check(lib.ame_scalar_build_segment_at(self._ptr,i,ct.byref(x)),'segment_at')
            out.append(NativeScalarSegment(x.pass_index,_PASS[x.pass_kind],_MODE[x.mode],_EVENT[x.event],x.traversal_index,x.piece_index,x.initial_knot_index,bool(x.boundary_start),x.offset0,x.L_used,x.sigma,x.abs0,x.abs1,x.direction,x.w0,x.k0,x.w1,x.k1))
        return out
    def envelope(self):
        out=[]
        for i in range(self.envelope_count):
            x=_Env();_check(lib.ame_scalar_build_envelope_at(self._ptr,i,ct.byref(x)),'envelope_at')
            out.append(NativeEnvelopePiece(_PASS[x.source],x.source_index,x.pass_index,x.abs0,x.abs1,x.local0,x.local1))
        return out
    def time_value(self):
        x=ct.c_double();_check(lib.ame_scalar_build_time_value(self._ptr,ct.byref(x)),'scalar_time');return x.value
    def time_value_and_gradient(self):
        v=ct.c_double();g=(ct.c_double*self.raw_count)();_check(lib.ame_scalar_build_time_value_gradient(self._ptr,ct.byref(v),g,self.raw_count),'scalar_time_value_gradient');return v.value,list(g)
    def promote_time(self):return self._promote(False)
    def promote_full(self):return self._promote(True)
    def _promote(self,full):
        p=ct.c_void_p();_check((lib.ame_reverse_promote_full if full else lib.ame_reverse_promote_time)(self._ptr,ct.byref(p)),'promote_full' if full else 'promote_time');return NativeReverseBuild(p,self.raw_count,self,full)

class NativeReverseBuild:
    __slots__=('_ptr','raw_count','_scalar','full')
    def __init__(self,ptr,raw_count,scalar,full):self._ptr=ptr;self.raw_count=int(raw_count);self._scalar=scalar;self.full=bool(full)
    def close(self):
        if self._ptr:lib.ame_reverse_build_destroy(self._ptr);self._ptr=None
    def __del__(self):
        try:self.close()
        except Exception:pass
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb): self.close(); return False
    @property
    def closed(self): return not bool(self._ptr)
    def stats(self):
        x=_ReverseStats();_check(lib.ame_reverse_build_get_stats(self._ptr,ct.byref(x)),'reverse_stats');return {k:int(getattr(x,k)) for k,_ in x._fields_}
    def time_value_and_gradient(self):
        v=ct.c_double();g=(ct.c_double*self.raw_count)();_check(lib.ame_reverse_time_value_gradient(self._ptr,ct.byref(v),g,self.raw_count),'time_value_gradient');return v.value,list(g)
    def final_state_rows(self,which='forward'):
        if not self.full: raise NativeReverseError(7,'final_state_rows','full promotion required')
        w=(ct.c_double*self.raw_count)();k=(ct.c_double*self.raw_count)();kind=1 if which=='forward' else 2 if which=='backward' else 0
        _check(lib.ame_reverse_final_state_rows(self._ptr,kind,w,k,self.raw_count),'final_state_rows');return list(w),list(k)

def build_scalar_native(raw_params,**kwargs):
    raw=[float(x) for x in raw_params];a=(ct.c_double*len(raw))(*raw);o=_options(**kwargs);p=ct.c_void_p();_check(lib.ame_scalar_build_create(a,len(raw),ct.byref(o),ct.byref(p)),'scalar_build');return NativeScalarBuild(p,len(raw))

def time_value_and_gradient_native(raw_params,**kwargs):
    raw=[float(x) for x in raw_params];a=(ct.c_double*len(raw))(*raw);o=_options(**kwargs);v=ct.c_double();g=(ct.c_double*len(raw))();_check(lib.ame_reverse_time_value_gradient_raw(a,len(raw),ct.byref(o),ct.byref(v),g,len(raw)),'time_value_gradient_raw');return v.value,list(g)
