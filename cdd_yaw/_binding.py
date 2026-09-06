from __future__ import annotations
import ctypes as ct
from pathlib import Path
from dataclasses import dataclass
from segment.base import SegmentType
from segment.differential_drive import DifferentialDriveParameters
from segment.physics_profiles import PhysicsProfile, RED_COMET_2017_NOMINAL
from optimization.dd_yaw_speed_profile import DDPassKind

class _P(ct.Structure): _fields_=[("beta",ct.c_double),("eta",ct.c_double),("q0",ct.c_double),("v_free",ct.c_double),("h_floor",ct.c_double),("c_floor",ct.c_double),("speed_margin",ct.c_double)]
class _Q(ct.Structure): _fields_=[("a_max",ct.c_double),("b_emf",ct.c_double),("a_brake",ct.c_double),("mu_g",ct.c_double)]
class _C(ct.Structure): _fields_=[("value",ct.c_double),("dw",ct.c_double),("dk",ct.c_double),("dsigma",ct.c_double)]
class _I(ct.Structure): _fields_=[("lower",ct.c_double),("upper",ct.c_double),("margin",ct.c_double),("motor_upper",ct.c_double),("grip",ct.c_double),("left_lower",ct.c_double),("left_upper",ct.c_double),("right_lower",ct.c_double),("right_upper",ct.c_double)]
class _A(ct.Structure): _fields_=[("w",ct.c_double),("state_jac",ct.c_double*8),("time",ct.c_double),("time_jac",ct.c_double*4)]
class _MVC(ct.Structure): _fields_=[("w",ct.c_double),("margin",ct.c_double),("hard_cap_w",ct.c_double),("upper_mode",ct.c_int),("lower_mode",ct.c_int),("status",ct.c_int)]

def _load():
 p=Path(__file__).resolve().parents[1]/"native"/"dd_yaw"/"libame_dd_yaw.so"
 if not p.exists(): raise ImportError(f"DD/yaw native library not built: {p}; run `make -C native/dd_yaw`")
 l=ct.CDLL(str(p));P=ct.POINTER(_P);Q=ct.POINTER(_Q)
 l.ame_dd_candidate_eval.argtypes=[P,Q,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_double,ct.POINTER(_C)];l.ame_dd_candidate_eval.restype=ct.c_int
 l.ame_dd_interval_eval.argtypes=[P,Q,ct.c_double,ct.c_double,ct.c_double,ct.POINTER(_I)];l.ame_dd_interval_eval.restype=ct.c_int
 l.ame_dd_segment_eval.argtypes=[P,Q,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.POINTER(_A)];l.ame_dd_segment_eval.restype=ct.c_int
 l.ame_dd_mvc_scan_bulk.argtypes=[P,Q,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_size_t,ct.c_int,ct.POINTER(_MVC)];l.ame_dd_mvc_scan_bulk.restype=ct.c_int
 l.ame_dd_status_name.argtypes=[ct.c_int];l.ame_dd_status_name.restype=ct.c_char_p
 return l
lib=_load()
_MODE={SegmentType.GRIP:1,SegmentType.MOTOR:2,SegmentType.BRAKE:3,SegmentType.SIDE_RIGHT:4,SegmentType.SIDE_LEFT:5}
_KIND={DDPassKind.FORWARD:1,DDPassKind.BACKWARD:-1}
def _pp(p):return _P(p.beta,p.eta,p.q0,p.side_free_speed_grid,p.h_floor,p.c_floor,p.speed_margin)
def _qq(q):return _Q(q.a_max,q.b_emf,q.a_brake,q.mu_g)
def _check(s):
 if s:
  x=lib.ame_dd_status_name(s); raise RuntimeError(f"native DD/yaw failed: {x.decode() if x else s}")
def candidate_native(mode,w,kappa,sigma,kind,params,profile=RED_COMET_2017_NOMINAL):
 p=_pp(params);q=_qq(profile);o=_C();_check(lib.ame_dd_candidate_eval(ct.byref(p),ct.byref(q),_MODE[mode],_KIND[kind],w,kappa,sigma,ct.byref(o)));return o.value,o.dw,o.dk,o.dsigma
def interval_native(w,kappa,sigma,params,profile=RED_COMET_2017_NOMINAL):
 p=_pp(params);q=_qq(profile);o=_I();_check(lib.ame_dd_interval_eval(ct.byref(p),ct.byref(q),w,kappa,sigma,ct.byref(o)));return tuple(getattr(o,x) for x in ("lower","upper","margin","motor_upper","grip","left_lower","left_upper","right_lower","right_upper"))
def segment_native(L,sigma,w0,k0,mode,kind,params,profile=RED_COMET_2017_NOMINAL,ds=None):
 p=_pp(params);q=_qq(profile);o=_A();d=L if ds is None else ds;_check(lib.ame_dd_segment_eval(ct.byref(p),ct.byref(q),_MODE[mode],_KIND[kind],L,sigma,w0,k0,d,ct.byref(o)));return o.w,tuple(o.state_jac),o.time,tuple(o.time_jac)


_MODE_NAME={0:"DOMAIN",1:"GRIP",2:"MOTOR",3:"BRAKE",4:"SIDE_RIGHT",5:"SIDE_LEFT"}
def mvc_scan_bulk_native(kappas,sigmas,params,profile=RED_COMET_2017_NOMINAL,n_scan=160):
 import numpy as np
 ka=np.ascontiguousarray(kappas,dtype=np.float64).reshape(-1); sg=np.ascontiguousarray(sigmas,dtype=np.float64).reshape(-1)
 if ka.size!=sg.size: raise ValueError("kappas and sigmas must have equal size")
 p=_pp(params);q=_qq(profile); arr=(_MVC*int(ka.size))()
 _check(lib.ame_dd_mvc_scan_bulk(ct.byref(p),ct.byref(q),ka.ctypes.data_as(ct.POINTER(ct.c_double)),sg.ctypes.data_as(ct.POINTER(ct.c_double)),ka.size,int(n_scan),arr))
 return [(float(x.w),float(x.margin),float(x.hard_cap_w),_MODE_NAME.get(int(x.upper_mode),"DOMAIN"),_MODE_NAME.get(int(x.lower_mode),"DOMAIN"),int(x.status)) for x in arr]
