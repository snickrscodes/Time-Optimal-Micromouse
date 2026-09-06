from enum import Enum, auto
from typing import Tuple

class SegmentType(Enum):
    GRIP = auto()
    MOTOR = auto()
    BRAKE = auto()
    # DD/yaw profile-gated modes are appended so legacy numeric values remain stable.
    SIDE_RIGHT = auto()
    SIDE_LEFT = auto()

# evaluation implementations
class EvalSegment:
    def __init__(self, L: float, sigma: float, w0: float, k0: float, mode: SegmentType):
        self.L = L
        self.sigma = sigma
        self.w0 = w0
        self.k0 = k0
        self.mode = mode

    def w(self, ds: float) -> float:
        ...

# differentiable implementations
class DiffSegment:
    def __init__(self, L: float, sigma: float, w0: float, k0: float, mode: SegmentType):
        self.L = L
        self.sigma = sigma
        self.w0 = w0
        self.k0 = k0
        self.mode = mode

    def w(self, ds: float) -> float:
        # returns w(s)
        ...

    def w_and_jac(self, ds: float) -> Tuple[float, Tuple[float, float, float, float, float, float, float, float]]:
        # returns w(s), ∂(w(s), k(s))/∂(s, sigma, w0, k0) (flattened)
        ...
    
    def time_and_jac(self, ds: float) -> Tuple[float, Tuple[float, float, float, float]]:
        # returns T(s) = integral_0^s 1 / sqrt(w(s)) ds, ∂T(s)/∂(s, sigma, w0, k0)
        ...