from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

# Provenance of the user-supplied production archive from which this benchmark
# productization pass started. Git commit is the preferred public identifier.
IMPORTED_SOURCE_ARCHIVE_SHA256 = "b284fb1d3bb219b183aca9b3bda8f2de8cb6ed2072b188c93dd01137d250c496"


@dataclass(frozen=True, slots=True)
class MazeCase:
    name: str
    width: int
    height: int
    seed: int
    extra_openings: int
    opening_seed: int | None = None

    def resolved_opening_seed(self) -> int:
        return self.seed ^ 0x5EED5EED if self.opening_seed is None else self.opening_seed

    def to_dict(self) -> dict:
        value = asdict(self)
        value["opening_seed"] = self.resolved_opening_seed()
        return value


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    init_w: float = 0.8
    curvature_iterations: int = 20
    length_iterations: int = 20
    time_iterations: int = 20
    class_time_pilot_iterations: int = 6
    maximum_exchange_rounds: int = 3
    feasibility_tolerance: float = 2.0e-7
    curvature_regularization: float = 0.0
    length_curvature_regularization: float = 1.0e-5
    corridor_mode: str = "overlapping_cover"
    body_length: float = 5.0 / 9.0
    body_height: float = 4.0 / 9.0
    geometry_refinement: int = 1
    curvature_slope_limit: float | None = 50.0
    n_scan: int = 32
    envelope_scan: int = 16
    domain_scan: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullOcpSensitivityCase:
    name: str
    structured_refinement: int | None
    ocp_meshes: tuple[int, ...]
    note: str


@dataclass(frozen=True, slots=True)
class SuiteDefinition:
    name: str
    module: str
    public_title: str
    tier: Literal["official", "supplemental"] = "official"
    requires_topology: bool = False
    requires_native: bool = True
    requires_ocp_dependency: bool = False
    timeout_core_seconds: float = 300.0
    timeout_smoke_seconds: float = 90.0


@dataclass(frozen=True, slots=True)
class ProfileDefinition:
    name: str
    suites: tuple[str, ...]
    numerical_profile: Literal["core", "smoke"]


# ---------------------------------------------------------------------------
# Published deterministic cases and numerical policies.
# ---------------------------------------------------------------------------

CORE_TOPOLOGY_CASES = (
    MazeCase("cyclic_4x4_s007", 4, 4, 7, 3),
    MazeCase("cyclic_4x4_s019", 4, 4, 19, 3),
    MazeCase("cyclic_4x4_s043", 4, 4, 43, 3),
    MazeCase("cyclic_4x4_s101", 4, 4, 101, 3),
    MazeCase("cyclic_4x4_s313", 4, 4, 313, 3),
    MazeCase("cyclic_4x4_s911", 4, 4, 911, 3),
)

BOUND_CASES = (
    MazeCase("exhaustive_3x3_s007", 3, 3, 7, 2),
    MazeCase("exhaustive_3x3_s019", 3, 3, 19, 2),
    MazeCase("exhaustive_3x3_s043", 3, 3, 43, 2),
)
MAX_EXHAUSTIVE_PATHS = 64

SMOKE_TOPOLOGY_CASES = (MazeCase("smoke_3x3_s007", 3, 3, 7, 2),)
SMOKE_BOUND_CASES = (MazeCase("smoke_exhaustive_3x3_s019", 3, 3, 19, 2),)

SMOKE_OPTIMIZATION = OptimizationConfig(
    curvature_iterations=6,
    length_iterations=6,
    time_iterations=6,
    class_time_pilot_iterations=2,
    maximum_exchange_rounds=2,
    n_scan=16,
    envelope_scan=8,
    domain_scan=16,
)
CORE_OPTIMIZATION = OptimizationConfig()
BOUND_OPTIMIZATION = OptimizationConfig(
    curvature_iterations=10,
    length_iterations=10,
    time_iterations=10,
    class_time_pilot_iterations=3,
    maximum_exchange_rounds=2,
    n_scan=24,
    envelope_scan=12,
    domain_scan=24,
)

DIRECT_TRANSCRIPTION_CASE_NAMES = (
    "cyclic_4x4_s007",
    "cyclic_4x4_s019",
    "cyclic_4x4_s043",
)
DIRECT_TRANSCRIPTION_MESHES = (64, 128, 256, 512, 1024)
DIRECT_TRANSCRIPTION_WARM_MESHES = (256,)
SMOKE_DIRECT_TRANSCRIPTION_CASE_NAMES = ("smoke_3x3_s007",)
SMOKE_DIRECT_TRANSCRIPTION_MESHES = (1024,)

FULL_OCP_OPTIMIZATION = OptimizationConfig(
    curvature_iterations=100,
    length_iterations=120,
    time_iterations=60,
    class_time_pilot_iterations=8,
    maximum_exchange_rounds=5,
    feasibility_tolerance=2.0e-7,
    curvature_regularization=0.0,
    length_curvature_regularization=1.0e-5,
    corridor_mode="overlapping_cover",
    body_length=5.0 / 9.0,
    body_height=4.0 / 9.0,
    geometry_refinement=1,
    curvature_slope_limit=50.0,
    n_scan=48,
    envelope_scan=24,
    domain_scan=48,
)
FULL_OCP_CASE_NAME = "cyclic_4x4_s007"
FULL_OCP_MESHES = (24, 48, 96)
FULL_OCP_STRUCTURED_REFINEMENTS = (1, 2, 4, 9)
SMOKE_FULL_OCP_CASE_NAME = "smoke_3x3_s007"
SMOKE_FULL_OCP_MESHES = (12,)

# Follow-up sensitivity cases are predeclared here rather than selected by
# outcome. They are supplemental to the canonical single-topology Level-2 run.
FULL_OCP_SENSITIVITY_CASES = (
    FullOcpSensitivityCase(
        "cyclic_4x4_s019",
        structured_refinement=2,
        ocp_meshes=(28, 56),
        note="two-turn zig-zag; exact 2x structured control plus OCP refinement",
    ),
    FullOcpSensitivityCase(
        "cyclic_4x4_s043",
        structured_refinement=2,
        ocp_meshes=(16, 32),
        note="simpler cornering path; structured 2x attempt is retained even if it fails",
    ),
)


SMOKE_FULL_OCP_SENSITIVITY_CASES = (
    FullOcpSensitivityCase(
        "smoke_3x3_s007",
        structured_refinement=2,
        ocp_meshes=(12,),
        note="tiny infrastructure-only sensitivity case",
    ),
)




SUITES: dict[str, SuiteDefinition] = {
    "topology_search": SuiteDefinition(
        "topology_search", "benchmarks.topology_search",
        "Benchmark 1 — Shortest-distance A* topology vs kinodynamic topology search",
        timeout_core_seconds=180.0,
    ),
    "lower_bounds": SuiteDefinition(
        "lower_bounds", "benchmarks.lower_bounds",
        "Benchmark 2 — Lower-bound admissibility, tightness, and pruning",
        timeout_core_seconds=240.0,
    ),
    "gradients": SuiteDefinition(
        "gradients", "benchmarks.gradients",
        "Benchmark 3 — Speed-profile gradient correctness",
        requires_topology=True, timeout_core_seconds=120.0,
    ),
    "warm_start": SuiteDefinition(
        "warm_start", "benchmarks.warm_start",
        "Benchmark 4 — Geometry optimizer and warm-start ablation",
        requires_topology=True, timeout_core_seconds=180.0,
    ),
    "native_stack": SuiteDefinition(
        "native_stack", "benchmarks.native_stack",
        "Benchmark 5 — Native C/C++ stack performance and equivalence",
        requires_topology=True, timeout_core_seconds=180.0,
    ),
    "resolution": SuiteDefinition(
        "resolution", "benchmarks.resolution",
        "Benchmark 6 — Parameterization/resolution sensitivity",
        requires_topology=True, timeout_core_seconds=180.0,
    ),
    "direct_transcription": SuiteDefinition(
        "direct_transcription", "benchmarks.direct_transcription",
        "Benchmark 7 — Fixed-geometry direct-transcription baseline",
        requires_topology=True, requires_ocp_dependency=True, timeout_core_seconds=300.0,
    ),
    "full_ocp": SuiteDefinition(
        "full_ocp", "benchmarks.orchestration.full_ocp",
        "Benchmark 8 — Simultaneous fixed-topology OCP baseline",
        requires_topology=True, requires_ocp_dependency=True, timeout_core_seconds=420.0,
    ),
    "full_ocp_sensitivity": SuiteDefinition(
        "full_ocp_sensitivity", "benchmarks.orchestration.full_ocp_sensitivity",
        "Benchmark 8S — Cross-topology OCP sensitivity",
        tier="supplemental", requires_topology=True, requires_ocp_dependency=True, timeout_core_seconds=420.0,
    ),
}


PROFILES: dict[str, ProfileDefinition] = {
    "smoke": ProfileDefinition("smoke", tuple(SUITES), "smoke"),
    "core": ProfileDefinition(
        "core",
        ("topology_search", "lower_bounds", "gradients", "warm_start", "native_stack", "resolution"),
        "core",
    ),
    "transcription": ProfileDefinition("transcription", ("direct_transcription",), "core"),
    "full-ocp": ProfileDefinition("full-ocp", ("full_ocp", "full_ocp_sensitivity"), "core"),
    "all": ProfileDefinition("all", tuple(SUITES), "core"),
}
