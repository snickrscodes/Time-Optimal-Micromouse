from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from typing import TYPE_CHECKING

from planning.certification import certify_parameters_on_problem as _production_certify_parameters_on_problem

if TYPE_CHECKING:
    from main import OptimizedRoute
from planning import build_route_optimization_problem

from .io import json_safe


@dataclass(frozen=True, slots=True)
class CertificationResult:
    """Independent continuous geometry/corridor certification result.

    Optimizer termination status is intentionally not represented here: benchmark
    quality statistics accept a trajectory only when this independent certificate
    says it is feasible.
    """

    certified: bool
    endpoint_error: float
    corridor_upper_bound: float
    final_state: Any

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def certify_parameters_on_problem(
    problem: Any,
    parameters: Sequence[float],
    *,
    tolerance: float,
    maximum_abs_sigma: float | None,
) -> dict[str, Any]:
    return _production_certify_parameters_on_problem(
        problem, parameters, tolerance=tolerance, maximum_abs_sigma=maximum_abs_sigma
    )


def certify_optimized_route(route: "OptimizedRoute", config: Any) -> dict[str, Any]:
    problem = build_route_optimization_problem(
        route.cells,
        body_length=config.body_length,
        body_height=config.body_height,
        corridor_mode=config.corridor_mode,
        refinement_factor=config.geometry_refinement,
    )
    return certify_parameters_on_problem(
        problem,
        route.parameters,
        tolerance=config.feasibility_tolerance,
        maximum_abs_sigma=config.curvature_slope_limit,
    )


def certification_passed(record: Mapping[str, Any], key: str = "certification") -> bool:
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, Mapping):
            return False
        value = value.get(part)
    return bool(isinstance(value, Mapping) and value.get("certified", False))


def certified_only(records: Iterable[dict[str, Any]], key: str = "certification") -> list[dict[str, Any]]:
    """Return only independently certified rows for trajectory-quality aggregates."""
    return [record for record in records if certification_passed(record, key)]


def require_certified(record: Mapping[str, Any], key: str = "certification", *, context: str = "trajectory") -> None:
    if not certification_passed(record, key):
        raise RuntimeError(f"{context} is not independently certified")
