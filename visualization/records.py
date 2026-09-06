"""Self-contained route snapshots for expensive-run postprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from optimization import GeometryState, knot_parameters_to_raw


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def optimized_clothoid_length(raw_parameters: Sequence[float]) -> float:
    return float(math.fsum(float(v) for v in raw_parameters[0::2]))


def route_snapshot(
    route: Any,
    *,
    config: Mapping[str, Any] | None = None,
    certification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize all geometry needed to recreate figures without re-solving."""

    raw = knot_parameters_to_raw(route.parameters, initial_k=route.initial_state.k)
    stages = [_json_safe(stage) for stage in getattr(route, "stages", ())]
    warm = getattr(route, "warm_start_record", None)
    sparse = getattr(route, "sparse_specialist_record", None)
    active_basis = getattr(route, "active_basis_record", None)
    active_basis_json = None if active_basis is None else _json_safe(active_basis)
    # Make expensive active-basis results self-contained for later rendering.
    # Hybrid corridor assignments are normally persisted as an .npy checkpoint;
    # inline the small integer assignment into the JSON snapshot as well so a
    # copied metadata file is sufficient to rebuild the exact final corridor.
    if isinstance(active_basis_json, dict):
        segment_path = active_basis_json.get("final_segment_cells")
        if segment_path:
            try:
                active_basis_json["final_segment_cells_values"] = [
                    int(v) for v in np.load(segment_path).tolist()
                ]
            except (OSError, ValueError, TypeError):
                active_basis_json["final_segment_cells_values"] = None
    return {
        "cells": [list(cell) for cell in route.cells],
        "parameters": np.asarray(route.parameters, dtype=float).tolist(),
        "raw_parameters": [float(v) for v in raw],
        "initial_state": _json_safe(route.initial_state),
        "initial_k": float(route.initial_state.k),
        "time": float(route.time),
        "selected_stage": str(route.selected_stage),
        "architecture": str(getattr(route, "architecture", "unknown")),
        "stages": stages,
        "warm_start": None if warm is None else _json_safe(warm.to_json_dict() if hasattr(warm, "to_json_dict") else warm),
        "sparse_specialist": None if sparse is None else _json_safe(sparse.to_json_dict() if hasattr(sparse, "to_json_dict") else sparse),
        "active_basis": active_basis_json,
        "topology_centerline_length": float(max(0, len(route.cells) - 1)),
        "optimized_clothoid_length": optimized_clothoid_length(raw),
        "config": {} if config is None else _json_safe(config),
        "certification": None if certification is None else _json_safe(certification),
    }


def initial_state_from_record(record: Mapping[str, Any]) -> GeometryState:
    value = record["initial_state"]
    if isinstance(value, Mapping):
        return GeometryState(float(value["x"]), float(value["y"]), float(value["theta"]), float(value["k"]))
    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("initial_state record must be a 4-vector or mapping")
    return GeometryState(*(float(v) for v in value))


@dataclass(frozen=True, slots=True)
class RouteSnapshotView:
    cells: tuple[tuple[int, int], ...]
    parameters: np.ndarray
    raw_parameters: np.ndarray
    initial_state: GeometryState
    time: float
    selected_stage: str
    architecture: str
    config: Mapping[str, Any]


def route_view_from_record(record: Mapping[str, Any]) -> RouteSnapshotView:
    cells = tuple((int(cell[0]), int(cell[1])) for cell in record["cells"])
    parameters = np.asarray(record["parameters"], dtype=float)
    initial_state = initial_state_from_record(record)
    if "raw_parameters" in record:
        raw = np.asarray(record["raw_parameters"], dtype=float)
    else:
        raw = np.asarray(
            knot_parameters_to_raw(parameters, initial_k=initial_state.k),
            dtype=float,
        )
    return RouteSnapshotView(
        cells=cells,
        parameters=parameters,
        raw_parameters=raw,
        initial_state=initial_state,
        time=float(record["time"]),
        selected_stage=str(record.get("selected_stage", "unknown")),
        architecture=str(record.get("architecture", "unknown")),
        config=dict(record.get("config", {})),
    )
