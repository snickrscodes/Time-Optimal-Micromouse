"""Stable identifiers for the Red Comet case-study topologies.

The public/historical green overlay corresponds to one of the three 121-step
simple graph paths. A final overlay audit selected the upper-left-prefix
variant that stays on the left wall one cell longer before entering the top row.  Enumeration order is an implementation detail, so the
case-study code identifies it by the SHA-256 of the complete *source-coordinate*
cell path rather than by a transient ``P2`` label.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence


Cell = tuple[int, int]

# Verified against the historical green overlay after the final prefix correction
# audit and the transcribed maze. In the current deterministic preflight
# enumeration this is one of the 121-step green-like variants,
# but downstream experiments must use the digest rather than that numeric ID.
HISTORICAL_GREEN_SOURCE_PATH_SHA256 = (
    "9d9291150d7cb3f8b6ff517270fabab8b5e0cb88950f307aedc5bbf18e450789"
)


def source_cell_path_sha256(cells: Iterable[Cell]) -> str:
    payload = ";".join(f"{int(x)},{int(y)}" for x, y in cells).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def path_digest_in_source_coordinates(scenario, cells: Sequence[Cell]) -> str:
    return source_cell_path_sha256(scenario.to_source_cell(tuple(cell)) for cell in cells)


def is_historical_green(scenario, cells: Sequence[Cell]) -> bool:
    return (
        path_digest_in_source_coordinates(scenario, cells)
        == HISTORICAL_GREEN_SOURCE_PATH_SHA256
    )


def find_historical_green_path(scenario, cell_paths: Iterable[Sequence[Cell]]) -> tuple[Cell, ...]:
    matches = [tuple(cells) for cells in cell_paths if is_historical_green(scenario, cells)]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one historical-green topology in the Red Comet maze; "
            f"found {len(matches)}"
        )
    return matches[0]


__all__ = [
    "HISTORICAL_GREEN_SOURCE_PATH_SHA256",
    "source_cell_path_sha256",
    "path_digest_in_source_coordinates",
    "is_historical_green",
    "find_historical_green_path",
]
