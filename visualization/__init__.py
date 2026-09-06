"""Reusable, solver-neutral visualization helpers.

The package intentionally contains no optimization policy or search logic.  It
turns already-produced maze/route/speed data into traces and Matplotlib layers
that can be reused by the production CLI, experiments, and README artifact
scripts.
"""

from .sampling import GeometryTrace, sample_geometry_parameters, sample_raw_geometry, sample_geometry_stations
from .trajectory import body_polygon, topology_centerline
from .speed import SpeedEvent, SpeedModeInterval, SpeedProfileTrace, build_speed_profile_trace
from .animation import PlaybackTrace, build_playback_trace

__all__ = [
    "GeometryTrace",
    "PlaybackTrace",
    "SpeedEvent",
    "SpeedModeInterval",
    "SpeedProfileTrace",
    "body_polygon",
    "build_playback_trace",
    "build_speed_profile_trace",
    "sample_geometry_parameters",
    "sample_geometry_stations",
    "sample_raw_geometry",
    "topology_centerline",
]
