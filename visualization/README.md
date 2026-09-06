# Visualization and release rendering

`visualization/` is the reusable **post-processing layer** for planner and benchmark results. It does not choose routes, optimize geometry, modify constraints, or certify trajectories.

The design goal is simple:

> **An expensive optimization run should be renderable many times without rerunning the optimization.**

Saved result records retain the exact route parameters and enough metadata to reconstruct the continuous geometry, speed traces, comparisons, and animations used by the public documentation.

## Numerical-authority boundary

Visualization is intentionally downstream of the solver.

- Saved clothoid parameters define the geometry.
- Saved/certified objective values define reported traversal time.
- Plotting samples are for rendering, not for redefining the path.
- Dense playback quadrature is allowed for `t -> s` animation mapping only after it is checked against the authoritative time value.
- Rendering does not recertify a trajectory.
- Benchmark timings never use visualization quadrature.

If a figure disagrees with the saved numerical result, the figure is wrong.

## Package map

| Module | Role |
|---|---|
| `sampling.py` | exact clothoid geometry sampling / station queries |
| `maze.py` | maze walls, semantic goals, goal-entry rendering |
| `trajectory.py` | route/topology and continuous trajectory drawing |
| `records.py` | shared route/result serialization helpers |
| `comparison.py` | reusable A* / optimized-route comparison figures |
| `detail.py` | body footprint and corridor-detail rendering |
| `speed.py` | introspectable hybrid speed-profile trace |
| `speed_plot.py` | speed/curvature/mode plots |
| `animation.py` | physical-time trajectory playback/GIF rendering |
| `case_study.py` | presentation-oriented A* vs B&B case-study loading/rendering |
| `search.py` | small diagnostic branch-and-bound search trees |
| `architecture.py` | programmatic conceptual architecture renderer |
| `style.py` | shared presentation primitives |

`main.py` retains compatibility wrappers around the reusable rendering functions, but new visualization work should use this package directly.

## 1. Exact geometry traces

`sampling.py` compiles the production clothoid path and exposes a `GeometryTrace` containing quantities such as:

```text
s, x, y, theta, kappa
segment ownership
exact knot stations
```

Sampling density controls rendering smoothness, not optimization resolution.

Because the trace is derived from the saved continuous clothoid parameters, a figure can be regenerated at higher visual resolution without changing the trajectory being depicted.

## 2. Maze and topology rendering

`maze.py` renders:

- exterior/interior walls;
- semantic multi-cell goal regions;
- the derived single goal-entry edge;
- coordinate-consistent custom/historical mazes.

`trajectory.py` can overlay:

- cell topology;
- exact sampled continuous trajectory;
- start/goal annotations.

These functions are used by both ordinary planner outputs and benchmark/case-study figures.

## 3. Body/corridor detail

The geometry-detail layer visualizes the same physical objects used by optimization/certification rather than inventing a separate display model.

Helpers reconstruct:

- transformed `RectangleBody` footprints at exact path states;
- convex corridor polygons from the production half-space representation;
- selected overlapping/reduced corridor regions.

This is useful when debugging why an apparently harmless centerline fails the rectangular-body corridor certificate.

## 4. Speed-profile traces

`speed.py` builds an introspectable `SpeedProfileTrace` containing:

- path station `s`;
- cumulative playback time `t`;
- squared speed `w`;
- speed `v`;
- longitudinal acceleration;
- curvature;
- active hybrid intervals/events.

For the legacy MOTOR/GRIP/BRAKE model, visualization temporarily selects the Python reference reverse backend to recover the detailed hybrid topology. The caller's previous backend is restored afterward.

The cumulative visualization time uses dense quadrature and is refined until it agrees with the authoritative production objective within the requested tolerance.

`station_at_time()` then provides the mapping used for physical-time animation.

## 5. Physical-time animation

`animation.py` combines the speed trace with exact geometry station queries.

A `PlaybackTrace` stores one exact path state per constant-time frame together with speed/mode metadata. `render_trajectory_gif()` animates the actual rectangular body on a fixed maze camera.

The robot therefore moves according to **physical playback time**, not at constant arc-length increments.

This is why synchronized route comparisons are visually meaningful: a faster route reaches the finish and freezes while the slower trajectory continues.

## 6. Release Red Comet assets

The canonical Red Comet renderer is:

```bash
python -m tools.visuals.red_comet_release
```

By default it reads:

```text
analysis/red_comet_2017/final_result.json
```

and regenerates the run-derived release assets under `assets/red_comet/`.

Current public outputs include:

| Asset | Purpose |
|---|---|
| `red_comet_astar_hero.gif` | single winning-route hero animation |
| `red_comet_astar_vs_green_side_by_side.gif` | synchronized A* vs historical animation |
| `red_comet_astar_vs_historical.svg` | static topology/geometry comparison |
| `red_comet_topology_summary.svg` | exhaustive ten-topology ranking |
| `red_comet_astar_geometry.svg` | final active-basis geometry detail |
| `red_comet_astar_speed.svg` | SI-unit speed/curvature profile |
| `visual_summary.json` | machine-readable manifest of regenerated release assets and route times |

The command does **not** rerun branch-and-bound or continuous optimization.

The architecture diagram is intentionally separate because it is architecture-derived, not result-derived. Its maintained source is:

```text
tools/visuals/planner_architecture_tikz.tex
```

## 7. OCP comparison figure

The normalized Benchmark 8 figure is regenerated with:

```bash
python -m tools.visuals.ocp_comparison
```

It consumes only:

```text
benchmark_results/reference/full_ocp_resolution_control.json
```

and writes the release comparison under `assets/benchmarks/`.

The renderer compares:

- multilevel structured 22/44/99-segment controls;
- simultaneous OCP 24/48/96-interval objectives;
- production hybrid replay on the OCP-discovered geometry.

It does not launch IPOPT or the production optimizer.

## 8. Generic A* vs B&B case-study renderer

For benchmark/planner metadata outside Red Comet:

```bash
python -m tools.visuals.astar_vs_bnb \
  --result benchmark_results/reference/topology_search.json \
  --output astar_vs_bnb.svg
```

Optional arguments allow selecting a specific Benchmark-1 case, overriding a custom maze path, hiding topology overlays, and choosing DPI.

The comparison is deliberately fair:

```text
shortest-distance topology + production continuous optimizer
versus
B&B-selected topology + the same production continuous optimizer
```

The renderer does not compare an optimized route against an unoptimized A* centerline.

## 9. Search-trace visualization

`planning.branch_and_bound_junction_paths()` can accept a diagnostic `search_trace_observer`.

When enabled, the observer records immutable events such as:

- node generation/expansion;
- bound pruning;
- reachability pruning;
- visit rejection;
- complete-route evaluation;
- incumbent update.

`visualization.search` turns those events into a compact search tree.

This is intended for small pedagogical/debugging cases. A full Micromouse search tree can be far too large to be a useful static graphic.

## 10. Development smoke renderers

Two smoke tools exercise the reusable visualization layers using checked-in result data.

### Geometry / corridor / speed smoke

```bash
python -m tools.visuals.smoke_v1_v3 \
  --result benchmark_results/reference/topology_search.json \
  --output-dir visualization_smoke
```

No optimizer is run.

### Animation / OCP / search / architecture smoke

```bash
python -m tools.visuals.smoke_v4_v7 \
  --topology-result benchmark_results/reference/topology_search.json \
  --ocp-result benchmark_results/reference/full_ocp_resolution_control.json \
  --output-dir visualization_smoke_v4_v7
```

This generates diagnostic artifacts for:

- a physical-time hero animation;
- normalized OCP comparison;
- a tiny deterministic B&B search tree;
- the programmatic conceptual architecture renderer.

Only the tiny search-tree smoke performs a cheap discrete B&B traversal; no continuous optimization is launched.

## 11. Architecture renderers

There are two architecture representations in the repository:

1. `visualization.architecture` — a programmatic semantic renderer useful for development/smoke tests;
2. `tools/visuals/planner_architecture_tikz.tex` — the hand-maintained publication architecture figure used by the main documentation.

The latter is the release authority for layout/typography. The former is useful when testing conceptual changes without manually editing TikZ first.

## 12. Adding a new visualization

Prefer the following pattern:

1. consume a saved route/result record;
2. reconstruct exact geometry through `visualization.sampling`;
3. derive display-only samples/quantities in a pure postprocessing function;
4. keep numerical authority and certification metadata untouched;
5. put generic reusable primitives in `visualization/`;
6. put one-off release orchestration/CLI in `tools/visuals/`;
7. add a smoke test that consumes checked-in compact data rather than launching an expensive optimizer.

Do not embed research/optimization logic in a renderer just because a figure needs a derived statistic.

## 13. Units

Internal solver geometry is normalized to maze grid units. Public historical figures may convert to SI using the scenario/profile scale.

For the Red Comet release:

- one cell pitch = 0.18 m;
- path station is displayed in meters;
- speed is displayed in m/s;
- curvature is displayed in 1/m.

The conversion is visual/reporting metadata only and does not modify saved optimizer variables.

## Related documentation

- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
- [`../docs/RED_COMET_CASE_STUDY.md`](../docs/RED_COMET_CASE_STUDY.md)
- [`../docs/OCP_COMPARISON.md`](../docs/OCP_COMPARISON.md)
- [`../examples/mazes/FORMAT.md`](../examples/mazes/FORMAT.md)
