# Planner architecture

This document describes the **final production architecture** of the time-optimal Micromouse planner. It is the best place to start after the root [`README.md`](../README.md) if you want to understand how the major subsystems fit together without reading implementation code.

<p align="center">
  <img src="../assets/red_comet/planner_architecture.svg" alt="Time-optimal Micromouse planner architecture" width="900">
</p>

The central design choice is to separate two different optimization problems:

- **discrete topology selection** — which connected maze corridor should the robot follow?;
- **continuous trajectory optimization** — within one chosen corridor, what smooth path and speed profile minimize traversal time while remaining physically feasible?

The planner does not attach a heuristic speed score to an A* path. Every complete topology that competes for the incumbent is converted into the same continuous trajectory problem, optimized under the selected vehicle model, and independently certified before its time is accepted.

## End-to-end data flow

```text
maze JSON / generated maze
        │
        ▼
MazeScenario + cell graph
        │
        ▼
compressed junction graph
        │
        ├──────── shortest-distance A* ────────┐
        │                                      │
        ▼                                      ▼
branch-and-bound prefixes               first complete topology
        │                                      │
        └──────────── complete topology ◄──────┘
                         │
                         ▼
                corridor construction
                         │
                         ▼
              reduced clothoid basis
                         │
                         ▼
         curvature / time continuation
                         │
                         ▼
          selective active-basis branch
                         │
                         ▼
               optimized geometry
                         │
                         ▼
          profile-selected speed solver
                         │
                         ▼
          independent geometry + dynamics
                    certification
                         │
                         ▼
               certified route time
                         │
                         └──── incumbent / B&B feedback
```

The implementation is layered for the same reason. The graph search never needs to reason directly about hundreds of nonlinear trajectory variables, and the continuous optimizer never has to choose among combinatorial maze routes.

## 1. Maze scenario and graph model

A planning request starts from either:

- a generated maze used by the benchmark/demo runner; or
- an `ame-maze-v1` JSON file loaded through `planning.maze_io`.

A `MazeScenario` retains the physical and semantic information that is not naturally represented by the legacy grid alone: source coordinate convention, scale, all semantic goal cells, the derived single-entry planning goal, and optional start heading.

Custom maze format details live in [`examples/mazes/FORMAT.md`](../examples/mazes/FORMAT.md).

The cell graph is then compressed into a **junction graph**. Long degree-2 corridors do not represent discrete decisions, so they can be represented as graph edges between branching/terminal junctions while still retaining the exact cell path behind each edge.

This graph is the domain of the discrete planner.

## 2. Discrete topology search

The production search has two roles for shortest-distance A*:

1. provide a deterministic first complete route;
2. provide the first **continuously optimized and certified time incumbent**.

A* is therefore a seed, not the final physical objective.

`planning.branch_and_bound` then explores competing junction paths. The search is finite under the configured visit policy (`maximum_node_visits=1` in the canonical simple-path experiments) and uses several conservative reductions before paying for a continuous leaf solve:

- **block-cut pruning** removes graph structure that cannot participate in a valid start-to-goal simple path;
- **residual reachability** rejects prefixes that can no longer reach the goal without violating the visit policy;
- **topology quotienting** can merge certified corridor-equivalent variants in supported open-room cases;
- **time lower bounds** estimate the best possible completion of a prefix under relaxed geometry/dynamics;
- the current certified incumbent turns those lower bounds into branch-and-bound pruning decisions.

A complete leaf is not ranked using topological length. It is passed to the route optimizer and becomes competitive only if that optimizer returns a certified trajectory.

The benchmark suite checks this separation explicitly. On the six deterministic cyclic topology cases, branch-and-bound selected a different certified topology from shortest-distance A* in 2/6 cases, with a maximum time improvement of 27.09%. On three tractable finite cases, B&B matched exhaustive simple-path enumeration exactly.

For the search rationale, including why a simple Pareto-label A* was not retained, see [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md#24-why-the-search-ended-up-as-branch-and-bound).

## 3. Fixed topology → continuous corridor problem

Once a cell topology is fixed, discrete search is finished. The route is converted into a sequence of convex corridor regions and a continuous path problem.

The path uses **piecewise-linear curvature**. On a segment of arc length `L`,

\[
\kappa(s)=\kappa_0+\sigma s,
\]

with planar evolution

\[
\frac{dx}{ds}=\cos\theta,\qquad
\frac{dy}{ds}=\sin\theta,\qquad
\frac{d\theta}{ds}=\kappa(s).
\]

Each segment is therefore an Euler spiral / clothoid; straight lines and circular arcs appear as limiting cases.

The geometry layer propagates these curves directly rather than approximating the optimization variables with a dense polyline. The robot is also modeled as a rectangle, not a point.

Two ideas are important here:

- **optimization constraints and physical authority are separate** — a finite active constraint pool makes the NLP tractable, while independent continuous geometry certification decides whether the final path is physically acceptable;
- the optimizer uses **positive log-length coordinates** internally, so segment ordering/positivity are structural properties rather than a long chain of fragile station inequalities.

The complete geometry/constraint architecture is documented in [`GEOMETRY_OPTIMIZATION.md`](GEOMETRY_OPTIMIZATION.md).

## 4. Route-optimization architectures and active basis

The repository retains three explicit route-optimization architecture modes:

| Mode | Role |
|---|---|
| `legacy_v9` | compatibility path for the older always-both warm-start architecture |
| `integrated_v10` | current generic `main.py` default with bounded warm-start/backend scheduling |
| `active_basis_v11` | qualified selective-basis architecture used by the historical-five campaign and final Red Comet case study |

The deeper active-basis algorithm documented here is therefore an explicit production architecture, not the implicit default of every CLI invocation. Select it with `--planner-architecture active_basis_v11` when that qualified workflow is intended.

`active_basis_v11` does not expose the densest corridor basis immediately.

The high-level state machine is:

1. build a **reduced maximal-run basis**;
2. establish a stable geometric basin through curvature-oriented continuation;
3. optimize traversal time on the reduced basis;
4. inspect exact support around individual turns;
5. activate additional turn-pair children only where the current solution can initialize them above the required conditioning threshold;
6. optimize the selectively expanded basis while relaxing a segment-length conditioning floor;
7. declare convergence only after both the reduced schedule and selective-basis closure semantics are satisfied.

The frozen curvature guard schedule is

\[
0.01 \rightarrow 0.005 \rightarrow 0.0025,
\]

and the current selective-basis child-length floor schedule is

\[
7.5\times10^{-4}
\rightarrow 7.0\times10^{-4}
\rightarrow 6.5\times10^{-4}
\rightarrow 6.0\times10^{-4}.
\]

Every richer-basis branch retains the previously certified solution as an incumbent. If a branch fails, times out, or cannot establish structural closure, it cannot destroy that incumbent; it is reported as incomplete rather than silently relabeled converged.

The current implementation is deliberately single-threaded at the route/branch level. Worker processes are used as killable safety/watchdog boundaries, not as route parallelism.

## 5. Time-model dispatch

Geometry optimization repeatedly asks a simple-looking question:

> What is the fastest feasible traversal time for this exact geometry, and what is its derivative with respect to the geometry parameters?

The implementation dispatches that question according to the selected physics profile.

### Legacy benchmark model

`legacy_grid_v1` uses the historically qualified hybrid **MOTOR / GRIP / BRAKE** reverse speed-profile solver. It constructs forward and backward extremals, locates physical mode crossings/events, inserts internal maximum-velocity-curve anchors when necessary, and merges all candidates into the final lower envelope.

GRIP flow evaluation is supplied by the native Cflow kernel; the native Segment/Crossing/Reverse stack owns the production hot path. The Python solver remains an independent reference implementation.

### Red Comet DD/yaw model

`red_comet_2017_dd_yaw_v1` selects a separate one-state differential-drive/yaw backend. It keeps the spatial state `w = v²`, but the allowable longitudinal acceleration interval also depends on left/right drivetrain limits and yaw dynamics. Its candidate limits include MOTOR, BRAKE, GRIP, SIDE_LEFT, and SIDE_RIGHT, together with actuator maximum-velocity-curve anchors.

The two models share the outer planner/geometry architecture but should not be conflated. The legacy benchmark model is not a hidden approximation to the Red Comet backend, and Red-Comet-only terms are profile-gated so they do not perturb the historically qualified legacy path.

See [`SPEED_PROFILE_SOLVER.md`](SPEED_PROFILE_SOLVER.md) for the full speed-solver design.

## 6. Scalar topology discovery and differentiable replay

A major performance boundary inside the legacy speed solver is the split between:

- **scalar topology construction** — determine which physical segments, events, anchors, and envelope pieces actually define the solution;
- **differentiable promotion/replay** — compile only the prefixes needed to evaluate derivatives on that already-authoritative scalar topology.

This means a scalar line-search request does not construct an unnecessarily large differentiable representation. If the optimizer later requests a gradient at the same point, the retained scalar build can be promoted rather than rediscovered.

The native reverse solver mirrors this ownership explicitly:

```text
ame_scalar_build
    owns physical topology + envelope
        │
        ├── time-only promotion ──► ame_reverse_build
        └── full promotion      ──► ame_reverse_build
```

The scalar build remains the numerical authority and must outlive any promoted reverse handle.

## 7. Certification is not optimizer termination

The planner treats three questions separately:

1. **Did the nonlinear optimizer terminate?**
2. **Is the returned geometry continuously feasible?**
3. **Does the selected dynamics implementation independently reproduce the claimed time/physical profile?**

A route comparison is accepted only after the relevant certification checks pass.

Geometry certification checks quantities such as endpoint residual, continuous rectangular-body corridor separation, finite parameters, and configured curvature-slope limits.

Dynamic certification depends on the profile. For the Red Comet release result, the final A* geometry is independently replayed by Python and native DD/yaw implementations; their reported times differ by about `2.06e-11 s`, far below the `2e-8 s` case-study certification tolerance.

This separation is why the repository avoids phrases such as “the optimizer succeeded, therefore the trajectory is valid.”

## 8. Native stack and reference stack

The numerically repetitive legacy hot path is implemented as a modular native stack:

```text
libcflow.so
    ▲
libame_segment.so
    ▲
libame_crossing.so
    ▲
libame_reverse.so
```

`libreverse_eta.so` is a narrowly gated derivative accelerator used only for qualified difficult reverse-time Cflow tangents. `libame_dd_yaw.so` is the separate Red Comet DD/yaw kernel.

Python remains the orchestration layer and, critically, an independent reference path. Native code was promoted layer-by-layer only after numerical equivalence, derivative, sanitizer, regression, and application-level gates.

Subsystem implementation documentation:

- [`native/cflow/README.md`](../native/cflow/README.md)
- [`native/reverse/README.md`](../native/reverse/README.md)
- [`native/reverse_eta/README.md`](../native/reverse_eta/README.md)

## 9. Evidence and postprocessing are separate from production

The repository deliberately separates three additional layers from planner code:

- `benchmarks/` calls production APIs but production modules do not import benchmark code;
- `visualization/` consumes saved route/result records and does not optimize or certify trajectories;
- `benchmark_results/reference/` and `analysis/red_comet_2017/` contain compact checked-in evidence sufficient to reproduce reports/figures without rerunning multi-hour campaigns.

This keeps scientific/release evidence inspectable without turning the production solver into a benchmarking framework.

See:

- [`BENCHMARKS.md`](../BENCHMARKS.md) — results and interpretation;
- [`benchmarks/README.md`](../benchmarks/README.md) — benchmark harness/reproduction;
- [`visualization/README.md`](../visualization/README.md) — plotting/animation API and numerical-authority boundary;
- [`RED_COMET_CASE_STUDY.md`](RED_COMET_CASE_STUDY.md) — final historical case study;
- [`OCP_COMPARISON.md`](OCP_COMPARISON.md) — independent simultaneous-OCP experiment.

## 10. Architectural invariants

Several rules survived enough failed implementations that they are now part of the architecture rather than incidental coding style:

1. **Exploit mathematical structure before adding generic numerical work.** Analytic MOTOR/BRAKE flow replaced RK2; the speed profile is event-driven rather than a dense NLP inside every objective call.
2. **Keep one state authority.** Derivatives, integrals, diagnostics, and event machinery may augment the physical trajectory but may not silently substitute a different one.
3. **Separate discovery from differentiation.** Scalar line searches should not pay for full derivative construction.
4. **Separate optimization constraints from final certification.** A finite active pool helps solve the problem; a continuous oracle decides whether the result is physically acceptable.
5. **Retain certified incumbents through risky continuation.** A failed richer representation cannot erase a known-good route.
6. **Treat continuous optimality claims as local.** Certification proves feasibility, not a global solution of the nonlinear trajectory problem.
7. **Keep the reference path alive.** Native speed is valuable only if numerical equivalence can still be checked independently.
8. **Measure end-to-end economics.** A faster kernel or theoretically richer optimizer is not promoted unless it improves the actual planner workload without weakening numerical authority.

These invariants explain much of the otherwise unusual structure in the codebase. The historical path that produced them is documented in [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md).
