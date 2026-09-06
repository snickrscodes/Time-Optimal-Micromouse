# Independent simultaneous-OCP comparison

Benchmark 8 asks a deliberately different question from the rest of the evaluation suite:

> **On one fixed maze topology, does an independent simultaneous optimal-control formulation discover the same local geometry as the structured production optimizer?**

The result is one of the most important limitations/validation findings in the repository. After controlling for clothoid resolution, the independent CasADi/IPOPT formulation still finds a lower-time geometry basin.

<p align="center">
  <img src="../assets/benchmarks/full_ocp_comparison.svg" alt="Structured multilevel refinement versus simultaneous OCP" width="880">
</p>

## 1. Why this benchmark exists

The production planner is highly specialized:

- geometry is represented by piecewise clothoids;
- corridor feasibility is handled through structured constraints and exchange;
- traversal time is evaluated by a specialized event-driven hybrid speed solver;
- derivatives are propagated through the hybrid solution;
- the active-basis optimizer changes representation and continuation stages deliberately.

That specialization is the source of much of the planner's performance, but it also creates a risk: the structured optimizer may be biased toward a particular local geometry basin.

A generic simultaneous OCP provides an independent formulation with different variables and globalization behavior.

The benchmark is **not** intended to prove that IPOPT is globally optimal or that one software package is universally superior. It is a controlled disagreement experiment.

## 2. Fixed benchmark case

The canonical case is

```text
cyclic_4x4_s007
```

with the discrete topology fixed before either continuous formulation is compared.

This is important. Benchmark 8 does not mix topology-search differences with continuous-optimization differences. Both methods solve the same route-level physical problem as closely as practical.

## 3. Structured production formulation

The structured side uses the same family of clothoid geometry and continuous hybrid time model used by the production planner.

The original certified structured reference had **11 clothoid segments** and time

```text
1.1713553788696385 s
```

A direct comparison between that coarse representation and a much denser OCP would be ambiguous, so the final benchmark includes a stronger multilevel resolution control described below.

## 4. Simultaneous CasADi/IPOPT formulation

The independent OCP optimizes geometry and squared speed simultaneously over a spatial mesh.

The benchmark uses fresh cold-start solves at:

```text
24 intervals
48 intervals
96 intervals
```

The OCP includes discretized geometry/dynamics/physical constraints corresponding to the same fixed route problem, but its optimization variables and numerical formulation are intentionally different from the structured production solver.

Every headline OCP result must pass the benchmark's independent continuous certification rather than being accepted solely because IPOPT reports success.

The implementation lives in:

- `benchmarks/full_ocp.py`;
- `benchmarks/full_ocp_generic.py`;
- `benchmarks/full_ocp_reference.py`;
- `benchmarks/orchestration/full_ocp.py`.

## 5. Why the first comparison was not enough

A denser formulation finding a lower objective than an 11-segment structured trajectory does not immediately imply a better optimizer basin.

There are at least two possible explanations:

1. **representation resolution** — 11 clothoid segments may simply be too coarse;
2. **local-basin/formulation behavior** — the two nonlinear optimizers may land on genuinely different geometries.

The final release therefore does not use a raw coarse-structured comparison as the headline result.

Instead, it performs an exact multilevel structured control.

## 6. Exact structured prolongation and reoptimization

Starting from the certified 11-segment structured trajectory, the path is exactly resegmented/prolonged through

```text
11 → 22 → 44 → 99 segments
```

The newly introduced geometry degrees of freedom are **not frozen**. Each level is locally reoptimized in deterministic bounded transactions.

At every stage the previously certified trajectory is retained as an incumbent. A new candidate can replace it only after the normal independent certification checks.

This gives a much stronger resolution test than simply subdividing the original curve and evaluating the same geometry at more segments.

## 7. Canonical results

| Structured / OCP resolution | Structured optimized time | OCP objective | Production hybrid replay on OCP geometry |
|---|---:|---:|---:|
| 22 segments / 24 intervals | **1.154339753 s** | 1.164119079 s | **1.140520046 s** |
| 44 segments / 48 intervals | 1.153401215 s | **1.147023932 s** | **1.134811789 s** |
| 99 segments / 96 intervals | 1.153401215 s | **1.137372894 s** | **1.131273794 s** |

The structured refinement itself improves substantially:

```text
11 segments: 1.171355379 s
22 segments: 1.154339753 s
44 segments: 1.153401215 s
99 segments: 1.153401215 s
```

Relative to the original 11-segment solution, the final structured control improves by about **1.53%**.

## 8. Why the 44 → 99 result matters

The exact 44→99 lift preserves the structured time to floating-point precision before additional optimization.

The first 99-segment local transaction gives

```text
1.153401215127105 s
```

with a gain of only approximately

```text
-1.38e-10 s
```

relative to the exact lifted start.

That does not mathematically prove resolution convergence, but it is strong evidence that **insufficient clothoid segmentation is no longer the main explanation** for the remaining gap in this local basin.

The release wording therefore says the multilevel structured control is **stable at the tested final refinement**, not a theorem-level claim of global or asymptotic resolution convergence.

## 9. Separate the OCP objective from production hybrid replay

The OCP speed state is discretized. A lower OCP objective could therefore, in principle, reflect the speed discretization rather than a genuinely better geometry.

To separate those effects, the OCP-discovered geometry is handed back to the production continuous hybrid speed solver.

At the finest level:

```text
structured 99-segment time        1.153401215 s
OCP 96-interval objective         1.137372894 s
production hybrid replay on OCP   1.131273794 s
```

The OCP objective is **1.390% lower** than the refined structured result.

The production hybrid solver evaluates the OCP geometry **1.918% lower** than the structured result.

The second comparison is especially informative: the specialized production speed solver itself prefers the geometry found by the independent OCP.

## 10. Interpretation

The strongest supported interpretation is:

> **The remaining difference is primarily a local-geometry-basin / formulation effect, not an obvious artifact of insufficient structured clothoid resolution.**

This means the structured outer optimizer is not globally optimal on the tested fixed topology.

It does **not** mean:

- the simultaneous OCP is globally optimal;
- IPOPT is universally better than the structured optimizer;
- the structured speed solver is inaccurate;
- the full maze-planning architecture should be replaced with a simultaneous OCP at every branch-and-bound leaf.

Both continuous formulations are nonconvex local methods.

## 11. Relationship to Benchmark 7

Benchmark 7 and Benchmark 8 test different layers.

### Benchmark 7: fixed geometry

The geometry is held fixed and only the speed profile is reformulated as a dense generic transcription.

At the finest 1024-interval mesh, the generic transcription agrees closely with the production hybrid speed solver (about **0.092% median** time difference across the three cases) but is about **934.9×** slower at the median for the measured fixed-geometry solve.

That supports the specialized **speed solver**.

### Benchmark 8: geometry + speed

Both geometry and speed are optimized simultaneously.

The independent formulation finds a different lower geometry basin.

That exposes a limitation in the **outer continuous geometry optimization**, not a contradiction of Benchmark 7.

Together the two experiments provide a useful decomposition:

```text
specialized fixed-geometry speed solver: strongly validated
outer nonconvex geometry basin: not globally resolved
```

## 12. Why this does not replace the production architecture

The production planner may evaluate many complete topologies inside branch-and-bound. Embedding a large simultaneous generic OCP inside every leaf would change the computational economics dramatically.

The structured architecture exists so that:

- hybrid speed evaluation is cheap enough to call repeatedly;
- gradients exploit event/segment structure;
- geometry constraints can be generated incrementally;
- certified incumbents can be preserved across staged basis changes;
- topology search can remain a separate finite layer.

The OCP comparison is therefore best understood as an **independent scientific cross-check and local-optimality probe**, not a drop-in production backend recommendation.

## 13. Reproducing the canonical comparison

Install the OCP dependencies:

```bash
python -m pip install -r requirements-benchmarks-ocp.txt
```

Run the canonical full-OCP benchmark:

```bash
python -m benchmarks.run --profile full-ocp
```

The checked-in multilevel fairness control is stored at:

```text
benchmark_results/reference/full_ocp_resolution_control.json
```

The release graphic is regenerated without launching IPOPT or the structured optimizer:

```bash
python -m tools.visuals.ocp_comparison
```

That renderer consumes only the compact normalized control JSON.

## 14. Evidence files

| Artifact | Role |
|---|---|
| `benchmark_results/reference/full_ocp.json` | canonical simultaneous-OCP run data |
| `benchmark_results/reference/full_ocp_resolution_control.json` | normalized multilevel structured fairness control |
| `assets/benchmarks/full_ocp_comparison.svg` | release comparison figure |
| `benchmark_results/reference/BENCHMARK_REPORT.md` | generated reference campaign report |

The checked-in JSON, not prose copied into a README, is the numerical source of truth.

## Related documentation

- [`GEOMETRY_OPTIMIZATION.md`](GEOMETRY_OPTIMIZATION.md) — structured outer optimizer
- [`SPEED_PROFILE_SOLVER.md`](SPEED_PROFILE_SOLVER.md) — production hybrid speed solver
- [`../BENCHMARKS.md`](../BENCHMARKS.md) — full benchmark campaign
- [`../benchmarks/README.md`](../benchmarks/README.md) — harness/reproduction details
