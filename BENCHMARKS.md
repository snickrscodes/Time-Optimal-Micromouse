# Benchmark campaign

The repository contains a reproducible evaluation layer around the production planner. The benchmark package calls production APIs; production modules do **not** import benchmark code.

The campaign is designed to answer separate questions instead of collapsing the project into one vague “planner speedup” number.

## Headline results

| ID | Benchmark | Main question | Checked reference result |
|---:|---|---|---|
| 1 | Topology search | Can kinodynamic topology selection beat shortest-distance A* after the same continuous optimization? | **6/6 certified pairs**, 2 topology changes, up to **27.09%** lower time |
| 2 | Lower bounds | Does B&B match exhaustive finite enumeration and avoid expensive leaves? | **3/3 exact matches**, **50% median** complete solves avoided, **0/33** tested prefix admissibility violations |
| 3 | Gradients | Do Python/native derivatives and finite differences agree? | Python/native exact at report precision; FD median relative error **4.04e-11** over 58 smooth coordinates |
| 4 | Warm starts | Does staged initialization materially affect certification/basin quality? | production and length→time certify both checked routes; direct-time / curvature→time do not under the common budget |
| 5 | Native stack | Does native ownership preserve numerics and improve runtime? | **2.18× scalar**, **1.71× time+gradient**, **1.18× end-to-end optimization** median speedups |
| 6 | Resolution | Does simply increasing clothoid resolution give a monotone quality story? | exact prolongation is geometrically exact; independent reoptimization exposes basin/certification sensitivity |
| 7 | Fixed-geometry transcription | Does the specialized hybrid speed solver agree with a generic dense speed NLP? | 3/3 finest cold solves certify; **0.092% median** time difference; **934.9×** median IPOPT/production solve-time ratio |
| 8 | Full simultaneous OCP | Does an independent geometry+speed formulation find the same fixed-topology local geometry? | after structured 11→22→44→99 refinement, 96-interval OCP is **1.390% lower**; production hybrid replay on its geometry is **1.918% lower** |

These are different claims. Benchmark 5 measures implementation economics; Benchmark 7 isolates the fixed-geometry speed solver; Benchmark 8 challenges the outer nonconvex geometry optimizer.

## Quick reproduction

Core production-focused campaign:

```bash
python -m pip install -r requirements-benchmarks.txt
make native
python -m benchmarks.run --profile core
```

Generic CasADi/IPOPT baselines additionally require:

```bash
python -m pip install -r requirements-benchmarks-ocp.txt
python -m benchmarks.run --profile transcription
python -m benchmarks.run --profile full-ocp
```

Complete campaign:

```bash
python -m benchmarks.run --profile all
```

Fast public smoke campaign:

```bash
make benchmark-smoke
```

Ordinary runs are written under `benchmark_results/runs/`. The checked-in reference campaign lives under `benchmark_results/reference/`.

The reference directory is protected from ordinary overwrite. Regeneration is explicit:

```bash
python -m benchmarks.run --profile all --reference --overwrite
```

For the exact Python package versions used by the checked reference run:

```bash
python -m pip install -r requirements-reference.txt
```

## Benchmark 1 — topology search

**Question:** Does shortest grid distance remain the best physical route after every candidate topology receives the same continuous optimization/certification treatment?

The campaign uses six deterministic 4×4 cyclic mazes.

Reference result:

- 6/6 A*/B&B pairs are independently certified;
- B&B selects a different topology in 2/6 cases;
- maximum certified time improvement: **27.0919%**;
- median improvement: 0% because A* remains best in four cases.

No case in this specific benchmark is “longer but faster” by raw cell count; the point is that **different topology**, not simply distance, can change the best continuous racing line.

## Benchmark 2 — lower bounds

**Question:** Is branch-and-bound making the same discrete decision as exhaustive enumeration on tractable cases, and does the bound save expensive leaf optimizations?

Reference result across three 3×3 cyclic mazes:

- B&B winner matches exhaustive simple-path winner: **3/3**;
- total simple paths enumerated: 10;
- tested prefixes: 33;
- observed admissibility violations: **0** at the benchmark tolerance;
- median fraction of complete route optimizations avoided: **50%**.

This is finite empirical evidence around the production bound, not a replacement for the mathematical reasoning behind admissibility.

The Red Comet case study provides an instructive counterexample for **usefulness**: its conservative bound prunes no complete candidate, even though it remains suitable for a correct B&B search.

## Benchmark 3 — gradients

**Question:** Are the derivatives used by the geometry optimizer consistent across independent implementations and finite differences on smooth hybrid topologies?

Reference result:

- Python/native implementation cases: 5;
- maximum time-value difference: 0 at report precision;
- maximum gradient difference: 0 at report precision;
- finite-difference coordinates attempted: 66;
- smooth coordinates admitted to the aggregate: 58;
- excluded nonsmooth/invalid coordinates: 8;
- median FD relative error: **4.035e-11**.

Nonsmooth event/tie coordinates are excluded explicitly rather than being counted as derivative successes.

## Benchmark 4 — warm starts

**Question:** How much does trajectory staging matter before the final time-minimization budget?

The benchmark compares direct-time, curvature→time, length→time, and production staging under a common final polish budget.

Reference result on the two checked routes:

| Variant | Certification rate |
|---|---:|
| direct time | 0/2 |
| curvature → time | 0/2 |
| length → time | **2/2** |
| production warm start | **2/2** |

The benchmark is intentionally not interpreted as a universal ranking of initialization schedules. It demonstrates that warm-start policy is an algorithmic part of this nonconvex problem, not cosmetic setup.

## Benchmark 5 — native stack

**Question:** Does moving the Segment/Crossing/Reverse hot path to native ownership preserve numerics and improve realistic workloads?

Reference aggregate:

- median scalar speedup: **2.182×**;
- median time+gradient speedup: **1.714×**;
- median representative complete-optimization speedup: **1.183×**;
- maximum scalar time discrepancy: 0 at report precision;
- maximum gradient discrepancy: 0 at report precision;
- all end-to-end comparison routes certified.

The complete-optimizer speedup is intentionally reported separately from much larger cheap-profile microbenchmarks.

## Benchmark 6 — resolution sensitivity

**Question:** Is a reported structured trajectory obviously an artifact of a coarse clothoid basis?

The benchmark performs exact `N → 2N` prolongation before reoptimization. Exact prolongation preserves the continuous curve to floating-point precision; the measured maximum initial position error in the reference campaign is about `1.88e-15`.

The independent higher-resolution local solves do **not** produce a simple monotone quality story. In the checked campaign, `s007` fails at `N` with an internal-cap condition but certifies at `2N` at **1.222499 s**, while `s019` reaches the internal-cap condition at both `N` and `2N`. There are therefore **0/2 certified N/2N pairs** from which to report a paired time difference. The benchmark still establishes exact geometric prolongation to floating-point precision and, importantly, exposes optimizer/certification sensitivity instead of hiding failed solves.

For that reason the release interpretation separates “representation refinement” from “optimizer robustness.” Benchmark 8 later adds a stronger controlled multilevel refinement on one case.

## Benchmark 7 — fixed-geometry direct transcription

**Question:** On the same certified geometry, does the specialized continuous hybrid speed solver agree with a generic dense CasADi/IPOPT speed formulation?

Predeclared meshes:

```text
64, 128, 256, 512, 1024 intervals
```

At the finest 1024-interval cold-start comparison:

- 3/3 cases pass the independent continuous physics check;
- median absolute time difference from production: **0.0924%**;
- median IPOPT solve time: about **1.733 s**;
- median production scalar evaluation time: about **0.00327 s**;
- median runtime ratio: **934.95×**.

This is a fixed-geometry benchmark. It validates the specialized speed-profile formulation but says nothing about whether the outer geometry optimizer finds the best nonconvex basin.

## Benchmark 8 — simultaneous full OCP

**Question:** Does an independent generic geometry+speed NLP find the same fixed-topology continuous solution as the structured planner?

Canonical case:

```text
cyclic_4x4_s007
```

OCP meshes:

```text
24, 48, 96 intervals
```

The final fairness control exactly prolongs and **reoptimizes** the structured trajectory through:

```text
11 → 22 → 44 → 99 clothoid segments
```

while retaining each previous certified trajectory as an incumbent.

| Structured / OCP | Structured T | OCP objective T | Production hybrid replay on OCP geometry |
|---|---:|---:|---:|
| 22 / 24 | **1.154339753 s** | 1.164119079 s | **1.140520046 s** |
| 44 / 48 | 1.153401215 s | **1.147023932 s** | **1.134811789 s** |
| 99 / 96 | 1.153401215 s | **1.137372894 s** | **1.131273794 s** |

The structured solution improves from **1.171355379 s** at 11 segments to **1.153401215 s**. The exact 44→99 lift is essentially stationary under the first 99-segment transaction (`~1.4e-10 s` change).

At the finest comparison:

- OCP discretized objective: **1.390% lower** than structured;
- production continuous hybrid replay on OCP geometry: **1.918% lower**.

The second number is the key diagnostic: the production speed solver itself prefers the geometry found by the independent OCP. The remaining discrepancy is therefore best interpreted as **different local geometry basins / formulation behavior**, not obviously inadequate clothoid resolution.

Neither formulation is claimed globally optimal.

Detailed discussion: [`docs/OCP_COMPARISON.md`](docs/OCP_COMPARISON.md).

## Supplemental Benchmark 8S

`full_ocp_sensitivity` is retained as **Benchmark 8S**, a supplemental cross-topology probe.

It is not assigned the same evidentiary status as the original official 1–8 campaign because those cases were motivated after the canonical Benchmark 8 observation. Failures, timeouts, internal-cap events, and certification failures remain visible in its raw result rather than being replaced with favorable cases.

## Interpretation rules

The release follows several reporting rules:

- no continuous NLP result is called globally optimal;
- “exhaustive” refers only to the explicitly finite topology set under the declared visit policy;
- optimizer termination is not physical certification;
- quality aggregates include only independently certified trajectories where certification is part of the claim;
- generic OCP objective, independent discretized-profile recheck, and production hybrid replay are kept separate;
- timing suites run serially and incompatible numerical solver/plugin families are isolated in fresh process groups;
- failed rows remain failed rows rather than disappearing from the dataset.

## Evidence layout

```text
benchmark_results/reference/
├── manifest.json
├── topology_search.json
├── lower_bounds.json
├── gradients.json
├── warm_start.json
├── native_stack.json
├── resolution.json
├── direct_transcription.json
├── full_ocp.json
├── full_ocp_resolution_control.json
├── ...
└── BENCHMARK_REPORT.md
```

The JSON is the numerical source of truth. `BENCHMARK_REPORT.md` is generated from those result files and should not be hand-edited.

## More detail

- [`benchmarks/README.md`](benchmarks/README.md) — harness architecture, profiles, schemas, isolation, tests, and adding/running suites
- [`benchmark_results/reference/BENCHMARK_REPORT.md`](benchmark_results/reference/BENCHMARK_REPORT.md) — generated checked reference report
- [`docs/OCP_COMPARISON.md`](docs/OCP_COMPARISON.md) — Benchmark 8 interpretation
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the benchmarked production layers fit together
