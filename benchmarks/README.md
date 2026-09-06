# Benchmark harness and reproduction guide

`benchmarks/` is the reproducible evaluation layer for the production Micromouse planner. It is intentionally **downstream** of production code: benchmark modules call planner/optimizer APIs, while production modules do not import the benchmark package.

For the measured results and their interpretation, read [`../BENCHMARKS.md`](../BENCHMARKS.md). This file is the operator/developer reference for the harness itself.

## 1. Installation

Production numerical dependencies:

```bash
python -m pip install -r requirements.txt
```

Core benchmark tooling/tests:

```bash
python -m pip install -r requirements-benchmarks.txt
```

Benchmarks 7 and 8 additionally require CasADi/IPOPT:

```bash
python -m pip install -r requirements-benchmarks-ocp.txt
```

For exact reproduction of the checked reference environment:

```bash
python -m pip install -r requirements-reference.txt
```

The native stack is built through the repository's normal build path:

```bash
make native
```

The benchmark runner verifies required shared libraries before launching suites. `--build-native` is only a convenience wrapper around that same root build target; it does not define a second native configuration.

## 2. Canonical runner

Default (`core`) profile:

```bash
python -m benchmarks.run
```

Explicit profiles:

```bash
python -m benchmarks.run --profile smoke
python -m benchmarks.run --profile core
python -m benchmarks.run --profile transcription
python -m benchmarks.run --profile full-ocp
python -m benchmarks.run --profile all
```

Individual suites can be selected with repeated arguments:

```bash
python -m benchmarks.run \
  --suite topology_search \
  --suite gradients
```

A suite that consumes fixed certified routes reuses `topology_search.json` from the selected results directory or generates the prerequisite when absent.

Official timing runs are serial. The public runner intentionally restricts `--jobs` to 1 so suite timing is not contaminated by cross-suite CPU contention.

## 3. Make targets

Convenience targets from the repository root:

```bash
make benchmark-smoke
make benchmark-core
make benchmark-transcription
make benchmark-full-ocp
make benchmark-all
make benchmark-reference
make benchmark-tests
```

`benchmark-reference` is consequential: it explicitly regenerates the checked reference campaign.

## 4. Output layout

Ordinary runs use unique timestamped directories:

```text
benchmark_results/
├── runs/
│   └── YYYYMMDDTHHMMSSZ_<profile>/
│       ├── manifest.json
│       ├── <suite>.json
│       ├── plots/
│       └── BENCHMARK_REPORT.md
└── reference/
    ├── manifest.json
    ├── <suite>.json
    ├── plots/
    └── BENCHMARK_REPORT.md
```

`benchmark_results/runs/` is ignored by Git. `benchmark_results/reference/` is the compact checked evidence intended for version control.

Ordinary commands do not overwrite reference data.

Reference regeneration requires the complete profile and explicit overwrite:

```bash
python -m benchmarks.run \
  --profile all \
  --reference \
  --overwrite
```

## 5. Generated report

`BENCHMARK_REPORT.md` is generated **only from result JSON**.

Regenerate an existing results directory without rerunning numerical experiments:

```bash
python -m benchmarks.report \
  --results-dir benchmark_results/reference \
  --output benchmark_results/reference/BENCHMARK_REPORT.md
```

Do not hand-edit the generated report. If a headline/interpretation is wrong, fix the source JSON schema/aggregation/report generator and regenerate it.

## 6. Official suites

Benchmarks 1–8 are the primary campaign:

| Suite | Role |
|---|---|
| `topology_search` | A* vs kinodynamic B&B after equal continuous optimization |
| `lower_bounds` | exhaustive finite replay, empirical prefix admissibility, pruning/ablation |
| `gradients` | Python/native reverse equivalence + deterministic finite differences |
| `warm_start` | initialization/staging reliability and basin comparison |
| `native_stack` | native scalar/gradient microbenchmarks + complete optimizer A/B |
| `resolution` | exact `N → 2N` trajectory prolongation/reoptimization |
| `direct_transcription` | fixed-geometry generic CasADi/IPOPT speed baseline |
| `full_ocp` | simultaneous fixed-topology geometry+speed OCP |

`full_ocp_sensitivity` is supplemental **Benchmark 8S**. It is retained explicitly as post-canonical robustness evidence and is not silently folded into the original predeclared headline set.

## 7. Deterministic configuration

Published cases and budgets live in [`config.py`](config.py), not inside ad-hoc benchmark scripts.

Current main deterministic sets include:

### Topology search

Six 4×4 cyclic mazes with seeds:

```text
7, 19, 43, 101, 313, 911
```

and three deterministic extra openings.

### Lower-bound exhaustive cases

Three 3×3 cyclic mazes with seeds:

```text
7, 19, 43
```

with two extra openings and a finite path-count safeguard.

### Benchmark 7 speed meshes

```text
64, 128, 256, 512, 1024
```

### Benchmark 8 OCP meshes

```text
24, 48, 96
```

on canonical case:

```text
cyclic_4x4_s007
```

The normalized post-campaign structured fairness control is stored separately as `full_ocp_resolution_control.json` because it exactly reoptimizes the structured representation through 22/44/99 segments.

Benchmark modules must not select cases based on observed performance.

## 8. Dependency resolution between suites

Some suites consume certified routes produced by earlier suites.

The runner resolves these dependencies through the selected results directory. For example, gradient/native/resolution experiments can reuse deterministic topology-search route records rather than independently solving a different route and calling it comparable.

This keeps benchmark questions aligned without coupling production modules to benchmark state.

## 9. Process isolation

Numerical extension libraries and solver plugins can retain threads/process state during interpreter teardown. The harness therefore runs expensive numerical jobs in fresh process groups.

Worker protocol:

1. run the numerical experiment;
2. write and fsync the result JSON;
3. write a completion marker;
4. attempt normal interpreter teardown;
5. if teardown stalls past the grace period, terminate the **entire worker process group**.

A forced teardown after the completion marker is an orchestration event, not a numerical timeout: the result and internal timings have already been written durably.

This prevents state from CasADi/IPOPT, native libraries, warm-start supervisors, or solver grandchildren from contaminating later suites.

## 10. Timing policy

Timing benchmarks are intentionally conservative:

- suites execute serially;
- warm-up is separated from recorded repetitions where appropriate;
- import/build time is excluded from microbenchmarks unless the benchmark specifically studies startup;
- generic solver workers record their internal solve time before teardown;
- end-to-end optimizer comparisons are reported separately from microkernel timings.

This is why Benchmark 5 publishes scalar, time+gradient, and complete-optimizer speedups separately.

## 11. Certification boundary

Optimizer success is not equivalent to benchmark success when the claim is physical quality.

Shared certification utilities live in:

```text
benchmarks/common/certification.py
```

Quality aggregates admit only independently certified trajectories when certification is part of the benchmark question.

### Benchmark 7

A discrete IPOPT speed solution must pass an independent continuous fixed-geometry physics check. Coarse NLP success can therefore still become `certification_failure`.

### Benchmark 8

Each accepted OCP result keeps three time quantities distinct:

1. generic discretized OCP objective;
2. independent re-evaluation of the discretized piecewise speed profile;
3. production continuous hybrid speed solve on the OCP-discovered geometry.

The reconstructed clothoid geometry must also pass the production endpoint/corridor certificate.

The release structured comparison is the multilevel control in:

```text
benchmark_results/reference/full_ocp_resolution_control.json
```

not a coarse/unreoptimized structured reference.

## 12. Failure taxonomy

Rows/components use explicit states rather than ambiguous `null` values:

```text
success
optimizer_failure
certification_failure
timeout
numerical_failure
internal_cap
not_requested
```

The versioned result schema validates `execution_status` values against this taxonomy.

A failed row remains visible in raw output and report generation.

## 13. Result schema

`schema.py` contains the shared validation helpers/versioning conventions for public benchmark JSON.

New suites should expose enough structure to distinguish:

- numerical execution status;
- optimizer status;
- certification status;
- quality metric(s);
- timing metric(s);
- deterministic case identity;
- configuration identity;
- environment/provenance.

Avoid one opaque “success + score” payload when multiple failure modes matter to interpretation.

## 14. Provenance

Every result records source identity.

When Git metadata is available, provenance includes commit/branch/dirty state. In unpacked release archives without `.git`, the harness computes a deterministic source-tree SHA-256 instead.

Generated benchmark result directories and compiled native artifacts are excluded from the fallback source hash so a rerun does not change its own source identity.

## 15. Environment capture

Environment metadata includes, as applicable:

- UTC timestamp;
- OS/kernel/architecture;
- CPU model and logical CPU count;
- Python implementation/version;
- NumPy/SciPy versions;
- C/C++ compiler strings;
- relevant thread-environment variables;
- native-library hashes/presence;
- active reverse backend;
- optimizer configuration;
- CasADi package/runtime information.

Timing comparisons should be interpreted together with this metadata rather than treated as universal hardware-independent constants.

## 16. Reference manifest

`benchmark_results/reference/manifest.json` ties the checked campaign together and records hashes/metadata for generated result artifacts.

If reference JSON/report content changes intentionally, regenerate/update the manifest through the supported benchmark/report workflow rather than hand-editing hashes.

## 17. Tests

Fast benchmark infrastructure qualification:

```bash
make benchmark-tests
```

or:

```bash
python -m pytest -q benchmarks/tests -m "not slow"
```

The fast tests cover areas such as:

- deterministic maze construction;
- lower-bound failure/admissibility handling;
- reverse-backend restoration;
- finite-difference machinery;
- certified-only aggregation;
- direct-transcription continuous checking;
- full-OCP mesh determinism;
- result schema/status validation;
- profile dependency resolution;
- native preflight;
- output/reference overwrite policy;
- Git/tree provenance;
- environment capture;
- public runner smoke behavior.

Slow tests exercise tiny end-to-end numerical versions of the public suites and are marked `slow` rather than being placed on every push.

## 18. Adding a benchmark suite

A new suite should follow this pattern:

1. define deterministic cases/budgets in `config.py`;
2. implement the experiment as a module that calls production APIs;
3. use shared environment/provenance/status/certification helpers;
4. isolate solver/plugin-heavy work in `orchestration/` when teardown or deadlines require it;
5. emit versioned machine-readable JSON;
6. keep failures explicit;
7. add report aggregation/rendering only after the raw schema is stable;
8. add fast infrastructure tests and, where appropriate, a marked slow numerical smoke;
9. do not import the new benchmark from production code.

## 19. Package map

```text
benchmarks/
├── common/                 shared certification, IO, timing, provenance, status
├── orchestration/          isolated worker/process-group management
├── config.py               deterministic cases/profiles/budgets
├── run.py                  public runner
├── report.py               generated Markdown report
├── schema.py               result validation/status conventions
├── topology_search.py
├── lower_bounds.py
├── gradients.py
├── warm_start.py
├── native_stack.py
├── resolution.py
├── direct_transcription.py
├── full_ocp.py
├── full_ocp_sensitivity.py
└── tests/
```

## Related documentation

- [`../BENCHMARKS.md`](../BENCHMARKS.md) — measured results and interpretation
- [`../docs/OCP_COMPARISON.md`](../docs/OCP_COMPARISON.md) — Benchmark 8 deep dive
- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — production system being benchmarked
- [`../benchmark_results/reference/BENCHMARK_REPORT.md`](../benchmark_results/reference/BENCHMARK_REPORT.md) — generated reference report
