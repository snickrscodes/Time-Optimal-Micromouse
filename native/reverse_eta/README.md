# Reverse-only regularized Cflow tangent cocycle

`native/reverse_eta` is a narrowly gated derivative accelerator for expensive **reverse-time GRIP tangent propagation**. It does not replace Cflow state evaluation and it does not build hybrid speed-profile topology.

The kernel uses the regularized variable

```text
eta = tan(delta)
```

to avoid the severe numerical/work growth of the ordinary variational representation on a qualified class of contracting near-grazing trajectories.

## Numerical-authority boundary

The separation from [`native/cflow`](../cflow/README.md) is intentional.

Production semantics are:

1. ordinary Cflow builds/evaluates the scalar physical trajectory;
2. the reverse solver establishes the authoritative hybrid topology;
3. only then may the derivative path consider the eta cocycle;
4. any eta failure, nonfinite result, or failed gate falls back to the ordinary production derivative path.

The eta kernel therefore supplies **missing tangent work**, not a second state/value authority.

## Dispatch gate

The production selector uses a grazing-tail work predictor

```text
N_asym = h (Q0 + Qh) (Q0^2 + Qh^2) / (0.0985 B)
```

and currently requires:

```text
N_asym >= 3000
```

plus the qualified sign-stable contracting chart and

```text
eta0 <= 30
```

The gate is intentionally strict. Broader exploratory low/mid tiers produced false-positive slowdowns on widened qualification corpora; the retained gate showed no measured slowdowns in that qualification and preserved substantial raw-kernel margin on the cases where it dispatched.

Boundary starts, expanding `|q|`, curvature-sign crossings, ordinary cheap requests, and states outside the qualified envelope remain on standard production derivatives.

## Why reverse-only?

Several candidate architectures were explored during Cflow performance research.

The winning production design reuses the already-authoritative scalar build and computes only the expensive tangent information that is missing during reverse replay.

A more self-contained “fresh production value + eta gradient” design was numerically strong but repeated scalar/value work that the reverse solver had already completed. In the current retained-build architecture that duplication made it materially slower.

The reverse-only kernel is therefore the smallest architecture that captures the measured derivative win without creating another value/topology surface.

## Build

From the repository root:

```bash
make native
```

or directly:

```bash
make -C native/reverse_eta
```

The root build places this library alongside Cflow/Segment/Crossing/Reverse so relative native linkage remains self-contained.

## Differential benchmarking switch

Set before Python import:

```bash
AME_ROBOT_DISABLE_REVERSE_ETA=1
```

to disable eta dispatch and force the ordinary production derivative path.

This is intended for A/B qualification and performance diagnosis, not as a separate physics mode.

## Provenance

[`PROVENANCE`](PROVENANCE) records the frozen source/history information associated with the shipped kernel.

Changes to the gate or transformed equations require application-level reverse/gradient qualification, not just local unit tests.

## Related documentation

- [`../cflow/README.md`](../cflow/README.md) — authoritative nonlinear flow runtime
- [`../reverse/README.md`](../reverse/README.md) — scalar-build/promote/reverse ownership
- [`../../docs/SPEED_PROFILE_SOLVER.md`](../../docs/SPEED_PROFILE_SOLVER.md) — planner-level mathematics
- [`../../docs/PROJECT_HISTORY.md`](../../docs/PROJECT_HISTORY.md) — derivative research history
