# Native reverse solver

`native/reverse` contains the production C++ implementation of the legacy hybrid scalar speed-profile builder and differentiable reverse replay.

It mirrors the architecture of `optimization.reverse_solver` but moves the high-frequency segment/crossing/envelope orchestration into one native owner. The Python implementation remains available as an explicit reference backend.

For the mathematical speed-profile design, see [`docs/SPEED_PROFILE_SOLVER.md`](../../docs/SPEED_PROFILE_SOLVER.md).

## 1. Why this layer exists

After Cflow, segment dynamics, and crossing calculations were already native, profiling still showed a large Python orchestration cost during scalar profile construction:

- constructing thousands of segment objects;
- crossing/domain calls;
- pass records;
- internal-cap anchor rounds;
- lower-envelope assembly;
- repeated ownership/cache transitions.

The native reverse port therefore moved **ownership of the complete scalar hybrid topology** into C++ instead of adding more per-segment FFI calls.

The Python binding performs coarse operations such as “build this scalar profile” or “promote and compute the gradient,” not a Python↔C transition for every physical segment.

## 2. Why it is called a reverse solver

The fixed-geometry time objective is differentiated by replaying the already-selected hybrid speed profile and then propagating sensitivities backward through the segment/event composition.

The name therefore refers to the derivative/reverse-mode architecture, not to graph search.

The scalar layer itself constructs both forward and backward physical extremals before selecting the final lower envelope.

## 3. Ownership model

The public C ABI is in [`ame_reverse.h`](ame_reverse.h).

There are two primary opaque owners.

### `ame_scalar_build`

Owns the authoritative scalar solution:

- raw trajectory parameters;
- forward/backward traversal geometry;
- native scalar segment objects;
- crossing/domain decisions;
- internal-cap candidate passes;
- selected lower envelope;
- scalar time state/caches/statistics.

No differentiable replay segment is required to create this object.

### `ame_reverse_build`

A differentiable promotion of one `ame_scalar_build`.

Two promotion modes exist:

- `ame_reverse_promote_time()` — compile only what is required for the time objective;
- `ame_reverse_promote_full()` — compile complete replay passes for objectives/rows that require final-state sensitivities.

A reverse build **borrows** its scalar build. The scalar owner must remain alive until all promoted reverse handles are destroyed.

This borrowing relation is intentional: the scalar topology remains the physical/event authority.

## 4. Lifecycle

Conceptually:

```text
raw parameters
      │
      ▼
ame_scalar_build_create
      │
      ├── ame_scalar_build_time_value
      │
      ├── ame_scalar_build_time_value_gradient
      │
      └── promote
             │
             ▼
       ame_reverse_build
             │
             ├── ame_reverse_time_value_gradient
             └── ame_reverse_final_state_rows
```

The Python optimizer cache releases scalar handles deterministically when a cached point is superseded or no longer needed.

## 5. Scalar topology responsibilities

The scalar builder performs the same physical responsibilities as the qualified Python implementation:

- decode raw clothoid segment parameters;
- launch endpoint forward/backward passes;
- compile MOTOR / GRIP / BRAKE segments;
- locate switching and domain events through the native crossing layer;
- insert internal friction-cap anchors when needed;
- assemble all candidate passes;
- construct the final lower envelope;
- evaluate scalar travel time.

Positive GRIP→MOTOR roots and physical endpoint/domain authority remain in the separately qualified Segment/Crossing/Cflow layers rather than being reimplemented with different formulas in the reverse owner.

## 6. Differentiable replay

Time promotion uses the scalar envelope to identify the prefixes that actually contribute to the objective.

Only those prefixes need differentiable segment construction. Event sensitivities and local flow derivatives are assembled, after which the reverse sweep is largely local algebra.

This separation gives two advantages:

1. scalar line searches do not pay for unused derivative work;
2. the derivative path cannot silently choose a different hybrid topology from the scalar objective at the same point.

## 7. Public operations

The public ABI exposes:

### Scalar construction / inspection

```c
ame_scalar_build_create
ame_scalar_build_destroy
ame_scalar_build_get_stats
ame_scalar_build_pass_count
ame_scalar_build_segment_count
ame_scalar_build_envelope_count
ame_scalar_build_segment_at
ame_scalar_build_envelope_at
```

### Scalar objectives

```c
ame_scalar_build_time_value
ame_scalar_build_time_value_gradient
```

### Promotion / reverse objectives

```c
ame_reverse_promote_time
ame_reverse_promote_full
ame_reverse_build_destroy
ame_reverse_time_value_gradient
ame_reverse_final_state_rows
```

### One-shot convenience

```c
ame_reverse_time_value_gradient_raw
```

The one-shot API still uses the same scalar-build/promotion separation internally; it does not define a second numerical method.

## 8. Status model

The native layer reports explicit failure categories:

- `AME_REVERSE_OK`;
- invalid argument;
- physical/domain failure;
- conditioning failure;
- numerical failure;
- topology failure;
- allocation failure;
- unsupported operation.

The Python binding converts these into controlled exceptions/fallback behavior rather than allowing corrupted/nonfinite profiles to enter the optimizer.

## 9. Statistics and profiling

`ame_scalar_build_stats` records quantities such as:

- number of passes/segments/envelope pieces;
- possible/inserted anchor counts;
- segment compiles;
- crossing calls;
- Cflow calls/local steps.

`ame_reverse_stats` records promoted/replayed segments and Cflow derivative work.

These counters are used by benchmark/profiling code to distinguish:

- scalar topology construction cost;
- differentiable promotion cost;
- reverse sweep cost.

This is why the project reports several speedups rather than one ambiguous “native is X× faster” number.

## 10. Python binding

The `creverse` package is the coarse Python binding used by the optimizer/planner.

The packaged default backend is native. The Python reference can be selected with:

```bash
AME_REVERSE_BACKEND=python
```

or in-process through:

```python
with using_reverse_backend("python"):
    ...
```

The reference path is retained for A/B qualification, tests, diagnostics, and visualization introspection.

## 11. Build graph

The root `make native` target builds the native stack in dependency order.

Relevant legacy chain:

```text
native/cflow/libcflow.so
        ▲
native/segment/libame_segment.so
        ▲
native/crossing/libame_crossing.so
        ▲
native/reverse/libame_reverse.so
```

`native/reverse_eta/libreverse_eta.so` is also built because the segment/Cflow derivative path may dispatch to that narrowly gated accelerator.

From the repository root:

```bash
make native
```

or directly:

```bash
make -C native/reverse
```

## 12. Relocatable modular libraries

The native components use relative RUNPATHs so the packaged `native/` directory can move as a unit.

The modular shared-library boundaries are intentional. A monolithic/LTO experiment on the frozen reverse workload did not produce a meaningful staged speedup and made some one-shot measurements worse, so independent component auditability was kept instead of pursuing one giant binary.

## 13. Performance interpretation

The native reverse port produced its largest gains in scalar construction, where Python previously orchestrated many cheap operations.

On synthetic profiles where physical segment formulas are cheap, removing Python control-flow overhead can produce very large per-call speedups as segment count grows.

The public benchmark reports the more representative complete-stack numbers:

- median scalar speedup: **2.18×**;
- median time+gradient speedup: **1.71×**;
- median representative end-to-end optimization speedup: **1.18×**;
- maximum reported time/gradient discrepancy: zero at benchmark precision.

These are intentionally separated from microbenchmarks.

## 14. Qualification boundary

The native port is an implementation optimization, not a new physics formulation.

Production promotion required comparison against the Python reference across:

- scalar times;
- hybrid topology/event records;
- gradients;
- internal-cap cases;
- route corpora;
- optimizer behavior;
- sanitizer runs;
- unchanged parent regression tests.

The Python reference remains available specifically so this boundary can continue to be tested after future changes.

## 15. Relationship to DD/yaw

This library implements the **legacy MOTOR/GRIP/BRAKE reverse stack**.

The Red Comet `dd_yaw_v1` time model is profile-selected separately and has its own native kernel under `native/dd_yaw/` together with Python reference code under `optimization/dd_yaw_*`.

Do not interpret `libame_reverse.so` as the Red Comet dynamics implementation.

## Related documentation

- [`../../docs/SPEED_PROFILE_SOLVER.md`](../../docs/SPEED_PROFILE_SOLVER.md)
- [`../cflow/README.md`](../cflow/README.md)
- [`../reverse_eta/README.md`](../reverse_eta/README.md)
- [`../../BENCHMARKS.md`](../../BENCHMARKS.md)
