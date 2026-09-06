# Cflow: native nonlinear GRIP flow kernel

Cflow is the dependency-light C11 runtime used by the legacy hybrid speed solver for the nonlinear GRIP-limited flow. It exists to hide the numerical complexity of that flow behind a small, testable ABI.

For the planner-level mathematics and how GRIP fits into the forward/backward envelope solver, see [`docs/SPEED_PROFILE_SOLVER.md`](../../docs/SPEED_PROFILE_SOLVER.md). For the history of the earlier RK2, Chebyshev, and Taylor implementations, see [`docs/PROJECT_HISTORY.md`](../../docs/PROJECT_HISTORY.md).

## 1. Problem solved by Cflow

In normalized variables the general GRIP branch is

\[
\frac{dX}{ds}=2\sqrt{1-\left[(q_0+bs)X\right]^2},
\]

with the additive observable

\[
J=\int \frac{ds}{\sqrt{X}}.
\]

The application mapping used by the legacy segment layer is

```text
X  = w / MU_G
q0 = k0
b  = sigma
h  = requested path prefix
```

where `w = v²`, `k0` is initial curvature, and `sigma = dk/ds` on the clothoid segment.

Cflow evaluates the normalized flow; the caller applies the physical scaling needed by the full time model.

## 2. Why this is a separate library

Earlier versions of the project evaluated GRIP with:

1. low-order RK2;
2. a global Chebyshev surrogate after nondimensionalization;
3. an adaptive high-order Taylor-series compiler.

The Chebyshev coordinate system became poorly conditioned as curvature slope approached zero, while the Taylor implementation became too complex and expensive to expose directly to the optimizer.

Cflow preserves the successful mathematical pieces while presenting one ordinary flow primitive to the segment/reverse layers.

The runtime is intentionally independent of Python/scientific-library dependencies. Generated coefficient tables are checked in; a production build requires only a C compiler, the C runtime, and `libm`.

## 3. Numerical-authority rule

Cflow follows one rule that is critical to the rest of the hybrid solver:

> **Endpoint state is authoritative.**

Derivative, integral, event, and diagnostic machinery may augment the endpoint state but may not silently replace it with a separately approximated physical trajectory.

This is why the endpoint value path remains the authority for state and event classification even when augmented/integral machinery performs additional work.

The reverse solver relies on this property when it retains a scalar hybrid topology and later promotes that exact topology for derivatives.

## 4. Public ABI

The public header is [`include/cflow.h`](include/cflow.h). Hidden visibility is used for internal helpers; only the application-facing functions are exported.

### General interior evaluation

```c
cflow_eval(...)
cflow_eval_jacobian(...)
cflow_integral_value(...)
cflow_integral_jacobian(...)
cflow_eval_all(...)
cflow_first_event(...)
```

### Exact friction-cap inward continuation

```c
cflow_boundary_eval(...)
cflow_boundary_eval_jacobian(...)
cflow_boundary_integral_value(...)
cflow_boundary_integral_jacobian(...)
cflow_boundary_eval_all(...)
```

The boundary functions are one-sided continuation APIs for exact cap starts. They avoid forcing a caller to perturb an exact physical boundary merely to enter the numerical chart.

## 5. Status model

The ABI reports explicit statuses rather than encoding every failure as NaN.

Current statuses include:

- `CFLOW_OK`;
- `CFLOW_EVENT`;
- `CFLOW_BEYOND_EVENT`;
- `CFLOW_OUTSIDE_REAL_DOMAIN`;
- `CFLOW_CONDITIONING_LIMIT`;
- `CFLOW_NUMERICAL_FAILURE`;
- `CFLOW_NO_EVENT_WITHIN_HORIZON`;
- `CFLOW_INVALID_ARGUMENT`;
- `CFLOW_OUTSIDE_INTEGRAL_DOMAIN`.

The application layer treats these statuses as part of hybrid physical/event semantics, not merely low-level error codes.

## 6. Numerical regions

Cflow uses multiple internal numerical representations because one global polynomial/chart was not reliable or efficient over the complete physical domain.

The implementation contains specialized components for:

- a core map;
- high-`|qX|` panels;
- far-field behavior;
- exact `b = 0` structure;
- terminal/domain events;
- exact boundary continuation;
- integral/observable evaluation;
- fallback numerical integration where specialized maps are not authoritative.

Dispatch remains internal; callers see one flow API.

## 7. High-`|qX|` Tucker maps

The high-`|qX|` endpoint panels use Tucker-compressed Chebyshev tensors.

Frozen endpoint ranks are:

| Panel | Tucker ranks |
|---|---|
| mid | `(12, 7, 8)` |
| upper | `(11, 7, 8)` |
| near | `(10, 6, 7)` |

The dense high-`z` tensor is retained where required by the integral defect/correction path; compression does not redefine terminal/event semantics.

## 8. Fixed-face specialization

For accepted maximal high-`z` steps, the active limiting face can be known structurally from the step-cap decision. When the C or `s` admissibility cap is the limiter, the accepted safety face is the fixed normalized coordinate `±0.985`.

The production fast path therefore evaluates precomputed face maps directly rather than recomputing an interior tensor and then testing whether the result happened to land near a face.

Endpoint maps use reconstructed two-dimensional Chebyshev value/normal-derivative surfaces. Integral maps use the representation selected by measurement:

- mid/upper: direct 2-D faces;
- near: precontracted Tucker face cores (`6,4,4`).

The Jacobian kernels fuse value/tangential and normal contractions where possible.

If the public horizon truncates the maximal step, another chart wins dispatch, or fallback quadrature asks for a shorter substep, Cflow returns to the original full interior evaluator.

Compile-time differential switch:

```text
CFLOW_DISABLE_HIGHZ_FACE_SPECIALIZATION
```

## 9. Separatrix and conditioning policy

Production no longer performs heuristic runtime snapping onto an asymptotic separatrix.

Ordinary binary64 inputs are never projected onto that trajectory merely because they are numerically close. Large-state asymptotics are retained only for **scale-aware conditioning analysis**.

`CFLOW_CONDITIONING_LIMIT` is reserved for expanding trajectories whose propagated input-quantization uncertainty reaches the configured endpoint-state uncertainty budget.

The frozen policy and constants are recorded in [`PROVENANCE`](PROVENANCE); there is no separate external separatrix-policy document in the release tree.

## 10. Semigroup and additive-observable composition

Cflow is designed for repeated prefix queries. The state map follows the flow semigroup structure, and the time observable follows the corresponding additive cocycle.

That enables continuation/caching without repeatedly restarting every prefix from `s = 0` and reduces the chance that two channels reconstruct subtly different trajectories.

This composition structure is important to the reverse solver's retained-build architecture.

## 11. Integral channel

The integral API evaluates the normalized observable and, where requested, derivatives with respect to the initial state/flow parameters.

The integral machinery may use augmented RK or defect/correction paths internally, but it may not replace the endpoint state authority described above.

This rule is particularly important because the project previously encountered cases where a state channel could be correct while a derived time scaling was not; the calibrated Red Comet native-build campaign added explicit profile-relative state+time differential checks for that reason.

## 12. Core scalar/Jacobian alignment

The current release aligns one core endpoint-state degree-scaling expression across scalar and Jacobian paths by using

```c
CFLOW_CORE_RSTAR * CFLOW_CORE_RSTAR
```

in both paths rather than two adjacent binary64 expressions for the same mathematical constant.

The change adds no algorithmic work; it removes a one-ULP literal discrepancy that could otherwise break exact scalar/Jacobian endpoint-state identity after tiny upstream rounding changes.

## 13. Generated tables and provenance

Generated coefficient/table sources live under [`generated/`](generated/). Their hashes and the frozen numerical/build policy are recorded in [`PROVENANCE`](PROVENANCE).

The checked-in generator for high-`z` face tables is:

```text
tools/generate_highz_face_tables.py
```

relative to this `native/cflow/` directory.

It reconstructs face data from the shipped binary64 Tucker coefficients at high precision and emits binary64 hexadecimal literals. Python/mpmath are generation-time tools only; they are not runtime dependencies.

Do not casually regenerate coefficient tables as part of an ordinary build. Table changes are numerical-source changes and require requalification.

## 14. Build

From the repository root:

```bash
make native
```

or directly:

```bash
make -C native/cflow
```

The production build baseline is `-O3` with C11 warnings enabled, floating-point contraction allowed, and `-fno-fast-math` retained.

Useful targets:

```bash
make -C native/cflow print-config
make -C native/cflow audit
make -C native/cflow clean
```

`audit` prints linkage and exported `cflow_*` symbols when the platform provides the relevant tools.

## 15. Runtime dependencies

Cflow is designed to remain small and relocatable with the rest of the native stack.

Runtime dependencies:

- C runtime;
- `libm`.

No Python, NumPy, SciPy, BLAS, or external ODE library is required by `libcflow` itself.

## 16. Relationship to the reverse-only eta kernel

[`native/reverse_eta`](../reverse_eta/README.md) contains a separate regularized tangent cocycle for a narrow class of expensive contracting/grazing reverse derivatives.

That library is **not** a second Cflow state authority:

- scalar topology/state are built by ordinary production Cflow;
- the eta kernel is considered only after scalar success;
- its dispatch is gated by a work predictor and qualified chart conditions;
- any failure/nonfinite result falls back to production derivatives.

Keeping it separate makes the numerical-authority boundary explicit.

## 17. Relationship to the rest of the native stack

```text
libcflow.so
    ▲
libame_segment.so
    ▲
libame_crossing.so
    ▲
libame_reverse.so
```

Cflow does not know about maze topology, corridor geometry, branch-and-bound, or optimizer state. It is deliberately a low-level normalized flow primitive.

That narrow boundary is part of what made the numerical implementation independently auditable and replaceable during development.

## 18. Qualification philosophy

Changes to Cflow are judged at more than the local kernel level.

A candidate may need to pass:

- scalar state parity;
- event/domain parity;
- Jacobian checks;
- integral/time checks;
- exact boundary-start cases;
- semigroup/cocycle composition checks;
- application-level reverse-gradient checks;
- full route/optimizer regressions;
- performance comparisons.

A locally more accurate or faster numerical method is not automatically a production improvement if it perturbs hybrid topology or worsens the complete optimizer workload.

## Related documentation

- [`../../docs/SPEED_PROFILE_SOLVER.md`](../../docs/SPEED_PROFILE_SOLVER.md)
- [`../../docs/PROJECT_HISTORY.md`](../../docs/PROJECT_HISTORY.md)
- [`../reverse/README.md`](../reverse/README.md)
- [`../reverse_eta/README.md`](../reverse_eta/README.md)
