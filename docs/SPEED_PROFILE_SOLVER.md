# Hybrid speed-profile solver

This document explains the time-model layer used by the continuous trajectory optimizer. It focuses on the **mathematical architecture**: spatial dynamics, forward/backward extremals, hybrid events, maximum-velocity curves, differentiation, and the separation between the legacy benchmark model and the Red Comet differential-drive/yaw model.

For the native implementation details, see [`native/cflow/README.md`](../native/cflow/README.md) and [`native/reverse/README.md`](../native/reverse/README.md).

<p align="center">
  <img src="../assets/red_comet/red_comet_astar_speed.svg" alt="Red Comet A-star speed and curvature profile" width="900">
</p>

## 1. Why use arc length and `w = v²`?

The geometry is already parameterized by path station `s`, so the longitudinal problem is written in arc length rather than clock time.

Define

\[
w(s)=v(s)^2.
\]

Because

\[
\frac{dv}{dt}=a,\qquad \frac{ds}{dt}=v,
\]

we obtain

\[
\frac{dw}{ds}=2a.
\]

This turns many longitudinal acceleration limits into first-order spatial dynamics and avoids carrying `v` through repeated square-root chain rules. Traversal time is recovered from

\[
T=\int_0^L \frac{ds}{\sqrt{w(s)}}.
\]

A fixed geometry therefore induces a one-dimensional hybrid optimal-control problem in `w(s)`, with curvature and curvature slope supplied by the clothoid path.

## 2. The envelope viewpoint

For a fixed path, the fastest feasible profile is constructed from reachable extremals rather than by placing an independent speed variable at every spatial sample.

Conceptually:

1. propagate the maximum reachable profile **forward** from the initial speed;
2. propagate the maximum profile that can still satisfy the terminal boundary **backward** from the goal;
3. add internal maximum-velocity-curve extremals where endpoint passes alone cannot cover the feasible envelope;
4. select the pointwise lower envelope of all relevant candidates.

This is the modern form of the very first solver in the project. The original implementation also used a forward/backward intersection, but each physical regime was stepped with low-order RK2 on a spatial mesh. The current solver makes mode switches and physical-domain boundaries explicit events.

## 3. Legacy MOTOR / GRIP / BRAKE model

The historically qualified `legacy_grid_v1` profile uses three active regimes.

### 3.1 BRAKE

With constant braking magnitude `A_brake`,

\[
\frac{dw}{ds}=-2A_{\mathrm{brake}}.
\]

The solution is linear in distance. Because clothoid curvature is also linear in `s`, several BRAKE crossing calculations reduce to algebraic/root-location problems rather than numerical ODE integration.

### 3.2 MOTOR

The motor-limited branch uses

\[
\frac{dw}{ds}=2\left(A-B\sqrt{w}\right),
\]

where `A` is the low-speed acceleration limit and `B` is the back-EMF coefficient (`A / v_max` in the frozen profile).

After substituting `y = sqrt(w)`, the equation is separable and can be inverted with the Lambert `W` function. Production segment code therefore propagates the MOTOR state analytically rather than numerically integrating it. The project includes a small purpose-built Lambert-`W` evaluator based on Halley iteration.

### 3.3 GRIP

The friction-circle branch is the nonlinear part. With linearly varying curvature

\[
\kappa(s)=\kappa_0+\sigma s,
\]

longitudinal acceleration is limited by the remaining grip after lateral demand. In the normalized Cflow variables

\[
X=\frac{w}{\mu g},\qquad q(s)=q_0+bs,
\]

the general branch is represented as

\[
\frac{dX}{ds}=2\sqrt{1-\left[(q_0+bs)X\right]^2}.
\]

This is the flow handled by the native Cflow kernel. Cflow also evaluates the additive travel-time observable

\[
J=\int \frac{ds}{\sqrt{X}},
\]

together with state/integral derivatives and first-domain-event information.

The path from RK2 to Cflow — including the abandoned Chebyshev and Taylor implementations — is described in [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md).

## 4. Regime switches are first-class events

A hybrid solver is only useful if transitions between active physical constraints are located consistently.

The legacy solver explicitly handles transitions such as:

```text
GRIP -> MOTOR
MOTOR -> GRIP
GRIP -> BRAKE
BRAKE -> GRIP
```

as well as piece ends and friction-domain boundaries.

The implementation does not simply sample until one mode happens to become numerically smaller than another. Crossing routines use analytic/algebraic structure where available and bracketed/refined root location where required. Event residuals, directional semantics, exact-boundary starts, and near-zero roots are handled explicitly because an incorrect switch can change the entire lower-envelope topology.

The scalar reverse builder records the actual emitted prefix length separately from the full candidate segment length. An event-truncated segment can therefore be represented without pretending the original candidate ended there.

## 5. Internal maximum-velocity-curve anchors

Forward and backward endpoint passes are not always sufficient.

A path may contain an **internal bottleneck** where the maximum admissible speed curve has a local minimum or where the admissible acceleration interval collapses. Such a point must seed additional reachable extremals; otherwise the lower envelope can contain an uncovered interval or incorrectly pass through an infeasible region.

The legacy solver therefore detects candidate internal friction-cap anchors and inserts forward/backward passes in batches until the candidate family forms a continuous valid envelope.

The Red Comet DD/yaw backend generalizes this idea further: actuator-side constraints can create maximum-velocity-curve anchors even when a simple friction cap would not.

## 6. Scalar topology construction

The production hot path deliberately separates **what the solution is** from **how its derivative is computed**.

A scalar build owns:

- the raw clothoid parameters;
- forward/backward traversal pieces;
- compiled scalar physical segments;
- crossing and domain decisions;
- inserted internal-cap/MVC candidate passes;
- the selected lower-envelope pieces;
- scalar time evaluation on that envelope.

This scalar topology is authoritative. A later gradient request is not allowed to rediscover a slightly different event topology and quietly differentiate that instead.

For the native implementation this object is `ame_scalar_build`; the Python reference solver follows the same conceptual split.

## 7. Differentiable promotion and replay

When the optimizer asks for a gradient at the same point, the retained scalar topology is **promoted**.

Time-only promotion compiles only the envelope-relevant replay prefixes needed for the time objective. Full promotion is available when final-state rows are required.

This matters because nonlinear optimizers often evaluate the scalar objective more frequently than the gradient. Rebuilding every possible differentiable pass on every line-search trial would spend most of the runtime on derivatives that are never consumed.

The final architecture therefore looks like:

```text
scalar raw geometry
      │
      ▼
physical event/topology discovery
      │
      ├── scalar time
      │
      └── differentiable promotion
               │
               ▼
          reverse sweep
```

The reverse sweep itself is local algebra once segment/prefix sensitivities and event derivatives have been assembled.

## 8. Differentiating through events

The time objective is smooth only while the active hybrid topology remains unchanged. Inside one smooth topology region, derivatives propagate through:

- segment state maps;
- segment time integrals;
- curvature/length parameter mappings;
- implicit crossing locations;
- envelope-used prefixes.

For a switching equation

\[
f(s,\theta)=0,
\]

the event location derivative is obtained from the implicit relation

\[
\frac{ds_*}{d\theta}
=-\frac{\partial f/\partial\theta}{\partial f/\partial s}
\]

when the directional root is regular.

Exact event ties and topology changes are genuinely nonsmooth. The benchmark suite therefore excludes coordinates classified as nonsmooth from its finite-difference accuracy aggregate rather than pretending every attempted coordinate should match a smooth derivative formula.

In the checked gradient campaign, Python and native reverse implementations agreed exactly at report precision. Five-point finite differences checked 58 smooth coordinates with a median relative error of approximately `4.04e-11`; eight nonsmooth/invalid coordinates were reported as excluded rather than successes.

## 9. One authoritative state trajectory

Cflow development produced a rule that now applies throughout the legacy speed solver:

> **There is one authoritative physical state trajectory.**

State derivatives, time integrals, integral derivatives, event analysis, conditioning diagnostics, and specialized accelerators may augment that trajectory, but they may not silently replace the endpoint state with a different approximation.

This is why Cflow distinguishes endpoint-state authority from auxiliary augmented/integral machinery, and why the reverse-only regularized tangent kernel is gated as a derivative accelerator rather than a second scalar solver.

It is also why the production reverse build borrows the scalar build instead of owning a separately reconstructed topology.

## 10. Prefix continuation, semigroup, and cocycle structure

Repeated optimizer evaluations often query many prefixes of the same GRIP flow. Restarting every request from the beginning would waste work and amplify numerical inconsistency.

The state flow obeys the semigroup composition principle

\[
\Phi_{h_1+h_2}(x)
=\Phi_{h_2}(\Phi_{h_1}(x)),
\]

while additive observables such as travel time obey the corresponding cocycle relation. Cflow and its callers exploit this structure for continuation/caching rather than treating each prefix as an unrelated integration problem.

This became especially important after the Taylor-series stage, where repeated recentering and prefix compilation had become a large fraction of objective cost.

## 11. Scalar time vs full value+gradient evaluation

The optimizer exposes both a scalar-only and differentiable evaluation path.

`optimization.scalar_reverse_solver.evaluate_time_scalar()` reuses the authoritative scalar topology builder and integrates the selected envelope without constructing differentiable replay segments or running a reverse sweep.

`optimization.reverse_solver.time_value_and_gradient()` (or the profile dispatcher) promotes/replays the retained topology and returns derivatives.

The distinction is visible in the native benchmark results: native scalar construction benefits more strongly from moving topology ownership out of Python, while differentiable replay gains less because much of the expensive GRIP work was already native.

## 12. Native and Python legacy backends

The packaged production default is the native reverse backend. The Python implementation remains selectable through

```bash
AME_REVERSE_BACKEND=python
```

or the process-local `using_reverse_backend("python")` context manager.

The reference backend is intentionally kept alive for:

- numerical A/B qualification;
- event/topology diagnostics;
- visualization traces that need introspection;
- regression tests;
- independent derivative checks.

The native backend is not a different mathematical method; it is a coarse-grained implementation of the same qualified scalar-build/promote/reverse architecture.

## 13. Red Comet differential-drive/yaw backend

The historical Red Comet case study uses a different profile-selected time model: `red_comet_2017_dd_yaw_v1`.

It still uses one spatial speed state,

\[
\frac{dw}{ds}=2a,
\]

but `a` must lie inside an acceleration interval generated by several competing constraints.

The forward candidates include:

- MOTOR;
- GRIP;
- SIDE_RIGHT;
- SIDE_LEFT.

The backward candidates include:

- BRAKE;
- GRIP;
- SIDE_RIGHT;
- SIDE_LEFT.

The side limits arise from aggregate left/right drivetrain authority. With path curvature `kappa`, curvature slope `sigma`, and yaw rate approximately `omega = v*kappa`, yaw acceleration contains a term proportional to

\[
a\kappa+w\sigma.
\]

Consequently, a path with aggressive curvature change can become actuator/yaw-limited even if a scalar friction-circle model would permit more longitudinal acceleration.

The DD/yaw solver detects internal actuator maximum-velocity-curve points, launches signed extremals from those anchors, and merges them with the endpoint passes. The checked release snapshot for the final Red Comet A* route contains 124 active anchors and 250 candidate passes before envelope selection.

The Python implementation is the independent reference; the native DD/yaw kernel is separately qualified and profile-gated so it cannot alter the legacy benchmark model.

## 14. Dynamic certification

A speed solver returning a finite time is not enough for a headline result.

The dynamic certificate checks the model-specific physical profile and, where applicable, performs independent implementation replay.

For the final Red Comet A* geometry:

- Python DD/yaw reference: `7.7446749805802995 s`;
- native DD/yaw: `7.7446749806008865 s`;
- difference: `2.06e-11 s`;
- case-study tolerance: `2e-8 s`.

The release certificate also records minimum speed, acceleration-interval margin, MVC violations, anchor counts, and model identity/source hashes.

## 15. Independent fixed-geometry transcription benchmark

Benchmark 7 asks whether the specialized hybrid legacy speed solver agrees with a much more generic dense formulation on **the same fixed geometry**.

The comparison uses CasADi/IPOPT with increasingly fine speed meshes and then continuously recertifies each discrete solution. At the finest predeclared 1024-interval mesh:

- 3/3 cold-start cases passed continuous certification;
- the median absolute time difference from the production solver was about **0.092%**;
- the median IPOPT/production scalar solve-time ratio was about **934.9×**.

This benchmark supports the numerical correctness and specialization value of the speed solver. It is intentionally narrower than the simultaneous geometry+speed OCP comparison in [`OCP_COMPARISON.md`](OCP_COMPARISON.md).

## 16. What this solver does not prove

The fixed-geometry envelope construction is highly structured, but the complete planner still contains a nonconvex outer geometry optimization.

Therefore:

- agreement between Python/native speed implementations does not prove the geometry is globally optimal;
- a certified speed profile proves feasibility for the selected geometry/model, not global optimality over all paths;
- hybrid event boundaries create genuine nonsmooth regions;
- the Red Comet DD/yaw model is a calibrated case-study model, not a full race-day vehicle reconstruction.

Those claim boundaries are important when interpreting both the Red Comet result and the independent simultaneous-OCP comparison.

## Related documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — end-to-end planner design
- [`GEOMETRY_OPTIMIZATION.md`](GEOMETRY_OPTIMIZATION.md) — outer clothoid/corridor NLP
- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — RK2 → Chebyshev → Taylor → Cflow development history
- [`native/cflow/README.md`](../native/cflow/README.md) — Cflow implementation and ABI
- [`native/reverse/README.md`](../native/reverse/README.md) — native scalar-build/promote/reverse ownership
- [`BENCHMARKS.md`](../BENCHMARKS.md) — evaluation campaign
