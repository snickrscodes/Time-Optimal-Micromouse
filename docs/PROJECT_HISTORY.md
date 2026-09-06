# Project history: from RK2 maze solver to structured optimal control

The current planner was not designed as one coherent architecture from the beginning. It grew through a long sequence of models, numerical methods, optimizer experiments, performance rewrites, and qualification campaigns.

This document preserves that progression — including substantial approaches that were eventually removed from production — because many of the final design choices only make sense in light of what failed before them.

## Origin: the Micromouse question that started the project

The project began during a senior-year high-school engineering project embedded in an integrated mathematics/engineering class. The team chose to build a maze-solving robot inspired by Micromouse, and the software problem became the part that continued long after the class project itself ended.

The specific motivation was the Red Comet 2017 story popularized by a Veritasium video: the visibly shortest maze route was not necessarily the fastest racing route. A longer path could trade distance for smoother turns and longer high-speed sections.

That suggested a more interesting question than ordinary shortest-path search:

> **If a planner were given only maze geometry and vehicle dynamics, could racing-line behavior emerge from minimum-time optimization rather than being hard-coded as motion primitives?**

The classroom project eventually disappeared and no physical robot was completed. The algorithm continued independently because the mathematical problem was more interesting than the original embedded implementation. The codebase eventually became far larger and more computationally ambitious than an onboard Micromouse controller would justify.

What survived from the original motivation was the desire for **one general physical objective** that could produce corner cutting, braking, acceleration, and route choice without separately programming each behavior.

---

## Condensed evolution

```text
maze graph / shortest-path prototypes
        ↓
continuous clothoid geometry + w = v² spatial dynamics
        ↓
RK2 forward/backward speed envelopes
        ↓
explicit MOTOR / GRIP / BRAKE hybrid regimes
        ↓
closed-form BRAKE + Lambert-W MOTOR propagation
        ↓
Chebyshev surrogate for nonlinear GRIP flow
        ↓
local high-order Taylor GRIP compiler
        ↓
compiled Cflow state/integral/Jacobian kernel
        ↓
scalar topology discovery + differentiable reverse replay
        ↓
continuous corridor constraints + independent certification
        ↓
SLSQP / trust-constr / sparse-filter-SQP / Hessian research
        ↓
reduced basis + curvature homotopy + selective active basis
        ↓
native Segment / Crossing / Reverse stack
        ↓
branch-and-bound topology search + benchmark qualification
        ↓
custom historical mazes + Red Comet DD/yaw model
        ↓
Red Comet exhaustive case study
        ↓
independent simultaneous-OCP comparison
```

The development was not perfectly linear. Geometry, dynamics, search, native acceleration, and optimizer work overlapped. The sequence above is the clearest **causal** history: each stage solved a failure exposed by the previous one.

---

# Part I — The speed solver

## 1. The first continuous formulation: forward/backward RK2

The earliest useful continuous solver already contained the core idea that remains today.

For a fixed geometric path, it propagated the maximum reachable speed forward from the start and backward from the goal, then intersected the two profiles. The spatial state was

\[
w=v^2,
\]

with traversal time

\[
T=\int \frac{ds}{\sqrt{w(s)}}.
\]

But the physical dynamics were evaluated with a low-order RK2 discretization across all active regimes. MOTOR-, GRIP-, and BRAKE-limited sections were sampled on a spatial mesh, and mode changes were discovered by stepping until a constraint changed sign.

That prototype demonstrated that the forward/backward envelope formulation worked, but it also exposed the weaknesses of putting a low-order integrator inside a nonlinear optimizer:

- time changed noticeably with spatial resolution;
- switch locations moved with the mesh;
- gradients inherited integration noise;
- every objective call performed many tiny steps;
- analytically solvable branches were being integrated numerically for no reason.

The first durable lesson of the project was therefore:

> **Do not numerically integrate structure that can be solved exactly.**

## 2. Explicit hybrid regimes

The speed model was reorganized as a true hybrid system with explicit MOTOR / GRIP / BRAKE modes.

For constant braking,

\[
\frac{dw}{ds}=-2A_{\mathrm{brake}},
\]

so BRAKE propagation is exactly linear in distance.

The MOTOR branch is

\[
\frac{dw}{ds}=2\left(A-B\sqrt{w}\right).
\]

After substituting `y = sqrt(w)`, the equation is separable and invertible with Lambert `W`. A custom Halley-based Lambert-`W` implementation turned MOTOR into an analytic propagation primitive as well.

Mode transitions became explicit root/crossing problems instead of “take another RK2 step and inspect the minimum constraint.”

After this change, numerical integration remained necessary only for the genuinely nonlinear GRIP branch.

## 3. Chebyshev GRIP surrogate

GRIP quickly became the dominant numerical problem.

The first serious replacement for RK2 was an offline Chebyshev approximation. The idea was attractive:

1. nondimensionalize the linearly varying-curvature GRIP ODE;
2. solve it accurately over a canonical parameter region;
3. fit a compact Chebyshev representation;
4. evaluate that representation cheaply inside the optimizer.

This was not a toy experiment. The Chebyshev implementation progressed far enough to support the gradient factory and inner speed-profile solver.

The fatal problem was the chosen nondimensionalization. Its coordinates depended too strongly on curvature slope. As the slope parameter `b → 0`, the transformed system became badly behaved even though the physical GRIP problem smoothly approaches a regular constant-curvature limit.

A global fitted representation also became awkward around:

- exact friction-boundary starts;
- event location;
- parameter sensitivities;
- separatrix-like behavior;
- coordinate regions that were numerically difficult for reasons unrelated to the original physical ODE.

The lesson was more important than the lost implementation:

> **Numerical coordinates should become singular only where the physical problem is singular — not because a convenient global fit chose a fragile scaling.**

## 4. Local high-order Taylor flow maps

The next solver abandoned a single global approximation and moved to local high-order Taylor series.

Coefficients were generated efficiently through recurrences, and a GRIP interval was covered with adaptively sized Taylor pieces. The implementation eventually supported:

- state propagation;
- state sensitivities;
- travel-time integrals;
- integral sensitivities;
- exact friction-boundary continuation;
- domain/event detection;
- composition of local pieces;
- recentering/splitting when one local series no longer covered the requested prefix.

This was a major correctness improvement and proved that the GRIP flow could be differentiated accurately enough for outer geometry optimization.

But it created a new problem: **software and runtime complexity**.

One physical GRIP segment could become a tree of compiled Taylor pieces. Difficult segments could exhaust piece budgets and trigger recursive recentering. Scalar topology discovery and differentiable replay both needed to understand Taylor-specific failure modes. Python-level cache/compilation logic became a large share of objective cost.

The Taylor stage therefore produced another architectural insight:

> **The optimizer should see one GRIP segment abstraction, not the machinery required to numerically construct its local flow.**

## 5. Cflow

Cflow extracted the normalized GRIP initial-value problem into its own numerical kernel.

In the final normalized variables,

\[
\frac{dX}{ds}=2\sqrt{1-[(q_0+bs)X]^2}.
\]

The C runtime exposes state evaluation, state Jacobians, time integrals, integral Jacobians, first-event queries, exact inward-boundary continuation, and conditioning/status information.

The important change was not merely “rewrite Taylor in C.” The numerical architecture itself was redesigned around several principles:

- direct mapping from physical variables without reintroducing the failed global nondimensionalization;
- specialized numerical charts/approximations only where they improved the physical flow evaluation;
- semigroup-consistent continuation;
- cocycle-consistent additive observables;
- explicit first-domain-event authority;
- a single authoritative endpoint state.

Once Cflow passed application-level value, Jacobian, integral, event, boundary-start, gradient, and performance gates, the runtime Taylor package was removed from production.

Later Cflow work continued far beyond the first port — high-`z` tensor specialization, boundary maps, conditioning analysis, separatrix policy, face-specialized Chebyshev/Tucker evaluation, and regularized derivative experiments — but the public abstraction remained compact.

## 6. “One authoritative state trajectory”

A particularly important numerical rule emerged during Cflow qualification:

> **There is one authoritative physical state trajectory.**

It is easy for an optimized numerical implementation to accidentally evaluate state, derivative, integral, and event channels with slightly different approximations. On a hybrid solver, those low-bit differences can change switch topology or local optimizer basin.

The final design therefore allows derivative/integral machinery to augment the state trajectory but not silently replace it.

This rule later shaped:

- Cflow state vs integral authority;
- reverse-only derivative acceleration;
- scalar topology retention;
- native/Python cross-checks;
- case-study model hashing and certification.

## 7. The reverse solver became a real solver

As the segment primitives stabilized, the original “two passes and intersect them” implementation grew into an explicit hybrid-profile engine.

The scalar layer now:

- constructs forward/backward extremals;
- locates mode switches and physical-domain events;
- inserts internal maximum-velocity-curve anchors when endpoint passes are insufficient;
- assembles the selected lower envelope;
- evaluates scalar travel time from that envelope.

Then a second layer **promotes** the retained scalar topology only when derivatives are requested.

This scalar-discovery/differentiable-replay split was important for both correctness and performance. Line searches could perform cheap scalar evaluations without compiling every derivative object, while gradient calls reused the exact same hybrid topology instead of rediscovering a nearby one.

Crossing sensitivities became implicit event derivatives rather than finite differences.

---

# Part II — The continuous path problem

## 8. From maze cells to clothoid geometry

In parallel with the dynamics work, the route representation evolved from a cell-center path into a true continuous trajectory problem.

Piecewise Euler spirals / clothoids became the geometry basis:

\[
\kappa(s)=\kappa_0+\sigma s.
\]

The path was propagated through exact/numerically careful clothoid geometry rather than a dense polyline approximation.

The robot also stopped being treated as a point. A rectangular body footprint was checked against the corridor induced by the route.

That immediately made constraint handling much more important.

## 9. Log-length coordinates

The original knot description used cumulative stations. That gives the optimizer an ordering problem: every knot station must remain greater than the previous one.

Production moved to positive log-length coordinates,

\[
L_i=L_{i,\mathrm{ref}}e^{z_i},
\]

so segment positivity and ordering became structural.

This did not eliminate collapsed/poorly conditioned segments, but it turned those into explicit conditioning/curvature-slope questions rather than station-ordering failures.

## 10. Continuous corridor authority and constraint generation

Finite sampled wall constraints are useful to an NLP, but they are not sufficient as a physical guarantee.

The optimizer therefore developed:

- convex corridor representations;
- exact geometry Jacobians;
- endpoint equalities;
- a finite active constraint pool;
- continuous violation search;
- constraint generation/exchange;
- independent final geometry certification.

The durable rule became:

> **Finite constraint pool for optimization; continuous oracle for authority.**

That separation is now central to the release semantics: optimizer success is not considered proof of feasibility.

---

# Part III — Optimizer research

## 11. SLSQP as the first serious workhorse

Dense SciPy SLSQP became the original main nonlinear solver because its framework overhead was extremely small and it behaved surprisingly well at the modest problem sizes of the early trajectory representation.

Its weakness was the line search. A single accepted major iteration could require several expensive scalar speed-profile builds.

As the speed solver itself became faster, optimizer callback economics became the next bottleneck.

## 12. `trust-constr`

SciPy `trust-constr` was explored as a way to reduce repeated objective evaluations and use a more explicit constrained trust-region architecture.

On some fixed-active-set problems it needed fewer expensive objective calls. But it also introduced higher framework/linear-algebra overhead and was less reliable on several signed/multi-event or degeneracy-sensitive cases.

The result was not “trust regions are bad.” It was narrower:

> A generic backend that looks better by iteration count does not necessarily reduce end-to-end wall time or preserve the same basin/certification behavior.

## 13. Sparse/filter SQP

The project then went much deeper and built custom sparse/filter-SQP machinery around HiGHS subproblems.

That research included:

- sparse active constraint representations;
- filter globalization;
- quasi-Newton updates;
- restoration/Phase-I paths;
- persistent continuation state;
- independent QP checking;
- bounded accepted-step batches;
- supervised worker processes;
- deterministic incumbent/checkpoint recovery;
- planner-level promotion/shadow policy.

On representative small problems it dramatically reduced scalar-build counts. One qualification seed dropped from 181 scalar builds under SLSQP to 63 under filter-SQP and returned a slightly better certified local result.

But broad promotion exposed the real difficulty: restoration, changing active sets, degeneracy/rank behavior, deployment portability, and failure semantics mattered as much as the local QP solves.

The final result is deliberately conservative. Filter-SQP is retained as a **qualified narrow primary backend** for the small-problem envelope where it was demonstrated, with SLSQP fallback outside that envelope. Sparse-specialist machinery remains supervised and fail-closed.

## 14. Hessian research

A separate campaign asked whether the remaining optimizer cost was fundamentally caused by missing second-order information.

Oracle/finite-difference high-quality Lagrangian curvature was used to estimate the ceiling before implementing a large analytic-Hessian or Hessian-vector-product system.

The result did not justify that effort. Active-set changes, cut generation, near-collapsed segments, and representation structure remained major sources of work/basin behavior even when more accurate curvature information was available.

This was an important negative result because it prevented a large amount of mathematically impressive but weakly justified implementation work.

## 15. Stop asking the solver to fix the representation

The biggest optimizer-side change came from reframing the problem.

Instead of asking “which NLP solver can survive the giant full corridor basis?”, the project began asking “why expose all of those degrees of freedom at once?”

The result was the active-basis architecture.

---

# Part IV — Active basis and qualification

## 16. Reduced maximal-run basis

The final route optimizer starts with a lower-dimensional **maximal-run corridor basis**.

This basis is easier to initialize and condition. It intentionally suppresses some local turn freedom until the current trajectory demonstrates that the extra freedom can matter.

## 17. Curvature homotopy

The reduced basis is first optimized through a curvature-oriented continuation with the frozen guard sequence

\[
0.01\rightarrow0.005\rightarrow0.0025.
\]

The purpose is to establish a stable geometric basin before direct time minimization dominates.

## 18. Reduced time transactions

The reduced trajectory is then optimized for traversal time through bounded transaction/trust schedules. Certified incumbents are checkpointed throughout.

Convergence is defined by exhaustion of the qualified schedule semantics, not by one generic optimizer message.

## 19. Selective turn-pair activation

After reduced convergence, the optimizer analyzes exact support around individual turns.

Additional full-basis children are activated only for coherent turn pairs that can be initialized above a conditioning threshold. The branch then optimizes behind a decreasing segment-length floor:

\[
7.5\times10^{-4}
\rightarrow7.0\times10^{-4}
\rightarrow6.5\times10^{-4}
\rightarrow6.0\times10^{-4}.
\]

Previously inactive turns can be reanalyzed after the geometry changes.

The reduced certified solution is always retained, so a failed expanded branch cannot erase a valid incumbent.

## 20. Historical-five qualification

The active-basis state machine was not promoted from one convenient example. It was exercised and frozen against a five-route historical qualification corpus.

That campaign drove several production requirements:

- explicit curvature→time switching criteria;
- transaction/trust policy;
- reduced→full promotion rules;
- append-only convergence logging;
- deterministic state-machine semantics;
- worker failure containment;
- single-threaded execution for qualification reproducibility.

The current code still treats those policies as frozen architecture rather than casual tuning knobs.

---

# Part V — Moving the hot path to C/C++

## 21. Segment and crossing ports

Once the mathematical structure stabilized, profiling showed that Python orchestration still controlled thousands of small segment/crossing/envelope operations.

Native work therefore moved upward in layers:

1. Cflow numerical GRIP kernel;
2. native segment dynamics;
3. native crossing/event calculations;
4. complete native scalar reverse construction;
5. native differentiable promotion/reverse sweep.

Each stage was qualified against the previous implementation before becoming production authority.

## 22. Native reverse ownership

The most important native architectural step was not simply translating the Python reverse file.

The C++ layer gained explicit lifetime semantics:

- `ame_scalar_build` owns physical topology and envelope;
- `ame_reverse_build` is a promotion of that scalar build;
- promoted handles borrow the scalar authority;
- Python crosses the FFI boundary once per coarse operation, not once per segment.

That removed large amounts of Python object/control-flow overhead while preserving the same numerical architecture.

Microbenchmarks on cheap analytic profiles showed very large orchestration speedups, but the project reports the more conservative end-to-end benchmark separately: about 2.18× median scalar speedup, 1.71× time+gradient speedup, and 1.18× median representative complete-optimizer speedup.

This distinction became another recurring rule:

> **Do not market a microkernel speedup as a planner speedup.**

## 23. Regularized reverse-only derivative work

Difficult near-grazing Cflow derivatives motivated a separate regularized tangent representation (`eta = tan(delta)`).

The best reverse-only architecture could dramatically accelerate qualified pathological tangent work. A more self-contained value+gradient variant was also investigated but repeated scalar work that the retained reverse topology had already performed, making it slower in the current architecture.

The production result is therefore narrowly gated: the reverse-only eta kernel is used only when a work predictor and contracting-chart conditions justify it; otherwise production Cflow derivatives remain authoritative.

---

# Part VI — Returning to topology search

## 24. Why the search ended up as branch-and-bound

The project began as a maze-search project, but serious topology selection had to wait until a fixed route could be optimized and certified consistently.

An attractive intermediate idea was a Pareto-frontier A* that would retain multiple prefixes with different time/distance/arrival-speed tradeoffs.

The problem is that ordinary scalar Pareto dominance is not generally safe here. The future value of a prefix depends on a continuous boundary state. The correct object is an **arrival-cost function** over that state:

\[
A_\pi(z).
\]

Prefix 1 safely dominates prefix 2 only if

\[
A_1(z)\le A_2(z)
\qquad\text{for every future-relevant }z.
\]

Two such functions can cross. A prefix that is worse at one arrival speed may be uniquely better at another.

Without a rigorous cheap containment/dominance test, an ordinary Pareto rule could prune the true optimum.

The project therefore chose the less glamorous but defensible option: **finite branch-and-bound** with conservative lower bounds, reachability, block-cut structure, topology quotienting where certified, and a bounded visit policy.

The result has worse worst-case combinatorics but a clear correctness story over the declared finite topology set.

## 25. Benchmarking topology search

Once topology search returned, it was benchmarked independently from the continuous solver.

The six-case cyclic benchmark set found:

- all A*/B&B pairs continuously certified;
- B&B selected a different topology in 2/6 cases;
- maximum certified time improvement: 27.09%.

On three small cases where exhaustive simple-path enumeration was tractable, B&B matched the exhaustive winner 3/3 times and avoided 50% median of complete route optimizations.

That result justified keeping topology search as a real layer instead of assuming shortest distance would always survive continuous optimization.

---

# Part VII — Benchmarking as a first-class subsystem

## 26. Why a benchmark framework was added

By this stage the planner had accumulated enough specialized machinery that “it runs on my example” was no longer meaningful evidence.

A separate benchmark package was created around production APIs. The production code does not import benchmark code.

The official campaign tests different claims separately:

1. topology selection;
2. lower-bound correctness/usefulness;
3. derivatives;
4. warm starts;
5. native equivalence/performance;
6. trajectory resolution;
7. fixed-geometry direct transcription;
8. full simultaneous geometry+speed OCP.

The campaign also formalized process isolation, deterministic cases, certification-only quality aggregates, machine-readable JSON results, pinned reference dependencies, and generated reports.

This changed the project substantially: optimization and numerical-method claims now had to survive a reproducible independent experiment rather than just local debugging.

---

# Part VIII — Historical mazes and Red Comet

## 27. Custom maze format

The final research stage required real historical Micromouse layouts rather than generated small mazes.

A compact `ame-maze-v1` format was introduced with:

- exact binary wall topology;
- explicit coordinate convention;
- physical scale metadata;
- all semantic goal cells;
- derived single-entry planning goal;
- optional start heading validation;
- source-file hash for reproducibility.

Three historical/reference mazes were transcribed and wall-audited.

## 28. Red Comet required a different physical model

The old benchmark constants were never meant to reconstruct Red Comet hardware. Using them for the historical comparison would answer the wrong question.

A separate calibration therefore used published/cross-year vehicle evidence to define a Red-Comet-specific physical profile. The scalar calibration included:

- 180 mm cell pitch;
- 76 × 45 mm body;
- 30.2 g mass;
- published straight-speed evidence;
- published turn-speed range mapped to an effective grip bound;
- a same-vehicle longitudinal-acceleration proxy from nearby-year technical data.

The historical winning race time was **not** used as a fitting target.

A subsequent `red_comet_2017_dd_yaw_v1` model added aggregate left/right drivetrain authority, motor torque-speed limits, effective track width, and yaw inertia while preserving one spatial speed state.

This model is explicitly a case-study approximation, not an instrumented reconstruction of race-day controller/tire/suction/battery behavior.

## 29. A calibration-specific native bug changed the qualification philosophy

During the calibrated-native campaign, an important correctness bug was discovered.

The native GRIP state equations were rebuilt with the Red Comet `MU_G`, but one derived inverse-square-root scale used by the travel-time integral still contained the legacy constant. State propagation matched Python while GRIP time was wrong by a deterministic scale factor.

The fix derived the scale directly from the active `MU_G` and added a profile-relative post-build Python/native state+time differential gate.

The episode reinforced an important project rule:

> **A parameterized native build must qualify derived constants and observables, not merely its main state equation.**

No optimizer/active-basis policy was changed as part of the fix.

## 30. Red Comet exhaustive experiment

The final Red Comet campaign proceeded in controlled stages:

1. verify maze transcription;
2. verify the historical route overlay;
3. optimize A* and historical routes as fixed-topology controls;
4. run topology/search preflight;
5. exhaustively search the ten simple topologies admitted by the production no-revisit policy;
6. investigate a better A* local basin exposed by continuation history;
7. close/certify the final A* incumbent;
8. perform a final overlay audit that corrected a small prefix transcription error in the historical route;
9. package compact result data and release visuals.

Final frozen-model result:

| Route | Grid steps | Optimized time |
|---|---:|---:|
| A* shortest topology | 99 | **7.744675 s** |
| nearest alternative | 103 | 8.176914 s |
| corrected historical Red Comet route | 121 | **8.666065 s** |

The result was the opposite of the motivating expectation: under this model, the shortest topology is decisively faster than the historical line.

The benchmark suite still contains cases where a longer/non-A* topology wins, so the conclusion is not “shortest path always equals fastest path.” It is subtler: **the physical advantage of a longer topology has to be earned by the actual geometry/dynamics; it does not follow automatically from having fewer turns or longer straights.**

The case study is documented in [`RED_COMET_CASE_STUDY.md`](RED_COMET_CASE_STUDY.md).

## 31. Red Comet also exposed a weak lower bound

The small benchmark suite showed useful B&B pruning, but Red Comet exposed the opposite regime.

The production lower bound was too conservative to prune a complete topology. The search exhausted all ten routes; most wall time was spent in continuous leaf optimization rather than graph expansion.

That is an important limitation of the final discrete layer and a clear future research direction: stronger admissible kinodynamic bounds could reduce expensive complete-route solves without weakening the search guarantee.

---

# Part IX — Independent simultaneous OCP

## 32. Why an OCP comparison was necessary

The structured planner is highly specialized. Its speed solver is event-driven and analytic/native where possible, and the outer geometry representation has many hand-designed continuation policies.

That raises a natural validation question:

> **Does an independent generic simultaneous formulation discover the same fixed-topology local solution?**

A CasADi/IPOPT benchmark was therefore added that optimizes geometry and squared-speed variables together.

## 33. The first comparison exposed a fairness problem

The original structured reference used only 11 clothoid segments. A much denser simultaneous OCP could find a lower objective, but that result alone could be explained by trajectory representation resolution rather than optimization basin.

A stronger control was therefore added after the initial campaign.

## 34. Multilevel structured resolution control

The certified 11-segment structured solution was exactly prolonged and reoptimized through

```text
11 → 22 → 44 → 99 clothoid segments
```

while retaining the previous certified curve as an incumbent.

The structured time improved from

```text
1.171355379 s  (11 segments)
```

to

```text
1.153401215 s  (44 segments)
1.153401215 s  (99 segments)
```

The exact 44→99 lift preserves time to floating-point precision, and the first 99-segment local transaction changes time by only about `1.4e-10 s`.

So insufficient clothoid resolution is no longer a convincing explanation for the remaining difference.

## 35. OCP finds a different geometry basin

The finest 96-interval simultaneous OCP reports

```text
1.137372894 s
```

which is 1.390% below the 99-segment structured control.

More importantly, when the **OCP-discovered geometry** is evaluated with the production continuous hybrid speed solver, the time is

```text
1.131273794 s
```

or 1.918% below the refined structured result.

That means the difference is not merely the OCP’s discretized speed objective. The production speed solver itself prefers the geometry found by the independent formulation.

The strongest interpretation is therefore **different nonconvex local geometry basins / formulation behavior**.

Neither solver is claimed globally optimal.

See [`OCP_COMPARISON.md`](OCP_COMPARISON.md).

---

# What survived all of these iterations

Several principles recur throughout the project.

## Exploit structure before brute force

RK2 disappeared from analytically solvable regimes. A global fitted GRIP surrogate gave way to local flow machinery. A giant full-basis NLP gave way to staged basis activation.

## Keep numerical authority explicit

State, derivative, integral, event topology, optimizer termination, and physical certification are related but not interchangeable.

## Preserve an independent reference

Native implementations were only promoted after parity/gradient/certification gates. Python remains useful as a reference even when it is not the fastest production path.

## Measure whole-system economics

Several locally faster or more accurate ideas were rejected because they did not improve the real optimizer workload enough to justify their complexity.

## Failed approaches are evidence

Chebyshev scaling failure, Taylor piece complexity, sparse-SQP restoration problems, Hessian limits, resolution sensitivity, native low-bit effects, and the weak Red Comet lower bound all directly influenced the final architecture.

## Certification is separate from optimization

A solver status is never the physical certificate. This became increasingly important as the system grew more complex.

## Continuous optimality remains local

The project can make stronger claims about its finite discrete search than about its nonconvex continuous trajectory solve. The independent OCP comparison makes that limitation visible rather than hiding it.

---

# Production vs research-history components

| Component / idea | Final status |
|---|---|
| RK2 speed integration | removed |
| closed-form BRAKE | production |
| Lambert-W MOTOR | production |
| global Chebyshev GRIP surrogate | removed |
| runtime Taylor GRIP compiler | removed |
| Cflow | production |
| scalar topology + differentiable replay | production |
| SLSQP finite solver | production/fallback |
| `trust-constr` trajectory backend | research only |
| sparse/filter SQP | narrowly qualified production policy + research machinery |
| oracle/exact-Hessian campaign | research; not promoted |
| full dense corridor basis from initialization | replaced by reduced/selective architecture |
| active-basis continuation | production |
| Python reverse backend | reference/qualification |
| native Segment/Crossing/Reverse | production |
| reverse-only eta tangent kernel | narrowly gated production accelerator |
| Pareto-frontier topology A* | abandoned as unsafe without functional dominance |
| finite B&B topology search | production |
| generic direct transcription | benchmark only |
| simultaneous CasADi/IPOPT OCP | benchmark only |
| Red Comet DD/yaw model | case-study production profile |

---

# Related documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — current production pipeline
- [`SPEED_PROFILE_SOLVER.md`](SPEED_PROFILE_SOLVER.md) — current hybrid time solver
- [`GEOMETRY_OPTIMIZATION.md`](GEOMETRY_OPTIMIZATION.md) — current continuous optimizer
- [`RED_COMET_CASE_STUDY.md`](RED_COMET_CASE_STUDY.md) — historical case-study result
- [`OCP_COMPARISON.md`](OCP_COMPARISON.md) — independent formulation comparison
- [`../BENCHMARKS.md`](../BENCHMARKS.md) — measured evaluation campaign
