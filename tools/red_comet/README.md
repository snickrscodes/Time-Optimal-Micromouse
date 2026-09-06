# Red Comet campaign tools

`tools/red_comet/` contains the specialized orchestration used to qualify and reproduce the Red Comet case study. Scientific interpretation and final results live in [`docs/RED_COMET_CASE_STUDY.md`](../../docs/RED_COMET_CASE_STUDY.md); this file is only an operator reference.

The commands are deliberately separated by cost and purpose.

## 1. Cheap topology/search preflight

```bash
python -m tools.red_comet.preflight \
  --maze examples/mazes/historical/red_comet_reference_maze.json \
  --output-dir analysis/red_comet_preflight
```

This performs **no continuous trajectory optimization**.

It:

- loads the historical maze;
- builds the production compressed junction graph;
- applies block-cut relevant-subgraph reduction;
- exhaustively enumerates simple start-to-goal junction paths under the same no-revisit policy;
- records distance/turn/run statistics;
- evaluates production lower bounds for complete topologies;
- performs dry B&B experiments against synthetic incumbents for bound-analysis purposes;
- optionally renders a topology gallery.

Useful options:

```text
--maximum-paths N
--no-gallery
--body-length ...
--body-width ...
--no-topology-quotient
```

The synthetic incumbent used by lower-bound analysis is only an evaluation trigger and is never reported as a trajectory result.

## 2. Controlled fixed-route comparison

```bash
python -m tools.red_comet.fixed_route_compare \
  --maze examples/mazes/historical/red_comet_reference_maze.json \
  --physics-profile red_comet_2017_dd_yaw_v1 \
  --output analysis/red_comet_calibration/fixed_route_comparison.json
```

This is intentionally **not** branch-and-bound. It optimizes only:

- production A* topology;
- corrected historical Red Comet topology.

The two routes use the same physical profile, body dimensions, endpoint conditions, optimizer architecture, and certification policy.

By default the tool prepares an isolated native-profile sandbox so calibrated native constants can be rebuilt without mutating the checked-in legacy-qualified native libraries.

A slower Python/reference path can be selected for differential/debug use through the command's backend option.

Useful controls include:

```text
--iterations ...
--seconds-per-route ...
--init-w ...
--terminal-w-max ...
--keep-native-sandbox
```

## 3. Resumable held-out campaign

```bash
python -m tools.red_comet.final_run \
  --maze examples/mazes/historical/red_comet_reference_maze.json \
  --output-dir analysis/red_comet_final
```

This is the expensive case-study orchestration.

High-level stages:

1. build matching calibrated native kernels in an isolated worktree;
2. solve/certify A* and historical fixed routes;
3. record the fixed-route diagnostic under the frozen model;
4. unless requested otherwise, launch blind branch-and-bound without seeding the historical route into the search;
5. reuse persistent active-basis route work by request/model digest when restarted.

Useful modes:

```text
--fixed-only    stop after the two controlled fixed routes
--bnb-only      reuse an existing compatible fixed-route result and run only B&B
```

The route cache lives below the chosen output directory rather than inside the temporary native sandbox, so restarting the command can reuse completed expensive route solves.

## 4. Release visualization

The final public figures do not require any of the expensive commands above.

```bash
python -m tools.visuals.red_comet_release
```

This reads the compact checked-in release snapshot:

```text
analysis/red_comet_2017/final_result.json
```

and regenerates the run-derived assets under `assets/red_comet/`.

See [`visualization/README.md`](../../visualization/README.md) for the rendering boundary and output list.

## 5. Physics/profile isolation

Red Comet uses the profile-selected `red_comet_2017_dd_yaw_v1` time model, while the repository's historical benchmark corpus uses `legacy_grid_v1`.

Campaign tooling must not silently mix a Python profile selection with native libraries built for another profile. The fixed/final tools therefore isolate calibrated native builds and check model identity/signatures.

This separation was strengthened after the calibration campaign exposed a derived GRIP-time scale that had remained tied to the legacy constant even though the main GRIP state equation had been rebuilt correctly.

## 6. Expensive outputs vs public release artifacts

Raw campaign work directories can contain:

- native build sandboxes/logs;
- active-basis checkpoints;
- per-route worker records;
- B&B traces;
- continuation state;
- supervisor logs.

Those are development/recovery artifacts and are intentionally not required by the public repository snapshot.

The compact release keeps only the final route/certificate/search information needed for public figures and reported comparisons.

## Related documentation

- [`../../docs/RED_COMET_CASE_STUDY.md`](../../docs/RED_COMET_CASE_STUDY.md) — scientific result/provenance
- [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — planner architecture
- [`../../examples/mazes/FORMAT.md`](../../examples/mazes/FORMAT.md) — maze format
- [`../../visualization/README.md`](../../visualization/README.md) — release rendering
