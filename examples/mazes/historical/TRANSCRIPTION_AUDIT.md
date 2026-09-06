# Historical/custom maze transcription audit

These three `ame-maze-v1` files were transcribed from the three reference PNGs supplied with the project discussion and then independently checked before being admitted as example inputs.

| File | Reachable cells | Unique goal entrance (source coordinates) | Derived canonical goal | Source JSON SHA-256 |
|---|---:|---|---|---|
| `micromouse_maze_01.json` | 256/256 | `[9, 7] -> [8, 7]` | `[8, 7]` | `ddb08e772adf2a4853c124e52c705910bab4ce9e318ffcdaaa73f6cb96b688c8` |
| `micromouse_maze_02.json` | 256/256 | `[7, 9] -> [7, 8]` | `[7, 8]` | `f7d4ac337b8f8983feba1190c25e36e136fa1a18a042b98200469577e2faea13` |
| `red_comet_reference_maze.json` | 256/256 | `[7, 9] -> [7, 8]` | `[7, 8]` | `0ac12673d57941c34bdd8c2f06be4dd6919adaae864499b4cbccbdff30d611fb` |

## Structural loader validation

Every file passes the production `planning.maze_io.load_maze_scenario` checks:

- 16x16 dimensions;
- 17 horizontal rows of 16 binary characters;
- 16 vertical rows of 17 binary characters;
- all exterior boundaries closed;
- southwest source coordinates normalized deterministically to the planner's northwest convention;
- start cell `[0,0]` has exactly one opening, north to `[0,1]`;
- the four center goal cells form one open connected region;
- exactly one goal-region entrance exists;
- the canonical single planning goal is derived from that entrance rather than supplied manually;
- all 256 cells are reachable from the start.

## Independent image-wall verification

The wall strings were also checked directly against the reference images, independently of the JSON loader. For each PNG, the 17x17 cell-boundary lattice was located from the red maze-wall raster. The red occupancy of every horizontal and vertical cell-boundary segment was classified as wall/open and compared bit-for-bit with the submitted JSON.

Result: **all 3 mazes match their source images with 0 horizontal-wall mismatches and 0 vertical-wall mismatches.** Colored route overlays, goal shading/labels, and background grid lines were ignored; only red wall occupancy was used.

This audit verifies the wall topology represented by the supplied PNGs. It does not independently establish the historical provenance or competition identity of a maze; those are metadata claims and should retain their source citation when used publicly.
