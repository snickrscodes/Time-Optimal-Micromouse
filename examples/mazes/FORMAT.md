# `ame-maze-v1` custom maze format

`ame-maze-v1` is the repository's small JSON format for deterministic hand-transcribed Micromouse mazes. It records exact wall topology, start/goal semantics, coordinate convention, and physical scale without introducing adapters for unrelated legacy maze formats.

A loaded file becomes a `planning.maze_scenario.MazeScenario` and then uses the same junction graph, A*/B&B, continuous trajectory optimizer, speed solver, and certification stack as generated mazes.

## Quick example

```json
{
  "format": "ame-maze-v1",
  "name": "Example 4x4",
  "dimensions": {"width": 4, "height": 4},
  "coordinates": {"origin": "southwest"},
  "start": {"cell": [1, 0], "heading": "north"},
  "goals": [[1,1], [2,1], [1,2], [2,2]],
  "goal_policy": "single_entry",
  "walls": {
    "row_order": "north_to_south",
    "horizontal": ["1111", "1111", "1001", "1011", "1111"],
    "vertical": ["11111", "11011", "11011", "11111"]
  },
  "scale": {
    "unit_mm": 12.0,
    "wall_thickness_units": 1.0,
    "cell_pitch_units": 15.0
  }
}
```

The full checked example is [`ame_maze_v1_example.json`](ame_maze_v1_example.json).

## Field summary

| Field | Required | Meaning |
|---|---|---|
| `format` | yes | must be exactly `"ame-maze-v1"` |
| `name` | yes | nonempty human-readable maze name |
| `dimensions.width` / `height` | yes | positive cell dimensions |
| `coordinates.origin` | optional | `southwest` (default) or `northwest` |
| `start.cell` | yes | `[x,y]` in the declared source coordinate system |
| `start.heading` | optional | `north`, `east`, `south`, or `west` |
| `goals` | yes | all semantic goal cells |
| `goal_policy` | optional | currently only `single_entry` |
| `require_all_cells_reachable` | optional | require complete maze reachability when true |
| `walls.row_order` | optional | currently only `north_to_south` |
| `walls.horizontal` | yes | `H+1` binary strings of width `W` |
| `walls.vertical` | yes | `H` binary strings of width `W+1` |
| `scale` | optional | physical conversion metadata; project wall convention is fixed |
| `metadata` | optional | arbitrary non-planning JSON object |

## 1. Cell coordinate convention

Cells are integer pairs:

```text
[x, y]
```

with `x` increasing to the right.

### Southwest origin

```json
"coordinates": {"origin": "southwest"}
```

- `(0,0)` is the lower-left cell;
- `y` increases upward.

This is the recommended convention for manual Micromouse transcriptions.

### Northwest origin

```json
"coordinates": {"origin": "northwest"}
```

- `(0,0)` is the upper-left cell;
- `y` increases downward.

The loader normalizes either convention to the production planner's internal northwest-origin coordinates.

`start`, `goals`, and other semantic cell coordinates use the declared coordinate origin.

## 2. Wall row order is always visual north → south

Wall matrices deliberately use a different convention from semantic cell coordinates:

> `walls.horizontal` and `walls.vertical` rows are always written from the **top of the maze to the bottom**.

This makes image transcription direct: read the drawing top-to-bottom without first flipping the wall matrix because the semantic cell origin is southwest.

The only supported value is:

```json
"row_order": "north_to_south"
```

## 3. Wall encoding

Walls are encoded with one-character binary strings:

```text
1 = wall present
0 = opening
```

For a maze of width `W` and height `H`:

### Horizontal wall rows

`walls.horizontal` contains exactly `H + 1` strings, each of length `W`.

```text
horizontal[0]      north exterior grid line
horizontal[1]      between visual cell rows 0 and 1
...
horizontal[H-1]
horizontal[H]      south exterior grid line
```

Character `x` describes the horizontal edge spanning column `x`.

### Vertical wall rows

`walls.vertical` contains exactly `H` strings, each of length `W + 1`.

```text
vertical[y][0]     west exterior edge of visual row y
vertical[y][1]     edge between columns 0 and 1
...
vertical[y][W]     east exterior edge
```

### 4×4 indexing picture

```text
                  x=0 x=1 x=2 x=3
horizontal[0]      --- --- --- ---
vertical[0]       |   |   |   |   |
horizontal[1]      --- --- --- ---
vertical[1]       |   |   |   |   |
horizontal[2]      --- --- --- ---
vertical[2]       |   |   |   |   |
horizontal[3]      --- --- --- ---
vertical[3]       |   |   |   |   |
horizontal[4]      --- --- --- ---
```

All four exterior maze boundaries must be walls.

Each shared cell edge appears only once in the file, so two neighboring cells cannot disagree about whether their common boundary is open.

## 4. Semantic goals and the canonical planning goal

`goals` stores **all semantic goal cells**.

A standard 16×16 four-cell center using southwest coordinates is typically:

```json
"goals": [[7,7], [8,7], [7,8], [8,8]]
```

The currently supported policy is:

```json
"goal_policy": "single_entry"
```

The loader then:

1. verifies that the goal cells form one connected open region;
2. scans every open connection from a goal cell to a non-goal cell;
3. requires exactly **one** such external entrance;
4. uses the inside cell of that entrance as the planner's canonical single goal.

Do **not** provide a separate canonical-goal field. The wall data determines it.

This lets the legacy graph/A*/B&B stack retain one goal vertex while `MazeScenario` preserves the complete semantic goal region.

## 5. Start cell and optional heading

Example:

```json
"start": {
  "cell": [0,0],
  "heading": "north"
}
```

Accepted headings:

```text
north
east
south
west
```

If `heading` is supplied, the loader currently requires the start cell to have **exactly one opening**, and that opening must point in the declared heading.

This guarantees that the existing route initializer's first path direction agrees with the physical start pose without adding a separate free-heading branch to the planner.

If heading semantics are not needed, omit `start.heading`.

## 6. Physical scale

The project wall convention is:

```json
"scale": {
  "unit_mm": 12.0,
  "wall_thickness_units": 1.0,
  "cell_pitch_units": 15.0
}
```

Therefore:

- one project wall unit = **12 mm**;
- wall thickness = **12 mm**;
- standard cell pitch = `15 × 12 mm = 180 mm`.

The current grid planner still uses **one maze cell = one planner length unit** internally. `MazeScale` records the physical conversion for reporting/case-study rendering; loading a file does not silently rescale already-qualified dynamics.

The loader enforces `12 mm/unit` and one-unit wall thickness in `ame-maze-v1`. A positive nonstandard `cell_pitch_units` may be recorded, but physical case-study models must still be selected consistently with that scale.

## 7. Optional metadata

`metadata` may contain any JSON object. It does not affect planning.

Suggested historical fields:

```json
"metadata": {
  "competition": "All Japan Micromouse Contest",
  "year": 2017,
  "round": "Final",
  "source": "description or URL used for transcription",
  "notes": "optional transcription notes"
}
```

Use metadata for provenance/context, not for values the planner needs to interpret the maze.

## 8. Complete valid example

The repository example:

```json
{
  "format": "ame-maze-v1",
  "name": "Loader example 4x4",
  "dimensions": {"width": 4, "height": 4},
  "coordinates": {"origin": "southwest"},

  "start": {
    "cell": [1, 0],
    "heading": "north"
  },

  "goals": [[1,1], [2,1], [1,2], [2,2]],
  "goal_policy": "single_entry",
  "require_all_cells_reachable": false,

  "walls": {
    "row_order": "north_to_south",
    "horizontal": [
      "1111",
      "1111",
      "1001",
      "1011",
      "1111"
    ],
    "vertical": [
      "11111",
      "11011",
      "11011",
      "11111"
    ]
  },

  "scale": {
    "unit_mm": 12.0,
    "wall_thickness_units": 1.0,
    "cell_pitch_units": 15.0
  },

  "metadata": {
    "purpose": "format example"
  }
}
```

In southwest coordinates, the unique goal entrance is:

```text
outside cell:   [1,0]
inside cell:    [1,1]
canonical goal: [1,1]
```

Internally the production northwest-origin representation becomes:

```text
(1,3) -> (1,2)
```

## 9. Validation rules

The loader rejects a file when any of the following is true:

- `format` is not `ame-maze-v1`;
- `name` is empty;
- dimensions are nonpositive or malformed;
- semantic coordinates are outside the maze;
- wall row counts/string lengths are incorrect;
- wall strings contain characters other than `0`/`1`;
- an exterior boundary is open;
- goals are empty or duplicated;
- goal cells do not form one connected open region;
- `goal_policy` is not `single_entry`;
- the goal region has anything other than one external opening;
- the derived canonical goal is unreachable from the start;
- `require_all_cells_reachable` is true and any cell is unreachable;
- a supplied start heading does not match a unique start opening;
- `metadata` is not an object;
- scale contradicts the fixed 12 mm/unit + one-unit wall-thickness convention;
- cell pitch is nonpositive.

Invalid files raise `planning.maze_io.MazeFormatError` rather than being silently repaired.

## 10. Source hashing

`load_maze_scenario()` computes SHA-256 over the **exact source JSON bytes** and stores it in the resulting `MazeScenario`.

This allows historical case-study records to identify the precise wall file used by an optimization/certificate.

Formatting-only edits therefore intentionally change the source hash even if they encode the same wall topology.

## 11. Running a loaded maze

```bash
python main.py \
  --maze-file examples/mazes/my_maze.json \
  --optimization-mode time
```

When `--maze-file` is used, the scenario supplies:

- maze width/height;
- normalized start;
- derived canonical goal;
- semantic goal metadata;
- source scale/provenance.

The rest of the planner stack is unchanged: compressed junction graph, A*, branch-and-bound, continuous route optimization, profile-selected speed model, and independent certification.

Useful planner options can be inspected with:

```bash
python main.py --help
```

## 12. Historical transcriptions

Checked historical/reference mazes live under:

```text
examples/mazes/historical/
```

The wall-level audit summary is:

[`historical/TRANSCRIPTION_AUDIT.md`](historical/TRANSCRIPTION_AUDIT.md)

The Red Comet use of the format is documented in [`../../docs/RED_COMET_CASE_STUDY.md`](../../docs/RED_COMET_CASE_STUDY.md).
