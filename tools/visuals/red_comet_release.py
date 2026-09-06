"""Render the publication-ready Red Comet 2017 case-study assets.

This is postprocessing only.  It consumes the compact, checked-in
``analysis/red_comet_2017/final_result.json`` release snapshot and performs no
route search or continuous optimization.  The hand-maintained TikZ architecture
diagram is intentionally outside this run-derived renderer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path




def _combine_gifs_side_by_side(
    left: Path,
    right: Path,
    output: Path,
    *,
    gap: int = 12,
    fps: float = 20.0,
) -> Path:
    """Combine two fixed-size hero GIFs on one synchronized real-time canvas.

    The shorter animation freezes on its final frame while the longer one
    continues, preserving the physical-time comparison.
    """
    from PIL import Image

    if gap < 0 or fps <= 0.0:
        raise ValueError("gap must be nonnegative and fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(left) as left_gif, Image.open(right) as right_gif:
        left_frames = int(getattr(left_gif, "n_frames", 1))
        right_frames = int(getattr(right_gif, "n_frames", 1))
        frame_count = max(left_frames, right_frames)
        left_size = left_gif.size
        right_size = right_gif.size
        width = left_size[0] + gap + right_size[0]
        height = max(left_size[1], right_size[1])
        frames = []
        for frame in range(frame_count):
            left_gif.seek(min(frame, left_frames - 1))
            right_gif.seek(min(frame, right_frames - 1))
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(left_gif.convert("RGB"), (0, (height - left_size[1]) // 2))
            canvas.paste(
                right_gif.convert("RGB"),
                (left_size[0] + gap, (height - right_size[1]) // 2),
            )
            frames.append(canvas.quantize(colors=256))
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return output


def _draw_route(ax, maze, route, *, scenario, title: str, subtitle: str) -> None:
    from visualization.maze import draw_goal_region, draw_maze_walls, draw_start
    from visualization.sampling import sample_geometry_parameters
    from visualization.trajectory import draw_geometry_trace, draw_topology

    draw_maze_walls(ax, maze)
    draw_topology(ax, route.cells, label="topology centerline")
    trace = sample_geometry_parameters(
        route.parameters,
        route.initial_state,
        samples_per_unit=100.0,
        minimum_samples_per_segment=12,
    )
    draw_geometry_trace(ax, trace, label="optimized clothoid trajectory")
    draw_start(ax, route.cells[0], label="start")
    draw_goal_region(ax, scenario=scenario)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)


def _route_problem(record: dict):
    """Rebuild the exact corridor model needed for active-basis geometry plots."""
    from dataclasses import replace
    import numpy as np
    from optimization import CorridorModel
    from tools.geometry_homotopy.goal_entry import build_goal_entry_problem

    cfg = record.get("config", {})
    cells = tuple(tuple(int(v) for v in c) for c in record["cells"])
    kwargs = dict(
        body_length=float(cfg.get("body_length", 5.0 / 9.0)),
        body_height=float(cfg.get("body_height", 4.0 / 9.0)),
        clearance=float(cfg.get("clearance", 0.0)),
        refinement_factor=int(cfg.get("geometry_refinement", 1)),
    )
    active = record.get("active_basis") or {}
    kind = str(active.get("basis_kind", "full"))
    if kind == "reduced":
        problem = build_goal_entry_problem(cells, corridor_mode="maximal_runs", **kwargs)
    else:
        problem = build_goal_entry_problem(cells, corridor_mode="overlapping_cover", **kwargs)
        if kind == "hybrid":
            assignment = active.get("final_segment_cells_values")
            if assignment is None:
                raise RuntimeError("hybrid release snapshot is missing final_segment_cells_values")
            corridor = CorridorModel(
                problem.corridor.cells,
                tuple(int(v) for v in assignment),
                problem.corridor.body,
                problem.corridor.clearance,
            )
            problem = replace(problem, corridor=corridor)
    return replace(problem, initial_parameters=np.asarray(record["parameters"], dtype=float))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("analysis/red_comet_2017/final_result.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/red_comet"),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    doc = json.loads(args.result.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from optimization import RectangleBody
    from planning.maze_io import load_maze_scenario
    from visualization.animation import build_playback_trace, render_trajectory_gif
    from visualization.detail import draw_geometry_detail
    from visualization.records import route_view_from_record
    from visualization.speed import speed_profile_trace_from_dd_record
    from visualization.speed_plot import draw_speed_profile

    scenario = load_maze_scenario(Path(doc["maze_file"]))
    maze = scenario.maze
    astar_record = doc["astar"]
    green_record = doc["historical_green"]
    astar = route_view_from_record(astar_record)
    green = route_view_from_record(green_record)
    comp = doc["comparison"]

    # Hero comparison: the public Red Comet story in one figure.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.25), sharex=True, sharey=True)
    _draw_route(
        axes[0], maze, astar, scenario=scenario,
        title="Shortest topology (A*)",
        subtitle=f"{astar.time:.3f} s · {comp['astar_grid_steps']} grid steps",
    )
    _draw_route(
        axes[1], maze, green, scenario=scenario,
        title="Historical Red Comet topology",
        subtitle=f"{green.time:.3f} s · {comp['historical_green_grid_steps']} grid steps",
    )
    fig.suptitle("Red Comet 2017 case study · calibrated differential-drive/yaw model", fontsize=14)
    fig.text(
        0.5, 0.045,
        f"Historical topology: +{comp['green_minus_astar_seconds']:.3f} s "
        f"(+{comp['green_slower_percent']:.1f}%) · exhaustive simple-topology search: "
        f"{doc['topology_search']['simple_topologies']}/10 topologies evaluated",
        ha="center", va="center", fontsize=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), frameon=False,
                   bbox_to_anchor=(0.5, 0.075))
    fig.tight_layout(rect=(0.0, 0.115, 1.0, 0.93))
    comparison_svg = args.output_dir / "red_comet_astar_vs_historical.svg"
    fig.savefig(comparison_svg, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    # Compact supporting evidence: all ten simple topologies from the exhaustive run.
    topologies = doc["topology_search"]["complete_topologies"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    alternatives = [r for r in topologies if not r["best"] and not r["historical_green"]]
    astar_row = next(r for r in topologies if r["best"])
    green_row = next(r for r in topologies if r["historical_green"])
    ax.scatter(
        [r["cells"] - 1 for r in alternatives],
        [r["optimized_time_seconds"] for r in alternatives],
        s=55, label="other simple topologies",
    )
    ax.scatter([astar_row["cells"] - 1], [astar_row["optimized_time_seconds"]], s=90, label="A* shortest")
    ax.scatter([green_row["cells"] - 1], [green_row["optimized_time_seconds"]], s=90, label="historical green")
    ax.annotate(
        f"A*  {astar_row['optimized_time_seconds']:.3f} s",
        (astar_row["cells"] - 1, astar_row["optimized_time_seconds"]),
        xytext=(8, 10), textcoords="offset points",
    )
    ax.annotate(
        f"historical  {green_row['optimized_time_seconds']:.3f} s",
        (green_row["cells"] - 1, green_row["optimized_time_seconds"]),
        xytext=(8, 8), textcoords="offset points",
    )
    ax.set_xlabel("Topology length (grid steps)")
    ax.set_ylabel("Optimized traversal time (s)")
    ax.set_title("Exhaustive simple-topology comparison")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    topology_svg = args.output_dir / "red_comet_topology_summary.svg"
    fig.savefig(topology_svg, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    # Final A* active-basis geometry.
    problem = _route_problem(astar_record)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_geometry_detail(
        ax,
        maze=maze,
        problem=problem,
        parameters=astar.parameters,
        cells=astar.cells,
        footprint_count=11,
        show_topology=True,
    )
    ax.set_title(f"Final A* active-basis trajectory · {astar.time:.3f} s")
    fig.tight_layout()
    geometry_svg = args.output_dir / "red_comet_astar_geometry.svg"
    fig.savefig(geometry_svg, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    # Final A* speed profile and real-time playback use the persisted DD/yaw trace.
    profile = doc["physics_model_identity"]
    cell_pitch_m = float(profile.get("cell_pitch_m", scenario.scale.cell_pitch_m))
    astar_speed = speed_profile_trace_from_dd_record(
        astar_record["speed_trace"],
        exact_total_time=astar.time,
        cell_pitch_m=cell_pitch_m,
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.5), sharex=True)
    draw_speed_profile(axes[0], axes[1], astar_speed, shade_modes=False, show_events=False, cell_pitch_m=cell_pitch_m)
    axes[0].set_title(f"Final A* speed profile · T={astar.time:.6f} s")
    fig.tight_layout()
    speed_svg = args.output_dir / "red_comet_astar_speed.svg"
    fig.savefig(speed_svg, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    playback = build_playback_trace(astar.raw_parameters, astar.initial_state, astar_speed, fps=args.fps)
    cfg = astar.config
    body = RectangleBody.centered(float(cfg.get("body_length", 5 / 9)), float(cfg.get("body_height", 4 / 9)))
    hero_gif = render_trajectory_gif(
        args.output_dir / "red_comet_astar_hero.gif",
        maze=maze,
        playback=playback,
        body=body,
        cells=astar.cells,
        scenario=scenario,
        fps=args.fps,
        dpi=90,
        title=f"Red Comet · A* optimized · {astar.time:.3f} s",
        cell_pitch_m=cell_pitch_m,
    )

    green_speed = speed_profile_trace_from_dd_record(
        green_record["speed_trace"],
        exact_total_time=green.time,
        cell_pitch_m=cell_pitch_m,
    )
    green_playback = build_playback_trace(
        green.raw_parameters, green.initial_state, green_speed, fps=args.fps
    )
    green_cfg = green.config
    green_body = RectangleBody.centered(
        float(green_cfg.get("body_length", 5 / 9)),
        float(green_cfg.get("body_height", 4 / 9)),
    )
    import tempfile
    side_by_side_gif = args.output_dir / "red_comet_astar_vs_green_side_by_side.gif"
    with tempfile.TemporaryDirectory() as tmpdir_str:
        green_hero_tmp = Path(tmpdir_str) / "historical_route_tmp.gif"
        render_trajectory_gif(
            green_hero_tmp,
            maze=maze,
            playback=green_playback,
            body=green_body,
            cells=green.cells,
            scenario=scenario,
            fps=args.fps,
            dpi=90,
            title=f"Red Comet · historical green · {green.time:.3f} s",
            cell_pitch_m=cell_pitch_m,
        )
        _combine_gifs_side_by_side(
            Path(hero_gif), green_hero_tmp, side_by_side_gif, fps=args.fps
        )

    summary = {
        "schema": "red-comet-release-visuals-v2",
        "source": str(args.result),
        "astar_time_seconds": astar.time,
        "historical_green_time_seconds": green.time,
        "comparison_svg": str(comparison_svg),
        "topology_svg": str(topology_svg),
        "geometry_svg": str(geometry_svg),
        "speed_svg": str(speed_svg),
        "hero_gif": str(hero_gif),
        "side_by_side_gif": str(side_by_side_gif),
        "playback_frames": len(playback.time),
        "historical_green_playback_frames": len(green_playback.time),
    }
    (args.output_dir / "visual_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
