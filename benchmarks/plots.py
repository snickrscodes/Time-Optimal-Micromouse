from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    if path.suffix.lower() == ".png":
        image = Image.open(path)
        try:
            if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                white = Image.new("RGB", image.size, (255, 255, 255))
                alpha = image.getchannel("A") if "A" in image.getbands() else None
                white.paste(image.convert("RGB"), mask=alpha) if alpha is not None else white.paste(image.convert("RGB"))
                white.save(path)
            elif image.mode != "RGB":
                image.convert("RGB").save(path)
        finally:
            image.close()


def topology_plot(data: dict[str, Any], path: Path) -> None:
    plt = _plt()
    rows = [r for r in data["cases"] if r["astar"]["certification"]["certified"] and r["branch_and_bound"]["certification"]["certified"]]
    names = [r["case"]["name"].replace("cyclic_4x4_", "") for r in rows]
    astar = [r["astar"]["route"]["time"] for r in rows]
    bb = [r["branch_and_bound"]["route"]["time"] for r in rows]
    x = np.arange(len(rows)); width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x-width/2, astar, width, label="A* topology + continuous optimizer")
    ax.bar(x+width/2, bb, width, label="B&B topology + same optimizer")
    ax.set_xticks(x, names); ax.set_ylabel("Certified traversal time (s)")
    ax.set_title("Shortest-distance topology vs kinodynamic topology search")
    ax.legend(); ax.grid(True, axis="y", alpha=0.25); fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)


def bounds_plot(data: dict[str, Any], path: Path) -> None:
    plt = _plt()
    levels = ["basic", "two_sided", "projection12", "complete_cover"]
    tight = [] ; avoided=[]
    for level in levels:
        medians = [c["ablation"][level]["tightness_ratio"]["median"] for c in data["cases"]]
        av = [c["ablation"][level]["search"]["complete_route_optimizations_avoided_fraction"] for c in data["cases"]]
        tight.append(float(np.median(medians))); avoided.append(float(np.median(av)))
    x=np.arange(len(levels))
    fig, ax=plt.subplots(figsize=(9,4.8))
    ax.plot(x, tight, marker="o", label="Median LB / best completion")
    ax.plot(x, avoided, marker="s", label="Median complete solves avoided")
    ax.set_xticks(x,["basic","+ two-sided","+ 12-dir","+ eroded cover"])
    ax.set_ylim(bottom=0); ax.set_title("Lower-bound hierarchy: tightness and pruning")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)


def gradients_plot(data: dict[str, Any], path: Path) -> dict[str, int]:
    plt = _plt()
    checked = [r for r in data["finite_difference_checks"] if r.get("checked")]
    values = np.asarray([max(float(r["relative_error"]), 1.0e-18) for r in checked], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if values.size:
        lo_exp = np.floor(np.log10(values.min()))
        hi_exp = np.ceil(np.log10(values.max()))
        if hi_exp <= lo_exp:
            hi_exp = lo_exp + 1.0
        bins = np.geomspace(10.0 ** lo_exp, 10.0 ** hi_exp, 15)
        counts, _, _ = ax.hist(values, bins=bins)
        nonempty_bins = int(np.count_nonzero(counts))
        ax.set_xscale("log")
        median = float(np.median(values))
        p95 = float(np.quantile(values, 0.95))
        ax.axvline(median, linestyle="--", linewidth=1.3, label=f"median {median:.2e}")
        ax.axvline(p95, linestyle=":", linewidth=1.5, label=f"p95 {p95:.2e}")
        ax.legend(frameon=False, fontsize=8)
    else:
        nonempty_bins = 0
    ax.set_xlabel("Relative error")
    ax.set_ylabel("Coordinates")
    ax.set_title("5-point finite-difference gradient spot checks")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, path, dpi=160)
    plt.close(fig)
    return {"checked_coordinates": int(values.size), "nonempty_bins": nonempty_bins}


def warm_plot(data: dict[str, Any], path: Path) -> dict[str, int]:
    plt = _plt()
    variants = ["direct_time", "length_then_time", "curvature_then_time", "production_warm_start"]
    labels = {
        "direct_time": "direct time",
        "length_then_time": "length → time",
        "curvature_then_time": "curvature → time",
        "production_warm_start": "production warm start",
    }
    routes = sorted({r["route_name"] for r in data["rows"]})
    x = np.arange(len(routes), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ymax = max(float(r["total_wall_seconds"]) for r in data["rows"] if r.get("total_wall_seconds") is not None)
    plotted = 0
    failed = 0
    certified_count = 0
    for i, variant in enumerate(variants):
        rows = [next(r for r in data["rows"] if r["route_name"] == route and r["variant"] == variant) for route in routes]
        vals = [float(r["total_wall_seconds"]) for r in rows]
        positions = x + (i - 1.5) * width
        bars = ax.bar(positions, vals, width, label=labels[variant])
        for bar, row in zip(bars, rows):
            plotted += 1
            certified = bool(row["certification"]["certified"])
            if not certified:
                failed += 1
                bar.set_hatch("///")
                bar.set_alpha(0.72)
                text = "failed"
            else:
                certified_count += 1
                text = f"T={float(row['final_travel_time']):.3f}s"
            ax.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90 if not certified else 0,
            )
    ax.set_xticks(x, [r.replace("cyclic_4x4_", "") for r in routes])
    ax.set_ylabel("Total wall time (s)")
    ax.set_title("Warm-start ablation: certification outcome and wall cost")
    ax.set_ylim(0.0, ymax * 1.14)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.99, 0.02, "hatched = failed certification", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path, dpi=160)
    plt.close(fig)
    return {"rows_plotted": plotted, "failed_rows": failed, "certified_rows": certified_count}


def native_plot(data: dict[str, Any], path: Path) -> None:
    plt=_plt(); rows=data["microbenchmarks"]
    names=[r["name"].replace("cyclic_4x4_","") for r in rows]
    py=[1000*r["python"]["time_gradient_median_seconds"] for r in rows]
    na=[1000*r["native"]["time_gradient_median_seconds"] for r in rows]
    x=np.arange(len(rows)); width=.38
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(x-width/2,py,width,label="Python"); ax.bar(x+width/2,na,width,label="Native")
    ax.set_xticks(x,names); ax.set_ylabel("Median time+gradient evaluation (ms)")
    ax.set_title("Reverse-solver backend performance"); ax.legend(); ax.grid(True,axis="y",alpha=.25); fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)


def resolution_plot(data: dict[str, Any], path: Path) -> dict[str, int]:
    plt = _plt()
    rows = list(data["rows"])
    names = [r["name"].replace("cyclic_4x4_", "") for r in rows]
    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    outcomes_plotted = 0
    failed = 0
    certified_count = 0
    for offset, key, marker, label in [(-0.12, "N", "o", "N"), (0.12, "2N", "s", "2N")]:
        outcomes = [1.0 if bool(r[key]["certified"]) else 0.0 for r in rows]
        points = ax.scatter(x + offset, outcomes, s=90, marker=marker, label=label, zorder=3)
        color = points.get_facecolor()[0] if len(points.get_facecolor()) else None
        for xx, yy, row in zip(x + offset, outcomes, rows):
            outcomes_plotted += 1
            record = row[key]
            if record["certified"] and record.get("time") is not None:
                certified_count += 1
                text = f"{float(record['time']):.4f}s"
                dy = 8
                va = "bottom"
            else:
                failed += 1
                status = str(record.get("execution_status") or "failed").replace("_", " ")
                text = status
                dy = -10
                va = "top"
            ax.annotate(text, (xx, yy), xytext=(0, dy), textcoords="offset points", ha="center", va=va, fontsize=7.5, color=color)
    ax.set_xticks(x, names)
    ax.set_yticks([0.0, 1.0], ["failed", "certified"])
    ax.set_ylim(-0.32, 1.30)
    ax.set_ylabel("Independent reoptimization outcome")
    ax.set_title("Resolution sensitivity: N → 2N reoptimization outcomes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    max_pos = data.get("aggregate", {}).get("maximum_initial_position_prolongation_error")
    if max_pos is not None:
        ax.text(
            0.01, 0.98,
            f"Exact prolongation max position error: {float(max_pos):.2e}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        )
    fig.tight_layout()
    _save_figure(fig, path, dpi=160)
    plt.close(fig)
    return {"rows_plotted": len(rows), "outcomes_plotted": outcomes_plotted, "failed_outcomes": failed, "certified_outcomes": certified_count}



def direct_transcription_plot(data: dict[str, Any], path: Path) -> None:
    plt=_plt()
    rows=[r for r in data["rows"] if r["initialization_label"]=="cold"]
    meshes=sorted({r["base_uniform_intervals"] for r in rows})
    fig,ax=plt.subplots(figsize=(8,4.8))
    for name in sorted({r["name"] for r in rows}):
        route=[r for r in rows if r["name"]==name]
        x=[r["base_uniform_intervals"] for r in route]
        y=[100.0*abs(r["relative_time_difference"]) for r in route]
        ax.plot(x,y,marker="o",label=name.replace("cyclic_4x4_", ""))
        for r,xx,yy in zip(route,x,y):
            if not r["certification"]["certified"]:
                ax.scatter([xx],[yy],marker="x",s=70)
    ax.set_xscale("log",base=2)
    ax.set_xticks(meshes, [str(m) for m in meshes])
    ax.set_xlabel("Uniform base intervals (exact clothoid knots also inserted)")
    ax.set_ylabel("|transcription - production| / production (%)")
    ax.set_title("CasADi/IPOPT fixed-geometry speed transcription convergence")
    ax.legend(title="Maze")
    ax.grid(True,alpha=.25)
    fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)


def generate_all(output_dir: Path, plots_dir: Path) -> dict[str, str]:
    import json
    functions = {
        "topology_search": topology_plot,
        "lower_bounds": bounds_plot,
        "gradients": gradients_plot,
        "warm_start": warm_plot,
        "native_stack": native_plot,
        "resolution": resolution_plot,
        "direct_transcription": direct_transcription_plot,
    }
    outputs={}
    for name,fn in functions.items():
        src=output_dir/f"{name}.json"
        if not src.exists(): continue
        data=json.loads(src.read_text())
        target=plots_dir/f"{name}.png"
        fn(data,target); outputs[name]=str(target)
    return outputs
