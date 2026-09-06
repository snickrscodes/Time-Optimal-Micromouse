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


def gradients_plot(data: dict[str, Any], path: Path) -> None:
    plt=_plt()
    checked=[r for r in data["finite_difference_checks"] if r.get("checked")]
    values=[max(r["relative_error"],1e-18) for r in checked]
    fig,ax=plt.subplots(figsize=(8,4.8))
    if values:
        ax.hist(values,bins=min(20,max(5,len(values)//2)))
        ax.set_xscale("log")
    ax.set_xlabel("Relative error"); ax.set_ylabel("Coordinates")
    ax.set_title("5-point finite-difference gradient spot checks")
    ax.grid(True,alpha=0.25); fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)


def warm_plot(data: dict[str, Any], path: Path) -> None:
    plt=_plt(); variants=["direct_time","length_then_time","curvature_then_time","production_warm_start"]
    routes=sorted({r["route_name"] for r in data["rows"]})
    x=np.arange(len(routes)); width=0.18
    fig,ax=plt.subplots(figsize=(10,4.8))
    for i,v in enumerate(variants):
        vals=[]
        for route in routes:
            row=next(r for r in data["rows"] if r["route_name"]==route and r["variant"]==v)
            vals.append(row["final_travel_time"] if row["certification"]["certified"] else np.nan)
        ax.bar(x+(i-1.5)*width,vals,width,label=v)
    ax.set_xticks(x,[r.replace("cyclic_4x4_","") for r in routes]); ax.set_ylabel("Certified final time (s)")
    ax.set_title("Warm-start ablation"); ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=0.25); fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)


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


def resolution_plot(data: dict[str, Any], path: Path) -> None:
    plt=_plt(); rows=[r for r in data["rows"] if r["N"]["certified"] and r["2N"]["certified"]]
    names=[r["name"].replace("cyclic_4x4_","") for r in rows]; n=[r["N"]["time"] for r in rows]; n2=[r["2N"]["time"] for r in rows]
    x=np.arange(len(rows)); width=.38
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(x-width/2,n,width,label="N"); ax.bar(x+width/2,n2,width,label="2N")
    ax.set_xticks(x,names); ax.set_ylabel("Certified traversal time (s)"); ax.set_title("Resolution sensitivity")
    ax.legend(); ax.grid(True,axis="y",alpha=.25); fig.tight_layout()
    _save_figure(fig, path, dpi=160); plt.close(fig)



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
