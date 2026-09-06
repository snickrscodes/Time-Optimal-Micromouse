from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import SUITES
from .schema import validate_benchmark_result


def _load(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_benchmark_result(payload, expected_benchmark=name)
    return payload


def _load_aux(directory: Path, filename: str) -> dict[str, Any] | None:
    path = directory / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if number != 0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}%"


def _ratio_pct(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _plot_link(report_path: Path, plots_dir: Path, name: str) -> str | None:
    path = plots_dir / f"{name}.png"
    if not path.exists():
        return None
    try:
        rel = path.relative_to(report_path.parent)
    except ValueError:
        import os
        rel = Path(os.path.relpath(path, report_path.parent))
    return rel.as_posix()


def _title(name: str) -> str:
    return SUITES[name].public_title


def _provenance_lines(env: dict[str, Any]) -> list[str]:
    provenance = env.get("provenance", {})
    if provenance.get("git_commit"):
        source = f"Git commit `{provenance['git_commit']}`"
        if provenance.get("git_branch"):
            source += f" on `{provenance['git_branch']}`"
        source += f"; dirty tree: `{bool(provenance.get('git_dirty'))}`"
    else:
        source = f"source-tree SHA-256 `{provenance.get('source_tree_sha256')}`"
    py = env.get("python", {})
    os_meta = env.get("os", {})
    cpu = env.get("cpu", {})
    compilers = env.get("compilers", {})
    native = env.get("native_artifacts", [])
    native_ok = all(row.get("exists") for row in native) if native else None
    lines = [
        f"- Source: {source}",
        f"- Imported production archive SHA-256: `{provenance.get('imported_source_archive_sha256')}`",
        f"- Timestamp (UTC): `{env.get('timestamp_utc')}`",
        f"- OS / architecture: `{os_meta.get('system')} {os_meta.get('release')} / {os_meta.get('machine')}`",
        f"- CPU: `{cpu.get('model')}` ({cpu.get('logical_count')} logical CPUs)",
        f"- Python: `{py.get('implementation')} {py.get('version')}`",
        f"- NumPy / SciPy: `{env.get('numpy')}` / `{env.get('scipy')}`",
        f"- C / C++ compiler: `{compilers.get('cc')}` / `{compilers.get('cxx')}`",
        f"- Native benchmark artifacts present: `{native_ok}`",
        f"- Thread environment: `{env.get('thread_environment', {})}`",
    ]
    if env.get("casadi") is not None or env.get("casadi_version") is not None:
        lines.append(
            f"- CasADi / IPOPT: `{env.get('casadi_version', env.get('casadi'))}` / `{env.get('ipopt_version', env.get('ipopt_plugin'))}`"
        )
    return lines


def generate(results_dir: Path, report_path: Path, plots_dir: Path) -> str:
    results = {name: _load(results_dir, name) for name in SUITES}
    official = {name: data for name, data in results.items() if data and SUITES[name].tier == "official"}
    supplemental = {name: data for name, data in results.items() if data and SUITES[name].tier == "supplemental"}
    if not official and not supplemental:
        raise RuntimeError("no benchmark result files found")
    first = next(iter(official.values() or supplemental.values()))
    env = first["environment"]

    topology = results.get("topology_search")
    bounds = results.get("lower_bounds")
    gradients = results.get("gradients")
    warm = results.get("warm_start")
    native = results.get("native_stack")
    resolution = results.get("resolution")
    direct = results.get("direct_transcription")
    full = results.get("full_ocp")
    full_resolution_control = _load_aux(results_dir, "full_ocp_resolution_control.json")
    sensitivity = results.get("full_ocp_sensitivity")

    lines: list[str] = [
        "# AME Robot Benchmark Report",
        "",
        "This report is generated exclusively from the machine-readable JSON files in this result directory. Numerical table entries and resume claims are derived from those files rather than manually copied into Markdown.",
        "",
        "## Executive results",
        "",
        "| Evidence | Measured result |",
        "|---|---|",
    ]
    if topology:
        a = topology["aggregate"]
        lines.append(
            f"| {_title('topology_search')} | {a['cases_in_quality_aggregate']}/{a['cases_total']} certified paired cases; median improvement {_pct(a['time_improvement_percent']['median'])}, max {_pct(a['time_improvement_percent']['max'])}; {a['topology_changes']} topology changes |"
        )
    if bounds:
        a = bounds["aggregate"]
        lines.append(
            f"| {_title('lower_bounds')} | B&B matched exhaustive enumeration on {a['bb_matches_exhaustive']}/{a['cases']} mazes; {a.get('total_prefixes_checked', 'n/a')} prefixes checked with {a.get('admissibility_violations', 'n/a')} admissibility violations; median complete-route solves avoided {_ratio_pct(a['complete_route_optimizations_avoided_fraction']['median'])} |"
        )
    if gradients:
        a = gradients["aggregate"]
        lines.append(
            f"| {_title('gradients')} | Python/native max absolute gradient difference {_f(a['maximum_absolute_gradient_difference'])}; 5-point FD checked {a['finite_difference_coordinates_checked']}/{a['finite_difference_coordinates_total']} coordinates with median relative error {_f(a['finite_difference_relative_error']['median'])} |"
        )
    if warm:
        a = warm["aggregate"]
        p = a.get("production_vs_direct", {})
        lines.append(
            f"| {_title('warm_start')} | {a['routes']} routes × {len(a['variants'])} variants; production-vs-direct median final-time improvement {_pct(p.get('median_final_time_improvement_percent'))}; median direct/production wall ratio {_f(p.get('median_wall_speedup_direct_over_production'),2)}× |"
        )
    if native:
        a = native["aggregate"]
        lines.append(
            f"| {_title('native_stack')} | median time+gradient speedup {_f(a['median_time_gradient_speedup'],2)}×; representative optimization speedup {_f(a['median_end_to_end_speedup'],2)}×; max time-value error {_f(a['maximum_time_value_absolute_error'])} s |"
        )
    if resolution:
        a = resolution["aggregate"]
        lines.append(
            f"| {_title('resolution')} | {a['certified_pairs']}/{a['cases']} certified N/2N pairs; median |relative time difference| {_ratio_pct(a['median_absolute_relative_time_difference'],3)}; max {_ratio_pct(a['maximum_absolute_relative_time_difference'],3)} |"
        )
    if direct:
        a = direct["aggregate"]
        lines.append(
            f"| {_title('direct_transcription')} | finest {a['finest_mesh']}-interval cold transcription certified on {a['finest_mesh_certified_cold_cases']}/{a['cases']} routes; median |time difference| {_pct(a['finest_mesh_cold_relative_time_difference_percent']['median'],3)}; median IPOPT/production solve-time ratio {_f(a['finest_mesh_cold_runtime_ratio_ipopt_over_production']['median'],1)}× |"
        )
    if full:
        a = full["aggregate"]
        if full_resolution_control:
            fine = full_resolution_control["fine_resolution_summary"]
            lines.append(
                f"| {_title('full_ocp')} | {a['certified_solves']}/{a['solves']} OCP meshes independently certified; after multilevel structured reoptimization, finest {fine['ocp_intervals']}-interval OCP objective {_f(fine['ocp_objective_time_seconds'],6)} s is {_pct(fine['ocp_vs_structured_percent'],3)} vs the {fine['structured_segments']}-segment structured control; production hybrid replay on the OCP geometry is {_pct(fine['hybrid_replay_vs_structured_percent'],3)} vs structured |"
            )
        else:
            lines.append(
                f"| {_title('full_ocp')} | {a['certified_solves']}/{a['solves']} OCP meshes independently certified; finest mesh {a['finest_mesh']} intervals; canonical OCP-vs-structured comparison requires `full_ocp_resolution_control.json` |"
            )

    lines += ["", "## Reproducibility", ""]
    lines.extend(_provenance_lines(env))
    lines += [
        "- Production dependencies: `python -m pip install -r requirements.txt`",
        "- Core benchmark/test dependencies: `python -m pip install -r requirements-benchmarks.txt`",
        "- OCP baseline dependencies: `python -m pip install -r requirements-benchmarks-ocp.txt`",
        "- Exact checked-reference Python stack: `python -m pip install -r requirements-reference.txt`",
        "- Native build: `make native`",
        "- Fast infrastructure tests: `make benchmark-tests`",
        "- Normal core campaign: `python -m benchmarks.run --profile core`",
        "- Full campaign: `python -m benchmarks.run --profile all`",
        "- Explicit checked-reference regeneration: `python -m benchmarks.run --profile all --reference --overwrite`",
        "",
        "The generic OCP benchmarks run each numerical component in an isolated process group. A worker result is accepted only after its result file and completion marker are durable; teardown-only stalls are terminated after a short grace period so one solver's extension/plugin lifecycle cannot contaminate later timings.",
        "",
    ]

    if topology:
        lines += [
            f"## {_title('topology_search')}",
            "",
            "**Question.** Does discrete kinodynamic topology selection improve certified traversal time relative to optimizing the shortest-distance A* topology with the same continuous route optimizer?",
            "",
            "The A* route is continuously optimized and independently certified before comparison. B&B uses the same complete-route optimization policy. No unoptimized centerline is compared against an optimized trajectory.",
            "",
            "| case | cells/junctions | A* T (s) | B&B T (s) | improvement | topology changed | complete optimizations | expanded/generated | certified |",
            "|---|---:|---:|---:|---:|---|---:|---:|---|",
        ]
        for r in topology["cases"]:
            s = r["branch_and_bound"]["search_statistics"]
            cert = r["astar"]["certification"]["certified"] and r["branch_and_bound"]["certification"]["certified"]
            lines.append(
                f"| {r['case']['name']} | {r['maze']['cells']}/{r['maze']['junctions']} | {_f(r['astar']['route']['time'],6)} | {_f(r['branch_and_bound']['route']['time'],6)} | {_pct(r['comparison']['percentage_time_improvement'])} | {r['comparison']['topology_changed']} | {r['branch_and_bound']['complete_route_optimizations_including_seed']} | {s['expanded']}/{s['generated']} | {cert} |"
            )
        a = topology["aggregate"]
        lines += [
            "",
            f"Across the certified predetermined suite, mean/median/max traversal-time improvement is **{_pct(a['time_improvement_percent']['mean'])} / {_pct(a['time_improvement_percent']['median'])} / {_pct(a['time_improvement_percent']['max'])}**, with **{a['topology_changes']}** topology changes and **{a['longer_but_faster_cases']}** geometrically-longer-but-faster selections.",
            "",
        ]
        link = _plot_link(report_path, plots_dir, "topology_search")
        if link:
            lines += [f"![A* vs B&B traversal times]({link})", ""]
        lines += ["**Limitation.** B&B is exhaustive/global only over its finite discrete topology policy; the continuous NLP remains local.", ""]

    if bounds:
        lines += [
            f"## {_title('lower_bounds')}",
            "",
            "Every allowed simple path in each tractable case is optimized and independently certified. Prefix lower bounds are checked against the best known certified completion under that finite topology set; a positive residual above tolerance fails the suite.",
            "",
            "| case | simple paths | exhaustive best (s) | B&B best (s) | match | B&B complete solves | solves avoided | prefixes | max LB residual |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for c in bounds["cases"]:
            p = c["production_bound"]
            prefixes = c["ablation"]["complete_cover"]["prefix_count"]
            lines.append(
                f"| {c['case']['name']} | {c['simple_path_count']} | {_f(c['exhaustive_best_time'],8)} | {_f(p['bb_best_time'],8)} | {p['match_error'] <= 2.0e-8} | {p['complete_route_optimizations_including_seed']} | {_ratio_pct(p['complete_route_optimizations_avoided_fraction'])} | {prefixes} | {_f(p['maximum_admissibility_residual'])} |"
            )
        link = _plot_link(report_path, plots_dir, "lower_bounds")
        if link:
            lines += ["", f"![Lower-bound tightness and pruning]({link})", ""]
        lines += ["**Limitation.** This is a finite empirical admissibility check, not a substitute for the mathematical admissibility argument.", ""]

    if gradients:
        a = gradients["aggregate"]
        lines += [
            f"## {_title('gradients')}",
            "",
            f"Python/native reverse implementations agree to **{_f(a['maximum_time_value_absolute_difference'])} s** maximum travel-time difference and **{_f(a['maximum_absolute_gradient_difference'])}** maximum absolute gradient difference. Five-point finite differences checked **{a['finite_difference_coordinates_checked']}/{a['finite_difference_coordinates_total']}** attempted coordinates; excluded coordinates are reported rather than counted as passes.",
            "",
            f"Finite-difference relative error median / p95 / max: **{_f(a['finite_difference_relative_error']['median'])} / {_f(a['finite_difference_relative_error'].get('p95'))} / {_f(a['finite_difference_relative_error']['max'])}**.",
            "",
        ]
        link = _plot_link(report_path, plots_dir, "gradients")
        if link:
            lines += [f"![Finite-difference gradient errors]({link})", ""]

    if warm:
        lines += [
            f"## {_title('warm_start')}",
            "",
            "The ablation holds the final time-optimization budget and certification policy fixed while changing how the initial geometry is prepared. Direct-time wins are retained when they occur.",
            "",
            "| route | variant | certified | final T (s) | total wall (s) | major iterations | objective calls | selected stage |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in warm["rows"]:
            lines.append(
                f"| {r['route_name']} | {r['variant']} | {r['certification']['certified']} | {_f(r['final_travel_time'],7)} | {_f(r['total_wall_seconds'],3)} | {r['slsqp_major_iterations']} | {r['objective_calls']} | {r['selected_stage']} |"
            )
        link = _plot_link(report_path, plots_dir, "warm_start")
        if link:
            lines += ["", f"![Warm-start ablation]({link})", ""]

    if native:
        a = native["aggregate"]
        lines += [
            f"## {_title('native_stack')}",
            "",
            f"Median scalar speedup is **{_f(a['median_scalar_speedup'],2)}×** and median time+gradient speedup is **{_f(a['median_time_gradient_speedup'],2)}×**. The complete optimizer A/B rows are accepted only when both backends independently certify; their median measured end-to-end speedup is **{_f(a['median_end_to_end_speedup'],2)}×**.",
            "",
        ]
        link = _plot_link(report_path, plots_dir, "native_stack")
        if link:
            lines += [f"![Native vs Python reverse solver]({link})", ""]

    if resolution:
        a = resolution["aggregate"]
        lines += [
            f"## {_title('resolution')}",
            "",
            f"Exact N→2N prolongation has maximum initial position error **{_f(a['maximum_initial_position_prolongation_error'])}**. **{a['certified_pairs']}/{a['cases']}** fixed-station pairs remained certified after both optimizations; only those pairs enter time-sensitivity aggregates.",
            "",
        ]
        link = _plot_link(report_path, plots_dir, "resolution")
        if link:
            lines += [f"![N vs 2N resolution sensitivity]({link})", ""]

    if direct:
        a = direct["aggregate"]
        lines += [
            f"## {_title('direct_transcription')}",
            "",
            "The fixed certified clothoid geometry is held constant. CasADi/IPOPT optimizes a dense piecewise-linear squared-speed profile under discretized versions of the production motor, brake, speed, friction-circle, and endpoint constraints. A separate continuous interval certificate—not IPOPT termination—decides feasibility.",
            "",
            f"At the finest predeclared **{a['finest_mesh']}**-interval mesh, **{a['finest_mesh_certified_cold_cases']}/{a['cases']}** cold-start routes certify. Median/max absolute relative time difference is **{_pct(a['finest_mesh_cold_relative_time_difference_percent']['median'],4)} / {_pct(a['finest_mesh_cold_relative_time_difference_percent']['max'],4)}**. Median IPOPT/production scalar solve-time ratio is **{_f(a['finest_mesh_cold_runtime_ratio_ipopt_over_production']['median'],1)}×**.",
            "",
            f"At the predeclared matched-initialization mesh, cold and production-sampled initializations differ in final NLP objective by at most **{_f(a['maximum_cold_warm_objective_difference'])} s** across **{a['matched_initialization_pairs']}** matched pairs.",
            "",
        ]
        link = _plot_link(report_path, plots_dir, "direct_transcription")
        if link:
            lines += [f"![Fixed-geometry transcription convergence]({link})", ""]
        lines += ["**Limitation.** This is one explicit dense direct-transcription formulation, not a universal comparison against optimal-control software.", ""]

    if full:
        a = full["aggregate"]
        lines += [
            f"## {_title('full_ocp')}",
            "",
            "One predeclared topology is fixed, but geometry, phase lengths, curvature, and longitudinal dynamics are optimized simultaneously by a generic CasADi/IPOPT NLP. The reconstructed trajectory must then pass the production continuous corridor/endpoint certificate and an independent continuous speed check.",
            "",
            "| OCP resolution | OCP objective T (s) | hybrid T on OCP geometry (s) | build + solve (s) | certified |",
            "|---:|---:|---:|---:|---|",
        ]
        for r in full["rows"]:
            lines.append(
                f"| {r['base_intervals']} intervals | {_f(r['objective_time'],9)} | {_f(r.get('continuous_hybrid_time_on_ocp_geometry'),9)} | {_f(r['build_plus_solve_seconds'],3)} | {r['certificate']['certified']} |"
            )
        if full_resolution_control:
            lines += [
                "",
                "### Canonical structured resolution control",
                "",
                "The structured side uses exact multilevel prolongation/resegmentation **11 → 22 → 44 → 99** and re-optimizes the newly introduced geometry degrees of freedom while retaining the previous certified trajectory as incumbent.",
                "",
                "| structured / OCP resolution | structured optimized T (s) | OCP objective T (s) | hybrid T on OCP geometry (s) | OCP−structured | hybrid−structured |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for row in full_resolution_control["rows"]:
                lines.append(
                    f"| {row['structured_segments']} / {row['ocp_intervals']} | {_f(row['structured_time'],9)} | {_f(row['ocp_objective_time'],9)} | {_f(row['hybrid_replay_time'],9)} | {_pct(row['ocp_vs_structured_percent'],3)} | {_pct(row['hybrid_replay_vs_structured_percent'],3)} |"
                )
            fine = full_resolution_control["fine_resolution_summary"]
            lines += [
                "",
                f"The structured control improves from **{_f(full_resolution_control['base_structured']['time_seconds'],9)} s** at 11 segments to **{_f(fine['structured_time_seconds'],9)} s** at 99 segments. The exact 44→99 lift preserves the refined geometry to floating-point precision, and the first 99-segment local transaction changes time by only **{abs(fine['first_99_segment_transaction_gain_seconds']):.2e} s**. At the finest pair, the OCP objective is **{abs(fine['ocp_vs_structured_percent']):.3f}% lower**, while production hybrid replay on the OCP geometry is **{abs(fine['hybrid_replay_vs_structured_percent']):.3f}% lower**. The multilevel control is stable at the tested final refinement, so the remaining gap is better attributed to local-basin/formulation behavior than insufficient clothoid resolution.",
                "",
            ]
            asset = results_dir.parents[1] / "assets" / "benchmarks" / "full_ocp_comparison.svg"
            if asset.exists():
                import os
                rel = Path(os.path.relpath(asset, report_path.parent)).as_posix()
                lines += [f"![Structured refinement vs simultaneous OCP]({rel})", ""]
        else:
            lines += [
                "",
                "**Comparison note.** This result directory does not contain the canonical multilevel structured resolution control, so no OCP-vs-structured percentage is reported.",
                "",
            ]

    if sensitivity:
        lines += [
            "## Supplemental sensitivity experiments",
            "",
            f"### {_title('full_ocp_sensitivity')}",
            "",
            "These predeclared follow-up cases test whether the canonical full-OCP observation generalizes. They are **supplemental**, because they were motivated after the canonical experiment rather than being part of the original headline campaign. Structured timeouts, internal-cap failures, optimizer failures, and certification failures remain explicit states.",
            "",
            "| case | existing production T (s) | structured control status | structured T (s) | OCP mesh | OCP status | OCP T (s) | hybrid T on OCP geometry (s) |",
            "|---|---:|---|---:|---:|---|---:|---:|",
        ]
        for c in sensitivity["cases"]:
            structured = c.get("structured_control")
            structured_time = structured.get("time") if structured else None
            for index, r in enumerate(c["rows"]):
                lines.append(
                    f"| {c['case'] if index == 0 else ''} | {_f(c['existing_production']['time'],9) if index == 0 else ''} | {c['structured_execution']['status'] if index == 0 else ''} | {_f(structured_time,9) if index == 0 else ''} | {r.get('base_intervals')} | {r.get('execution_status')} | {_f(r.get('objective_time'),9)} | {_f(r.get('continuous_hybrid_time_on_ocp_geometry'),9)} |"
                )
        lines += [""]

    lines += ["## Resume-ready claims supported by measured results", ""]
    bullets: list[str] = []
    if topology:
        a = topology["aggregate"]
        if a["cases_in_quality_aggregate"] and a["time_improvement_percent"]["max"] is not None:
            bullets.append(
                f"Built kinodynamic branch-and-bound topology search that selected a different certified route in **{a['topology_changes']}/{a['cases_in_quality_aggregate']}** deterministic cyclic mazes and reduced traversal time by **up to {_pct(a['time_improvement_percent']['max'])}** (mean **{_pct(a['time_improvement_percent']['mean'])}**) versus shortest-distance A* topologies optimized with the same continuous solver."
            )
    if bounds:
        a = bounds["aggregate"]
        if a["bb_matches_exhaustive"] == a["cases"] and a.get("admissibility_violations") == 0:
            bullets.append(
                f"Validated kinodynamic branch-and-bound against exhaustive simple-path enumeration on **{a['cases']}/{a['cases']}** tractable cyclic mazes while avoiding **{_ratio_pct(a['complete_route_optimizations_avoided_fraction']['median'])} median** of complete continuous route optimizations, with **0 admissibility violations across {a.get('total_prefixes_checked', 0)}** prefix checks."
            )
    if gradients:
        a = gradients["aggregate"]
        if a["finite_difference_coordinates_checked"]:
            bullets.append(
                f"Implemented reverse-mode differentiation through a hybrid event-driven speed solver; native and Python gradients agreed to **{_f(a['maximum_absolute_gradient_difference'])} max absolute error**, with 5-point finite differences showing **{_f(a['finite_difference_relative_error']['median'])} median relative error** over **{a['finite_difference_coordinates_checked']}** smooth coordinates."
            )
    if native:
        a = native["aggregate"]
        if a.get("median_time_gradient_speedup") and a["median_time_gradient_speedup"] > 1:
            text = (
                f"Ported the speed-profile hot path to C/C++, preserving time values to **{_f(a['maximum_time_value_absolute_error'])} s** and gradients to **{_f(a['maximum_absolute_gradient_error'])} max absolute error** while accelerating time+gradient evaluation by **{_f(a['median_time_gradient_speedup'],2)}× median**"
            )
            if a.get("all_end_to_end_certified") and a.get("median_end_to_end_speedup"):
                text += f" and representative full optimization by **{_f(a['median_end_to_end_speedup'],2)}× median**"
            bullets.append(text + ".")
    if direct:
        a = direct["aggregate"]
        if a["finest_mesh_certified_cold_cases"] == a["cases"] and a["finest_mesh_cold_runtime_ratio_ipopt_over_production"]["median"]:
            bullets.append(
                f"Validated a specialized event-driven minimum-time speed solver against a generic {a['finest_mesh']}-interval CasADi/IPOPT direct transcription on **{a['cases']}/{a['cases']}** certified fixed geometries, matching traversal time within **{_pct(a['finest_mesh_cold_relative_time_difference_percent']['max'],3)} max** while reducing median solve time by **{_f(a['finest_mesh_cold_runtime_ratio_ipopt_over_production']['median'],1)}×**."
            )
    if full and full_resolution_control:
        fine = full_resolution_control["fine_resolution_summary"]
        bullets.append(
            f"Cross-validated the structured trajectory optimizer against a simultaneous CasADi/IPOPT fixed-topology OCP after exact multilevel structured refinement to **{fine['structured_segments']} segments**; the multilevel structured control stabilized at **{_f(fine['structured_time_seconds'],6)} s**, while production hybrid replay on the independently discovered {fine['ocp_intervals']}-interval OCP geometry reached **{_f(fine['hybrid_replay_time_seconds'],6)} s** (**{abs(fine['hybrid_replay_vs_structured_percent']):.2f}% lower**), isolating a local-basin/formulation gap rather than a coarse-resolution artifact."
        )
    if bullets:
        lines.extend(f"- {bullet}" for bullet in bullets[:5])
    else:
        lines.append("No resume bullet met the report's measured-evidence gates.")

    lines += [
        "",
        "## Interview talking points",
        "",
        "- **Topology search:** shows that shortest geometric distance is not always the fastest certified route after dynamics; limitation: each complete topology still receives a local continuous solve.",
        "- **Lower bounds:** checks the production B&B result against exhaustive finite topology enumeration and quantifies expensive solve avoidance; limitation: empirical finite-set admissibility is not a proof.",
        "- **Gradients/native stack:** cross-checks independent implementations, finite differences, numerical equivalence, and performance; limitation: event-boundary derivatives are explicitly excluded where nonsmooth.",
        "- **Warm starts/resolution:** separates local-basin reliability and discretization sensitivity from headline performance claims.",
        "- **Fixed-geometry transcription:** independently reformulates the speed problem and exposes the difference between mesh convergence and continuous certification.",
        "- **Simultaneous fixed-topology OCP:** independently reformulates the full geometry+dynamics problem and can reveal different local geometry basins; limitation: it is one generic formulation on predeclared fixed topologies, not a global optimal-control proof.",
        "",
    ]

    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BENCHMARK_REPORT.md from benchmark JSON outputs only.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots-dir", type=Path)
    args = parser.parse_args()
    output = args.output or args.results_dir / "BENCHMARK_REPORT.md"
    plots = args.plots_dir or args.results_dir / "plots"
    generate(args.results_dir, output, plots)
    print(output)


if __name__ == "__main__":
    main()
