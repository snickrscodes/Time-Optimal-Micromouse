from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from .common.environment import environment_metadata
from .common.io import read_json, sha256_file, write_json
from .common.native import build_native, ensure_native_available
from .config import PROFILES, SUITES
from .orchestration.process import run_python_module
from .schema import validate_benchmark_result, validate_manifest


def _resolve_suites(
    profile: str,
    requested: Iterable[str] | None,
    *,
    topology_available: bool = False,
) -> tuple[str, ...]:
    selected = list(PROFILES[profile].suites if not requested else requested)
    resolved: list[str] = []
    for name in selected:
        definition = SUITES[name]
        if (
            definition.requires_topology
            and "topology_search" not in resolved
            and "topology_search" not in selected
            and not topology_available
        ):
            resolved.append("topology_search")
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def _suite_command(name: str, *, numerical_profile: str, output_dir: Path) -> tuple[str, list[str]]:
    definition = SUITES[name]
    args = ["--profile", numerical_profile, "--output-dir", str(output_dir)]
    if definition.module.startswith("benchmarks.orchestration."):
        return definition.module, args
    return "benchmarks.orchestration.worker", ["--suite", name, *args]


def _unique_run_directory(root: Path, profile_name: str) -> Path:
    base = root / "benchmark_results" / "runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / f"{stamp}_{profile_name}"
    counter = 2
    while candidate.exists():
        candidate = base / f"{stamp}_{profile_name}_{counter}"
        counter += 1
    return candidate


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _ensure_ocp_dependency(suites: Iterable[str]) -> None:
    if not any(SUITES[name].requires_ocp_dependency for name in suites):
        return
    try:
        version("casadi")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "the selected benchmark profile requires CasADi/IPOPT; install "
            "`requirements-benchmarks-ocp.txt`"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic AME robot benchmarks in isolated processes."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="core")
    parser.add_argument(
        "--suite",
        action="append",
        choices=tuple(SUITES),
        help="run one suite; repeat to select multiple suites (overrides profile suite list)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="regenerate the checked-in benchmark_results/reference campaign; requires --profile all",
    )
    parser.add_argument("--build-native", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse existing validated suite JSON and run only missing suites")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="alias for --profile smoke")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.jobs != 1:
        parser.error("official timing suites run serially; --jobs must be 1")
    if args.reference and (args.output_dir is not None or args.quick):
        parser.error("--reference cannot be combined with --output-dir or --quick")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    profile_name = "smoke" if args.quick else args.profile
    if args.reference and profile_name != "all":
        parser.error("--reference requires --profile all")

    profile = PROFILES[profile_name]
    root = Path(__file__).resolve().parents[1]
    if args.reference:
        output_dir = root / "benchmark_results" / "reference"
    elif args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = _unique_run_directory(root, profile_name)

    if args.build_native:
        build_native(root)
    ensure_native_available(root)

    suites = _resolve_suites(
        profile_name,
        args.suite,
        topology_available=(output_dir / "topology_search.json").exists(),
    )
    _ensure_ocp_dependency(suites)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and not args.resume:
        collisions = [name for name in suites if (output_dir / f"{name}.json").exists()]
        if collisions:
            parser.error(
                "result files already exist for " + ", ".join(collisions) +
                "; pass --overwrite, --resume, or choose a different --output-dir"
            )

    env_metadata = environment_metadata(root=root, include_reverse_backend=False)
    provenance = env_metadata.get("provenance", {})
    if provenance.get("git_dirty"):
        print("[benchmarks] WARNING: running from a dirty Git working tree", file=sys.stderr, flush=True)

    run_id = output_dir.name
    manifest = {
        "schema_version": 1,
        "benchmark": "manifest",
        "status": "complete",
        "profile": profile_name,
        "numerical_profile": profile.numerical_profile,
        "reference_run": bool(args.reference),
        "run_id": run_id,
        "requested_suites": list(args.suite or ()),
        "resolved_suites": list(suites),
        "environment": env_metadata,
        "runs": [],
    }

    for name in suites:
        definition = SUITES[name]
        existing_output = output_dir / f"{name}.json"
        if args.resume and existing_output.exists():
            validate_benchmark_result(
                read_json(existing_output),
                expected_benchmark=name,
                expected_profile=profile.numerical_profile,
            )
            manifest["runs"].append(
                {
                    "suite": name,
                    "public_title": definition.public_title,
                    "tier": definition.tier,
                    "reused": True,
                    "orchestration_wall_seconds": 0.0,
                    "returncode": 0,
                    "completion_marker_seen": None,
                    "forced_teardown": False,
                    "output": _relative_or_absolute(existing_output, root),
                    "sha256": sha256_file(existing_output),
                }
            )
            print(f"[benchmarks] reusing validated {definition.public_title}", flush=True)
            continue
        timeout = (
            definition.timeout_smoke_seconds
            if profile.numerical_profile == "smoke"
            else definition.timeout_core_seconds
        )
        module, worker_args = _suite_command(
            name, numerical_profile=profile.numerical_profile, output_dir=output_dir
        )
        completion_marker = output_dir / ".components" / f"{name}.suite.done"
        worker_args = [*worker_args, "--done", str(completion_marker)]
        print(f"[benchmarks] running {definition.public_title} ({profile.numerical_profile})", flush=True)
        started = time.perf_counter()
        result = run_python_module(
            module, worker_args, cwd=root, timeout=timeout, completion_marker=completion_marker
        )
        elapsed = time.perf_counter() - started
        output = output_dir / f"{name}.json"
        if not output.exists():
            raise RuntimeError(f"suite {name} exited successfully but did not produce {output}")
        validate_benchmark_result(
            read_json(output), expected_benchmark=name, expected_profile=profile.numerical_profile
        )
        manifest["runs"].append(
            {
                "suite": name,
                "public_title": definition.public_title,
                "tier": definition.tier,
                "reused": False,
                "orchestration_wall_seconds": float(elapsed),
                "returncode": result.returncode,
                "completion_marker_seen": result.completion_marker_seen,
                "forced_teardown": result.forced_teardown,
                "output": _relative_or_absolute(output, root),
                "sha256": sha256_file(output),
            }
        )
        print(f"[benchmarks] {name}: {elapsed:.3f}s orchestration wall", flush=True)

    if not args.no_plots:
        from .plots import generate_all
        plots_dir = output_dir / "plots"
        manifest["plots"] = {
            name: _relative_or_absolute(Path(path), root)
            for name, path in generate_all(output_dir, plots_dir).items()
        }
    if not args.no_report:
        from .report import generate
        report_path = output_dir / "BENCHMARK_REPORT.md"
        generate(output_dir, report_path, output_dir / "plots")
        manifest["report"] = _relative_or_absolute(report_path, root)
        manifest["report_sha256"] = sha256_file(report_path)

    validate_manifest(manifest)
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": _relative_or_absolute(output_dir, root),
                "reference_run": bool(args.reference),
                "runs": manifest["runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
