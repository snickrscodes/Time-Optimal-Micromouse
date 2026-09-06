from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(input_json: Path, output_svg: Path) -> None:
    import matplotlib.pyplot as plt

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    labels = [f"{r['structured_segments']} / {r['ocp_intervals']}" for r in rows]
    x = list(range(len(rows)))
    structured = [r["structured_time"] for r in rows]
    ocp = [r["ocp_objective_time"] for r in rows]
    replay = [r["hybrid_replay_time"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(x, structured, marker="o", linewidth=2.0, label="Structured multilevel control")
    ax.plot(x, ocp, marker="s", linewidth=2.0, label="Simultaneous OCP objective")
    ax.plot(x, replay, marker="^", linewidth=2.0, label="Hybrid replay on OCP geometry")

    ax.set_xticks(x, labels)
    ax.set_xlabel("Structured segments / OCP intervals")
    ax.set_ylabel("Certified traversal time (s)")
    ax.set_title("Fixed-topology OCP comparison after structured resolution control")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)

    lo = min(structured + ocp + replay)
    hi = max(structured + ocp + replay)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    fine = payload["fine_resolution_summary"]
    ax.annotate(
        f"Hybrid replay: {abs(fine['hybrid_replay_vs_structured_percent']):.2f}% lower",
        xy=(x[-1], replay[-1]),
        xytext=(-8, -24),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    ax.text(
        0.01,
        0.01,
        "All plotted trajectories independently certified; local solutions only.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
    )

    fig.tight_layout()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Render the release OCP comparison figure.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "benchmark_results/reference/full_ocp_resolution_control.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "assets/benchmarks/full_ocp_comparison.svg",
    )
    args = parser.parse_args()
    render(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
