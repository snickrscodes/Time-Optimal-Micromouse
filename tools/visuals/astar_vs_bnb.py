"""Render the final A* vs kinodynamic B&B case-study figure from saved JSON.

This command is postprocessing-only: it never runs A*, B&B, or trajectory
optimization.  It accepts either ``main.py`` metadata or Benchmark-1 JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visualization.case_study import load_case_study, render_astar_vs_bnb_case_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--maze", type=Path, help="override custom maze path embedded in main.py metadata")
    parser.add_argument("--case", help="Benchmark-1 case name; defaults to largest measured improvement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-topology", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    case = load_case_study(args.result, maze_override=args.maze, case_name=args.case)
    render_astar_vs_bnb_case_study(
        case,
        args.output,
        show_topology=not args.no_topology,
        dpi=args.dpi,
    )
    print(args.output)


if __name__ == "__main__":
    main()
