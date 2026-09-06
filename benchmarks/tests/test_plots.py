from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from benchmarks.plots import gradients_plot, resolution_plot, warm_plot


def _reference(name: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "benchmark_results" / "reference" / f"{name}.json").read_text())


def _assert_white_rgb(path: Path) -> None:
    image = Image.open(path)
    try:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
    finally:
        image.close()


def test_warm_plot_keeps_failed_variants_visible(tmp_path: Path) -> None:
    output = tmp_path / "warm.png"
    summary = warm_plot(_reference("warm_start"), output)
    assert summary == {"rows_plotted": 8, "failed_rows": 4, "certified_rows": 4}
    _assert_white_rgb(output)


def test_resolution_plot_keeps_uncertified_outcomes_visible(tmp_path: Path) -> None:
    output = tmp_path / "resolution.png"
    summary = resolution_plot(_reference("resolution"), output)
    assert summary == {"rows_plotted": 2, "outcomes_plotted": 4, "failed_outcomes": 3, "certified_outcomes": 1}
    _assert_white_rgb(output)


def test_gradient_plot_uses_informative_log_bins(tmp_path: Path) -> None:
    output = tmp_path / "gradients.png"
    summary = gradients_plot(_reference("gradients"), output)
    assert summary["checked_coordinates"] == 58
    assert summary["nonempty_bins"] >= 5
    _assert_white_rgb(output)
