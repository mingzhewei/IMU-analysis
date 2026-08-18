"""Structural regression tests for the Allan figure layout and styling.

These tests inspect Matplotlib artists instead of rendered pixels.  That keeps
them independent of DPI, installed fonts and antialiasing while still guarding
the user-visible contracts introduced for the detailed Allan plot.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest


MODULE_NAMES = (
    "IMU_Analysis_huayi",
    "IMU_Analysis_yuanshen",
)


def _write_device_csv(module_name: str, path: Path, rows: int = 1200,
                      fs: float = 200.0) -> None:
    """Write a small six-axis input accepted by the selected device adapter."""
    rng = np.random.default_rng(20260818)
    t = np.arange(rows, dtype=float) / fs
    signals = {
        "acc_x": 0.003 * rng.normal(size=rows) + 2e-4 * np.sin(2 * np.pi * t),
        "acc_y": 0.003 * rng.normal(size=rows),
        "acc_z": 9.80665 + 0.003 * rng.normal(size=rows),
        "gyro_x": 0.01 * rng.normal(size=rows),
        "gyro_y": 0.01 * rng.normal(size=rows),
        "gyro_z": 0.01 * rng.normal(size=rows),
    }
    frame = pd.DataFrame(signals)
    if module_name.endswith("huayi"):
        frame.insert(0, "frameCount", np.arange(1, rows + 1))
    else:
        frame.insert(0, "TID", np.arange(1, rows + 1))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _capture_allan_figure(module_name: str, tmp_path: Path,
                          monkeypatch: pytest.MonkeyPatch):
    """Run the real small pipeline while retaining its Figure for inspection."""
    module = importlib.import_module(module_name)
    source = tmp_path / f"{module_name}.csv"
    _write_device_csv(module_name, source)
    analyzer = module.IMUDataAnalyzer(str(source), sample_rate=200.0)

    captured: dict[str, matplotlib.figure.Figure] = {}
    def spy_savefig(figure, *args, **kwargs):
        output = str(args[0]) if args else str(kwargs.get("fname", ""))
        if output.endswith("03_零偏稳定性分析图.png"):
            captured["figure"] = figure
        # The test validates artists, not the PNG encoder.  Avoid the expensive
        # raster write without changing the production code path around it.

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy_savefig)
    output = analyzer.plot_allan_variance()

    assert output is not None
    assert "figure" in captured
    return captured["figure"], analyzer


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_allan_plot_has_six_curve_and_six_separate_info_axes(
    module_name, tmp_path, monkeypatch
):
    figure, _ = _capture_allan_figure(module_name, tmp_path, monkeypatch)
    # Use labels as the stable public seam, then assert the expected complete
    # label set independently of Matplotlib's axes creation order.
    curve_by_label = {
        ax.get_label(): ax for ax in figure.axes
        if ax.get_label().startswith("allan_curve_")
    }
    info_by_label = {
        ax.get_label(): ax for ax in figure.axes
        if ax.get_label().startswith("allan_info_")
    }
    assert set(curve_by_label) == {f"allan_curve_{i}" for i in range(6)}
    assert set(info_by_label) == {f"allan_info_{i}" for i in range(6)}

    for index in range(6):
        curve_ax = curve_by_label[f"allan_curve_{index}"]
        info_ax = info_by_label[f"allan_info_{index}"]
        assert curve_ax.get_box_aspect() == pytest.approx(4 / 6)
        assert not info_ax.axison

        info_body = "\n".join(text.get_text() for text in info_ax.texts)
        assert info_body.strip()
        assert any(key in info_body for key in ("VRW", "ARW"))

        # Result prose belongs to the dedicated information axes.  Curve axes
        # may still contain short failure/status labels, but never the full
        # VRW/ARW/BI result block.
        curve_body = "\n".join(text.get_text() for text in curve_ax.texts)
        assert "VRW" not in curve_body
        assert "ARW" not in curve_body


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_allan_curve_lines_have_no_point_markers_and_use_thin_support_styles(
    module_name, tmp_path, monkeypatch
):
    figure, _ = _capture_allan_figure(module_name, tmp_path, monkeypatch)
    curve_axes = [
        ax for ax in figure.axes if ax.get_label().startswith("allan_curve_")
    ]
    assert len(curve_axes) == 6

    for ax in curve_axes:
        # Scatter markers for tau~1 s, BI and the minimum reference are Path
        # Collections, not these support-class Line2D artists.
        support_lines = [line for line in ax.lines if "n_terms" in line.get_label()]
        assert len(support_lines) >= 3
        for line in support_lines:
            assert line.get_marker() in (None, "None", "")
            assert 1.0 <= line.get_linewidth() <= 1.2
        assert any("低支持尾部" in line.get_label() for line in support_lines)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_allan_figure_discloses_support_thresholds_are_engineering_not_standard(
    module_name, tmp_path, monkeypatch
):
    figure, analyzer = _capture_allan_figure(module_name, tmp_path, monkeypatch)
    figure_text = "\n".join(text.get_text() for text in figure.texts)

    assert "20/5" in figure_text
    assert "工程可视化门槛" in figure_text
    assert "非标准规定" in figure_text
    assert "n_terms>=20" in figure_text
    assert "5..19" in figure_text
    assert "<5" in figure_text

    # The generated analysis metadata repeats the disclosure so that textual
    # reports do not lose this qualification when the PNG is viewed separately.
    report_text = "\n".join(str(value) for value in analyzer.report_data.values())
    assert "20/5" in report_text
    assert "工程可视化门槛" in report_text
    assert "非标准规定" in report_text


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_allan_information_text_stays_within_its_own_axes(
    module_name, tmp_path, monkeypatch
):
    """Guard against text/bbox artists widening the tight-saved canvas."""
    figure, _ = _capture_allan_figure(module_name, tmp_path, monkeypatch)
    renderer = figure.canvas.get_renderer()
    info_axes = [
        ax for ax in figure.axes if ax.get_label().startswith("allan_info_")
    ]
    assert len(info_axes) == 6

    for ax in info_axes:
        axes_bbox = ax.get_window_extent(renderer)
        for artist in ax.texts:
            text_bbox = artist.get_window_extent(renderer)
            # One-pixel tolerance absorbs backend rounding only; the prose
            # must remain inside the dedicated information panel.
            assert text_bbox.x0 >= axes_bbox.x0 - 1
            assert text_bbox.x1 <= axes_bbox.x1 + 1
            assert text_bbox.y0 >= axes_bbox.y0 - 1
            assert text_bbox.y1 <= axes_bbox.y1 + 1
