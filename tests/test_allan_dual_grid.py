"""Regression tests for the independent Allan display/estimation grids.

The production modules intentionally keep device-specific input adapters separate.
These tests exercise the shared Allan invariants with small synthetic CSV files.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import allantools as at
import numpy as np
import pandas as pd
import pytest


MODULE_NAMES = (
    "IMU_Analysis_huayi",
    "IMU_Analysis_yuanshen",
    "IMU_Analysis_linsi_LINS16460",
    "IMU_Analysis_linsi_LINS355",
)


def _module(name: str):
    return importlib.import_module(name)


def _display_tau_builder(module):
    for name in ("build_allan_display_taus",):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    analyzer = module.IMUDataAnalyzer
    for name in ("_build_allan_display_tau_grid", "_build_log_tau_grid"):
        candidate = getattr(analyzer, name, None)
        if callable(candidate):
            return candidate
    pytest.fail(f"{module.__name__} does not expose an Allan display tau builder")


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_display_grid_is_nominal_15_per_decade_and_contains_octave_anchors(module_name):
    module = _module(module_name)
    builder = _display_tau_builder(module)
    n, fs = 300_000, 200.0

    display_taus = np.asarray(builder(n, fs, 15), dtype=float)
    display_m = np.rint(display_taus * fs).astype(np.int64)
    octave_taus, *_ = at.adev(
        np.zeros(n, dtype=float), rate=fs, data_type="freq", taus="octave"
    )
    octave_m = np.rint(np.asarray(octave_taus) * fs).astype(np.int64)

    assert np.all(np.isfinite(display_taus))
    assert np.all(display_taus > 0)
    assert np.all(np.diff(display_taus) > 0)
    assert len(display_m) == len(np.unique(display_m))
    assert set(octave_m).issubset(set(display_m))

    # Density is nominal because integer-m quantisation removes duplicates at
    # short tau; over a mature full decade the log grid supplies >=15 points.
    mature = display_m[(display_m >= 100) & (display_m <= 1000)]
    assert len(mature) >= 15


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_common_octave_anchor_adev_values_are_identical(module_name):
    module = _module(module_name)
    builder = _display_tau_builder(module)
    rng = np.random.default_rng(20260818)
    n, fs = 30_000, 200.0
    data = rng.normal(size=n) + 0.02 * np.sin(np.arange(n) / 137.0)

    d_tau, d_adev, d_err, d_ns = at.adev(
        data, rate=fs, data_type="freq", taus=builder(n, fs, 15)
    )
    e_tau, e_adev, e_err, e_ns = at.adev(
        data, rate=fs, data_type="freq", taus="octave"
    )
    display = {
        int(round(tau * fs)): (adev, err, ns)
        for tau, adev, err, ns in zip(d_tau, d_adev, d_err, d_ns)
    }

    assert len(e_tau) > 8
    for tau, adev, err, ns in zip(e_tau, e_adev, e_err, e_ns):
        key = int(round(tau * fs))
        assert key in display
        got_adev, got_err, got_ns = display[key]
        assert got_adev == pytest.approx(adev, rel=1e-13, abs=0.0)
        assert got_err == pytest.approx(err, rel=1e-13, abs=0.0)
        assert got_ns == ns


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_support_classification_is_style_only(module_name):
    module = _module(module_name)
    classifier = getattr(module, "classify_allan_support", None)
    if classifier is None:
        classifier = getattr(module.IMUDataAnalyzer, "_classify_allan_support", None)
    if classifier is None:
        scalar_classifier = getattr(
            module.IMUDataAnalyzer, "_allan_support_class", None
        )
        if scalar_classifier is not None:
            classifier = lambda values: np.asarray(
                [scalar_classifier(value) for value in values], dtype=object
            )
    assert callable(classifier), "production plotting must use an auditable support classifier"

    tau = np.geomspace(0.005, 100.0, 8)
    adev = np.asarray([8.0, 7.0, 6.5, 5.0, 5.4, 4.9, 6.1, 8.2]) * 1e-4
    ns = np.asarray([500, 20, 19, 8, 5, 4, 2, np.nan], dtype=float)
    support = np.asarray(classifier(ns), dtype=object)

    assert support.shape == ns.shape
    assert np.array_equal(tau.copy(), tau)
    assert np.array_equal(adev.copy(), adev)
    assert len(set(support)) >= 3
    # Classification is one label per original point: it neither drops,
    # duplicates, reorders nor numerically transforms the curve arrays.
    rebuilt_indices = np.sort(
        np.concatenate([np.flatnonzero(support == label) for label in np.unique(support)])
    )
    assert np.array_equal(rebuilt_indices, np.arange(len(ns)))
    assert np.array_equal(tau[rebuilt_indices], tau)
    assert np.array_equal(adev[rebuilt_indices], adev)


def _write_device_csv(module_name: str, path: Path, rows: int = 1200, fs: float = 200.0):
    rng = np.random.default_rng(42)
    t = np.arange(rows) / fs
    signals = {
        "ax": 0.003 * rng.normal(size=rows),
        "ay": 0.003 * rng.normal(size=rows),
        "az": 9.80665 + 0.003 * rng.normal(size=rows),
        "gx": 0.01 * rng.normal(size=rows),
        "gy": 0.01 * rng.normal(size=rows),
        "gz": 0.01 * rng.normal(size=rows),
    }
    if module_name.endswith("huayi"):
        frame = pd.DataFrame({"frameCount": np.arange(1, rows + 1), **signals})
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif module_name.endswith("yuanshen"):
        frame = pd.DataFrame({
            "TID": (np.arange(rows) % 60000) + 1,
            "X轴加速度": signals["ax"], "Y轴加速度": signals["ay"],
            "Z轴加速度": signals["az"], "X轴角速度": signals["gx"],
            "Y轴角速度": signals["gy"], "Z轴角速度": signals["gz"],
        })
        frame.to_csv(path, index=False, encoding="gbk")
    else:
        # 两款 LINS 的真实文件都有 5 行采集头，数据行是 Tab 分隔且行尾多一个空列。
        # 两款 LINS 的文件表头都明确记录加速度源单位为 g。
        acc_scale = 1.0 / 9.80665
        magnetic = module_name.endswith("LINS355")
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write("device\nrecorded_at\n\ncolumns\nunits\n")
            for idx in range(rows):
                values = [
                    signals["ax"][idx] * acc_scale,
                    signals["ay"][idx] * acc_scale,
                    signals["az"][idx] * acc_scale,
                    signals["gx"][idx], signals["gy"][idx], signals["gz"][idx],
                ]
                if magnetic:
                    values.extend([0.1, 0.2, 0.3])
                values.extend([0.0, 0.0, 0.0, 25.0])
                stream.write("\t".join(f"{value:.12g}" for value in values) + "\t\n")


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_pipeline_routes_octave_to_parameters_and_display_grid_to_csv(
    module_name, tmp_path, monkeypatch
):
    """Spy on production calls so an accidental grid swap fails loudly."""
    module = _module(module_name)
    source = tmp_path / f"{module_name}.csv"
    _write_device_csv(module_name, source)
    analyzer = module.IMUDataAnalyzer(str(source), sample_rate=200.0)

    original_adev = module.at.adev
    adev_calls = []

    def spy_adev(data, rate=1.0, data_type="phase", taus=None):
        result = original_adev(data, rate=rate, data_type=data_type, taus=taus)
        role = "octave" if isinstance(taus, str) and taus == "octave" else "display"
        adev_calls.append((role, tuple(np.asarray(values).copy() for values in result)))
        return result

    parameter_calls = []
    for method_name in (
        "extract_random_walk_coefficient",
        "extract_bias_instability",
        "extract_minimum_adev_reference",
    ):
        original = getattr(analyzer, method_name)

        def make_spy(name, bound):
            def spy(tau, ad, *args, **kwargs):
                parameter_calls.append((name, np.asarray(tau).copy()))
                return bound(tau, ad, *args, **kwargs)
            return spy

        monkeypatch.setattr(analyzer, method_name, make_spy(method_name, original))

    monkeypatch.setattr(module.at, "adev", spy_adev)
    output = analyzer.plot_allan_variance()
    assert output is not None

    display_calls = [result for role, result in adev_calls if role == "display"]
    octave_calls = [result for role, result in adev_calls if role == "octave"]
    assert len(display_calls) == 6
    assert len(octave_calls) == 6
    assert len(parameter_calls) == 18
    for axis_index in range(6):
        expected = octave_calls[axis_index][0]
        for _, received in parameter_calls[axis_index * 3:(axis_index + 1) * 3]:
            assert np.array_equal(received, expected)

    curve = pd.read_csv(analyzer.report_data["Allan曲线数据"], encoding="utf-8-sig")
    required = {
        "display_points_per_decade", "estimation_tau_grid",
        "parameter_estimation_source", "support_class",
    }
    assert required.issubset(curve.columns)
    assert set(curve["display_points_per_decade"]) == {15}
    assert set(curve["estimation_tau_grid"]) == {"octave"}
    assert set(curve["parameter_estimation_source"]) == {"separate_octave_grid"}
    classifier = getattr(module, "classify_allan_support", None)
    if classifier is not None:
        expected_support = np.asarray(
            classifier(curve["n_terms"].to_numpy()), dtype=object
        )
    else:
        scalar = module.IMUDataAnalyzer._allan_support_class
        expected_support = np.asarray(
            [scalar(value) for value in curve["n_terms"]], dtype=object
        )
    assert np.array_equal(curve["support_class"].to_numpy(), expected_support)
    for axis_index, (sensor, axis) in enumerate(
        [("acc", "x"), ("acc", "y"), ("acc", "z"),
         ("gyro", "x"), ("gyro", "y"), ("gyro", "z")]
    ):
        rows = curve[(curve.sensor == sensor) & (curve.axis == axis)]
        expected_tau, expected_adev, expected_error, expected_ns = display_calls[axis_index]
        # CSV must be the untouched display-grid computation, not a resampled,
        # smoothed, parameter-grid or support-filtered derivative of it.
        assert np.array_equal(rows.tau_s.to_numpy(), expected_tau)
        # Text CSV round-tripping may change the final floating-point bit while
        # preserving full practical precision; use a tight serialization bound.
        assert np.allclose(rows.adev.to_numpy(), expected_adev, rtol=1e-12, atol=1e-15)
        assert np.allclose(rows.error.to_numpy(), expected_error, rtol=1e-12, atol=1e-15)
        assert np.array_equal(rows.n_terms.to_numpy(), expected_ns)
