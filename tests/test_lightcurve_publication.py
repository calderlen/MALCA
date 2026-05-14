from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from malca.lightcurve_publication import (
    filter_lightcurve,
    load_lightcurve,
    normalize_lightcurve_frame,
    plot_lightcurve_panel,
    plot_lightcurve,
    plot_phase_panel,
    plot_residual_panel,
    resolve_time_axis,
)


def test_generic_csv_aliases_are_normalized(tmp_path):
    csv_path = tmp_path / "generic.csv"
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0, 59003.0],
            "magnitude": [14.1, 14.2, 14.3, 14.4],
            "mag_err": [0.02, 0.03, 5.0, 0.02],
            "passband": ["g", "V", "g", "V"],
            "camera_id": ["a", "a", "b", "b"],
            "quality": ["G", "G", "G", "B"],
        }
    ).to_csv(csv_path, index=False)

    lc = load_lightcurve(csv_path)
    filtered = filter_lightcurve(lc)

    assert lc.time_column == "mjd"
    assert lc.y_kind == "mag"
    assert filtered["band"].tolist() == ["g", "V"]
    assert filtered["camera"].tolist() == ["a", "a"]


def test_dat2_file_uses_existing_loader(tmp_path):
    dat_path = tmp_path / "123.dat2"
    dat_path.write_text(
        "\n".join(
            [
                "7479.8 14.10 0.02 1 4 0 0 ba/F1",
                "7480.8 14.20 0.03 1 5 1 0 bb/F1",
            ]
        )
    )

    lc = load_lightcurve(dat_path)
    filtered = filter_lightcurve(lc)

    assert filtered["band"].tolist() == ["g", "V"]
    assert filtered["camera"].tolist() == ["4", "5"]


def test_resolve_time_axis_auto_for_full_jd():
    plotted, label = resolve_time_axis(
        pd.Series([2457000.0, 2457001.0]),
        source_column="JD",
        offset="auto",
    )

    assert label == "JD - 2450000"
    assert np.allclose(plotted, [7000.0, 7001.0])


def test_plot_lightcurve_writes_output(tmp_path):
    csv_path = tmp_path / "sky.csv"
    pd.DataFrame(
        {
            "JD": [2457000.0, 2457001.0, 2457002.0],
            "Mag": [13.0, 13.1, 13.05],
            "Mag Error": [0.01, 0.02, 0.02],
            "Filter": ["g", "g", "V"],
            "Quality": ["G", "G", "G"],
            "Camera": ["ba", "ba", "bb"],
        }
    ).to_csv(csv_path, index=False)
    lc = load_lightcurve(csv_path)
    filtered = filter_lightcurve(lc)
    output = tmp_path / "plot.png"

    fig, ax = plot_lightcurve(lc, filtered, output=output, title="", group_by="band")

    assert output.exists()
    assert output.stat().st_size > 0
    assert fig is ax.figure
    plt.close(fig)


def test_direct_dataframe_normalization_and_axes_reuse():
    frame = pd.DataFrame(
        {
            "JD": [2457000.0, 2457001.0, 2457002.0],
            "mag": [13.0, 13.2, 13.1],
            "error": [0.02, 0.03, 0.02],
            "camera_field": ["ba/F1", "ba/F1", "bb/F2"],
        }
    )
    lc = normalize_lightcurve_frame(frame.rename(columns={"camera_field": "camera"}), "inline")
    assert len(lc.df) == 3

    fig, ax = plt.subplots()
    result = plot_lightcurve_panel(
        ax,
        frame,
        group_by="group",
        group_col="camera_field",
        show_errorbars=True,
        legend="none",
    )

    assert result.ax is ax
    assert "time_plot" in result.frame.columns
    assert len(ax.collections) + len(ax.lines) > 0
    plt.close(fig)


def test_panel_overlays_baseline_events_and_highlights():
    frame = pd.DataFrame(
        {
            "JD": [2457000.0, 2457001.0, 2457002.0, 2457003.0],
            "mag": [13.0, 13.1, 13.4, 13.05],
            "error": [0.02, 0.02, 0.02, 0.02],
            "v_g_band": [0, 0, 0, 0],
            "baseline": [13.0, 13.0, 13.0, 13.0],
        }
    )
    fig, ax = plt.subplots()
    result = plot_lightcurve_panel(
        ax,
        frame,
        baseline=frame,
        baseline_col="baseline",
        vertical_lines=[2457002.0],
        vertical_spans=[(2457001.5, 2457002.5)],
        event_runs=[{"start_jd": 2457001.8, "end_jd": 2457002.2, "params": {"t0": 2457002.0}}],
        highlight_mask=[False, False, True, False],
        legend="none",
    )

    assert result.frame["plot_group"].tolist() == ["g", "g", "g", "g"]
    assert len(ax.lines) >= 4
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_residual_panel_renders_thresholds():
    frame = pd.DataFrame(
        {
            "JD": [2457000.0, 2457001.0, 2457002.0],
            "resid": [0.0, 0.25, -0.15],
            "error": [0.02, 0.02, 0.02],
            "camera#": [1, 1, 2],
        }
    )
    fig, ax = plt.subplots()
    result = plot_residual_panel(
        ax,
        frame,
        threshold=0.2,
        group_by="camera",
        show_errorbars=True,
        legend="none",
    )

    assert result.ax is ax
    assert ax.get_ylabel() == "Residual [mag]"
    assert len(ax.lines) >= 3
    plt.close(fig)


def test_phase_panel_uses_shared_phase_fold_helper():
    frame = pd.DataFrame(
        {
            "JD": [2457000.0, 2457000.5, 2457001.0, 2457001.5],
            "mag": [13.0, 13.2, 13.0, 13.2],
            "error": [0.02, 0.02, 0.02, 0.02],
            "v_g_band": [0, 0, 1, 1],
            "camera#": [1, 1, 2, 2],
        }
    )
    fig, ax = plt.subplots()
    result = plot_phase_panel(
        ax,
        frame,
        period_days=1.0,
        align_v_to_g=False,
        legend="none",
    )

    assert result.diagnostics is not None
    assert result.diagnostics["period_days"] == 1.0
    assert result.frame["time_plot"].between(0.0, 2.0).all()
    plt.close(fig)
