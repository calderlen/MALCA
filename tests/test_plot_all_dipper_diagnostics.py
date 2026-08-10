from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.stv import dimming_window


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_all_dipper_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("plot_all_dipper_diagnostics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


def test_fractional_depth_delta_mag_conversion_round_trips() -> None:
    fractional_depth = np.array([0.0, 0.1, 0.25, 0.6])

    delta_mag = DIAGNOSTICS._fractional_depth_to_delta_mag(fractional_depth)

    assert delta_mag == pytest.approx(
        [0.0, 0.11439373, 0.31234684, 0.99485002]
    )
    assert DIAGNOSTICS._delta_mag_to_fractional_depth(delta_mag) == pytest.approx(
        fractional_depth
    )


def test_pipeline_run_overlay_metrics_keep_run_and_atlas_definitions_separate() -> None:
    runs = pd.DataFrame(
        {
            "run_start_jd": [100.0, 160.0, 300.0],
            "run_end_jd": [110.0, 170.0, 310.0],
            "trigger_jds_json": ["[101.0, 102.0]", "[161.0]", "[301.0]"],
        }
    )

    metrics = DIAGNOSTICS._pipeline_run_overlay_metrics(
        runs,
        event_start_jd=105.0,
        event_end_jd=165.0,
        peak_jd=108.0,
    )

    assert metrics == {
        "pipeline_dip_run_count": 3,
        "pipeline_dip_runs_overlapping_complex": 2,
        "pipeline_trigger_point_count": 4,
        "atlas_peak_inside_pipeline_dip_run": True,
    }


def test_read_pipeline_dip_runs_filters_candidates_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "triggered_dip_runs.parquet"
    pd.DataFrame(
        {
            "event_id": ["a:0", "b:0"],
            "candidate_id": ["a", "b"],
            "run_number": [0, 0],
            "run_start_jd": [100.0, 200.0],
            "run_end_jd": [101.0, 202.0],
            "dip_jd": [100.5, 201.0],
            "n_trigger_points": [2, 3],
            "run_peak_event_probability": [0.9, 0.8],
            "trigger_jds_json": ["[100.0, 101.0]", "[200.0, 201.0, 202.0]"],
            "detector_commit": ["deadbeef", "deadbeef"],
            "run_table_schema_version": [2, 2],
        }
    ).to_parquet(replay_path, index=False)

    selected = DIAGNOSTICS.read_pipeline_dip_runs(replay_path, ["b"])

    assert selected["candidate_id"].tolist() == ["b"]
    assert selected["event_id"].tolist() == ["b:0"]
    assert selected.attrs["source_path"] == str(replay_path.resolve())


def test_dimming_complex_duration_is_separate_and_censor_aware() -> None:
    bounded = DIAGNOSTICS._dimming_complex_metrics(
        100.0,
        120.0,
        "baseline_bounded",
        False,
    )
    assert bounded["dimming_complex_duration_days"] == pytest.approx(20.0)
    assert bounded["dimming_complex_duration_lower_days"] == pytest.approx(20.0)
    assert bounded["dimming_complex_duration_upper_days"] == pytest.approx(20.0)
    assert bounded["dimming_complex_is_lower_limit"] is False
    assert bounded["dimming_complex_status"] == "recovery_bounded"

    ongoing = DIAGNOSTICS._dimming_complex_metrics(
        100.0,
        1_144.12,
        "ongoing_right_censored",
        True,
    )
    assert ongoing["dimming_complex_duration_days"] == pytest.approx(1_044.12)
    assert ongoing["dimming_complex_duration_lower_days"] == pytest.approx(1_044.12)
    assert np.isnan(ongoing["dimming_complex_duration_upper_days"])
    assert ongoing["dimming_complex_is_lower_limit"] is True
    assert ongoing["dimming_complex_status"] == "right_censored"


@pytest.mark.parametrize(
    ("sed_alpha_class", "expected"),
    [
        ("Class II", "Class II"),
        ("Class III/photosphere", "Class III/photosphere"),
        ("Flat", "Flat"),
        ("Class I", "Class I"),
        ("unknown", "Unknown"),
        (None, "Unknown"),
        ("unexpected", "Unknown"),
    ],
)
def test_display_sed_alpha_class_maps_known_labels(
    sed_alpha_class: str | None,
    expected: str,
) -> None:
    assert DIAGNOSTICS._display_sed_alpha_class(sed_alpha_class) == expected


def test_fwhm_proxy_bounds_do_not_promote_interval_midpoint() -> None:
    frame = pd.DataFrame(
        {
            "duration_status": ["resolved", "interval_censored", "right_censored"],
            "duration_plot_days": [10.0, 20.0, 30.0],
            "duration_lower_days": [10.0, 12.0, 30.0],
            "duration_upper_days": [10.0, 40.0, np.nan],
            "duration_is_lower_limit": [False, False, True],
            "tau_peak": [0.2, 0.2, 0.2],
        }
    )

    result = DIAGNOSTICS.add_fwhm_proxy_bounds(frame)

    assert result.loc[0, "a_proxy_lower_au"] == pytest.approx(
        result.loc[0, "a_proxy_upper_au"]
    )
    assert result.loc[1, "a_proxy_lower_au"] < result.loc[1, "a_proxy_upper_au"]
    assert result.loc[1, "p_ecl_proxy_lower"] < result.loc[1, "p_ecl_proxy_upper"]
    assert result.loc[1, "fwhm_proxy_duration_lower_days"] == pytest.approx(12.0)
    assert result.loc[1, "fwhm_proxy_duration_upper_days"] == pytest.approx(40.0)
    assert np.isnan(result.loc[2, "a_proxy_upper_au"])
    assert result.loc[2, "p_ecl_proxy_lower"] == pytest.approx(0.0)
    assert np.isfinite(result.loc[2, "p_ecl_proxy_upper"])


def test_persistent_recovery_requires_five_sixths_for_seven_days_in_one_block() -> None:
    dense_times = np.arange(8, dtype=float)
    five_sixths_dense = np.array([True, True, True, False, True, True, True, True])
    assert DIAGNOSTICS._persistent_recovery_windows(
        dense_times,
        five_sixths_dense,
        0,
        7,
        5.0,
    ) == [(0, 7)]

    spaced_times = np.arange(6, dtype=float) * 1.5
    assert DIAGNOSTICS._persistent_recovery_windows(
        spaced_times,
        np.array([True, True, False, True, False, True]),
        0,
        5,
        5.0,
    ) == []
    assert DIAGNOSTICS._persistent_recovery_windows(
        np.arange(6, dtype=float),
        np.array([True, True, True, False, True, True]),
        0,
        5,
        5.0,
    ) == []
    assert DIAGNOSTICS._persistent_recovery_windows(
        np.array([0.0, 1.5, 3.0, 12.0, 13.5, 15.0]),
        np.array([True, True, True, False, True, True]),
        0,
        5,
        5.0,
    ) == []


def test_persistent_half_depth_resolves_after_five_of_six_recovery() -> None:
    times = np.arange(19, dtype=float) * 1.5
    residual = np.zeros_like(times)
    residual[6:13] = 0.10
    residual[15] = 0.10  # one tolerated discordant night in the recovery run

    result = DIAGNOSTICS._persistent_half_depth_measurement(
        times,
        residual,
        residual,
        half_level=0.05,
        anchor=9,
        event_start=0,
        event_stop=len(times) - 1,
        crossing_gap_limit_days=5.0,
        left_event_boundary_type="recovery",
        right_event_boundary_type="recovery",
    )

    assert result["left"]["status"] == "exact"
    assert result["right"]["status"] == "exact"
    assert result["left"]["point"] == pytest.approx(8.25)
    assert result["right"]["point"] == pytest.approx(18.75)
    assert result["duration_status"] == "resolved"
    assert result["duration_plot_days"] == pytest.approx(10.5)


def test_persistent_half_depth_returns_interval_for_gap_hidden_crossing() -> None:
    times = np.r_[np.arange(6, dtype=float) * 1.5, np.arange(9.0, 19.5, 1.5), np.arange(100.0, 109.0, 1.5)]
    residual = np.zeros_like(times)
    residual[6:13] = 0.10

    result = DIAGNOSTICS._persistent_half_depth_measurement(
        times,
        residual,
        residual,
        half_level=0.05,
        anchor=9,
        event_start=0,
        event_stop=len(times) - 1,
        crossing_gap_limit_days=5.0,
        left_event_boundary_type="recovery",
        right_event_boundary_type="recovery",
    )

    assert result["left"]["status"] == "exact"
    assert result["right"]["status"] == "interval"
    assert result["right"]["lower"] == pytest.approx(18.0)
    assert result["right"]["upper"] == pytest.approx(100.0)
    assert result["duration_status"] == "interval_censored"
    assert result["duration_lower_days"] == pytest.approx(9.0)
    assert result["duration_upper_days"] == pytest.approx(92.5)


def test_persistent_half_depth_keeps_true_data_edge_open() -> None:
    times = np.arange(21, dtype=float) * 1.5
    residual = np.zeros_like(times)
    residual[6:] = 0.10

    result = DIAGNOSTICS._persistent_half_depth_measurement(
        times,
        residual,
        residual,
        half_level=0.05,
        anchor=12,
        event_start=0,
        event_stop=len(times) - 1,
        crossing_gap_limit_days=5.0,
        left_event_boundary_type="recovery",
        right_event_boundary_type="data_edge",
    )

    assert result["left"]["status"] == "exact"
    assert result["right"]["status"] == "censored"
    assert result["duration_status"] == "right_censored"
    assert result["duration_lower_days"] == pytest.approx(21.75)


def test_fwhm_duration_depth_plot_uses_tau_axis_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "duration_plot_days": [10.0, 25.0],
            "tau_peak": [0.2, 0.15],
            "plot_class": ["Class II", "Class III/photosphere"],
            "duration_status": ["resolved", "interval_censored"],
            "duration_is_interval_censored": [False, True],
            "duration_is_lower_limit": [False, False],
            "duration_lower_days": [10.0, 12.0],
            "duration_upper_days": [10.0, 40.0],
            "duration_mc_reporting_status": ["not_evaluated", "structurally_censored"],
            "duration_mc_err_minus": [np.nan, np.nan],
            "duration_mc_err_plus": [np.nan, np.nan],
            "tau_peak_mc_err_minus": [0.01, 0.01],
            "tau_peak_mc_err_plus": [0.02, 0.02],
        }
    )
    captured: dict[str, object] = {}

    def capture_figure(fig, *_args, **_kwargs) -> None:
        fig.canvas.draw()
        captured["figure"] = fig

    monkeypatch.setattr(DIAGNOSTICS, "_save_figure", capture_figure)

    coverage = DIAGNOSTICS.plot_fwhm_duration_depth(frame, tmp_path)

    fig = captured["figure"]
    primary_axis = fig.axes[0]
    assert primary_axis.get_xlabel() == r"$\tau_{\mathrm{FWHM}}$ [days]"
    assert primary_axis.get_xlim() == pytest.approx(
        (DIAGNOSTICS.DURATION_DEPTH_XMIN_DAYS, DIAGNOSTICS.DURATION_DEPTH_XMAX_DAYS)
    )
    from matplotlib.collections import PathCollection

    marker_count = sum(
        collection.get_offsets().shape[0]
        for collection in primary_axis.collections
        if isinstance(collection, PathCollection)
    )
    assert marker_count == 2
    assert coverage["resolved"] == 1
    assert coverage["interval_censored"] == 1
    DIAGNOSTICS.plt.close(fig)


def test_eclipse_probability_panel_is_square_and_secondary_axis_is_vector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "duration_plot_days": [10.0],
            "tau_peak": [0.2],
            "plot_class": ["Class III/photosphere"],
            "duration_status": ["resolved"],
            "duration_is_interval_censored": [False],
            "duration_is_lower_limit": [False],
            "duration_lower_days": [10.0],
            "duration_upper_days": [10.0],
            "duration_mc_err_minus": [np.nan],
            "duration_mc_err_plus": [np.nan],
            "tau_peak_mc_err_minus": [np.nan],
            "tau_peak_mc_err_plus": [np.nan],
        }
    )
    captured: dict[str, object] = {}

    def capture_figure(fig, *_args, **_kwargs) -> None:
        fig.canvas.draw()
        captured["figure"] = fig

    monkeypatch.setattr(DIAGNOSTICS, "_save_figure", capture_figure)

    DIAGNOSTICS.plot_eclipse_probability_proxy(frame, tmp_path)

    fig = captured["figure"]
    primary_axis = fig.axes[0]
    panel = primary_axis.get_position()
    panel_width = panel.width * fig.get_figwidth()
    panel_height = panel.height * fig.get_figheight()
    assert panel_width == pytest.approx(panel_height, abs=0.02)
    assert primary_axis.get_xlim() == pytest.approx(
        (
            DIAGNOSTICS.DURATION_DEPTH_XMIN_DAYS,
            DIAGNOSTICS.DURATION_DEPTH_XMAX_DAYS,
        )
    )

    secondary_axes = [
        child
        for child in primary_axis.get_children()
        if type(child).__name__ == "SecondaryAxis"
    ]
    assert len(secondary_axes) == 1
    assert secondary_axes[0].get_zorder() > 1
    assert secondary_axes[0].get_ylabel() == r"$\Delta m$ [mag]"

    renderer = fig.canvas.get_renderer()
    ylabel_bounds = primary_axis.yaxis.label.get_window_extent(renderer)
    assert ylabel_bounds.x0 >= 0.05 * fig.dpi
    legend = primary_axis.get_legend()
    assert {text.get_fontsize() for text in legend.get_texts()} == {5.4}
    legend_bounds = legend.get_window_extent(renderer)
    probability_1e4_label = next(
        text for text in primary_axis.texts if text.get_text() == r"$10^{-4}$"
    )
    assert not legend_bounds.overlaps(probability_1e4_label.get_window_extent(renderer))
    DIAGNOSTICS.plt.close(fig)


def _measure_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    times: np.ndarray,
    residual_mag: np.ndarray,
    *,
    sigma_mag: float = 0.008,
) -> dict[str, object]:
    """Run the public estimator while replacing only light-curve I/O and GP fitting."""
    frame = pd.DataFrame(
        {
            "JD": np.asarray(times, dtype=float),
            "resid": np.asarray(residual_mag, dtype=float),
            "sigma_eff": np.full(len(times), sigma_mag, dtype=float),
        }
    )
    monkeypatch.setattr(dimming_window, "load_lightcurve_df", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(dimming_window, "to_asassn_algorithm_frame", lambda data: data.copy())
    monkeypatch.setattr(dimming_window, "clean_lc", lambda data: data.copy())
    monkeypatch.setattr(
        dimming_window,
        "per_camera_gp_baseline_masked",
        lambda data, **kwargs: data.copy(),
    )
    return DIAGNOSTICS.measure_half_depth_event(
        "synthetic",
        Path("unused.csv"),
        include_trace=True,
    )


def _assert_measurement_invariants(result: dict[str, object]) -> None:
    assert result["measurement_error"] == ""
    status = result["event_window_status"]
    trace = result["_trace"]
    assert trace is not None
    times = trace["epochs"]["t"].to_numpy(float)
    event_start = int(trace["event_start"])
    event_stop = int(trace["event_stop"])

    assert result["uncertainty_method"] == "conditional_nightly_parametric_mc_v2"
    assert result["uncertainty_draws"] == DIAGNOSTICS.DEFAULT_EVENT_MC_DRAWS
    for field in ("tau_peak_mc_err_minus", "tau_peak_mc_err_plus"):
        assert np.isfinite(result[field])
        assert result[field] >= 0

    assert 0 <= event_start <= int(trace["anchor_index"]) <= event_stop < len(times)
    assert result["event_window_start_jd"] == pytest.approx(times[event_start])
    assert result["event_window_end_jd"] == pytest.approx(times[event_stop])
    assert result["event_window_duration_days"] == pytest.approx(
        times[event_stop] - times[event_start]
    )
    assert result["event_metrics_schema_version"] == DIAGNOSTICS.EVENT_METRICS_SCHEMA_VERSION
    assert result["dimming_complex_start_jd"] == pytest.approx(times[event_start])
    assert result["dimming_complex_end_jd"] == pytest.approx(times[event_stop])
    assert result["dimming_complex_duration_days"] == pytest.approx(
        result["event_window_duration_days"]
    )
    assert result["dimming_complex_duration_lower_days"] == pytest.approx(
        result["event_window_duration_days"]
    )

    if status == "baseline_bounded":
        assert result["left_baseline_recovered"] is True
        assert result["right_baseline_recovered"] is True
        assert result["left_event_boundary_type"] == "recovery"
        assert result["right_event_boundary_type"] == "recovery"
        assert result["event_window_is_lower_limit"] is False
        assert result["dimming_complex_is_lower_limit"] is False
        assert result["dimming_complex_status"] == "recovery_bounded"
        assert result["dimming_complex_duration_upper_days"] == pytest.approx(
            result["event_window_duration_days"]
        )
        assert bool(trace["recovery_mask"][event_start])
        assert bool(trace["recovery_mask"][event_stop])
        assert result["left_recovery_jd"] == pytest.approx(times[event_start])
        assert result["right_recovery_jd"] == pytest.approx(times[event_stop])
    elif status == "ongoing_right_censored":
        assert result["left_baseline_recovered"] is True
        assert result["right_baseline_recovered"] is False
        assert result["left_edge_dim_confirmed"] is False
        assert result["right_edge_dim_confirmed"] is True
        assert result["left_event_boundary_type"] == "recovery"
        assert result["right_event_boundary_type"] == "data_edge"
        assert result["event_window_is_lower_limit"] is True
        assert bool(trace["recovery_mask"][event_start])
        assert result["left_recovery_jd"] == pytest.approx(times[event_start])
        assert result["event_window_end_jd"] == pytest.approx(times[-1])
        assert trace["epochs"]["resid"].to_numpy(float)[-1] >= result["detection_threshold_mag"]
    elif status == "ongoing_left_censored":
        assert result["left_baseline_recovered"] is False
        assert result["right_baseline_recovered"] is True
        assert result["left_edge_dim_confirmed"] is True
        assert result["right_edge_dim_confirmed"] is False
        assert result["left_event_boundary_type"] == "data_edge"
        assert result["right_event_boundary_type"] == "recovery"
        assert result["event_window_is_lower_limit"] is True
        assert bool(trace["recovery_mask"][event_stop])
        assert result["right_recovery_jd"] == pytest.approx(times[event_stop])
        assert result["event_window_start_jd"] == pytest.approx(times[0])
        assert trace["epochs"]["resid"].to_numpy(float)[0] >= result["detection_threshold_mag"]
    elif status == "left_recovery_unconfirmed":
        assert result["left_baseline_recovered"] is False
        assert result["right_baseline_recovered"] is True
        assert result["left_event_boundary_type"] == "unconfirmed_recovery"
        assert result["right_event_boundary_type"] == "recovery"
        assert bool(trace["recovery_mask"][event_start])
        assert bool(trace["right_recovery_anchor_mask"][event_stop])
        assert result["event_window_is_lower_limit"] is True
    elif status == "right_recovery_unconfirmed":
        assert result["left_baseline_recovered"] is True
        assert result["right_baseline_recovered"] is False
        assert result["left_event_boundary_type"] == "recovery"
        assert result["right_event_boundary_type"] == "unconfirmed_recovery"
        assert bool(trace["left_recovery_anchor_mask"][event_start])
        assert bool(trace["recovery_mask"][event_stop])
        assert result["event_window_is_lower_limit"] is True
    elif status == "right_gap_censored":
        assert result["left_baseline_recovered"] is True
        assert result["right_baseline_recovered"] is False
        assert result["left_event_boundary_type"] == "recovery"
        assert result["right_event_boundary_type"] == "gap"
        assert result["right_gap_boundary_state"] in {"baseline", "dim"}
        assert result["event_window_is_lower_limit"] is True
        assert bool(trace["recovery_mask"][event_start])
        assert result["left_recovery_jd"] == pytest.approx(times[event_start])
    elif status == "left_gap_censored":
        assert result["left_baseline_recovered"] is False
        assert result["right_baseline_recovered"] is True
        assert result["left_event_boundary_type"] == "gap"
        assert result["right_event_boundary_type"] == "recovery"
        assert result["left_gap_boundary_state"] in {"baseline", "dim"}
        assert result["event_window_is_lower_limit"] is True
        assert bool(trace["recovery_mask"][event_stop])
        assert result["right_recovery_jd"] == pytest.approx(times[event_stop])
    else:
        pytest.fail(f"successful measurement has unsupported event status {status!r}")

    if status != "baseline_bounded":
        assert result["dimming_complex_is_lower_limit"] is True
        assert np.isnan(result["dimming_complex_duration_upper_days"])

    duration_status = result["duration_status"]
    left_status = result["left_crossing_status"]
    right_status = result["right_crossing_status"]
    assert result["fwhm_method_version"] == DIAGNOSTICS.FWHM_METHOD_VERSION
    assert result["left_bracketed"] is (left_status in {"exact", "interval"})
    assert result["right_bracketed"] is (right_status in {"exact", "interval"})
    assert result["left_crossing_resolved"] is (left_status == "exact")
    assert result["right_crossing_resolved"] is (right_status == "exact")
    assert result["left_crossing_gap_censored"] is (left_status == "interval")
    assert result["right_crossing_gap_censored"] is (right_status == "interval")
    for side, side_status in (("left", left_status), ("right", right_status)):
        point = result[f"{side}_crossing_time"]
        lower = result[f"{side}_crossing_lower_jd"]
        upper = result[f"{side}_crossing_upper_jd"]
        if side_status == "exact":
            assert np.isfinite(point)
            assert np.isfinite(lower)
            assert np.isfinite(upper)
            assert lower <= point <= upper
        elif side_status == "interval":
            assert np.isnan(point)
            assert np.isfinite(lower)
            assert np.isfinite(upper)
            assert lower < upper
        else:
            assert side_status == "censored"
            assert np.isnan(point)
    if duration_status == "resolved":
        assert left_status == "exact"
        assert right_status == "exact"
        assert result["duration_is_interval_censored"] is False
        assert result["duration_is_lower_limit"] is False
        if result["duration_mc_reporting_status"] == "reported_resolved":
            assert result["duration_mc_resolved_fraction"] >= 0.9
            for field in ("duration_mc_err_minus", "duration_mc_err_plus"):
                assert np.isfinite(result[field])
                assert result[field] >= 0
        else:
            assert np.isnan(result["duration_mc_err_minus"])
            assert np.isnan(result["duration_mc_err_plus"])
    elif duration_status == "interval_censored":
        assert left_status in {"exact", "interval"}
        assert right_status in {"exact", "interval"}
        assert "interval" in {left_status, right_status}
        assert result["duration_is_interval_censored"] is True
        assert result["duration_lower_days"] <= result["duration_plot_days"]
        assert result["duration_plot_days"] <= result["duration_upper_days"]
        assert result["duration_mc_reporting_status"] == "structurally_censored"
        assert np.isnan(result["duration_mc_err_minus"])
        assert np.isnan(result["duration_mc_err_plus"])
    elif duration_status == "right_censored":
        assert left_status in {"exact", "interval"}
        assert right_status == "censored"
        assert result["duration_is_lower_limit"] is True
        assert result["duration_is_interval_censored"] is False
        assert np.isnan(result["right_crossing_time"])
        assert result["duration_plot_days"] == pytest.approx(result["duration_lower_days"])
        assert result["duration_mc_reporting_status"] == "structurally_censored"
        assert np.isnan(result["duration_mc_err_minus"])
        assert np.isnan(result["duration_mc_err_plus"])
    elif duration_status == "left_censored":
        assert left_status == "censored"
        assert right_status in {"exact", "interval"}
        assert result["duration_is_lower_limit"] is True
        assert result["duration_is_interval_censored"] is False
        assert np.isnan(result["left_crossing_time"])
        assert result["duration_plot_days"] == pytest.approx(result["duration_lower_days"])
        assert result["duration_mc_reporting_status"] == "structurally_censored"
        assert np.isnan(result["duration_mc_err_minus"])
        assert np.isnan(result["duration_mc_err_plus"])
    elif duration_status == "both_censored":
        assert left_status == "censored"
        assert right_status == "censored"
        assert result["duration_is_lower_limit"] is True
        assert result["duration_is_interval_censored"] is False
        assert np.isnan(result["left_crossing_time"])
        assert np.isnan(result["right_crossing_time"])
        assert result["duration_plot_days"] == pytest.approx(result["duration_lower_days"])
        assert result["duration_mc_reporting_status"] == "structurally_censored"
        assert np.isnan(result["duration_mc_err_minus"])
        assert np.isnan(result["duration_mc_err_plus"])


def test_completed_event_uses_baseline_recoveries_not_half_depth_island(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(41, dtype=float) + 0.2
    residual = np.zeros_like(times)
    residual[10:14] = 0.04  # shallow ingress below the half-depth level
    residual[14:27] = 0.12
    residual[27:31] = 0.04  # shallow egress below the half-depth level

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert result["event_window_start_jd"] < times[10]
    assert result["event_window_end_jd"] > times[30]
    assert result["left_crossing_time"] > result["event_window_start_jd"]
    assert result["right_crossing_time"] < result["event_window_end_jd"]
    assert result["event_window_duration_days"] > result["half_depth_duration_days"]


@pytest.mark.parametrize(
    ("side", "expected_event_status", "expected_duration_status"),
    [
        ("right", "ongoing_right_censored", "right_censored"),
        ("left", "ongoing_left_censored", "left_censored"),
    ],
)
def test_ongoing_event_has_one_recovered_side_and_censors_only_at_data_edge(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
    expected_event_status: str,
    expected_duration_status: str,
) -> None:
    times = np.arange(32, dtype=float) + 0.2
    residual = np.zeros_like(times)
    residual[8:11] = 0.04
    residual[11:] = 0.12
    if side == "left":
        residual = residual[::-1].copy()

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == expected_event_status
    assert result["duration_status"] == expected_duration_status
    if side == "right":
        assert result["event_window_end_jd"] == pytest.approx(times[-1])
    else:
        assert result["event_window_start_jd"] == pytest.approx(times[0])


@pytest.mark.parametrize("edge_side", ["left", "right"])
def test_baseline_edge_epoch_disqualifies_false_ongoing_event_and_selects_valid_dip(
    monkeypatch: pytest.MonkeyPatch,
    edge_side: str,
) -> None:
    times = np.arange(61, dtype=float) + 0.2
    residual = np.zeros_like(times)
    # This is the larger apparent event, but its actual edge epoch is already
    # back at baseline.  A nearest-three smoothing operation must not turn it
    # into a supposedly ongoing event.
    residual[1:10] = 0.20
    # A smaller, genuinely baseline-bounded dip remains a valid candidate.
    residual[31:38] = 0.12
    if edge_side == "right":
        residual = residual[::-1].copy()
        valid_peak_range = (times[23], times[29])
    else:
        valid_peak_range = (times[31], times[37])

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert result["left_edge_dim_confirmed"] is False
    assert result["right_edge_dim_confirmed"] is False
    assert valid_peak_range[0] <= result["peak_jd"] <= valid_peak_range[1]


@pytest.mark.parametrize("edge_side", ["left", "right"])
def test_baseline_edge_epoch_disqualifies_false_ongoing_event_when_no_other_dip(
    monkeypatch: pytest.MonkeyPatch,
    edge_side: str,
) -> None:
    times = np.arange(30, dtype=float) + 0.2
    residual = np.zeros_like(times)
    residual[1:10] = 0.20
    if edge_side == "right":
        residual = residual[::-1].copy()

    result = _measure_synthetic(monkeypatch, times, residual)

    assert result["measurement_error"]
    assert result["event_window_status"] == "measurement_failed"
    assert result["left_edge_dim_confirmed"] is False
    assert result["right_edge_dim_confirmed"] is False
    assert result["_trace"] is None


def test_event_without_any_baseline_recovery_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(30, dtype=float) + 0.2
    residual = np.full_like(times, 0.12)

    result = _measure_synthetic(monkeypatch, times, residual)

    assert result["measurement_error"]
    assert result["event_window_status"] == "measurement_failed"
    assert result["_trace"] is None


def test_strong_terminal_state_across_gap_is_ongoing_but_gap_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.r_[np.arange(18, dtype=float), 120.0] + 0.2
    residual = np.zeros_like(times)
    residual[8:] = 0.14

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "ongoing_right_censored"
    assert result["right_edge_dim_confirmed"] is True
    assert result["right_event_boundary_type"] == "data_edge"
    assert result["event_window_end_jd"] == pytest.approx(times[-1])
    assert result["event_window_gap_count"] == 1
    assert result["event_window_is_lower_limit"] is True


def test_isolated_near_baseline_points_do_not_count_as_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(30, dtype=float) + 0.2
    residual = np.full_like(times, 0.12)
    residual[[0, 14, 29]] = 0.0

    result = _measure_synthetic(monkeypatch, times, residual)

    assert result["measurement_error"]
    assert result["event_window_status"] == "measurement_failed"
    assert result["left_baseline_recovered"] is False
    assert result["right_baseline_recovered"] is False


def test_dim_supported_seasonal_gap_is_not_truncated_to_peak_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.r_[np.arange(12, dtype=float), np.arange(100, 113, dtype=float)] + 0.2
    residual = np.zeros_like(times)
    residual[5] = 0.04
    residual[6:17] = 0.12  # half-depth state continues across the 90-day gap
    residual[17] = 0.04

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert result["left_event_boundary_type"] == "recovery"
    assert result["right_event_boundary_type"] == "recovery"
    assert result["event_window_start_jd"] < times[6]
    assert result["event_window_end_jd"] > times[16]
    assert result["event_window_duration_days"] > 90.0
    assert result["duration_status"] == "left_censored"
    assert result["left_crossing_status"] == "censored"
    assert result["right_crossing_status"] == "exact"
    assert result["duration_is_lower_limit"] is True
    assert result["duration_is_interval_censored"] is False
    assert result["internal_gap_count"] == 1
    assert result["event_window_gap_count"] == 1
    assert result["event_continuity_assumed"] is True
    assert result["duration_plot_days"] == pytest.approx(result["duration_lower_days"])
    assert result["duration_plot_days"] > 90.0


def test_short_intermediate_recovery_does_not_split_gap_spanning_half_depth_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.r_[np.arange(15, dtype=float), np.arange(100, 116, dtype=float)] + 0.2
    residual = np.zeros_like(times)
    # Episode one has a left recovery, then only two near-baseline nights
    # before the seasonal gap: not enough for a recovery plateau.
    residual[5:13] = 0.12
    # A distinct and deeper post-gap episode begins dim and recovers locally.
    residual[15:24] = 0.16

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert result["left_event_boundary_type"] == "recovery"
    assert result["right_event_boundary_type"] == "recovery"
    assert result["left_baseline_recovered"] is True
    assert result["right_baseline_recovered"] is True
    assert result["event_window_is_lower_limit"] is False
    assert result["event_window_start_jd"] < times[5]
    assert result["event_window_end_jd"] < times[-1]
    assert result["event_window_gap_count"] == 1
    assert result["event_continuity_assumed"] is True
    assert result["event_window_duration_days"] > 90.0
    assert result["peak_jd"] >= 100.0
    assert result["duration_status"] == "both_censored"
    assert result["left_crossing_status"] == "censored"
    assert result["right_crossing_status"] == "censored"
    assert np.isnan(result["left_crossing_time"])
    assert result["duration_plot_days"] > 90.0


def test_recovery_anchor_validation_uses_only_the_quiescent_side() -> None:
    times = np.arange(9, dtype=float)
    residual = np.array([0.21, -0.17, 0.16, -0.010, -0.0002, -0.014, -0.023, -0.025, 0.0])
    sigma = np.full_like(times, 0.01)
    left = DIAGNOSTICS._directional_recovery_anchor_mask(
        residual,
        sigma,
        times,
        0.01,
        30.0,
        side="left",
    )
    right = DIAGNOSTICS._directional_recovery_anchor_mask(
        residual,
        sigma,
        times,
        0.01,
        30.0,
        side="right",
    )

    assert left[4] is np.False_
    assert right[4] is np.True_


def test_mixed_boundary_is_not_reported_as_confirmed_recovery() -> None:
    times = np.arange(21, dtype=float)
    recovery = np.zeros_like(times, dtype=bool)
    recovery[[5, 15]] = True
    support = recovery.copy()
    left_anchor = np.zeros_like(recovery)
    right_anchor = np.zeros_like(recovery)
    right_anchor[15] = True

    envelopes = DIAGNOSTICS._event_envelopes_from_recovery(
        recovery,
        support,
        left_anchor,
        right_anchor,
        times,
        30.0,
    )
    middle = next(item for item in envelopes if item["start"] == 5 and item["stop"] == 15)

    assert middle["left_boundary_type"] == "unconfirmed_recovery"
    assert middle["left_recovery"] is None
    assert middle["right_boundary_type"] == "recovery"
    assert middle["right_recovery"] == 15


def test_half_depth_state_continuing_back_from_seasonal_gap_is_left_censored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.r_[np.arange(12, dtype=float), np.arange(100, 113, dtype=float)] + 0.2
    residual = np.zeros_like(times)
    residual[5] = 0.04
    residual[6:12] = 0.10
    residual[12:17] = 0.14  # the selected peak is in the post-gap component
    residual[17] = 0.04

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert result["duration_status"] == "left_censored"
    assert result["left_bracketed"] is False
    assert np.isnan(result["left_crossing_time"])
    assert result["duration_is_lower_limit"] is True
    assert result["duration_is_interval_censored"] is False
    assert result["internal_gap_count"] == 1
    assert result["event_window_gap_count"] == 1
    trace = result["_trace"]
    assert trace["epochs"].iloc[int(trace["left_inside"])]["t"] == pytest.approx(6.2)
    assert result["duration_plot_days"] > 90.0


def test_half_depth_support_spans_two_gaps_when_recovery_is_not_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.r_[
        np.arange(12, dtype=float),
        np.arange(100, 106, dtype=float),
        np.arange(200, 212, dtype=float),
    ] + 0.2
    residual = np.zeros_like(times)
    residual[5] = 0.04
    residual[6:12] = 0.10
    residual[12:18] = 0.14  # selected peak component lies between two gaps
    residual[18:24] = 0.10
    residual[24] = 0.04

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert result["duration_status"] == "both_censored"
    assert result["left_bracketed"] is False
    assert result["right_bracketed"] is False
    assert result["duration_is_lower_limit"] is True
    assert result["duration_is_interval_censored"] is False
    assert result["event_window_gap_count"] == 2
    assert result["observed_half_depth_span_days"] == pytest.approx(199.0)
    assert result["internal_gap_count"] == 2
    assert result["duration_plot_days"] == pytest.approx(199.0)


def test_multiple_dips_are_ranked_as_separate_baseline_bounded_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(61, dtype=float) + 0.2
    residual = np.zeros_like(times)
    residual[8:10] = 0.03
    residual[10:15] = 0.08
    residual[15:17] = 0.03
    residual[32:35] = 0.04
    residual[35:44] = 0.16
    residual[44:47] = 0.04

    result = _measure_synthetic(monkeypatch, times, residual)

    _assert_measurement_invariants(result)
    assert result["event_window_status"] == "baseline_bounded"
    assert times[35] <= result["peak_jd"] <= times[43]
    assert result["event_window_start_jd"] > times[17]
    assert result["event_window_end_jd"] < times[-1]


def test_noisy_baseline_and_one_near_baseline_dip_epoch_do_not_split_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(51, dtype=float) + 0.2
    baseline_pattern = np.array([0.006, -0.008, 0.011, -0.004, 0.003])
    residual = np.resize(baseline_pattern, times.shape).astype(float)
    residual[15:18] = 0.04
    residual[18:34] = 0.13
    residual[25] = 0.0  # a single near-baseline epoch inside the physical dip
    residual[34:37] = 0.04

    result = _measure_synthetic(monkeypatch, times, residual, sigma_mag=0.01)

    _assert_measurement_invariants(result)
    trace = result["_trace"]
    assert result["event_window_status"] == "baseline_bounded"
    assert result["event_window_start_jd"] < times[15]
    assert result["event_window_end_jd"] > times[36]
    assert not bool(trace["recovery_mask"][25])
    assert int(trace["event_start"]) < 25 < int(trace["event_stop"])
