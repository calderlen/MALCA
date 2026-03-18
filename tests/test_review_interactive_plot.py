from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import malca.review.interactive_plot as interactive_plot_module
from malca.review.interactive_plot import (
    build_interactive_lightcurve_figure,
    normalize_external_lc_dataframe,
    resolve_lightcurve_path,
)


def _write_skypatrol_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "JD": [2459000.1, 2459001.1, 2459010.1, 2459011.1],
            "Flux": [1000.0, 1012.0, 990.0, 995.0],
            "Flux Error": [8.0, 8.2, 8.4, 8.1],
            "Mag": [14.0, 13.95, 14.12, 14.08],
            "Mag Error": [0.03, 0.03, 0.04, 0.04],
            "Limit": [99.0, 99.0, 99.0, 99.0],
            "FWHM": [2.5, 2.7, 2.6, 2.8],
            "Filter": ["g", "V", "g", "V"],
            "Quality": ["G", "G", "G", "G"],
            "Camera": ["camA", "camA", "camB", "camB"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_phase_offset_case(path: Path) -> None:
    df = pd.DataFrame(
        {
            "JD": [2459000.0, 2459001.0, 2459000.5, 2459001.5],
            "Flux": [1000.0, 1008.0, 990.0, 996.0],
            "Flux Error": [8.0, 8.0, 8.0, 8.0],
            "Mag": [14.0, 14.1, 13.8, 13.9],
            "Mag Error": [0.03, 0.03, 0.03, 0.03],
            "Limit": [99.0, 99.0, 99.0, 99.0],
            "FWHM": [2.5, 2.5, 2.5, 2.5],
            "Filter": ["g", "g", "V", "V"],
            "Quality": ["G", "G", "G", "G"],
            "Camera": ["camG", "camG", "camV", "camV"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_reduced_jd_case(path: Path) -> None:
    df = pd.DataFrame(
        {
            "JD": [6731.0, 6732.0, 6733.0, 6734.0],
            "Flux": [1000.0, 1012.0, 990.0, 995.0],
            "Flux Error": [8.0, 8.2, 8.4, 8.1],
            "Mag": [14.0, 13.95, 14.12, 14.08],
            "Mag Error": [0.03, 0.03, 0.04, 0.04],
            "Limit": [99.0, 99.0, 99.0, 99.0],
            "FWHM": [2.5, 2.7, 2.6, 2.8],
            "Filter": ["g", "V", "g", "V"],
            "Quality": ["G", "G", "G", "G"],
            "Camera": ["camA", "camA", "camB", "camB"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_resolve_lightcurve_path_uses_bundle_assets_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    plot_dir = run_dir / "plots"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    plot_dir.mkdir(parents=True)
    bundle_dir.mkdir(parents=True)

    lc_file = bundle_dir / "ASASSN-TEST-123.csv"
    _write_skypatrol_csv(lc_file)

    payload = {"path": "/not/present/ASASSN-TEST-123.csv"}
    resolved = resolve_lightcurve_path(payload, plot_dir)
    assert resolved == lc_file


def test_interactive_plot_camera_filter_changes_visible_extent(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True)
    lc_file = tmp_path / "ASASSN-TEST-456.csv"
    _write_skypatrol_csv(lc_file)

    payload = {
        "path": str(lc_file),
        "asas_sn_id": "ASASSN-TEST-456",
        "dip_run_count": 1,
        "jump_run_count": 0,
        "dipper_score": 2.5,
    }

    full = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=None,
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=False,
        show_diagnostics=True,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="case-full",
    )

    filtered = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=["camB"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=False,
        show_diagnostics=True,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="case-filtered",
    )

    full_min_x = min(min(trace.x) for trace in full["figure"].data if len(trace.x) > 0)
    filtered_min_x = min(min(trace.x) for trace in filtered["figure"].data if len(trace.x) > 0)

    assert filtered_min_x > full_min_x
    assert {opt["value"] for opt in full["camera_options"]} == {"camA", "camB"}
    assert filtered["camera_values"] == ["camB"]
    assert full["figure"].layout.uirevision == "case-full"


def test_interactive_plot_band_filter_replaces_legend_toggles(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True)
    lc_file = tmp_path / "ASASSN-TEST-BANDS.csv"
    _write_skypatrol_csv(lc_file)

    payload = {
        "path": str(lc_file),
        "asas_sn_id": "ASASSN-TEST-BANDS",
    }

    full = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=None,
        filter_bad_cameras=False,
        show_baseline=False,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="bands-full",
    )

    g_only = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=None,
        selected_bands=["g"],
        filter_bad_cameras=False,
        show_baseline=False,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="bands-g-only",
    )

    assert full["status"] == "ok"
    assert len(full["figure"].data) == 4
    assert g_only["status"] == "ok"
    assert len(g_only["figure"].data) == 2
    assert {trace.name for trace in g_only["figure"].data} == {"camA (g)", "camB (g)"}


def test_interactive_plot_missing_file_has_actionable_status(tmp_path: Path) -> None:
    payload = {"path": "/missing/lightcurve.csv", "asas_sn_id": "ASASSN-TEST-999"}
    out = build_interactive_lightcurve_figure(
        payload,
        plot_dir=tmp_path / "plots",
        selected_cameras=None,
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=False,
        show_diagnostics=True,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="missing",
    )
    assert out["status"] == "missing-file"
    assert "Missing light-curve file" in out["status_message"]


@pytest.mark.parametrize("baseline_func", ["gp", "gp_masked"])
def test_interactive_plot_uses_run_param_gp_settings(tmp_path: Path, monkeypatch, baseline_func: str) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True)
    lc_file = tmp_path / f"ASASSN-TEST-{baseline_func}.csv"
    _write_skypatrol_csv(lc_file)

    payload = {
        "path": str(lc_file),
        "asas_sn_id": f"ASASSN-TEST-{baseline_func}",
    }

    captured: dict[str, object] = {}

    def fake_baseline(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured["kwargs"] = dict(kwargs)
        out = df.copy()
        out["baseline"] = out["mag"].to_numpy(dtype=float)
        out["resid"] = 0.0
        out["sigma_eff"] = out["error"].to_numpy(dtype=float)
        out["baseline_source"] = baseline_func
        return out

    interactive_plot_module._BASELINE_CACHE.clear()
    monkeypatch.setitem(interactive_plot_module.BASELINE_FUNCTIONS, baseline_func, fake_baseline)

    out = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=None,
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=False,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={
            "baseline_func": baseline_func,
            "baseline_s0": 0.11,
            "baseline_w0": 0.22,
            "baseline_q": 0.33,
            "baseline_jitter": 0.44,
            "baseline_sigma_floor": 0.55,
        },
        uirevision_key=f"baseline-{baseline_func}",
    )

    assert out["status"] == "ok"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["S0"] == pytest.approx(0.11)
    assert kwargs["w0"] == pytest.approx(0.22)
    assert kwargs["q"] == pytest.approx(0.33)
    assert kwargs["jitter"] == pytest.approx(0.44)
    assert kwargs["sigma_floor"] == pytest.approx(0.55)
    assert kwargs["add_sigma_eff_col"] is True


def test_phase_fold_uses_shared_epoch_across_bands(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True)
    lc_file = tmp_path / "ASASSN-TEST-PHASE.csv"
    _write_phase_offset_case(lc_file)

    payload = {
        "path": str(lc_file),
        "asas_sn_id": "ASASSN-TEST-PHASE",
    }

    out = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=None,
        filter_bad_cameras=False,
        show_baseline=False,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=True,
        show_raw_mag=False,
        override_period=1.0,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="phase-shared-epoch",
    )

    assert out["status"] == "ok"
    assert len(out["figure"].data) == 2
    v_band_trace = out["figure"].data[1]
    assert min(v_band_trace.x) == pytest.approx(0.5)


def test_external_overlay_uses_plot_jd_frame_for_reduced_native_jd(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True)
    lc_file = tmp_path / "ASASSN-TEST-REDUCED.csv"
    _write_reduced_jd_case(lc_file)

    ps1_path = tmp_path / "ps1_lc_ASASSN-TEST-REDUCED.parquet"
    pd.DataFrame(
        {
            "mjd": [55026.608101],
            "filter": ["g_ps"],
            "mag": [17.7],
            "mag_err": [0.07],
        }
    ).to_parquet(ps1_path, index=False)

    payload = {
        "path": str(lc_file),
        "asas_sn_id": "ASASSN-TEST-REDUCED",
    }

    out = build_interactive_lightcurve_figure(
        payload,
        plot_dir=plot_dir,
        selected_cameras=None,
        filter_bad_cameras=False,
        show_baseline=False,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="external-jd-frame",
        external_lcs={"ps1": ps1_path},
    )

    assert out["status"] == "ok"
    ps1_trace = out["figure"].data[-1]
    assert float(ps1_trace.x[0]) == pytest.approx(-2972.891899)


@pytest.mark.parametrize(
    ("source_name", "frame", "checks"),
    [
        (
            "atlas",
            pd.DataFrame({"MJD": [2459000.5], "F": ["C"], "m": [13.1], "dm": [0.03]}),
            {
                "mjd": 59000.0,
                "filter": "c",
                "mag_err": 0.03,
            },
        ),
        (
            "ztf",
            pd.DataFrame({"hjd": [2459000.5], "filtercode": [2], "mag": [13.2], "magerr": [0.04]}),
            {
                "mjd": 59000.0,
                "band": "zr",
                "mag_err": 0.04,
            },
        ),
        (
            "gaia_epoch",
            pd.DataFrame({"g_transit_time": [123.4], "g_transit_mag": [15.2], "g_transit_mag_error": [0.02], "band": ["g"]}),
            {
                "time": 123.4,
                "band": "G",
                "mag_err": 0.02,
            },
        ),
        (
            "aavso",
            pd.DataFrame({"JD": [2459000.5], "Mag": [13.7], "Err": [0.08], "Filter": ["v"]}),
            {
                "mjd": 59000.0,
                "filter": "V",
                "mag_err": 0.08,
            },
        ),
        (
            "ps1",
            pd.DataFrame({"obsTime": [2459000.5], "filterID": [1], "psfFlux": [1000.0], "psfFluxErr": [10.0]}),
            {
                "mjd": 59000.0,
                "filter": "g_ps",
            },
        ),
        (
            "crts",
            pd.DataFrame({"ObsTime": [2459000.5], "Mag": [14.1], "e_Mag": [0.12]}),
            {
                "mjd": 59000.0,
                "mag_err": 0.12,
            },
        ),
    ],
)
def test_normalize_external_lc_dataframe_handles_schema_aliases(source_name: str, frame: pd.DataFrame, checks: dict) -> None:
    out = normalize_external_lc_dataframe(source_name, frame)

    for key, expected in checks.items():
        if isinstance(expected, float):
            assert float(out.iloc[0][key]) == pytest.approx(expected)
        else:
            assert out.iloc[0][key] == expected

    if source_name == "ps1":
        assert float(out.iloc[0]["mag"]) == pytest.approx(1.4)
        assert float(out.iloc[0]["mag_err"]) == pytest.approx(0.0108)
