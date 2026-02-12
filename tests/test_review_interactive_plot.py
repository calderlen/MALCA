from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.review.interactive_plot import build_interactive_lightcurve_figure, resolve_lightcurve_path


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
        show_diagnostics=True,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="missing",
    )
    assert out["status"] == "missing-file"
    assert "Missing light-curve file" in out["status_message"]
