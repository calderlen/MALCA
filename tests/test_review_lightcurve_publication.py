from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


if "celerite2" not in sys.modules:
    sys.modules["celerite2"] = types.SimpleNamespace(
        GaussianProcess=object,
        terms=types.SimpleNamespace(SHOTerm=object),
    )

from malca.review.lightcurve_publication import _axis_label_for_offset, build_review_lightcurve_publication_pdf
from malca.review.lightcurve_assembly import PlotTrace
from malca.review.lightcurve_pdf import _plot_trace
from malca.review.interactive_plot import build_interactive_lightcurve_figure


def _write_dat2(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2458000.0 14.10 0.02 1 4 0 0 ba/F1",
                "2458001.0 14.18 0.02 1 4 0 0 ba/F1",
                "2458002.0 14.55 0.03 1 4 0 0 ba/F1",
                "2458003.0 14.15 0.02 1 4 0 0 ba/F1",
                "2458000.5 13.95 0.02 1 5 1 0 bb/F1",
                "2458001.5 14.02 0.02 1 5 1 0 bb/F1",
                "2458002.5 14.34 0.03 1 5 1 0 bb/F1",
                "2458003.5 14.00 0.02 1 5 1 0 bb/F1",
            ]
        ),
        encoding="ascii",
    )


def _write_tess_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "time": [1000.0, 1001.0, 1002.0],
            "flux": [1.0, 0.9, 1.1],
            "flux_err": [0.01, 0.01, 0.02],
            "quality": [0, 0, 0],
            "sector": [1, 1, 1],
        }
    ).to_parquet(path, index=False)


def _write_neowise_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0],
            "w1mpro": [12.1, 12.4, 12.2],
            "w1sigmpro": [0.03, 0.04, 0.03],
            "w2mpro": [11.7, 11.8, 11.6],
            "w2sigmpro": [0.04, 0.04, 0.05],
        }
    ).to_parquet(path, index=False)


def _write_crts_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "mjd": [56000.0, 56001.0],
            "mag": [15.1, 15.2],
            "mag_err": [0.08, 0.09],
        }
    ).to_parquet(path, index=False)


def test_review_time_axis_label_includes_days() -> None:
    label = _axis_label_for_offset(2458000.0)

    assert "2458000" in label
    assert r"\mathrm{d}" in label


def test_review_lightcurve_pdf_uses_adaptive_small_markers() -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.calls = []

        def errorbar(self, *args, **kwargs) -> None:
            self.calls.append(("errorbar", args, kwargs))

        def scatter(self, *args, **kwargs) -> None:
            self.calls.append(("scatter", args, kwargs))

    raw_axis = FakeAxis()
    _plot_trace(
        raw_axis,
        PlotTrace(
            panel_id="raw",
            x=np.arange(500, dtype=float),
            y=np.ones(500),
            yerr=np.full(500, 0.02),
            marker_size=7.0,
        ),
    )
    assert raw_axis.calls[0][2]["markersize"] == pytest.approx(1.9)
    assert raw_axis.calls[0][2]["markeredgewidth"] == pytest.approx(0.15)
    assert raw_axis.calls[0][2]["elinewidth"] == pytest.approx(0.3)

    resid_axis = FakeAxis()
    _plot_trace(
        resid_axis,
        PlotTrace(
            panel_id="resid",
            x=np.arange(50, dtype=float),
            y=np.ones(50),
            marker_size=6.0,
        ),
    )
    assert resid_axis.calls[0][2]["s"] == pytest.approx(2.4**2)
    assert resid_axis.calls[0][2]["linewidths"] == pytest.approx(0.15)


def test_review_lightcurve_publication_pdf_uses_native_matplotlib_data_path(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "dip_best_t0": 2458002.0,
        "dip_best_width_param": 0.4,
        "dip_bayes_factor": 20.0,
        "dip_run_count": 1,
        "period_consensus_days": 2.0,
    }

    pdf = build_review_lightcurve_publication_pdf(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=True,
        confidence_colors=True,
        run_params={"baseline_func": "global_median"},
        yaxis_mode="mag",
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_interactive_event_overlays_do_not_expand_raw_y_autorange(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "dip_best_t0": 2458002.0,
        "dip_best_width_param": 0.4,
        "dip_bayes_factor": 20.0,
        "jump_best_t0": 2458000.5,
        "jump_best_width_param": 0.4,
        "jump_bayes_factor": 20.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=False,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=True,
        confidence_colors=True,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    fig = result["figure"]
    assert result["status"] == "ok"
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]
    assert fig.layout.yaxis2.autorange == "reversed"
    assert min(fig.layout.yaxis.range) > 13.0
    event_shapes = [
        shape
        for shape in fig.layout.shapes
        if shape.type == "rect" or getattr(getattr(shape, "line", None), "dash", None) == "dash"
    ]
    assert event_shapes
    assert all(str(shape.yref).endswith(" domain") for shape in event_shapes)
    assert all(
        not (
            getattr(trace, "mode", None) == "markers"
            and getattr(getattr(trace, "marker", None), "symbol", None) == "diamond"
        )
        for trace in fig.data
    )
    event_annotations = [
        annotation
        for annotation in fig.layout.annotations
        if annotation.text == "◆"
        or str(annotation.text).startswith("Dip thr")
        or str(annotation.text).startswith("Jump thr")
    ]
    assert event_annotations
    assert all(str(annotation.yref) == "y" for annotation in event_annotations)
    dip_label = next(annotation for annotation in event_annotations if str(annotation.text).startswith("Dip thr"))
    jump_label = next(annotation for annotation in event_annotations if str(annotation.text).startswith("Jump thr"))
    dip_marker = next(
        annotation
        for annotation in event_annotations
        if annotation.text == "◆" and str(annotation.font.color).startswith("rgba(255")
    )
    jump_marker = next(
        annotation
        for annotation in event_annotations
        if annotation.text == "◆" and str(annotation.font.color).startswith("rgba(0,150,255")
    )
    raw_xy = [
        (float(x), float(y))
        for trace in fig.data
        if getattr(trace, "mode", None) == "markers" and getattr(trace, "showlegend", None) is not False
        for x, y in zip(trace.x, trace.y)
    ]
    raw_y = [y for _, y in raw_xy]
    dip_local_y = [y for x, y in raw_xy if abs(x - 2.0) <= 0.8]
    assert float(dip_label.y) > max(dip_local_y)
    assert float(dip_marker.y) > max(dip_local_y)
    assert float(jump_label.y) < min(raw_y)
    assert float(jump_marker.y) < min(raw_y)


def test_interactive_event_overlays_convert_skypatrol_reduced_t0_to_plot_axis(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "dip_best_t0": 8002.0,
        "dip_best_width_param": 0.4,
        "dip_bayes_factor": 20.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=False,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=True,
        confidence_colors=True,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    fig = result["figure"]
    assert result["status"] == "ok"
    dip_line = next(
        shape
        for shape in fig.layout.shapes
        if getattr(getattr(shape, "line", None), "dash", None) == "dash"
    )
    dip_label = next(
        annotation
        for annotation in fig.layout.annotations
        if str(annotation.text).startswith("Dip thr")
    )
    dip_marker = next(annotation for annotation in fig.layout.annotations if annotation.text == "◆")

    assert float(dip_line.x0) == pytest.approx(2.0)
    assert float(dip_line.x1) == pytest.approx(2.0)
    assert float(dip_label.x) == pytest.approx(2.0)
    assert float(dip_marker.x) == pytest.approx(2.0)


def test_interactive_phase_panel_shows_placeholder_without_period(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    fig = result["figure"]
    assert result["status"] == "ok"
    assert any("no valid period" in warning.lower() for warning in result["warnings"])
    assert fig.layout.yaxis3.title.text == r"$\Delta m$ [mag]"
    assert any(
        str(annotation.text).startswith("No phase period available")
        and str(annotation.xref).endswith(" domain")
        and str(annotation.yref).endswith(" domain")
        for annotation in fig.layout.annotations
    )


def test_interactive_phase_panel_labels_pending_auto_pdm_without_period(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=None,
        phase_period_pending=True,
        phase_period_pending_source="Auto PDM",
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    assert result["status"] == "ok"
    assert result["status_message"] == "Auto PDM: searching..."
    assert any("auto pdm is running" in warning.lower() for warning in result["warnings"])
    assert any(str(annotation.text).startswith("Auto PDM is running") for annotation in result["figure"].layout.annotations)


def test_interactive_phase_panel_uses_vsx_while_harmonic_check_pending(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "vsx_period": 9.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=None,
        phase_period_pending=True,
        suppress_catalog_phase_period=False,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    assert result["status"] == "ok"
    assert "P=9.00000 d" in result["status_message"]
    assert "source=vsx_period" in result["status_message"]
    assert not any("auto harmonic check is running" in warning.lower() for warning in result["warnings"])


def test_interactive_phase_panel_uses_completed_harmonic_check_over_vsx(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "vsx_period": 9.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=1.5,
        override_period_source="Auto harmonic check",
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    assert result["status"] == "ok"
    assert "Phase-fold P=1.50000 d" in result["status_message"]
    assert "source=Auto harmonic check" in result["status_message"]
    assert "9.00000" not in result["status_message"]
    assert result["figure"].layout.yaxis3.autorange == "reversed"


def test_interactive_phase_time_panel_uses_cycle_axis_and_residual_color(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "period_consensus_days": 2.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=True,
        phase_panel_mode="time",
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
    )

    fig = result["figure"]
    phase_time_traces = [trace for trace in fig.data if "cycle:" in str(getattr(trace, "hovertemplate", ""))]

    assert result["status"] == "ok"
    assert "Phase-time P=2.00000 d" in result["status_message"]
    assert fig.layout.yaxis3.title.text == "Cycle E"
    assert fig.layout.xaxis3.range == (-0.02, 2.02)
    assert phase_time_traces
    assert any(max(trace.x) > 1.0 for trace in phase_time_traces)
    assert any(getattr(trace.marker, "showscale", False) for trace in phase_time_traces)


def test_interactive_tess_overlay_appears_as_relative_magnitude_in_mag_mode(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    tess_path = tmp_path / "tess_lc_123.parquet"
    _write_dat2(lc_path)
    _write_tess_parquet(tess_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "baseline_mag": 14.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
        external_lcs={"tess": tess_path},
        external_source_view=["asassn", "tess"],
    )

    tess_trace = next(trace for trace in result["figure"].data if trace.name == "TESS rel. Δm")
    assert tess_trace.y[0] == pytest.approx(14.0)
    assert tess_trace.y[1] > 14.0
    assert tess_trace.y[2] < 14.0
    assert "raw flux" in tess_trace.hovertemplate
    assert any("relative magnitude" in warning for warning in result["warnings"])
    assert any("not calibrated TESS-band magnitude" in warning for warning in result["warnings"])


def test_interactive_external_sources_can_split_selected_neowise_only(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    neowise_path = tmp_path / "neowise_lc_123.parquet"
    crts_path = tmp_path / "crts_lc_123.parquet"
    _write_dat2(lc_path)
    _write_neowise_parquet(neowise_path)
    _write_crts_parquet(crts_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=False,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
        external_lcs={"neowise": neowise_path, "crts": crts_path},
        external_source_view=["asassn", "neowise"],
        external_panel_mode="split",
    )

    fig = result["figure"]
    names = [trace.name for trace in fig.data]
    assert result["status"] == "ok"
    assert "NEOWISE W1" in names
    assert "NEOWISE W2" in names
    assert "CRTS CV" not in names
    assert fig.layout.yaxis2.title.text == r"$m$ [mag]"
    assert fig.layout.yaxis2.autorange == "reversed"
    assert any(annotation.text == "NEOWISE W1/W2" for annotation in fig.layout.annotations)
    assert all(getattr(trace, "yaxis", None) == "y2" for trace in fig.data if str(trace.name).startswith("NEOWISE"))


def test_interactive_external_sources_render_new_lc_products(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    paths = {
        "kepler": tmp_path / "kepler_lc_123.parquet",
        "aavso": tmp_path / "aavso_lc_123.parquet",
        "ogle": tmp_path / "ogle_lc_123.parquet",
        "stripe82": tmp_path / "stripe82_lc_123.parquet",
        "allwise_mep": tmp_path / "allwise_mep_lc_123.parquet",
        "vvvx_virac": tmp_path / "vvvx_virac_lc_123.parquet",
    }
    pd.DataFrame({"time": [100.0, 101.0], "flux": [1.0, 1.2], "flux_err": [0.01, 0.02]}).to_parquet(paths["kepler"], index=False)
    pd.DataFrame({"mjd": [59000.0, 59001.0], "band": ["V", "B"], "mag": [13.0, 13.4], "mag_err": [0.02, 0.03]}).to_parquet(paths["aavso"], index=False)
    pd.DataFrame({"mjd": [57000.0, 57001.0], "band": ["I", "V"], "mag": [15.0, 15.3], "mag_err": [0.02, 0.03]}).to_parquet(paths["ogle"], index=False)
    pd.DataFrame({"mjd": [52000.0, 52001.0], "band": ["g", "r"], "mag": [18.0, 17.6], "mag_err": [0.02, 0.03]}).to_parquet(paths["stripe82"], index=False)
    pd.DataFrame(
        {
            "mjd": [55400.0, 55401.0],
            "w1mpro": [12.0, 12.3],
            "w1sigmpro": [0.03, 0.04],
            "w3mpro": [8.0, 8.1],
            "w3sigmpro": [0.2, 0.25],
        }
    ).to_parquet(paths["allwise_mep"], index=False)
    pd.DataFrame({"mjd": [57000.0, 57001.0], "band": ["ks", "j"], "mag": [14.0, 15.0], "mag_err": [0.05, 0.06]}).to_parquet(paths["vvvx_virac"], index=False)

    result = build_interactive_lightcurve_figure(
        {
            "candidate_id": "123",
            "asas_sn_id": "123",
            "lc_path": str(lc_path),
            "baseline_mag": 14.0,
        },
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="mag",
        external_lcs=paths,
        external_source_view=["asassn", *paths.keys()],
    )

    names = {trace.name for trace in result["figure"].data}
    assert result["status"] == "ok"
    assert {
        "Kepler/K2 rel. Δm",
        "AAVSO V",
        "OGLE I",
        "Stripe 82 g",
        "AllWISE W1",
        "VVVX Ks",
    }.issubset(names)


def test_interactive_tess_overlay_appears_as_flux_in_flux_mode(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    tess_path = tmp_path / "tess_lc_123.parquet"
    _write_dat2(lc_path)
    _write_tess_parquet(tess_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "baseline_mag": 14.0,
    }

    result = build_interactive_lightcurve_figure(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=False,
        show_residuals=False,
        show_phase_fold=False,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        yaxis_mode="flux",
        external_lcs={"tess": tess_path},
        external_source_view=["asassn", "tess"],
    )

    fig = result["figure"]
    tess_trace = next(trace for trace in fig.data if trace.name == "TESS flux")
    assert list(tess_trace.y) == [1.0, 0.9, 1.1]
    assert "F: %{y:.4e}" in tess_trace.hovertemplate
    assert fig.layout.yaxis.range[0] == pytest.approx(0.0)
    assert fig.layout.yaxis.range[1] > max(tess_trace.y)
    assert not any("not calibrated TESS-band magnitude" in warning for warning in result["warnings"])
