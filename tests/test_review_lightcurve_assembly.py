from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd
import pytest

if "celerite2" not in sys.modules:
    sys.modules["celerite2"] = types.SimpleNamespace(
        GaussianProcess=object,
        terms=types.SimpleNamespace(SHOTerm=object),
    )

from malca.plotting.lightcurve_publication import BAND_COLORS
from malca.review.lightcurve_assembly import ReviewPlotRequest, assemble_review_lightcurve_plot
from malca.review.lightcurve_pdf import render_review_lightcurve_pdf
from malca.review.lightcurve_plotly import render_review_lightcurve_plotly


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


def _base_request(payload: dict, **kwargs) -> ReviewPlotRequest:
    params = dict(
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
        yaxis_mode="mag",
        external_source_view=["asassn"],
        external_panel_mode="overlay",
        candidate_id=None,
        discover_external=False,
    )
    params.update(kwargs)
    return ReviewPlotRequest.from_kwargs(payload, **params)


def test_assembler_external_overlay_on_raw_panel(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    tess_path = tmp_path / "tess_lc_123.parquet"
    _write_dat2(lc_path)
    _write_tess_parquet(tess_path)
    payload = {"candidate_id": "123", "asas_sn_id": "123", "lc_path": str(lc_path), "baseline_mag": 14.0}

    spec = assemble_review_lightcurve_plot(
        _base_request(
            payload,
            external_lcs={"tess": tess_path},
            external_source_view=["asassn", "tess"],
        )
    )

    assert spec.status == "ok"
    external = [trace for trace in spec.traces if trace.label and "TESS" in trace.label]
    assert external
    assert all(trace.panel_id == "raw" for trace in external)


def test_assembler_split_layout_adds_external_panel(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    neowise_path = tmp_path / "neowise_lc_123.parquet"
    _write_dat2(lc_path)
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0],
            "w1mpro": [12.1, 12.4, 12.2],
            "w1sigmpro": [0.03, 0.04, 0.03],
            "w2mpro": [11.7, 11.8, 11.6],
            "w2sigmpro": [0.04, 0.04, 0.05],
        }
    ).to_parquet(neowise_path, index=False)
    payload = {"candidate_id": "123", "asas_sn_id": "123", "lc_path": str(lc_path)}

    spec = assemble_review_lightcurve_plot(
        _base_request(
            payload,
            show_baseline=False,
            external_lcs={"neowise": neowise_path},
            external_source_view=["asassn", "neowise"],
            external_panel_mode="split",
        )
    )

    assert spec.status == "ok"
    panel_ids = [panel.panel_id for panel in spec.panels]
    assert "external:neowise" in panel_ids
    neowise_traces = [trace for trace in spec.traces if trace.panel_id == "external:neowise"]
    assert neowise_traces


def test_assembler_populates_coordinate_headers(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "ra": 147.2,
        "dec": -54.9997,
    }
    spec = assemble_review_lightcurve_plot(_base_request(payload))
    assert spec.header_left is not None and spec.header_left.startswith("J")
    assert spec.header_right is not None and r"\alpha" in spec.header_right


def test_plotly_and_pdf_backends_share_external_overlay(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    tess_path = tmp_path / "tess_lc_123.parquet"
    _write_dat2(lc_path)
    _write_tess_parquet(tess_path)
    payload = {"candidate_id": "123", "asas_sn_id": "123", "lc_path": str(lc_path), "baseline_mag": 14.0}
    request = _base_request(
        payload,
        external_lcs={"tess": tess_path},
        external_source_view=["asassn", "tess"],
    )
    spec = assemble_review_lightcurve_plot(request)

    plotly = render_review_lightcurve_plotly(spec, theme="black", uirevision_key="test")
    pdf = render_review_lightcurve_pdf(spec)

    assert plotly["status"] == "ok"
    assert any("TESS" in str(trace.name) for trace in plotly["figure"].data)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_assembler_native_band_color_mode_groups_cameras_by_band(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {"candidate_id": "123", "asas_sn_id": "123", "lc_path": str(lc_path), "baseline_mag": 14.0}

    camera_spec = assemble_review_lightcurve_plot(_base_request(payload, native_color_mode="camera"))
    band_spec = assemble_review_lightcurve_plot(_base_request(payload, native_color_mode="band"))

    camera_raw = [trace for trace in camera_spec.traces if trace.panel_id == "raw"]
    band_raw = [trace for trace in band_spec.traces if trace.panel_id == "raw"]
    assert len(camera_raw) >= 2
    assert len({trace.color for trace in camera_raw if trace.color}) >= 2
    assert {trace.color for trace in band_raw if trace.color} == {BAND_COLORS["g"], BAND_COLORS["V"]}
    assert sum(trace.color == BAND_COLORS["g"] for trace in band_raw) == 1
    assert sum(trace.color == BAND_COLORS["V"] for trace in band_raw) == 1
