from __future__ import annotations

import base64
import json
import sqlite3

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from malca.review.diagnostic_plots import (
    build_publication_diagnostic_pdf,
    build_teff_sed_alpha_figure,
    _background_density,
)
from malca.review.store import get_diagnostic_background
from malca.review.publication import _numeric_sequence, _trace_array, graph_config_without_image_export, render_publication_pdf


def _payload() -> dict[str, float]:
    return {
        "phot_g_mean_mag": 12.0,
        "bp_rp": 0.7,
        "distance_gspphot": 1000.0,
        "A_v_3d": 0.4,
        "tmass_h": 10.1,
        "tmass_k": 10.0,
        "w1": 9.9,
        "w2": 9.88,
        "teff_gspphot": 6500.0,
        "logg_gspphot": 4.1,
        "pmra": 5.0,
        "pmdec": 3.0,
        "dipper_score": 12.0,
        "jumper_score": 0.5,
        "stats_photometry_robust_sigma_mag": 0.04,
        "stats_variability_stetson_J": 1.2,
        "stats_skew": 0.3,
        "stats_max_slope": 12.0,
        "period_n_sources": 3.0,
        "dip_run_count": 4.0,
        "dip_inter_event_spacing_median": 100.0,
        "dip_inter_event_spacing_std": 30.0,
        "dip_amplitude_consistency": 0.8,
        "dip_duration_consistency": 0.7,
        "sed_alpha": -0.55,
    }


def _background() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12)
    return {
        "cmd_bprp0": rng.normal(1.0, 0.35, 300),
        "cmd_mg0": rng.normal(5.0, 1.1, 300),
        "ir_w1w2": rng.normal(0.0, 0.09, 300),
        "ir_hk": rng.normal(0.05, 0.12, 300),
        "kiel_teff": rng.normal(6200.0, 1000.0, 300),
        "kiel_logg": rng.normal(4.0, 0.35, 300),
        "plane_teff_alpha_x": rng.normal(5200.0, 1200.0, 300),
        "plane_teff_alpha_y": rng.normal(-0.8, 0.6, 300),
        "rpm_bprp": rng.normal(1.0, 0.45, 300),
        "rpm_hg": rng.normal(8.0, 2.0, 300),
        "metric_dipper_score": rng.gamma(1.8, 2.0, 300),
        "metric_jumper_score": rng.gamma(1.2, 1.1, 300),
        "plane_var_strength_x": rng.lognormal(-3.2, 0.6, 300),
        "plane_var_strength_y": rng.gamma(1.8, 2.0, 300),
        "plane_stetson_x": rng.lognormal(-3.2, 0.6, 300),
        "plane_stetson_y": rng.gamma(1.2, 1.0, 300),
        "plane_shape_x": rng.normal(0.0, 0.7, 300),
        "plane_shape_y": rng.lognormal(2.2, 0.8, 300),
        "plane_catalog_support_x": rng.integers(0, 5, 300),
        "plane_catalog_support_y": rng.integers(0, 8, 300),
        "plane_recurrence_regularity_x": rng.lognormal(4.0, 0.65, 300),
        "plane_recurrence_regularity_y": rng.lognormal(3.2, 0.75, 300),
        "plane_dip_repeatability_x": rng.uniform(0.0, 1.0, 300),
        "plane_dip_repeatability_y": rng.uniform(0.0, 1.0, 300),
    }


def test_publication_diagnostic_pdf_renderers_emit_pdf_bytes() -> None:
    payload = _payload()
    background = _background()

    for name in (
        "cmd",
        "ir_colorcolor",
        "kiel",
        "teff_sed_alpha",
        "rpm",
        "score_balance",
        "catalog_support",
        "recurrence_regularity",
        "dip_repeatability",
        "variability_strength",
        "stetson_scatter",
        "shape_impulsiveness",
    ):
        pdf = build_publication_diagnostic_pdf(name, payload, background)
        assert isinstance(pdf, bytes)
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 1000


def test_diagnostic_publication_pdfs_avoid_in_plot_titles_and_legends(monkeypatch) -> None:
    import matplotlib.axes

    def fail_title(*_args, **_kwargs):
        raise AssertionError("diagnostic publication export should not add a large in-figure title")

    def fail_legend(*_args, **_kwargs):
        raise AssertionError("diagnostic publication export should not put a legend inside the plot")

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", fail_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", fail_legend)

    for name in ("cmd", "ir_colorcolor", "kiel", "teff_sed_alpha"):
        pdf = build_publication_diagnostic_pdf(name, _payload(), _background())
    assert pdf.startswith(b"%PDF")


def test_metric_publication_pdfs_avoid_in_plot_titles_and_legends(monkeypatch) -> None:
    import matplotlib.axes

    def fail_title(*_args, **_kwargs):
        raise AssertionError("metric publication exports should not add large in-figure titles")

    def fail_legend(*_args, **_kwargs):
        raise AssertionError("metric publication exports should not put legends inside the plot")

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", fail_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", fail_legend)

    for name in ("score_balance", "catalog_support", "recurrence_regularity", "dip_repeatability"):
        pdf = build_publication_diagnostic_pdf(name, _payload(), _background())
        assert pdf.startswith(b"%PDF")


def test_publication_diagnostic_pdfs_do_not_add_candidate_text(monkeypatch) -> None:
    import matplotlib.axes

    original_annotate = matplotlib.axes.Axes.annotate

    def reject_candidate_text(self, text, *args, **kwargs):
        if str(text or "").strip().lower() == "candidate":
            raise AssertionError("diagnostic publication export should use marker color, not candidate text")
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", reject_candidate_text)

    for name in (
        "cmd",
        "ir_colorcolor",
        "kiel",
        "teff_sed_alpha",
        "rpm",
        "score_balance",
        "catalog_support",
        "recurrence_regularity",
        "dip_repeatability",
        "variability_strength",
        "stetson_scatter",
        "shape_impulsiveness",
    ):
        pdf = build_publication_diagnostic_pdf(name, _payload(), _background())
        assert pdf.startswith(b"%PDF")


def test_publication_diagnostic_pdf_returns_none_for_unknown_name() -> None:
    assert build_publication_diagnostic_pdf("not-a-static-plot", _payload(), _background()) is None


def test_diagnostic_background_uses_payload_json_when_columns_are_absent(tmp_path) -> None:
    db_path = tmp_path / "review.db"
    payload = {
        "teff50": 6100.0,
        "logg50": 4.2,
        "phot_g_mean_mag": 12.0,
        "bp_rp": 0.8,
        "parallax": 2.0,
        "A_v_3d": 0.1,
        "H_K": 0.08,
        "w1_w2": 0.04,
        "pmra": 8.0,
        "pmdec": 6.0,
        "stats_photometry_robust_sigma_mag": 0.03,
        "stats_variability_stetson_J": 0.9,
        "stats_skew": 0.2,
        "stats_max_slope": 30.0,
        "dipper_score": 7.0,
        "sed_alpha": -0.6,
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, payload_json TEXT NOT NULL)")
        conn.execute("INSERT INTO candidates VALUES (?, ?)", ("C1", json.dumps(payload)))
        bg = get_diagnostic_background(conn)

    assert bg["cmd_bprp0"].shape == (1,)
    assert bg["ir_w1w2"].tolist() == [0.04]
    assert bg["rpm_bprp"].shape == (1,)
    assert bg["plane_teff_alpha_x"].tolist() == [6100.0]
    assert bg["plane_teff_alpha_y"].tolist() == [-0.6]
    assert bg["plane_stetson_x"].tolist() == [0.03]
    assert bg["plane_shape_y"].tolist() == [30.0]


def test_teff_sed_alpha_diagnostic_requires_teff_and_alpha() -> None:
    assert build_teff_sed_alpha_figure({"teff50": 4300.0}, "white", background=_background()) is None
    fig = build_teff_sed_alpha_figure(_payload(), "white", background=_background())

    assert fig is not None
    assert fig.layout.yaxis.title.text == "SED alpha"
    assert len(fig.layout.shapes) >= 3


def test_publication_background_density_uses_unsmoothed_circle_bins(monkeypatch) -> None:
    import matplotlib.axes
    import matplotlib.pyplot as plt

    recorded = {}

    def fail_imshow(*_args, **_kwargs):
        raise AssertionError("publication density should render occupied bins as circles, not image pixels")

    def record_scatter(_self, x, y, *args, **kwargs):
        recorded["x"] = np.asarray(x)
        recorded["y"] = np.asarray(y)
        recorded["counts"] = np.asarray(kwargs.get("c"))
        recorded["marker"] = kwargs.get("marker")
        recorded["linewidths"] = kwargs.get("linewidths")
        return object()

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", fail_imshow)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", record_scatter)

    fig, ax = plt.subplots()
    try:
        x = np.asarray([0.5] * 5 + [1.5] * 2, dtype=float)
        y = np.asarray([0.5] * 5 + [1.5] * 2, dtype=float)
        _background_density(ax, x, y, extent=(0.0, 2.0, 0.0, 2.0))
    finally:
        plt.close(fig)

    assert recorded["marker"] == "o"
    assert recorded["linewidths"] == 0.0
    assert recorded["x"].shape == (1,)
    assert recorded["y"].shape == (1,)
    assert recorded["counts"].tolist() == [5.0]


def test_publication_background_density_falls_back_for_sparse_samples(monkeypatch) -> None:
    import matplotlib.axes
    import matplotlib.pyplot as plt

    recorded = {}

    def record_scatter(_self, x, y, *args, **kwargs):
        recorded["x"] = np.asarray(x)
        recorded["y"] = np.asarray(y)
        recorded["color"] = kwargs.get("c") or kwargs.get("color")
        recorded["linewidths"] = kwargs.get("linewidths")
        return object()

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", record_scatter)

    fig, ax = plt.subplots()
    try:
        x = np.asarray([0.3, 1.7], dtype=float)
        y = np.asarray([0.4, 1.6], dtype=float)
        _background_density(ax, x, y, extent=(0.0, 2.0, 0.0, 2.0))
    finally:
        plt.close(fig)

    assert recorded["x"].tolist() == [0.3, 1.7]
    assert recorded["y"].tolist() == [0.4, 1.6]
    assert recorded["color"] == "#4f7fa7"
    assert recorded["linewidths"] == 0.0


def test_generic_publication_pdf_renderer_emits_pdf_bytes() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[1, 2, 3], y=[2, 3, 5], mode="markers", name="points"))
    fig.update_layout(title="Generic Export", xaxis_title="JD", yaxis_title="relative flux")

    pdf = render_publication_pdf(fig, title="Generic Export")

    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_generic_publication_pdf_renderer_decodes_plotly_typed_arrays() -> None:
    x = np.asarray([1.0, 2.0, 3.0], dtype="float64")
    y = np.asarray([2.0, 3.0, 5.0], dtype="float64")
    marker_color = np.asarray([0.1, 0.5, 0.9], dtype="float64")
    trace = {
        "type": "scatter",
        "mode": "markers",
        "x": {"dtype": "f8", "bdata": base64.b64encode(x.tobytes()).decode("ascii")},
        "y": {"dtype": "f8", "bdata": base64.b64encode(y.tobytes()).decode("ascii")},
        "marker": {"color": {"dtype": "f8", "bdata": base64.b64encode(marker_color.tobytes()).decode("ascii")}},
    }

    assert _trace_array(trace, "x") == [1.0, 2.0, 3.0]
    assert _trace_array(trace, "y") == [2.0, 3.0, 5.0]
    assert _numeric_sequence(trace["marker"]["color"]) == [0.1, 0.5, 0.9]

    pdf = render_publication_pdf({"data": [trace], "layout": {"xaxis": {"title": "x"}, "yaxis": {"title": "y"}}})
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_graph_config_without_image_export_preserves_controls() -> None:
    config = graph_config_without_image_export(
        {
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d"],
        }
    )

    assert config["displaylogo"] is False
    assert config["scrollZoom"] is True
    assert config["modeBarButtonsToRemove"] == ["lasso2d", "toImage"]


def test_generic_publication_pdf_renderer_omits_titles(monkeypatch) -> None:
    import matplotlib.figure

    def fail_suptitle(*_args, **_kwargs):
        raise AssertionError("publication exports should not add generic in-figure titles")

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", fail_suptitle)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], mode="lines", name="curve"))
    fig.update_layout(title="Generic Export", xaxis_title="JD", yaxis_title="relative flux")

    pdf = render_publication_pdf(fig, title="Generic Export")

    assert pdf.startswith(b"%PDF")


def test_generic_publication_pdf_preserves_vertical_subplot_domains(monkeypatch) -> None:
    import matplotlib.figure

    bounds: list[list[float]] = []
    original_add_axes = matplotlib.figure.Figure.add_axes

    def record_add_axes(self, *args, **kwargs):
        if args and isinstance(args[0], list):
            bounds.append(list(args[0]))
        return original_add_axes(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "add_axes", record_add_axes)

    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.08)
    for row in (1, 2, 3):
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[row, row + 0.1, row], mode="markers", name=f"panel {row}"), row=row, col=1)
    fig.update_xaxes(title_text="JD - 2458000", row=2, col=1)
    fig.update_xaxes(title_text=r"$\phi$", row=3, col=1)

    pdf = render_publication_pdf(fig, title="Light curve", width=1400, height=900)

    assert pdf.startswith(b"%PDF")
    assert len(bounds) >= 3
    assert bounds[0][1] > bounds[1][1] > bounds[2][1]


def test_generic_publication_pdf_suppresses_inner_xlabels(monkeypatch) -> None:
    import matplotlib.axes

    labels: list[str] = []
    original_set_xlabel = matplotlib.axes.Axes.set_xlabel

    def record_xlabel(self, xlabel, *args, **kwargs):
        if str(xlabel or "").strip():
            labels.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", record_xlabel)

    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.08)
    for row in (1, 2, 3):
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[row, row + 0.1, row], mode="markers", name=f"panel {row}"), row=row, col=1)
    fig.update_xaxes(title_text="JD - 2458000", row=2, col=1)
    fig.update_xaxes(title_text=r"$\phi$", row=3, col=1)

    pdf = render_publication_pdf(fig, title="Light curve", width=1400, height=900)

    assert pdf.startswith(b"%PDF")
    assert not any("JD" in label for label in labels)
    assert any("phi" in label or r"\phi" in label for label in labels)
