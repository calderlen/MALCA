from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from malca.review.diagnostic_plots import build_publication_diagnostic_pdf, _background_density
from malca.review.publication import graph_config_without_image_export, render_publication_pdf


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
        "dipper_score": 12.0,
        "jumper_score": 0.5,
        "period_n_sources": 3.0,
        "dip_run_count": 4.0,
        "dip_inter_event_spacing_median": 100.0,
        "dip_inter_event_spacing_std": 30.0,
        "dip_amplitude_consistency": 0.8,
        "dip_duration_consistency": 0.7,
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
        "metric_dipper_score": rng.gamma(1.8, 2.0, 300),
        "metric_jumper_score": rng.gamma(1.2, 1.1, 300),
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
        "score_balance",
        "catalog_support",
        "recurrence_regularity",
        "dip_repeatability",
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

    for name in ("cmd", "ir_colorcolor", "kiel"):
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


def test_publication_diagnostic_pdf_returns_none_for_unknown_name() -> None:
    assert build_publication_diagnostic_pdf("not-a-static-plot", _payload(), _background()) is None


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


def test_generic_publication_pdf_renderer_emits_pdf_bytes() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[1, 2, 3], y=[2, 3, 5], mode="markers", name="points"))
    fig.update_layout(title="Generic Export", xaxis_title="JD", yaxis_title="relative flux")

    pdf = render_publication_pdf(fig, title="Generic Export")

    assert isinstance(pdf, bytes)
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
