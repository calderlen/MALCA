from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, no_update

from malca.review import app as review_app


def test_update_sed_panel_renders_graph(monkeypatch) -> None:
    def fake_load(candidate_id, extinction_mode, theme_mode):
        fig = go.Figure()
        rows = pd.DataFrame({
            "source": ["AllWISE"],
            "lambda_l_lambda": [1.0e32],
        })
        return fig, rows, []

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)

    graph, status = review_app.update_sed_panel("cand-1", "observed", "black")

    assert isinstance(graph, dcc.Graph)
    assert "1 SED points" in status
    assert "AllWISE" in status


def test_export_sed_pdf_reports_kaleido_failure(monkeypatch) -> None:
    def fake_load(candidate_id, extinction_mode, theme_mode):
        return go.Figure(), pd.DataFrame(), []

    def fail_to_image(*args, **kwargs):
        raise RuntimeError("kaleido unavailable")

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)
    monkeypatch.setattr(review_app.pio, "to_image", fail_to_image)

    data, message = review_app.export_sed_plot(1, "cand-1", "observed", "black")

    assert data is no_update
    assert "Export failed (SED PDF)" in message
    assert "kaleido" in message


def test_export_sed_pdf_uses_white_theme_and_dark_model_line(monkeypatch) -> None:
    seen = {}

    def fake_load(candidate_id, extinction_mode, theme_mode):
        seen["theme_mode"] = theme_mode
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            mode="lines",
            name="Castelli/Kurucz dereddened fit",
            line=dict(color="#f8fafc", width=2.0),
        ))
        return fig, pd.DataFrame({"source": ["APASS"], "lambda_l_lambda": [1.0]}), []

    def fake_to_image(fig, *args, **kwargs):
        seen["line_color"] = fig.data[0].line.color
        seen["width"] = kwargs.get("width")
        seen["height"] = kwargs.get("height")
        return b"%PDF"

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)
    monkeypatch.setattr(review_app.pio, "to_image", fake_to_image)

    data, message = review_app.export_sed_plot(1, "cand-1", "corrected", "black")

    assert seen["theme_mode"] == "white"
    assert seen["line_color"] == "#111827"
    assert seen["width"] == 1200
    assert seen["height"] == 820
    assert data is not no_update
    assert "Exported" in message
