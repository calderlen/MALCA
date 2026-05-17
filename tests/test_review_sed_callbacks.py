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
        seen["legend_orientation"] = fig.layout.legend.orientation
        seen["legend_x"] = fig.layout.legend.x
        seen["right_margin"] = fig.layout.margin.r
        seen["width"] = kwargs.get("width")
        seen["height"] = kwargs.get("height")
        return b"%PDF"

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)
    monkeypatch.setattr(review_app.pio, "to_image", fake_to_image)

    data, message = review_app.export_sed_plot(1, "cand-1", "corrected", "black")

    assert seen["theme_mode"] == "white"
    assert seen["line_color"] == "#111827"
    assert seen["legend_orientation"] == "v"
    assert seen["legend_x"] > 1.0
    assert seen["right_margin"] >= 250
    assert seen["width"] == 1200
    assert seen["height"] == 820
    assert data is not no_update
    assert "Exported" in message


def test_external_followup_panel_renders_without_details_open_state(monkeypatch) -> None:
    seen = {}

    def fake_candidate_context(candidate_id):
        return {"candidate_id": candidate_id, "simbad_main_id": "Star"}, None, None

    def fake_render(payload, candidate_id, theme):
        seen["candidate_id"] = candidate_id
        seen["theme"] = theme
        return ["external-card"]

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_render_external_followup", fake_render)

    panel = review_app.update_external_followup_panel("cand-1", "black")

    assert panel == ["external-card"]
    assert seen == {"candidate_id": "cand-1", "theme": "black"}


def test_diagnostic_plots_render_without_details_open_state(monkeypatch) -> None:
    seen = {}

    def fake_candidate_context(candidate_id):
        return {"candidate_id": candidate_id}, None, None

    def fake_render(payload, theme, background=None):
        seen["candidate_id"] = payload["candidate_id"]
        seen["theme"] = theme
        seen["background"] = background
        return ["diagnostic-card"]

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda _path: "sig")
    monkeypatch.setattr(review_app, "_get_cached_diagnostic_background", lambda _sig: {"ready": True})
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", fake_render)

    panel, status = review_app.update_diagnostic_plots(
        "cand-1",
        "black",
        {"ready": True, "signature": "sig"},
    )

    assert panel == ["diagnostic-card"]
    assert status == ""
    assert seen == {"candidate_id": "cand-1", "theme": "black", "background": {"ready": True}}


def test_diagnostic_background_prepares_even_when_details_closed(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda _path: "sig")
    monkeypatch.setattr(review_app, "_get_cached_diagnostic_background", lambda _sig: {"cached": True})

    state = review_app._prepare_diagnostic_background(
        False,
        None,
        None,
        {"signature": "", "ready": False, "cached": False, "token": 3},
    )

    assert state["signature"] == "sig"
    assert state["ready"] is True
    assert state["cached"] is True
    assert state["token"] == 4
