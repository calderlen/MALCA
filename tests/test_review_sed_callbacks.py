from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, no_update

from malca.review import app as review_app
from malca.review.dustycult import DustyCultAvailability


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


def test_dustycult_result_panel_renders_unavailable_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(review_app, "DB_PATH", str(tmp_path / "review.db"))
    monkeypatch.setattr(
        review_app,
        "check_dustycult_available",
        lambda: DustyCultAvailability(False, "julia", Path("missing"), Path("missing/scripts/fit_lightcurve.jl"), "Julia executable not found"),
    )

    panel, status = review_app.update_dustycult_result_panel("cand-1", "black", 0)
    rendered = "\n".join([status] + _collect_text(panel))

    assert "Julia executable not found" in rendered
    assert "No DustyCult fit" in rendered


def test_update_dustycult_controls_uses_candidate_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(review_app, "DB_PATH", str(tmp_path / "review.db"))
    monkeypatch.setattr(review_app, "_candidate_context", lambda candidate_id: ({"candidate_id": candidate_id}, "/tmp/cand.dat2", "source"))
    monkeypatch.setattr(review_app, "_effective_local_lc_path", lambda payload, stored_lc_path=None, source_path=None: stored_lc_path)
    monkeypatch.setattr(review_app, "_load_run_params_for_plot_dir", lambda _plot_dir: {"baseline_func": "global_median"})
    monkeypatch.setattr(
        review_app,
        "control_defaults_for_candidate",
        lambda *_args, **_kwargs: {
            "source": "stored_event_columns",
            "message": "Loaded dip defaults from stored event columns.",
            "start_jd": 1.0,
            "end_jd": 3.0,
            "t0_jd": 2.0,
            "t0_width_days": 0.5,
            "log_v_width": 1.0,
            "b_center": 0.0,
            "b_width": 0.5,
            "log_tau0_width": 1.5,
            "alpha_center": 0.0,
            "alpha_width": 2.0,
            "log_sigma_width": 0.75,
            "star_R": 1.1,
            "star_u1": 0.2,
            "star_u2": 0.3,
        },
    )

    values = review_app.update_dustycult_controls("cand-1", 0)

    assert values[0:3] == (1.0, 3.0, 2.0)
    assert values[-4:-1] == (1.1, 0.2, 0.3)
    assert "stored_event_columns" in values[-1]


def _collect_text(node) -> list[str]:
    text: list[str] = []

    def walk(item) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if isinstance(item, str):
            text.append(item)
            return
        if item is None or isinstance(item, (int, float, bool)):
            return
        walk(getattr(item, "children", None))

    walk(node)
    return text
