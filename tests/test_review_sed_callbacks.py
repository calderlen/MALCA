from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, no_update


def _install_review_app_import_stubs() -> None:
    if "celerite2" not in sys.modules and importlib.util.find_spec("celerite2") is None:
        fake_celerite2 = types.ModuleType("celerite2")
        fake_terms = types.ModuleType("celerite2.terms")

        class _FakeGaussianProcess:
            def __init__(self, *args, **kwargs):
                pass

        class _FakeTerm:
            def __init__(self, *args, **kwargs):
                pass

            def __add__(self, other):
                return self

        fake_terms.SHOTerm = _FakeTerm
        fake_terms.RealTerm = _FakeTerm
        fake_celerite2.GaussianProcess = _FakeGaussianProcess
        fake_celerite2.terms = fake_terms
        sys.modules["celerite2"] = fake_celerite2
        sys.modules["celerite2.terms"] = fake_terms

    if "multiprocess" not in sys.modules and importlib.util.find_spec("multiprocess") is None:
        fake_multiprocess = types.ModuleType("multiprocess")
        fake_multiprocess.get_all_start_methods = lambda: ["spawn"]
        fake_multiprocess.set_start_method = lambda *args, **kwargs: None
        sys.modules["multiprocess"] = fake_multiprocess


_install_review_app_import_stubs()

from malca.review import app as review_app
from malca.review.dustycult import DustyCultAvailability, upsert_dustycult_fit
from malca.review.store import upsert_candidates_frame


def test_update_sed_panel_renders_graph(monkeypatch) -> None:
    def fake_load(candidate_id, extinction_mode, theme_mode):
        fig = go.Figure()
        rows = pd.DataFrame({
            "source": ["AllWISE"],
            "lambda_l_lambda": [1.0e32],
        })
        return fig, rows, []

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)
    monkeypatch.setattr(review_app, "_load_sed_source_status_for_candidate", lambda _candidate_id: [])

    children, status = review_app.update_sed_panel("cand-1", "observed", "black")

    graph = children[0]
    assert isinstance(graph, dcc.Graph)
    assert "toImage" in graph.config.get("modeBarButtonsToRemove", [])
    assert "1 SED points" in status
    assert "AllWISE" in status


def test_sed_status_text_reports_mixed_fetch_provenance() -> None:
    rows = pd.DataFrame({
        "source": ["Gaia GSPC"],
        "lambda_l_lambda": [1.0e32],
    })
    statuses = [
        {"key": "gaia_gspc", "label": "Gaia GSPC", "status": "hit", "n_rows": 1},
        {"key": "ps1", "label": "PS1", "status": "miss", "n_rows": 0},
        {"key": "sdss", "label": "SDSS", "status": "not_queried", "n_rows": 0},
    ]

    text = review_app._sed_status_text(rows, [], statuses)

    assert "1 SED points from Gaia GSPC" in text
    assert "no match from PS1" in text
    assert "not queried: SDSS" in text


def test_export_sed_pdf_reports_matplotlib_failure(monkeypatch) -> None:
    def fake_load(candidate_id, extinction_mode, theme_mode):
        return go.Figure(), pd.DataFrame(), []

    def fail_render(*args, **kwargs):
        raise RuntimeError("matplotlib unavailable")

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)
    monkeypatch.setattr(review_app, "render_publication_pdf", fail_render)

    data, message = review_app.export_sed_plot(1, "cand-1", "observed", "black")

    assert data is no_update
    assert "Export failed (SED PDF)" in message
    assert "matplotlib" in message


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

    def fake_render(fig, *args, **kwargs):
        seen["line_color"] = fig.data[0].line.color
        seen["line_width"] = fig.data[0].line.width
        seen["marker_size"] = fig.data[0].marker.size
        seen["marker_opacity"] = fig.data[0].marker.opacity
        seen["right_margin"] = kwargs.get("right_margin")
        seen["width"] = kwargs.get("width")
        seen["height"] = kwargs.get("height")
        seen["title"] = kwargs.get("title")
        return b"%PDF"

    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fake_load)
    monkeypatch.setattr(review_app, "render_publication_pdf", fake_render)

    data, message = review_app.export_sed_plot(1, "cand-1", "corrected", "black")

    assert seen["theme_mode"] == "white"
    assert seen["line_color"] == "#111827"
    assert seen["line_width"] == 2.6
    assert seen["marker_size"] == 11.0
    assert seen["marker_opacity"] == 1.0
    assert seen["right_margin"] >= 250
    assert seen["width"] == 1200
    assert seen["height"] == 820
    assert seen["title"] == "Spectral Energy Distribution"
    assert data is not no_update
    assert "Exported" in message


def test_export_eda_plot_pdf_uses_publication_renderer(monkeypatch) -> None:
    seen = {}
    frame = pd.DataFrame(
        {
            "candidate_id": ["B", "A"],
            "period_n_sources": [2, 0],
            "dipper_score": [4.0, 1.0],
            "status": ["reviewed", "pending"],
        }
    )

    def fake_render(fig, *args, **kwargs):
        seen["fig"] = fig
        seen.update(kwargs)
        return b"%PDF"

    monkeypatch.setattr(review_app, "_current_eda_frame", lambda: frame)
    monkeypatch.setattr(review_app, "render_publication_pdf", fake_render)

    data, message = review_app.export_eda_plot_pdf(
        1,
        {"candidate_ids": ["B", "A"]},
        0,
        "period_n_sources",
        "dipper_score",
        "status",
        None,
        [],
    )

    assert data is not no_update
    assert data["filename"] == "review_eda_dipper_score_vs_period_n_sources.pdf"
    assert "Exported review_eda_dipper_score_vs_period_n_sources.pdf" in message
    assert seen["title"] == "dipper_score vs period_n_sources"
    assert seen["width"] == 1200
    assert seen["height"] == 820
    assert seen["style"] is False
    assert seen["fig"].layout.paper_bgcolor == "white"
    assert seen["fig"].data[-1].name == "current"
    assert seen["fig"].data[-1].customdata[0][0] == "B"


def test_export_eda_plot_pdf_reports_empty_queue() -> None:
    data, message = review_app.export_eda_plot_pdf(
        1,
        {"candidate_ids": []},
        0,
        "period_n_sources",
        "dipper_score",
        None,
        None,
        [],
    )

    assert data is no_update
    assert "No active review queue" in message


def test_export_eda_plot_pdf_reports_missing_metric(monkeypatch) -> None:
    frame = pd.DataFrame({"candidate_id": ["A"], "period_n_sources": [1]})
    monkeypatch.setattr(review_app, "_current_eda_frame", lambda: frame)

    data, message = review_app.export_eda_plot_pdf(
        1,
        {"candidate_ids": ["A"]},
        0,
        "period_n_sources",
        "dipper_score",
        None,
        None,
        [],
    )

    assert data is no_update
    assert "Missing EDA metric: dipper_score" in message


def test_export_eda_plot_pdf_reports_log_filtered_rows(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A"],
            "period_n_sources": [0],
            "dipper_score": [2.0],
        }
    )
    monkeypatch.setattr(review_app, "_current_eda_frame", lambda: frame)

    data, message = review_app.export_eda_plot_pdf(
        1,
        {"candidate_ids": ["A"]},
        0,
        "period_n_sources",
        "dipper_score",
        None,
        None,
        ["logx"],
    )

    assert data is no_update
    assert "log-axis filtering" in message


def test_export_active_plot_png_mode_uses_matplotlib_lightcurve_renderer(monkeypatch) -> None:
    seen = {}

    def fake_candidate_context(candidate_id):
        seen["candidate_id"] = candidate_id
        return {"candidate_id": candidate_id, "asas_sn_id": "asas-1"}, None, None

    def fake_send_file(*args, **kwargs):
        raise AssertionError("PNG display mode should not pass through the old PNG file")

    def fake_build(payload, **kwargs):
        seen["payload"] = payload
        seen.update(kwargs)
        return b"%PDF-1.4\n%%EOF"

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_load_run_params_for_plot_dir", lambda _plot_dir: {"baseline_func": "global_median"})
    monkeypatch.setattr(review_app, "_configured_plot_dir", lambda: None)
    monkeypatch.setattr(review_app.dcc, "send_file", fake_send_file)
    monkeypatch.setattr(review_app, "build_review_lightcurve_publication_pdf", fake_build)

    data, message = review_app.export_active_plot(
        1,
        None,
        "png",
        "/plots/old-review-plot.png",
        3,
        "cand-1",
        {
            "state": {
                "candidate_id": "cand-1",
                "overlay_values": ["raw", "residuals", "phase", "markers", "filter_bad_cameras"],
                "selected_cameras": ["ba"],
                "selected_bands": ["g"],
                "override_period": 2.5,
                "residual_height": 0.24,
                "baseline_opacity": 0.4,
                "yaxis_mode": "flux",
            }
        },
    )

    assert data is not no_update
    assert "malca_plot_4_asas-1.pdf" in message
    assert seen["payload"]["candidate_id"] == "cand-1"
    assert seen["selected_cameras"] == ["ba"]
    assert seen["selected_bands"] == ["g"]
    assert seen["filter_bad_cameras"] is True
    assert seen["show_raw_mag"] is True
    assert seen["show_residuals"] is True
    assert seen["show_phase_fold"] is True
    assert seen["override_period"] == 2.5
    assert seen["yaxis_mode"] == "flux"


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


def test_diagnostic_plots_load_background_when_state_is_stale(monkeypatch) -> None:
    seen = {}

    def fake_candidate_context(candidate_id):
        return {"candidate_id": candidate_id}, None, None

    def fake_render(payload, theme, background=None):
        seen["candidate_id"] = payload["candidate_id"]
        seen["theme"] = theme
        seen["background"] = background
        return ["diagnostic-card"]

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda _path: "fresh-sig")
    monkeypatch.setattr(review_app, "_get_cached_diagnostic_background", lambda _sig: None)
    monkeypatch.setattr(review_app, "_load_or_cache_diagnostic_background", lambda _sig: ({"fresh": True}, False))
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", fake_render)

    panel, status = review_app.update_diagnostic_plots(
        "cand-1",
        "black",
        {"ready": True, "signature": "stale-sig"},
    )

    assert panel == ["diagnostic-card"]
    assert status == ""
    assert seen == {"candidate_id": "cand-1", "theme": "black", "background": {"fresh": True}}


def test_vetting_known_preset_can_leave_uncertain_types_visible() -> None:
    options = {col: [] for col in review_app.VETTING_KNOWN_SELECT_FILTERS}
    options["tns_type"] = [
        {"label": "SN Ia", "value": "SN Ia"},
        {"label": "CV candidate", "value": "CV candidate"},
        {"label": "SN?", "value": "SN?"},
        {"label": "possible SN", "value": "possible SN"},
    ]
    options["vsx_class"] = [
        {"label": "GCAS", "value": "GCAS"},
        {"label": "BE", "value": "BE"},
        {"label": "EA", "value": "EA"},
        {"label": "DSCT", "value": "DSCT"},
        {"label": "EA:", "value": "EA:"},
        {"label": "DSCT:+VAR", "value": "DSCT:+VAR"},
        {"label": "VAR", "value": "VAR"},
    ]
    options["asassn_var_type"] = [
        {"label": "ROT", "value": "ROT"},
        {"label": "ROT:", "value": "ROT:"},
        {"label": "VAR", "value": "VAR"},
    ]
    options["simbad_otype"] = [
        {"label": "V*", "value": "V*"},
        {"label": "Y*?", "value": "Y*?"},
        {"label": "*", "value": "*"},
    ]

    bool_values, select_values = review_app._vetting_known_filter_preset(
        options,
        include_uncertain=False,
    )
    select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, select_values))
    bool_values_by_col = dict(zip(review_app.VETTING_KNOWN_BOOL_FILTERS, bool_values))

    assert bool_values_by_col["vetting_likely_known"] == "Any"
    assert bool_values_by_col["microlens_match"] == "False"
    assert select_values_by_col["vsx_class"] == ["GCAS", "BE", "EA", "DSCT"]
    assert select_values_by_col["asassn_var_type"] == ["ROT"]
    assert select_values_by_col["tns_type"] == ["SN Ia"]
    assert select_values_by_col["simbad_otype"] == ["V*"]


def test_vetting_known_preset_broad_mode_keeps_existing_behavior() -> None:
    options = {col: [] for col in review_app.VETTING_KNOWN_SELECT_FILTERS}
    options["tns_type"] = [
        {"label": "SN Ia", "value": "SN Ia"},
        {"label": "CV candidate", "value": "CV candidate"},
        {"label": "SN?", "value": "SN?"},
    ]
    options["vsx_class"] = [
        {"label": "GCAS", "value": "GCAS"},
        {"label": "BE", "value": "BE"},
        {"label": "EA", "value": "EA"},
        {"label": "VAR", "value": "VAR"},
        {"label": "EA:", "value": "EA:"},
        {"label": "DSCT:+VAR", "value": "DSCT:+VAR"},
    ]

    bool_values, select_values = review_app._vetting_known_filter_preset(
        options,
        include_uncertain=True,
    )
    select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, select_values))
    bool_values_by_col = dict(zip(review_app.VETTING_KNOWN_BOOL_FILTERS, bool_values))

    assert bool_values_by_col["vetting_likely_known"] == "False"
    assert bool_values_by_col["microlens_match"] == "False"
    assert select_values_by_col["vsx_class"] == ["GCAS", "BE", "EA", "VAR", "EA:", "DSCT:+VAR"]
    assert select_values_by_col["tns_type"] == ["SN Ia", "CV candidate", "SN?"]


def test_vetting_known_options_loader_includes_backfilled_vsx_classes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "review.db"
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    with review_app.db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "cand-gcas", "source_path": "run-a", "vsx_class": "GCAS"},
                    {"candidate_id": "cand-be", "source_path": "run-a", "vsx_class": "BE"},
                    {"candidate_id": "cand-var", "source_path": "run-a", "vsx_class": "VAR"},
                    {"candidate_id": "cand-other", "source_path": "run-b", "vsx_class": "EA"},
                ]
            ),
        )

    options = review_app._load_vetting_known_select_options(
        {"source_paths": ["run-a/results/candidates.parquet"]}
    )

    assert options["vsx_class"] == [
        {"label": "BE", "value": "BE"},
        {"label": "GCAS", "value": "GCAS"},
        {"label": "VAR", "value": "VAR"},
    ]


def test_vetting_known_options_refresh_replaces_stale_vsx_classes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "review.db"
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    with review_app.db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "cand-gcas", "source_path": "run-a", "vsx_class": "GCAS"},
                    {"candidate_id": "cand-be", "source_path": "run-a", "vsx_class": "BE"},
                    {"candidate_id": "cand-var", "source_path": "run-a", "vsx_class": "VAR"},
                ]
            ),
        )
    stale_options = {col: [] for col in review_app.VETTING_KNOWN_SELECT_FILTERS}
    stale_options["vsx_class"] = [{"label": "EA", "value": "EA"}]

    options = review_app._fresh_vetting_known_select_options(
        {"source_paths": ["run-a/results/candidates.parquet"]},
        stale_options,
    )
    _bool_values, select_values = review_app._vetting_known_filter_preset(
        options,
        include_uncertain=False,
    )
    select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, select_values))

    assert options["vsx_class"] == [
        {"label": "BE", "value": "BE"},
        {"label": "GCAS", "value": "GCAS"},
        {"label": "VAR", "value": "VAR"},
    ]
    assert select_values_by_col["vsx_class"] == ["BE", "GCAS"]


def test_diagnostic_background_signature_tracks_wal_file(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    db_path.write_text("db", encoding="utf-8")
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    before = review_app._diagnostic_background_signature()
    Path(f"{db_path}-wal").write_text("wal", encoding="utf-8")
    after = review_app._diagnostic_background_signature()

    assert before != after


def test_diagnostic_background_prepares_even_when_details_closed(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda _path: "sig")
    monkeypatch.setattr(review_app, "_load_or_cache_diagnostic_background", lambda _sig: ({"cached": True}, True))

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


def test_dustycult_publication_figures_use_best_fit(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    posterior = {
        "t0": {"median": 10.0},
        "v": {"median": 1.0},
        "b": {"median": 0.0},
        "tau0": {"median": 0.4},
        "lambda0": {"median": 510.0},
        "alpha": {"median": 0.0},
        "sigma_y": {"median": 0.25},
        "sigma_x_plus": {"median": 0.25},
        "sigma_x_minus": {"median": 0.25},
    }
    with review_app.db_connect(db_path) as conn:
        upsert_dustycult_fit(
            conn,
            {
                "candidate_id": "cand-1",
                "mode": "quick",
                "status": "ok",
                "posterior_json": review_app.json.dumps(posterior),
                "stellar_json": review_app.json.dumps({"R": 1.0, "u1": 0.1, "u2": 0.1}),
                "t0_jd": 10.0,
            },
            pd.DataFrame(
                {
                    "candidate_id": ["cand-1"],
                    "mode": ["quick"],
                    "point_id": [1],
                    "time": [10.0],
                    "band": ["g"],
                    "observed": [0.9],
                    "error": [0.02],
                    "lower95": [0.8],
                    "lower68": [0.85],
                    "median": [0.9],
                    "upper68": [0.95],
                    "upper95": [1.0],
                }
            ),
        )
        fit_fig, fit_mode = review_app._dustycult_fit_publication_figure(conn, "cand-1")
        occulter_fig, occulter_mode = review_app._dustycult_occulter_publication_figure(conn, "cand-1")

    assert fit_mode == "quick"
    assert occulter_mode == "quick"
    assert fit_fig.layout.legend.x > 1.0
    assert "$F/F" in fit_fig.layout.yaxis.title.text
    assert "Occulter" in occulter_fig.layout.title.text
    assert any(getattr(trace, "type", "") == "heatmap" for trace in occulter_fig.data)


def test_mini_plot_export_helper_uses_matplotlib_export_path(monkeypatch) -> None:
    seen = {}
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[1, 2], y=[3, 4], mode="markers", name="points"))
    fig.update_layout(title="Mini Plot")

    def fake_render(export_fig, *args, **kwargs):
        seen["trace_type"] = export_fig.data[0].type
        seen["title"] = kwargs.get("title")
        seen["width"] = kwargs.get("width")
        seen["right_margin"] = kwargs.get("right_margin")
        return b"%PDF"

    monkeypatch.setattr(review_app, "render_publication_pdf", fake_render)

    data, message = review_app._export_mini_plot_pdf_from_state(
        {"panel": "external", "name": "cmd"},
        [{"panel": "external", "name": "cmd"}],
        [1],
        [{"panel": "external", "name": "cmd"}],
        [fig],
        "cand-1",
    )

    assert seen["trace_type"] == "scattergl"
    assert seen["title"] == "Mini Plot"
    assert seen["width"] == 1200
    assert seen["right_margin"] == 260
    assert data is not no_update
    assert "malca_external_cmd_cand-1.pdf" in message


def test_mini_plot_export_helper_uses_static_diagnostic_renderer(monkeypatch) -> None:
    seen = {}

    def fake_static_pdf(name, payload, background):
        seen["name"] = name
        seen["payload"] = payload
        seen["background"] = background
        return b"%PDF static"

    def fail_render(*args, **kwargs):
        raise AssertionError("diagnostic static export should not call the generic renderer")

    monkeypatch.setattr(review_app, "_candidate_context", lambda candidate_id: ({"candidate_id": candidate_id}, None, None))
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda _path: "sig")
    monkeypatch.setattr(review_app, "_load_or_cache_diagnostic_background", lambda _sig: ({"cmd_bprp0": [1.0]}, True))
    monkeypatch.setattr(review_app, "build_publication_diagnostic_pdf", fake_static_pdf)
    monkeypatch.setattr(review_app, "render_publication_pdf", fail_render)

    data, message = review_app._export_mini_plot_pdf_from_state(
        {"panel": "diagnostic", "name": "cmd"},
        [{"panel": "diagnostic", "name": "cmd"}],
        [1],
        [{"panel": "diagnostic", "name": "cmd"}],
        [go.Figure()],
        "cand-1",
    )

    assert seen == {
        "name": "cmd",
        "payload": {"candidate_id": "cand-1"},
        "background": {"cmd_bprp0": [1.0]},
    }
    assert data is not no_update
    assert "malca_diagnostic_cmd_cand-1.pdf" in message


def test_mini_plot_export_helper_ignores_zero_click_insertions(monkeypatch) -> None:
    def fail_render(*args, **kwargs):
        raise AssertionError("zero-click callback should not export")

    monkeypatch.setattr(review_app, "render_publication_pdf", fail_render)

    data, message = review_app._export_mini_plot_pdf_from_state(
        {"panel": "diagnostic", "name": "cmd"},
        [{"panel": "diagnostic", "name": "cmd"}],
        [0],
        [{"panel": "diagnostic", "name": "cmd"}],
        [go.Figure()],
        "cand-1",
    )

    assert data is no_update
    assert message is no_update


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
