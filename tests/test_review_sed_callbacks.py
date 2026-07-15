from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pandas as pd
import plotly.graph_objects as go
import pytest
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


def test_queue_filter_state_restores_catalog_neighbor_radius_and_presets() -> None:
    saved = {
        "catalog_neighbor_radius_arcsec": 99,
        "exclude_known_catalog_neighbors": ["yes"],
        "exclude_dipper_catalog_neighbors": [],
    }

    normalized = review_app._normalize_saved_queue_filter_ui_state(saved)
    values = review_app._queue_filter_ui_values_from_state(saved)
    params = review_app._queue_filter_params_from_ui_state(normalized, None, None)

    assert normalized["catalog_neighbor_radius_arcsec"] == 30.0
    assert values[3:6] == (30.0, ["yes"], [])
    assert params["catalog_neighbor_radius_arcsec"] == 30.0
    assert params["exclude_known_catalog_neighbors"] is True
    assert params["exclude_dipper_catalog_neighbors"] is False


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


def test_batch_export_searches_missing_phase_period(tmp_path: Path, monkeypatch) -> None:
    seen = {}
    progress_messages = []

    def fake_candidate_context(candidate_id):
        return {"candidate_id": candidate_id, "asas_sn_id": "asas-1"}, None, None

    def fake_search(payload, **kwargs):
        seen["search_payload"] = payload
        seen["search_kwargs"] = kwargs
        return {"best_period": 2.75, "method": "PDM"}, "PDM: P=2.75000 d"

    def fake_build(payload, **kwargs):
        seen["payload"] = payload
        seen["build_kwargs"] = kwargs
        return b"%PDF-1.4\n%%EOF"

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_review_plot_dir_for_context", lambda _source_path: None)
    monkeypatch.setattr(review_app, "_load_run_params_for_plot_dir", lambda _plot_dir: {})
    monkeypatch.setattr(review_app, "_run_period_search_for_payload", fake_search)
    monkeypatch.setattr(review_app, "build_review_lightcurve_publication_pdf", fake_build)

    message = review_app.handle_batch_export(
        progress_messages.append,
        1,
        "all",
        str(tmp_path),
        {"candidate_ids": ["cand-1"]},
        {"state": {"overlay_values": ["phase"], "selected_bands": ["g"]}},
        0.2,
        5.0,
        "pdm",
    )

    assert (tmp_path / "malca_plot_asas-1.pdf").exists()
    assert seen["search_payload"]["candidate_id"] == "cand-1"
    assert seen["search_kwargs"] == {"min_period": 0.2, "max_period": 5.0, "method": "pdm"}
    assert seen["build_kwargs"]["show_phase_fold"] is True
    assert seen["build_kwargs"]["override_period"] == 2.75
    assert seen["build_kwargs"]["override_period_source"] == "Batch auto-search (PDM)"
    assert "phase found: 1" in message
    assert "phase skipped: 0" in message
    assert any("PDF ok:" in item for item in progress_messages)


def test_batch_export_skips_phase_panel_when_period_search_fails(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_candidate_context(candidate_id):
        return {"candidate_id": candidate_id, "asas_sn_id": "asas-1"}, None, None

    def fake_build(payload, **kwargs):
        seen["payload"] = payload
        seen["build_kwargs"] = kwargs
        return b"%PDF-1.4\n%%EOF"

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_review_plot_dir_for_context", lambda _source_path: None)
    monkeypatch.setattr(review_app, "_load_run_params_for_plot_dir", lambda _plot_dir: {})
    monkeypatch.setattr(review_app, "_run_period_search_for_payload", lambda *args, **kwargs: (None, "No valid period"))
    monkeypatch.setattr(review_app, "build_review_lightcurve_publication_pdf", fake_build)

    message = review_app.handle_batch_export(
        lambda _message: None,
        1,
        "all",
        str(tmp_path),
        {"candidate_ids": ["cand-1"]},
        {"state": {"overlay_values": ["phase"], "selected_bands": ["g"]}},
        None,
        None,
        "pdm",
    )

    assert (tmp_path / "malca_plot_asas-1.pdf").exists()
    assert seen["build_kwargs"]["show_phase_fold"] is False
    assert seen["build_kwargs"]["override_period"] is None
    assert seen["build_kwargs"]["phase_period_pending"] is False
    assert seen["build_kwargs"]["suppress_catalog_phase_period"] is False
    assert "phase found: 0" in message
    assert "phase skipped: 1" in message


def test_external_followup_panel_renders_without_details_open_state(monkeypatch) -> None:
    seen = {}

    def fake_candidate_context(candidate_id):
        return {"candidate_id": candidate_id, "simbad_main_id": "Star"}, None, None

    def fake_render(payload, candidate_id, theme, **_kwargs):
        seen["candidate_id"] = candidate_id
        seen["theme"] = theme
        return ["external-card"]

    monkeypatch.setattr(review_app, "_candidate_context", fake_candidate_context)
    monkeypatch.setattr(review_app, "_render_external_followup", fake_render)

    panel, status = review_app.update_external_followup_panel("cand-1", "black")

    assert panel == ["external-card"]
    assert status == "Loaded external data for cand-1."
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


def test_diagnostic_plots_show_loading_when_background_state_is_stale(monkeypatch) -> None:
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
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", fake_render)

    panel, status = review_app.update_diagnostic_plots(
        "cand-1",
        "black",
        {"ready": True, "signature": "stale-sig"},
    )

    assert "Loading population background" in status
    assert "lazy-panel-placeholder" in getattr(panel, "className", "")
    assert seen == {}


def test_vetting_known_variable_preset_leaves_uncertain_types_visible() -> None:
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
        {"label": "UXOR", "value": "UXOR"},
        {"label": "YSO/DIP", "value": "YSO/DIP"},
        {"label": "DSCT", "value": "DSCT"},
        {"label": "EA:", "value": "EA:"},
        {"label": "EA/SD:", "value": "EA/SD:"},
        {"label": "BE:", "value": "BE:"},
        {"label": "BY+Microlens:", "value": "BY+Microlens:"},
        {"label": "ACEP|CEP", "value": "ACEP|CEP"},
        {"label": "BE|GCAS|SDOR|WR", "value": "BE|GCAS|SDOR|WR"},
        {"label": "DSCT:+VAR", "value": "DSCT:+VAR"},
        {"label": "VAR", "value": "VAR"},
        {"label": "nan", "value": "nan"},
    ]
    options["asassn_var_type"] = [
        {"label": "ROT", "value": "ROT"},
        {"label": "YSO", "value": "YSO"},
        {"label": "ROT:", "value": "ROT:"},
        {"label": "VAR", "value": "VAR"},
    ]
    options["gaia_var_class"] = [
        {"label": "ECL", "value": "ECL"},
        {"label": "YSO", "value": "YSO"},
        {"label": "RR", "value": "RR"},
        {"label": "DSCT|GDOR|SXPHE", "value": "DSCT|GDOR|SXPHE"},
    ]
    options["ztf_var_type"] = [
        {"label": "EA", "value": "EA"},
        {"label": "RSCVN", "value": "RSCVN"},
    ]
    options["alerce_lc_class"] = [
        {"label": "Periodic", "value": "Periodic"},
        {"label": "YSO", "value": "YSO"},
        {"label": "CV/Nova", "value": "CV/Nova"},
    ]
    options["yso_class"] = [
        {"label": "Class II", "value": "Class II"},
        {"label": "Main Sequence", "value": "Main Sequence"},
    ]
    options["simbad_otype"] = [
        {"label": "EB*", "value": "EB*"},
        {"label": "SB*", "value": "SB*"},
        {"label": "V*", "value": "V*"},
        {"label": "LP*", "value": "LP*"},
        {"label": "Or*", "value": "Or*"},
        {"label": "Y*O", "value": "Y*O"},
        {"label": "Y*?", "value": "Y*?"},
        {"label": "TT*", "value": "TT*"},
        {"label": "Be*", "value": "Be*"},
        {"label": "IR", "value": "IR"},
        {"label": "*", "value": "*"},
    ]

    bool_values, select_values = review_app._vetting_known_filter_preset(
        options,
        policy="known_variables",
    )
    select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, select_values))
    bool_values_by_col = dict(zip(review_app.VETTING_KNOWN_BOOL_FILTERS, bool_values))

    assert bool_values_by_col["vetting_likely_known"] == "False"
    assert bool_values_by_col["microlens_match"] == "False"
    assert bool_values_by_col["nearby_vsx_dipper_contaminant"] == "False"
    assert select_values_by_col["vsx_class"] == [
        "GCAS",
        "BE",
        "EA",
        "UXOR",
        "YSO/DIP",
        "DSCT",
        "ACEP|CEP",
        "BE|GCAS|SDOR|WR",
    ]
    assert select_values_by_col["asassn_var_type"] == ["ROT", "YSO"]
    assert select_values_by_col["gaia_var_class"] == ["ECL", "YSO", "RR", "DSCT|GDOR|SXPHE"]
    assert select_values_by_col["ztf_var_type"] == ["EA", "RSCVN"]
    assert select_values_by_col["alerce_lc_class"] == ["Periodic", "YSO", "CV/Nova"]
    assert select_values_by_col["yso_class"] == []
    assert select_values_by_col["tns_type"] == ["SN Ia"]
    assert select_values_by_col["simbad_otype"] == ["EB*", "SB*", "V*", "LP*", "Or*"]

    dipper_bool_values, dipper_select_values = review_app._vetting_known_filter_preset(
        options,
        policy="dipper_contaminants",
    )
    dipper_select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, dipper_select_values))
    dipper_bool_values_by_col = dict(zip(review_app.VETTING_KNOWN_BOOL_FILTERS, dipper_bool_values))
    assert dipper_bool_values_by_col["vetting_likely_known"] == "Any"
    assert dipper_bool_values_by_col["microlens_match"] == "Any"
    assert dipper_bool_values_by_col["nearby_vsx_dipper_contaminant"] == "False"
    assert dipper_select_values_by_col["vsx_class"] == ["GCAS", "BE", "EA", "UXOR", "YSO/DIP", "BE|GCAS|SDOR|WR"]
    assert dipper_select_values_by_col["asassn_var_type"] == ["YSO"]
    assert dipper_select_values_by_col["gaia_var_class"] == ["ECL", "YSO"]
    assert dipper_select_values_by_col["ztf_var_type"] == ["EA"]
    assert dipper_select_values_by_col["alerce_lc_class"] == ["YSO", "CV/Nova"]
    assert dipper_select_values_by_col["yso_class"] == ["Class II"]
    assert dipper_select_values_by_col["simbad_otype"] == ["EB*", "SB*", "Or*", "Y*O", "TT*", "Be*"]


def test_vetting_dipper_contaminant_preset_selects_target_safe_simbad_values() -> None:
    options = {col: [] for col in review_app.VETTING_KNOWN_SELECT_FILTERS}
    options["simbad_otype"] = [
        {"label": "EB*", "value": "EB*"},
        {"label": "SB*", "value": "SB*"},
        {"label": "RR*", "value": "RR*"},
        {"label": "CV*", "value": "CV*"},
        {"label": "No*", "value": "No*"},
        {"label": "HXB", "value": "HXB"},
        {"label": "V*", "value": "V*"},
        {"label": "LP*", "value": "LP*"},
        {"label": "Mi*", "value": "Mi*"},
        {"label": "Or*", "value": "Or*"},
        {"label": "Y*O", "value": "Y*O"},
        {"label": "Y*?", "value": "Y*?"},
        {"label": "TT*", "value": "TT*"},
        {"label": "Be*", "value": "Be*"},
        {"label": "Em*", "value": "Em*"},
        {"label": "IR", "value": "IR"},
        {"label": "Rad", "value": "Rad"},
        {"label": "X", "value": "X"},
        {"label": "FIR", "value": "FIR"},
        {"label": "*", "value": "*"},
    ]

    bool_values, select_values = review_app._vetting_known_filter_preset(
        options,
        policy="dipper_contaminants",
    )
    select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, select_values))
    bool_values_by_col = dict(zip(review_app.VETTING_KNOWN_BOOL_FILTERS, bool_values))

    assert bool_values_by_col["vetting_likely_known"] == "Any"
    assert bool_values_by_col["microlens_match"] == "Any"
    assert bool_values_by_col["nearby_vsx_dipper_contaminant"] == "False"
    assert select_values_by_col["simbad_otype"] == [
        "EB*",
        "SB*",
        "CV*",
        "No*",
        "HXB",
        "Or*",
        "Y*O",
        "TT*",
        "Be*",
        "Em*",
    ]


def test_vetting_filter_toggles_preserve_other_filters_and_shared_values() -> None:
    options = {col: [] for col in review_app.VETTING_KNOWN_SELECT_FILTERS}
    options["vsx_class"] = [
        {"label": "EA", "value": "EA"},
        {"label": "UXOR", "value": "UXOR"},
        {"label": "DSCT", "value": "DSCT"},
        {"label": "USER_ONLY", "value": "USER_ONLY"},
    ]
    options["yso_class"] = [
        {"label": "Class II", "value": "Class II"},
        {"label": "Main Sequence", "value": "Main Sequence"},
    ]

    vsx_idx = review_app.VETTING_KNOWN_SELECT_FILTERS.index("vsx_class")
    yso_idx = review_app.VETTING_KNOWN_SELECT_FILTERS.index("yso_class")
    current_select_values = [[] for _col in review_app.VETTING_KNOWN_SELECT_FILTERS]
    current_select_values[vsx_idx] = ["USER_ONLY"]

    known_bool_values, known_select_values = review_app._apply_vetting_known_filter_toggle(
        options,
        current_bool_values=["Any", "Any", "Any"],
        current_select_values=current_select_values,
        policy="known_variables",
        enabled=True,
    )

    assert known_bool_values == ["False", "False", "False"]
    assert known_select_values[vsx_idx] == ["USER_ONLY", "EA", "UXOR", "DSCT"]
    assert known_select_values[yso_idx] == []

    both_bool_values, both_select_values = review_app._apply_vetting_known_filter_toggle(
        options,
        current_bool_values=known_bool_values,
        current_select_values=known_select_values,
        policy="dipper_contaminants",
        enabled=True,
        other_policy="known_variables",
        other_enabled=True,
    )

    assert both_bool_values == ["False", "False", "False"]
    assert both_select_values[vsx_idx] == ["USER_ONLY", "EA", "UXOR", "DSCT"]
    assert both_select_values[yso_idx] == ["Class II"]

    dipper_only_bool_values, dipper_only_select_values = review_app._apply_vetting_known_filter_toggle(
        options,
        current_bool_values=both_bool_values,
        current_select_values=both_select_values,
        policy="known_variables",
        enabled=False,
        other_policy="dipper_contaminants",
        other_enabled=True,
    )

    assert dipper_only_bool_values == ["Any", "Any", "False"]
    assert dipper_only_select_values[vsx_idx] == ["USER_ONLY", "EA", "UXOR"]
    assert dipper_only_select_values[yso_idx] == ["Class II"]

    user_only_bool_values, user_only_select_values = review_app._apply_vetting_known_filter_toggle(
        options,
        current_bool_values=dipper_only_bool_values,
        current_select_values=dipper_only_select_values,
        policy="dipper_contaminants",
        enabled=False,
        other_policy="known_variables",
        other_enabled=False,
    )

    assert user_only_bool_values == ["Any", "Any", "Any"]
    assert user_only_select_values[vsx_idx] == ["USER_ONLY"]
    assert user_only_select_values[yso_idx] == []


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

    assert [option["value"] for option in options["vsx_class"]] == ["BE", "GCAS", "VAR"]


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
        policy="dipper_contaminants",
    )
    select_values_by_col = dict(zip(review_app.VETTING_KNOWN_SELECT_FILTERS, select_values))

    assert [option["value"] for option in options["vsx_class"]] == ["BE", "GCAS", "VAR"]
    assert select_values_by_col["vsx_class"] == ["BE", "GCAS"]


def test_diagnostic_background_signature_tracks_wal_file(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    db_path.write_text("db", encoding="utf-8")
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    before = review_app._diagnostic_background_signature()
    Path(f"{db_path}-wal").write_text("wal", encoding="utf-8")
    after = review_app._diagnostic_background_signature()

    assert before != after


def test_diagnostic_background_skips_when_details_closed(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda _path: "sig")

    def fail(*_args, **_kwargs):
        raise AssertionError("closed diagnostics should not prepare background")

    monkeypatch.setattr(review_app, "_load_or_cache_diagnostic_background", fail)

    try:
        review_app._prepare_diagnostic_background(
            False,
            None,
            None,
            {"signature": "", "ready": False, "cached": False, "token": 3},
        )
    except review_app.dash.exceptions.PreventUpdate:
        return
    raise AssertionError("closed diagnostics should raise PreventUpdate")


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
    assert values[-5:-2] == (1.1, 0.2, 0.3)
    assert "stored_event_columns" in values[-2]
    assert values[-1]["candidate_id"] == "cand-1"
    assert values[-1]["source"] == "stored_event_columns"


def test_dustycult_warning_renders_but_failed_does_not_render_model_graph(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(
        review_app,
        "check_dustycult_available",
        lambda: DustyCultAvailability(True, "julia", tmp_path, tmp_path / "fit_lightcurve.jl", "available"),
    )
    posterior = {
        "t0": {"median": 10.0, "p16": 9.9, "p84": 10.1},
        "v": {"median": 1.0, "p16": 0.9, "p84": 1.1},
        "b": {"median": 0.0, "p16": -0.1, "p84": 0.1},
        "tau0": {"median": 0.4, "p16": 0.3, "p84": 0.5},
        "lambda0": {"median": 510.0, "p16": 500.0, "p84": 520.0},
        "alpha": {"median": 0.0, "p16": -0.1, "p84": 0.1},
        "sigma_y": {"median": 0.25, "p16": 0.2, "p84": 0.3},
        "sigma_x_plus": {"median": 0.25, "p16": 0.2, "p84": 0.3},
        "sigma_x_minus": {"median": 0.25, "p16": 0.2, "p84": 0.3},
    }
    curves = pd.DataFrame(
        {
            "candidate_id": ["cand-1", "cand-1"],
            "mode": ["quick", "quick"],
            "point_id": [1, 2],
            "time": [9.5, 10.5],
            "band": ["g", "V"],
            "observed": [0.95, 0.96],
            "error": [0.02, 0.02],
            "lower95": [0.9, 0.91],
            "lower68": [0.93, 0.94],
            "median": [0.95, 0.96],
            "upper68": [0.98, 0.99],
            "upper95": [1.0, 1.01],
        }
    )
    with review_app.db_connect(db_path) as conn:
        upsert_dustycult_fit(
            conn,
            {
                "candidate_id": "cand-1",
                "mode": "quick",
                "status": "warning",
                "error": "one-band warning",
                "posterior_json": review_app.json.dumps(posterior),
                "stellar_json": review_app.json.dumps({"R": 1.0, "u1": 0.1, "u2": 0.1}),
                "t0_jd": 10.0,
            },
            curves,
        )

    panel = review_app._render_dustycult_result_panel("cand-1", "black", 0)
    assert "dustycult-fit-plot" in _collect_component_ids(panel)
    assert "one-band warning" in "\n".join(_collect_text(panel))

    with review_app.db_connect(db_path) as conn:
        upsert_dustycult_fit(
            conn,
            {
                "candidate_id": "cand-1",
                "mode": "quick",
                "status": "failed",
                "error": "quality gate failed",
            },
            pd.DataFrame(),
        )

    failed_panel = review_app._render_dustycult_result_panel("cand-1", "black", 1)
    assert "dustycult-fit-plot" not in _collect_component_ids(failed_panel)
    assert "quality gate failed" in "\n".join(_collect_text(failed_panel))


def test_dustycult_fit_click_recomputes_stale_controls_and_preserves_current_manual(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(review_app, "DB_PATH", str(tmp_path / "review.db"))
    monkeypatch.setattr(review_app, "_candidate_context", lambda candidate_id: ({"candidate_id": candidate_id}, "/tmp/cand.dat2", "source"))
    monkeypatch.setattr(review_app, "_effective_local_lc_path", lambda payload, stored_lc_path=None, source_path=None: stored_lc_path)
    monkeypatch.setattr(review_app, "_review_plot_dir_for_context", lambda _source_path: tmp_path)
    monkeypatch.setattr(review_app, "_load_run_params_for_plot_dir", lambda _plot_dir: {"baseline_func": "global_median"})
    defaults = {
        "source": "stored_event_columns",
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
    }
    default_calls = {"count": 0}
    captured: list[dict[str, object]] = []

    def fake_defaults(*_args, **_kwargs):
        default_calls["count"] += 1
        return dict(defaults)

    def fake_run(_conn, _candidate_id, _payload, **kwargs):
        captured.append(dict(kwargs["controls"]))
        return {"status": "ok", "runtime_sec": 0.1, "artifact_dir": "/tmp/dustycult"}

    monkeypatch.setattr(review_app, "control_defaults_for_candidate", fake_defaults)
    monkeypatch.setattr(review_app, "run_dustycult_fit", fake_run)

    stale_values = [99.0 for _key, _field_id, _label, _step in review_app._DUSTYCULT_CONTROL_FIELDS]
    message, _token = review_app._dustycult_fit_callback_impl(
        "dustycult-quick-fit-btn",
        1,
        0,
        "cand-1",
        0,
        {"candidate_id": "other", "source": "manual_controls"},
        *stale_values,
    )

    assert "Status: ok" in message
    assert default_calls["count"] == 1
    assert captured[-1]["start_jd"] == 1.0
    assert captured[-1]["_dustycult_window_source"] == "stored_event_columns"

    manual = dict(defaults)
    manual.update({"start_jd": 10.0, "end_jd": 20.0, "t0_jd": 15.0})
    current_values = [manual.get(key) for key, _field_id, _label, _step in review_app._DUSTYCULT_CONTROL_FIELDS]
    review_app._dustycult_fit_callback_impl(
        "dustycult-quick-fit-btn",
        2,
        0,
        "cand-1",
        1,
        {"candidate_id": "cand-1", "source": "manual_controls"},
        *current_values,
    )

    assert default_calls["count"] == 1
    assert captured[-1]["start_jd"] == 10.0
    assert captured[-1]["_dustycult_window_source"] == "manual_controls"


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
                "status": "warning",
                "error": "quality warning",
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
        upsert_dustycult_fit(
            conn,
            {
                "candidate_id": "cand-1",
                "mode": "quick",
                "status": "failed",
                "error": "quality failed",
            },
            pd.DataFrame(),
        )
        with pytest.raises(ValueError, match="not exportable"):
            review_app._dustycult_fit_publication_figure(conn, "cand-1")

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


def _collect_component_ids(node) -> list[str]:
    ids: list[str] = []

    def walk(item) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        component_id = getattr(item, "id", None)
        if component_id is not None:
            ids.append(str(component_id))
        walk(getattr(item, "children", None))

    walk(node)
    return ids
