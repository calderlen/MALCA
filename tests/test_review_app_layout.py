from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
import types
from pathlib import Path

import dash
import pandas as pd


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
from malca.review.app import EXTERNAL_SOURCE_VIEW_OPTIONS, _render_external_followup, app
from malca.review.store import count_queue, db_connect, upsert_candidates_frame
from malca.review.cutouts import CUTOUT_SURVEYS, DEFAULT_CUTOUT_SURVEY_KEY


_APP_SOURCE_DIR = Path(review_app.__file__).resolve().parent


def _component_ids_in_order(node: object) -> list[object]:
    ids: list[object] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        cid = getattr(item, "id", None)
        if cid is not None:
            ids.append(cid)
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return ids


def _component_text_in_order(node: object) -> list[str]:
    texts: list[str] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None:
            return
        if isinstance(item, (str, int, float, bool)):
            texts.append(str(item))
            return
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return texts


def _graph_configs_in_order(node: object) -> list[dict]:
    configs: list[dict] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        config = getattr(item, "config", None)
        if isinstance(config, dict):
            configs.append(config)
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return configs


def _components_by_type(node: object, target_type: str) -> list[object]:
    matches: list[object] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        if item.__class__.__name__ == target_type:
            matches.append(item)
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return matches


def _component_by_id(node: object, target_id: object) -> object | None:
    found: object | None = None

    def walk(item: object) -> None:
        nonlocal found
        if found is not None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        if getattr(item, "id", None) == target_id:
            found = item
            return
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return found


def _components_with_class(node: object, class_fragment: str) -> list[object]:
    matches: list[object] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        class_name = str(getattr(item, "className", "") or "")
        if class_fragment in class_name:
            matches.append(item)
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return matches


def _props(component: object) -> dict:
    if component is None:
        return {}
    return component.to_plotly_json().get("props", {})


def test_candidate_panels_appear_before_diagnostic_plots() -> None:
    ids = _component_ids_in_order(app.layout)

    external_idx = ids.index("external-followup-details")
    sed_idx = ids.index("sed-details")
    dustycult_idx = ids.index("dustycult-details")
    sed_button_idx = ids.index("rerun-stage-sed-photometry-btn")
    multi_survey_button_idx = ids.index("rerun-stage-multi-survey-btn")
    candidate_panels_idx = ids.index("candidate-panels-details")
    diagnostic_idx = ids.index("diagnostic-plots-details")
    metadata_idx = ids.index("candidate-info-grid")
    run_config_idx = ids.index("run-config-details")

    assert external_idx < metadata_idx
    assert external_idx < sed_idx < metadata_idx
    assert sed_idx < dustycult_idx < candidate_panels_idx < metadata_idx < diagnostic_idx
    assert sed_button_idx < metadata_idx
    assert multi_survey_button_idx < metadata_idx
    assert diagnostic_idx < run_config_idx


def test_layout_graphs_disable_plotly_image_export() -> None:
    configs = _graph_configs_in_order(app.layout)

    assert configs
    assert all("toImage" in config.get("modeBarButtonsToRemove", []) for config in configs)


def test_dash_runtime_endpoints_are_not_cached_by_browsers() -> None:
    client = app.server.test_client()

    for path in ("/_dash-layout", "/_dash-dependencies"):
        response = client.get(path)
        assert response.status_code == 200
        assert "no-store" in response.headers.get("Cache-Control", "")
        assert response.headers.get("Pragma") == "no-cache"


def test_layout_graphs_enable_dash_mathjax_without_dash_responsive_prop() -> None:
    graphs = _components_by_type(app.layout, "Graph")

    assert graphs
    assert all(_props(graph).get("mathjax") is True for graph in graphs)
    assert all(_props(graph).get("responsive") is not True for graph in graphs)


def test_metadata_markdown_enables_dash_mathjax() -> None:
    value_markdowns = _components_by_type(review_app._copyable_math_value(r"$\mathrm{JD}$"), "Markdown")
    stats_markdowns = _components_by_type(review_app._render_stat_cards([("stats_period", "1.0")]), "Markdown")

    assert value_markdowns
    assert stats_markdowns
    assert all(_props(markdown).get("mathjax") is True for markdown in [*value_markdowns, *stats_markdowns])


def test_review_app_source_avoids_chromium_sensitive_graph_flags() -> None:
    source = "\n".join(path.read_text() for path in _APP_SOURCE_DIR.rglob("*.py"))

    assert "responsive=True" not in source


def test_primary_light_curve_graph_uses_plotly_config_responsiveness_only() -> None:
    graph = _component_by_id(app.layout, "interactive-plot")

    assert graph is not None
    props = _props(graph)
    assert props["className"] == "plot-native"
    assert props.get("responsive") is not True
    assert props["config"]["responsive"] is True
    assert "figure" not in props
    assert props["style"] == {
        "display": "block",
        "width": "100%",
        "height": "100%",
        "min-height": "600px",
    }


def test_primary_light_curve_css_does_not_force_plotly_internal_containers() -> None:
    assert ".plot-native .plot-container" not in app.index_string
    assert ".plot-native .svg-container" not in app.index_string
    assert ".plot-frame #plot-image" not in app.index_string


def test_primary_light_curve_has_no_custom_resize_callback() -> None:
    scripts = "\n".join(getattr(app, "_inline_scripts", []))
    resize_callbacks = [
        spec for output, spec in app.callback_map.items()
        if "plot-resize-trigger.data" in output
    ]

    assert not resize_callbacks
    assert "plot-resize-trigger" not in scripts
    assert "window.__malcaInteractivePlotState" not in scripts
    assert "window.Plotly.react(root, data, layout, plotConfig)" not in scripts
    assert "MutationObserver" not in scripts
    assert "setInterval(function()" not in scripts


def test_layout_embeds_eda_panel() -> None:
    ids = _component_ids_in_order(app.layout)

    assert "eda-panel-toggle" in ids
    assert "eda-splitter" in ids
    assert "eda-drag-handle" in ids
    assert "eda-panel" in ids
    assert "eda-panel-state" in ids
    assert "eda-selection-candidate-ids" in ids
    assert "eda-collapse-btn" in ids
    assert "eda-expand-btn" in ids
    assert "eda-x-metric" in ids
    assert "eda-y-metric" in ids
    assert "eda-color-metric" in ids
    assert "eda-symbol-metric" in ids
    assert "eda-selection-mode" in ids
    assert "eda-clear-selection-btn" in ids
    assert "eda-custom-graph" in ids
    assert "eda-candidate-table" in ids
    assert "eda-plot-export-download" in ids
    assert "eda-export-pdf-btn" in ids
    assert "eda-export-status" in ids


def test_layout_exposes_lazy_feedback_and_copy_initializer() -> None:
    ids = _component_ids_in_order(app.layout)

    assert "metadata-copy-init" in ids
    assert "cutout-selected-survey" in ids
    assert "external-followup-summary" in ids
    assert "sed-summary" in ids
    assert "dustycult-summary" in ids
    assert "phoebe-summary" in ids
    assert "diagnostic-plots-summary" in ids
    assert "external-followup-status" in ids
    assert "sed-status" in ids
    assert "dustycult-config-status" in ids
    assert "phoebe-config-status" in ids
    assert "diagnostic-plots-status" in ids
    assert "run-config-status" in ids


def test_lazy_panel_callbacks_use_summary_clicks() -> None:
    callback_inputs_by_output = {
        output: {(item["id"], item["property"]) for item in spec.get("inputs", [])}
        for output, spec in app.callback_map.items()
    }

    expected = {
        "external-followup-panel.children": ("external-followup-summary", "n_clicks"),
        "sed-plot-panel.children": ("sed-summary", "n_clicks"),
        "dustycult-result-panel.children": ("dustycult-summary", "n_clicks"),
        "phoebe-result-panel.children": ("phoebe-summary", "n_clicks"),
        "diagnostic-plots-panel.children": ("diagnostic-plots-summary", "n_clicks"),
        "diagnostic-background-state.data": ("diagnostic-plots-summary", "n_clicks"),
    }
    for output_id, input_id in expected.items():
        matches = [
            inputs
            for output, inputs in callback_inputs_by_output.items()
            if output_id in output
        ]
        assert matches, output_id
        assert input_id in matches[0]
        assert (output_id.rsplit(".", 1)[0].replace("-panel", "-details"), "open") not in matches[0]


def test_external_followup_renders_static_cutout_panel() -> None:
    cards = _render_external_followup(
        {"candidate_id": "C1", "ra": 240.48595227, "dec": 20.0},
        "C1",
    )

    survey_select = _component_by_id(cards, "cutout-survey-select")
    image = _component_by_id(cards, "cutout-image")
    fwhm_overlay = _component_by_id(cards, "cutout-asassn-fwhm-overlay")
    source_link = _component_by_id(cards, "cutout-source-link")
    status = _component_by_id(cards, "cutout-status")

    select_props = _props(survey_select)
    image_props = _props(image)
    overlay_props = _props(fwhm_overlay)
    link_props = _props(source_link)
    assert select_props["value"] == DEFAULT_CUTOUT_SURVEY_KEY
    assert select_props["disabled"] is False
    assert [option["label"] for option in select_props["options"]] == [survey.label for survey in CUTOUT_SURVEYS]
    assert "CDS%2FP%2FPanSTARRS%2FDR1%2Fcolor-i-r-g" in image_props["src"]
    assert "fov=0.03333333333" in image_props["src"]
    assert "title" not in overlay_props
    assert "aria-label" not in overlay_props
    assert overlay_props["style"]["width"] == "13.33%"
    assert overlay_props["style"]["height"] == "13.33%"
    assert "display" not in overlay_props["style"]
    assert link_props["href"] == image_props["src"]
    assert "PanSTARRS DR1 color" in str(_props(status).get("children"))


def test_external_followup_uses_dss2_default_for_southern_cutout() -> None:
    cards = _render_external_followup(
        {"candidate_id": "C1", "ra": 240.48595227, "dec": -55.342371},
        "C1",
        selected_cutout_survey="vtss-ha",
    )

    survey_select = _component_by_id(cards, "cutout-survey-select")
    image = _component_by_id(cards, "cutout-image")
    status = _component_by_id(cards, "cutout-status")

    assert _props(survey_select)["value"] == "dss2"
    assert "CDS%2FP%2FDSS2%2Fcolor" in _props(image)["src"]
    assert "DSS2" in str(_props(status).get("children"))


def test_external_followup_cutout_handles_missing_coordinates() -> None:
    cards = _render_external_followup({"candidate_id": "C2"}, "C2")

    survey_select = _component_by_id(cards, "cutout-survey-select")
    image = _component_by_id(cards, "cutout-image")
    fwhm_overlay = _component_by_id(cards, "cutout-asassn-fwhm-overlay")
    source_link = _component_by_id(cards, "cutout-source-link")
    status = _component_by_id(cards, "cutout-status")

    assert _props(survey_select)["disabled"] is True
    assert _props(image)["src"] == ""
    assert _props(fwhm_overlay)["style"]["display"] == "none"
    assert _props(source_link)["href"] == "#"
    assert "RA/Dec" in str(_props(status).get("children"))


def test_cutout_fwhm_overlay_uses_simple_negative_circle_without_crosshair_or_glow() -> None:
    css = (_APP_SOURCE_DIR / "styles.py").read_text(encoding="utf-8")

    assert ".cutout-crosshair" not in css
    assert "mix-blend-mode: difference" in css
    assert "border: 2px solid #fff" in css
    overlay_block = css.split(".cutout-asassn-fwhm-overlay", 1)[1].split("}", 1)[0]
    assert "box-shadow" not in overlay_block


def test_cutout_selector_callback_is_separate_from_plot_rendering() -> None:
    callback_specs = [
        spec
        for output, spec in app.callback_map.items()
        if "cutout-image.src" in output
    ]

    assert callback_specs
    inputs = {(item["id"], item["property"]) for item in callback_specs[0].get("inputs", [])}
    outputs = str(callback_specs[0].get("output", ""))
    assert ("cutout-survey-select", "value") in inputs
    assert ("current-candidate-id", "data") not in inputs
    assert ("plot-render-request", "data") not in inputs
    assert "plot-render-request" not in outputs


def test_stat_cards_are_full_width_and_copyable() -> None:
    cards = review_app._render_stat_cards([
        ("stats_photometry_weighted_mean_sem", r"2.64723 \times 10^{-4}"),
    ])

    assert len(cards) == 1
    stats = cards[0]
    assert getattr(stats, "className", "") == "stats-details"
    inner = getattr(stats, "children", [None, None])[1]
    assert "stats-sections-grid" in getattr(inner, "className", "")

    buttons = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        if getattr(item, "className", "") == "metadata-copy-btn":
            buttons.append(item)
        walk(getattr(item, "children", None))

    walk(stats)
    assert buttons
    assert buttons[0].to_plotly_json()["props"]["data-copy-text"] == r"2.64723 \times 10^{-4}"


def test_streamlined_metadata_layout_renders_summary_as_normal_rows() -> None:
    payload = {
        "asas_sn_id": "618475536448",
        "gaia_id": "5885468452501959424",
        "dipper_score": 18.6,
        "dip_significant": True,
        "jump_count": 0,
        "baseline_mag": 12.696,
    }
    grouped = review_app.extract_review_metadata_grouped(payload, round_sigfigs=True)
    feature_rows = review_app.extract_review_metadata_feature_rows(payload, round_sigfigs=True)

    layout = review_app._render_metadata_review_layout(
        payload,
        grouped,
        [("stats_cadence_mean_dt_days", "2.83856")],
        feature_rows,
    )

    class_order = [
        str(getattr(component, "className", "") or "")
        for component in _components_with_class(layout, "review-feature-section")
    ]
    assert any("review-feature-section-review-summary" in class_name for class_name in class_order)
    assert any("review-feature-section-dip-evidence" in class_name for class_name in class_order)
    assert any("all-features-details" in class_name for class_name in class_order)
    assert next(
        i for i, class_name in enumerate(class_order) if "review-feature-section-review-summary" in class_name
    ) < next(
        i for i, class_name in enumerate(class_order) if "review-feature-section-dip-evidence" in class_name
    )
    assert next(
        i for i, class_name in enumerate(class_order) if "review-feature-section-review-summary" in class_name
    ) < next(
        i for i, class_name in enumerate(class_order) if "all-features-details" in class_name
    )

    summary = _components_with_class(layout, "review-feature-section-review-summary")[0]
    assert _props(summary).get("open") is True
    assert _components_with_class(summary, "meta-field-row")
    assert _components_with_class(layout, "review-summary-tile") == []


def test_streamlined_metadata_layout_opens_dip_and_collapses_empty_jump() -> None:
    payload = {
        "dipper_score": 18.6,
        "dip_significant": True,
        "jumper_score": 0,
        "jump_count": 0,
        "jump_significant": False,
    }
    grouped = review_app.extract_review_metadata_grouped(payload, round_sigfigs=True)
    feature_rows = review_app.extract_review_metadata_feature_rows(payload, round_sigfigs=True)
    layout = review_app._render_metadata_review_layout(payload, grouped, [], feature_rows)

    dip_section = _components_with_class(layout, "review-feature-section-dip-evidence")[0]
    jump_section = _components_with_class(layout, "review-feature-section-jump-evidence")[0]

    assert _props(dip_section).get("open") is True
    assert _props(jump_section).get("open") is False


def test_all_features_plain_list_contains_metadata_and_stats_rows() -> None:
    payload = {
        "candidate_id": "618475536448",
        "asas_sn_id": "618475536448",
        "dipper_score": 18.6,
        "dip_significant": True,
        "jump_count": 0,
        "gaia_var_flag": False,
        "tmass_j": 11.5,
        "tmass_j_err": 0.02,
        "ra_deg": 12.34,
        "period_source_periods": "asassn_var:1.2345",
        "derived_harmonics_r32": 0.77,
        "char_status_yso": "ok",
        "sed_model_fit_checked": True,
        "ztf_lc_n_det": 8,
        "ms_feature_status": "ok",
        "mystery_payload_field": "present",
        "payload_json": '{"too": "large"}',
        "lc_stats": '{"stats_amplitude": 0.1}',
    }
    grouped = review_app.extract_review_metadata_grouped(payload, round_sigfigs=True)
    feature_rows = review_app.extract_review_metadata_feature_rows(payload, round_sigfigs=True)
    layout = review_app._render_metadata_review_layout(
        payload,
        grouped,
        [("stats_cadence_mean_dt_days", "2.83856")],
        feature_rows,
    )
    all_features = _components_with_class(layout, "all-features-details")[0]
    copy_buttons = _components_with_class(all_features, "all-features-copy-btn")
    raw_lines = _components_with_class(all_features, "all-features-line")

    assert _component_by_id(layout, "all-features-table") is None
    assert _components_by_type(layout, "DataTable") == []
    assert _props(all_features).get("open") is False
    assert copy_buttons
    copy_text = copy_buttons[0].to_plotly_json()["props"]["data-copy-text"]
    for expected in (
        "asas_sn_id",
        "dip_significant",
        "jump_count",
        "gaia_var_flag",
        "tmass_j",
        "stats_cadence_mean_dt_days",
        "candidate_id",
        "ra_deg",
        "period_source_periods",
        "derived_harmonics_r32",
        "char_status_yso",
        "sed_model_fit_checked",
        "ztf_lc_n_det",
        "ms_feature_status",
        "tmass_j_err",
        "mystery_payload_field",
    ):
        assert expected in copy_text
    assert "Other Payload Fields / Record & Run Context / Candidate ID / candidate_id = 618475536448" in copy_text
    assert "Other Payload Fields / Coordinates & Gaia / RA Deg / ra_deg = 12.34" in copy_text
    assert "Other Payload Fields / Period & Filter Flags / Period Source Periods / period_source_periods = asassn_var:1.2345" in copy_text
    assert "Other Payload Fields / Derived Feature Extras / Derived Harmonics R32 / derived_harmonics_r32 = 0.77" in copy_text
    assert "Other Payload Fields / Enrichment Stage Status / Char Status YSO / char_status_yso = ok" in copy_text
    assert "Other Payload Fields / SED Pipeline Status / Sed Model Fit Checked / sed_model_fit_checked = True" in copy_text
    assert "Other Payload Fields / External LC Coverage / ZTF LC N Det / ztf_lc_n_det = 8" in copy_text
    assert "Other Payload Fields / Multi-Survey Features / Ms Feature Status / ms_feature_status = ok" in copy_text
    assert "Other Payload Fields / Photometric Error Columns / Tmass J Err / tmass_j_err = 0.02" in copy_text
    assert "Other Payload Fields / Miscellaneous / Mystery Payload Field / mystery_payload_field = present" in copy_text
    subsection_titles = {
        str(getattr(component, "children", ""))
        for component in _components_with_class(all_features, "all-features-subsection-title")
    }
    for expected in (
        "Record & Run Context (1)",
        "Coordinates & Gaia (1)",
        "Period & Filter Flags (1)",
        "Derived Feature Extras (1)",
        "Enrichment Stage Status (1)",
        "SED Pipeline Status (1)",
        "External LC Coverage (1)",
        "Multi-Survey Features (1)",
        "Photometric Error Columns (1)",
        "Miscellaneous (1)",
    ):
        assert expected in subsection_titles
    assert "payload_json" not in copy_text
    assert "lc_stats" not in copy_text
    assert raw_lines


def test_layout_exposes_phase_time_toggle() -> None:
    control = _component_by_id(app.layout, "phase-panel-mode")

    assert control is not None
    assert getattr(control, "value", None) == "fold"
    values = {str(option.get("value")) for option in getattr(control, "options", [])}
    assert values == {"fold", "time"}


def test_taxonomy_subtype_render_event_does_not_select_brightening_detail(monkeypatch) -> None:
    trigger = {
        "type": "taxonomy-option-btn",
        "menu": "morphology_secondary",
        "value": "single_brightening",
    }
    monkeypatch.setattr(
        review_app,
        "callback_context",
        types.SimpleNamespace(triggered_id=trigger, inputs_list=[[{"id": trigger}]]),
    )

    out = review_app.click_taxonomy_option(
        [0],
        {"morphology_primary": "brightening_event"},
        "morphology_secondary",
    )

    assert out == (
        review_app.no_update,
        review_app.no_update,
        review_app.no_update,
        review_app.no_update,
    )


def test_taxonomy_subtype_positive_click_selects_brightening_detail(monkeypatch) -> None:
    trigger = {
        "type": "taxonomy-option-btn",
        "menu": "morphology_secondary",
        "value": "single_brightening",
    }
    monkeypatch.setattr(
        review_app,
        "callback_context",
        types.SimpleNamespace(triggered_id=trigger, inputs_list=[[{"id": trigger}]]),
    )

    selection, active_menu, submenu, note = review_app.click_taxonomy_option(
        [1],
        {"morphology_primary": "brightening_event"},
        "morphology_secondary",
    )

    assert selection["morphology_secondary"] == "single_brightening"
    assert selection["morphology_secondary_list"] == ["single_brightening"]
    assert json.loads(selection["morphology_secondary_json"]) == ["single_brightening"]
    assert active_menu == "morphology_secondary"
    assert submenu == "brightening_event"
    assert "single brightening" in note


def test_auto_period_on_navigate_queues_harmonic_check_with_stored_period(monkeypatch) -> None:
    monkeypatch.setattr(
        review_app,
        "_candidate_context",
        lambda _candidate_id: ({"candidate_id": "cand-1", "period_consensus_days": 8.0}, None, None),
    )

    result, label, manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        10.0,
        {},
        {"nonce": 4},
    )

    assert result["pending"] is True
    assert result["source"] == "Auto harmonic check"
    assert result["search_method"] == "alias_check"
    assert result["base_period"] == 8.0
    assert label == "Auto harmonic check: checking aliases..."
    assert manual_period is None
    assert cache_update is review_app.no_update
    assert request["candidate_id"] == "cand-1"
    assert request["method"] == "alias_check"
    assert request["base_period"] == 8.0
    assert request["nonce"] == 5


def test_auto_period_on_navigate_queues_fallback_pdm_without_stored_period(monkeypatch) -> None:
    monkeypatch.setattr(
        review_app,
        "_candidate_context",
        lambda _candidate_id: ({"candidate_id": "cand-1"}, None, None),
    )

    result, label, manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        10.0,
        {},
        {"nonce": 4},
    )

    assert result["pending"] is True
    assert result["source"] == "Auto PDM"
    assert result["search_method"] == "pdm"
    assert result["reason"] == "no stored period"
    assert label == "No stored period; running auto PDM..."
    assert manual_period is None
    assert cache_update is review_app.no_update
    assert request["candidate_id"] == "cand-1"
    assert request["method"] == "pdm"
    assert request["reason"] == "no stored period"


def test_auto_period_cache_reuses_only_matching_bounds(monkeypatch) -> None:
    cached_result = {
        "candidate_id": "cand-1",
        "search_method": "alias_check",
        "method": "Harmonic check",
        "source": "Auto harmonic check",
        "best_period": 2.5,
        "min_period": 0.1,
        "max_period": 10.0,
        "base_period": 10.0,
        "auto": True,
    }
    key = review_app._period_cache_key("cand-1", "alias_check", 0.1, 10.0, 10.0)
    cache = {
        key: {
            "candidate_id": "cand-1",
            "method": "alias_check",
            "min_period": 0.1,
            "max_period": 10.0,
            "base_period": 10.0,
            "result": cached_result,
            "label": "Auto harmonic check: P=2.50000 d",
        }
    }

    monkeypatch.setattr(
        review_app,
        "_candidate_context",
        lambda _candidate_id: ({"candidate_id": "cand-1", "period_consensus_days": 10.0}, None, None),
    )
    result, label, _manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        10.0,
        cache,
        {"nonce": 1},
    )
    assert result == cached_result
    assert label == "Auto harmonic check: P=2.50000 d"
    assert cache_update is review_app.no_update
    assert request is review_app.no_update

    changed_result, changed_label, _manual_period, _cache_update, changed_request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        12.0,
        cache,
        {"nonce": 1},
    )
    assert changed_result["pending"] is True
    assert changed_label == "Auto harmonic check: checking aliases..."
    assert changed_request["method"] == "alias_check"
    assert changed_request["max_period"] == 12.0


def test_auto_period_cache_reuses_only_matching_base_period(monkeypatch) -> None:
    cached_result = {
        "candidate_id": "cand-1",
        "search_method": "alias_check",
        "method": "Harmonic check",
        "source": "Auto harmonic check",
        "best_period": 2.5,
        "min_period": 0.1,
        "max_period": 10.0,
        "base_period": 10.0,
        "auto": True,
    }
    key = review_app._period_cache_key("cand-1", "alias_check", 0.1, 10.0, 10.0)
    cache = {
        key: {
            "candidate_id": "cand-1",
            "method": "alias_check",
            "min_period": 0.1,
            "max_period": 10.0,
            "base_period": 10.0,
            "result": cached_result,
            "label": "Auto harmonic check: P=2.50000 d",
        }
    }
    monkeypatch.setattr(
        review_app,
        "_candidate_context",
        lambda _candidate_id: ({"candidate_id": "cand-1", "period_consensus_days": 8.0}, None, None),
    )

    result, label, _manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        10.0,
        cache,
        {"nonce": 1},
    )
    assert result["pending"] is True
    assert label == "Auto harmonic check: checking aliases..."
    assert cache_update is review_app.no_update
    assert request["method"] == "alias_check"
    assert request["base_period"] == 8.0


def test_run_auto_period_search_marks_result_and_cache_as_harmonic_check(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_candidate_context", lambda _candidate_id: ({"candidate_id": "cand-1"}, None, None))
    monkeypatch.setattr(
        review_app,
        "_run_period_search_for_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("navigation must not run full period search")),
    )

    def fake_check(_payload, *, min_period, max_period):
        assert min_period == 0.1
        assert max_period == 10.0
        return {
            "best_period": 1.234,
            "method": "Harmonic check",
            "base_period": 4.936,
        }, "Auto harmonic check: P=1.23400 d"

    monkeypatch.setattr(review_app, "_run_harmonic_check_for_payload", fake_check)

    result, label, cache = review_app.run_auto_period_search(
        {"nonce": 2, "candidate_id": "cand-1", "min_period": 0.1, "max_period": 10.0, "method": "alias_check", "base_period": 4.936},
        {},
    )

    key = review_app._period_cache_key("cand-1", "alias_check", 0.1, 10.0, 4.936)
    assert label == "Auto harmonic check: P=1.23400 d"
    assert result["auto"] is True
    assert result["source"] == "Auto harmonic check"
    assert result["search_method"] == "alias_check"
    assert result["candidate_id"] == "cand-1"
    assert cache[key]["method"] == "alias_check"
    assert cache[key]["result"]["best_period"] == 1.234


def test_run_auto_period_search_runs_fallback_pdm_without_stored_period(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_candidate_context", lambda _candidate_id: ({"candidate_id": "cand-1"}, None, None))
    monkeypatch.setattr(
        review_app,
        "_run_harmonic_check_for_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("missing-period fallback must run full search")),
    )

    def fake_search(_payload, *, min_period, max_period, method):
        assert min_period == 0.1
        assert max_period == 10.0
        assert method == "pdm"
        return {
            "best_period": 3.21,
            "method": "PDM",
        }, "PDM: P=3.21000 d"

    monkeypatch.setattr(review_app, "_run_period_search_for_payload", fake_search)

    result, label, cache = review_app.run_auto_period_search(
        {"nonce": 2, "candidate_id": "cand-1", "min_period": 0.1, "max_period": 10.0, "method": "pdm", "reason": "no stored period"},
        {},
    )

    key = review_app._period_cache_key("cand-1", "pdm", 0.1, 10.0)
    assert label == "Auto PDM: P=3.21000 d"
    assert result["auto"] is True
    assert result["source"] == "Auto PDM"
    assert result["search_method"] == "pdm"
    assert result["candidate_id"] == "cand-1"
    assert result["reason"] == "no stored period"
    assert cache[key]["method"] == "pdm"
    assert cache[key]["result"]["best_period"] == 3.21


def test_plot_render_request_manual_overrides_pending_harmonic_check() -> None:
    pending_result = {
        "pending": True,
        "candidate_id": "cand-1",
        "search_method": "alias_check",
        "method": "Harmonic check",
        "source": "Auto harmonic check",
        "min_period": 0.1,
        "max_period": 10.0,
        "base_period": 8.0,
    }

    request = review_app.queue_plot_render_request(
        0,
        "cand-1",
        "native",
        ["phase"],
        [],
        "Diagnostics",
        0.3,
        "black",
        1,
        0,
        0.5,
        ["g", "V"],
        ["yes"],
        10.0,
        pending_result,
        0.1,
        10.0,
        3.25,
        "mag",
        "fold",
        "asassn",
        {"nonce": 9},
    )

    state = request["state"]
    assert state["override_period"] == 3.25
    assert state["override_period_source"] == "manual/search"
    assert state["phase_period_pending"] is False
    assert state["suppress_catalog_phase_period"] is False


def test_plot_render_request_keeps_catalog_period_while_harmonic_check_pending() -> None:
    pending_result = {
        "pending": True,
        "candidate_id": "cand-1",
        "search_method": "alias_check",
        "method": "Harmonic check",
        "source": "Auto harmonic check",
        "min_period": 0.1,
        "max_period": 10.0,
        "base_period": 8.0,
    }

    request = review_app.queue_plot_render_request(
        0,
        "cand-1",
        "native",
        ["phase"],
        [],
        "Diagnostics",
        0.3,
        "black",
        1,
        0,
        0.5,
        ["g", "V"],
        ["yes"],
        10.0,
        pending_result,
        0.1,
        10.0,
        None,
        "mag",
        "fold",
        "asassn",
        {"nonce": 9},
    )

    state = request["state"]
    assert state["override_period"] is None
    assert state["phase_period_pending"] is True
    assert state["phase_period_pending_source"] == "Auto harmonic check"
    assert state["suppress_catalog_phase_period"] is False


def test_plot_render_request_labels_pending_fallback_pdm() -> None:
    pending_result = {
        "pending": True,
        "candidate_id": "cand-1",
        "search_method": "pdm",
        "method": "PDM",
        "source": "Auto PDM",
        "min_period": 0.1,
        "max_period": 10.0,
        "reason": "no stored period",
    }

    request = review_app.queue_plot_render_request(
        0,
        "cand-1",
        "native",
        ["phase"],
        [],
        "Diagnostics",
        0.3,
        "black",
        1,
        0,
        0.5,
        ["g", "V"],
        ["yes"],
        10.0,
        pending_result,
        0.1,
        10.0,
        None,
        "mag",
        "fold",
        "asassn",
        {"nonce": 9},
    )

    state = request["state"]
    assert state["override_period"] is None
    assert state["phase_period_pending"] is True
    assert state["phase_period_pending_source"] == "Auto PDM"
    assert state["suppress_catalog_phase_period"] is False


def test_plot_render_request_keeps_catalog_period_for_stale_auto_result() -> None:
    stale_result = {
        "candidate_id": "old-cand",
        "search_method": "alias_check",
        "method": "Harmonic check",
        "source": "Auto harmonic check",
        "best_period": 9.0,
        "min_period": 0.1,
        "max_period": 10.0,
        "base_period": 9.0,
    }

    request = review_app.queue_plot_render_request(
        0,
        "cand-1",
        "native",
        ["phase"],
        [],
        "Diagnostics",
        0.3,
        "black",
        1,
        0,
        0.5,
        ["g", "V"],
        ["yes"],
        10.0,
        stale_result,
        0.1,
        10.0,
        None,
        "mag",
        "fold",
        "asassn",
        {"nonce": 9},
    )

    state = request["state"]
    assert state["override_period"] is None
    assert state["phase_period_pending"] is False
    assert state["suppress_catalog_phase_period"] is False


def test_eda_table_uses_server_side_sorting_filtering_and_paging() -> None:
    table = _component_by_id(app.layout, "eda-candidate-table")

    assert table is not None
    assert getattr(table, "page_action", None) == "custom"
    assert getattr(table, "page_current", None) == 0
    assert getattr(table, "page_size", None) == 12
    assert getattr(table, "page_count", None) == 0
    assert getattr(table, "sort_action", None) == "custom"
    assert getattr(table, "sort_mode", None) == "multi"
    assert getattr(table, "sort_by", None) == []
    assert getattr(table, "filter_action", None) == "custom"
    assert getattr(table, "filter_query", None) == ""
    assert getattr(table, "hidden_columns", None) in (None, [])
    assert all(column.get("id") != "candidate_key" for column in table.columns)


def test_initial_eda_metric_sync_defers_db_load(monkeypatch) -> None:
    def fail_load():
        raise AssertionError("EDA frame should not load during initial hydration")

    monkeypatch.setattr(review_app, "_current_eda_frame", fail_load)

    result = review_app.sync_eda_metric_controls(
        {"candidate_ids": ["A"], "queue_size": 1},
        "scope",
        None,
        None,
        "open",
        0,
        None,
        None,
        None,
        None,
    )

    assert result == ([], None, [], None, [], None, [], None)


def test_initial_eda_panel_returns_placeholder_before_startup(monkeypatch) -> None:
    def fail_load():
        raise AssertionError("EDA frame should not load during initial hydration")

    monkeypatch.setattr(review_app, "_current_eda_frame", fail_load)

    status, fig, rows, page_count, style, status_base, trace_idx = review_app.update_eda_panel(
        {"candidate_ids": ["A"], "queue_size": 1},
        None,
        None,
        None,
        None,
        [],
        [],
        [],
        "black",
        None,
        "scope",
        "open",
        0,
        0,
        12,
        [],
        "",
        "A",
    )

    assert "startup" in status.lower()
    assert "startup" in fig.layout.annotations[0].text.lower()
    assert rows == []
    assert page_count == 0
    assert style == []
    assert status_base == status
    assert trace_idx is None


def test_eda_plot_selection_filters_table_only_when_enabled(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C"],
            "dipper_score": [1.0, 2.0, 3.0],
            "interest_score": [4.0, 5.0, 6.0],
        }
    )
    queue_data = {"candidate_ids": ["A", "B", "C"], "queue_size": 3}
    monkeypatch.setattr(review_app, "_current_eda_frame", lambda: frame)

    _status, fig, rows, page_count, _style, _status_base, _trace_idx = review_app.update_eda_panel(
        queue_data,
        "dipper_score",
        "interest_score",
        None,
        None,
        [],
        [],
        ["B", "C"],
        "black",
        0,
        "scope",
        "open",
        1,
        0,
        12,
        [],
        "",
        "A",
    )

    assert [row["candidate_id"] for row in rows] == ["A", "B", "C"]
    assert page_count == 1
    assert fig.layout.dragmode == "zoom"

    graph_fig = review_app.eda_scatter_figure(
        frame,
        x_metric="dipper_score",
        y_metric="interest_score",
        selected_candidate_id="A",
    )
    selected_ids = review_app.capture_eda_selection(
        {"points": [{"curveNumber": 0, "pointNumber": 1}, {"curveNumber": 0, "pointNumber": 2}]},
        graph_fig.to_dict(),
        ["table"],
    )

    assert selected_ids == ["B", "C"]

    status, fig, rows, page_count, _style, _status_base, _trace_idx = review_app.update_eda_panel(
        queue_data,
        "dipper_score",
        "interest_score",
        None,
        None,
        [],
        ["table"],
        selected_ids,
        "black",
        0,
        "scope",
        "open",
        1,
        0,
        12,
        [],
        "",
        "A",
    )

    assert [row["candidate_id"] for row in rows] == ["B", "C"]
    assert page_count == 1
    assert "Selected: 2" in status
    assert fig.layout.dragmode == "select"

    status, _fig, rows, page_count, _style, _status_base, _trace_idx = review_app.update_eda_panel(
        queue_data,
        "dipper_score",
        "interest_score",
        None,
        None,
        [],
        ["table"],
        ["B", "C"],
        "black",
        0,
        "scope",
        "open",
        1,
        0,
        12,
        [],
        "",
        "A",
    )

    assert [row["candidate_id"] for row in rows] == ["B", "C"]
    assert page_count == 1
    assert "Selected: 2" in status

    status, _fig, rows, page_count, _style, _status_base, _trace_idx = review_app.update_eda_panel(
        queue_data,
        "dipper_score",
        "interest_score",
        None,
        None,
        [],
        ["table"],
        [],
        "black",
        0,
        "scope",
        "open",
        1,
        0,
        12,
        [],
        "",
        "A",
    )

    assert [row["candidate_id"] for row in rows] == ["A", "B", "C"]
    assert page_count == 1
    assert "Selected:" not in status


def test_eda_panel_sorts_filters_and_pages_table_server_side(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C", "D"],
            "dipper_score": [1.0, 4.0, 3.0, 2.0],
            "interest_score": [4.0, 5.0, 6.0, 7.0],
        }
    )
    queue_data = {"candidate_ids": ["A", "B", "C", "D"], "queue_size": 4}
    monkeypatch.setattr(review_app, "_current_eda_frame", lambda: frame)

    status, _fig, rows, page_count, _style, _status_base, _trace_idx = review_app.update_eda_panel(
        queue_data,
        "dipper_score",
        "interest_score",
        None,
        None,
        [],
        [],
        [],
        "black",
        0,
        "scope",
        "open",
        1,
        0,
        1,
        [{"column_id": "dipper_score", "direction": "desc"}],
        "{dipper_score} > 0",
        "A",
    )

    assert [row["candidate_id"] for row in rows] == ["B"]
    assert page_count == 4
    assert "Table filtered: 4" in status


def test_eda_navigation_callback_returns_patch_without_large_state(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "dipper_score": [1.0, 2.0],
            "interest_score": [4.0, 5.0],
        }
    )
    monkeypatch.setattr(review_app, "_current_eda_frame", lambda: frame)

    status, patch, style = review_app.update_eda_current_candidate(
        "B",
        "dipper_score",
        "interest_score",
        [],
        "black",
        "Queue rows: 2/2 | Plotted: 2",
        1,
        {"candidate_ids": ["A", "B"], "queue_size": 2},
        "open",
        1,
    )

    assert status.endswith("Current: B")
    assert isinstance(patch, dash.Patch)
    assert style[0]["if"] == {"filter_query": '{candidate_id} = "B"'}

    callback = None
    for meta in review_app.app.callback_map.values():
        outputs = meta.get("output")
        output_text = str(outputs)
        if "eda-custom-graph.figure" in output_text and "eda-candidate-table.style_data_conditional" in output_text:
            inputs = {item["id"] for item in meta.get("inputs", [])}
            if "current-candidate-id" in inputs:
                callback = meta
                break

    assert callback is not None
    state_props = {(item["id"], item["property"]) for item in callback.get("state", [])}
    assert ("eda-custom-graph", "figure") not in state_props
    assert ("eda-candidate-table", "data") not in state_props


def test_full_eda_rebuild_callback_does_not_listen_to_navigation() -> None:
    callback = None
    for meta in review_app.app.callback_map.values():
        output_text = str(meta.get("output"))
        if "eda-candidate-table.data" in output_text:
            callback = meta
            break

    assert callback is not None
    input_ids = {item["id"] for item in callback.get("inputs", [])}
    state_ids = {item["id"] for item in callback.get("state", [])}
    assert "current-index" not in input_ids
    assert "last-candidate-saved" not in input_ids
    assert "current-candidate-id" in state_ids


def test_eda_splitter_reuses_metadata_splitter_style() -> None:
    splitter = _component_by_id(app.layout, "eda-splitter")

    assert splitter is not None
    assert "panel-splitter-vertical" in str(getattr(splitter, "className", ""))


def test_eda_panel_drag_can_collapse_to_zero_width() -> None:
    source = inspect.getsource(review_app)

    assert re.search(r"\.eda-panel \{[^}]*min-width: 0;", app.index_string, re.S)
    assert re.search(
        r"var storageKey = 'malca\.review\.eda_panel\.width\.v1';[^}]*var minWidth = 0;",
        source,
        re.S,
    )
    assert "if (numeric < minWidth) numeric = minWidth;" in source


def test_eda_panel_state_callback_supports_collapse_restore_and_wide(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "callback_context", types.SimpleNamespace(triggered_id="eda-collapse-btn"))
    panel_class, splitter_class, _wide_text, _wide_title, toggle_text, _toggle_title, state = review_app.toggle_eda_panel(1, 0, 0, "open")
    assert panel_class == "eda-panel is-collapsed"
    assert splitter_class == "eda-splitter panel-splitter-vertical collapsed"
    assert toggle_text == "EDA"
    assert state == "collapsed"

    monkeypatch.setattr(review_app, "callback_context", types.SimpleNamespace(triggered_id="eda-panel-toggle"))
    panel_class, splitter_class, _wide_text, _wide_title, _toggle_text, _toggle_title, state = review_app.toggle_eda_panel(0, 0, 1, "collapsed")
    assert panel_class == "eda-panel"
    assert splitter_class == "eda-splitter panel-splitter-vertical"
    assert state == "open"

    monkeypatch.setattr(review_app, "callback_context", types.SimpleNamespace(triggered_id="eda-expand-btn"))
    panel_class, splitter_class, wide_text, _wide_title, _toggle_text, _toggle_title, state = review_app.toggle_eda_panel(0, 1, 0, "open")
    assert panel_class == "eda-panel is-expanded"
    assert splitter_class == "eda-splitter panel-splitter-vertical"
    assert wide_text == "Restore"
    assert state == "expanded"


def test_external_source_selector_exposes_tess() -> None:
    values = {str(option.get("value")) for option in EXTERNAL_SOURCE_VIEW_OPTIONS}

    assert {
        "tess",
        "kepler",
        "aavso",
        "ogle",
        "stripe82",
        "allwise_mep",
        "vvvx_virac",
    }.issubset(values)


def test_review_plot_dir_infers_run_bundle_from_db_path(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "march18_bundle"
    (run_dir / "review").mkdir(parents=True)
    (run_dir / "bundle_assets" / "lightcurves").mkdir(parents=True)
    db_path = run_dir / "review" / "review.db"
    db_path.write_bytes(b"")

    monkeypatch.setattr(review_app, "PLOT_DIR", None)
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    assert review_app._review_plot_dir_for_context() == run_dir / "plots"


def test_effective_local_lc_path_uses_inferred_bundle_without_plot_dir(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "march18_bundle"
    lc_dir = run_dir / "bundle_assets" / "lightcurves"
    (run_dir / "review").mkdir(parents=True)
    lc_dir.mkdir(parents=True)
    db_path = run_dir / "review" / "review.db"
    db_path.write_bytes(b"")
    lc_path = lc_dir / "C1.dat3"
    lc_path.write_text("", encoding="ascii")

    monkeypatch.setattr(review_app, "PLOT_DIR", None)
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    resolved = review_app._effective_local_lc_path(
        {"candidate_id": "C1", "asas_sn_id": "C1", "lc_path": "/old/root/C1.dat3"},
        stored_lc_path="/old/root/C1.dat3",
        source_path="/old/root",
    )

    assert resolved == str(lc_path)


def test_plot_resolution_prefers_local_db_context_over_stale_payload_paths(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "march18_bundle"
    lc_dir = run_dir / "bundle_assets" / "lightcurves"
    plot_dir = run_dir / "plots"
    (run_dir / "review").mkdir(parents=True)
    lc_dir.mkdir(parents=True)
    plot_dir.mkdir()
    db_path = run_dir / "review" / "review.db"
    db_path.write_bytes(b"")
    lc_path = lc_dir / "C1.dat3"
    lc_path.write_text("", encoding="ascii")

    monkeypatch.setattr(review_app, "PLOT_DIR", None)
    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    payload = {
        "candidate_id": "C1",
        "asas_sn_id": "C1",
        "source_path": "/old/root",
        "path": "/old/root/C1.dat3",
        "lc_path": "/old/root/C1.dat3",
    }

    assert review_app._review_plot_dir_for_context(payload["source_path"]) == plot_dir
    assert review_app._plot_search_root_for_payload(payload) == plot_dir
    assert review_app._effective_local_lc_path(
        payload,
        stored_lc_path="/old/root/C1.dat3",
        source_path="/old/root",
    ) == str(lc_path)
    assert review_app._baseline_provenance_warning(
        payload,
        plot_dir=plot_dir,
        run_params={"baseline_func": "per_camera_median"},
        stored_lc_path="/old/root/C1.dat3",
        source_path="/old/root",
    ) is None


def test_queue_scope_for_migrated_path_falls_back_to_run_token(tmp_path) -> None:
    db_path = tmp_path / "review.db"
    old_run_dir = "/home/calder/code/malca/output/runs/march18_bundle"
    new_run_dir = tmp_path / "output_migrated" / "runs" / "march18_bundle"
    results_file = new_run_dir / "results" / "lc_events_vetted.parquet"
    results_file.parent.mkdir(parents=True)
    results_file.write_text("", encoding="ascii")

    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "asas_sn_id": "C1",
                        "source_path": old_run_dir,
                    }
                ]
            ),
        )
        exact_scope = {"source_paths": [str(new_run_dir)]}
        portable_scope = review_app._queue_scope_filter_kwargs(
            review_app._queue_scope_from_import_text(str(results_file))
        )

        assert count_queue(conn, filters=exact_scope) == 0
        assert portable_scope["source_paths"] == [str(new_run_dir)]
        assert portable_scope["source_path_fallback_like_any"] == ["march18_bundle"]
        assert count_queue(conn, filters=portable_scope) == 1


def test_review_db_for_plot_dir_prefers_populated_sibling_over_empty_review_db(tmp_path) -> None:
    run_dir = tmp_path / "march18_bundle"
    review_dir = run_dir / "review"
    (run_dir / "bundle_assets" / "lightcurves").mkdir(parents=True)
    review_dir.mkdir(parents=True)
    empty_db = review_dir / "review.db"
    empty_db.write_bytes(b"")
    populated_db = review_dir / "review.taxonomy_filled.db"

    with db_connect(populated_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "C1"}]),
        )

    assert review_app._review_db_for_plot_dir(str(run_dir)) == populated_db.resolve()


def test_resolve_db_cli_path_prefers_populated_appended_db_sibling(tmp_path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    typo_db = review_dir / "review.taxonomy_filled"
    populated_db = review_dir / "review.taxonomy_filled.db"

    with db_connect(typo_db):
        pass
    with db_connect(populated_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "C1"}]),
        )

    assert review_app._resolve_db_cli_path(str(typo_db)) == populated_db.resolve()


def test_closed_lazy_panels_skip_candidate_work(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("closed panel should not render")

    monkeypatch.setattr(review_app, "_candidate_context", fail)
    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", fail)
    monkeypatch.setattr(review_app, "_render_dustycult_result_panel", fail)
    monkeypatch.setattr(review_app, "_render_phoebe_result_panel", fail)
    monkeypatch.setattr(review_app, "control_defaults_for_candidate", fail)
    monkeypatch.setattr(review_app, "infer_period_days", fail)

    assert review_app.update_external_followup_panel("C1", "black", False) == (review_app.no_update, review_app.no_update)
    assert review_app.update_sed_panel("C1", "observed", "black", False) == (review_app.no_update, review_app.no_update)
    assert review_app.update_diagnostic_plots("C1", "black", {"ready": False}, False) == (review_app.no_update, review_app.no_update)
    assert review_app.update_dustycult_result_panel("C1", "black", 0, False) == (review_app.no_update, review_app.no_update)
    assert review_app.update_phoebe_result_panel("C1", "black", 0, False) == (review_app.no_update, review_app.no_update)
    assert review_app.update_phoebe_period_control("C1", False) == (review_app.no_update, review_app.no_update)

    dustycult_controls = review_app.update_dustycult_controls("C1", 0, False)
    assert dustycult_controls == tuple(
        [review_app.no_update] * len(review_app._DUSTYCULT_CONTROL_FIELDS) + [review_app.no_update]
    )


def test_open_lazy_panels_render_current_candidate(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_candidate_context", lambda *_args: ({"candidate_id": "C1"}, None, None))
    monkeypatch.setattr(review_app, "_render_external_followup", lambda *_args, **_kwargs: ["external"])
    monkeypatch.setattr(review_app, "_load_sed_figure_for_candidate", lambda *_args, **_kwargs: ({"data": [], "layout": {}}, [], []))
    monkeypatch.setattr(review_app, "_load_sed_source_status_for_candidate", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(review_app, "_render_sed_fetch_provenance", lambda *_args, **_kwargs: "provenance")
    monkeypatch.setattr(review_app, "_sed_status_text", lambda *_args, **_kwargs: "sed status")
    monkeypatch.setattr(review_app, "_diagnostic_background_signature", lambda *_args, **_kwargs: "sig")
    monkeypatch.setattr(review_app, "_get_cached_diagnostic_background", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", lambda *_args, **_kwargs: ["diag"])
    monkeypatch.setattr(review_app, "_render_dustycult_result_panel", lambda *_args, **_kwargs: ["dust"])
    monkeypatch.setattr(review_app, "_dustycult_config_status_text", lambda: "dust config")
    monkeypatch.setattr(review_app, "_render_phoebe_result_panel", lambda *_args, **_kwargs: ["phoebe"])
    monkeypatch.setattr(review_app, "_phoebe_config_status_text", lambda: "phoebe config")
    monkeypatch.setattr(review_app, "infer_period_days", lambda *_args, **_kwargs: (2.5, "test"))

    assert review_app.update_external_followup_panel("C1", "black", True) == (["external"], "Loaded external data for C1.")
    sed_children, sed_status = review_app.update_sed_panel("C1", "observed", "black", True)
    assert len(sed_children) == 2
    assert isinstance(sed_status, str)
    assert review_app.update_diagnostic_plots("C1", "black", {"ready": True, "signature": "sig"}, True) == (["diag"], "")
    assert review_app.update_dustycult_result_panel("C1", "black", 0, True) == (["dust"], "dust config")
    assert review_app.update_phoebe_result_panel("C1", "black", 0, True) == (["phoebe"], "phoebe config")
    assert review_app.update_phoebe_period_control("C1", True) == (2.5, "Using test period.")


def test_dustycult_publication_export_controls_are_present() -> None:
    ids = _component_ids_in_order(app.layout)

    assert "dustycult-export-download" in ids
    assert "mini-plot-export-download" in ids
    assert "dustycult-export-fit-btn" in ids
    assert "dustycult-export-occulter-btn" in ids


def test_vetting_filter_group_has_known_variable_and_dipper_contaminant_presets() -> None:
    ids = _component_ids_in_order(app.layout)
    texts = _component_text_in_order(app.layout)

    assert "vetting-known-variables-btn" in ids
    assert "vetting-dipper-contaminants-btn" in ids
    assert "Exclude Known Variables" in texts
    assert "Exclude Dipper Contaminants" in texts


def test_sidebar_filter_options_show_classification_labels_without_changing_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "gaia_var_class": "DSCT|GDOR|SXPHE",
                        "vsx_class": "EA/SD:",
                        "simbad_otype": "EB?",
                    }
                ]
            ),
        )

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    payload = review_app._load_sidebar_filter_payload(True, None)
    select_options = payload[len(review_app._TEXT_STATES):]
    options_by_filter = {
        filter_key: options
        for (_cid, filter_key), options in zip(review_app._SELECT_STATES, select_options)
    }

    gaia_option = options_by_filter["exclude_gaia_var_class"][0]
    vsx_option = options_by_filter["exclude_vsx_class"][0]
    simbad_option = options_by_filter["exclude_simbad_otype"][0]

    assert gaia_option["value"] == "DSCT|GDOR|SXPHE"
    assert "Delta Scuti" in gaia_option["label"]
    assert vsx_option["value"] == "EA/SD:"
    assert "candidate/uncertain" in vsx_option["label"]
    assert simbad_option["value"] == "EB?"
    assert simbad_option["label"] == "EB? - candidate/uncertain Eclipsing binary [SIMBAD]"


def test_external_followup_exposes_multi_survey_summary() -> None:
    panel = _render_external_followup(
        {
            "ms_feature_status": "ok",
            "ms_event_type": "dip",
            "ms_event_t0_jd": 2458500.0,
            "ms_ztf_gr_delta": 0.3,
            "ms_ztf_gr_event_pairs": 1,
            "ms_neowise_w1_delta": 1.0,
            "ms_tess_flux_frac_delta": -0.1,
            "ms_gaia_epoch_g_delta": 0.7,
        },
        "C1",
    )

    text: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        if isinstance(node, str):
            text.append(node)
            return
        if node is None or isinstance(node, (int, float, bool)):
            return
        walk(getattr(node, "children", None))

    walk(panel)
    rendered = "\n".join(text)
    assert "Multi-survey Features" in rendered
    assert "ZTF g-r delta" in rendered
    assert "TESS delta F/F" in rendered
