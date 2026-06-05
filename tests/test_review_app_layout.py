from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
import types


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
from malca.review.cutouts import CUTOUT_SURVEYS, DEFAULT_CUTOUT_SURVEY_KEY


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


def test_layout_embeds_eda_panel() -> None:
    ids = _component_ids_in_order(app.layout)

    assert "eda-panel-toggle" in ids
    assert "eda-splitter" in ids
    assert "eda-drag-handle" in ids
    assert "eda-panel" in ids
    assert "eda-panel-state" in ids
    assert "eda-collapse-btn" in ids
    assert "eda-expand-btn" in ids
    assert "eda-x-metric" in ids
    assert "eda-y-metric" in ids
    assert "eda-color-metric" in ids
    assert "eda-symbol-metric" in ids
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
    source_link = _component_by_id(cards, "cutout-source-link")
    status = _component_by_id(cards, "cutout-status")

    select_props = _props(survey_select)
    image_props = _props(image)
    link_props = _props(source_link)
    assert select_props["value"] == DEFAULT_CUTOUT_SURVEY_KEY
    assert select_props["disabled"] is False
    assert [option["label"] for option in select_props["options"]] == [survey.label for survey in CUTOUT_SURVEYS]
    assert "CDS%2FP%2FPanSTARRS%2FDR1%2Fcolor-i-r-g" in image_props["src"]
    assert "fov=0.03333333333" in image_props["src"]
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
    source_link = _component_by_id(cards, "cutout-source-link")
    status = _component_by_id(cards, "cutout-status")

    assert _props(survey_select)["disabled"] is True
    assert _props(image)["src"] == ""
    assert _props(source_link)["href"] == "#"
    assert "RA/Dec" in str(_props(status).get("children"))


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


def test_auto_period_on_navigate_queues_pdm_even_with_external_period(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_has_external_period", lambda _payload: True)

    result, label, manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        10.0,
        {},
        {"nonce": 4},
    )

    assert result["pending"] is True
    assert result["source"] == "Auto PDM"
    assert label == "Auto PDM: searching..."
    assert manual_period is None
    assert cache_update is review_app.no_update
    assert request["candidate_id"] == "cand-1"
    assert request["method"] == "pdm"
    assert request["nonce"] == 5


def test_auto_period_cache_reuses_only_matching_bounds() -> None:
    cached_result = {
        "candidate_id": "cand-1",
        "search_method": "pdm",
        "method": "PDM",
        "source": "Auto PDM",
        "best_period": 2.5,
        "min_period": 0.1,
        "max_period": 10.0,
        "auto": True,
    }
    key = review_app._period_cache_key("cand-1", "pdm", 0.1, 10.0)
    cache = {
        key: {
            "candidate_id": "cand-1",
            "method": "pdm",
            "min_period": 0.1,
            "max_period": 10.0,
            "result": cached_result,
            "label": "Auto PDM: P=2.50000 d",
        }
    }

    result, label, _manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "cand-1",
        0.1,
        10.0,
        cache,
        {"nonce": 1},
    )
    assert result == cached_result
    assert label == "Auto PDM: P=2.50000 d"
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
    assert changed_label == "Auto PDM: searching..."
    assert changed_request["method"] == "pdm"
    assert changed_request["max_period"] == 12.0


def test_run_auto_period_search_marks_result_and_cache_as_auto_pdm(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_candidate_context", lambda _candidate_id: ({"candidate_id": "cand-1"}, None, None))

    def fake_search(_payload, *, min_period, max_period, method):
        assert method == "pdm"
        assert min_period == 0.1
        assert max_period == 10.0
        return {"best_period": 1.234, "method": "PDM"}, "PDM: P=1.23400 d"

    monkeypatch.setattr(review_app, "_run_period_search_for_payload", fake_search)

    result, label, cache = review_app.run_auto_period_search(
        {"nonce": 2, "candidate_id": "cand-1", "min_period": 0.1, "max_period": 10.0, "method": "pdm"},
        {},
    )

    key = review_app._period_cache_key("cand-1", "pdm", 0.1, 10.0)
    assert label == "Auto PDM: P=1.23400 d"
    assert result["auto"] is True
    assert result["source"] == "Auto PDM"
    assert result["candidate_id"] == "cand-1"
    assert cache[key]["method"] == "pdm"
    assert cache[key]["result"]["best_period"] == 1.234


def test_plot_render_request_manual_overrides_pending_auto_pdm() -> None:
    pending_result = {
        "pending": True,
        "candidate_id": "cand-1",
        "search_method": "pdm",
        "method": "PDM",
        "source": "Auto PDM",
        "min_period": 0.1,
        "max_period": 10.0,
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


def test_plot_render_request_suppresses_catalog_until_matching_pdm_result() -> None:
    stale_result = {
        "candidate_id": "old-cand",
        "search_method": "pdm",
        "method": "PDM",
        "source": "Auto PDM",
        "best_period": 9.0,
        "min_period": 0.1,
        "max_period": 10.0,
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
    assert state["phase_period_pending"] is True
    assert state["suppress_catalog_phase_period"] is True


def test_eda_table_has_native_sorting_and_filtering() -> None:
    table = _component_by_id(app.layout, "eda-candidate-table")

    assert table is not None
    assert getattr(table, "sort_action", None) == "native"
    assert getattr(table, "sort_mode", None) == "multi"
    assert getattr(table, "filter_action", None) == "native"
    assert getattr(table, "hidden_columns", None) in (None, [])
    assert all(column.get("id") != "candidate_key" for column in table.columns)


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

    assert "tess" in values


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


def test_vetting_filter_group_has_definite_known_type_preset() -> None:
    ids = _component_ids_in_order(app.layout)

    assert "vetting-known-types-btn" in ids
    assert "vetting-definite-known-types-btn" in ids


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
