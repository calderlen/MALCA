from __future__ import annotations

import importlib.util
import inspect
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
