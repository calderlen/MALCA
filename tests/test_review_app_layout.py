from __future__ import annotations

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


def test_external_and_diagnostic_panels_are_above_long_metadata() -> None:
    ids = _component_ids_in_order(app.layout)

    external_idx = ids.index("external-followup-details")
    sed_idx = ids.index("sed-details")
    dustycult_idx = ids.index("dustycult-details")
    sed_button_idx = ids.index("rerun-stage-sed-photometry-btn")
    multi_survey_button_idx = ids.index("rerun-stage-multi-survey-btn")
    diagnostic_idx = ids.index("diagnostic-plots-details")
    metadata_idx = ids.index("candidate-info-grid")
    run_config_idx = ids.index("run-config-details")

    assert external_idx < metadata_idx
    assert external_idx < sed_idx < metadata_idx
    assert sed_idx < dustycult_idx < diagnostic_idx < metadata_idx
    assert sed_button_idx < metadata_idx
    assert multi_survey_button_idx < metadata_idx
    assert metadata_idx < run_config_idx


def test_external_source_selector_exposes_tess() -> None:
    values = {str(option.get("value")) for option in EXTERNAL_SOURCE_VIEW_OPTIONS}

    assert "tess" in values


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
