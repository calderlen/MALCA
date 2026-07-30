from __future__ import annotations

from malca.review.tui_filters import FilterEditor
from malca.review.tui_photometry import (
    DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES,
    TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES,
    TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES,
)
from malca.review.tui_service import NumericRange, QueueFilterSpec


_HIERARCHICAL_PROBABILITY_FIELDS = (
    "prob_hierarchical_artifact_or_nonvariable",
    "prob_usable_astrophysical_variable",
    "prob_dipper_dimming",
    "prob_eb_geometric_periodic",
    "prob_long_timescale_variable",
    "prob_brightening_transient",
    "prob_other_structured_variable",
    "prob_quasi_periodic_hierarchical",
    "prob_microlensing_hierarchical",
    "prob_long_period_variable_hierarchical",
    "prob_long_term_variable_hierarchical",
    "prob_recurrent_dipper_hierarchical",
    "prob_single_dipper_hierarchical",
    "prob_quasi_periodic_given_usable",
    "prob_microlensing_given_brightening",
    "prob_long_period_variable_given_long_timescale",
    "prob_long_term_variable_given_long_timescale",
    "prob_recurrent_given_dipper",
    "prob_single_given_dipper",
)


def _move_to(editor: FilterEditor, key: str) -> None:
    while editor.active_key != key:
        editor.move(1)


def test_filter_editor_cycles_curated_presets_and_resets() -> None:
    editor = FilterEditor(QueueFilterSpec.default())
    _move_to(editor, "high_pm")
    assert editor.rows()[editor.cursor].value == "exclude"
    editor.cycle(1)
    assert editor.spec.high_pm == "any"

    _move_to(editor, "prob_dipper_dimming")

    editor.cycle(2)
    assert editor.spec.prob_dipper_dimming == NumericRange(0.5)
    assert editor.rows()[editor.cursor].value == ">= 0.50"

    _move_to(editor, "sort_by")
    while editor.spec.sort_by != "prob_dipper_dimming":
        editor.cycle(1)
    assert editor.spec.sort_by == "prob_dipper_dimming"
    _move_to(editor, "sort_desc")
    editor.cycle(1)
    assert editor.spec.sort_desc is True

    editor.reset()
    assert editor.spec == QueueFilterSpec.default()
    assert editor.cursor == 0


def test_external_photometry_master_and_individual_sources_are_configurable() -> None:
    editor = FilterEditor(QueueFilterSpec.default())

    _move_to(editor, "show_external_lightcurves")
    assert editor.rows()[editor.cursor].value == "enabled"
    editor.cycle(1)
    assert editor.spec.show_external_lightcurves is False
    assert editor.rows()[editor.cursor].value == "disabled"

    _move_to(editor, "external_source_ztf")
    assert editor.external_source_enabled("ztf") is True
    editor.cycle(1)
    assert editor.external_source_enabled("ztf") is False
    assert editor.rows()[editor.cursor].value == "disabled"

    _move_to(editor, "external_source_gaia_epoch")
    assert editor.external_source_enabled("gaia_epoch") is False
    assert editor.toggle_external_source("gaia_epoch") is True
    assert editor.rows()[editor.cursor].value == "enabled"
    assert editor.spec.external_lightcurve_sources == tuple(
        source
        for source in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES
        if source
        in (
            set(DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES)
            - {"ztf"}
            | {"gaia_epoch"}
        )
    )


def test_external_photometry_availability_is_independent_from_display() -> None:
    editor = FilterEditor(
        QueueFilterSpec.default(),
        external_photometry_counts={"neowise": 8197},
    )
    _move_to(editor, "external_availability_neowise")

    assert editor.rows()[editor.cursor].value == "any · n=8,197"
    editor.cycle(1)
    assert editor.external_availability_mode("neowise") == "required"
    assert editor.spec.required_external_photometry_sources == ("neowise",)
    assert editor.external_source_enabled("neowise") is True

    editor.cycle(1)
    assert editor.external_availability_mode("neowise") == "absent"
    assert editor.spec.required_external_photometry_sources == ()
    assert editor.spec.excluded_external_photometry_sources == ("neowise",)

    editor.cycle(1)
    assert editor.external_availability_mode("neowise") == "any"

    row_keys = {row.key for row in editor.rows()}
    assert {
        f"external_availability_{source}"
        for source in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES
    }.issubset(row_keys)
    assert "external_availability_tess" in row_keys
    assert "external_availability_kepler" in row_keys
    assert "external_source_tess" not in row_keys
    assert "external_source_kepler" not in row_keys


def test_all_hierarchical_probabilities_offer_less_than_and_greater_than_presets() -> None:
    for field_name in _HIERARCHICAL_PROBABILITY_FIELDS:
        greater_editor = FilterEditor(QueueFilterSpec.default())
        _move_to(greater_editor, field_name)
        greater_editor.cycle(1)
        assert getattr(greater_editor.spec, field_name) == NumericRange(minimum=0.25)
        assert greater_editor.rows()[greater_editor.cursor].value == ">= 0.25"

        less_editor = FilterEditor(QueueFilterSpec.default())
        _move_to(less_editor, field_name)
        less_editor.cycle(5)
        assert getattr(less_editor.spec, field_name) == NumericRange(maximum=0.25)
        assert less_editor.rows()[less_editor.cursor].value == "<= 0.25"


def test_all_hierarchical_probabilities_are_sortable_in_both_directions() -> None:
    for field_name in _HIERARCHICAL_PROBABILITY_FIELDS:
        editor = FilterEditor(QueueFilterSpec.default())
        _move_to(editor, "sort_by")
        while editor.spec.sort_by != field_name:
            editor.cycle(1)
        assert editor.spec.sort_by == field_name

        _move_to(editor, "sort_desc")
        assert editor.rows()[editor.cursor].value == "ascending"
        editor.cycle(1)
        assert editor.rows()[editor.cursor].value == "descending"


def test_filter_editor_taxonomy_is_multi_select_and_clearable() -> None:
    editor = FilterEditor(QueueFilterSpec.default())
    _move_to(editor, "morphology_primary")

    assert editor.toggle_taxonomy("dimming_event") is True
    assert editor.toggle_taxonomy("periodic") is True
    assert editor.spec.morphology_primary == ("dimming_event", "periodic")
    assert "+1" in editor.rows()[editor.cursor].value

    assert editor.toggle_taxonomy("dimming_event") is False
    assert editor.spec.morphology_primary == ("periodic",)
    editor.clear_taxonomy()
    assert editor.spec.morphology_primary == ()


def test_filter_editor_catalog_types_are_explicit_keep_exclude_choices() -> None:
    editor = FilterEditor(QueueFilterSpec.default())
    _move_to(editor, "catalog_vsx")

    assert editor.rows()[editor.cursor].value == "all kept"
    assert editor.catalog_type_kept("vsx", "EA") is True

    assert editor.toggle_catalog_type("vsx", "EA") is False
    assert editor.catalog_type_kept("vsx", "EA") is False
    assert editor.spec.excluded_vsx_types == ("EA",)
    assert editor.rows()[editor.cursor].value == "1 excluded"

    assert editor.set_catalog_type_kept("asassn", "YSO", False) is False
    assert editor.spec.excluded_asassn_var_types == ("YSO",)
    assert editor.set_catalog_type_kept("vsx", "EA", True) is True
    assert editor.spec.excluded_vsx_types == ()

    _move_to(editor, "catalog_asassn")
    assert editor.rows()[editor.cursor].value == "1 excluded"
    editor.keep_all_catalog_types("asassn")
    assert editor.rows()[editor.cursor].value == "all kept"


def test_filter_editor_exposes_all_major_filter_rows() -> None:
    keys = {row.key for row in FilterEditor(QueueFilterSpec.default()).rows()}

    assert {
        "queue_state",
        "signal_lane",
        "show_external_lightcurves",
        "external_source_atlas",
        "external_source_ztf",
        "external_source_gaia_epoch",
        "external_source_neowise",
        "external_source_allwise_mep",
        "external_availability_neowise",
        "external_availability_allwise_mep",
        "external_source_ps1",
        "external_source_vvvx_virac",
        "external_source_asas3",
        "external_source_crts",
        "external_source_dasch",
        "known_objects",
        "catalog_vsx",
        "catalog_gaia",
        "catalog_asassn",
        "catalog_simbad",
        "catalog_ztf",
        "catalog_microlens",
        "catalog_tns",
        "catalog_alerce",
        "catalog_yso",
        "high_ruwe",
        "high_pm",
        "exclude_known_neighbors",
        "exclude_dipper_contaminants",
        "morphology_primary",
        "physical_primary",
        "prob_hierarchical_artifact_or_nonvariable",
        "prob_usable_astrophysical_variable",
        "prob_dipper_dimming",
        "prob_eb_geometric_periodic",
        "prob_long_timescale_variable",
        "prob_brightening_transient",
        "prob_other_structured_variable",
        "prob_quasi_periodic_hierarchical",
        "prob_microlensing_hierarchical",
        "prob_long_period_variable_hierarchical",
        "prob_long_term_variable_hierarchical",
        "prob_recurrent_dipper_hierarchical",
        "prob_single_dipper_hierarchical",
        "prob_quasi_periodic_given_usable",
        "prob_microlensing_given_brightening",
        "prob_long_period_variable_given_long_timescale",
        "prob_long_term_variable_given_long_timescale",
        "prob_recurrent_given_dipper",
        "prob_single_given_dipper",
        "predicted_hierarchy_gate",
        "predicted_primary_morphology",
        "predicted_hierarchical_class",
        "predicted_quasi_periodic",
        "predicted_microlensing_like",
        "predicted_long_timescale_subtype",
        "predicted_dipper_recurrence",
        "dipper_score",
        "jumper_score",
        "q",
        "m",
        "g_magnitude",
        "period_days",
        "sort_by",
        "sort_desc",
        "categorical_logic",
    } <= keys


def test_hierarchy_filter_sections_are_ordered_and_headings_are_not_selectable() -> None:
    editor = FilterEditor(QueueFilterSpec.default())
    rows = editor.rows()
    headings = [row.label for row in rows if row.kind == "heading"]

    assert headings == [
        "PHOTOMETRY DISPLAY (ASAS-SN ALWAYS ON)",
        "PHOTOMETRY AVAILABILITY FILTERS (AND WITH ML)",
        "ML GATE",
        "ML GLOBAL PRIMARY",
        "ML GLOBAL SUBTYPES",
        "ML CONDITIONAL HEADS",
        "ML PREDICTED CLASSES",
        "TRIAGE METRICS",
    ]

    visited = set()
    for _ in range(editor.row_count * 2):
        visited.add(editor.active_key)
        assert editor.active_kind != "heading"
        editor.move(1)
    assert "prob_hierarchical_artifact_or_nonvariable" in visited
    assert "prob_long_term_variable_hierarchical" in visited
    assert "prob_single_dipper_hierarchical" in visited
    assert "prob_long_term_variable_given_long_timescale" in visited
    assert "prob_single_given_dipper" in visited
    assert "predicted_hierarchical_class" in visited


def test_predicted_hierarchy_class_selectors_cycle_through_named_classes() -> None:
    editor = FilterEditor(QueueFilterSpec.default())
    _move_to(editor, "predicted_hierarchical_class")

    assert editor.rows()[editor.cursor].value == "any"
    editor.cycle(1)
    assert editor.spec.predicted_hierarchical_class == "artifact_or_nonvariable"
    assert editor.rows()[editor.cursor].value == "artifact / nonvariable"
