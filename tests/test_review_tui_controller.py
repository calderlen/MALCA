from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from malca.review.taxonomy import keyboard_payload
from malca.review.tui_controller import (
    MORPHOLOGY_PRIMARY_BY_KEY,
    MORPHOLOGY_PRIMARY_ITEMS,
    MORPHOLOGY_SECONDARY_BY_KEY,
    MORPHOLOGY_SECONDARY_ITEMS,
    PHYSICAL_PRIMARY_BY_KEY,
    PHYSICAL_PRIMARY_ITEMS,
    PHYSICAL_SECONDARY_BY_KEY,
    PHYSICAL_SECONDARY_ITEMS,
    DetailSection,
    ReviewDraft,
    build_review_display_title,
    build_review_identity_line,
    compact_detail_lines,
    detail_sections,
    external_catalog_labels,
    physical_primary_item_for_key,
    physical_secondary_item_for_key,
    physical_secondary_items_for,
    primary_item_for_key,
    secondary_item_for_key,
    secondary_items_for,
)


def test_menu_items_are_immutable_views_of_canonical_keyboard_payload() -> None:
    payload = keyboard_payload()

    assert [
        {"key": item.key, "value": item.value, "label": item.label}
        for item in MORPHOLOGY_PRIMARY_ITEMS
    ] == payload["morphology_primary"]
    assert [item.value for item in secondary_items_for("dimming_event")] == [
        item["value"] for item in payload["morphology_secondary"]["dimming_event"]
    ]
    assert primary_item_for_key("E").value == "dimming_event"
    assert secondary_item_for_key("dimming_event", "K").value == "recurrent_dips"
    assert secondary_items_for(None) == ()
    assert secondary_item_for_key(None, "a") is None
    assert [
        {"key": item.key, "value": item.value, "label": item.label}
        for item in PHYSICAL_PRIMARY_ITEMS
    ] == payload["physical_primary"]
    assert physical_primary_item_for_key("Y").value == "young_stellar_object_or_pms"
    assert [
        {"key": item.key, "value": item.value, "label": item.label}
        for item in physical_secondary_items_for("pulsating_variable")
    ] == payload["physical_secondary"]["pulsating_variable"]
    assert (
        physical_secondary_item_for_key("pulsating_variable", "s").value
        == "rr_lyrae"
    )
    assert physical_secondary_items_for(None) == ()

    with pytest.raises(TypeError):
        MORPHOLOGY_PRIMARY_BY_KEY["!"] = MORPHOLOGY_PRIMARY_ITEMS[0]
    with pytest.raises(TypeError):
        PHYSICAL_PRIMARY_BY_KEY["!"] = PHYSICAL_PRIMARY_ITEMS[0]
    with pytest.raises(TypeError):
        MORPHOLOGY_SECONDARY_ITEMS["dimming_event"] = ()
    with pytest.raises(TypeError):
        MORPHOLOGY_SECONDARY_BY_KEY["dimming_event"]["!"] = secondary_items_for(
            "dimming_event"
        )[0]
    with pytest.raises(TypeError):
        PHYSICAL_SECONDARY_ITEMS["pulsating_variable"] = ()
    with pytest.raises(TypeError):
        PHYSICAL_SECONDARY_BY_KEY["pulsating_variable"]["!"] = (
            physical_secondary_items_for("pulsating_variable")[0]
        )
    with pytest.raises(FrozenInstanceError):
        MORPHOLOGY_PRIMARY_ITEMS[0].label = "changed"


def test_from_review_preserves_current_secondary_json_semantics_and_is_clean() -> None:
    draft = ReviewDraft.from_review(
        {
            "morphology_primary": "dimming_event",
            "morphology_secondary": "recurrent_dips",
            "morphology_secondary_json": json.dumps(
                ["multi_depth_dips", "recurrent_dips", "multi_depth_dips"]
            ),
            "classification_confidence": "3",
            "physical_primary": "young_stellar_object_or_pms",
            "physical_secondary": "yso_dipper",
            "workflow_status": "needs_followup",
        }
    )

    assert draft.morphology_primary == "dimming_event"
    assert draft.morphology_secondaries == ["multi_depth_dips", "recurrent_dips"]
    assert draft.morphology_secondary == "multi_depth_dips"
    assert draft.morphology_secondary_json == '["multi_depth_dips","recurrent_dips"]'
    assert draft.confidence == 3
    assert draft.physical_primary == "young_stellar_object_or_pms"
    assert draft.physical_secondary == "yso_dipper"
    assert draft.needs_followup is True
    assert draft.dirty is False


def test_from_review_promotes_legacy_scalar_and_honors_explicit_followup() -> None:
    draft = ReviewDraft.from_review(
        {
            "morphology_primary": "dimming_event",
            "morphology_secondary": "sharp_dip",
            "classification_confidence": 2,
            "workflow_status": "needs_followup",
            "needs_followup": False,
        }
    )

    assert draft.morphology_secondaries == ["sharp_dip"]
    assert draft.needs_followup is False


def test_select_primary_only_clears_subtypes_when_primary_changes() -> None:
    draft = ReviewDraft.from_review(
        {
            "morphology_primary": "dimming_event",
            "morphology_secondary_json": '["sharp_dip","recurrent_dips"]',
        }
    )

    assert draft.select_primary("dimming_event") is False
    assert draft.morphology_secondaries == ["sharp_dip", "recurrent_dips"]
    assert draft.dirty is False

    assert draft.select_primary("brightening_event") is True
    assert draft.morphology_secondaries == []
    assert draft.dirty is True

    with pytest.raises(ValueError, match="Unknown morphology primary"):
        draft.select_primary("not_a_primary")


def test_toggle_subtype_validates_membership_and_preserves_selection_order() -> None:
    draft = ReviewDraft()
    with pytest.raises(ValueError, match="Select a morphology primary"):
        draft.toggle_subtype("sharp_dip")

    draft.select_primary("dimming_event")
    assert draft.toggle_subtype("recurrent_dips") is True
    assert draft.toggle_subtype("sharp_dip") is True
    assert draft.toggle_subtype("recurrent_dips") is False
    assert draft.toggle_subtype("broad_dip") is True
    assert draft.morphology_secondaries == ["sharp_dip", "broad_dip"]

    with pytest.raises(ValueError, match="does not belong"):
        draft.toggle_subtype("single_brightening")


def test_clear_confidence_followup_validation_and_mark_saved() -> None:
    draft = ReviewDraft()

    assert draft.validate() == (
        "Morphology is required",
        "Confidence must be from 1 to 4",
    )
    draft.select_primary("dimming_event")
    assert draft.validate() == ("Confidence must be from 1 to 4",)
    draft.set_confidence("4")
    assert draft.validate() == ()
    assert draft.toggle_subtype("sharp_dip") is True
    assert draft.clear_subtypes() is True
    assert draft.clear_subtypes() is False
    assert draft.toggle_followup() is True
    assert draft.dirty is True

    draft.mark_saved()
    assert draft.dirty is False
    assert draft.toggle_followup() is False
    assert draft.dirty is True

    for invalid in (None, 0, 5, "high"):
        with pytest.raises(ValueError, match="1 to 4"):
            draft.set_confidence(invalid)


def test_select_physical_primary_tracks_dirty_state_and_can_clear() -> None:
    draft = ReviewDraft.from_review(
        {
            "physical_primary": "microlensing",
            "physical_secondary": "point_lens_candidate",
        }
    )

    assert draft.select_physical_primary("microlensing") is False
    assert draft.dirty is False
    assert draft.select_physical_primary("eclipsing_or_geometric_binary") is True
    assert draft.physical_primary == "eclipsing_or_geometric_binary"
    assert draft.physical_secondary is None
    assert draft.dirty is True
    assert draft.select_physical_primary(None) is True
    assert draft.physical_primary is None

    with pytest.raises(ValueError, match="Unknown physical primary"):
        draft.select_physical_primary("not_a_physical_primary")

    invalid = ReviewDraft(physical_primary="not_a_physical_primary")
    assert "Physical hypothesis is not recognized" in invalid.validate()


def test_physical_subtype_is_single_select_toggle_and_notes_are_dirty() -> None:
    draft = ReviewDraft.from_review(
        {
            "physical_primary": "pulsating_variable",
            "physical_secondary": "rr_lyrae",
            "notes": "browser note",
        }
    )

    assert draft.toggle_physical_subtype("rr_lyrae") is False
    assert draft.physical_secondary is None
    assert draft.toggle_physical_subtype("cepheid") is True
    assert draft.physical_secondary == "cepheid"
    assert draft.clear_physical_subtype() is True
    assert draft.clear_physical_subtype() is False
    draft.set_notes("terminal note")
    assert draft.notes == "terminal note"
    assert draft.dirty is True

    with pytest.raises(ValueError, match="does not belong"):
        draft.toggle_physical_subtype("yso_dipper")


def test_to_selection_uses_first_secondary_as_scalar_and_returns_a_copy() -> None:
    draft = ReviewDraft.from_review(
        {
            "morphology_primary": "dimming_event",
            "morphology_secondary_json": '["sharp_dip","recurrent_dips"]',
            "classification_confidence": 3,
        }
    )

    selection = draft.to_selection()
    assert selection == {
        "morphology_primary": "dimming_event",
        "morphology_secondary": "sharp_dip",
        "morphology_secondary_list": ["sharp_dip", "recurrent_dips"],
        "morphology_secondary_json": '["sharp_dip","recurrent_dips"]',
        "physical_primary": None,
        "physical_secondary": None,
        "classification_confidence": 3,
    }

    selection["morphology_secondary_list"].append("multi_depth_dips")
    assert draft.morphology_secondaries == ["sharp_dip", "recurrent_dips"]


def test_compact_detail_lines_formats_payload_and_explicit_phase_result() -> None:
    lines = compact_detail_lines(
        {
            "phot_g_mean_mag": 13.26446,
            "bp_rp": 0.99987,
            "ruwe": 1.04214,
            "prob_hierarchical_artifact_or_nonvariable": 0.06,
            "prob_usable_astrophysical_variable": 0.94,
            "prob_brightening_transient": 0.12,
            "prob_dipper_dimming": 0.826,
            "prob_eb_geometric_periodic": 0.10,
            "prob_long_period_variable_hierarchical": 0.03,
            "prob_long_term_variable_hierarchical": 0.01,
            "prob_long_timescale_variable": 0.04,
            "prob_other_structured_variable": 0.03,
            "prob_microlensing_hierarchical": 0.01,
            "prob_quasi_periodic_hierarchical": 0.02,
            "prob_recurrent_dipper_hierarchical": 0.07,
            "prob_single_dipper_hierarchical": 0.756,
            "dipper_score": 21.944,
            "jumper_score": 18.563,
            "stats_variability_quasi_periodicity_q": 0.85657,
            "stats_variability_flux_asymmetry_m": -0.01671,
            "periodicity_period": 5.99884,
            "vsx_class": "EA",
            "gaia_var_class": "ECL",
            "vetting_likely_known": False,
        },
        phase_period_days=0.3121309405940594,
        phase_period_source="Auto PDM",
    )

    assert lines == (
        "ML Reject .06   Usable .94",
        "ML Dipper .83   EB .10   Long .04",
        "ML Bright .12   Other .03",
        "ML QP .02   Micro .01   LPV .03",
        "ML LTV .01   Recur .07   Single .76",
        "Q .86   M -.02   period 0.31 d (Auto PDM)",
        "RUWE 1.04",
        "VSX EA   Gaia VAR ECL   likely known no",
    )


def test_compact_detail_lines_handles_aliases_missing_values_and_tristates() -> None:
    assert compact_detail_lines(
        {
            "gaia_phot_g_mean_mag": "14.5",
            "derived_bp_rp": 1.25,
            "ruwe_gaia": float("nan"),
            "phase_period_days": 2.5,
            "phase_source": "stored",
            "gaia_var_flag": "true",
            "vetting_likely_known": None,
        }
    ) == (
        "ML Reject —   Usable —",
        "ML Dipper —   EB —   Long —",
        "ML Bright —   Other —",
        "ML QP —   Micro —   LPV —",
        "ML LTV —   Recur —   Single —",
        "Q —   M —   period 2.5 d (stored)",
        "RUWE —",
        "VSX —   Gaia VAR yes   likely known —",
    )


def test_detail_sections_group_and_align_portrait_metadata() -> None:
    sections = detail_sections(
        {
            "phot_g_mean_mag": 13.26446,
            "bp_rp": 0.99987,
            "ruwe": 1.04214,
            "prob_hierarchical_artifact_or_nonvariable": 0.06,
            "prob_usable_astrophysical_variable": 0.94,
            "prob_brightening_transient": 0.12,
            "prob_dipper_dimming": 0.826,
            "prob_eb_geometric_periodic": 0.10,
            "prob_long_period_variable_hierarchical": 0.03,
            "prob_long_term_variable_hierarchical": 0.01,
            "prob_long_timescale_variable": 0.04,
            "prob_other_structured_variable": 0.03,
            "prob_microlensing_hierarchical": 0.01,
            "prob_quasi_periodic_hierarchical": 0.02,
            "prob_recurrent_dipper_hierarchical": 0.07,
            "prob_single_dipper_hierarchical": 0.756,
            "dipper_score": 21.944,
            "jumper_score": 18.563,
            "stats_variability_quasi_periodicity_q": 0.85657,
            "stats_variability_flux_asymmetry_m": -0.01671,
            "periodicity_period": 5.99884,
            "vsx_class": "EA",
            "gaia_var_class": "ECL",
            "vetting_likely_known": False,
        },
        phase_period_days=0.3121309405940594,
        phase_period_source="Auto PDM",
    )

    assert sections == (
        DetailSection(
            key="ml_class_scores",
            title="ML CLASS SCORES",
            lines=(
                "  P(reject) .06",
                "  P(usable) .94",
                "  P(dip)    .83",
                "  P(EB)     .10",
                "  P(long)   .04",
                "  P(bright) .12",
                "  P(other)  .03",
                "  P(QP)     .02",
                "  P(micro)  .01",
                "  P(LPV)    .03",
                "  P(LTV)    .01",
                "  P(recur)  .07",
                "  P(single) .76",
            ),
        ),
        DetailSection(
            key="signal",
            title="SIGNAL",
            lines=(
                "  Q         .86",
                "  M         -.02",
                "  mean mag  —",
                "  period    0.31 d",
                "  source    Auto PDM",
            ),
        ),
        DetailSection(
            key="astrometry",
            title="ASTROMETRY",
            lines=(
                "  RUWE      1.04",
                "  PM        —",
                "  α_SED     —",
                "  α class   —",
            ),
        ),
        DetailSection(
            key="starhorse",
            title="STARHORSE",
            lines=(
                "  Teff      —",
                "  type      —",
                "  log g     —",
                "  [M/H]     —",
                "  Mass      —",
                "  A_V       —",
            ),
        ),
        DetailSection(
            key="context",
            title="CONTEXT",
            lines=(
                "  BANYAN    —",
                "  EB        none",
            ),
        ),
        DetailSection(
            key="catalogs",
            title="CATALOGS",
            lines=(
                "  VSX       EA",
                "  Gaia VAR  ECL",
                "  ASAS-SN   —",
                "  SIMBAD    —",
                "  known     no",
            ),
        ),
    )


def test_detail_sections_use_aliases_missing_markers_and_bound_line_width() -> None:
    sections = detail_sections(
        {
            "gaia_phot_g_mean_mag": "14.5",
            "derived_bp_rp": 1.25,
            "ruwe_gaia": float("nan"),
            "phase_period_days": 2.5,
            "phase_source": "a deliberately long stored-period source",
            "period_vsx_class": "LONG-CATALOG-CLASS",
            "gaia_var_flag": "true",
            "vetting_likely_known": None,
        },
        width=24,
    )

    assert [section.key for section in sections] == [
        "ml_class_scores",
        "signal",
        "astrometry",
        "starhorse",
        "context",
        "catalogs",
    ]
    assert sections[1].lines[-2:] == (
        "  period    2.5 d",
        "  source    a deliberat…",
    )
    assert sections[2].lines == (
        "  RUWE      —",
        "  PM        —",
        "  α_SED     —",
        "  α class   —",
    )
    assert sections[5].lines == (
        "  VSX       LONG-CATALO…",
        "  Gaia VAR  yes",
        "  ASAS-SN   —",
        "  SIMBAD    —",
        "  known     —",
    )
    assert all(
        len(line) <= 24 for section in sections for line in section.lines
    )


def test_detail_sections_and_compact_output_use_hierarchical_scores() -> None:
    payload = {
        "prob_dipper_dimming": 0.25,
        "phase_period_days": 8.103976,
        "phase_source": "Auto PDM",
    }

    detail_sections(payload, width=22)

    assert compact_detail_lines(payload) == (
        "ML Reject —   Usable —",
        "ML Dipper .25   EB —   Long —",
        "ML Bright —   Other —",
        "ML QP —   Micro —   LPV —",
        "ML LTV —   Recur —   Single —",
        "Q —   M —   period 8.1 d (Auto PDM)",
        "RUWE —",
        "VSX —   Gaia VAR —   likely known —",
    )


def test_build_review_identity_line_shows_asas_and_gaia_ids_only() -> None:
    line = build_review_identity_line(
        {"source_id": "1234567890123456789"},
        asas_sn_id="68720730002",
    )
    assert line == "ASAS-SN: 68720730002  GAIA: 1234567890123456789"


def test_build_review_display_title_includes_ids_and_catalog_classes() -> None:
    payload = {
        "asas_sn_id": "68720730002",
        "gaia_id": "1234567890123456789",
        "vsx_class": "SRS",
        "gaia_var_class": "LPV",
    }
    title = build_review_display_title(payload, asas_sn_id="68720730002")
    assert title == (
        "ASAS-SN 68720730002  ·  Gaia 1234567890123456789  ·  "
        "VSX SRS  ·  Gaia LPV"
    )
    assert external_catalog_labels({"vsx_class": "—"}) == ()
    assert build_review_display_title({}) == "MALCA Review"


def test_detail_sections_include_simbad_catalog_label() -> None:
    sections = detail_sections({"simbad_otype": "WR*"})
    catalogs = next(section for section in sections if section.key == "catalogs")
    assert "  SIMBAD    WR* - Wolf-Rayet star" in catalogs.lines
    assert external_catalog_labels({"simbad_otype": "WR*"}) == (
        "SIMBAD WR* - Wolf-Rayet star",
    )


def test_detail_sections_include_canonical_asassn_variable_catalog_fields() -> None:
    payload = {
        "asassn_var_name": "ASASSN-V J012345.67-123456.7",
        "asassn_var_type": "YSO",
        "asassn_var_period": 12.34567,
    }
    sections = detail_sections(payload)
    catalogs = next(section for section in sections if section.key == "catalogs")

    assert "  ASAS-SN   YSO P=12.3457d" in catalogs.lines
    assert external_catalog_labels(payload) == ("ASAS-SN YSO",)


def test_detail_sections_pm_from_pmra_pmdec() -> None:
    sections = detail_sections({"pmra": 80.0, "pmdec": 60.0})
    astrometry = next(section for section in sections if section.key == "astrometry")
    assert "  PM        100.0" in astrometry.lines


def test_detail_sections_pm_prefers_pm_total() -> None:
    sections = detail_sections({"pm_total": 42.5, "pmra": 80.0, "pmdec": 60.0})
    astrometry = next(section for section in sections if section.key == "astrometry")
    assert "  PM        42.5" in astrometry.lines


def test_detail_sections_starhorse_uses_max_asymmetric_uncertainty() -> None:
    sections = detail_sections(
        {
            "teff50": 3510.0,
            "teff16": 3501.0,
            "teff84": 3519.0,
            "logg50": 4.77,
            "logg16": 4.77,
            "logg84": 4.87,
            "met50": -0.59,
            "met16": -0.59,
            "met84": -0.59,
        }
    )
    starhorse = next(section for section in sections if section.key == "starhorse")
    assert starhorse.lines[:4] == (
        "  Teff      3510±9",
        "  type      ≈M2 V",
        "  log g     4.77±0.10",
        "  [M/H]     -0.59±0.00",
    )


def test_approximate_stellar_type_propagates_teff_and_logg_ranges() -> None:
    from malca.review.tui_controller import _approximate_stellar_type

    assert _approximate_stellar_type(
        15_060.0,
        12_856.0,
        16_300.0,
        3.94,
        3.81,
        3.98,
    ) == "≈B4–B7 IV"
    assert _approximate_stellar_type(
        5_770.0, None, None, 4.44, None, None
    ) == "≈G2 V"
    assert _approximate_stellar_type(
        5_770.0, None, None, None, None, None
    ) == "≈G2"
    assert _approximate_stellar_type(
        None, None, None, 4.44, None, None
    ) == "—"


def test_format_period_uses_at_most_two_decimal_places() -> None:
    from malca.review.tui_controller import _format_period, _format_period_days

    assert _format_period_days(310.0) == "310"
    assert _format_period(310.0) == "310 d"
    assert _format_period_days(0.3121309405940594) == "0.31"
    assert _format_period_days(3.4303091769975387e-19) == "3.43e-19"
    assert _format_period(2.5) == "2.5 d"
    assert _format_period(8.103976) == "8.1 d"
