from __future__ import annotations

from malca.review.classification_labels import (
    classification_tokens,
    format_catalog_class_label,
    resolve_catalog_class,
)


def test_gaia_uses_exact_archive_class_before_bar_fallback() -> None:
    resolved = resolve_catalog_class("gaia_var_class", "DSCT|GDOR|SXPHE")

    assert resolved.tokens == ("DSCT|GDOR|SXPHE",)
    assert resolved.matched is True
    assert "Delta Scuti" in resolved.label
    assert "[Gaia DR3]" in resolved.label


def test_alerce_treats_slash_and_hyphen_classes_as_exact_labels() -> None:
    assert classification_tokens("alerce_lc_class", "CV/Nova") == ["CV/Nova"]
    assert classification_tokens("alerce_lc_class", "Periodic-Other") == ["Periodic-Other"]
    assert "Cataclysmic variable or nova" in format_catalog_class_label(
        "alerce_lc_class",
        "CV/Nova",
    )


def test_simbad_preserves_uncertain_candidate_codes() -> None:
    label = format_catalog_class_label("simbad_otype", "EB?")

    assert "EB? - candidate/uncertain Eclipsing binary [SIMBAD]" == label


def test_vsx_composite_display_reuses_existing_tokenizer() -> None:
    assert classification_tokens("vsx_class", "EA/SD:") == ["EA", "SD"]

    label = format_catalog_class_label("vsx_class", "EA/SD:")
    assert "candidate/uncertain" in label
    assert "EA: Algol-type eclipsing binary" in label
    assert "SD: Semi-detached eclipsing system" in label
    assert "[VSX]" in label


def test_vsx_aavso_titles_cover_dipper_contaminant_tokens() -> None:
    assert "EX Lupi-type" in format_catalog_class_label("vsx_class", "EXOR")
    assert "FU Orionis-type" in format_catalog_class_label("vsx_class", "FUOR")
    assert "T Tauri-type Orion variable with abrupt fadings" in format_catalog_class_label(
        "vsx_class",
        "INAT",
    )
    assert "VY Sculptoris-type" in format_catalog_class_label("vsx_class", "VY")


def test_asassn_uses_asassn_exact_map_then_vsx_fallback() -> None:
    assert "Rotational variable" in format_catalog_class_label("asassn_var_type", "ROT:")
    assert "Gamma Cassiopeiae-type" in format_catalog_class_label(
        "asassn_var_type",
        "GCAS:",
    )


def test_ztf_and_tns_exact_lookup_labels() -> None:
    assert "BY Draconis rotational variable" in format_catalog_class_label(
        "ztf_var_type",
        "BYDra",
    )
    assert "Cataclysmic variable candidate" in format_catalog_class_label(
        "tns_type",
        "CV candidate",
    )


def test_unknown_values_fall_back_to_raw_source_label() -> None:
    assert format_catalog_class_label("tns_type", "strange new thing") == "strange new thing [TNS]"
    assert format_catalog_class_label("not_a_catalog_type", "ABC") == "ABC"
