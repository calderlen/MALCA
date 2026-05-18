from __future__ import annotations

from malca.review.filter_schema import is_definite_known_type_value


def test_vsx_uncertainty_and_generic_classes_stay_visible() -> None:
    assert is_definite_known_type_value("vsx_class", "EA") is True
    assert is_definite_known_type_value("vsx_class", "EA:") is False
    assert is_definite_known_type_value("vsx_class", "DSCT:+VAR") is False
    assert is_definite_known_type_value("vsx_class", "VAR") is False
    assert is_definite_known_type_value("vsx_class", "MISC") is False
    assert is_definite_known_type_value("vsx_class", "None") is False
    assert is_definite_known_type_value("vsx_class", "*") is False


def test_asassn_uncertainty_and_generic_classes_stay_visible() -> None:
    assert is_definite_known_type_value("asassn_var_type", "ROT") is True
    assert is_definite_known_type_value("asassn_var_type", "ROT:") is False
    assert is_definite_known_type_value("asassn_var_type", "GCAS:") is False
    assert is_definite_known_type_value("asassn_var_type", "VAR") is False


def test_simbad_uncertainty_and_generic_types_stay_visible() -> None:
    assert is_definite_known_type_value("simbad_otype", "EB*") is True
    assert is_definite_known_type_value("simbad_otype", "EB?") is False
    assert is_definite_known_type_value("simbad_otype", "Y*O") is True
    assert is_definite_known_type_value("simbad_otype", "Y*?") is False
    assert is_definite_known_type_value("simbad_otype", "s?r") is False
    assert is_definite_known_type_value("simbad_otype", "*") is False
    assert is_definite_known_type_value("simbad_otype", "**") is False
    assert is_definite_known_type_value("simbad_otype", "G") is False
    assert is_definite_known_type_value("simbad_otype", "FIR") is False
    assert is_definite_known_type_value("simbad_otype", "mul") is False


def test_tns_definite_types_and_candidate_text() -> None:
    assert is_definite_known_type_value("tns_type", "SN Ia") is True
    assert is_definite_known_type_value("tns_type", "SN II") is True
    assert is_definite_known_type_value("tns_type", "Nova") is True
    assert is_definite_known_type_value("tns_type", "CV") is True
    assert is_definite_known_type_value("tns_type", "TDE") is True
    assert is_definite_known_type_value("tns_type", "CV candidate") is False
    assert is_definite_known_type_value("tns_type", "SN?") is False
    assert is_definite_known_type_value("tns_type", "Fading blue star") is False
    assert is_definite_known_type_value("tns_type", "star with a deep dimming event") is False


def test_classifier_sources_are_definite_when_non_empty() -> None:
    assert is_definite_known_type_value("gaia_var_class", "ECL") is True
    assert is_definite_known_type_value("ztf_var_type", "EA") is True
    assert is_definite_known_type_value("alerce_lc_class", "Periodic") is True
    assert is_definite_known_type_value("gaia_var_class", "") is False
