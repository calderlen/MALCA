from __future__ import annotations

from malca.review.metadata import bracket_unit_label


def test_bracket_unit_label_updates_units_only() -> None:
    assert bracket_unit_label("Period consensus (d)") == "Period consensus [d]"
    assert bracket_unit_label("Trend slope (mag/yr)") == "Trend slope [mag/yr]"
    assert bracket_unit_label("T_eff (K)") == "T_eff [K]"
    assert bracket_unit_label('SIMBAD sep (")') == "SIMBAD sep [arcsec]"
    assert bracket_unit_label("H-K (dered)") == "H-K (dered)"
    assert bracket_unit_label("A_V (3D)") == "A_V (3D)"
    assert bracket_unit_label("Teff (GSP-Phot)") == "Teff (GSP-Phot)"
