from __future__ import annotations

from malca.plotting.color_color_labels import (
    EFFECTIVE_WAVELENGTH_UM,
    LABEL_H_KS,
    LABEL_KS_W4,
    LABEL_W1_W2,
    color_color_mag_label,
)


def test_color_color_mag_label_annotates_each_band() -> None:
    label = color_color_mag_label(r"H", r"K_s", "H", "Ks")
    assert r"\mu\mathrm{m}" in label
    assert r"\mathrm{[mag]}" in label
    assert "1.7" in label
    assert "2.2" in label
    assert label == LABEL_H_KS


def test_w1_w2_label_shows_both_band_wavelengths() -> None:
    label = color_color_mag_label(r"W_1", r"W_2", "W1", "W2")
    assert "3.4" in label
    assert "4.6" in label
    assert label == LABEL_W1_W2


def test_disk_color_labels_show_both_band_wavelengths() -> None:
    assert "2.2" in LABEL_KS_W4
    assert "22" in LABEL_KS_W4
    assert LABEL_W1_W2.startswith("$W_1")


def test_effective_wavelengths_match_bandpass_pivots() -> None:
    assert EFFECTIVE_WAVELENGTH_UM["W3"] == 11.561
    assert EFFECTIVE_WAVELENGTH_UM["Ha"] == 0.657
