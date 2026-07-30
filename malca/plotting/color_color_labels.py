"""Axis labels for color-color diagrams with per-band wavelength annotations."""

from __future__ import annotations

# Effective wavelengths [micron] from malca/review/sed.py bandpass pivots.
EFFECTIVE_WAVELENGTH_UM: dict[str, float] = {
    "H": 1.662,
    "Ks": 2.159,
    "W1": 3.353,
    "W2": 4.603,
    "W3": 11.561,
    "W4": 22.088,
    "r": 0.623,
    "i": 0.763,
    "Ha": 0.657,
}


def _format_lambda_um(lam_um: float) -> str:
    if lam_um >= 10.0:
        text = f"{lam_um:.0f}"
    elif lam_um >= 1.0:
        text = f"{lam_um:.1f}".rstrip("0").rstrip(".")
    else:
        text = f"{lam_um:.2f}".rstrip("0").rstrip(".")
    return text


def lambda_um_bracket(lam_um: float) -> str:
    """LaTeX bracket annotation for math-mode axis labels."""
    return rf"[{_format_lambda_um(lam_um)}\,\mu\mathrm{{m}}]"


def color_color_mag_label(
    band_a_tex: str,
    band_b_tex: str,
    band_a: str,
    band_b: str,
) -> str:
    """Matplotlib axis label for a color index with per-band ``[mag]``."""
    lam_a = lambda_um_bracket(EFFECTIVE_WAVELENGTH_UM[band_a])
    lam_b = lambda_um_bracket(EFFECTIVE_WAVELENGTH_UM[band_b])
    return rf"${band_a_tex}\ {lam_a} - {band_b_tex}\ {lam_b}\ \mathrm{{[mag]}}$"


def band_mag_label(band_tex: str, band: str) -> str:
    """Matplotlib axis label for a single-band magnitude axis."""
    return (
        rf"${band_tex}\ {lambda_um_bracket(EFFECTIVE_WAVELENGTH_UM[band])}"
        rf"\ \mathrm{{[mag]}}$"
    )


def color_color_title_plain(
    band_a_name: str,
    band_b_name: str,
    band_a: str,
    band_b: str,
) -> str:
    """Plotly/plain axis title for a color index."""
    lam_a = _format_lambda_um(EFFECTIVE_WAVELENGTH_UM[band_a])
    lam_b = _format_lambda_um(EFFECTIVE_WAVELENGTH_UM[band_b])
    return f"{band_a_name} [{lam_a} μm] - {band_b_name} [{lam_b} μm]"


# Common matplotlib labels used across publication scripts.
LABEL_W1_W2 = color_color_mag_label(r"W_1", r"W_2", "W1", "W2")
LABEL_W2_W3 = color_color_mag_label(r"W_2", r"W_3", "W2", "W3")
LABEL_W3_W4 = color_color_mag_label(r"W_3", r"W_4", "W3", "W4")
LABEL_W1_W4 = color_color_mag_label(r"W_1", r"W_4", "W1", "W4")
LABEL_H_KS = color_color_mag_label(r"H", r"K_s", "H", "Ks")
LABEL_R_I = color_color_mag_label(r"r", r"i", "r", "i")
LABEL_R_HALPHA = color_color_mag_label(r"r", r"\mathrm{H}\alpha", "r", "Ha")
LABEL_R_HALPHA_PLAIN = color_color_mag_label(r"r", r"H\alpha", "r", "Ha")
LABEL_KS_W3 = color_color_mag_label(r"K_s", r"W_3", "Ks", "W3")
LABEL_KS_W4 = color_color_mag_label(r"K_s", r"W_4", "Ks", "W4")
LABEL_W3 = band_mag_label("W_3", "W3")

TITLE_W1_W2 = color_color_title_plain("W1", "W2", "W1", "W2")
TITLE_H_K = color_color_title_plain("H", "K", "H", "Ks")
