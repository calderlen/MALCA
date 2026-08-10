"""Shared foreground-extinction coefficients for derived photometric products.

The mid-infrared coefficients are monochromatic ``A_lambda / A_V`` values from
the Gordon et al. (2023; G23) ``R_V = 3.1`` relation evaluated at each
registry nominal wavelength.  Response-integrated SED forward modelling still
uses the full G23 curve; these scalar values support catalog-magnitude plots,
legacy normalized rows, and other products that do not carry a spectrum.
"""

from __future__ import annotations


MID_IR_EXTINCTION_POLICY_VERSION = "g23-rv31-nominal-wavelength-v1"
MID_IR_EXTINCTION_LAW = "G23"
MID_IR_EXTINCTION_RV = 3.1


# Keys are normalized as ``(source.casefold(), band.casefold())``.  Values were
# evaluated with ``dust_extinction.parameter_averages.G23(Rv=3.1)`` at the
# nominal wavelengths stored in ``malca.review.sed.SED_BANDPASSES``.
MID_IR_AV_COEFFICIENTS: dict[tuple[str, str], float] = {
    ("allwise", "w3"): 0.0440068055,
    ("allwise", "w4"): 0.0387976733,
    ("spitzer seip", "irac1"): 0.0443382959,
    ("spitzer seip", "irac2"): 0.0319828578,
    ("spitzer seip", "irac3"): 0.0267811426,
    ("spitzer seip", "irac4"): 0.0293044181,
    ("spitzer seip", "mips24"): 0.0367636023,
    ("akari", "s9w"): 0.0564159428,
    ("akari", "l18w"): 0.0422707742,
    ("iras", "12"): 0.0385226596,
    ("iras", "25"): 0.0352095443,
}


def mid_ir_av_coefficient(source: object, band: object) -> float | None:
    """Return the active scalar G23 mid-infrared ``A_lambda / A_V`` value."""
    key = (str(source or "").strip().casefold(), str(band or "").strip().casefold())
    return MID_IR_AV_COEFFICIENTS.get(key)
