"""IRAC Vega-colour utilities for diagnostic colour-colour plots."""

from __future__ import annotations

import numpy as np


# IRAC zero-magnitude flux densities [Jy] used by the IRSA/GLIMPSE catalog
# convention (Reach et al. 2005): 3.6, 4.5, 5.8, and 8.0 micron respectively.
IRAC_VEGA_ZERO_POINT_JY: dict[str, float] = {
    "IRAC1": 280.9,
    "IRAC2": 179.7,
    "IRAC3": 115.0,
    "IRAC4": 64.13,
}


def irac_vega_magnitude(flux_nu_jy: np.ndarray, band: str) -> np.ndarray:
    """Convert IRAC flux density in Jy to a Vega magnitude."""
    flux = np.asarray(flux_nu_jy, dtype=float)
    zero_point = IRAC_VEGA_ZERO_POINT_JY[band]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0.0, -2.5 * np.log10(flux / zero_point), np.nan)


def irac_vega_magnitude_error(
    flux_nu_jy: np.ndarray,
    flux_nu_jy_err: np.ndarray,
) -> np.ndarray:
    """Propagate flux-density uncertainty into a Vega magnitude uncertainty."""
    flux = np.asarray(flux_nu_jy, dtype=float)
    flux_err = np.asarray(flux_nu_jy_err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (flux > 0.0) & (flux_err >= 0.0),
            2.5 / np.log(10.0) * flux_err / flux,
            np.nan,
        )
