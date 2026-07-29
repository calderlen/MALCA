"""Small, dependency-light astrometric helpers for catalog identity checks."""

from __future__ import annotations

import numpy as np


GAIA_REFERENCE_EPOCH_JYEAR = 2016.0
JYEAR_DAYS = 365.25


def propagate_linear_icrs(
    ra_deg: float,
    dec_deg: float,
    target_mjd: np.ndarray | float,
    *,
    pmra_mas_per_year: float | None = None,
    pmdec_mas_per_year: float | None = None,
    reference_epoch_jyear: float = GAIA_REFERENCE_EPOCH_JYEAR,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Propagate ICRS coordinates with the standard ``mu_alpha*`` convention.

    Missing proper motion is not replaced by a scientific zero: coordinates are
    returned unchanged and the method explicitly reports ``static_missing_pm``.
    """
    mjd = np.asarray(target_mjd, dtype=float)
    ra = np.full(mjd.shape, float(ra_deg), dtype=float)
    dec = np.full(mjd.shape, float(dec_deg), dtype=float)
    try:
        pmra = float(pmra_mas_per_year) if pmra_mas_per_year is not None else np.nan
        pmdec = float(pmdec_mas_per_year) if pmdec_mas_per_year is not None else np.nan
    except (TypeError, ValueError):
        pmra = pmdec = np.nan
    if not (np.isfinite(pmra) and np.isfinite(pmdec)):
        return ra, dec, "static_missing_pm"

    reference_mjd = 51544.5 + (float(reference_epoch_jyear) - 2000.0) * JYEAR_DAYS
    delta_year = (mjd - reference_mjd) / JYEAR_DAYS
    cos_dec = np.cos(np.deg2rad(float(dec_deg)))
    if abs(cos_dec) < 1.0e-8:
        return ra, dec, "static_near_pole"
    ra = np.mod(ra + pmra * delta_year / (3.6e6 * cos_dec), 360.0)
    dec = dec + pmdec * delta_year / 3.6e6
    return ra, dec, "proper_motion_linear"


def angular_separation_arcsec(
    ra1_deg: np.ndarray | float,
    dec1_deg: np.ndarray | float,
    ra2_deg: np.ndarray | float,
    dec2_deg: np.ndarray | float,
) -> np.ndarray:
    """Vectorized great-circle separation in arcseconds."""
    ra1 = np.deg2rad(np.asarray(ra1_deg, dtype=float))
    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=float))
    ra2 = np.deg2rad(np.asarray(ra2_deg, dtype=float))
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=float))
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 3600.0
