"""Fetch ASAS-SN light curves via the skypatrol (pyasassn) client.

Provides three entry points:
  - download_lightcurve_by_id(asas_sn_id)
  - download_lightcurve_by_gaia_id(gaia_id)
  - cone_search(ra, dec, radius_arcsec)
  - download_lightcurve_for_target(asas_sn_id)

Downloaded CSVs are saved in SkyPatrol web-CSV format so that
``malca.utils.read_skypatrol_csv`` can read them unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Lazy singleton for the SkyPatrolClient
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        from pyasassn.client import SkyPatrolClient
        _client = SkyPatrolClient(verbose=False)
    return _client


# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------
_DEFAULT_CACHE = Path("~/.malca/cache/skypatrol").expanduser()


def _ensure_cache(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ---------------------------------------------------------------------------
# Column remapping: pyasassn -> SkyPatrol web-CSV format
# ---------------------------------------------------------------------------
# pyasassn LightCurve.data has columns:
#   jd, flux, flux_err, mag, mag_err, limit, fwhm, asas_sn_id, cam
#
# SkyPatrol web CSV (what read_skypatrol_csv expects):
#   JD, Flux, Flux Error, Mag, Mag Error, Limit, FWHM, Filter, Quality, Camera
#
_PYASASSN_TO_WEB = {
    "jd": "JD",
    "flux": "Flux",
    "flux_err": "Flux Error",
    "mag": "Mag",
    "mag_err": "Mag Error",
    "limit": "Limit",
    "fwhm": "FWHM",
    "cam": "Camera",
}


def _save_lc_as_skypatrol_csv(lc_data: pd.DataFrame, out_path: Path, filter_band: str = "g") -> Path:
    """Remap pyasassn DataFrame columns and save as SkyPatrol web-CSV format."""
    df = lc_data.rename(columns=_PYASASSN_TO_WEB).copy()

    # pyasassn has no Filter or Quality columns — synthesize them
    if "Filter" not in df.columns:
        df["Filter"] = filter_band
    if "Quality" not in df.columns:
        df["Quality"] = "G"  # mark all as good by default

    # Keep only the web-CSV columns in order
    web_cols = ["JD", "Flux", "Flux Error", "Mag", "Mag Error", "Limit", "FWHM", "Filter", "Quality", "Camera"]
    for c in web_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[web_cols]

    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_lightcurve_by_id(
    asas_sn_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> Path:
    """Download a light curve by ASAS-SN ID.  Returns path to saved CSV."""
    cache = _ensure_cache(cache_dir)
    out = cache / f"{asas_sn_id}.csv"
    if out.exists() and out.stat().st_size > 0:
        return out

    client = _get_client()
    result = client.query_list(
        [int(asas_sn_id)],
        id_col="asas_sn_id",
        catalog="master_list",
        download=True,
    )
    lc = result.data
    if lc is None or lc.empty:
        raise RuntimeError(f"No light curve returned for ASAS-SN ID {asas_sn_id}")

    _save_lc_as_skypatrol_csv(lc, out)
    return out


def download_lightcurve_by_gaia_id(
    gaia_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> Path:
    """Download a light curve by Gaia DR3 source_id.  Returns path to saved CSV."""
    cache = _ensure_cache(cache_dir)
    out = cache / f"gaia_{gaia_id}.csv"
    if out.exists() and out.stat().st_size > 0:
        return out

    client = _get_client()
    result = client.query_list(
        [int(gaia_id)],
        id_col="gaia_id",
        catalog="stellar_main",
        download=True,
    )
    lc = result.data
    if lc is None or lc.empty:
        raise RuntimeError(f"No light curve returned for Gaia ID {gaia_id}")

    _save_lc_as_skypatrol_csv(lc, out)
    return out


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    catalog: str = "stellar_main",
) -> pd.DataFrame:
    """Cone search on SkyPatrol.  Returns catalog DataFrame (no LC download)."""
    client = _get_client()
    radius_deg = radius_arcsec / 3600.0
    result = client.cone_search(
        ra_deg=ra,
        dec_deg=dec,
        radius=radius_deg,
        units="deg",
        catalog=catalog,
        download=False,
    )
    if result is None or (hasattr(result, "catalog_info") and result.catalog_info.empty):
        return pd.DataFrame()

    return result.catalog_info


def download_lightcurve_for_target(
    asas_sn_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> Path:
    """Alias for download_lightcurve_by_id."""
    return download_lightcurve_by_id(asas_sn_id, cache_dir=cache_dir)
