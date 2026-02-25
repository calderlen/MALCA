"""Fetch ASAS-SN light curves via the skypatrol (pyasassn) client.

Provides three entry points:
  - download_lightcurve_by_id(asas_sn_id)
  - download_lightcurve_by_gaia_id(gaia_id)
  - cone_search(ra, dec, radius_arcsec)

Downloaded CSVs are saved in SkyPatrol web-CSV format so that
``malca.utils.read_skypatrol_csv`` can read them unchanged.
"""

from __future__ import annotations

from pathlib import Path

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


def _save_lc_as_skypatrol_csv(lc_data: pd.DataFrame, out_path: Path) -> Path:
    """Remap pyasassn DataFrame columns and save as SkyPatrol web-CSV format."""
    df = lc_data.rename(columns=_PYASASSN_TO_WEB).copy()

    # Derive filter from camera name if missing.
    # ASAS-SN convention: lowercase cam prefix (b*) = g-band, uppercase (B*) = V-band.
    if "Filter" not in df.columns and "Camera" in df.columns:
        df["Filter"] = df["Camera"].astype(str).apply(
            lambda c: "V" if c and c[0].isupper() else "g"
        )
    elif "Filter" not in df.columns:
        df["Filter"] = "g"

    if "Quality" not in df.columns:
        df["Quality"] = "G"

    web_cols = ["JD", "Flux", "Flux Error", "Mag", "Mag Error",
                "Limit", "FWHM", "Filter", "Quality", "Camera"]
    for c in web_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[web_cols]

    df.to_csv(out_path, index=False)
    return out_path


def _query_catalog_info(
    target_ids: list,
    id_col: str = "asas_sn_id",
    catalog: str = "stellar_main",
) -> dict:
    """Query SkyPatrol catalog for ALL metadata columns (no download).

    Returns a dict with all catalog columns for the first matched target.
    Uses cols=None which tells pyasassn to return ALL columns.
    """
    client = _get_client()

    # Get ALL columns from the catalog (not just default 3)
    all_cols = list(client.catalogs[catalog]["col_names"])
    try:
        result = client.query_list(
            target_ids, id_col=id_col, catalog=catalog,
            cols=all_cols, download=False
        )
    except Exception:
        # Fallback to default columns if full query fails
        result = client.query_list(
            target_ids, id_col=id_col, catalog=catalog,
            download=False
        )

    if result is None or (isinstance(result, pd.DataFrame) and result.empty):
        return {}
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        row = result.iloc[0].to_dict()
        # Clean NaN values
        return {k: v for k, v in row.items()
                if v is not None and not (isinstance(v, float) and pd.isna(v))}
    return {}


def _download_lc(
    target_ids: list,
    id_col: str = "asas_sn_id",
    catalog: str = "master_list",
) -> pd.DataFrame | None:
    """Download light curve data. Returns the LC DataFrame or None.

    Downloads from master_list (which indexes all targets).
    """
    client = _get_client()
    result = client.query_list(
        target_ids, id_col=id_col, catalog=catalog, download=True
    )

    if result is None:
        return None
    if hasattr(result, "data"):
        return result.data
    if isinstance(result, pd.DataFrame) and not result.empty:
        if "jd" in result.columns or "JD" in result.columns:
            return result
    return None

def _cone_search_catalog(
    ra: float,
    dec: float,
    radius_arcsec: float = 3.0,
    catalog: str = "stellar_main",
) -> dict:
    """Cone search stellar_main at coords, return first match as dict with all columns."""
    client = _get_client()
    all_cols = list(client.catalogs[catalog]["col_names"])
    radius_deg = radius_arcsec / 3600.0
    try:
        result = client.cone_search(
            ra_deg=ra, dec_deg=dec, radius=radius_deg, units="deg",
            catalog=catalog, cols=all_cols, download=False,
        )
        if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
            row = result.iloc[0].to_dict()
            return {k: v for k, v in row.items()
                    if v is not None and not (isinstance(v, float) and pd.isna(v))}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_lightcurve_by_id(
    asas_sn_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> tuple[Path, dict]:
    """Download a light curve by ASAS-SN ID.

    Returns (path_to_csv, catalog_info_dict).
    Queries stellar_main for full metadata (47 cols), downloads LC from master_list.
    """
    cache = _ensure_cache(cache_dir)
    out = cache / f"{asas_sn_id}.csv"

    # Query stellar_main for full catalog metadata (parallax, PM, Gaia mag, etc.)
    catalog_info = _query_catalog_info(
        [int(asas_sn_id)], id_col="asas_sn_id", catalog="stellar_main"
    )

    # Fallback: if not in stellar_main, get coords from master_list,
    # then cone search stellar_main at those coords.
    if not catalog_info:
        ml_info = _query_catalog_info(
            [int(asas_sn_id)], id_col="asas_sn_id", catalog="master_list"
        )
        if ml_info:
            catalog_info = dict(ml_info)  # at least get RA/Dec
            ra = ml_info.get("ra_deg")
            dec = ml_info.get("dec_deg")
            if ra is not None and dec is not None:
                enriched = _cone_search_catalog(ra, dec, radius_arcsec=3.0)
                if enriched:
                    # Merge: stellar_main data takes priority, keep master_list RA/Dec
                    enriched.update({k: v for k, v in catalog_info.items()
                                     if k in ("asas_sn_id", "ra_deg", "dec_deg")})
                    catalog_info = enriched

    if out.exists() and out.stat().st_size > 0:
        return out, catalog_info

    # Download LC from master_list (which indexes all targets)
    lc = _download_lc([int(asas_sn_id)], id_col="asas_sn_id", catalog="master_list")
    if lc is None or lc.empty:
        raise RuntimeError(f"No light curve returned for ASAS-SN ID {asas_sn_id}")

    _save_lc_as_skypatrol_csv(lc, out)
    return out, catalog_info


def download_lightcurve_by_gaia_id(
    gaia_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> tuple[Path, dict]:
    """Download a light curve by Gaia DR3 source_id.

    Returns (path_to_csv, catalog_info_dict).
    """
    cache = _ensure_cache(cache_dir)
    out = cache / f"gaia_{gaia_id}.csv"

    # stellar_main has gaia_id column
    catalog_info = _query_catalog_info(
        [int(gaia_id)], id_col="gaia_id", catalog="stellar_main"
    )

    if out.exists() and out.stat().st_size > 0:
        return out, catalog_info

    # Download via stellar_main (gaia_id lookup)
    lc = _download_lc([int(gaia_id)], id_col="gaia_id", catalog="stellar_main")
    if lc is None or lc.empty:
        raise RuntimeError(f"No light curve returned for Gaia ID {gaia_id}")

    _save_lc_as_skypatrol_csv(lc, out)
    return out, catalog_info


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    catalog: str = "stellar_main",
) -> pd.DataFrame:
    """Cone search on SkyPatrol. Returns catalog DataFrame (no LC download)."""
    client = _get_client()
    radius_deg = radius_arcsec / 3600.0

    # Get all cols from catalog
    all_cols = list(client.catalogs[catalog]["col_names"])
    result = client.cone_search(
        ra_deg=ra,
        dec_deg=dec,
        radius=radius_deg,
        units="deg",
        catalog=catalog,
        cols=all_cols,
        download=False,
    )
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        return result
    if hasattr(result, "catalog_info"):
        return result.catalog_info
    return pd.DataFrame()
