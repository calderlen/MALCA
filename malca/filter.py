"""
Filters that run AFTER events.py.
Most filters depend only on the output columns from events.py; the camera
median validation also reads per-camera stats from .raw2 files via path.

Filters:
7. filter_evidence_strength - require strong Bayes factors
8. filter_significant_detection - require explicit significant run/peak evidence
9. filter_run_robustness - require sufficient run count and points
10. filter_morphology - require specific morphology with good BIC
11. filter_score - require minimum dipper/jumper scores (log10 event score)

Validation filters (expensive, run on candidates only):
11. validate_periodicity - bootstrap PDM/CE to check if source is periodic
12. validate_gaia_ruwe - flag/reject high RUWE sources from Gaia
13. validate_gaia_proper_motion - flag/reject high proper motion sources
14. validate_periodic_catalog - cross-match against known periodic catalogs

Required input columns (from events.py):
    dip_significant, jump_significant,
    dip_count, jump_count,
    dip_bayes_factor, jump_bayes_factor,
    dip_max_log_bf_local, jump_max_log_bf_local,
    dip_run_count, jump_run_count,
    dip_max_run_points, jump_max_run_points,
    dip_max_run_cameras, jump_max_run_cameras,
    dip_best_morph, jump_best_morph,
    dip_best_delta_bic, jump_best_delta_bic,
    dipper_score (for score filter),
    path (for logging and camera median validation)
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from multiprocessing import Pool, cpu_count
from pathlib import Path
from pathlib import Path as WorkerPath
from time import perf_counter
import argparse
import json
import math
import re
import shlex
import sys
import time

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyvo

from malca.config import (
    MIN_BAYES_FACTOR,
    POST_FILTER_MIN_RUN_CAMERAS,
    POST_FILTER_MIN_RUN_POINTS,
    POST_FILTER_MAX_RUWE,
    POST_FILTER_MAX_PM,
    POST_FILTER_MAX_SEP_ARCSEC,
    POST_FILTER_REL_TOL,
    POST_FILTER_MIN_DELTA_BIC,
    POST_FILTER_MIN_DIP_SCORE,
    POST_FILTER_MIN_JUMP_SCORE,
    POST_FILTER_PDM_SNR_THRESHOLD,
    POST_FILTER_CE_SNR_THRESHOLD,
    POST_FILTER_PDM_METHOD,
    POST_FILTER_PDM_MIN_THETA,
    POST_FILTER_CE_MIN_ENTROPY,
    POST_FILTER_PERIODICITY_SCORE,
    POST_FILTER_COORD_CHUNK_SIZE,
)
from malca.config import PARQUET_CACHE_COMPRESSION, PARQUET_OUTPUT_COMPRESSION
from malca.config import (
    DEFAULT_CACHE_DIR,
    GAIA_AIP_TAP_URL,
    GAIA_LOCAL_CATALOG,
    VSX_CROSSMATCH_PATH,
)
from malca.config import ASASSN_INDEX_PATH
from malca.config import WORKERS, MIN_MAG_OFFSET
from malca.config import PDM_METHOD_CHOICES
from malca.phase import align_v_to_g_magnitude
from malca.stats import compute_pdm_stats, compute_ce_stats
from malca.utils import log_rejections
from malca.utils import read_lc_dat2






# =============================================================================
# Catalog Query Helpers (with caching)
# =============================================================================

DEFAULT_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser()

PERIOD_SOURCE_PRIORITY = (
    "gaia_eb",
    "vsx",
    "asassn_var",
    "ztf_periodic",
    "ogle",
)

PERIOD_HARMONIC_FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)

HOME_ONLY_FILTER_LABELS = (
    "periodic_catalog",
    "gaia_ruwe",
    "gaia_pm",
)

PERIODIC_CATALOG_MERGE_COLS = (
    "catalog_match",
    "catalog_period",
    "catalog_class",
    "catalog_source",
    "period_sources",
    "period_n_sources",
    "period_consensus_days",
    "period_consensus_agree",
    "period_conflict_flag",
    "period_consensus_support",
    "period_primary_source",
    "period_source_periods",
    "period_gaia_eb_match",
    "period_gaia_eb_days",
    "period_gaia_eb_class",
    "period_gaia_eb_sep_arcsec",
    "period_vsx_match",
    "period_vsx_days",
    "period_vsx_class",
    "period_vsx_sep_arcsec",
    "period_asassn_var_match",
    "period_asassn_var_days",
    "period_asassn_var_class",
    "period_asassn_var_sep_arcsec",
    "period_ztf_periodic_match",
    "period_ztf_periodic_days",
    "period_ztf_periodic_class",
    "period_ztf_periodic_sep_arcsec",
    "period_ogle_match",
    "period_ogle_days",
    "period_ogle_class",
    "period_ogle_sep_arcsec",
)

GAIA_RUWE_MERGE_COLS = (
    "ruwe",
    "high_ruwe_flag",
)

GAIA_PM_MERGE_COLS = (
    "pmra",
    "pmdec",
    "pm_total",
    "high_pm_flag",
)

PERIODICITY_MERGE_COLS = (
    "lsp_power",
    "lsp_period",
    "lsp_bootstrap_sig",
    "lsp_is_alias",
    "lsp_is_significant",
    "pdm_method",
    "pdm_period",
    "pdm_theta",
    "pdm_snr",
    "pdm_bootstrap_sig",
    "pdm_is_significant",
    "ce_period",
    "ce_entropy",
    "ce_snr",
    "ce_bootstrap_sig",
    "ce_is_significant",
    "periodicity_bootstrap_sig",
    "periodicity_is_significant",
    "periodicity_score",
    "periodic_flag",
)

HOME_ONLY_CLEAR_DEFAULTS: dict[str, dict[str, object]] = {
    "periodic_catalog": {
        "catalog_match": False,
        "catalog_period": np.nan,
        "catalog_class": "",
        "catalog_source": "",
        "period_sources": "",
        "period_n_sources": 0,
        "period_consensus_days": np.nan,
        "period_consensus_agree": False,
        "period_conflict_flag": False,
        "period_consensus_support": np.nan,
        "period_primary_source": "",
        "period_source_periods": "",
        "period_gaia_eb_match": False,
        "period_gaia_eb_days": np.nan,
        "period_gaia_eb_class": "",
        "period_gaia_eb_sep_arcsec": np.nan,
        "period_vsx_match": False,
        "period_vsx_days": np.nan,
        "period_vsx_class": "",
        "period_vsx_sep_arcsec": np.nan,
        "period_asassn_var_match": False,
        "period_asassn_var_days": np.nan,
        "period_asassn_var_class": "",
        "period_asassn_var_sep_arcsec": np.nan,
        "period_ztf_periodic_match": False,
        "period_ztf_periodic_days": np.nan,
        "period_ztf_periodic_class": "",
        "period_ztf_periodic_sep_arcsec": np.nan,
        "period_ogle_match": False,
        "period_ogle_days": np.nan,
        "period_ogle_class": "",
        "period_ogle_sep_arcsec": np.nan,
    },
    "gaia_ruwe": {
        "ruwe": np.nan,
        "high_ruwe_flag": False,
    },
    "gaia_pm": {
        "pmra": np.nan,
        "pmdec": np.nan,
        "pm_total": np.nan,
        "high_pm_flag": False,
    },
}


def _parse_asassn_id(value: object) -> str | None:
    """Normalize ASAS-SN ID-like values to digit strings."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    return s


def _extract_asassn_ids(df: pd.DataFrame) -> pd.Series:
    """Extract ASAS-SN IDs from asas_sn_id/source_id/path columns."""
    if "asas_sn_id" in df.columns:
        raw = df["asas_sn_id"]
    elif "source_id" in df.columns:
        raw = df["source_id"]
    elif "path" in df.columns:
        raw = df["path"].astype(str).map(lambda p: Path(p).stem)
    else:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    out = raw.astype(str).str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out


def _pick_coord_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Pick available candidate coordinate columns."""
    for ra_col, dec_col in (("ra_deg", "dec_deg"), ("ra", "dec")):
        if ra_col in df.columns and dec_col in df.columns:
            return ra_col, dec_col
    return None, None


def _periods_agree(period_a: float, period_b: float, *, rel_tol: float = POST_FILTER_REL_TOL) -> bool:
    """Return True when periods agree directly or via common harmonics."""
    if not (np.isfinite(period_a) and np.isfinite(period_b)):
        return False
    if period_a <= 0 or period_b <= 0:
        return False
    ratio = max(period_a, period_b) / min(period_a, period_b)
    for factor in PERIOD_HARMONIC_FACTORS:
        if factor <= 0:
            continue
        if abs(ratio - factor) / factor <= rel_tol:
            return True
    return False


def _normalize_period_to_reference(period: float, reference: float) -> float:
    """Map period onto the closest harmonic around reference."""
    if not (np.isfinite(period) and np.isfinite(reference)):
        return period
    if period <= 0 or reference <= 0:
        return period

    candidates = [
        period,
        period / 2.0,
        period * 2.0,
        period / 3.0,
        period * 3.0,
    ]
    best = min(candidates, key=lambda p: abs(math.log10(p) - math.log10(reference)) if p > 0 else np.inf)
    return float(best)


def _choose_consensus_period(
    periods_by_source: dict[str, float],
    *,
    rel_tol: float = POST_FILTER_REL_TOL,
) -> tuple[float, bool, bool, float, str]:
    """Return consensus period + agreement metadata.

    Returns
    -------
    tuple
        (period_consensus_days, period_consensus_agree,
         period_conflict_flag, consensus_support_fraction, period_primary_source)
    """
    valid = {
        src: float(p)
        for src, p in periods_by_source.items()
        if np.isfinite(p) and float(p) > 0
    }
    if not valid:
        return np.nan, False, False, np.nan, ""

    ordered_sources = sorted(
        valid.keys(),
        key=lambda s: PERIOD_SOURCE_PRIORITY.index(s) if s in PERIOD_SOURCE_PRIORITY else len(PERIOD_SOURCE_PRIORITY),
    )
    if len(valid) == 1:
        src = ordered_sources[0]
        return float(valid[src]), True, False, 1.0, src

    best_source = ""
    best_support = -1
    for src in ordered_sources:
        p = valid[src]
        support = sum(_periods_agree(p, q, rel_tol=rel_tol) for q in valid.values())
        if support > best_support:
            best_support = support
            best_source = src

    reference = valid[best_source]
    inlier_sources = [src for src, p in valid.items() if _periods_agree(p, reference, rel_tol=rel_tol)]
    normalized = [_normalize_period_to_reference(valid[src], reference) for src in inlier_sources]
    consensus = float(np.median(normalized)) if normalized else float(reference)

    n_sources = len(valid)
    support_fraction = float(len(inlier_sources) / n_sources) if n_sources else np.nan
    agree = bool(len(inlier_sources) == n_sources)
    conflict = bool((n_sources >= 2) and (not agree))
    return consensus, agree, conflict, support_fraction, best_source


def fetch_chen2020_ztf_periodic(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """
    Fetch Chen+2020 ZTF periodic variable catalog from VizieR.

    VizieR ID: J/ApJS/249/18
    Contains 781,602 periodic variables with periods and classifications.

    Parameters
    ----------
    cache_dir : Path | None
        Directory to cache downloaded catalog (default: ~/.cache/malca/catalogs)
    force_download : bool
        Re-download even if cached file exists
    show_tqdm : bool
        Show progress messages

    Returns
    -------
    pd.DataFrame
        Catalog with columns: ra, dec, period, var_type
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "chen2020_ztf_periodic.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_chen2020] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_chen2020] Querying VizieR J/ApJS/249/18 (this may take a few minutes)...")

    try:
        v = Vizier(columns=["RAJ2000", "DEJ2000", "Per", "Type", "GaiaEDR3"], row_limit=-1)
        tables = v.get_catalogs("J/ApJS/249/18")

        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()

        df = pd.DataFrame({
            "ra": cat["RAJ2000"].astype(float),
            "dec": cat["DEJ2000"].astype(float),
            "period": cat["Per"].astype(float),
            "var_type": cat["Type"].astype(str),
        })

        if "GaiaEDR3" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaEDR3"], errors="coerce").astype("Int64")
        elif "GaiaDR3" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaDR3"], errors="coerce").astype("Int64")
        elif "GaiaDR2" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaDR2"], errors="coerce").astype("Int64")

        df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        if show_tqdm:
            tqdm.write(f"[fetch_chen2020] Cached {len(df)} sources to {cache_file}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch Chen+2020 catalog from VizieR: {e}")


def fetch_asassn_variable_catalog(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch ASAS-SN variable star catalog (VizieR II/366/catv2021)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "asassn_var_ii366.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_asassn_var] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_asassn_var] Querying VizieR II/366/catv2021 (this may take a few minutes)...")

    try:
        v = Vizier(columns=["ASASSN-V", "RAJ2000", "DEJ2000", "Per", "Type", "GaiaDR3"], row_limit=-1)
        tables = v.get_catalogs("II/366/catv2021")
        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()
        df = pd.DataFrame(
            {
                "source_name": cat.get("ASASSN-V", pd.Series(index=cat.index)).astype(str),
                "ra": pd.to_numeric(cat.get("RAJ2000"), errors="coerce"),
                "dec": pd.to_numeric(cat.get("DEJ2000"), errors="coerce"),
                "period": pd.to_numeric(cat.get("Per"), errors="coerce"),
                "var_type": cat.get("Type", pd.Series(index=cat.index)).astype(str),
                "gaia_id": pd.to_numeric(cat.get("GaiaDR3"), errors="coerce").astype("Int64"),
            }
        )

        df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        if show_tqdm:
            tqdm.write(f"[fetch_asassn_var] Cached {len(df)} sources to {cache_file}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ASAS-SN variable catalog from VizieR: {e}")


def fetch_ogle_periodic_catalog(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch OGLE periodic variable catalog (VizieR II/213/pvar)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ogle_ii213_pvar.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_ogle] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_ogle] Querying VizieR II/213/pvar...")

    try:
        v = Vizier(columns=["OGLE", "RAJ2000", "DEJ2000", "Per", "Type"], row_limit=-1)
        tables = v.get_catalogs("II/213/pvar")
        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()
        df = pd.DataFrame(
            {
                "source_name": cat.get("OGLE", pd.Series(index=cat.index)).astype(str),
                "ra": pd.to_numeric(cat.get("RAJ2000"), errors="coerce"),
                "dec": pd.to_numeric(cat.get("DEJ2000"), errors="coerce"),
                "period": pd.to_numeric(cat.get("Per"), errors="coerce"),
                "var_type": cat.get("Type", pd.Series(index=cat.index)).astype(str),
            }
        )

        df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        if show_tqdm:
            tqdm.write(f"[fetch_ogle] Cached {len(df)} sources to {cache_file}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OGLE periodic catalog from VizieR: {e}")


def fetch_vsx_period_catalog(
    vsx_crossmatch_csv: str | Path = VSX_CROSSMATCH_PATH,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Load VSX crossmatch table and expose ASAS-SN keyed periods/classes."""
    path = Path(vsx_crossmatch_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"VSX crossmatch file not found: {path}")

    if show_tqdm:
        tqdm.write(f"[fetch_vsx_period] Loading VSX crossmatch from {path}")

    xmatch = pd.read_csv(path, low_memory=False)
    rename_map: dict[str, str] = {}
    if "sep_arcsec" in xmatch.columns and "vsx_sep_arcsec" not in xmatch.columns:
        rename_map["sep_arcsec"] = "vsx_sep_arcsec"
    if "class" in xmatch.columns and "vsx_class" not in xmatch.columns:
        rename_map["class"] = "vsx_class"
    if rename_map:
        xmatch = xmatch.rename(columns=rename_map)

    required_cols = {"asas_sn_id"}
    missing = [c for c in required_cols if c not in xmatch.columns]
    if missing:
        raise ValueError(f"VSX crossmatch file missing required columns: {missing}")

    keep_cols = [c for c in ["asas_sn_id", "period", "vsx_class", "vsx_sep_arcsec", "gaia_id", "ra", "dec"] if c in xmatch.columns]
    out = xmatch[keep_cols].copy()
    out["asas_sn_id"] = out["asas_sn_id"].astype(str).str.strip()
    if "period" in out.columns:
        out["period"] = pd.to_numeric(out["period"], errors="coerce")
    if "gaia_id" in out.columns:
        out["gaia_id"] = pd.to_numeric(out["gaia_id"], errors="coerce").astype("Int64")

    if "vsx_sep_arcsec" in out.columns:
        out["vsx_sep_arcsec"] = pd.to_numeric(out["vsx_sep_arcsec"], errors="coerce")
        out = out.sort_values("vsx_sep_arcsec", na_position="last").drop_duplicates("asas_sn_id", keep="first")
    else:
        out = out.drop_duplicates("asas_sn_id", keep="first")

    out = out.rename(columns={"vsx_class": "var_type"})
    return out.reset_index(drop=True)


def fetch_gaia_dr3_eb_periods(
    source_ids: list[int] | None,
    *,
    cache_dir: Path | None = None,
    chunk_size: int = 1000,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch Gaia DR3 eclipsing-binary periods for source IDs (cached)."""
    if source_ids is None or len(source_ids) == 0:
        return pd.DataFrame(columns=["source_id", "period", "var_type", "global_ranking"])

    requested_ids = sorted({int(sid) for sid in source_ids})
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gaia_dr3_eb_periods.parquet"

    cached_df = pd.DataFrame(columns=["source_id", "period", "var_type", "global_ranking"])
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if "source_id" in cached_df.columns:
                cached_df["source_id"] = pd.to_numeric(cached_df["source_id"], errors="coerce").astype("Int64")
        except Exception:
            cached_df = pd.DataFrame(columns=["source_id", "period", "var_type", "global_ranking"])

    cached_ids: set[int] = set()
    if "source_id" in cached_df.columns:
        cached_ids = {
            int(v)
            for v in pd.to_numeric(cached_df["source_id"], errors="coerce").dropna().tolist()
        }

    missing_ids = [sid for sid in requested_ids if sid not in cached_ids]
    if show_tqdm and missing_ids:
        tqdm.write(f"[fetch_gaia_eb] Querying Gaia TAP for {len(missing_ids)} uncached source IDs")

    new_rows: list[dict[str, object]] = []
    if missing_ids:
        tap = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)
        chunks = range(0, len(missing_ids), max(1, int(chunk_size)))
        iterator = tqdm(chunks, desc="fetch_gaia_eb", leave=False, disable=not show_tqdm)
        for i in iterator:
            chunk = missing_ids[i : i + max(1, int(chunk_size))]
            ids_str = ",".join(str(sid) for sid in chunk)
            query = f"""
                SELECT source_id, frequency, model_type, global_ranking
                FROM gaiadr3.vari_eclipsing_binary
                WHERE source_id IN ({ids_str})
            """
            attempt = 0
            while True:
                attempt += 1
                try:
                    result = tap.run_sync(query)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    delay = min(5.0 * attempt, 60.0)
                    if show_tqdm:
                        msg = str(e).splitlines()[0].strip()
                        tqdm.write(
                            f"[fetch_gaia_eb] chunk query failed on attempt {attempt}; "
                            f"retrying in {delay:.0f}s: {msg}"
                        )
                    time.sleep(delay)

            for row in result:
                sid = row["source_id"]
                freq = row["frequency"]
                period = np.nan
                if freq is not None:
                    try:
                        fv = float(freq)
                        if np.isfinite(fv) and fv > 0:
                            period = 1.0 / fv
                    except Exception:
                        period = np.nan

                new_rows.append(
                    {
                        "source_id": int(sid),
                        "period": period,
                        "var_type": str(row["model_type"]) if row["model_type"] is not None else "",
                        "global_ranking": float(row["global_ranking"]) if row["global_ranking"] is not None else np.nan,
                    }
                )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        full_df = pd.concat([cached_df, new_df], ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["source_id"], keep="last")
        full_df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
    else:
        full_df = cached_df

    if full_df.empty:
        return full_df

    full_df["source_id"] = pd.to_numeric(full_df["source_id"], errors="coerce").astype("Int64")
    keep = full_df["source_id"].isin(requested_ids)
    return full_df.loc[keep].reset_index(drop=True)


def fetch_gaia_dr3_ruwe(
    source_ids: list[int] | None = None,
    show_tqdm: bool = True,
    **_kwargs,
) -> pd.DataFrame:
    """
    Look up Gaia DR3 RUWE values from the local Gaia catalog.

    The catalog is produced by ``malca gaia-fetch``.  No network call is made.

    Parameters
    ----------
    source_ids : list[int] | None
        Gaia source IDs to look up
    show_tqdm : bool
        Show progress messages

    Returns
    -------
    pd.DataFrame
        Subset with columns: source_id, ruwe, and optional astrometry columns
        available in the local cache (ra, dec, pmra, pmdec).
    """
    if source_ids is None or len(source_ids) == 0:
        raise ValueError("Must provide source_ids")

    catalog_path = GAIA_LOCAL_CATALOG if GAIA_LOCAL_CATALOG.exists() else None
    if catalog_path is None:
        raise FileNotFoundError(
            "Local Gaia catalog not found. Run:\n"
            "  malca gaia-fetch --input <your_candidates.parquet>\n"
            "to download Gaia DR3 data before running filter RUWE validation."
        )

    if show_tqdm:
        tqdm.write(f"[fetch_gaia_dr3_ruwe] Loading local Gaia catalog from {catalog_path}")

    gaia_df = pd.read_parquet(catalog_path)
    if "source_id" not in gaia_df.columns or "ruwe" not in gaia_df.columns:
        raise ValueError(f"Local Gaia catalog at {catalog_path} missing required columns (source_id, ruwe).")

    gaia_df["source_id"] = gaia_df["source_id"].astype(int)
    requested_ids = set(int(sid) for sid in source_ids)
    optional_cols = [c for c in ("ra", "dec", "pmra", "pmdec") if c in gaia_df.columns]
    selected_cols = ["source_id", "ruwe"] + optional_cols
    result_df = gaia_df[gaia_df["source_id"].isin(requested_ids)][selected_cols].copy()

    if show_tqdm:
        tqdm.write(f"[fetch_gaia_dr3_ruwe] Matched {len(result_df)}/{len(requested_ids)} sources from local catalog")

    return result_df.reset_index(drop=True)


def _parse_gaia_id_int(value: object) -> int | None:
    """Parse Gaia source ID-like values to int when possible."""
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return None

    if d != d.to_integral_value():
        return None

    try:
        return int(d)
    except Exception:
        return None


def _to_bool_mask(series: pd.Series) -> pd.Series:
    """Convert mixed boolean-like values into a pandas bool mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _passing_mask_from_failures(
    df: pd.DataFrame,
    *,
    include_labels: tuple[str, ...] | list[str] | None = None,
    ignore_labels: tuple[str, ...] | list[str] | None = None,
) -> pd.Series:
    """Return rows with no failures in the selected failed_* columns."""
    mask = pd.Series(True, index=df.index, dtype=bool)

    if include_labels is not None:
        failure_cols = [f"failed_{label}" for label in include_labels]
    else:
        ignored = {"failed_any"}
        if ignore_labels is not None:
            ignored.update(f"failed_{label}" for label in ignore_labels)
        failure_cols = [
            col
            for col in df.columns
            if col.startswith("failed_") and col not in ignored
        ]

    for col in failure_cols:
        if col in df.columns:
            mask &= ~_to_bool_mask(df[col])

    return mask


def _clear_annotation_columns(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    defaults: dict[str, object] | None,
) -> pd.DataFrame:
    """Reset annotation columns for rows intentionally skipped this pass."""
    if not defaults:
        return df

    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        if bool(mask.any()):
            out.loc[mask, col] = default
    return out


def _match_period_catalog(
    df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    *,
    source_label: str,
    max_sep_arcsec: float,
    period_col: str = "period",
    class_col: str = "var_type",
    gaia_col: str = "gaia_id",
    ra_col: str = "ra",
    dec_col: str = "dec",
    candidate_asassn_ids: pd.Series | None = None,
    catalog_asassn_col: str | None = None,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """Match one catalog to candidates and return per-source period columns."""
    n0 = len(df)
    match = np.zeros(n0, dtype=bool)
    period = np.full(n0, np.nan, dtype=float)
    cls = np.array([""] * n0, dtype=object)
    sep = np.full(n0, np.nan, dtype=float)

    if catalog_df is None or catalog_df.empty:
        return pd.DataFrame(
            {
                f"period_{source_label}_match": match,
                f"period_{source_label}_days": period,
                f"period_{source_label}_class": cls,
                f"period_{source_label}_sep_arcsec": sep,
            },
            index=df.index,
        )

    cat = catalog_df.copy()
    if period_col in cat.columns:
        cat[period_col] = pd.to_numeric(cat[period_col], errors="coerce")
    else:
        cat[period_col] = np.nan
    if class_col in cat.columns:
        cat[class_col] = cat[class_col].fillna("").astype(str)
    else:
        cat[class_col] = ""

    # 1) ID-level match by Gaia source_id (preferred)
    if gaia_col in cat.columns and "gaia_id" in df.columns:
        cand_gaia = pd.Series([_parse_gaia_id_int(v) for v in df["gaia_id"].tolist()], index=df.index, dtype="object")
        cat_gaia = pd.Series([_parse_gaia_id_int(v) for v in cat[gaia_col].tolist()], index=cat.index, dtype="object")
        cat_valid = cat.loc[cat_gaia.notna()].copy()
        if not cat_valid.empty:
            cat_valid["_gaia_id"] = cat_gaia.loc[cat_valid.index].astype(int)
            if "vsx_sep_arcsec" in cat_valid.columns:
                cat_valid["vsx_sep_arcsec"] = pd.to_numeric(cat_valid["vsx_sep_arcsec"], errors="coerce")
                cat_valid = cat_valid.sort_values("vsx_sep_arcsec", na_position="last")
            cat_valid = cat_valid.drop_duplicates(subset=["_gaia_id"], keep="first").set_index("_gaia_id")

            mapped_period = cand_gaia.map(cat_valid[period_col])
            mapped_class = cand_gaia.map(cat_valid[class_col]).fillna("")
            valid_period = mapped_period.notna() & np.isfinite(mapped_period.to_numpy(dtype=float)) & (mapped_period.to_numpy(dtype=float) > 0)
            if valid_period.any():
                match[valid_period.to_numpy()] = True
                period[valid_period.to_numpy()] = mapped_period.loc[valid_period].to_numpy(dtype=float)
                cls[valid_period.to_numpy()] = mapped_class.loc[valid_period].astype(str).to_numpy()
                sep[valid_period.to_numpy()] = 0.0

    # 2) ID-level match by ASAS-SN ID for VSX-like sources
    if catalog_asassn_col and (catalog_asassn_col in cat.columns) and (candidate_asassn_ids is not None):
        cat_asas = cat[catalog_asassn_col].astype(str).str.strip()
        cat_valid = cat.loc[cat_asas.notna() & cat_asas.ne("")].copy()
        if not cat_valid.empty:
            cat_valid["_asas_id"] = cat_asas.loc[cat_valid.index]
            if "vsx_sep_arcsec" in cat_valid.columns:
                cat_valid["vsx_sep_arcsec"] = pd.to_numeric(cat_valid["vsx_sep_arcsec"], errors="coerce")
                cat_valid = cat_valid.sort_values("vsx_sep_arcsec", na_position="last")
            cat_valid = cat_valid.drop_duplicates(subset=["_asas_id"], keep="first").set_index("_asas_id")

            mapped_period = candidate_asassn_ids.map(cat_valid[period_col])
            mapped_class = candidate_asassn_ids.map(cat_valid[class_col]).fillna("")
            mapped_sep = (
                candidate_asassn_ids.map(cat_valid["vsx_sep_arcsec"])
                if "vsx_sep_arcsec" in cat_valid.columns
                else pd.Series(np.nan, index=df.index)
            )

            valid_period = mapped_period.notna() & np.isfinite(mapped_period.to_numpy(dtype=float)) & (mapped_period.to_numpy(dtype=float) > 0)
            if valid_period.any():
                idx_mask = valid_period.to_numpy() & (~match)
                match[idx_mask] = True
                period[idx_mask] = mapped_period.loc[idx_mask].to_numpy(dtype=float)
                cls[idx_mask] = mapped_class.loc[idx_mask].astype(str).to_numpy()
                sep[idx_mask] = pd.to_numeric(mapped_sep.loc[idx_mask], errors="coerce").to_numpy(dtype=float)

    # 3) Coordinate fallback for remaining unmatched rows
    ra_cand_col, dec_cand_col = _pick_coord_columns(df)
    if ra_cand_col is not None and dec_cand_col is not None and ra_col in cat.columns and dec_col in cat.columns:
        remaining = ~match
        cand_ra = pd.to_numeric(df[ra_cand_col], errors="coerce").to_numpy(dtype=float)
        cand_dec = pd.to_numeric(df[dec_cand_col], errors="coerce").to_numpy(dtype=float)
        valid_cand = remaining & np.isfinite(cand_ra) & np.isfinite(cand_dec)

        cat_ra = pd.to_numeric(cat[ra_col], errors="coerce").to_numpy(dtype=float)
        cat_dec = pd.to_numeric(cat[dec_col], errors="coerce").to_numpy(dtype=float)
        cat_period = pd.to_numeric(cat[period_col], errors="coerce").to_numpy(dtype=float)
        cat_class = cat[class_col].astype(str).to_numpy(dtype=object)

        valid_cat = np.isfinite(cat_ra) & np.isfinite(cat_dec) & np.isfinite(cat_period) & (cat_period > 0)
        if valid_cand.any() and valid_cat.any():
            cat_coords = SkyCoord(ra=cat_ra[valid_cat] * u.deg, dec=cat_dec[valid_cat] * u.deg)
            cat_period_valid = cat_period[valid_cat]
            cat_class_valid = cat_class[valid_cat]

            cand_indices = np.flatnonzero(valid_cand)
            chunk_size = POST_FILTER_COORD_CHUNK_SIZE
            iterator = range(0, len(cand_indices), chunk_size)
            if show_tqdm and len(cand_indices) > chunk_size:
                iterator = tqdm(iterator, desc=f"match_{source_label}_coords", leave=False)
            for start in iterator:
                sub_idx = cand_indices[start : start + chunk_size]
                cand_coords = SkyCoord(ra=cand_ra[sub_idx] * u.deg, dec=cand_dec[sub_idx] * u.deg)
                idx_cat, sep2d, _ = cand_coords.match_to_catalog_sky(cat_coords)
                sep_arcsec = sep2d.to(u.arcsec).value
                within = sep_arcsec <= float(max_sep_arcsec)
                if not np.any(within):
                    continue

                out_idx = sub_idx[within]
                src_idx = idx_cat[within]
                match[out_idx] = True
                period[out_idx] = cat_period_valid[src_idx]
                cls[out_idx] = cat_class_valid[src_idx]
                sep[out_idx] = sep_arcsec[within]

    return pd.DataFrame(
        {
            f"period_{source_label}_match": match,
            f"period_{source_label}_days": period,
            f"period_{source_label}_class": cls,
            f"period_{source_label}_sep_arcsec": sep,
        },
        index=df.index,
    )


def filter_evidence_strength(
    df: pd.DataFrame,
    *,
    min_bayes_factor: float = MIN_BAYES_FACTOR,
    require_finite_local_bf: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_bayes_factor or jump_bayes_factor > threshold.
    Optionally require dip_max_log_bf_local or jump_max_log_bf_local to be finite.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_evidence_strength", leave=False) if show_tqdm else None

    # At least one of dip or jump BF must exceed threshold
    mask = (df["dip_bayes_factor"].fillna(0) > min_bayes_factor) | \
           (df["jump_bayes_factor"].fillna(0) > min_bayes_factor)

    # Require finite local BF if requested
    if require_finite_local_bf:
        is_finite_dip = df["dip_max_log_bf_local"].notna() & np.isfinite(df["dip_max_log_bf_local"])
        is_finite_jump = df["jump_max_log_bf_local"].notna() & np.isfinite(df["jump_max_log_bf_local"])
        mask &= (is_finite_dip | is_finite_jump)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_evidence_strength] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_evidence_strength", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out



# =============================================================================
# Filter 7.5: Signal amplitude
# =============================================================================

def filter_signal_amplitude(
    df: pd.DataFrame,
    *,
    min_mag_offset: float = MIN_MAG_OFFSET,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Enforce |best_mag_event - baseline_mag| > threshold.

    This filter requires baseline_mag and event best-magnitude columns from
    events.py output. It is useful as a lightweight post-events gate.
    """
    n0 = len(df)
    required_cols = {"baseline_mag", "dip_best_mag_event", "jump_best_mag_event"}
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        if verbose:
            tqdm.write(
                "[filter_signal_amplitude] WARNING: missing columns "
                f"{missing}; skipping signal amplitude filter"
            )
        return df.copy()

    pbar = tqdm(total=2, desc="filter_signal_amplitude", leave=False) if show_tqdm else None

    dip_diff = np.abs(df["dip_best_mag_event"] - df["baseline_mag"])
    jump_diff = np.abs(df["jump_best_mag_event"] - df["baseline_mag"])
    mask = (dip_diff > min_mag_offset) | (jump_diff > min_mag_offset)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_signal_amplitude] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_signal_amplitude", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out



# =============================================================================
# Filter 9: Run robustness
# =============================================================================

def filter_run_robustness(
    df: pd.DataFrame,
    *,
    min_run_count: int = 1,
    max_run_count: int | None = None,
    min_run_points: int = 2,
    min_run_cameras: int = 2,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_run_count or jump_run_count in [min_run_count, max_run_count] (if max set).
    Require dip_max_run_points or jump_max_run_points >= min_run_points.
    Require dip_max_run_cameras or jump_max_run_cameras >= min_run_cameras.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_run_robustness", leave=False) if show_tqdm else None

    # Check run counts
    dip_counts = pd.to_numeric(df["dip_run_count"], errors="coerce").fillna(0)
    jump_counts = pd.to_numeric(df["jump_run_count"], errors="coerce").fillna(0)
    dip_count_ok = dip_counts >= min_run_count
    jump_count_ok = jump_counts >= min_run_count
    if max_run_count is not None:
        dip_count_ok &= dip_counts <= int(max_run_count)
        jump_count_ok &= jump_counts <= int(max_run_count)

    # Check run points
    dip_points_ok = df["dip_max_run_points"].fillna(0) >= min_run_points
    jump_points_ok = df["jump_max_run_points"].fillna(0) >= min_run_points

    # Check run cameras
    dip_cams_ok = df["dip_max_run_cameras"].fillna(0) >= min_run_cameras
    jump_cams_ok = df["jump_max_run_cameras"].fillna(0) >= min_run_cameras

    dip_ok = dip_count_ok & dip_points_ok & dip_cams_ok
    jump_ok = jump_count_ok & jump_points_ok & jump_cams_ok

    mask = dip_ok | jump_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_run_robustness] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_run_robustness", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 8.5: Significant run/peak gate
# =============================================================================

def filter_significant_detection(
    df: pd.DataFrame,
    *,
    require_significant_flag: bool = True,
    min_peak_count: int = 1,
    min_run_count: int = 1,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require at least one branch (dip or jump) to pass explicit significant detection gates.

    A branch passes when:
    - run_count >= min_run_count
    - peak_count >= min_peak_count
    - (optionally) corresponding *_significant flag is True
    """
    n0 = len(df)

    required_cols = {
        "dip_run_count", "jump_run_count",
        "dip_count", "jump_count",
    }
    if require_significant_flag:
        required_cols.update({"dip_significant", "jump_significant"})

    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        if verbose:
            tqdm.write(
                "[filter_significant_detection] WARNING: missing columns "
                f"{missing}; skipping significant detection gate"
            )
        return df.copy()

    dip_runs = pd.to_numeric(df["dip_run_count"], errors="coerce").fillna(0)
    jump_runs = pd.to_numeric(df["jump_run_count"], errors="coerce").fillna(0)
    dip_peaks = pd.to_numeric(df["dip_count"], errors="coerce").fillna(0)
    jump_peaks = pd.to_numeric(df["jump_count"], errors="coerce").fillna(0)

    dip_ok = (dip_runs >= int(min_run_count)) & (dip_peaks >= int(min_peak_count))
    jump_ok = (jump_runs >= int(min_run_count)) & (jump_peaks >= int(min_peak_count))

    if require_significant_flag:
        dip_ok &= _to_bool_mask(df["dip_significant"])
        jump_ok &= _to_bool_mask(df["jump_significant"])

    mask = dip_ok | jump_ok
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_significant_detection] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_significant_detection", rejected_log_csv)
    return out


# =============================================================================
# Filter 10: Morphology
# =============================================================================

def filter_morphology(
    df: pd.DataFrame,
    *,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = POST_FILTER_MIN_DELTA_BIC,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep runs whose best morphology is 'gaussian' for dips or 'paczynski' for jumps,
    with dip_best_delta_bic/jump_best_delta_bic >= threshold to reject noise-like runs.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_morphology", leave=False) if show_tqdm else None

    # Check morphology for dips
    dip_morph_ok = (df["dip_best_morph"].fillna("").str.lower() == dip_morphology.lower()) & \
                   (df["dip_best_delta_bic"].fillna(0) >= min_delta_bic)

    # Check morphology for jumps
    jump_morph_ok = (df["jump_best_morph"].fillna("").str.lower() == jump_morphology.lower()) & \
                    (df["jump_best_delta_bic"].fillna(0) >= min_delta_bic)

    mask = dip_morph_ok | jump_morph_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_morphology] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_morphology", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 10: Event score
# =============================================================================

def filter_score(
    df: pd.DataFrame,
    *,
    min_dip_score: float | None = None,
    min_jump_score: float | None = None,
    min_score: float | None = None,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dipper/jumper score thresholds with branch-aware limits.

    If ``min_score`` is provided, it is used as a legacy fallback for both
    branches unless branch-specific thresholds are also provided.

    Parameters
    ----------
    min_dip_score : float | None
        Minimum dipper_score threshold.
    min_jump_score : float | None
        Minimum jumper_score threshold.
    min_score : float | None
        Legacy threshold applied to both branches when branch-specific
        thresholds are not provided.
    """
    n0 = len(df)

    if min_score is not None:
        if min_dip_score is None:
            min_dip_score = float(min_score)
        if min_jump_score is None:
            min_jump_score = float(min_score)

    if min_dip_score is None:
        min_dip_score = POST_FILTER_MIN_DIP_SCORE
    if min_jump_score is None:
        min_jump_score = POST_FILTER_MIN_JUMP_SCORE

    has_dip = "dipper_score" in df.columns
    has_jump = "jumper_score" in df.columns
    if not has_dip and not has_jump:
        if verbose:
            tqdm.write("[filter_score] WARNING: score columns missing, skipping filter")
        return df.copy()

    dip_ok = pd.Series(False, index=df.index)
    jump_ok = pd.Series(False, index=df.index)
    if has_dip:
        dip_ok = pd.to_numeric(df["dipper_score"], errors="coerce").fillna(-np.inf) >= float(min_dip_score)
    if has_jump:
        jump_ok = pd.to_numeric(df["jumper_score"], errors="coerce").fillna(-np.inf) >= float(min_jump_score)

    mask = dip_ok | jump_ok
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_score] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_score", rejected_log_csv)

    return out


# =============================================================================
# Validation filters (expensive checks, run after event detection)
# =============================================================================

RAW_STATS_COLUMNS = [
    "camera",
    "median",
    "sig1_low",
    "sig1_high",
    "p90_low",
    "p90_high",
]


def _parse_mag_bin_range(mag_bin: str | None) -> tuple[float, float] | None:
    if not mag_bin:
        return None
    token = mag_bin.strip().replace("-", "_")
    parts = token.split("_")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _mag_bin_range_from_path(path: Path) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?_\d+(?:\.\d+)?)", str(path))
    if not match:
        return None
    return _parse_mag_bin_range(match.group(1))


def _find_raw_stats_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".raw2":
        return path
    return path.with_suffix(".raw2")


def _read_raw_camera_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=RAW_STATS_COLUMNS,
        comment="#",
        header=None,
    )
    for col in ("median", "sig1_low", "sig1_high", "p90_low", "p90_high"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["median"].notna()].reset_index(drop=True)
    return df


def _is_periodic_by_snr(pdm_snr: float, ce_snr: float) -> bool:
    try:
        pdm_val = float(pdm_snr)
        ce_val = float(ce_snr)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(pdm_val) or not np.isfinite(ce_val):
        return False
    return (
        pdm_val >= float(POST_FILTER_PDM_SNR_THRESHOLD)
        and ce_val >= float(POST_FILTER_CE_SNR_THRESHOLD)
    )


def _finite_float(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _candidate_lc_filenames(row: pd.Series | dict[str, object]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for key in ("path", "lc_path"):
        raw = row.get(key) if isinstance(row, dict) else row.get(key)
        text = str(raw or "").strip()
        if not text:
            continue
        candidate = WorkerPath(text).expanduser()
        for name in (candidate.name, candidate.with_suffix(".raw2").name if candidate.suffix in (".dat", ".dat2", ".dat3") else None):
            if name and name not in seen:
                seen.add(name)
                names.append(name)

    for key in ("candidate_id", "asas_sn_id"):
        raw = row.get(key) if isinstance(row, dict) else row.get(key)
        text = str(raw or "").strip()
        if not text:
            continue
        for ext in (".dat3", ".raw2", ".dat2", ".dat"):
            name = f"{text}{ext}"
            if name not in seen:
                seen.add(name)
                names.append(name)

    return names


def _resolve_periodicity_lightcurve_path(
    row: pd.Series | dict[str, object],
    lightcurve_bundle_dir: Path | None,
) -> Path | None:
    for key in ("lc_path", "path"):
        raw = row.get(key) if isinstance(row, dict) else row.get(key)
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            candidate = WorkerPath(text).expanduser()
            if candidate.exists():
                return candidate
        except Exception:
            continue

    if lightcurve_bundle_dir is None or not lightcurve_bundle_dir.exists():
        return None

    for name in _candidate_lc_filenames(row):
        candidate = lightcurve_bundle_dir / name
        if candidate.exists():
            return candidate
    return None


def _checkpoint_result_is_usable(
    result: dict[str, object],
    row: pd.Series,
    *,
    skip_if_consensus: bool,
    expected_pdm_method: str,
) -> bool:
    if not isinstance(result, dict) or not result:
        return False

    if str(result.get("error") or "").strip():
        return False

    catalog_match = False
    if skip_if_consensus and "catalog_match" in row.index:
        catalog_match = _to_bool_mask(pd.Series([row.get("catalog_match")]))[0]
    catalog_period = _finite_float(row.get("catalog_period"))
    if catalog_match and catalog_period is not None and catalog_period > 0:
        return _finite_float(result.get("lsp_period")) is not None

    cached_pdm_method = str(result.get("pdm_method") or "").strip().lower()
    if cached_pdm_method != str(expected_pdm_method).strip().lower():
        return False

    required_cols = (
        "pdm_period",
        "pdm_min_theta",
        "pdm_snr",
        "ce_period",
        "ce_min_entropy",
        "ce_snr",
        "periodicity_is_rejected",
    )
    if any(col not in result for col in required_cols):
        return False

    metric_cols = (
        "pdm_period",
        "pdm_min_theta",
        "pdm_snr",
        "pdm_bootstrap_sig",
        "ce_period",
        "ce_min_entropy",
        "ce_snr",
        "ce_bootstrap_sig",
        "periodicity_bootstrap_sig",
        "lsp_period",
        "lsp_bootstrap_sig",
    )
    if any(_finite_float(result.get(col)) is not None for col in metric_cols):
        return True

    n_points = _finite_float(row.get("n_points"))
    if n_points is not None and n_points < 50:
        return True

    return False


def _lsp_worker(args: tuple) -> dict:
    """
    Worker function for parallel periodicity computation (PDM + CE).

    Args:
        args: Tuple of (path_str, n_bootstrap, significance_level, exclude_alias_periods)

    Returns:
        Dict with path and periodicity results
    """

    if len(args) == 6:
        original_path, path_str, n_bootstrap, significance_level, exclude_alias_periods, pdm_method = args
    elif len(args) == 5:
        original_path, path_str, n_bootstrap, significance_level, exclude_alias_periods = args
        pdm_method = POST_FILTER_PDM_METHOD
    else:
        path_str, n_bootstrap, significance_level, exclude_alias_periods = args
        original_path = path_str
        pdm_method = POST_FILTER_PDM_METHOD
    _ = exclude_alias_periods

    try:
        path = WorkerPath(path_str)
        asassn_id = path.stem
        dir_path = str(path.parent)

        dfg, dfv = read_lc_dat2(asassn_id, dir_path)
        df_lc = pd.concat([dfg, dfv], ignore_index=True)
        df_lc_aligned, _ = align_v_to_g_magnitude(df_lc)

        jd = df_lc_aligned["JD"].values
        mag = df_lc_aligned["mag"].values
        err = df_lc_aligned["error"].values

        # PDM
        pdm_result = compute_pdm_stats(
            jd,
            mag,
            err,
            pdm_method=str(pdm_method),
            n_bootstrap=n_bootstrap,
            significance_level=significance_level,
        )

        # CE
        ce_result = compute_ce_stats(
            jd,
            mag,
            err,
            n_bootstrap=n_bootstrap,
            significance_level=significance_level,
        )

        # Decide if rejected (either method hits threshold)
        pdm_rej = (pdm_result["pdm_snr"] >= POST_FILTER_PDM_SNR_THRESHOLD) and (pdm_result["pdm_min_theta"] <= POST_FILTER_PDM_MIN_THETA)
        ce_rej = (ce_result["ce_snr"] >= POST_FILTER_CE_SNR_THRESHOLD) and (ce_result["ce_min_entropy"] <= POST_FILTER_CE_MIN_ENTROPY)

        pdm_boot_sig = pdm_result.get("pdm_bootstrap_sig", np.nan)
        ce_boot_sig = ce_result.get("ce_bootstrap_sig", np.nan)
        boot_sigs = np.array([pdm_boot_sig, ce_boot_sig], dtype=float)
        finite_boot = boot_sigs[np.isfinite(boot_sigs)]
        if finite_boot.size > 0:
            periodicity_bootstrap_sig = float(np.min(finite_boot))
            periodicity_is_significant = bool(periodicity_bootstrap_sig < float(significance_level))
        else:
            periodicity_bootstrap_sig = np.nan
            periodicity_is_significant = False

        is_rejected = pdm_rej or ce_rej

        best_period = pdm_result.get("pdm_period", np.nan)
        if np.isfinite(ce_boot_sig) and (not np.isfinite(pdm_boot_sig) or ce_boot_sig < pdm_boot_sig):
            best_period = ce_result.get("ce_period", np.nan)

        return {
            "path": original_path,
            "resolved_path": path_str,
            "lsp_power": np.nan,
            "lsp_period": best_period,
            "lsp_bootstrap_sig": periodicity_bootstrap_sig,
            "lsp_is_alias": False,
            "lsp_is_significant": periodicity_is_significant,
            "pdm_method": str(pdm_method),
            "pdm_period": pdm_result["pdm_period"],
            "pdm_min_theta": pdm_result["pdm_min_theta"],
            "pdm_snr": pdm_result["pdm_snr"],
            "pdm_bootstrap_sig": pdm_result.get("pdm_bootstrap_sig", np.nan),
            "pdm_is_significant": bool(pdm_result.get("pdm_is_significant", False)),
            "ce_period": ce_result["ce_period"],
            "ce_min_entropy": ce_result["ce_min_entropy"],
            "ce_snr": ce_result["ce_snr"],
            "ce_bootstrap_sig": ce_result.get("ce_bootstrap_sig", np.nan),
            "ce_is_significant": bool(ce_result.get("ce_is_significant", False)),
            "periodicity_bootstrap_sig": periodicity_bootstrap_sig,
            "periodicity_is_significant": periodicity_is_significant,
            "periodicity_is_rejected": is_rejected,
            "error": None,
        }
    except Exception as e:
        return {
            "path": original_path,
            "resolved_path": path_str,
            "pdm_method": str(pdm_method),
            "pdm_period": np.nan,
            "pdm_min_theta": np.nan,
            "pdm_snr": np.nan,
            "pdm_bootstrap_sig": np.nan,
            "pdm_is_significant": False,
            "ce_period": np.nan,
            "ce_min_entropy": np.nan,
            "ce_snr": np.nan,
            "ce_bootstrap_sig": np.nan,
            "ce_is_significant": False,
            "periodicity_bootstrap_sig": np.nan,
            "periodicity_is_significant": False,
            "periodicity_is_rejected": False,
            "error": str(e),
        }


def validate_periodicity(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    significance_level: float = 0.01,
    pdm_method: str = POST_FILTER_PDM_METHOD,
    exclude_alias_periods: bool = True,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
    workers: int = 1,
    checkpoint_dir: str | Path | None = None,
    skip_if_consensus: bool = True,
    lightcurve_bundle_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Detailed periodicity validation on candidates using PDM + CE.

    Uses Phase Dispersion Minimization and Conditional Entropy to identify:
    - Eclipsing binaries (short periods ~1 day)
    - Rotating variables (periods ~30 days)
    - Other periodic contamination

    Only run on detected candidates, not all sources.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates from events.py (must have 'path' column)
    n_bootstrap : int
        Bootstrap shuffles used for both PDM and CE significance.
    significance_level : float
        Bootstrap significance threshold (lower is more significant).
    exclude_alias_periods : bool
        Present for API compatibility (not applied to PDM/CE).
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates
    workers : int
        Number of parallel workers (default 1 = sequential)
    checkpoint_dir : str | Path | None
        Directory for checkpoint files (enables resume on restart)
    skip_if_consensus : bool
        Skip if a consensus period is already found in external catalogs (default True)
    lightcurve_bundle_dir : str | Path | None
        Optional local bundle directory used to resolve light curves when the
        parquet still points at cluster paths that are unavailable locally.

    Returns
    -------
    pd.DataFrame
        Candidates without strong periodic signals
    """



    n0 = len(df)
    paths = [str(p) for p in df["path"].astype(str).tolist()]
    bundle_dir = None
    if lightcurve_bundle_dir is not None:
        try:
            candidate = Path(lightcurve_bundle_dir).expanduser().resolve()
        except Exception:
            candidate = Path(lightcurve_bundle_dir).expanduser()
        if candidate.exists():
            bundle_dir = candidate

    # Checkpoint handling
    checkpoint_file = None
    completed_results: dict[str, dict[str, object]] = {}
    
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path / "lsp_checkpoint.parquet"
        
        # Load existing checkpoint if present
        if checkpoint_file.exists():
            try:
                checkpoint_df = pd.read_parquet(checkpoint_file)
                completed_results = {
                    str(row["path"]): row.to_dict()
                    for _, row in checkpoint_df.iterrows()
                    if str(row.get("path", "")).strip()
                }
                if show_tqdm:
                    tqdm.write(f"[validate_periodicity] Loaded {len(completed_results)} cached results from checkpoint")
            except Exception as e:
                if show_tqdm:
                    tqdm.write(f"[validate_periodicity] Warning: Could not load checkpoint: {e}")
    
    # Filter to rows not already processed
    worker_args: list[tuple[str, str, int, float, bool, str]] = []
    skipped_consensus: dict[str, dict[str, object]] = {}
    prefilled_errors: list[dict[str, object]] = []

    has_consensus = False
    if skip_if_consensus and "catalog_match" in df.columns:
        # Check if catalog_match is true (implies consensus/evidence found)
        # We can also check period_consensus_agree if we want to be stricter
        has_consensus = True

    if has_consensus:
        # Pre-fill results for consensus matches
        for _, row in df.iterrows():
            p = str(row["path"])

            # Use loose consensus check: any catalog match is treated as valid period evidence
            # to skip the expensive bootstrap check.
            if _to_bool_mask(pd.Series([row["catalog_match"]]))[0]:
                period = _finite_float(row.get("catalog_period"))
                if period is not None and period > 0:
                    skipped_consensus[p] = {
                        "path": p,
                        "resolved_path": None,
                        "lsp_power": np.nan,  # Not computed
                        "lsp_period": period, # Trust catalog period
                        "lsp_bootstrap_sig": 0.0, # Treat as highly significant
                        "lsp_is_alias": False,
                        "lsp_is_significant": True,
                        "pdm_method": str(pdm_method),
                        "periodicity_score": POST_FILTER_PERIODICITY_SCORE,
                        "error": None,
                    }
                    continue

            cached = completed_results.get(p)
            if cached is not None and _checkpoint_result_is_usable(
                cached,
                row,
                skip_if_consensus=skip_if_consensus,
                expected_pdm_method=str(pdm_method),
            ):
                continue
            completed_results.pop(p, None)

            resolved = _resolve_periodicity_lightcurve_path(row, bundle_dir)
            if resolved is None:
                prefilled_errors.append({
                    "path": p,
                    "resolved_path": None,
                    "pdm_method": str(pdm_method),
                    "pdm_period": np.nan,
                    "pdm_min_theta": np.nan,
                    "pdm_snr": np.nan,
                    "pdm_bootstrap_sig": np.nan,
                    "pdm_is_significant": False,
                    "ce_period": np.nan,
                    "ce_min_entropy": np.nan,
                    "ce_snr": np.nan,
                    "ce_bootstrap_sig": np.nan,
                    "ce_is_significant": False,
                    "periodicity_bootstrap_sig": np.nan,
                    "periodicity_is_significant": False,
                    "periodicity_is_rejected": False,
                    "error": f"Light curve file not found for periodicity validation: {p}",
                })
                continue

            worker_args.append((p, str(resolved), n_bootstrap, significance_level, exclude_alias_periods, str(pdm_method)))
    else:
        for _, row in df.iterrows():
            p = str(row["path"])
            cached = completed_results.get(p)
            if cached is not None and _checkpoint_result_is_usable(
                cached,
                row,
                skip_if_consensus=skip_if_consensus,
                expected_pdm_method=str(pdm_method),
            ):
                continue
            completed_results.pop(p, None)

            resolved = _resolve_periodicity_lightcurve_path(row, bundle_dir)
            if resolved is None:
                prefilled_errors.append({
                    "path": p,
                    "resolved_path": None,
                    "pdm_method": str(pdm_method),
                    "pdm_period": np.nan,
                    "pdm_min_theta": np.nan,
                    "pdm_snr": np.nan,
                    "pdm_bootstrap_sig": np.nan,
                    "pdm_is_significant": False,
                    "ce_period": np.nan,
                    "ce_min_entropy": np.nan,
                    "ce_snr": np.nan,
                    "ce_bootstrap_sig": np.nan,
                    "ce_is_significant": False,
                    "periodicity_bootstrap_sig": np.nan,
                    "periodicity_is_significant": False,
                    "periodicity_is_rejected": False,
                    "error": f"Light curve file not found for periodicity validation: {p}",
                })
                continue

            worker_args.append((p, str(resolved), n_bootstrap, significance_level, exclude_alias_periods, str(pdm_method)))

    if show_tqdm:
        n_cached = len(paths) - len(worker_args) - len(skipped_consensus) - len(prefilled_errors)
        msg = f"[validate_periodicity] {n_cached} cached"
        if skipped_consensus:
            msg += f", {len(skipped_consensus)} skipped (consensus)"
        if prefilled_errors:
            msg += f", {len(prefilled_errors)} unresolved"
        msg += f", processing {len(worker_args)}"
        tqdm.write(msg)

    # Process with multiprocessing or sequential based on workers
    new_results = list(prefilled_errors)
    n_errors = len(prefilled_errors)

    if workers > 1 and len(worker_args) > 0:
        # Parallel execution
        actual_workers = min(workers, cpu_count(), len(worker_args))
        chunksize = max(1, len(worker_args) // (actual_workers * 4))

        with Pool(processes=actual_workers, maxtasksperchild=50) as pool:
            iterator = pool.imap_unordered(_lsp_worker, worker_args, chunksize=chunksize)
            if show_tqdm:
                iterator = tqdm(iterator, total=len(worker_args), desc="Periodicity validation")
            
            checkpoint_batch = []
            checkpoint_interval = max(100, len(worker_args) // 20)  # Save every 5%
            
            for result in iterator:
                new_results.append(result)
                if result["error"] is not None:
                    n_errors += 1
                
                # Batch checkpoint saves
                if checkpoint_file is not None:
                    checkpoint_batch.append(result)
                    if len(checkpoint_batch) >= checkpoint_interval:
                        _save_checkpoint(checkpoint_file, completed_results, new_results)
                        checkpoint_batch = []
    else:
        # Sequential execution (workers=1 or no paths to process)
        iterator = worker_args
        if show_tqdm and len(worker_args) > 0:
            iterator = tqdm(worker_args, desc="Periodicity validation")
        
        for args in iterator:
            result = _lsp_worker(args)
            new_results.append(result)
            if result["error"] is not None:
                n_errors += 1
    
    # Final checkpoint save
    if checkpoint_file is not None and new_results:
        _save_checkpoint(checkpoint_file, completed_results, new_results)
        if show_tqdm:
            tqdm.write(f"[validate_periodicity] Saved checkpoint with {len(completed_results) + len(new_results)} entries")
    
    # Combine cached + new results + skipped consensus
    all_results = {**completed_results}
    for r in new_results:
        all_results[r["path"]] = r
    for p, r in skipped_consensus.items():
        all_results[p] = r
    
    # Build output columns
    powers = []
    periods = []
    bootstrap_significances = []
    is_alias = []
    is_significant = []
    
    pdm_methods = []
    pdm_periods = []
    pdm_thetas = []
    pdm_snrs = []
    pdm_bootstrap_significances = []
    pdm_significant_flags = []

    ce_periods = []
    ce_entropies = []
    ce_snrs = []
    ce_bootstrap_significances = []
    ce_significant_flags = []

    periodicity_bootstrap_significances = []
    periodicity_significant_flags = []
    
    periodicity_scores = []
    keep_flags = []
    
    for path_str in paths:
        result = all_results.get(path_str, {})
        powers.append(result.get("lsp_power", np.nan))
        periods.append(result.get("lsp_period", np.nan))
        sig = result.get("periodicity_bootstrap_sig", result.get("lsp_bootstrap_sig", np.nan))
        bootstrap_significances.append(sig)
        alias_flag = result.get("lsp_is_alias", False)
        is_alias.append(alias_flag)
        is_significant.append(bool(result.get("periodicity_is_significant", result.get("lsp_is_significant", False))))
        periodicity_bootstrap_significances.append(sig)
        periodicity_significant_flags.append(bool(result.get("periodicity_is_significant", result.get("lsp_is_significant", False))))

        # New PDM/CE columns
        pdm_methods.append(result.get("pdm_method", str(pdm_method)))
        pdm_periods.append(result.get("pdm_period", np.nan))
        pdm_thetas.append(result.get("pdm_min_theta", np.nan))
        pdm_snrs.append(result.get("pdm_snr", np.nan))
        pdm_bootstrap_significances.append(result.get("pdm_bootstrap_sig", np.nan))
        pdm_significant_flags.append(bool(result.get("pdm_is_significant", False)))

        ce_periods.append(result.get("ce_period", np.nan))
        ce_entropies.append(result.get("ce_min_entropy", np.nan))
        ce_snrs.append(result.get("ce_snr", np.nan))
        ce_bootstrap_significances.append(result.get("ce_bootstrap_sig", np.nan))
        ce_significant_flags.append(bool(result.get("ce_is_significant", False)))

        if np.isfinite(sig):
            min_p = max(1.0 / float(max(n_bootstrap, 1)), 1e-12)
            periodicity_scores.append(float(-np.log10(np.clip(sig, min_p, 1.0))))
        else:
            periodicity_scores.append(np.nan)
        
        # Use the combined rejection flag from the worker
        is_rej = result.get("periodicity_is_rejected", False)
        keep = not is_rej
        keep_flags.append(keep)

    df_out = df.copy()
    df_out["lsp_power"] = powers
    df_out["lsp_period"] = periods
    df_out["lsp_bootstrap_sig"] = bootstrap_significances
    df_out["lsp_is_alias"] = is_alias
    df_out["lsp_is_significant"] = is_significant
    
    df_out["pdm_method"] = pdm_methods
    df_out["pdm_period"] = pdm_periods
    df_out["pdm_theta"] = pdm_thetas
    df_out["pdm_snr"] = pdm_snrs
    df_out["pdm_bootstrap_sig"] = pdm_bootstrap_significances
    df_out["pdm_is_significant"] = pdm_significant_flags

    df_out["ce_period"] = ce_periods
    df_out["ce_entropy"] = ce_entropies
    df_out["ce_snr"] = ce_snrs
    df_out["ce_bootstrap_sig"] = ce_bootstrap_significances
    df_out["ce_is_significant"] = ce_significant_flags

    df_out["periodicity_bootstrap_sig"] = periodicity_bootstrap_significances
    df_out["periodicity_is_significant"] = periodicity_significant_flags
    
    df_out["periodicity_score"] = periodicity_scores

    periodic_flags = [not x for x in keep_flags]
    df_out["periodic_flag"] = periodic_flags

    if flag_only:
        df_filtered = df_out.reset_index(drop=True)
    else:
        df_filtered = df_out[keep_flags].reset_index(drop=True)

    if show_tqdm:
        n_flagged = int(np.sum(periodic_flags))
        if flag_only:
            tqdm.write(f"[validate_periodicity] flagged {n_flagged}/{n0} as periodic")
        else:
            tqdm.write(f"[validate_periodicity] kept {len(df_filtered)}/{n0}")
        if n_errors > 0:
            tqdm.write(f"[validate_periodicity] {n_errors} sources had errors (kept as-is)")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_periodicity", rejected_log_csv)

    return df_filtered


def _save_checkpoint(checkpoint_file: Path, completed: dict, new_results: list) -> None:
    """Save checkpoint to parquet file."""
    all_data = list(completed.values()) + new_results
    clean_data = []
    for r in all_data:
        clean_data.append({
            "path": r["path"],
            "resolved_path": r.get("resolved_path"),
            "lsp_power": r.get("lsp_power", np.nan),
            "lsp_period": r.get("lsp_period", np.nan),
            "lsp_bootstrap_sig": r.get("lsp_bootstrap_sig", np.nan),
            "lsp_is_alias": r.get("lsp_is_alias", False),
            "lsp_is_significant": r.get("lsp_is_significant", False),
            "pdm_method": r.get("pdm_method", str(POST_FILTER_PDM_METHOD)),
            "pdm_period": r.get("pdm_period", np.nan),
            "pdm_min_theta": r.get("pdm_min_theta", np.nan),
            "pdm_snr": r.get("pdm_snr", np.nan),
            "pdm_bootstrap_sig": r.get("pdm_bootstrap_sig", np.nan),
            "pdm_is_significant": r.get("pdm_is_significant", False),
            "ce_period": r.get("ce_period", np.nan),
            "ce_min_entropy": r.get("ce_min_entropy", np.nan),
            "ce_snr": r.get("ce_snr", np.nan),
            "ce_bootstrap_sig": r.get("ce_bootstrap_sig", np.nan),
            "ce_is_significant": r.get("ce_is_significant", False),
            "periodicity_bootstrap_sig": r.get("periodicity_bootstrap_sig", np.nan),
            "periodicity_is_significant": r.get("periodicity_is_significant", False),
            "periodicity_is_rejected": r.get("periodicity_is_rejected", False),
            "error": r.get("error"),
        })
    pd.DataFrame(clean_data).to_parquet(checkpoint_file, index=False, compression=PARQUET_CACHE_COMPRESSION)


def _infer_run_dir_for_periodicity(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    try:
        path = Path(path_like).expanduser().resolve()
    except Exception:
        path = Path(path_like).expanduser()

    candidates = [path]
    if path.is_file():
        candidates.extend([path.parent, path.parent.parent, path.parent.parent.parent])
    else:
        candidates.extend([path.parent, path.parent.parent])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "bundle_assets" / "lightcurves").is_dir():
            return candidate
    return None


def validate_gaia_ruwe(
    df: pd.DataFrame,
    *,
    max_ruwe: float = POST_FILTER_MAX_RUWE,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Validate candidates using Gaia RUWE (Renormalized Unit Weight Error).

    Queries Gaia DR3 via TAP for candidate coordinates.
    RUWE > 1.4 indicates potential companion (binary contamination).
    Paper identifies 5/81 candidates with high RUWE.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates (must have gaia_id column)
    max_ruwe : float
        RUWE threshold (default 1.4, from paper)
    flag_only : bool
        If True, add 'ruwe' and 'high_ruwe_flag' columns but don't reject
        If False, reject sources with RUWE > max_ruwe
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Candidates with RUWE information added

    Notes
    -----
    Paper approach:
    - RUWE ~ 1 consistent with single stars
    - RUWE > 1.4 indicates binarity
    - 5/81 candidates flagged (potential companions)
    - Still need follow-up (imaging, RV) to confirm
    """
    n0 = len(df)

    if "gaia_id" not in df.columns:
        raise ValueError("[validate_gaia_ruwe] Missing gaia_id column")

    # Get unique Gaia IDs (excluding NaN/invalid)
    parsed_ids = [_parse_gaia_id_int(v) for v in df["gaia_id"].tolist()]
    unique_ids = sorted({gid for gid in parsed_ids if gid is not None})

    if not unique_ids:
        if show_tqdm:
            tqdm.write("[validate_gaia_ruwe] No valid Gaia IDs - returning unchanged")
        df_out = df.copy()
        df_out["ruwe"] = np.nan
        df_out["high_ruwe_flag"] = False
        return df_out

    # Fetch RUWE from Gaia TAP by source_id
    if show_tqdm:
        tqdm.write(f"[validate_gaia_ruwe] Querying Gaia TAP for {len(unique_ids)} unique sources...")
    try:
        gaia_df = fetch_gaia_dr3_ruwe(
            source_ids=unique_ids,
            show_tqdm=show_tqdm,
        )
    except Exception as e:
        raise RuntimeError(f"[validate_gaia_ruwe] Gaia RUWE lookup failed: {e}") from e

    found_ids = set(pd.to_numeric(gaia_df.get("source_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    missing_ids = sorted(set(unique_ids) - found_ids)
    if missing_ids:
        preview = ", ".join(str(v) for v in missing_ids[:5])
        more = f" (+{len(missing_ids) - 5} more)" if len(missing_ids) > 5 else ""
        raise RuntimeError(
            f"[validate_gaia_ruwe] Local Gaia catalog is missing RUWE rows for source_id(s): {preview}{more}"
        )

    # Create lookup dict from Gaia results
    ruwe_lookup = dict(zip(gaia_df["source_id"].astype(int), gaia_df["ruwe"]))

    # Map RUWE values to candidates
    ruwes = []
    high_ruwe_flags = []
    for gid in parsed_ids:
        if gid is not None and gid in ruwe_lookup:
            ruwe_val = float(ruwe_lookup[gid])
            ruwes.append(ruwe_val)
            high_ruwe_flags.append(ruwe_val > max_ruwe)
        else:
            ruwes.append(np.nan)
            high_ruwe_flags.append(False)

    df_out = df.copy()
    df_out["ruwe"] = ruwes
    df_out["high_ruwe_flag"] = high_ruwe_flags

    if flag_only:
        df_filtered = df_out
    else:
        df_filtered = df_out[~df_out["high_ruwe_flag"]].reset_index(drop=True)

    if show_tqdm:
        n_flagged = sum(high_ruwe_flags)
        tqdm.write(f"[validate_gaia_ruwe] flagged {n_flagged}/{n0} with RUWE > {max_ruwe}")
        tqdm.write(f"[validate_gaia_ruwe] kept {len(df_filtered)}/{n0}")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_gaia_ruwe", rejected_log_csv)

    return df_filtered


def validate_gaia_proper_motion(
    df: pd.DataFrame,
    *,
    max_pm: float = POST_FILTER_MAX_PM,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Validate candidates using Gaia proper motion magnitude.

    Uses local Gaia cache values for ``pmra``/``pmdec`` (mas/yr), computes
    ``pm_total = sqrt(pmra^2 + pmdec^2)``, and flags or rejects sources above
    ``max_pm``.
    """
    _ = verbose
    n0 = len(df)

    if "gaia_id" not in df.columns:
        raise ValueError("[validate_gaia_proper_motion] Missing gaia_id column")

    df_out = df.copy()
    pmra = pd.to_numeric(df_out["pmra"], errors="coerce") if "pmra" in df_out.columns else pd.Series(np.nan, index=df_out.index, dtype=float)
    pmdec = pd.to_numeric(df_out["pmdec"], errors="coerce") if "pmdec" in df_out.columns else pd.Series(np.nan, index=df_out.index, dtype=float)

    gaia_ids = [_parse_gaia_id_int(v) for v in df_out["gaia_id"].tolist()]
    unique_ids = sorted({gid for gid in gaia_ids if gid is not None})

    if unique_ids:
        if show_tqdm:
            tqdm.write(f"[validate_gaia_proper_motion] Looking up PM for {len(unique_ids)} unique Gaia IDs...")
        try:
            gaia_df = fetch_gaia_dr3_ruwe(source_ids=unique_ids, show_tqdm=show_tqdm)
        except Exception as e:
            raise RuntimeError(f"[validate_gaia_proper_motion] Gaia PM lookup failed: {e}") from e

        if not gaia_df.empty:
            if "pmra" in gaia_df.columns and "pmdec" in gaia_df.columns:
                gaia_df = gaia_df.copy()
                gaia_df["source_id"] = pd.to_numeric(gaia_df["source_id"], errors="coerce")
                gaia_df["pmra"] = pd.to_numeric(gaia_df["pmra"], errors="coerce")
                gaia_df["pmdec"] = pd.to_numeric(gaia_df["pmdec"], errors="coerce")

                pm_lookup: dict[int, tuple[float, float]] = {}
                for _, row in gaia_df.iterrows():
                    sid = row.get("source_id")
                    if pd.isna(sid):
                        continue
                    pm_lookup[int(sid)] = (row.get("pmra", np.nan), row.get("pmdec", np.nan))

                for i, gid in enumerate(gaia_ids):
                    if gid is None:
                        continue
                    vals = pm_lookup.get(gid)
                    if vals is None:
                        continue
                    if pd.isna(pmra.iat[i]) and pd.notna(vals[0]):
                        pmra.iat[i] = float(vals[0])
                    if pd.isna(pmdec.iat[i]) and pd.notna(vals[1]):
                        pmdec.iat[i] = float(vals[1])
            elif show_tqdm:
                tqdm.write("[validate_gaia_proper_motion] Local Gaia cache has no pmra/pmdec columns - using existing PM columns only")
        elif pmra.isna().all() or pmdec.isna().all():
            raise RuntimeError("[validate_gaia_proper_motion] Local Gaia catalog returned no PM rows")
    elif show_tqdm:
        tqdm.write("[validate_gaia_proper_motion] No valid Gaia IDs - using existing PM columns only")

    valid_pm = pmra.notna() & pmdec.notna()
    pm_total = pd.Series(np.nan, index=df_out.index, dtype=float)
    pm_total.loc[valid_pm] = np.sqrt(pmra.loc[valid_pm] ** 2 + pmdec.loc[valid_pm] ** 2)
    high_pm_flags = (pm_total > max_pm).fillna(False)

    df_out["pmra"] = pmra
    df_out["pmdec"] = pmdec
    df_out["pm_total"] = pm_total
    df_out["high_pm_flag"] = high_pm_flags

    if flag_only:
        df_filtered = df_out
    else:
        df_filtered = df_out[~df_out["high_pm_flag"]].reset_index(drop=True)

    if show_tqdm:
        n_flagged = int(high_pm_flags.sum())
        tqdm.write(f"[validate_gaia_proper_motion] flagged {n_flagged}/{n0} with PM > {max_pm} mas/yr")
        tqdm.write(f"[validate_gaia_proper_motion] kept {len(df_filtered)}/{n0}")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_gaia_proper_motion", rejected_log_csv)

    return df_filtered


def validate_periodic_catalog(
    df: pd.DataFrame,
    *,
    max_sep_arcsec: float = POST_FILTER_MAX_SEP_ARCSEC,
    flag_only: bool = True,
    consensus_rel_tol: float = POST_FILTER_REL_TOL,
    use_gaia_eb: bool = True,
    use_asassn_var: bool = True,
    use_ztf_periodic: bool = True,
    use_vsx_period: bool = True,
    use_ogle_periodic: bool = True,
    vsx_crossmatch_csv: str | Path = VSX_CROSSMATCH_PATH,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Aggregate multi-catalog periodic evidence and compute period consensus.

    Evidence sources:
    - Gaia DR3 eclipsing binary table (period from frequency)
    - ASAS-SN variable catalog (II/366)
    - ZTF periodic variables (Chen+2020)
    - VSX periods from the ASAS-SN x VSX crossmatch table
    - OGLE periodic variables (II/213)

    Parameters
    ----------
    df : pd.DataFrame
        Candidate table (gaia_id, asas_sn_id/path, and/or coordinates used if available)
    max_sep_arcsec : float
        Maximum separation for coordinate fallback matches
    flag_only : bool
        If True, annotate only (default). If False, reject any catalog-matched rows.
    consensus_rel_tol : float
        Relative tolerance when checking period agreement/harmonics.
    use_* : bool
        Enable/disable each evidence source.
    vsx_crossmatch_csv : str | Path
        VSX crossmatch source used to recover VSX periods.
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Dataframe annotated with per-source period evidence and consensus fields:
        period_sources, period_n_sources, period_consensus_days,
        period_consensus_agree, period_conflict_flag, catalog_match.
    """
    n0 = len(df)

    df_out = df.copy()
    existing_output_cols = [col for col in PERIODIC_CATALOG_MERGE_COLS if col in df_out.columns]
    if existing_output_cols:
        df_out = df_out.drop(columns=existing_output_cols)
    candidate_asassn_ids = _extract_asassn_ids(df_out)
    source_frames: dict[str, pd.DataFrame] = {}

    def _safe_collect(source_label: str, fn, **kwargs) -> None:
        try:
            source_frames[source_label] = fn(**kwargs)
        except Exception as e:
            if show_tqdm:
                tqdm.write(f"[validate_periodic_catalog] {source_label} lookup failed: {e}")
            raise RuntimeError(f"[validate_periodic_catalog] {source_label} lookup failed: {e}") from e

    # Gaia EB periods
    if use_gaia_eb and "gaia_id" in df_out.columns:
        gaia_ids = [_parse_gaia_id_int(v) for v in df_out["gaia_id"].tolist()]
        gaia_ids = [gid for gid in gaia_ids if gid is not None]
        if gaia_ids:
            _safe_collect("gaia_eb", fetch_gaia_dr3_eb_periods, source_ids=gaia_ids, show_tqdm=show_tqdm)
            if not source_frames["gaia_eb"].empty:
                source_frames["gaia_eb"] = source_frames["gaia_eb"].rename(
                    columns={"source_id": "gaia_id", "global_ranking": "ranking"}
                )

    # ASAS-SN variable catalog
    if use_asassn_var:
        _safe_collect("asassn_var", fetch_asassn_variable_catalog, show_tqdm=show_tqdm)

    # ZTF periodic catalog
    if use_ztf_periodic:
        _safe_collect("ztf_periodic", fetch_chen2020_ztf_periodic, show_tqdm=show_tqdm)

    # VSX periods from crossmatch table
    if use_vsx_period:
        _safe_collect("vsx", fetch_vsx_period_catalog, vsx_crossmatch_csv=vsx_crossmatch_csv, show_tqdm=show_tqdm)

    # OGLE periodic catalog
    if use_ogle_periodic:
        _safe_collect("ogle", fetch_ogle_periodic_catalog, show_tqdm=show_tqdm)

    # Match each source to candidates and attach source columns
    for src in PERIOD_SOURCE_PRIORITY:
        cat_df = source_frames.get(src)
        if cat_df is None:
            continue

        if src == "vsx":
            src_match = _match_period_catalog(
                df_out,
                cat_df,
                source_label=src,
                max_sep_arcsec=max_sep_arcsec,
                period_col="period",
                class_col="var_type",
                gaia_col="gaia_id",
                catalog_asassn_col="asas_sn_id",
                candidate_asassn_ids=candidate_asassn_ids,
                show_tqdm=show_tqdm,
            )
        elif src == "gaia_eb":
            src_match = _match_period_catalog(
                df_out,
                cat_df,
                source_label=src,
                max_sep_arcsec=max_sep_arcsec,
                period_col="period",
                class_col="var_type",
                gaia_col="gaia_id",
                show_tqdm=show_tqdm,
            )
        else:
            src_match = _match_period_catalog(
                df_out,
                cat_df,
                source_label=src,
                max_sep_arcsec=max_sep_arcsec,
                period_col="period",
                class_col="var_type",
                gaia_col="gaia_id",
                show_tqdm=show_tqdm,
            )
        df_out = pd.concat([df_out, src_match], axis=1)

    period_sources_col = np.array([""] * n0, dtype=object)
    period_n_sources_col = np.zeros(n0, dtype=int)
    period_consensus_days_col = np.full(n0, np.nan, dtype=float)
    period_consensus_agree_col = np.zeros(n0, dtype=bool)
    period_conflict_flag_col = np.zeros(n0, dtype=bool)
    period_consensus_support_col = np.full(n0, np.nan, dtype=float)
    period_primary_source_col = np.array([""] * n0, dtype=object)
    period_source_periods_col = np.array([""] * n0, dtype=object)

    catalog_match_col = np.zeros(n0, dtype=bool)
    catalog_period_col = np.full(n0, np.nan, dtype=float)
    catalog_class_col = np.array([""] * n0, dtype=object)
    catalog_source_col = np.array([""] * n0, dtype=object)

    period_cols = {src: f"period_{src}_days" for src in PERIOD_SOURCE_PRIORITY if f"period_{src}_days" in df_out.columns}
    class_cols = {src: f"period_{src}_class" for src in PERIOD_SOURCE_PRIORITY if f"period_{src}_class" in df_out.columns}

    if period_cols:
        has_any_period = np.zeros(n0, dtype=bool)
        period_arrays: dict[str, np.ndarray] = {}
        class_arrays: dict[str, np.ndarray] = {}
        for src, col in period_cols.items():
            vals = pd.to_numeric(df_out[col], errors="coerce").to_numpy(dtype=float)
            period_arrays[src] = vals
            has_any_period |= np.isfinite(vals) & (vals > 0)
        for src, col in class_cols.items():
            class_arrays[src] = df_out[col].fillna("").astype(str).to_numpy(dtype=object)

        idx_with_periods = np.flatnonzero(has_any_period)
        for idx in idx_with_periods:
            periods_by_source = {
                src: float(vals[idx])
                for src, vals in period_arrays.items()
                if np.isfinite(vals[idx]) and vals[idx] > 0
            }
            if not periods_by_source:
                continue

            ordered = sorted(
                periods_by_source,
                key=lambda s: PERIOD_SOURCE_PRIORITY.index(s) if s in PERIOD_SOURCE_PRIORITY else len(PERIOD_SOURCE_PRIORITY),
            )
            consensus, agree, conflict, support, primary_source = _choose_consensus_period(
                periods_by_source,
                rel_tol=consensus_rel_tol,
            )

            period_sources_col[idx] = "|".join(ordered)
            period_n_sources_col[idx] = len(ordered)
            period_consensus_days_col[idx] = consensus
            period_consensus_agree_col[idx] = agree
            period_conflict_flag_col[idx] = conflict
            period_consensus_support_col[idx] = support
            period_primary_source_col[idx] = primary_source
            period_source_periods_col[idx] = ";".join(f"{src}:{periods_by_source[src]:.8g}" for src in ordered)

            # Backward-compatible aggregate fields
            catalog_match_col[idx] = True
            catalog_period_col[idx] = consensus
            catalog_source_col[idx] = primary_source

            cat_class = ""
            for src in ordered:
                cvals = class_arrays.get(src)
                if cvals is None:
                    continue
                cval = str(cvals[idx]).strip()
                if cval:
                    cat_class = cval
                    break
            catalog_class_col[idx] = cat_class

    df_out["period_sources"] = period_sources_col
    df_out["period_n_sources"] = period_n_sources_col
    df_out["period_consensus_days"] = period_consensus_days_col
    df_out["period_consensus_agree"] = period_consensus_agree_col
    df_out["period_conflict_flag"] = period_conflict_flag_col
    df_out["period_consensus_support"] = period_consensus_support_col
    df_out["period_primary_source"] = period_primary_source_col
    df_out["period_source_periods"] = period_source_periods_col

    df_out["catalog_match"] = catalog_match_col
    df_out["catalog_period"] = catalog_period_col
    df_out["catalog_class"] = catalog_class_col
    df_out["catalog_source"] = catalog_source_col

    if flag_only:
        df_filtered = df_out
    else:
        df_filtered = df_out[~df_out["catalog_match"]].reset_index(drop=True)

    if show_tqdm:
        n_matched = int(catalog_match_col.sum())
        n_conflict = int(period_conflict_flag_col.sum())
        tqdm.write(f"[validate_periodic_catalog] matched {n_matched}/{n0} with periodic evidence")
        tqdm.write(f"[validate_periodic_catalog] conflict flagged {n_conflict}/{n0}")
        tqdm.write(f"[validate_periodic_catalog] kept {len(df_filtered)}/{n0}")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_periodic_catalog", rejected_log_csv)

    return df_filtered


def annotate_phase_plot_candidates(
    df: pd.DataFrame,
    *,
    max_sig: float = 0.01,
    min_power: float | None = 0.3,
    allow_alias: bool = False,
) -> pd.DataFrame:
    """Annotate periodic candidates that are eligible for phase-fold plotting.

    Eligibility uses periodicity bootstrap significance and optional power thresholds.
    This is a metadata-only annotation step (no rows are filtered).
    """
    out = df.copy()

    out["phase_plot_ready"] = False
    out["phase_period_days"] = np.nan
    out["phase_source"] = ""
    out["phase_quality_score"] = np.nan

    if "lsp_period" not in out.columns or "lsp_bootstrap_sig" not in out.columns:
        return out

    period = pd.to_numeric(out["lsp_period"], errors="coerce")
    sig = pd.to_numeric(out["lsp_bootstrap_sig"], errors="coerce")

    ready = period.notna() & np.isfinite(period) & (period > 0)
    ready &= sig.notna() & np.isfinite(sig) & (sig <= float(max_sig))

    if min_power is not None:
        if "lsp_power" not in out.columns:
            ready &= False
        else:
            power = pd.to_numeric(out["lsp_power"], errors="coerce")
            ready &= power.notna() & np.isfinite(power) & (power >= float(min_power))

    if not allow_alias and "lsp_is_alias" in out.columns:
        alias = out["lsp_is_alias"].fillna(False).astype(bool)
        ready &= ~alias

    if "periodicity_score" in out.columns:
        quality = pd.to_numeric(out["periodicity_score"], errors="coerce")
    else:
        min_p = 1e-12
        with np.errstate(invalid="ignore"):
            quality = -np.log10(np.clip(sig.to_numpy(dtype=float), min_p, 1.0))
        quality = pd.Series(quality, index=out.index)

    out.loc[ready, "phase_plot_ready"] = True
    out.loc[ready, "phase_period_days"] = period[ready].astype(float)
    out.loc[ready, "phase_source"] = "lsp"
    out.loc[ready, "phase_quality_score"] = quality[ready].astype(float)
    return out


# =============================================================================
# Main orchestration
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    *,
    # Filter 7: evidence strength
    apply_evidence_strength: bool = True,
    min_bayes_factor: float = MIN_BAYES_FACTOR,
    require_finite_local_bf: bool = True,
    # Filter 8: explicit significant detection gate
    apply_significant_detection: bool = True,
    significant_require_flag: bool = True,
    significant_min_peak_count: int = 1,
    significant_min_run_count: int = 1,
    # Filter 9: run robustness
    apply_run_robustness: bool = True,
    min_run_count: int = 1,
    max_run_count: int | None = None,
    min_run_points: int = 2,
    min_run_cameras: int = 2,
    # Filter 10: morphology
    apply_morphology: bool = False,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = POST_FILTER_MIN_DELTA_BIC,
    # Filter 11: event score
    apply_score: bool = True,
    min_dip_score: float | None = 0.0,
    min_jump_score: float | None = 0.0,
    min_score: float | None = None,
    # Validation: periodicity
    apply_periodicity_validation: bool = False,
    periodicity_n_bootstrap: int = 1000,
    periodicity_significance: float = 0.01,
    periodicity_pdm_method: str = POST_FILTER_PDM_METHOD,
    periodicity_exclude_aliases: bool = True,
    periodicity_flag_only: bool = True,
    periodicity_workers: int = 1,
    periodicity_checkpoint_dir: Path | None = None,
    periodicity_lightcurve_dir: Path | None = None,
    periodicity_skip_if_consensus: bool = True,
    periodicity_all_candidates: bool = False,
    phase_plot_max_sig: float = 0.01,
    phase_plot_min_power: float | None = 0.3,
    phase_plot_allow_alias: bool = False,
    # Validation: Gaia RUWE
    apply_gaia_ruwe_validation: bool = True,
    gaia_max_ruwe: float = POST_FILTER_MAX_RUWE,
    gaia_flag_only: bool = True,
    # Validation: Gaia proper motion
    apply_gaia_pm_validation: bool = True,
    gaia_max_pm: float = POST_FILTER_MAX_PM,
    gaia_pm_flag_only: bool = True,
    # Validation: periodic catalog
    apply_periodic_catalog_validation: bool = True,
    periodic_catalog_max_sep: float = POST_FILTER_MAX_SEP_ARCSEC,
    periodic_catalog_flag_only: bool = True,
    periodic_catalog_consensus_rel_tol: float = POST_FILTER_REL_TOL,
    periodic_catalog_use_gaia_eb: bool = True,
    periodic_catalog_use_asassn_var: bool = True,
    periodic_catalog_use_ztf_periodic: bool = True,
    periodic_catalog_use_vsx_period: bool = True,
    periodic_catalog_use_ogle_periodic: bool = True,
    periodic_catalog_vsx_crossmatch_csv: str | Path = VSX_CROSSMATCH_PATH,
    home_passers_only: bool = False,
    # General
    show_tqdm: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply candidate filters after running events.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from events.py
    apply_* : bool
        Whether to apply each filter
    apply_periodicity_validation : bool
        Apply bootstrap PDM/CE validation (expensive, off by default)
    periodicity_all_candidates : bool
        Run periodicity validation on every row in the current table instead of
        only rows that pass the prerequisite failed_* filters.
    apply_gaia_ruwe_validation : bool
        Apply Gaia RUWE validation (queries Gaia TAP)
    apply_gaia_pm_validation : bool
        Apply Gaia proper-motion validation (uses local Gaia catalog)
    apply_periodic_catalog_validation : bool
        Apply periodic-catalog evidence and period-consensus validation
    home_passers_only : bool
        During home-stage revalidation, run home-only validations only on rows
        with no upstream failed_* flags while keeping the full output table.
    show_tqdm : bool
        Show progress bars
    verbose : bool
        Print per-filter summaries and totals

    Returns
    -------
    pd.DataFrame
        Full dataframe with added columns:
        - failed_<filter_name>: bool, True if row failed that filter
        - failed_any: bool, True if row failed any filter
    """
    df_filtered = df.copy()
    n_start = len(df_filtered)

    def _merge_columns_by_path(
        df_base: pd.DataFrame,
        df_updates: pd.DataFrame,
        *,
        include_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Merge columns from df_updates into df_base by stringified path."""
        if "path" not in df_base.columns:
            return df_base
        if df_updates is None or df_updates.empty or "path" not in df_updates.columns:
            return df_base

        updates = df_updates.copy()
        updates["_path_key"] = updates["path"].astype(str)
        updates = updates.drop_duplicates(subset=["_path_key"], keep="first")
        updates_idx = updates.set_index("_path_key")

        if include_columns is None:
            cols = [c for c in updates_idx.columns if c != "path"]
        else:
            cols = [c for c in include_columns if c in updates_idx.columns]
        if not cols:
            return df_base

        base_keys = df_base["path"].astype(str)
        matched = base_keys.isin(updates_idx.index)
        if not bool(matched.any()):
            return df_base

        for col in cols:
            mapped = base_keys.map(updates_idx[col])
            if col in df_base.columns:
                df_base.loc[matched, col] = mapped.loc[matched].to_numpy()
            else:
                df_base[col] = mapped.to_numpy()
        return df_base

    filters = []

    periodicity_prereq_labels: list[str] = []

    if apply_evidence_strength:
        periodicity_prereq_labels.append("posterior_strength")
        filters.append(("posterior_strength", filter_evidence_strength, {
            "min_bayes_factor": min_bayes_factor,
            "require_finite_local_bf": require_finite_local_bf,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_significant_detection:
        periodicity_prereq_labels.append("significant_detection")
        filters.append(("significant_detection", filter_significant_detection, {
            "require_significant_flag": significant_require_flag,
            "min_peak_count": significant_min_peak_count,
            "min_run_count": significant_min_run_count,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_run_robustness:
        periodicity_prereq_labels.append("run_robustness")
        filters.append(("run_robustness", filter_run_robustness, {
            "min_run_count": min_run_count,
            "max_run_count": max_run_count,
            "min_run_points": min_run_points,
            "min_run_cameras": min_run_cameras,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_morphology:
        periodicity_prereq_labels.append("morphology")
        filters.append(("morphology", filter_morphology, {
            "dip_morphology": dip_morphology,
            "jump_morphology": jump_morphology,
            "min_delta_bic": min_delta_bic,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_score:
        periodicity_prereq_labels.append("score")
        filters.append(("score", filter_score, {
            "min_dip_score": min_dip_score,
            "min_jump_score": min_jump_score,
            "min_score": min_score,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_periodic_catalog_validation:
        filters.append(("periodic_catalog", validate_periodic_catalog, {
            "max_sep_arcsec": periodic_catalog_max_sep,
            "flag_only": periodic_catalog_flag_only,
            "consensus_rel_tol": periodic_catalog_consensus_rel_tol,
            "use_gaia_eb": periodic_catalog_use_gaia_eb,
            "use_asassn_var": periodic_catalog_use_asassn_var,
            "use_ztf_periodic": periodic_catalog_use_ztf_periodic,
            "use_vsx_period": periodic_catalog_use_vsx_period,
            "use_ogle_periodic": periodic_catalog_use_ogle_periodic,
            "vsx_crossmatch_csv": periodic_catalog_vsx_crossmatch_csv,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }, list(PERIODIC_CATALOG_MERGE_COLS)))

    if apply_gaia_ruwe_validation:
        filters.append(("gaia_ruwe", validate_gaia_ruwe, {
            "max_ruwe": gaia_max_ruwe,
            "flag_only": gaia_flag_only,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }, list(GAIA_RUWE_MERGE_COLS)))

    if apply_gaia_pm_validation:
        filters.append(("gaia_pm", validate_gaia_proper_motion, {
            "max_pm": gaia_max_pm,
            "flag_only": gaia_pm_flag_only,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }, list(GAIA_PM_MERGE_COLS)))

    if apply_periodicity_validation:
        filters.append(("periodicity", validate_periodicity, {
            "n_bootstrap": periodicity_n_bootstrap,
            "significance_level": periodicity_significance,
            "pdm_method": periodicity_pdm_method,
            "exclude_alias_periods": periodicity_exclude_aliases,
            "flag_only": periodicity_flag_only,
            "workers": periodicity_workers,
            "checkpoint_dir": periodicity_checkpoint_dir,
            "lightcurve_bundle_dir": periodicity_lightcurve_dir,
            "skip_if_consensus": periodicity_skip_if_consensus,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }, list(PERIODICITY_MERGE_COLS)))

    subset_filter_configs: dict[str, dict[str, object]] = {
        "periodic_catalog": {
            "failure_indicator_col": "catalog_match",
            "clear_defaults": HOME_ONLY_CLEAR_DEFAULTS["periodic_catalog"] if home_passers_only else None,
            "eligible_mask_fn": (
                lambda frame: _passing_mask_from_failures(frame, ignore_labels=HOME_ONLY_FILTER_LABELS)
                if home_passers_only else pd.Series(True, index=frame.index, dtype=bool)
            ),
        },
        "gaia_ruwe": {
            "failure_indicator_col": "high_ruwe_flag",
            "clear_defaults": HOME_ONLY_CLEAR_DEFAULTS["gaia_ruwe"] if home_passers_only else None,
            "eligible_mask_fn": (
                lambda frame: _passing_mask_from_failures(frame, ignore_labels=HOME_ONLY_FILTER_LABELS)
                if home_passers_only else pd.Series(True, index=frame.index, dtype=bool)
            ),
        },
        "gaia_pm": {
            "failure_indicator_col": "high_pm_flag",
            "clear_defaults": HOME_ONLY_CLEAR_DEFAULTS["gaia_pm"] if home_passers_only else None,
            "eligible_mask_fn": (
                lambda frame: _passing_mask_from_failures(frame, ignore_labels=HOME_ONLY_FILTER_LABELS)
                if home_passers_only else pd.Series(True, index=frame.index, dtype=bool)
            ),
        },
        "periodicity": {
            "failure_indicator_col": "periodic_flag",
            "clear_defaults": None,
            "eligible_mask_fn": (
                (lambda frame: pd.Series(True, index=frame.index, dtype=bool))
                if periodicity_all_candidates else
                (lambda frame: _passing_mask_from_failures(
                    frame,
                    include_labels=tuple(periodicity_prereq_labels),
                ))
            ),
        },
    }

    def _run_subset_filter(
        df_base: pd.DataFrame,
        *,
        func,
        kwargs: dict[str, object],
        eligible_mask: pd.Series,
        merge_cols: list[str] | None,
        failure_indicator_col: str | None,
        clear_defaults: dict[str, object] | None,
    ) -> tuple[pd.DataFrame, pd.Series, int]:
        failed_mask = pd.Series(False, index=df_base.index, dtype=bool)
        checked_mask = eligible_mask.reindex(df_base.index, fill_value=False).astype(bool)
        skipped_mask = ~checked_mask

        out = _clear_annotation_columns(df_base, mask=skipped_mask, defaults=clear_defaults)
        n_checked = int(checked_mask.sum())
        if n_checked == 0:
            return out, failed_mask, 0

        df_to_check = out.loc[checked_mask].copy()
        subset_kwargs = dict(kwargs)
        reject_mode = bool(failure_indicator_col) and (not bool(subset_kwargs.get("flag_only", True)))
        if reject_mode:
            subset_kwargs["flag_only"] = True

        df_result = func(df_to_check, **subset_kwargs)

        if merge_cols:
            out = _merge_columns_by_path(out, df_result, include_columns=merge_cols)

        if reject_mode:
            if failure_indicator_col and failure_indicator_col in out.columns:
                checked_flags = _to_bool_mask(out.loc[checked_mask, failure_indicator_col])
                failed_mask.loc[checked_mask] = checked_flags.to_numpy()
            else:
                passed_paths = set(df_result["path"].astype(str))
                checked_paths = out.loc[checked_mask, "path"].astype(str)
                failed_mask.loc[checked_mask] = (~checked_paths.isin(passed_paths)).to_numpy()

        return out, failed_mask, n_checked

    # Apply filters and tag failures (all rows kept)
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_filters", leave=True, disable=not show_tqdm) as pbar:
            for filter_entry in filters:
                label, func, kwargs = filter_entry[0], filter_entry[1], filter_entry[2]
                merge_cols = filter_entry[3] if len(filter_entry) > 3 else None
                start = perf_counter()

                subset_cfg = subset_filter_configs.get(label)
                if subset_cfg is not None:
                    eligible_mask = subset_cfg["eligible_mask_fn"](df_filtered)
                    df_filtered, failed_mask, n_checked = _run_subset_filter(
                        df_filtered,
                        func=func,
                        kwargs=kwargs,
                        eligible_mask=eligible_mask,
                        merge_cols=merge_cols,
                        failure_indicator_col=subset_cfg["failure_indicator_col"],
                        clear_defaults=subset_cfg["clear_defaults"],
                    )
                    elapsed = perf_counter() - start
                    df_filtered[f"failed_{label}"] = failed_mask

                    n_failed = int(failed_mask.sum())
                    if verbose:
                        pbar.set_postfix_str(
                            f"{label}: checked {n_checked}/{n_start}, {n_failed}/{n_start} failed ({elapsed:.2f}s)"
                        )
                    else:
                        pbar.set_postfix_str("")
                    pbar.update(1)
                    continue

                # Run filter on full dataframe to identify which rows pass
                df_passed = func(df_filtered, **kwargs)
                elapsed = perf_counter() - start

                # Determine which rows failed by comparing paths
                passed_paths = set(df_passed["path"].astype(str))
                failed_mask = ~df_filtered["path"].astype(str).isin(passed_paths)
                df_filtered[f"failed_{label}"] = failed_mask

                # Merge annotation columns back (e.g. high_ruwe_flag, catalog_match)
                if merge_cols:
                    df_filtered = _merge_columns_by_path(
                        df_filtered, df_passed, include_columns=merge_cols,
                    )

                n_failed = int(failed_mask.sum())
                if verbose:
                    pbar.set_postfix_str(f"{label}: {n_failed}/{n_start} failed ({elapsed:.2f}s)")
                else:
                    pbar.set_postfix_str("")
                pbar.update(1)

    # Phase-fold plotting metadata (annotation only; no filtering)
    df_filtered = annotate_phase_plot_candidates(
        df_filtered,
        max_sig=phase_plot_max_sig,
        min_power=phase_plot_min_power,
        allow_alias=phase_plot_allow_alias,
    )

    # Add summary column
    failed_cols = [c for c in df_filtered.columns if c.startswith("failed_") and c != "failed_any"]
    if failed_cols:
        df_filtered["failed_any"] = df_filtered[failed_cols].any(axis=1)

    if show_tqdm and verbose:
        n_failed_any = int(df_filtered["failed_any"].sum()) if "failed_any" in df_filtered.columns else 0
        tqdm.write(f"\n[apply_filters] {n_failed_any}/{n_start} failed at least one filter")

    return df_filtered.reset_index(drop=True)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:


    parser = argparse.ArgumentParser(
        description="Apply candidate filters to events.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  malca filter --input results.csv --output results_filtered.csv
  malca filter --input results.csv --output results_filtered.csv --min-bayes-factor 20
  malca filter --input results.csv --output results_filtered.csv --apply-periodicity-validation
  malca filter --input results.csv --output results_filtered.csv --skip-gaia-ruwe-validation --skip-periodic-catalog-validation
"""
    )
    g_io = parser.add_argument_group("Input / output")
    g_evidence = parser.add_argument_group("Evidence & significance")
    g_run = parser.add_argument_group("Run robustness")
    g_morph = parser.add_argument_group("Morphology")
    g_score = parser.add_argument_group("Score")
    g_periodicity = parser.add_argument_group("Periodicity validation")
    g_gaia_ruwe = parser.add_argument_group("Gaia RUWE")
    g_gaia_pm = parser.add_argument_group("Gaia proper motion")
    g_periodic_catalog = parser.add_argument_group("Periodic catalog")
    g_general = parser.add_argument_group("General")

    g_io.add_argument("--detect-run", type=Path, default=None,
                        help="Detect run directory (e.g., output/runs/20250121_143052). If specified, reads from <detect-run>/results/ and writes filtered results there.")
    g_io.add_argument("--input", type=Path, default=None, help="Input CSV/Parquet from events.py (overrides --detect-run)")
    g_io.add_argument("--output", type=Path, default=None, help="Output CSV/Parquet path (overrides default location)")
    g_io.add_argument("--index-file", type=Path, default=ASASSN_INDEX_PATH,
                        help="ASAS-SN index file to join ra_deg/dec_deg coordinates")

    g_evidence.add_argument("--skip-evidence-strength", action="store_true", help="Skip evidence strength filter (Bayes factor threshold)")
    g_evidence.add_argument("--skip-significant-detection", action="store_true", help="Skip explicit significant run/peak gate")
    g_evidence.add_argument("--skip-run-robustness", action="store_true", help="Skip run robustness filter")
    g_evidence.add_argument("--apply-morphology", action="store_true", help="Apply morphology filter (off by default)")
    g_evidence.add_argument("--min-bayes-factor", type=float, default=MIN_BAYES_FACTOR,
                        help="Minimum Bayes factor for posterior strength filter (default: 10)")
    g_evidence.add_argument("--allow-infinite-local-bf", action="store_true",
                        help="Allow infinite local BF (default: require finite)")
    g_evidence.add_argument("--significant-no-require-flag", action="store_true",
                        help="Do not require dip_significant/jump_significant for significant detection gate")
    g_evidence.add_argument("--significant-min-peak-count", type=int, default=1,
                        help="Minimum dip_count/jump_count for significant detection gate (default: 1)")
    g_evidence.add_argument("--significant-min-run-count", type=int, default=1,
                        help="Minimum dip_run_count/jump_run_count for significant detection gate (default: 1)")

    g_run.add_argument("--min-run-count", type=int, default=1,
                        help="Minimum number of runs (default: 1)")
    g_run.add_argument("--max-run-count", type=int, default=None,
                        help="Maximum number of runs (default: disabled)")
    g_run.add_argument("--min-run-points", type=int, default=POST_FILTER_MIN_RUN_POINTS,
                        help="Minimum points per run (default: 2)")
    g_run.add_argument("--min-run-cameras", type=int, default=POST_FILTER_MIN_RUN_CAMERAS,
                        help="Minimum cameras per run (default: 2)")

    g_morph.add_argument("--dip-morphology", type=str, default="gaussian",
                        choices=["gaussian", "paczynski"],
                        help="Required morphology for dips (default: gaussian)")
    g_morph.add_argument("--jump-morphology", type=str, default="paczynski",
                        choices=["gaussian", "paczynski"],
                        help="Required morphology for jumps (default: paczynski)")
    g_morph.add_argument("--min-delta-bic", type=float, default=POST_FILTER_MIN_DELTA_BIC,
                        help="Minimum delta BIC for morphology filter (default: 10)")

    g_score.add_argument("--apply-score-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply event score filter (default: enabled)")
    g_score.add_argument("--min-score", type=float, default=POST_FILTER_MIN_DIP_SCORE,
                        help="Legacy minimum log10 event score applied to dip and jump branches (default: -3.0)")
    g_score.add_argument("--min-dip-score", type=float, default=None,
                        help="Minimum dipper_score threshold (overrides --min-score for dips)")
    g_score.add_argument("--min-jump-score", type=float, default=None,
                        help="Minimum jumper_score threshold (overrides --min-score for jumps)")

    g_periodicity.add_argument("--apply-periodicity-validation", action="store_true",
                        help="Apply bootstrap PDM/CE periodicity validation (off by default)")
    g_periodicity.add_argument("--periodicity-n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    g_periodicity.add_argument("--periodicity-significance", type=float, default=0.01,
                        help="Significance threshold (default: 0.01)")
    g_periodicity.add_argument("--periodicity-pdm-method", type=str, default=POST_FILTER_PDM_METHOD,
                        choices=list(PDM_METHOD_CHOICES),
                        help="PDM implementation for periodicity validation (default: plavchan)")
    g_periodicity.add_argument("--periodicity-no-exclude-aliases", action="store_true",
                        help="Do not exclude alias periods (1d, 29.53d, etc.)")
    g_periodicity.add_argument("--periodicity-reject", action="store_true",
                        help="Reject periodic candidates (default: flag only)")
    g_periodicity.add_argument("--periodicity-force-bootstrap", action="store_true",
                        help="Force bootstrap periodicity checks even if consensus period is found")
    g_periodicity.add_argument("--periodicity-all-candidates", action="store_true",
                        help="Run periodicity validation on all rows in the current input, not just prerequisite passers")
    g_periodicity.add_argument("--workers", type=int, default=WORKERS,
                        help="Number of parallel workers for periodicity validation (default: 10)")
    g_periodicity.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Directory for checkpoints (enables resume on restart)")
    g_periodicity.add_argument("--phase-plot-max-sig", type=float, default=0.01,
                        help="Require lsp_bootstrap_sig <= this for phase plots (default: 0.01)")
    g_periodicity.add_argument("--phase-plot-min-power", type=float, default=0.3,
                        help="Require lsp_power >= this for phase plots (default: 0.3)")
    g_periodicity.add_argument("--phase-plot-allow-alias", action="store_true",
                        help="Allow alias periods for phase plots (default: disabled)")

    g_gaia_ruwe.add_argument("--skip-gaia-ruwe-validation", action="store_true",
                        help="Skip Gaia RUWE validation (on by default, queries Gaia TAP)")
    g_gaia_ruwe.add_argument("--gaia-max-ruwe", type=float, default=POST_FILTER_MAX_RUWE,
                        help="Maximum RUWE to keep (default: 1.4)")
    g_gaia_ruwe.add_argument("--gaia-reject", action="store_true",
                        help="Reject high RUWE sources (default: flag only)")

    g_gaia_pm.add_argument("--skip-gaia-pm-validation", action="store_true",
                        help="Skip Gaia proper-motion validation (on by default, uses local Gaia cache)")
    g_gaia_pm.add_argument("--gaia-max-pm", type=float, default=POST_FILTER_MAX_PM,
                        help="Maximum total proper motion to keep in mas/yr (default: 100.0)")
    g_gaia_pm.add_argument("--gaia-pm-reject", action="store_true",
                        help="Reject high proper-motion sources (default: flag only)")

    g_periodic_catalog.add_argument("--skip-periodic-catalog-validation", action="store_true",
                        help="Skip periodic-catalog consensus validation (on by default)")
    g_periodic_catalog.add_argument("--periodic-catalog-max-sep", type=float, default=POST_FILTER_MAX_SEP_ARCSEC,
                        help="Maximum separation in arcsec for coordinate fallback matches (default: 3.0)")
    g_periodic_catalog.add_argument("--periodic-catalog-consensus-rel-tol", type=float, default=POST_FILTER_REL_TOL,
                        help="Relative tolerance for period-consensus agreement (default: 0.10)")
    g_periodic_catalog.add_argument("--periodic-catalog-vsx-crossmatch", type=Path, default=VSX_CROSSMATCH_PATH,
                        help="ASAS-SN x VSX crossmatch CSV used for VSX period lookup")
    g_periodic_catalog.add_argument("--periodic-catalog-no-gaia-eb", action="store_true",
                        help="Disable Gaia EB period evidence in periodic-catalog validation")
    g_periodic_catalog.add_argument("--periodic-catalog-no-asassn-var", action="store_true",
                        help="Disable ASAS-SN variable catalog evidence in periodic-catalog validation")
    g_periodic_catalog.add_argument("--periodic-catalog-no-ztf", action="store_true",
                        help="Disable ZTF periodic catalog evidence in periodic-catalog validation")
    g_periodic_catalog.add_argument("--periodic-catalog-no-vsx", action="store_true",
                        help="Disable VSX period evidence in periodic-catalog validation")
    g_periodic_catalog.add_argument("--periodic-catalog-no-ogle", action="store_true",
                        help="Disable OGLE period evidence in periodic-catalog validation")
    g_periodic_catalog.add_argument("--periodic-catalog-reject", action="store_true",
                        help="Reject catalog matches (default: flag only)")
    g_periodic_catalog.add_argument("--home-passers-only", action="store_true",
                        help="Run home-only validations only on rows that already pass upstream filters")

    g_general.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    g_general.add_argument("-v", "--verbose", action="store_true", help="Print per-filter summaries (default: off)")

    args = parser.parse_args()

    # Determine input path
    if args.input:
        input_path = args.input.expanduser()
    elif args.detect_run:
        detect_run = args.detect_run.expanduser()
        results_dir = detect_run / "results"
        # Look for events results file in the detect run directory
        candidates = list(results_dir.glob("*events_results.csv")) + list(results_dir.glob("*events_results.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No events results file found in {results_dir}")
        if len(candidates) > 1:
            print(f"Warning: Multiple results files found, using: {candidates[0]}")
        input_path = candidates[0]
    else:
        raise ValueError("Must specify either --input or --detect-run")

    # Load input
    if input_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows from {input_path}")

    # Join coordinates from index file
    index_path = args.index_file.expanduser()
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    if index_path.suffix.lower() in (".parquet", ".pq"):
        index_df = pd.read_parquet(index_path)
    else:
        index_df = pd.read_csv(index_path)

    # Determine join column (asas_sn_id or path stem)
    if "asas_sn_id" in df.columns and "asas_sn_id" in index_df.columns:
        join_col = "asas_sn_id"
        # Ensure same type
        df[join_col] = df[join_col].astype(str)
        index_df[join_col] = index_df[join_col].astype(str)
    elif "path" in df.columns:
        # Extract ID from path (as string)
        df["_join_id"] = df["path"].apply(lambda p: Path(p).stem).astype(str)
        if "asas_sn_id" in index_df.columns:
            index_df["_join_id"] = index_df["asas_sn_id"].astype(str)
        join_col = "_join_id"
    else:
        raise ValueError("Cannot determine join column between events results and index file")

    # Join gaia_id, ra_deg, dec_deg from index
    join_cols = ["gaia_id", "ra_deg", "dec_deg"]
    available_cols = [c for c in join_cols if c in index_df.columns]
    if "gaia_id" not in available_cols:
        raise ValueError(f"Index file missing gaia_id column")

    # Preserve Gaia IDs as exact digit strings (avoid float/scientific notation).
    gaia_series = pd.to_numeric(index_df["gaia_id"], errors="coerce")
    index_df["gaia_id"] = gaia_series.astype("Int64").astype(str)
    index_df.loc[gaia_series.isna(), "gaia_id"] = pd.NA

    # Replace any pre-existing joined columns from prior runs to avoid _x/_y suffixes.
    existing_join_cols = [c for c in join_cols if c in df.columns]
    if existing_join_cols:
        df = df.drop(columns=existing_join_cols)

    df = df.merge(
        index_df[[join_col] + available_cols].drop_duplicates(subset=[join_col]),
        on=join_col,
        how="left"
    )
    if "_join_id" in df.columns:
        df = df.drop(columns=["_join_id"])
    print(f"Joined {len(available_cols)} columns ({', '.join(available_cols)}) from {index_path}")

    # Determine output path
    if args.output:
        output_path = args.output.expanduser()
    elif args.detect_run:
        detect_run = args.detect_run.expanduser()
        results_dir = detect_run / "results"
        # Create filtered filename based on input filename
        base_name = input_path.stem.replace("_results", "").replace("events", "")
        if base_name:
            filtered_name = f"{base_name}_events_results_filtered{input_path.suffix}"
        else:
            filtered_name = f"events_results_filtered{input_path.suffix}"
        output_path = results_dir / filtered_name
    else:
        # Fallback: same directory as input
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"

    periodicity_lightcurve_dir = None
    if args.apply_periodicity_validation:
        run_dir_candidates = [
            detect_run if args.detect_run else None,
            input_path,
            output_path,
            args.checkpoint_dir.expanduser() if args.checkpoint_dir else None,
        ]
        for candidate in run_dir_candidates:
            run_dir = _infer_run_dir_for_periodicity(candidate)
            if run_dir is None:
                continue
            lc_dir = run_dir / "bundle_assets" / "lightcurves"
            if lc_dir.is_dir():
                periodicity_lightcurve_dir = lc_dir
                break
        if verbose and periodicity_lightcurve_dir is not None:
            print(f"Using local bundled light curves for periodicity validation: {periodicity_lightcurve_dir}")

    # Apply filters
    df_filtered = apply_filters(
        df,
        # Filter toggles
        apply_evidence_strength=not args.skip_evidence_strength,
        apply_significant_detection=not args.skip_significant_detection,
        apply_run_robustness=not args.skip_run_robustness,
        apply_morphology=args.apply_morphology,
        # Posterior strength
        min_bayes_factor=args.min_bayes_factor,
        require_finite_local_bf=not args.allow_infinite_local_bf,
        # Significant detection gate
        significant_require_flag=not args.significant_no_require_flag,
        significant_min_peak_count=args.significant_min_peak_count,
        significant_min_run_count=args.significant_min_run_count,
        # Run robustness
        min_run_count=args.min_run_count,
        max_run_count=args.max_run_count,
        min_run_points=args.min_run_points,
        min_run_cameras=args.min_run_cameras,
        # Morphology
        dip_morphology=args.dip_morphology,
        jump_morphology=args.jump_morphology,
        min_delta_bic=args.min_delta_bic,
        # Score
        apply_score=args.apply_score_filter,
        min_dip_score=args.min_dip_score,
        min_jump_score=args.min_jump_score,
        min_score=args.min_score,
        # Periodicity validation
        apply_periodicity_validation=args.apply_periodicity_validation,
        periodicity_n_bootstrap=args.periodicity_n_bootstrap,
        periodicity_significance=args.periodicity_significance,
        periodicity_pdm_method=args.periodicity_pdm_method,
        periodicity_exclude_aliases=not args.periodicity_no_exclude_aliases,
        periodicity_flag_only=not args.periodicity_reject,
        periodicity_workers=args.workers,
        periodicity_checkpoint_dir=args.checkpoint_dir.expanduser() if args.checkpoint_dir else (detect_run / "checkpoints" if args.detect_run and args.apply_periodicity_validation else None),
        periodicity_lightcurve_dir=periodicity_lightcurve_dir,
        periodicity_skip_if_consensus=not args.periodicity_force_bootstrap,
        periodicity_all_candidates=args.periodicity_all_candidates,
        phase_plot_max_sig=args.phase_plot_max_sig,
        phase_plot_min_power=args.phase_plot_min_power,
        phase_plot_allow_alias=args.phase_plot_allow_alias,
        # Gaia RUWE validation
        apply_gaia_ruwe_validation=not args.skip_gaia_ruwe_validation,
        gaia_max_ruwe=args.gaia_max_ruwe,
        gaia_flag_only=not args.gaia_reject,
        # Gaia PM validation
        apply_gaia_pm_validation=not args.skip_gaia_pm_validation,
        gaia_max_pm=args.gaia_max_pm,
        gaia_pm_flag_only=not args.gaia_pm_reject,
        # Periodic catalog validation
        apply_periodic_catalog_validation=not args.skip_periodic_catalog_validation,
        periodic_catalog_max_sep=args.periodic_catalog_max_sep,
        periodic_catalog_flag_only=not args.periodic_catalog_reject,
        periodic_catalog_consensus_rel_tol=args.periodic_catalog_consensus_rel_tol,
        periodic_catalog_use_gaia_eb=not args.periodic_catalog_no_gaia_eb,
        periodic_catalog_use_asassn_var=not args.periodic_catalog_no_asassn_var,
        periodic_catalog_use_ztf_periodic=not args.periodic_catalog_no_ztf,
        periodic_catalog_use_vsx_period=not args.periodic_catalog_no_vsx,
        periodic_catalog_use_ogle_periodic=not args.periodic_catalog_no_ogle,
        periodic_catalog_vsx_crossmatch_csv=args.periodic_catalog_vsx_crossmatch,
        home_passers_only=args.home_passers_only,
        # General
        show_tqdm=not args.no_progress,
        verbose=args.verbose,
    )

    # Generate filter log with comprehensive statistics
    if args.detect_run:
        try:





            detect_run = args.detect_run.expanduser()
            filter_log_file = detect_run / "filter_log.json"

            orig_argv = getattr(sys, "orig_argv", None)
            cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)

            filter_log = {
                "timestamp": datetime.now().isoformat(),
                "command": cmd,
                "input_file": str(input_path),
                "output_file": str(output_path),
                "filter_params": {
                    "apply_evidence_strength": not args.skip_evidence_strength,
                    "apply_significant_detection": not args.skip_significant_detection,
                    "apply_run_robustness": not args.skip_run_robustness,
                    "apply_morphology": args.apply_morphology,
                    "apply_score": args.apply_score_filter,
                    "apply_periodicity_validation": args.apply_periodicity_validation,
                    "periodicity_reject": args.periodicity_reject if args.apply_periodicity_validation else None,
                    "periodicity_all_candidates": args.periodicity_all_candidates if args.apply_periodicity_validation else None,
                    "phase_plot_max_sig": args.phase_plot_max_sig,
                    "phase_plot_min_power": args.phase_plot_min_power,
                    "phase_plot_allow_alias": args.phase_plot_allow_alias,
                    "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
                    "apply_gaia_pm_validation": not args.skip_gaia_pm_validation,
                    "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
                    "home_passers_only": args.home_passers_only,
                    "min_bayes_factor": args.min_bayes_factor,
                    "require_finite_local_bf": not args.allow_infinite_local_bf,
                    "significant_require_flag": not args.significant_no_require_flag,
                    "significant_min_peak_count": args.significant_min_peak_count,
                    "significant_min_run_count": args.significant_min_run_count,
                    "min_run_count": args.min_run_count,
                    "max_run_count": args.max_run_count,
                    "min_run_points": args.min_run_points,
                    "min_run_cameras": args.min_run_cameras,
                    "dip_morphology": args.dip_morphology if args.apply_morphology else None,
                    "jump_morphology": args.jump_morphology if args.apply_morphology else None,
                    "min_delta_bic": args.min_delta_bic if args.apply_morphology else None,
                    "min_score": args.min_score if args.apply_score_filter else None,
                    "min_dip_score": args.min_dip_score if args.apply_score_filter else None,
                    "min_jump_score": args.min_jump_score if args.apply_score_filter else None,
                    "gaia_max_ruwe": args.gaia_max_ruwe if not args.skip_gaia_ruwe_validation else None,
                    "gaia_reject": args.gaia_reject if not args.skip_gaia_ruwe_validation else None,
                    "gaia_max_pm": args.gaia_max_pm if not args.skip_gaia_pm_validation else None,
                    "gaia_pm_reject": args.gaia_pm_reject if not args.skip_gaia_pm_validation else None,
                    "periodic_catalog_max_sep": args.periodic_catalog_max_sep if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_consensus_rel_tol": args.periodic_catalog_consensus_rel_tol if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_use_gaia_eb": (not args.periodic_catalog_no_gaia_eb) if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_use_asassn_var": (not args.periodic_catalog_no_asassn_var) if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_use_ztf_periodic": (not args.periodic_catalog_no_ztf) if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_use_vsx_period": (not args.periodic_catalog_no_vsx) if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_use_ogle_periodic": (not args.periodic_catalog_no_ogle) if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_vsx_crossmatch": str(args.periodic_catalog_vsx_crossmatch) if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_reject": args.periodic_catalog_reject if not args.skip_periodic_catalog_validation else None,
                },
                "results": {
                    "total_rows": len(df_filtered),
                    "passed_all": int((~df_filtered.get("failed_any", pd.Series(False))).sum()),
                    "failed_any": int(df_filtered.get("failed_any", pd.Series(False)).sum()),
                    "per_filter_failures": {
                        col: int(df_filtered[col].sum())
                        for col in df_filtered.columns
                        if col.startswith("failed_") and col != "failed_any"
                    },
                },
            }

            with open(filter_log_file, "w") as f:
                json.dump(filter_log, f, indent=2, default=str)

            if args.verbose:
                print(f"Filter log saved to {filter_log_file}")

        except Exception as e:
            if args.verbose:
                print(f"Warning: could not write filter log: {e}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in (".parquet", ".pq"):
        df_filtered.to_parquet(output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    else:
        df_filtered.to_csv(output_path, index=False)

    n_failed = int(df_filtered["failed_any"].sum()) if "failed_any" in df_filtered.columns else 0
    n_passed = len(df_filtered) - n_failed
    print(f"\nWrote {len(df_filtered)} rows to {output_path}")
    print(f"Passed all filters: {n_passed}/{len(df_filtered)} ({n_passed/len(df_filtered)*100:.1f}%)")

    # Print per-filter failure counts
    failed_cols = [c for c in df_filtered.columns if c.startswith("failed_") and c != "failed_any"]
    if failed_cols:
        print("\nPer-filter failures:")
        for col in failed_cols:
            n = int(df_filtered[col].sum())
            print(f"  {col}: {n}/{len(df_filtered)} ({n/len(df_filtered)*100:.1f}%)")


if __name__ == "__main__":
    main()
