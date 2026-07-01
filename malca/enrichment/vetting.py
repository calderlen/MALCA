"""
Post-review vetting: check whether candidates are already known objects.

Queries:
 1. SIMBAD — object type, identifiers, bibliography count
 2. Gaia DR3 variability tables — variability flag + classification
 3. ASAS-SN Variable Stars Database (VizieR II/366) — known ASAS-SN variables
 4. OGLE EWS + KMTNet + MOA microlensing events — known microlensing surveys
 5. ZTF periodic variables (Chen+ 2020, VizieR J/ApJS/249/18) — recent ZTF discoveries
 6. TNS (Transient Name Server) — supernovae, novae, CVs, transients
 7. Gaia DR3 eclipsing binary parameters — periods for dominant contaminant class
 8. ALeRCE ZTF broker — ZTF ML classification
 9. ATLAS forced photometry — independent cyan/orange confirmation
10. Gaia DR3 epoch photometry — space-based variability confirmation
11. eROSITA X-ray catalog — youth indicator
12. Chandra CSC X-ray catalog — archival high-resolution X-ray detections
13. Proper motion consistency — cluster membership validation
14. NEOWISE light curves — IR time-series for dipper confirmation

Usage:
    from malca.enrichment.vetting import vet_candidates
    df_vetted = vet_candidates(df)
"""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tarfile
from typing import Callable, Literal
import argparse
import io
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urljoin

from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from astroquery.xmatch import XMatch
from tqdm import tqdm
import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd
import pyvo
import requests

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from malca.config import GAIA_CHUNK_SIZE
from malca.config import PARQUET_CACHE_COMPRESSION
from malca.config import VIZIER_TAP_URL
from malca.config import GAIA_AIP_TAP_URL
from malca.config import DEFAULT_CACHE_DIR, LEGACY_DEFAULT_CACHE_DIR
from malca.external_lc_manifest import upsert_external_lc_manifest_entry
from malca.config import (
    VETTING_SIMBAD_BATCH_SIZE,
    VETTING_SIMBAD_RETRY_DELAY,
    VETTING_SIMBAD_MAX_RETRIES,
    GAIA_ESA_TAP_URL,
    ALERCE_API_BASE,
    ATLAS_API_BASE,
    TNS_API_BASE,
    ASASSN_VAR_CATALOG_ID,
    ZTF_VAR_CATALOG_ID,
    EROSITA_CATALOG_ID,
    CHANDRA_CSC_CATALOG_ID,
    OGLE_MICROLENS_CATALOG_ID,
    ALERCE_RADIUS_ARCSEC as CFG_ALERCE_RADIUS_ARCSEC,
    ATLAS_MJD_MIN as CFG_ATLAS_MJD_MIN,
    ZTF_VAR_RADIUS_ARCSEC as CFG_ZTF_VAR_RADIUS_ARCSEC,
    TNS_RADIUS_ARCSEC as CFG_TNS_RADIUS_ARCSEC,
    EROSITA_RADIUS_ARCSEC as CFG_EROSITA_RADIUS_ARCSEC,
    CHANDRA_CSC_RADIUS_ARCSEC as CFG_CHANDRA_CSC_RADIUS_ARCSEC,
    OGLE_MICROLENS_RADIUS_ARCSEC as CFG_OGLE_MICROLENS_RADIUS_ARCSEC,
    NEOWISE_VET_MAX_SEP_ARCSEC,
    ZTF_LC_RADIUS_ARCSEC,
    CRTS_MATCH_RADIUS_ARCSEC,
    ALERCE_BATCH_SIZE as CFG_ALERCE_BATCH_SIZE,
    TNS_BATCH_SIZE as CFG_TNS_BATCH_SIZE,
    ALERCE_WORKERS as CFG_ALERCE_WORKERS,
    NEOWISE_VET_WORKERS as CFG_NEOWISE_VET_WORKERS,
    CRTS_CHUNK_SIZE as CFG_CRTS_CHUNK_SIZE,
    GAIA_EPOCH_VET_CHUNK_SIZE as CFG_GAIA_EPOCH_VET_CHUNK_SIZE,
    ATLAS_POLL_INTERVAL as CFG_ATLAS_POLL_INTERVAL,
    ATLAS_MAX_POLL as CFG_ATLAS_MAX_POLL,
    VETTING_HTTP_TIMEOUT,
    VETTING_BACKOFF_CAP,
    MJD_TO_JD,
    MICROLENS_OGLE_EWS_START_YEAR,
    MICROLENS_KMTNET_START_YEAR,
    MICROLENS_DEFAULT_END_YEAR,
    PANSTARRS_DEC_LIMIT,
    TESS_SEARCH_RADIUS_ARCSEC,
    AAVSO_MAX_PAGES,
    AAVSO_RESULTS_PER_PAGE,
)
from malca.catalogs.gaia_ids import parse_gaia_source_id
from malca.catalogs.neowise_filters import filter_neowise_single_exposure_lc
from malca.core.utils import batch_tap_crossmatch
from malca.products.candidates import select_passing_candidates_if_present
from malca.io.table_io import read_feature_table, write_feature_table






# Vetting configuration
SIMBAD_RADIUS_ARCSEC = 5.0
SIMBAD_BATCH_SIZE = VETTING_SIMBAD_BATCH_SIZE
SIMBAD_RETRY_DELAY = VETTING_SIMBAD_RETRY_DELAY
SIMBAD_MAX_RETRIES = VETTING_SIMBAD_MAX_RETRIES

# Gaia archive endpoints can reject very long IN(...) clauses; keep this
# conservative for robustness across mirrors.
GAIA_VAR_CHUNK_SIZE = min(GAIA_CHUNK_SIZE, 100)
GAIA_TAP_URLS = [GAIA_ESA_TAP_URL, GAIA_AIP_TAP_URL]
ASASSN_VAR_CATALOG = ASASSN_VAR_CATALOG_ID
ASASSN_VAR_LOCAL_CSV = Path(__file__).resolve().parent.parent / "input" / "asassn_variables_220326.csv"
ASASSN_VAR_RADIUS_ARCSEC = 5.0
GAIA_TAP_RETRY_BASE_DELAY = 5.0
GAIA_TAP_RETRY_MAX_DELAY = 60.0
VETTING_CACHE_FILES = {
    "simbad": "vetting_simbad.parquet",
    "gaia_variability": "vetting_gaia_variability.parquet",
    "gaia_epoch": "vetting_gaia_epoch.parquet",
    "gaia_eb": "vetting_gaia_eb.parquet",
    "alerce": "vetting_alerce.parquet",
}

# Module-level cache for the local ASAS-SN catalog
_asassn_cache: dict = {}

ALERCE_RADIUS_ARCSEC = CFG_ALERCE_RADIUS_ARCSEC
ALERCE_BATCH_SIZE = CFG_ALERCE_BATCH_SIZE
ALERCE_WORKERS = CFG_ALERCE_WORKERS

ATLAS_MJD_MIN = CFG_ATLAS_MJD_MIN
ATLAS_POLL_INTERVAL = CFG_ATLAS_POLL_INTERVAL
ATLAS_MAX_POLL = CFG_ATLAS_MAX_POLL


def _short_error(exc: Exception, max_len: int = 240) -> str:
    msg = str(exc).splitlines()[0].strip()
    return msg if len(msg) <= max_len else msg[:max_len - 3] + "..."


def _write_to_stream(stream: object | None, text: str) -> bool:
    if stream is None:
        return False
    try:
        stream.write(text)
        stream.flush()
        return True
    except Exception:
        return False


class _SafeOutputStream(io.TextIOBase):
    """Forward writes to a stream, falling back if that stream was closed."""

    def __init__(self, primary: object | None, fallback: object | None) -> None:
        super().__init__()
        self._primary = primary
        self._fallback = fallback

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        text = str(text)
        if not _write_to_stream(self._primary, text):
            _write_to_stream(self._fallback, text)
        return len(text)

    def flush(self) -> None:
        for stream in (self._primary, self._fallback):
            try:
                stream.flush()
                return
            except Exception:
                continue


def _safe_print(msg: str) -> None:
    text = f"{msg}\n"
    if _write_to_stream(sys.stdout, text):
        return
    _write_to_stream(getattr(sys, "__stderr__", None) or sys.stderr, text)


def _raise_lookup_failures(label: str, failures: list[str], n_total: int) -> None:
    if not failures:
        return
    detail = "; ".join(failures[:3])
    more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
    raise RuntimeError(
        f"{label}: lookup failed for {len(failures)}/{n_total} candidates: {detail}{more}"
    )


def _gaia_retry_delay(attempt: int) -> float:
    return min(GAIA_TAP_RETRY_BASE_DELAY * max(1, attempt), GAIA_TAP_RETRY_MAX_DELAY)


def _gaia_tap_error_is_nonretryable(exc: Exception) -> bool:
    text = str(exc).lower()
    nonretryable_markers = (
        "code='invalid'",
        'code="invalid"',
        "column ",
        " not found",
        "error while translating your query",
        "syntax error",
    )
    return any(marker in text for marker in nonretryable_markers)


def _parse_gaia_source_id_str(value: object) -> str | None:
    """Parse Gaia source ID-like values to a plain integer string."""
    return parse_gaia_source_id(value)


def _connect_gaia_taps_until_available(
    test_query: str,
    *,
    label: str,
    urls: list[str] | None = None,
    maxrec: int | None = None,
) -> list[tuple[str, pyvo.dal.TAPService]]:
    """Return working Gaia TAP services, retrying until interrupted."""
    tap_urls = list(urls or GAIA_TAP_URLS)
    attempt = 0
    while True:
        attempt += 1
        taps: list[tuple[str, pyvo.dal.TAPService]] = []
        errors: list[str] = []
        nonretryable_errors: list[str] = []
        for tap_url in tap_urls:
            try:
                tap = pyvo.dal.TAPService(tap_url)
                if maxrec is None:
                    tap.run_sync(test_query)
                else:
                    tap.run_sync(test_query, maxrec=maxrec)
                taps.append((tap_url, tap))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                msg = f"{tap_url}: {_short_error(exc)}"
                if _gaia_tap_error_is_nonretryable(exc):
                    nonretryable_errors.append(msg)
                else:
                    errors.append(msg)
        if taps:
            return taps
        if nonretryable_errors and not errors:
            detail = " | ".join(nonretryable_errors)
            raise RuntimeError(f"{label}: Gaia TAP test query is invalid: {detail}")

        delay = _gaia_retry_delay(attempt)
        detail = " | ".join(errors + nonretryable_errors) if (errors or nonretryable_errors) else "no TAP services configured"
        print(
            f"  {label}: all Gaia TAP servers unavailable "
            f"(attempt {attempt}); retrying in {delay:.0f}s. Last errors: {detail}"
        )
        time.sleep(delay)


def _run_gaia_tap_query_until_success(
    taps: list[tuple[str, pyvo.dal.TAPService]],
    query: str,
    *,
    label: str,
    maxrec: int | None = None,
):
    """Run a Gaia TAP query on available mirrors, retrying until interrupted."""
    attempt = 0
    while True:
        attempt += 1
        errors: list[str] = []
        nonretryable_errors: list[str] = []
        for i, (tap_url, tap) in enumerate(list(taps)):
            try:
                if maxrec is None:
                    result = tap.run_sync(query)
                else:
                    result = tap.run_sync(query, maxrec=maxrec)
                if i != 0:
                    taps.insert(0, taps.pop(i))
                return result
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                msg = f"{tap_url}: {_short_error(exc)}"
                if _gaia_tap_error_is_nonretryable(exc):
                    nonretryable_errors.append(msg)
                else:
                    errors.append(msg)
        if nonretryable_errors and not errors:
            detail = " | ".join(nonretryable_errors)
            raise RuntimeError(f"{label}: Gaia TAP query is invalid: {detail}")

        delay = _gaia_retry_delay(attempt)
        detail = " | ".join(errors + nonretryable_errors) if (errors or nonretryable_errors) else "no TAP services available"
        print(
            f"  {label}: Gaia TAP query failed on all mirrors "
            f"(attempt {attempt}); retrying in {delay:.0f}s. Last errors: {detail}"
        )
        time.sleep(delay)

ZTF_VAR_CATALOG = ZTF_VAR_CATALOG_ID

ZTF_VAR_RADIUS_ARCSEC = CFG_ZTF_VAR_RADIUS_ARCSEC
TNS_RADIUS_ARCSEC = CFG_TNS_RADIUS_ARCSEC
TNS_BATCH_SIZE = CFG_TNS_BATCH_SIZE
OGLE_MICROLENS_CATALOG = OGLE_MICROLENS_CATALOG_ID
OGLE_MICROLENS_RADIUS_ARCSEC = CFG_OGLE_MICROLENS_RADIUS_ARCSEC
MICROLENS_CACHE_DIR = (DEFAULT_CACHE_DIR / "microlensing").expanduser()
MICROLENS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TNS_LOCAL_INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
TNS_LOCAL_CSVS = [
    TNS_LOCAL_INPUT_DIR / "tns_public_objects.csv",
    TNS_LOCAL_INPUT_DIR / "tns_sne.csv",
    TNS_LOCAL_INPUT_DIR / "asassn_transients.csv",
]

# Module-level cache for the local TNS catalog
_tns_cache: dict = {}

EROSITA_CATALOG = EROSITA_CATALOG_ID
EROSITA_LOCAL_FITS = Path(__file__).resolve().parent.parent / "input" / "eRASS1_Main.v1.2.fits"
EROSITA_RADIUS_ARCSEC = CFG_EROSITA_RADIUS_ARCSEC
CHANDRA_CSC_CATALOG = CHANDRA_CSC_CATALOG_ID
CHANDRA_CSC_RADIUS_ARCSEC = CFG_CHANDRA_CSC_RADIUS_ARCSEC
EROSITA_XRAY_LABEL = "eROSITA"
CHANDRA_XRAY_LABEL = "Chandra CSC 2.1"

# Module-level cache for the local eROSITA catalog
_erosita_cache: dict = {}

NEOWISE_MAX_SEP_ARCSEC = NEOWISE_VET_MAX_SEP_ARCSEC
NEOWISE_VET_WORKERS = CFG_NEOWISE_VET_WORKERS
CRTS_CHUNK_SIZE = CFG_CRTS_CHUNK_SIZE
GAIA_EPOCH_VET_CHUNK_SIZE = CFG_GAIA_EPOCH_VET_CHUNK_SIZE


def _vetting_cache_path(cache_dir: Path | str | None, cache_name: str) -> Path:
    base = Path(cache_dir).expanduser() if cache_dir is not None else (DEFAULT_CACHE_DIR / "vetting").expanduser()
    return base / VETTING_CACHE_FILES[cache_name]


def _legacy_vetting_cache_path(cache_name: str) -> Path:
    return LEGACY_DEFAULT_CACHE_DIR.expanduser() / VETTING_CACHE_FILES[cache_name]


def _read_vetting_cache(cache_dir: Path | str | None, cache_name: str) -> pd.DataFrame:
    path = _vetting_cache_path(cache_dir, cache_name)
    if not path.exists() and cache_dir is None:
        legacy_path = _legacy_vetting_cache_path(cache_name)
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"  Cache warning: could not read {path}: {_short_error(exc)}")
        return pd.DataFrame()


def _write_vetting_cache(
    cache_dir: Path | str | None,
    cache_name: str,
    rows: pd.DataFrame,
    *,
    key_cols: list[str],
) -> None:
    if rows.empty:
        return
    path = _vetting_cache_path(cache_dir, cache_name)
    try:
        existing = _read_vetting_cache(cache_dir, cache_name) if not path.exists() else pd.read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False, compression=PARQUET_CACHE_COMPRESSION)
    except Exception as exc:
        print(f"  Cache warning: could not write {path}: {_short_error(exc)}")


def _coord_cache_key(ra: object, dec: object, radius_arcsec: float) -> str | None:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
        rad_f = float(radius_arcsec)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(ra_f) and np.isfinite(dec_f) and np.isfinite(rad_f)):
        return None
    return f"{ra_f:.7f}:{dec_f:.7f}:{rad_f:.3f}"


def _cached_rows_by_key(cache: pd.DataFrame, key_col: str, keys: set[str]) -> dict[str, pd.Series]:
    if cache.empty or key_col not in cache.columns:
        return {}
    cache = cache.copy()
    cache[key_col] = cache[key_col].astype(str)
    cache = cache[cache[key_col].isin(keys)]
    if cache.empty:
        return {}
    cache = cache.drop_duplicates(subset=key_col, keep="last")
    return {str(row[key_col]): row for _, row in cache.iterrows()}


EXTERNAL_LC_STATUS_FILE = "_external_lc_status.parquet"
OGLE_OCVS_BASE_URLS = (
    "https://ogle.astrouw.edu.pl/ogle/ogle4/OCVS",
    "https://www.astrouw.edu.pl/ogle/ogle4/OCVS",
)
STRIPE82_VARIABLES_URL = "https://faculty.washington.edu/ivezic/sdss/catalogs/S82variables.html"
STRIPE82_MASTER_FALLBACK_URLS = (
    "https://faculty.washington.edu/ivezic/sdss/catalogs/stripe82candidateVar_v1.1.dat.gz",
    "https://faculty.washington.edu/ivezic/sdss/catalogs/S82variables/S82variables.dat.gz",
    "https://faculty.washington.edu/ivezic/sdss/catalogs/S82variables.dat.gz",
)
STRIPE82_LC_ARCHIVE_FALLBACK_URLS = (
    "https://faculty.washington.edu/ivezic/sdss/catalogs/AllLCs.tar.gz",
    "https://faculty.washington.edu/ivezic/sdss/catalogs/S82variables/AllLCs.tar.gz",
)
ALLWISE_MEP_MAX_SEP_ARCSEC = 3.0
ALLWISE_MEP_MAX_ATTEMPTS = 3
ALLWISE_MEP_RETRY_BASE_DELAY = 1.0
ALLWISE_MEP_RETRY_MARKERS = (
    "502",
    "503",
    "504",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "transient",
    "internal server error",
)
VVVX_VIRAC_MAX_SEP_ARCSEC = 1.0
OGLE_LC_MAX_SEP_ARCSEC = 2.0
STRIPE82_MAX_SEP_ARCSEC = 1.5
ESO_TAP_CAT_URL = "https://archive.eso.org/tap_cat"
AAVSO_VSX_API_URLS = (
    "https://www.aavso.org/vsx/index.php",
    "https://vsx.aavso.org/index.php",
)
AAVSO_VSX_API_URL = AAVSO_VSX_API_URLS[0]
AAVSO_DEFAULT_FROM_JD = 2456000.5
AAVSO_MAX_POINTS = AAVSO_MAX_PAGES * AAVSO_RESULTS_PER_PAGE
AAVSO_NAME_COLUMNS = ("vsx_name", "asassn_var_name", "simbad_main_id", "tns_name", "ztf_var_name")
PANSTARRS_LC_RADIUS_DEG = 0.0015
PANSTARRS_LC_MAX_ATTEMPTS = 5
PANSTARRS_LC_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _candidate_cache_id(df: pd.DataFrame, idx) -> str:
    if "candidate_id" in df.columns:
        value = df.loc[idx, "candidate_id"]
        if pd.notna(value) and str(value).strip():
            return str(value)
    return str(idx)


def _external_lc_path(output_dir: Path | str | None, file_prefix: str, df: pd.DataFrame, idx) -> Path | None:
    if output_dir is None:
        return None
    return Path(output_dir) / f"{file_prefix}_{_candidate_cache_id(df, idx)}.parquet"


def _external_lc_status_path(output_dir: Path | str | None) -> Path | None:
    if output_dir is None:
        return None
    return Path(output_dir) / EXTERNAL_LC_STATUS_FILE


def _read_external_lc_status(output_dir: Path | str | None) -> pd.DataFrame:
    path = _external_lc_status_path(output_dir)
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        _safe_print(f"  External LC cache warning: could not read {path}: {_short_error(exc)}")
        return pd.DataFrame()


def _write_external_lc_status(output_dir: Path | str | None, rows: list[dict]) -> None:
    path = _external_lc_status_path(output_dir)
    if path is None or not rows:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with open(lock_path, "a", encoding="ascii") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                new = pd.DataFrame(rows)
                existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
                combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
                combined = combined.drop_duplicates(subset=["module", "candidate_id", "cache_key"], keep="last")
                combined.to_parquet(path, index=False, compression=PARQUET_CACHE_COMPRESSION)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        _safe_print(f"  External LC cache warning: could not write {path}: {_short_error(exc)}")


def _coord_lookup_cache_key(df: pd.DataFrame, idx, radius_arcsec: float, *extra: object) -> str | None:
    if "ra" not in df.columns or "dec" not in df.columns:
        return None
    key = _coord_cache_key(df.loc[idx, "ra"], df.loc[idx, "dec"], radius_arcsec)
    if key is None:
        return None
    if extra:
        return "|".join([key, *[str(x) for x in extra]])
    return key


def _source_lookup_cache_key(source_id: object, *extra: object) -> str | None:
    if source_id is None:
        return None
    try:
        if pd.isna(source_id):
            return None
    except Exception:
        pass
    text = str(source_id).strip()
    if not text:
        return None
    if extra:
        return "|".join([text, *[str(x) for x in extra]])
    return text


def _external_lc_status_hit(
    status_df: pd.DataFrame,
    module: str,
    candidate_id: str,
    cache_key: str | None,
    summary_cols: list[str],
) -> dict | None:
    if status_df.empty or cache_key is None:
        return None
    required = {"module", "candidate_id", "cache_key", "status"}
    if not required.issubset(status_df.columns):
        return None
    mask = (
        (status_df["module"].astype(str) == module)
        & (status_df["candidate_id"].astype(str) == candidate_id)
        & (status_df["cache_key"].astype(str) == str(cache_key))
    )
    rows = status_df[mask]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    if str(row.get("status", "")) != "no_data":
        return None
    return {col: row.get(col, np.nan) for col in summary_cols}


def _read_external_lc_file(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        if path.stat().st_size <= 0:
            return None
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def _summary_is_positive(summary: dict, match_col: str) -> bool:
    value = summary.get(match_col)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(pd.notna(value) and float(value) > 0)
    except Exception:
        return False


def _apply_external_lc_cache_hits(
    df: pd.DataFrame,
    valid_idx: list,
    *,
    output_dir: Path | str | None,
    refresh_cache: bool,
    module: str,
    file_prefix: str,
    summary_cols: list[str],
    match_col: str,
    cache_key_func: Callable,
    summarize_func: Callable[[pd.DataFrame], dict],
) -> tuple[list, int, int]:
    if output_dir is None or refresh_cache:
        return valid_idx, 0, 0

    status_df = _read_external_lc_status(output_dir)
    missing_idx = []
    n_cached = 0
    n_matched = 0

    for idx in valid_idx:
        cand_id = _candidate_cache_id(df, idx)
        lc_path = _external_lc_path(output_dir, file_prefix, df, idx)
        cached_lc = _read_external_lc_file(lc_path)
        summary = None
        if cached_lc is not None:
            try:
                summary = summarize_func(cached_lc)
            except Exception:
                summary = None
        if summary is None:
            cache_key = cache_key_func(idx)
            summary = _external_lc_status_hit(status_df, module, cand_id, cache_key, summary_cols)
        if summary is None:
            missing_idx.append(idx)
            continue

        for col in summary_cols:
            df.loc[idx, col] = summary.get(col, np.nan)
        n_cached += 1
        if _summary_is_positive(summary, match_col):
            n_matched += 1

    if n_cached:
        _safe_print(f"{module}: served {n_cached} from cache; fetching {len(missing_idx)} misses")
    return missing_idx, n_cached, n_matched


def _external_lc_status_row(
    df: pd.DataFrame,
    idx,
    *,
    module: str,
    cache_key: str | None,
    summary: dict,
    status: str,
) -> dict | None:
    if cache_key is None:
        return None
    row = {
        "module": module,
        "candidate_id": _candidate_cache_id(df, idx),
        "cache_key": cache_key,
        "status": status,
        "updated_unix": time.time(),
    }
    row.update(summary)
    return row


def _write_external_lc_file(output_dir: Path | str | None, file_prefix: str, df: pd.DataFrame, idx, lc_df: pd.DataFrame) -> None:
    path = _external_lc_path(output_dir, file_prefix, df, idx)
    if path is None or lc_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lc_df.to_parquet(path, index=False, compression=PARQUET_CACHE_COMPRESSION)
    if output_dir is not None:
        upsert_external_lc_manifest_entry(
            output_dir,
            candidate_id=_candidate_cache_id(df, idx),
            source=file_prefix,
            file_prefix=file_prefix,
            path=path,
        )


def _external_catalog_cache_dir(source: str) -> Path:
    path = Path(DEFAULT_CACHE_DIR) / "external_lcs" / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_to_cache(url: str, path: Path, *, timeout: float = 120.0) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    tmp.replace(path)
    return path


def _band_mag_range(lc: pd.DataFrame, band: str, *, band_col: str = "band", mag_col: str = "mag") -> float:
    if lc.empty or band_col not in lc.columns or mag_col not in lc.columns:
        return np.nan
    mags = pd.to_numeric(lc.loc[lc[band_col].astype(str).str.lower() == band.lower(), mag_col], errors="coerce").dropna()
    return float(mags.max() - mags.min()) if len(mags) >= 2 else np.nan


def _summarize_long_mag_lc(lc: pd.DataFrame, prefix: str, bands: tuple[str, ...]) -> dict:
    if lc.empty or "mag" not in lc.columns:
        raise ValueError(f"invalid {prefix} light curve")
    out = {f"{prefix}_n_points": int(len(lc))}
    for band in bands:
        out[f"{prefix}_{band.lower()}_range"] = _band_mag_range(lc, band)
    return out


def _summarize_atlas_lc(phot: pd.DataFrame) -> dict:
    if phot.empty:
        raise ValueError("empty ATLAS light curve")
    summary = {
        "atlas_has_phot": True,
        "atlas_n_det_cyan": 0,
        "atlas_n_det_orange": 0,
        "atlas_cyan_range": np.nan,
        "atlas_orange_range": np.nan,
    }
    if "F" in phot.columns:
        cyan = phot[phot["F"] == "c"]
        orange = phot[phot["F"] == "o"]
    elif "filter" in phot.columns:
        cyan = phot[phot["filter"] == "c"]
        orange = phot[phot["filter"] == "o"]
    else:
        return summary

    mag_col = "m" if "m" in phot.columns else "mag" if "mag" in phot.columns else None
    if mag_col is None:
        return summary

    c_mags = pd.to_numeric(cyan[mag_col], errors="coerce").dropna()
    o_mags = pd.to_numeric(orange[mag_col], errors="coerce").dropna()
    summary["atlas_n_det_cyan"] = len(c_mags)
    summary["atlas_n_det_orange"] = len(o_mags)
    if len(c_mags) >= 2:
        summary["atlas_cyan_range"] = round(float(c_mags.max() - c_mags.min()), 4)
    if len(o_mags) >= 2:
        summary["atlas_orange_range"] = round(float(o_mags.max() - o_mags.min()), 4)
    return summary


def _summarize_ztf_lc(lc: pd.DataFrame) -> dict:
    lc = _coalesce_duplicate_columns(lc)
    if lc.empty or "mag" not in lc.columns:
        raise ValueError("invalid ZTF light curve")
    mag = pd.to_numeric(lc["mag"], errors="coerce")
    band = lc["band"] if "band" in lc.columns else pd.Series("", index=lc.index)
    g_mags = mag[band == "zg"].dropna()
    r_mags = mag[band == "zr"].dropna()
    return {
        "ztf_lc_n_det": len(lc),
        "ztf_lc_g_range": float(g_mags.max() - g_mags.min()) if len(g_mags) >= 2 else np.nan,
        "ztf_lc_r_range": float(r_mags.max() - r_mags.min()) if len(r_mags) >= 2 else np.nan,
    }


def _summarize_gaia_epoch_lc(lc: pd.DataFrame) -> dict:
    if lc.empty or "mag" not in lc.columns:
        raise ValueError("invalid Gaia epoch light curve")
    g_mags = pd.to_numeric(lc["mag"], errors="coerce").dropna()
    return {
        "gaia_epoch_lc_n_g": len(lc),
        "gaia_epoch_lc_g_range": float(g_mags.max() - g_mags.min()) if len(g_mags) >= 2 else np.nan,
    }


def _gaia_epoch_cell_values(value: object) -> list:
    if isinstance(value, np.ma.MaskedArray):
        data = np.asarray(value.data, dtype=object).ravel()
        mask = np.ma.getmaskarray(value).ravel()
        return [np.nan if bool(is_masked) else item for item, is_masked in zip(data, mask)]
    if isinstance(value, np.ndarray):
        return np.ravel(value).tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        if value is None or pd.isna(value):
            return []
    except Exception:
        if value is None:
            return []
    return [value]


def _gaia_epoch_table_value(value: object) -> object:
    """Convert one Astropy table cell without forcing masked ints through NaN."""
    if np.ma.is_masked(value):
        return pd.NA
    if isinstance(value, np.ma.MaskedArray):
        if value.shape == ():
            mask = np.ma.getmaskarray(value)
            if bool(np.asarray(mask).item()):
                return pd.NA
            return np.asarray(value.data).item()
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _gaia_epoch_table_to_frame(table: object) -> pd.DataFrame:
    """Convert Gaia TAP tables to pandas while preserving masked nullable columns."""
    if table is None or len(table) == 0:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table.copy()
    colnames = getattr(table, "colnames", None)
    if colnames is None:
        return table.to_pandas()
    data = {
        name: [_gaia_epoch_table_value(value) for value in table[name]]
        for name in colnames
    }
    return pd.DataFrame(data, columns=list(colnames))


def _gaia_epoch_broadcast_values(value: object, n_rows: int, default: object) -> list:
    values = _gaia_epoch_cell_values(value)
    if not values:
        return [default] * n_rows
    if len(values) == n_rows:
        return values
    if len(values) == 1:
        return values * n_rows
    return [default] * n_rows


def _normalize_gaia_epoch_tap_lightcurve(lc_all: pd.DataFrame) -> pd.DataFrame:
    """Expand Gaia epoch TAP rows to one row per G-band transit."""
    columns = ["source_id", "transit_id", "time", "mag", "band", "mag_error"]
    if lc_all.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for _, row in lc_all.iterrows():
        time_values = _gaia_epoch_cell_values(row.get("time"))
        mag_values = _gaia_epoch_cell_values(row.get("mag"))
        n_rows = max(len(time_values), len(mag_values))
        if n_rows == 0:
            continue
        if len(time_values) == 1 and n_rows > 1:
            time_values = time_values * n_rows
        if len(mag_values) == 1 and n_rows > 1:
            mag_values = mag_values * n_rows
        if len(time_values) != n_rows or len(mag_values) != n_rows:
            continue

        source_values = _gaia_epoch_broadcast_values(row.get("source_id"), n_rows, pd.NA)
        transit_values = _gaia_epoch_broadcast_values(row.get("transit_id"), n_rows, pd.NA)
        band_values = _gaia_epoch_broadcast_values(row.get("band"), n_rows, "G")
        mag_error_values = _gaia_epoch_broadcast_values(row.get("mag_error"), n_rows, np.nan)

        for i in range(n_rows):
            band = band_values[i]
            try:
                if pd.isna(band):
                    band = "G"
            except Exception:
                pass
            rows.append(
                {
                    "source_id": source_values[i],
                    "transit_id": transit_values[i],
                    "time": time_values[i],
                    "mag": mag_values[i],
                    "band": band,
                    "mag_error": mag_error_values[i],
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    out["source_id"] = pd.to_numeric(out["source_id"], errors="coerce")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["mag"] = pd.to_numeric(out["mag"], errors="coerce")
    out["mag_error"] = pd.to_numeric(out["mag_error"], errors="coerce")
    out = out.dropna(subset=["source_id", "time", "mag"])
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["source_id"] = out["source_id"].astype(np.int64)
    out["band"] = out["band"].fillna("G").astype(str)
    return out[columns]


def _summarize_neowise_lc(lc: pd.DataFrame) -> dict:
    if lc.empty:
        raise ValueError("empty NEOWISE light curve")
    w1 = pd.to_numeric(lc["w1mpro"], errors="coerce").dropna() if "w1mpro" in lc.columns else pd.Series(dtype=float)
    w2 = pd.to_numeric(lc["w2mpro"], errors="coerce").dropna() if "w2mpro" in lc.columns else pd.Series(dtype=float)
    return {
        "neowise_n_epochs": len(lc),
        "neowise_w1_range": float(w1.max() - w1.min()) if len(w1) >= 2 else np.nan,
        "neowise_w2_range": float(w2.max() - w2.min()) if len(w2) >= 2 else np.nan,
    }


def _summarize_flux_lc(lc: pd.DataFrame, group_col: str, n_col: str, total_col: str, range_col: str) -> dict:
    if lc.empty or "flux" not in lc.columns:
        raise ValueError("invalid flux light curve")
    flux_vals = pd.to_numeric(lc["flux"], errors="coerce").dropna()
    n_groups = lc[group_col].nunique(dropna=True) if group_col in lc.columns else 0
    return {
        n_col: int(n_groups),
        total_col: len(lc),
        range_col: float(flux_vals.max() - flux_vals.min()) if len(flux_vals) >= 2 else np.nan,
    }


def _summarize_count_lc(lc: pd.DataFrame, n_col: str) -> dict:
    if lc.empty:
        raise ValueError("empty light curve")
    return {n_col: len(lc)}


def _summarize_ogle_lc(lc: pd.DataFrame) -> dict:
    return _summarize_long_mag_lc(lc, "ogle_lc", ("i", "v"))


def _summarize_stripe82_lc(lc: pd.DataFrame) -> dict:
    return _summarize_long_mag_lc(lc, "stripe82_lc", ("u", "g", "r", "i", "z"))


def _summarize_vvvx_virac_lc(lc: pd.DataFrame) -> dict:
    summary = _summarize_long_mag_lc(lc, "vvvx_virac", ("z", "y", "j", "h", "ks"))
    summary["vvvx_virac_n_epochs"] = summary.pop("vvvx_virac_n_points")
    return summary


def _summarize_allwise_mep_lc(lc: pd.DataFrame) -> dict:
    if lc.empty:
        raise ValueError("empty AllWISE MEP light curve")
    out = {"allwise_mep_n_epochs": int(len(lc))}
    for band in ("w1", "w2", "w3", "w4"):
        col = f"{band}mpro"
        vals = pd.to_numeric(lc[col], errors="coerce").dropna() if col in lc.columns else pd.Series(dtype=float)
        out[f"allwise_mep_{band}_range"] = float(vals.max() - vals.min()) if len(vals) >= 2 else np.nan
    return out


# =============================================================================
# SIMBAD BATCH QUERY
# =============================================================================


def _xmatch_row_scalar(row: pd.Series, key: str, default=None):
    """Return a scalar from an XMatch ``to_pandas()`` row (handles duplicate column names)."""
    if key not in row.index:
        return default
    value = row[key]
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        value = value.iloc[0]
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return default
    return value


def _simbad_designation_tail(s: str) -> bool:
    """True if *s* looks like the numeric tail of a catalog id (e.g. UCAC4 667-019722, TYC 3159-1869-1)."""
    t = str(s).strip()
    if not t:
        return False
    return bool(re.match(r"^\d+-\d+(-\d+)?$", t))


def _normalize_simbad_xmatch_fields(
    row: pd.Series, main_id, otype, ang_dist,
) -> tuple[str, str, float]:
    """
    Repair occasional CDS XMatch / pandas quirks: split ``main_id`` with the numeric
    tail in the ``otype`` column, and coerce separation from ``angDist``.
    """
    if main_id is None or (isinstance(main_id, float) and pd.isna(main_id)):
        mid = ""
    else:
        mid = str(main_id).strip()

    if otype is None or (isinstance(otype, float) and pd.isna(otype)):
        ot = ""
    else:
        ot = str(otype).strip()

    if mid and " " not in mid and _simbad_designation_tail(ot):
        mid = f"{mid} {ot}"
        ot_fix = _xmatch_row_scalar(row, "main_type", "")
        if ot_fix is None or (isinstance(ot_fix, float) and pd.isna(ot_fix)) or (
            isinstance(ot_fix, str) and not ot_fix.strip()
        ):
            ot_raw = _xmatch_row_scalar(row, "otype", "")
            tail = str(ot_raw).strip() if ot_raw is not None and not pd.isna(ot_raw) else ""
            if tail and not _simbad_designation_tail(tail):
                ot_fix = ot_raw
            else:
                ot_fix = ""
        if ot_fix is None or (isinstance(ot_fix, float) and pd.isna(ot_fix)):
            ot = ""
        else:
            ot = str(ot_fix).strip()

    sep_out = np.nan
    try:
        v = float(ang_dist)
        if np.isfinite(v) and 0.0 <= v <= 3600.0:
            sep_out = v
    except (TypeError, ValueError):
        pass

    return mid, ot, sep_out


def query_simbad_batch(
    df: pd.DataFrame,
    radius_arcsec: float = SIMBAD_RADIUS_ARCSEC,
    cache_dir: Path | str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Query SIMBAD by coordinates via CDS XMatch.

    Adds columns: simbad_main_id, simbad_otype, simbad_nbref, simbad_sep_arcsec.
    """
    df = df.copy()
    for col in ("simbad_main_id", "simbad_otype", "simbad_nbref", "simbad_sep_arcsec"):
        df[col] = np.nan if col == "simbad_sep_arcsec" or col == "simbad_nbref" else ""

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    cache_keys = pd.Series(index=df.index, dtype=object)
    for idx in df.index[valid]:
        cache_keys.loc[idx] = _coord_cache_key(df.loc[idx, "ra"], df.loc[idx, "dec"], radius_arcsec)

    valid = valid & cache_keys.notna()
    if not valid.any():
        return df

    all_keys = set(cache_keys.loc[valid].astype(str))
    cached_by_key: dict[str, pd.Series] = {}
    if not refresh_cache:
        cached_by_key = _cached_rows_by_key(_read_vetting_cache(cache_dir, "simbad"), "coord_key", all_keys)

    simbad_cols = ("simbad_main_id", "simbad_otype", "simbad_nbref", "simbad_sep_arcsec")
    cached_idx = []
    for idx in df.index[valid]:
        key = str(cache_keys.loc[idx])
        cached = cached_by_key.get(key)
        if cached is None:
            continue
        for col in simbad_cols:
            if col in cached.index:
                df.loc[idx, col] = cached[col]
        cached_idx.append(idx)

    missing = valid & ~df.index.isin(cached_idx)
    if not missing.any():
        print(f"SIMBAD: served {len(cached_idx)} candidates from cache")
        return df

    if cached_idx:
        print(f"SIMBAD: served {len(cached_idx)} candidates from cache; querying {int(missing.sum())} misses")

    n = int(missing.sum())
    queried = _simbad_via_xmatch(df.loc[missing].copy(), pd.Series(True, index=df.index[missing]), n, radius_arcsec)
    for col in simbad_cols:
        df.loc[queried.index, col] = queried[col]

    cache_rows = pd.DataFrame({
        "coord_key": cache_keys.loc[missing].astype(str).values,
        "radius_arcsec": float(radius_arcsec),
        "simbad_main_id": df.loc[missing, "simbad_main_id"].fillna("").astype(str).values,
        "simbad_otype": df.loc[missing, "simbad_otype"].fillna("").astype(str).values,
        "simbad_nbref": pd.to_numeric(df.loc[missing, "simbad_nbref"], errors="coerce").fillna(0).astype(int).values,
        "simbad_sep_arcsec": pd.to_numeric(df.loc[missing, "simbad_sep_arcsec"], errors="coerce").values,
    })
    _write_vetting_cache(cache_dir, "simbad", cache_rows, key_cols=["coord_key"])
    return df


def _simbad_via_xmatch(
    df: pd.DataFrame, valid, n: int, radius_arcsec: float,
) -> pd.DataFrame:
    """SIMBAD lookup via CDS XMatch (reliable for small batches)."""
    print(f"SIMBAD: querying {n} candidates via CDS XMatch (radius={radius_arcsec}\")")

    source_table = Table()
    source_table["_idx"] = np.array(df.index[valid])
    source_table["ra"] = df.loc[valid, "ra"].values
    source_table["dec"] = df.loc[valid, "dec"].values

    matched = 0
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2="simbad",
            max_distance=radius_arcsec * u.arcsec,
            colRA1="ra", colDec1="dec",
        )
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            # Normalise column names (XMatch may return main_id or main_type)
            col_map = {}
            has_otype = any(str(c).lower() == "otype" for c in result_df.columns)
            for c in result_df.columns:
                cl = c.lower()
                if cl == "main_id" and c != "main_id":
                    col_map[c] = "main_id"
                elif cl == "main_type" and not has_otype and c != "main_type":
                    col_map[c] = "otype"
                elif cl == "nbref" and c != "nbref":
                    col_map[c] = "nbref"
            result_df = result_df.rename(columns=col_map)
            if result_df.columns.duplicated().any():
                result_df = result_df.loc[:, ~result_df.columns.duplicated(keep="first")]

            if "nbref" in result_df.columns:
                result_df["nbref"] = pd.to_numeric(result_df["nbref"], errors="coerce").fillna(0).astype(int)
                result_df = result_df.sort_values("nbref", ascending=False).drop_duplicates(subset="_idx", keep="first")
            elif "angDist" in result_df.columns:
                result_df = result_df.sort_values("angDist").drop_duplicates(subset="_idx", keep="first")
            else:
                result_df = result_df.drop_duplicates(subset="_idx", keep="first")

            for _, row in result_df.iterrows():
                idx = int(row["_idx"])
                if idx in df.index:
                    main_id = _xmatch_row_scalar(row, "main_id", "")
                    otype = _xmatch_row_scalar(row, "otype", "")
                    nbref_val = _xmatch_row_scalar(row, "nbref", 0)
                    ang_dist = _xmatch_row_scalar(row, "angDist", np.nan)
                    main_id, otype, sep = _normalize_simbad_xmatch_fields(
                        row, main_id, otype, ang_dist,
                    )

                    df.loc[idx, "simbad_main_id"] = str(main_id) if main_id is not None else ""
                    df.loc[idx, "simbad_otype"] = str(otype) if otype is not None else ""

                    nbref_num = pd.to_numeric(nbref_val, errors="coerce")
                    df.loc[idx, "simbad_nbref"] = int(nbref_num) if pd.notna(nbref_num) else 0
                    df.loc[idx, "simbad_sep_arcsec"] = round(float(sep), 3) if pd.notna(sep) else np.nan
                    matched += 1
    except Exception as e:
        raise RuntimeError(f"SIMBAD XMatch lookup failed: {e}") from e

    print(f"SIMBAD: {matched}/{n} candidates matched")
    return df
# =============================================================================
# GAIA DR3 VARIABILITY TABLES
# =============================================================================


def query_gaia_variability(
    df: pd.DataFrame,
    chunk_size: int = GAIA_VAR_CHUNK_SIZE,
    cache_dir: Path | str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Query Gaia DR3 vari_summary + vari_classifier_result.

    Adds columns: gaia_var_flag (bool), gaia_var_class, gaia_var_score.
    Requires a 'gaia_id' column with Gaia DR3 source_ids.
    """
    df = df.copy()
    df["gaia_var_flag"] = False
    df["gaia_var_class"] = ""
    df["gaia_var_score"] = np.nan

    if "gaia_id" not in df.columns:
        print("Warning: Gaia variability query requires 'gaia_id' column, skipping")
        return df

    valid = df["gaia_id"].notna()
    if not valid.any():
        return df

    # Normalize IDs
    gaia_ids = []
    idx_map = {}  # gaia_id_str -> list of df indices
    for idx, val in df.loc[valid, "gaia_id"].items():
        sid = _parse_gaia_source_id_str(val)
        if sid is None:
            continue
        gaia_ids.append(sid)
        idx_map.setdefault(sid, []).append(idx)
    gaia_ids = list(set(gaia_ids))

    if not gaia_ids:
        return df

    gaia_ids = sorted(gaia_ids)
    all_ids = set(gaia_ids)
    cached_by_id: dict[str, pd.Series] = {}
    if not refresh_cache:
        cached_by_id = _cached_rows_by_key(
            _read_vetting_cache(cache_dir, "gaia_variability"),
            "source_id",
            all_ids,
        )

    def _apply_gaia_var(sid: str, flag: object, class_name: object, score: object) -> int:
        applied = 0
        is_var = bool(flag) or bool(_safe_text(class_name))
        for idx in idx_map.get(sid, []):
            df.loc[idx, "gaia_var_flag"] = is_var
            df.loc[idx, "gaia_var_class"] = _safe_text(class_name)
            try:
                df.loc[idx, "gaia_var_score"] = float(score) if pd.notna(score) else np.nan
            except (TypeError, ValueError):
                df.loc[idx, "gaia_var_score"] = np.nan
            applied += 1
        return applied

    cached_ids = set(cached_by_id)
    for sid, cached in cached_by_id.items():
        _apply_gaia_var(
            sid,
            cached.get("gaia_var_flag", False),
            cached.get("gaia_var_class", ""),
            cached.get("gaia_var_score", np.nan),
        )

    missing_ids = [sid for sid in gaia_ids if refresh_cache or sid not in cached_ids]
    if not missing_ids:
        flagged = int(pd.Series([df.loc[idxs[0], "gaia_var_flag"] for idxs in idx_map.values()]).fillna(False).astype(bool).sum())
        classified = int((df["gaia_var_class"].fillna("").astype(str).str.strip() != "").sum())
        print(f"Gaia variability: served {len(gaia_ids)} source_ids from cache; {flagged} flagged, {classified} classified")
        return df

    if cached_ids and not refresh_cache:
        print(f"Gaia variability: served {len(cached_ids)} source_ids from cache; querying {len(missing_ids)} misses")
    else:
        print(f"Gaia variability: querying {len(missing_ids)} source_ids")

    try:
        effective_chunk_size = max(1, min(int(chunk_size), 100))
    except Exception:
        effective_chunk_size = 100
    if effective_chunk_size != int(chunk_size):
        print(f"  Gaia variability: reducing chunk size {chunk_size} -> {effective_chunk_size} for TAP compatibility")

    # Build an ordered list of working TAP services.
    preferred_tap_urls = [GAIA_AIP_TAP_URL] + [u for u in GAIA_TAP_URLS if u != GAIA_AIP_TAP_URL]
    test_query = f"SELECT source_id FROM gaiadr3.vari_summary WHERE source_id = {missing_ids[0]}"
    taps = _connect_gaia_taps_until_available(
        test_query,
        label="Gaia variability",
        urls=preferred_tap_urls,
    )

    def _query_chunk_rows(ids_chunk: list[str], query_builder: Callable[[list[str]], str], *, label: str):
        """Execute one chunk, retrying until TAP succeeds or the run is interrupted."""
        query = query_builder(ids_chunk)
        return list(_run_gaia_tap_query_until_success(taps, query, label=label))

    def _summary_query(ids_chunk: list[str]) -> str:
        ids_str = ",".join(ids_chunk)
        return f"""
            SELECT source_id,
                   in_vari_classification_result
            FROM gaiadr3.vari_summary
            WHERE source_id IN ({ids_str})
        """

    def _classifier_query(ids_chunk: list[str]) -> str:
        ids_str = ",".join(ids_chunk)
        return f"""
            SELECT source_id, best_class_name, best_class_score
            FROM gaiadr3.vari_classifier_result
            WHERE source_id IN ({ids_str})
        """

    # Query vari_summary (is it flagged as variable?)
    summary_results = {}
    for i in tqdm(range(0, len(missing_ids), effective_chunk_size), desc="Gaia vari_summary"):
        chunk = missing_ids[i : i + effective_chunk_size]
        rows = _query_chunk_rows(chunk, _summary_query, label=f"Gaia vari_summary chunk {i}")
        for row in rows:
            sid = str(row["source_id"])
            summary_results[sid] = bool(row["in_vari_classification_result"])

    # Query vari_classifier_result (what class?)
    classifier_results = {}
    for i in tqdm(range(0, len(missing_ids), effective_chunk_size), desc="Gaia vari_classifier"):
        chunk = missing_ids[i : i + effective_chunk_size]
        rows = _query_chunk_rows(chunk, _classifier_query, label=f"Gaia vari_classifier chunk {i}")
        for row in rows:
            sid = str(row["source_id"])
            classifier_results[sid] = (
                str(row["best_class_name"]),
                float(row["best_class_score"]) if row["best_class_score"] is not None else np.nan,
            )

    # Apply queried results and cache both hits and misses.
    matched = 0
    cache_rows = []
    for sid in missing_ids:
        indices = idx_map.get(sid, [])
        cls_info = classifier_results.get(sid)
        is_var = summary_results.get(sid, False) or cls_info is not None
        for idx in indices:
            df.loc[idx, "gaia_var_flag"] = is_var
            if cls_info is not None:
                df.loc[idx, "gaia_var_class"] = cls_info[0]
                df.loc[idx, "gaia_var_score"] = cls_info[1]
                matched += 1
        cache_rows.append({
            "source_id": sid,
            "gaia_var_flag": bool(is_var),
            "gaia_var_class": cls_info[0] if cls_info is not None else "",
            "gaia_var_score": cls_info[1] if cls_info is not None else np.nan,
        })

    _write_vetting_cache(
        cache_dir,
        "gaia_variability",
        pd.DataFrame(cache_rows),
        key_cols=["source_id"],
    )

    flagged = sum(
        1
        for sid in gaia_ids
        if bool(df.loc[idx_map[sid][0], "gaia_var_flag"])
    )
    classified = int((df["gaia_var_class"].fillna("").astype(str).str.strip() != "").sum())
    print(f"Gaia variability: {flagged} flagged as variable, {classified} with classification")
    return df


# =============================================================================
# ASAS-SN VARIABLE STAR CATALOG (VizieR II/366)
# =============================================================================


def _load_asassn_local_catalog(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    if key in _asassn_cache:
        return _asassn_cache[key]
    if not path.is_file():
        raise FileNotFoundError(f"ASAS-SN variables local CSV not found: {path}")
    print(f"ASAS-SN variables: loading local catalog {path.name}...")
    cat = pd.read_csv(path, low_memory=False)
    need = {"RAJ2000", "DEJ2000", "ID"}
    if not need.issubset(cat.columns):
        raise ValueError(f"ASAS-SN variables CSV missing columns {need}, got {list(cat.columns)[:12]}...")
    _asassn_cache[key] = cat
    return cat


def _asassn_via_local(
    df: pd.DataFrame,
    valid: pd.Series,
    n_valid: int,
    radius_arcsec: float,
    local_csv: Path | str | None,
) -> pd.DataFrame:
    path = Path(local_csv) if local_csv is not None else ASASSN_VAR_LOCAL_CSV
    cat = _load_asassn_local_catalog(path)
    if cat.empty:
        return df

    print(f"ASAS-SN variables: crossmatching {n_valid} candidates to local catalog (radius={radius_arcsec}\")")
    src_index = df.index[valid]
    src_coords = SkyCoord(
        ra=df.loc[valid, "ra"].values,
        dec=df.loc[valid, "dec"].values,
        unit="deg",
    )
    cat_coords = SkyCoord(
        ra=cat["RAJ2000"].to_numpy(dtype=float),
        dec=cat["DEJ2000"].to_numpy(dtype=float),
        unit="deg",
    )
    idx_cat, idx_src, sep2d, _ = src_coords.search_around_sky(cat_coords, radius_arcsec * u.arcsec)
    if len(idx_src) == 0:
        print("ASAS-SN variables: 0 matches")
        return df

    matched = cat.iloc[np.asarray(idx_cat, dtype=int)].copy().reset_index(drop=True)
    matched["candidate_idx"] = src_index.to_numpy()[np.asarray(idx_src, dtype=int)]
    matched["sep_arcsec"] = np.asarray(sep2d.arcsec, dtype=float)
    type_col = "ML_classification" if "ML_classification" in matched.columns else None
    per_col = "Period" if "Period" in matched.columns else None

    n_matched = 0
    for cand_idx, group in matched.groupby("candidate_idx", sort=False):
        best = group.sort_values("sep_arcsec", na_position="last").iloc[0]
        df.loc[cand_idx, "asassn_var_name"] = _safe_text(best.get("ID"))
        if type_col:
            df.loc[cand_idx, "asassn_var_type"] = _safe_text(best.get(type_col))
        if per_col:
            try:
                df.loc[cand_idx, "asassn_var_period"] = float(best.get(per_col))
            except (TypeError, ValueError):
                pass
        n_matched += 1

    print(f"ASAS-SN variables: {n_matched} matches")
    return df


def _asassn_via_tap(
    df: pd.DataFrame,
    valid: pd.Series,
    n_valid: int,
    radius_arcsec: float,
    chunk_size: int,
) -> pd.DataFrame:
    print(f"ASAS-SN variables: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")
    coords_df = pd.DataFrame({
        "_idx": df.index[valid],
        "ra": df.loc[valid, "ra"].values,
        "dec": df.loc[valid, "dec"].values,
    })
    result = batch_tap_crossmatch(
        coords_df,
        tap_url=VIZIER_TAP_URL,
        catalog_table=f'"{ASASSN_VAR_CATALOG}"',
        select_cols='c."ASASSN-V", c."Per", c."Type"',
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        match_radius_arcsec=radius_arcsec,
        chunk_size=chunk_size,
        n_workers=4,
        verbose=True,
        desc="ASAS-SN II/366 TAP",
        raise_on_all_failed=True,
        raise_on_failed_chunk=True,
    )

    if result.empty:
        print("ASAS-SN variables: 0 matches")
        return df
    sep_col = "sep_arcsec" if "sep_arcsec" in result.columns else "angDist"
    if sep_col in result.columns:
        result = result.sort_values(sep_col).drop_duplicates(subset="_idx", keep="first")

    name_keys = ("ASASSN-V", "ASASSN_V", "ASASSNV")
    per_keys = ("Per", "PER")
    type_keys = ("Type", "TYPE")

    def _pick(row: pd.Series, keys: tuple[str, ...]) -> object:
        for k in keys:
            if k in row.index and pd.notna(row.get(k)):
                return row.get(k)
        return None

    matched = 0
    for _, row in result.iterrows():
        try:
            idx = int(row["_idx"])
        except (TypeError, ValueError):
            continue
        if idx not in df.index:
            continue
        name = _pick(row, name_keys)
        per = _pick(row, per_keys)
        vtype = _pick(row, type_keys)
        df.loc[idx, "asassn_var_name"] = _safe_text(name)
        df.loc[idx, "asassn_var_type"] = _safe_text(vtype)
        try:
            df.loc[idx, "asassn_var_period"] = float(per) if per is not None and pd.notna(per) else np.nan
        except (TypeError, ValueError):
            pass
        matched += 1

    print(f"ASAS-SN variables: {matched} matches")
    return df


def crossmatch_asassn_variables(
    df: pd.DataFrame,
    radius_arcsec: float = ASASSN_VAR_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "local"] = "local",
    local_csv: Path | str | None = None,
) -> pd.DataFrame:
    """
    Crossmatch against the ASAS-SN Variable Stars Database (VizieR II/366).
    Adds columns: asassn_var_name, asassn_var_type, asassn_var_period.

    method='local' — ``input/asassn_variables_*.csv`` (or *local_csv*) + SkyCoord.
    method='tap'    — VizieR TAP upload to ``II/366/catv2021``.
    """
    df = df.copy()
    df["asassn_var_name"] = ""
    df["asassn_var_type"] = ""
    df["asassn_var_period"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    if method == "tap":
        return _asassn_via_tap(df, valid, n_valid, radius_arcsec, chunk_size)
    return _asassn_via_local(df, valid, n_valid, radius_arcsec, local_csv)

# =============================================================================
# MICROLENSING EVENT CATALOGS
# =============================================================================


MICROLENS_SOURCE_PRIORITY = {
    "OGLE-EWS": 0,
    "KMTNet": 1,
    "MOA": 2,
    "OGLE-IV": 3,
}



def fetch_microlensing_event_catalog(
    cache_dir: Path | None = None,
    *,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    kmtnet_path = Path(__file__).resolve().parent.parent / "input" / "kmtnet_220326.csv"
    if kmtnet_path.exists():
        df_kmt = pd.read_csv(kmtnet_path)
        kmt_mapped = pd.DataFrame()
        kmt_mapped["event_id"] = df_kmt["event"].astype(str)
        kmt_mapped["source"] = "KMTNet"
        kmt_mapped["alias"] = ""
        kmt_mapped["ra"] = pd.to_numeric(df_kmt["ra_deg"], errors="coerce")
        kmt_mapped["dec"] = pd.to_numeric(df_kmt["dec_deg"], errors="coerce")
        kmt_mapped["timescale_days"] = pd.to_numeric(df_kmt["t_e"], errors="coerce")
        kmt_mapped["timescale_kind"] = "te"
        kmt_mapped["status"] = df_kmt["classification"].astype(str)
        kmt_mapped["source_url"] = ""
        kmt_mapped["event_year"] = kmt_mapped["event_id"].str.extract(r"-(20\d{2})-")[0].astype(float)
        frames.append(kmt_mapped)

    ogle_path = Path(__file__).resolve().parent.parent / "input" / "ogle_ews_220326.csv"
    if ogle_path.exists():
        df_ogle = pd.read_csv(ogle_path)
        ogle_mapped = pd.DataFrame()
        ogle_mapped["event_id"] = "OGLE-" + df_ogle["event"].astype(str)
        ogle_mapped["source"] = "OGLE-EWS"
        ogle_mapped["alias"] = df_ogle["event"].astype(str)
        ogle_mapped["ra"] = pd.to_numeric(df_ogle["ra_deg"], errors="coerce")
        ogle_mapped["dec"] = pd.to_numeric(df_ogle["dec_deg"], errors="coerce")
        ogle_mapped["timescale_days"] = pd.to_numeric(df_ogle["tau"], errors="coerce")
        ogle_mapped["timescale_kind"] = "tau"
        ogle_mapped["status"] = ""
        ogle_mapped["source_url"] = ""
        ogle_mapped["event_year"] = ogle_mapped["event_id"].str.extract(r"-(199\d|20\d{2})-")[0].astype(float)
        frames.append(ogle_mapped)

    asassn_ml_path = Path(__file__).resolve().parent.parent / "input" / "asas_sn_microlens.csv"
    if asassn_ml_path.exists():
        df_asassn_ml = pd.read_csv(asassn_ml_path, low_memory=False)
        if {"ra_deg", "dec_deg"}.issubset(df_asassn_ml.columns):
            ra_vals = pd.to_numeric(df_asassn_ml["ra_deg"], errors="coerce")
            dec_vals = pd.to_numeric(df_asassn_ml["dec_deg"], errors="coerce")
        elif {"raj2000", "dej2000"}.issubset(df_asassn_ml.columns):
            coords = []
            for ra_str, dec_str in zip(df_asassn_ml["raj2000"], df_asassn_ml["dej2000"]):
                try:
                    c = SkyCoord(ra=str(ra_str), dec=str(dec_str), unit=(u.hourangle, u.deg))
                    coords.append((c.ra.deg, c.dec.deg))
                except Exception:
                    coords.append((np.nan, np.nan))
            ra_vals = pd.Series([x[0] for x in coords], index=df_asassn_ml.index)
            dec_vals = pd.Series([x[1] for x in coords], index=df_asassn_ml.index)
        else:
            ra_vals = pd.Series(np.nan, index=df_asassn_ml.index)
            dec_vals = pd.Series(np.nan, index=df_asassn_ml.index)

        valid_asassn_ml = (
            ra_vals.notna()
            & dec_vals.notna()
            & np.isfinite(ra_vals)
            & np.isfinite(dec_vals)
            & ra_vals.between(0.0, 360.0)
            & dec_vals.between(-90.0, 90.0)
        )
        if valid_asassn_ml.any():
            asas_mapped = pd.DataFrame()
            asas_mapped["event_id"] = df_asassn_ml.get("id", df_asassn_ml.index).astype(str)
            asas_mapped["source"] = "ASAS-SN microlens"
            asas_mapped["alias"] = df_asassn_ml.get("other_names", "").astype(str) if "other_names" in df_asassn_ml.columns else ""
            asas_mapped["ra"] = ra_vals
            asas_mapped["dec"] = dec_vals
            asas_mapped["timescale_days"] = np.nan
            asas_mapped["timescale_kind"] = ""
            asas_mapped["status"] = df_asassn_ml.get("variable_type", "").astype(str) if "variable_type" in df_asassn_ml.columns else ""
            asas_mapped["source_url"] = ""
            asas_mapped["event_year"] = asas_mapped["event_id"].str.extract(r"(20\d{2})")[0].astype(float)
            frames.append(asas_mapped.loc[valid_asassn_ml].copy())
        else:
            print(
                "Microlensing catalogs: found input/asas_sn_microlens.csv but "
                "coordinates did not validate under the expected schema; skipping it"
            )

    if not frames:
        print("Microlensing catalogs: no local CSVs found in input/")
        return pd.DataFrame()

    union = pd.concat(frames, ignore_index=True)
    union = union.dropna(subset=["ra", "dec"])
    union = union[np.isfinite(union["ra"]) & np.isfinite(union["dec"])].copy()
    union = union.drop_duplicates(subset=["source", "event_id"], keep="first")
    union["source_rank"] = union["source"].map(MICROLENS_SOURCE_PRIORITY).fillna(999).astype(int)
    union = union.sort_values(["source_rank", "event_year", "event_id"], na_position="last").reset_index(drop=True)
    return union

def _dedupe_join(values: list[str]) -> str:
    out = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return ",".join(out)

def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _is_missing_catalog_text(value: object) -> bool:
    text = _safe_text(value).strip()
    return not text or text.lower() in {"-", "nan", "none", "null", "unknown"}


def _normalise_asassn_transient_name(row: pd.Series) -> str:
    for col in ("asassn_id", "other_ids", "atel_tns"):
        value = _safe_text(row.get(col))
        if _is_missing_catalog_text(value):
            continue
        return f"ASAS-SN:{value}"
    return "ASAS-SN:transient"


def _infer_asassn_transient_type(row: pd.Series) -> str:
    cls = _safe_text(row.get("spectroscopic_class"))
    if not _is_missing_catalog_text(cls):
        return cls

    comments = _safe_text(row.get("comments"))
    if not comments:
        return ""

    type_match = re.search(r"\bType\s+([A-Za-z0-9.+/-]+)", comments, flags=re.IGNORECASE)
    if type_match:
        return f"Type {type_match.group(1)}"

    lowered = comments.lower()
    if "cv candidate" in lowered or "cataclysmic" in lowered:
        return "CV candidate"
    if "microlensing" in lowered or "ulens" in lowered:
        return "Microlensing candidate"
    if "nova" in lowered:
        return "Nova candidate"
    if "sn candidate" in lowered or "supernova" in lowered:
        return "SN candidate"
    if "transient" in lowered:
        return "Transient"

    first_clause = comments.split(",", 1)[0].strip()
    return first_clause[:80]


def _parse_asassn_transient_redshift(row: pd.Series) -> float:
    for text in (_safe_text(row.get("comments")), _safe_text(row.get("spectroscopic_class"))):
        match = re.search(r"\bz\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def _normalise_asassn_discovery_date(value: object) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not match:
        return text[:10]
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

def crossmatch_microlensing_catalogs(
    df: pd.DataFrame,
    radius_arcsec: float = OGLE_MICROLENS_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "xmatch"] = "tap",
) -> pd.DataFrame:
    """
    Crossmatch against known microlensing event catalogs.

    Current coverage:
      - OGLE EWS archive/history
      - KMTNet yearly event lists
      - MOA alert/archive history

    The survey tables are fetched once, cached locally, and then crossmatched
    in-memory via SkyCoord. If all three survey adapters fail, this falls back
    to the older published OGLE-IV VizieR catalog path.

    Adds columns: microlens_match, microlens_catalog, microlens_name,
    microlens_alt_name, microlens_te_days, microlens_sep_arcsec.
    """
    df = df.copy()
    df["microlens_match"] = False
    df["microlens_catalog"] = ""
    df["microlens_name"] = ""
    df["microlens_alt_name"] = ""
    df["microlens_te_days"] = np.nan
    df["microlens_sep_arcsec"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    print(f"Microlensing catalogs: crossmatching {n_valid} candidates via cached survey union (radius={radius_arcsec}\")")
    catalog = fetch_microlensing_event_catalog(show_tqdm=True)

    if catalog.empty:
        print("Microlensing catalogs: 0 matches")
        return df

    src_index = df.index[valid]
    src_coords = SkyCoord(
        ra=df.loc[valid, "ra"].values,
        dec=df.loc[valid, "dec"].values,
        unit="deg",
    )
    cat_coords = SkyCoord(
        ra=catalog["ra"].to_numpy(dtype=float),
        dec=catalog["dec"].to_numpy(dtype=float),
        unit="deg",
    )

    idx_cat, idx_src, sep2d, _ = src_coords.search_around_sky(cat_coords, radius_arcsec * u.arcsec)
    if len(idx_src) == 0:
        print("Microlensing catalogs: 0 matches")
        return df

    matched_rows = catalog.iloc[np.asarray(idx_cat, dtype=int)].copy().reset_index(drop=True)
    matched_rows["candidate_idx"] = src_index.to_numpy()[np.asarray(idx_src, dtype=int)]
    matched_rows["sep_arcsec"] = np.asarray(sep2d.arcsec, dtype=float)
    matched_rows["source_rank"] = matched_rows["source"].map(MICROLENS_SOURCE_PRIORITY).fillna(999).astype(int)

    matched = 0
    for candidate_idx, group in matched_rows.groupby("candidate_idx", sort=False):
        ordered = group.sort_values(["sep_arcsec", "source_rank", "event_id"], na_position="last")
        primary = ordered.iloc[0]
        alt_bits = []
        primary_alias = _safe_text(primary.get("alias"))
        if primary_alias:
            alt_bits.append(primary_alias)
        for _, extra in ordered.iloc[1:].head(3).iterrows():
            alt_bits.append(f"{_safe_text(extra.get('source'))}:{_safe_text(extra.get('event_id'))}")

        timescale = pd.to_numeric(primary.get("timescale_days"), errors="coerce")
        kind = _safe_text(primary.get("timescale_kind")).lower()
        df.loc[candidate_idx, "microlens_match"] = True
        df.loc[candidate_idx, "microlens_catalog"] = _safe_text(primary.get("source"))
        df.loc[candidate_idx, "microlens_name"] = _safe_text(primary.get("event_id"))
        df.loc[candidate_idx, "microlens_alt_name"] = _dedupe_join(alt_bits)
        df.loc[candidate_idx, "microlens_te_days"] = float(timescale) if pd.notna(timescale) and kind in {"te", "tau"} else np.nan
        df.loc[candidate_idx, "microlens_sep_arcsec"] = float(primary["sep_arcsec"])
        matched += 1

    print(f"Microlensing catalogs: {matched} matches")
    return df


# =============================================================================
# ZTF PERIODIC VARIABLES (Chen+ 2020, VizieR J/ApJS/249/18)
# =============================================================================


def crossmatch_ztf_variables(
    df: pd.DataFrame,
    radius_arcsec: float = ZTF_VAR_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "xmatch"] = "tap",
) -> pd.DataFrame:
    """
    Crossmatch against ZTF periodic variable catalog (Chen+ 2020).

    method='tap'    — batch VizieR TAP upload (best for large batches).
    method='xmatch' — CDS XMatch service (reliable for small batches).

    ~781k periodic variables from ZTF DR2.  Adds columns: ztf_var_type, ztf_var_period, ztf_var_amp.
    """
    df = df.copy()
    df["ztf_var_type"] = ""
    df["ztf_var_period"] = np.nan
    df["ztf_var_amp"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())

    if method == "xmatch":
        print(f"ZTF variables: crossmatching {n_valid} candidates via CDS XMatch (radius={radius_arcsec}\")")
        source_table = Table()
        source_table["_idx"] = np.array(df.index[valid])
        source_table["ra"] = df.loc[valid, "ra"].values
        source_table["dec"] = df.loc[valid, "dec"].values

        try:
            result_tab = XMatch.query(
                cat1=source_table,
                cat2="vizier:J/ApJS/249/18/table2",
                max_distance=radius_arcsec * u.arcsec,
                colRA1="ra", colDec1="dec",
            )
            result = result_tab.to_pandas() if result_tab is not None and len(result_tab) > 0 else pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"ZTF variables XMatch lookup failed: {e}") from e

        if not result.empty and "angDist" in result.columns:
            result = result.sort_values("angDist").drop_duplicates(subset="_idx", keep="first")
    else:
        print(f"ZTF variables: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")
        coords_df = pd.DataFrame({
            "_idx": df.index[valid],
            "ra": df.loc[valid, "ra"].values,
            "dec": df.loc[valid, "dec"].values,
        })
        result = batch_tap_crossmatch(
            coords_df,
            tap_url=VIZIER_TAP_URL,
            catalog_table='"J/ApJS/249/18/table2"',
            select_cols='c."Type", c."Per", c."gAmp", c."rAmp"',
            ra_col="RAJ2000",
            dec_col="DEJ2000",
            match_radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            n_workers=4,
            verbose=True,
            desc="ZTF vars TAP",
            raise_on_all_failed=True,
            raise_on_failed_chunk=True,
        )
        if not result.empty:
            result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")

    matched = 0
    if not result.empty:
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "ztf_var_type"] = str(row.get("Type", "") or "")
                try:
                    df.loc[idx, "ztf_var_period"] = float(row["Per"]) if pd.notna(row.get("Per")) else np.nan
                except (ValueError, TypeError):
                    pass
                # Use g-band amplitude, fall back to r-band
                amp = np.nan
                for amp_col in ("gAmp", "rAmp"):
                    try:
                        v = row.get(amp_col)
                        if pd.notna(v):
                            amp = float(v)
                            break
                    except (ValueError, TypeError):
                        pass
                df.loc[idx, "ztf_var_amp"] = amp
                matched += 1

    print(f"ZTF variables: {matched} matches")
    return df


# =============================================================================
# TNS (TRANSIENT NAME SERVER)
# =============================================================================


def _load_tns_catalog(csv_paths: list[Path]) -> tuple[pd.DataFrame, SkyCoord] | None:
    """Load and merge TNS catalogs from CSVs. Returns (cat, cat_coord) or None if no data."""
    cache_key = tuple(str(p) for p in csv_paths)
    if cache_key in _tns_cache:
        return _tns_cache[cache_key]

    rows: list[dict] = []

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            continue
        try:
            if csv_path.name == "tns_public_objects.csv":
                cat = pd.read_csv(csv_path, skiprows=1, low_memory=False)
                if cat.empty:
                    continue
                # Columns: name_prefix, name, ra, declination, redshift, type, discoverydate
                for _, r in cat.iterrows():
                    ra = r.get("ra")
                    dec = r.get("declination")
                    if pd.isna(ra) or pd.isna(dec):
                        continue
                    try:
                        ra_f = float(ra)
                        dec_f = float(dec)
                    except (ValueError, TypeError):
                        continue
                    prefix = str(r.get("name_prefix", "") or "").strip()
                    name = str(r.get("name", "") or "").strip()
                    tns_name = f"{prefix}{name}" if prefix and name else (name or prefix)
                    rows.append({
                        "ra": ra_f,
                        "dec": dec_f,
                        "name": tns_name,
                        "type": str(r.get("type", "") or "").strip(),
                        "redshift": r.get("redshift"),
                        "discovery_date": str(r.get("discoverydate", "") or "")[:10] if pd.notna(r.get("discoverydate")) else "",
                    })
            elif csv_path.name == "tns_sne.csv":
                cat = pd.read_csv(csv_path, low_memory=False)
                if cat.empty:
                    continue
                for _, r in cat.iterrows():
                    ra_str = r.get("RA")
                    dec_str = r.get("DEC")
                    if pd.isna(ra_str) or pd.isna(dec_str):
                        continue
                    try:
                        c = SkyCoord(ra=str(ra_str), dec=str(dec_str), unit=(u.hourangle, u.deg))
                        ra_f = c.ra.deg
                        dec_f = c.dec.deg
                    except Exception:
                        continue
                    name = str(r.get("Name", "") or "").strip()
                    obj_type = str(r.get("Obj. Type", "") or "").strip()
                    disc_col = "Discovery Date (UT)"
                    disc_val = r.get(disc_col, "") if disc_col in r else ""
                    disc_date = str(disc_val)[:10] if pd.notna(disc_val) else ""
                    rows.append({
                        "ra": ra_f,
                        "dec": dec_f,
                        "name": name,
                        "type": obj_type,
                        "redshift": r.get("Redshift"),
                        "discovery_date": disc_date,
                    })
            elif csv_path.name == "asassn_transients.csv":
                cat = pd.read_csv(csv_path, low_memory=False)
                if cat.empty:
                    continue
                for _, r in cat.iterrows():
                    ra_f = pd.to_numeric(r.get("ra_deg"), errors="coerce")
                    dec_f = pd.to_numeric(r.get("dec_deg"), errors="coerce")
                    if pd.isna(ra_f) or pd.isna(dec_f):
                        ra_str = r.get("ra")
                        dec_str = r.get("dec")
                        if pd.isna(ra_str) or pd.isna(dec_str):
                            continue
                        try:
                            c = SkyCoord(ra=str(ra_str), dec=str(dec_str), unit=(u.hourangle, u.deg))
                            ra_f = c.ra.deg
                            dec_f = c.dec.deg
                        except Exception:
                            continue
                    try:
                        ra_f = float(ra_f)
                        dec_f = float(dec_f)
                    except (ValueError, TypeError):
                        continue
                    rows.append({
                        "ra": ra_f,
                        "dec": dec_f,
                        "name": _normalise_asassn_transient_name(r),
                        "type": _infer_asassn_transient_type(r),
                        "redshift": _parse_asassn_transient_redshift(r),
                        "discovery_date": _normalise_asassn_discovery_date(r.get("discovery_ut")),
                    })
        except Exception as e:
            print(f"TNS: warning loading {csv_path.name}: {e}")
            continue

    if not rows:
        _tns_cache[cache_key] = None
        return None

    cat = pd.DataFrame(rows)
    cat = cat.dropna(subset=["ra", "dec"])
    if cat.empty:
        _tns_cache[cache_key] = None
        return None
    cat_coord = SkyCoord(ra=cat["ra"].values, dec=cat["dec"].values, unit="deg")
    _tns_cache[cache_key] = (cat, cat_coord)
    print(f"TNS: loaded local catalog: {len(cat)} entries from {len(csv_paths)} file(s)")
    return cat, cat_coord


def crossmatch_tns(
    df: pd.DataFrame,
    radius_arcsec: float = TNS_RADIUS_ARCSEC,
    tns_api_key: str | None = None,
    local_csvs: list[Path] | None = None,
) -> pd.DataFrame:
    """
    Crossmatch against the Transient Name Server via local catalogs.

    Uses tns_public_objects.csv and tns_sne.csv in ~/code/malca/input (or local_csvs override).

    Adds columns: tns_name, tns_type, tns_redshift, tns_disc_date.
    """
    df = df.copy()
    df["tns_name"] = ""
    df["tns_type"] = ""
    df["tns_redshift"] = np.nan
    df["tns_disc_date"] = ""

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    paths = list(local_csvs) if local_csvs else TNS_LOCAL_CSVS
    loaded = _load_tns_catalog(paths)

    if loaded is None:
        print("TNS: no catalog data loaded (check input/tns_public_objects.csv, input/tns_sne.csv, input/asassn_transients.csv), skipping")
        return df

    cat, cat_coord = loaded
    print(f"TNS: crossmatching {n_valid} candidates via local catalog (radius={radius_arcsec}\")")

    src_coord = SkyCoord(
        ra=df.loc[valid, "ra"].values,
        dec=df.loc[valid, "dec"].values,
        unit="deg",
    )
    idx_cat, sep2d, _ = src_coord.match_to_catalog_sky(cat_coord)
    max_sep = radius_arcsec * u.arcsec

    matched = 0
    for i, df_idx in enumerate(df.index[valid]):
        if sep2d[i] <= max_sep:
            row = cat.iloc[idx_cat[i]]
            df.loc[df_idx, "tns_name"] = str(row.get("name", "") or "")
            df.loc[df_idx, "tns_type"] = str(row.get("type", "") or "")
            try:
                z = row.get("redshift")
                df.loc[df_idx, "tns_redshift"] = float(z) if pd.notna(z) and str(z).strip() else np.nan
            except (ValueError, TypeError):
                pass
            df.loc[df_idx, "tns_disc_date"] = str(row.get("discovery_date", "") or "")
            matched += 1

    print(f"TNS: {matched} transient matches")
    return df


# =============================================================================
# GAIA DR3 ECLIPSING BINARY PARAMETERS
# =============================================================================


def query_gaia_eb_params(
    df: pd.DataFrame,
    chunk_size: int = GAIA_VAR_CHUNK_SIZE,
    cache_dir: Path | str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Query Gaia DR3 vari_eclipsing_binary for detailed EB parameters.

    Only queries sources already classified as ECL by the Gaia classifier.
    Adds columns: gaia_eb_period, gaia_eb_morph, gaia_eb_global_ranking.
    """
    df = df.copy()
    df["gaia_eb_period"] = np.nan
    df["gaia_eb_morph"] = ""
    df["gaia_eb_global_ranking"] = np.nan

    if "gaia_id" not in df.columns:
        return df

    # Only look up sources classified as ECL
    ecl_mask = df.get("gaia_var_class", pd.Series("", index=df.index)).str.upper() == "ECL"
    if not ecl_mask.any():
        print("Gaia EB params: no ECL-classified sources, skipping")
        return df

    gaia_ids = []
    idx_map = {}
    for idx, val in df.loc[ecl_mask, "gaia_id"].items():
        sid = _parse_gaia_source_id_str(val)
        if sid is None:
            continue
        gaia_ids.append(sid)
        idx_map.setdefault(sid, []).append(idx)
    gaia_ids = sorted(set(gaia_ids))

    if not gaia_ids:
        return df

    all_ids = set(gaia_ids)
    cached_by_id: dict[str, pd.Series] = {}
    if not refresh_cache:
        cached_by_id = _cached_rows_by_key(_read_vetting_cache(cache_dir, "gaia_eb"), "source_id", all_ids)

    matched = 0
    for sid, cached in cached_by_id.items():
        period = pd.to_numeric(cached.get("gaia_eb_period"), errors="coerce")
        morph = _safe_text(cached.get("gaia_eb_morph"))
        ranking = pd.to_numeric(cached.get("gaia_eb_global_ranking"), errors="coerce")
        for idx in idx_map.get(sid, []):
            df.loc[idx, "gaia_eb_period"] = float(period) if pd.notna(period) else np.nan
            df.loc[idx, "gaia_eb_morph"] = morph
            df.loc[idx, "gaia_eb_global_ranking"] = float(ranking) if pd.notna(ranking) else np.nan
            if pd.notna(period):
                matched += 1

    missing_ids = [sid for sid in gaia_ids if refresh_cache or sid not in cached_by_id]
    if not missing_ids:
        print(f"Gaia EB params: served {len(gaia_ids)} ECL-classified sources from cache")
        print(f"Gaia EB params: {matched} sources with orbital parameters")
        return df

    n_ecl = len(missing_ids)
    if cached_by_id and not refresh_cache:
        print(f"Gaia EB params: served {len(cached_by_id)} sources from cache; querying {n_ecl} misses")
    else:
        print(f"Gaia EB params: querying {n_ecl} ECL-classified sources")
    test_query = f"SELECT source_id FROM gaiadr3.vari_eclipsing_binary WHERE source_id = {missing_ids[0]}"
    taps = _connect_gaia_taps_until_available(test_query, label="Gaia EB params")

    eb_results = {}
    for i in tqdm(range(0, len(missing_ids), chunk_size), desc="Gaia EB params"):
        chunk = missing_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id, frequency, model_type, global_ranking
            FROM gaiadr3.vari_eclipsing_binary
            WHERE source_id IN ({ids_str})
        """
        result = _run_gaia_tap_query_until_success(taps, query, label=f"Gaia EB chunk {i}")
        for row in result:
            sid = str(row["source_id"])
            freq = row["frequency"]
            period = 1.0 / float(freq) if freq and float(freq) > 0 else np.nan
            morph = str(row["model_type"]) if row["model_type"] else ""
            ranking = float(row["global_ranking"]) if row["global_ranking"] is not None else np.nan
            eb_results[sid] = (period, morph, ranking)

    cache_rows = []
    for sid in missing_ids:
        indices = idx_map.get(sid, [])
        info = eb_results.get(sid)
        if info is None:
            cache_rows.append({
                "source_id": sid,
                "gaia_eb_period": np.nan,
                "gaia_eb_morph": "",
                "gaia_eb_global_ranking": np.nan,
            })
            continue
        period, morph, ranking = info
        for idx in indices:
            df.loc[idx, "gaia_eb_period"] = period
            df.loc[idx, "gaia_eb_morph"] = morph
            df.loc[idx, "gaia_eb_global_ranking"] = ranking
            matched += 1
        cache_rows.append({
            "source_id": sid,
            "gaia_eb_period": period,
            "gaia_eb_morph": morph,
            "gaia_eb_global_ranking": ranking,
        })

    _write_vetting_cache(cache_dir, "gaia_eb", pd.DataFrame(cache_rows), key_cols=["source_id"])

    print(f"Gaia EB params: {matched} sources with orbital parameters")
    return df


# =============================================================================
# ALeRCE ZTF BROKER
# =============================================================================


def _alerce_request_with_retry(method, url, max_retries=VETTING_SIMBAD_MAX_RETRIES, **kwargs):
    """HTTP request with retry on 429 rate-limit responses."""
    kwargs.setdefault("timeout", VETTING_HTTP_TIMEOUT)
    for attempt in range(max_retries):
        try:
            resp = method(url, **kwargs)
            if resp.status_code == 429:
                time.sleep(min(2 ** attempt, VETTING_BACKOFF_CAP))
                continue
            return resp
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def _alerce_query_single(ra: float, dec: float, radius_arcsec: float) -> dict | None:
    """Cone search + probability lookup for one candidate. Returns result dict or None."""
    defaults = {
        "alerce_oid": "", "alerce_ndet": 0,
        "alerce_lc_class": "", "alerce_lc_prob": np.nan,
        "alerce_stamp_class": "", "alerce_stamp_prob": np.nan,
    }

    # Cone search
    resp = _alerce_request_with_retry(
        requests.get,
        f"{ALERCE_API_BASE}/ztf/v1/objects/",
        params={
            "ra": ra,
            "dec": dec,
            "radius": radius_arcsec,
            "page_size": 5,
            "order_by": "ndet",
            "order_mode": "DESC",
        },
    )
    if resp is None or resp.status_code != 200:
        return None
    items = resp.json().get("items", [])
    if not items:
        return None

    obj = items[0]
    oid = obj.get("oid", "")
    result = dict(defaults)
    result["alerce_oid"] = oid
    result["alerce_ndet"] = int(obj.get("ndet", 0))

    # Probability lookup
    if oid:
        resp = _alerce_request_with_retry(
            requests.get,
            f"{ALERCE_API_BASE}/ztf/v1/objects/{oid}/probabilities",
        )
        if resp is not None and resp.status_code == 200:
            probs = resp.json()
            lc_probs = [p for p in probs if p.get("classifier_name", "").startswith("lc_classifier")]
            if lc_probs:
                best_lc = max(lc_probs, key=lambda p: p.get("probability", 0))
                result["alerce_lc_class"] = best_lc.get("class_name", "")
                result["alerce_lc_prob"] = best_lc.get("probability", np.nan)
            stamp_probs = [p for p in probs if p.get("classifier_name", "").startswith("stamp_classifier")]
            if stamp_probs:
                best_stamp = max(stamp_probs, key=lambda p: p.get("probability", 0))
                result["alerce_stamp_class"] = best_stamp.get("class_name", "")
                result["alerce_stamp_prob"] = best_stamp.get("probability", np.nan)

    return result


def query_alerce(
    df: pd.DataFrame,
    radius_arcsec: float = ALERCE_RADIUS_ARCSEC,
    workers: int = 8,
    cache_dir: Path | str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Query ALeRCE ZTF broker for classification.

    Adds columns: alerce_oid, alerce_ndet, alerce_lc_class, alerce_lc_prob,
                  alerce_stamp_class, alerce_stamp_prob.
    """
    df = df.copy()
    df["alerce_oid"] = ""
    df["alerce_ndet"] = 0
    df["alerce_lc_class"] = ""
    df["alerce_lc_prob"] = np.nan
    df["alerce_stamp_class"] = ""
    df["alerce_stamp_prob"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    cache_keys = pd.Series(index=df.index, dtype=object)
    for idx in df.index[valid]:
        cache_keys.loc[idx] = _coord_cache_key(df.loc[idx, "ra"], df.loc[idx, "dec"], radius_arcsec)
    valid = valid & cache_keys.notna()
    if not valid.any():
        return df

    alerce_cols = (
        "alerce_oid",
        "alerce_ndet",
        "alerce_lc_class",
        "alerce_lc_prob",
        "alerce_stamp_class",
        "alerce_stamp_prob",
    )
    all_keys = set(cache_keys.loc[valid].astype(str))
    cached_by_key: dict[str, pd.Series] = {}
    if not refresh_cache:
        cached_by_key = _cached_rows_by_key(_read_vetting_cache(cache_dir, "alerce"), "coord_key", all_keys)

    cached_idx = []
    for idx in df.index[valid]:
        key = str(cache_keys.loc[idx])
        cached = cached_by_key.get(key)
        if cached is None:
            continue
        for col in alerce_cols:
            if col in cached.index:
                df.loc[idx, col] = cached[col]
        cached_idx.append(idx)

    missing = valid & ~df.index.isin(cached_idx)
    if not missing.any():
        matched = int((df.loc[valid, "alerce_oid"].fillna("").astype(str).str.strip() != "").sum())
        print(f"ALeRCE: served {len(cached_idx)} candidates from cache; {matched}/{int(valid.sum())} matched")
        return df

    n_valid = int(missing.sum())
    if cached_idx and not refresh_cache:
        print(f"ALeRCE: served {len(cached_idx)} candidates from cache; querying {n_valid} misses (radius={radius_arcsec}\", workers={workers})")
    else:
        print(f"ALeRCE: querying {n_valid} candidates (radius={radius_arcsec}\", workers={workers})")
    matched = int((df.loc[cached_idx, "alerce_oid"].fillna("").astype(str).str.strip() != "").sum()) if cached_idx else 0
    cache_payload: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_alerce_query_single, float(df.loc[idx, "ra"]),
                            float(df.loc[idx, "dec"]), radius_arcsec): idx
            for idx in df.index[missing]
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ALeRCE"):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception:
                result = None
            if result is None:
                result = {
                    "alerce_oid": "", "alerce_ndet": 0,
                    "alerce_lc_class": "", "alerce_lc_prob": np.nan,
                    "alerce_stamp_class": "", "alerce_stamp_prob": np.nan,
                }
                cache_payload[idx] = result
                continue
            for k, v in result.items():
                df.loc[idx, k] = v
            if _safe_text(result.get("alerce_oid")):
                matched += 1
            cache_payload[idx] = result

    cache_rows = []
    for idx in df.index[missing]:
        result = cache_payload.get(idx, {
            "alerce_oid": "", "alerce_ndet": 0,
            "alerce_lc_class": "", "alerce_lc_prob": np.nan,
            "alerce_stamp_class": "", "alerce_stamp_prob": np.nan,
        })
        row = {"coord_key": str(cache_keys.loc[idx]), "radius_arcsec": float(radius_arcsec)}
        row.update(result)
        cache_rows.append(row)
    _write_vetting_cache(cache_dir, "alerce", pd.DataFrame(cache_rows), key_cols=["coord_key"])

    print(f"ALeRCE: {matched}/{int(valid.sum())} candidates matched")
    return df


# =============================================================================
# ATLAS FORCED PHOTOMETRY
# =============================================================================


def _atlas_submit_job(
    ra: float, dec: float, token: str, mjd_min: float = ATLAS_MJD_MIN,
) -> str | None:
    """Submit an ATLAS forced photometry job. Returns task URL or None."""
    try:
        resp = requests.post(
            f"{ATLAS_API_BASE}/queue/",
            headers={"Authorization": f"Token {token}"},
            data={"ra": ra, "dec": dec, "mjd_min": mjd_min},
            timeout=30,
        )
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        return resp.url
    except Exception:
        return None


def _atlas_poll_result(task_url: str, token: str) -> pd.DataFrame | None:
    """Poll an ATLAS task until complete, return photometry DataFrame or None."""
    for _ in range(ATLAS_MAX_POLL):
        try:
            resp = requests.get(
                task_url,
                headers={"Authorization": f"Token {token}"},
                timeout=VETTING_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("finishtimestamp"):
                result_url = data.get("result_url")
                if result_url:
                    phot_resp = requests.get(
                        result_url,
                        headers={"Authorization": f"Token {token}"},
                        timeout=VETTING_HTTP_TIMEOUT,
                    )
                    phot_resp.raise_for_status()
                    text = phot_resp.text
                    # Strip comment lines
                    lines = [l for l in text.split("\n") if not l.startswith("###")]
                    if lines:
                        return pd.read_csv(io.StringIO("\n".join(lines)), delim_whitespace=True)
                return None
        except Exception:
            pass
        time.sleep(ATLAS_POLL_INTERVAL)
    return None


def query_atlas_forced_phot(
    df: pd.DataFrame,
    token: str | None = None,
    output_dir: Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Query ATLAS forced photometry for independent variability confirmation.

    Requires an ATLAS API token (register at https://fallingstar-data.com/forcedphot/).

    Adds columns: atlas_has_phot, atlas_n_det_cyan, atlas_n_det_orange,
                  atlas_cyan_range, atlas_orange_range.
    If *output_dir* is set, saves the full photometry DataFrame per candidate
    as ``atlas_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["atlas_has_phot"] = False
    df["atlas_n_det_cyan"] = 0
    df["atlas_n_det_orange"] = 0
    df["atlas_cyan_range"] = np.nan
    df["atlas_orange_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    valid_idx = df.index[valid].tolist()
    summary_cols = ["atlas_has_phot", "atlas_n_det_cyan", "atlas_n_det_orange", "atlas_cyan_range", "atlas_orange_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="ATLAS LCs",
        file_prefix="atlas_lc",
        summary_cols=summary_cols,
        match_col="atlas_has_phot",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, 0.0, "atlas", ATLAS_MJD_MIN),
        summarize_func=_summarize_atlas_lc,
    )
    if not valid_idx:
        print(f"ATLAS: {cached_matched}/{int(valid.sum())} candidates with photometry")
        return df

    token = token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN")
    if not token:
        print("ATLAS: no API token provided, skipping uncached candidates (register at https://fallingstar-data.com/forcedphot/)")
        return df

    n_valid = int(valid.sum())
    print(f"ATLAS: submitting {len(valid_idx)} forced photometry jobs")
    matched = cached_matched
    status_rows: list[dict] = []

    for idx in tqdm(valid_idx, desc="ATLAS forced phot"):
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        cache_key = _coord_lookup_cache_key(df, idx, 0.0, "atlas", ATLAS_MJD_MIN)

        task_url = _atlas_submit_job(ra, dec, token)
        if task_url is None:
            continue

        phot = _atlas_poll_result(task_url, token)
        if phot is None or phot.empty:
            row = _external_lc_status_row(
                df,
                idx,
                module="ATLAS LCs",
                cache_key=cache_key,
                summary={
                    "atlas_has_phot": False,
                    "atlas_n_det_cyan": 0,
                    "atlas_n_det_orange": 0,
                    "atlas_cyan_range": np.nan,
                    "atlas_orange_range": np.nan,
                },
                status="no_data",
            )
            if row is not None:
                status_rows.append(row)
            continue

        summary = _summarize_atlas_lc(phot)
        for col, value in summary.items():
            df.loc[idx, col] = value

        if output_dir and not phot.empty:
            _write_external_lc_file(output_dir, "atlas_lc", df, idx, phot)

        row = _external_lc_status_row(
            df,
            idx,
            module="ATLAS LCs",
            cache_key=cache_key,
            summary=summary,
            status="fetched",
        )
        if row is not None:
            status_rows.append(row)

        matched += 1

    _write_external_lc_status(output_dir, status_rows)
    print(f"ATLAS: {matched}/{n_valid} candidates with photometry")
    return df


# =============================================================================
# ZTF LIGHT CURVE FETCHING (IRSA API)
# =============================================================================

ZTF_LC_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
ZTF_LC_COLLECTION = "ztf_dr22"


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate labels by taking the first non-null value per row."""
    if df.empty or not df.columns.duplicated().any():
        return df
    out = pd.DataFrame(index=df.index)
    for col in pd.Index(df.columns).unique():
        subset = df.loc[:, df.columns == col]
        if subset.shape[1] == 1:
            out[col] = subset.iloc[:, 0]
        else:
            out[col] = subset.bfill(axis=1).iloc[:, 0]
    return out


def _normalize_ztf_api_lc(lc: pd.DataFrame) -> pd.DataFrame:
    """Normalize IRSA ZTF light-curve API output to MALCA's LC schema."""
    if lc.empty:
        return lc
    col_map = {}
    for col in lc.columns:
        cl = str(col).strip().lower()
        if cl in {"hjd", "hmjd", "mjd"}:
            col_map[col] = "mjd"
        elif cl in {"filtercode", "filterid", "fid", "filter", "band", "bandname"}:
            col_map[col] = "band"
        else:
            col_map[col] = cl.replace(" ", "_")
    lc = lc.rename(columns=col_map)
    lc = _coalesce_duplicate_columns(lc)

    if "catflags" in lc.columns:
        catflags = pd.to_numeric(lc["catflags"], errors="coerce")
        lc = lc.loc[catflags.fillna(0) == 0].copy()

    if "mjd" in lc.columns:
        lc["mjd"] = pd.to_numeric(lc["mjd"], errors="coerce")
        mask = lc["mjd"] > 2400000
        lc.loc[mask, "mjd"] = lc.loc[mask, "mjd"] - 2400000.5

    if "band" in lc.columns:
        def _band_name(value: object) -> str:
            text = str(value).strip().lower()
            if text.endswith(".0"):
                text = text[:-2]
            return {
                "1": "zg",
                "2": "zr",
                "3": "zi",
                "g": "zg",
                "r": "zr",
                "i": "zi",
                "zg": "zg",
                "zr": "zr",
                "zi": "zi",
            }.get(text, str(value))

        lc["band"] = lc["band"].map(_band_name)
    return lc


def fetch_ztf_lightcurves(
    df: pd.DataFrame,
    radius_arcsec: float = 2.0,
    output_dir: Path | None = None,
    workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch ZTF light curves from IRSA ZTF DR22.

    Uses IRSA's ZTF light-curve API. The API performs the object lookup and
    light-curve retrieval for the requested position/collection.

    Adds columns: ztf_lc_n_det, ztf_lc_g_range, ztf_lc_r_range.
    If *output_dir* is set, saves per-candidate parquet files as
    ``ztf_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["ztf_lc_n_det"] = 0
    df["ztf_lc_g_range"] = np.nan
    df["ztf_lc_r_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"ZTF LCs: fetching {n_valid} light curves")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["ztf_lc_n_det", "ztf_lc_g_range", "ztf_lc_r_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="ZTF LCs",
        file_prefix="ztf_lc",
        summary_cols=summary_cols,
        match_col="ztf_lc_n_det",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, radius_arcsec, "ztf_dr22"),
        summarize_func=_summarize_ztf_lc,
    )
    if not valid_idx:
        print(f"ZTF LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, np.nan, np.nan, None, None)

        cache_key = _coord_lookup_cache_key(df, idx, radius_arcsec, "ztf_dr22")

        try:
            response = requests.get(
                ZTF_LC_API_URL,
                params={
                    "POS": f"CIRCLE {ra:.7f} {dec:.7f} {radius_arcsec / 3600.0:.8f}",
                    "BAD_CATFLAGS_MASK": "32768",
                    "COLLECTION": ZTF_LC_COLLECTION,
                    "FORMAT": "CSV",
                },
                timeout=60,
            )
            response.raise_for_status()
            text = str(response.text or "").strip()
            if not text:
                return (idx, 0, np.nan, np.nan, None, cache_key)
            if "<html" in text[:200].lower():
                raise RuntimeError("IRSA ZTF light-curve API returned HTML instead of CSV")
            try:
                lc = pd.read_csv(io.StringIO(text), comment="#")
            except pd.errors.EmptyDataError:
                return (idx, 0, np.nan, np.nan, None, cache_key)
            lc = _normalize_ztf_api_lc(lc)
            if lc.empty:
                return (idx, 0, np.nan, np.nan, None, cache_key)

            summary = _summarize_ztf_lc(lc)

            if output_dir and not lc.empty:
                _write_external_lc_file(output_dir, "ztf_lc", df, idx, lc)

            return (
                idx,
                summary["ztf_lc_n_det"],
                summary["ztf_lc_g_range"],
                summary["ztf_lc_r_range"],
                None,
                cache_key,
            )
        except Exception as exc:
            return (idx, 0, np.nan, np.nan, f"{idx}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ZTF LCs"):
            idx, n_det, g_range, r_range, error, cache_key = fut.result()
            if error is not None:
                failures.append(error)
                summary = {"ztf_lc_n_det": 0, "ztf_lc_g_range": np.nan, "ztf_lc_r_range": np.nan}
                row = _external_lc_status_row(
                    df,
                    idx,
                    module="ZTF LCs",
                    cache_key=cache_key,
                    summary=summary,
                    status="failed",
                )
                if row is not None:
                    row["error"] = error
                    status_rows.append(row)
                continue
            df.loc[idx, "ztf_lc_n_det"] = n_det
            df.loc[idx, "ztf_lc_g_range"] = g_range
            df.loc[idx, "ztf_lc_r_range"] = r_range
            summary = {"ztf_lc_n_det": n_det, "ztf_lc_g_range": g_range, "ztf_lc_r_range": r_range}
            row = _external_lc_status_row(
                df,
                idx,
                module="ZTF LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if n_det > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)
            if n_det > 0:
                matched += 1

    _write_external_lc_status(output_dir, status_rows)
    if failures:
        detail = "; ".join(failures[:3])
        more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        print(f"ZTF LCs: lookup failed for {len(failures)}/{n_valid} candidates; keeping partial results: {detail}{more}")
    print(f"ZTF LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# GAIA DR3 EPOCH PHOTOMETRY
# =============================================================================


def query_gaia_epoch_photometry(
    df: pd.DataFrame,
    chunk_size: int = GAIA_VAR_CHUNK_SIZE,
    cache_dir: Path | str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Check Gaia DR3 epoch photometry availability and basic stats.

    Adds columns: gaia_epoch_available, gaia_epoch_n_obs, gaia_epoch_g_range.
    Requires 'gaia_id' column.
    """
    df = df.copy()
    df["gaia_epoch_available"] = False
    df["gaia_epoch_n_obs"] = 0
    df["gaia_epoch_g_range"] = np.nan

    if "gaia_id" not in df.columns:
        print("Warning: Gaia epoch photometry requires 'gaia_id' column, skipping")
        return df

    valid = df["gaia_id"].notna()
    if not valid.any():
        return df

    # Normalize IDs
    gaia_ids = []
    idx_map = {}
    for idx, val in df.loc[valid, "gaia_id"].items():
        sid = _parse_gaia_source_id_str(val)
        if sid is None:
            continue
        gaia_ids.append(sid)
        idx_map.setdefault(sid, []).append(idx)
    gaia_ids = sorted(set(gaia_ids))

    if not gaia_ids:
        return df

    all_ids = set(gaia_ids)
    cached_by_id: dict[str, pd.Series] = {}
    if not refresh_cache:
        cached_by_id = _cached_rows_by_key(_read_vetting_cache(cache_dir, "gaia_epoch"), "source_id", all_ids)

    matched = 0
    for sid, cached in cached_by_id.items():
        n_obs = pd.to_numeric(cached.get("gaia_epoch_n_obs"), errors="coerce")
        g_range = pd.to_numeric(cached.get("gaia_epoch_g_range"), errors="coerce")
        n_obs_int = int(n_obs) if pd.notna(n_obs) else 0
        for idx in idx_map.get(sid, []):
            df.loc[idx, "gaia_epoch_available"] = bool(n_obs_int > 0)
            df.loc[idx, "gaia_epoch_n_obs"] = n_obs_int
            df.loc[idx, "gaia_epoch_g_range"] = float(g_range) if pd.notna(g_range) else np.nan
            if n_obs_int > 0:
                matched += 1

    missing_ids = [sid for sid in gaia_ids if refresh_cache or sid not in cached_by_id]
    if not missing_ids:
        print(f"Gaia epoch photometry: served {len(gaia_ids)} source_ids from cache")
        print(f"Gaia epoch photometry: {matched} sources with time-series data")
        return df

    if cached_by_id and not refresh_cache:
        print(f"Gaia epoch photometry: served {len(cached_by_id)} source_ids from cache; checking {len(missing_ids)} misses")
    else:
        print(f"Gaia epoch photometry: checking {len(missing_ids)} source_ids")
    test_query = f"SELECT source_id FROM gaiadr3.vari_summary WHERE source_id = {missing_ids[0]}"
    taps = _connect_gaia_taps_until_available(test_query, label="Gaia epoch photometry")

    # Query vari_summary for observation counts and magnitude ranges
    # (epoch photometry itself is huge — we use vari_summary stats instead)
    epoch_results = {}
    for i in tqdm(range(0, len(missing_ids), chunk_size), desc="Gaia epoch stats"):
        chunk = missing_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id,
                   num_selected_g_fov,
                   range_mag_g_fov
            FROM gaiadr3.vari_summary
            WHERE source_id IN ({ids_str})
        """
        result = _run_gaia_tap_query_until_success(taps, query, label=f"Gaia epoch stats chunk {i}")
        for row in result:
            sid = str(row["source_id"])
            n_obs = int(row["num_selected_g_fov"]) if row["num_selected_g_fov"] is not None else 0
            g_range = float(row["range_mag_g_fov"]) if row["range_mag_g_fov"] is not None else np.nan
            epoch_results[sid] = (n_obs, g_range)

    # Apply
    cache_rows = []
    for sid in missing_ids:
        indices = idx_map.get(sid, [])
        info = epoch_results.get(sid)
        if info is None:
            cache_rows.append({
                "source_id": sid,
                "gaia_epoch_available": False,
                "gaia_epoch_n_obs": 0,
                "gaia_epoch_g_range": np.nan,
            })
            continue
        n_obs, g_range = info
        for idx in indices:
            df.loc[idx, "gaia_epoch_available"] = n_obs > 0
            df.loc[idx, "gaia_epoch_n_obs"] = n_obs
            df.loc[idx, "gaia_epoch_g_range"] = g_range
            matched += 1
        cache_rows.append({
            "source_id": sid,
            "gaia_epoch_available": bool(n_obs > 0),
            "gaia_epoch_n_obs": int(n_obs),
            "gaia_epoch_g_range": g_range,
        })

    _write_vetting_cache(cache_dir, "gaia_epoch", pd.DataFrame(cache_rows), key_cols=["source_id"])

    print(f"Gaia epoch photometry: {matched} sources with time-series data")
    return df


def fetch_gaia_epoch_lcs(
    df: pd.DataFrame,
    chunk_size: int = 50,
    output_dir: Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Download full Gaia DR3 epoch photometry time series.

    Only fetches for candidates with ``gaia_epoch_available == True`` (or that
    have a valid ``gaia_id``).  Stores per-candidate parquet files as
    ``gaia_epoch_lc_<candidate_id>.parquet``.

    Adds columns: gaia_epoch_lc_n_g, gaia_epoch_lc_g_range.
    """
    df = df.copy()
    df["gaia_epoch_lc_n_g"] = 0
    df["gaia_epoch_lc_g_range"] = np.nan

    if "gaia_id" not in df.columns:
        print("Gaia epoch LCs: requires 'gaia_id' column, skipping")
        return df

    # Only fetch for candidates with epoch photometry available
    if "gaia_epoch_available" in df.columns:
        valid = df["gaia_id"].notna() & df["gaia_epoch_available"].astype(bool)
    else:
        valid = df["gaia_id"].notna()
    if not valid.any():
        print("Gaia epoch LCs: no candidates with epoch photometry available")
        return df
    n_valid_total = int(valid.sum())

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    valid_idx = df.index[valid].tolist()
    summary_cols = ["gaia_epoch_lc_n_g", "gaia_epoch_lc_g_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="Gaia epoch LCs",
        file_prefix="gaia_epoch_lc",
        summary_cols=summary_cols,
        match_col="gaia_epoch_lc_n_g",
        cache_key_func=lambda idx: _source_lookup_cache_key(_parse_gaia_source_id_str(df.loc[idx, "gaia_id"]), "gaia_epoch_lc"),
        summarize_func=_summarize_gaia_epoch_lc,
    )
    if not valid_idx:
        print(f"Gaia epoch LCs: {cached_matched}/{n_valid_total} with time-series data")
        return df

    # Build gaia_id -> index mapping
    gaia_ids = []
    idx_map: dict[str, list] = {}
    for idx in valid_idx:
        val = df.loc[idx, "gaia_id"]
        sid = _parse_gaia_source_id_str(val)
        if sid is None:
            continue
        gaia_ids.append(sid)
        idx_map.setdefault(sid, []).append(idx)
    gaia_ids = list(set(gaia_ids))

    if not gaia_ids:
        return df

    n_total = len(gaia_ids)
    print(f"Gaia epoch LCs: downloading time series for {n_total} sources")

    # Find working TAP server
    test = f"SELECT source_id FROM gaiadr3.epoch_photometry WHERE source_id = {gaia_ids[0]} AND transit_id IS NOT NULL"
    taps = _connect_gaia_taps_until_available(test, label="Gaia epoch LC download", maxrec=1)

    matched = cached_matched
    seen_with_data: set[str] = set()
    status_rows: list[dict] = []
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia epoch LCs"):
        chunk = gaia_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id, transit_id,
                   g_transit_time AS "time",
                   g_transit_mag AS mag,
                   'G' AS band
            FROM gaiadr3.epoch_photometry
            WHERE source_id IN ({ids_str})
              AND g_transit_time IS NOT NULL
              AND g_transit_mag IS NOT NULL
            ORDER BY source_id, g_transit_time
        """
        result = _run_gaia_tap_query_until_success(taps, query, label=f"Gaia epoch LC chunk {i}")
        table = result.to_table()
        if table is None or len(table) == 0:
            continue
        lc_all = _normalize_gaia_epoch_tap_lightcurve(_gaia_epoch_table_to_frame(table))
        if lc_all.empty:
            continue

        # Process per source
        for sid in chunk:
            sid_int = int(sid)
            src_lc = lc_all[lc_all["source_id"] == sid_int].copy()
            if src_lc.empty:
                continue

            src_lc["time"] = pd.to_numeric(src_lc["time"], errors="coerce")
            src_lc["mag"] = pd.to_numeric(src_lc["mag"], errors="coerce")
            if "mag_error" in src_lc.columns:
                src_lc["mag_error"] = pd.to_numeric(src_lc["mag_error"], errors="coerce")
            else:
                src_lc["mag_error"] = np.nan
            src_lc = src_lc.dropna(subset=["time", "mag"])

            if src_lc.empty:
                continue

            summary = _summarize_gaia_epoch_lc(src_lc)
            n_g = int(summary["gaia_epoch_lc_n_g"])
            g_range = summary["gaia_epoch_lc_g_range"]

            for df_idx in idx_map.get(sid, []):
                df.loc[df_idx, "gaia_epoch_lc_n_g"] = n_g
                df.loc[df_idx, "gaia_epoch_lc_g_range"] = g_range
                row = _external_lc_status_row(
                    df,
                    df_idx,
                    module="Gaia epoch LCs",
                    cache_key=_source_lookup_cache_key(sid, "gaia_epoch_lc"),
                    summary=summary,
                    status="fetched",
                )
                if row is not None:
                    status_rows.append(row)

            if output_dir and not src_lc.empty:
                for df_idx in idx_map.get(sid, []):
                    _write_external_lc_file(output_dir, "gaia_epoch_lc", df, df_idx, src_lc)

            seen_with_data.add(sid)
            matched += len(idx_map.get(sid, []))

    for sid in gaia_ids:
        if sid in seen_with_data:
            continue
        summary = {"gaia_epoch_lc_n_g": 0, "gaia_epoch_lc_g_range": np.nan}
        for df_idx in idx_map.get(sid, []):
            row = _external_lc_status_row(
                df,
                df_idx,
                module="Gaia epoch LCs",
                cache_key=_source_lookup_cache_key(sid, "gaia_epoch_lc"),
                summary=summary,
                status="no_data",
            )
            if row is not None:
                status_rows.append(row)

    _write_external_lc_status(output_dir, status_rows)
    print(f"Gaia epoch LCs: {matched}/{n_valid_total} with time-series data")
    return df


# =============================================================================
# eROSITA X-RAY CATALOG
# =============================================================================


def _to_bool_flag(value: object) -> bool:
    """Coerce catalog flag values without treating arbitrary non-empty strings as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "detected"}


def _safe_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _init_xray_aggregate_columns(df: pd.DataFrame) -> None:
    if "xray_det" not in df.columns:
        df["xray_det"] = False
    if "xray_flux" not in df.columns:
        df["xray_flux"] = np.nan
    if "xray_sep_arcsec" not in df.columns:
        df["xray_sep_arcsec"] = np.nan
    if "xray_source_catalogs" not in df.columns:
        df["xray_source_catalogs"] = ""


def _sync_xray_aggregate_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Set generic xray_* fields from source-specific X-ray catalog matches."""
    df = df.copy()
    legacy_xray_available = "erosita_det" not in df.columns and "xray_det" in df.columns
    _init_xray_aggregate_columns(df)

    for idx, row in df.iterrows():
        matched_catalogs: list[str] = []
        candidates: list[tuple[float, int, str, float, float]] = []

        if "erosita_det" in df.columns:
            erosita_det = _to_bool_flag(row.get("erosita_det"))
            erosita_flux = _safe_float(row.get("erosita_flux"))
            erosita_sep = _safe_float(row.get("erosita_sep_arcsec"))
        elif legacy_xray_available:
            catalogs_text = str(row.get("xray_source_catalogs") or "")
            legacy_is_chandra = "chandra" in catalogs_text.lower()
            erosita_det = _to_bool_flag(row.get("xray_det")) and not legacy_is_chandra
            erosita_flux = _safe_float(row.get("xray_flux"))
            erosita_sep = _safe_float(row.get("xray_sep_arcsec"))
        else:
            erosita_det = False
            erosita_flux = np.nan
            erosita_sep = np.nan

        if erosita_det:
            matched_catalogs.append(EROSITA_XRAY_LABEL)
            sort_sep = erosita_sep if np.isfinite(erosita_sep) else np.inf
            candidates.append((sort_sep, 0, EROSITA_XRAY_LABEL, erosita_flux, erosita_sep))

        chandra_det = _to_bool_flag(row.get("chandra_det")) if "chandra_det" in df.columns else False
        if chandra_det:
            chandra_flux = _safe_float(row.get("chandra_flux_05_7"))
            if not np.isfinite(chandra_flux):
                chandra_flux = _safe_float(row.get("chandra_flux_broad"))
            chandra_sep = _safe_float(row.get("chandra_sep_arcsec"))
            matched_catalogs.append(CHANDRA_XRAY_LABEL)
            sort_sep = chandra_sep if np.isfinite(chandra_sep) else np.inf
            candidates.append((sort_sep, 1, CHANDRA_XRAY_LABEL, chandra_flux, chandra_sep))

        if candidates:
            _sort_sep, _order, _label, flux, sep = sorted(candidates)[0]
            df.loc[idx, "xray_det"] = True
            df.loc[idx, "xray_flux"] = flux if np.isfinite(flux) else np.nan
            df.loc[idx, "xray_sep_arcsec"] = round(float(sep), 3) if np.isfinite(sep) else np.nan
            df.loc[idx, "xray_source_catalogs"] = ",".join(matched_catalogs)
        else:
            df.loc[idx, "xray_det"] = False
            df.loc[idx, "xray_flux"] = np.nan
            df.loc[idx, "xray_sep_arcsec"] = np.nan
            df.loc[idx, "xray_source_catalogs"] = ""

    return df


def _init_erosita_columns(df: pd.DataFrame) -> None:
    df["erosita_det"] = False
    df["erosita_flux"] = np.nan
    df["erosita_sep_arcsec"] = np.nan
    _init_xray_aggregate_columns(df)


def _init_chandra_columns(df: pd.DataFrame) -> None:
    df["chandra_det"] = False
    df["chandra_source_id"] = ""
    df["chandra_flux_05_7"] = np.nan
    df["chandra_flux_broad"] = np.nan
    df["chandra_significance"] = np.nan
    df["chandra_likelihood"] = np.nan
    df["chandra_likelihood_class"] = ""
    df["chandra_pos_err_maj_arcsec"] = np.nan
    df["chandra_pos_err_min_arcsec"] = np.nan
    df["chandra_pos_err_pa_deg"] = np.nan
    df["chandra_extended_flag"] = False
    df["chandra_variable_flag"] = False
    df["chandra_sep_arcsec"] = np.nan
    _init_xray_aggregate_columns(df)


def crossmatch_erosita(
    df: pd.DataFrame,
    radius_arcsec: float = EROSITA_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "xmatch", "local"] = "tap",
    local_fits: Path | str | None = None,
) -> pd.DataFrame:
    """
    Crossmatch against eROSITA-DE DR1 (Merloni+2024).

    method='tap'    — batch VizieR TAP upload (best for large batches).
    method='xmatch' — CDS XMatch service (reliable for small batches).
    method='local'  — local FITS crossmatch via SkyCoord (instant, no network).

    X-ray detection is a strong youth indicator for YSO candidates.
    Adds source-specific erosita_* columns and updates aggregate xray_* columns.
    """
    df = df.copy()
    _init_erosita_columns(df)

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return _sync_xray_aggregate_fields(df)

    n_valid = int(valid.sum())

    if method == "local":
        return _erosita_via_local(df, valid, n_valid, radius_arcsec, local_fits)
    elif method == "xmatch":
        print(f"eROSITA: crossmatching {n_valid} candidates via CDS XMatch (radius={radius_arcsec}\")")
        source_table = Table()
        source_table["_idx"] = np.array(df.index[valid])
        source_table["ra"] = df.loc[valid, "ra"].values
        source_table["dec"] = df.loc[valid, "dec"].values

        try:
            result_tab = XMatch.query(
                cat1=source_table,
                cat2="vizier:J/A+A/682/A34/erass1-m",
                max_distance=radius_arcsec * u.arcsec,
                colRA1="ra", colDec1="dec",
            )
            result = result_tab.to_pandas() if result_tab is not None and len(result_tab) > 0 else pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"eROSITA XMatch lookup failed: {e}") from e

        sep_col = "angDist" if "angDist" in result.columns else "sep_arcsec"
        if not result.empty and sep_col in result.columns:
            result = result.sort_values(sep_col).drop_duplicates(subset="_idx", keep="first")
    else:
        print(f"eROSITA: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")
        coords_df = pd.DataFrame({
            "_idx": df.index[valid],
            "ra": df.loc[valid, "ra"].values,
            "dec": df.loc[valid, "dec"].values,
        })
        result = batch_tap_crossmatch(
            coords_df,
            tap_url=VIZIER_TAP_URL,
            catalog_table='"J/A+A/682/A34/erass1-m"',
            select_cols='c."MLFlux1"',
            ra_col="RA_ICRS",
            dec_col="DE_ICRS",
            match_radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            n_workers=4,
            verbose=True,
            desc="eROSITA TAP",
            raise_on_all_failed=True,
            raise_on_failed_chunk=True,
        )
        sep_col = "sep_arcsec"
        if not result.empty:
            result = result.sort_values(sep_col).drop_duplicates(subset="_idx", keep="first")

    matched = 0
    if not result.empty:
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "erosita_det"] = True
                sep = row.get("angDist", row.get("sep_arcsec", np.nan))
                df.loc[idx, "erosita_sep_arcsec"] = round(float(sep), 3) if pd.notna(sep) else np.nan
                try:
                    df.loc[idx, "erosita_flux"] = float(row["MLFlux1"])
                except (ValueError, TypeError, KeyError):
                    pass
                matched += 1

    print(f"eROSITA: {matched} X-ray matches")
    return _sync_xray_aggregate_fields(df)


def _erosita_via_local(
    df: pd.DataFrame, valid, n_valid: int, radius_arcsec: float,
    local_fits: Path | str | None = None,
) -> pd.DataFrame:
    """eROSITA crossmatch via local FITS + SkyCoord (instant)."""


    fits_path = Path(local_fits) if local_fits else EROSITA_LOCAL_FITS
    if not fits_path.exists():
        raise FileNotFoundError(f"eROSITA local FITS not found: {fits_path}")

    # Load and cache the catalog (only RA, DEC, ML_FLUX_1)
    cache_key = str(fits_path)
    if cache_key not in _erosita_cache:
        print(f"eROSITA: loading local catalog from {fits_path.name}...")
        with pyfits.open(fits_path, memmap=True) as hdul:
            tbl = hdul[1].data
            ra_arr = tbl["RA"].astype(np.float64)
            dec_arr = tbl["DEC"].astype(np.float64)
            flux_arr = tbl["ML_FLUX_1"].astype(np.float32)
        cat_coord = SkyCoord(ra=ra_arr, dec=dec_arr, unit="deg")
        _erosita_cache[cache_key] = (flux_arr, cat_coord)
        print(f"eROSITA: cached {len(ra_arr)} sources")

    flux_arr, cat_coord = _erosita_cache[cache_key]

    print(f"eROSITA: crossmatching {n_valid} candidates via local catalog (radius={radius_arcsec}\")")

    src_coord = SkyCoord(
        ra=df.loc[valid, "ra"].values, dec=df.loc[valid, "dec"].values, unit="deg",
    )
    idx_cat, sep2d, _ = src_coord.match_to_catalog_sky(cat_coord)
    max_sep = radius_arcsec * u.arcsec

    matched = 0
    for i, df_idx in enumerate(df.index[valid]):
        if sep2d[i] <= max_sep:
            df.loc[df_idx, "erosita_det"] = True
            df.loc[df_idx, "erosita_sep_arcsec"] = round(sep2d[i].arcsec, 3)
            try:
                df.loc[df_idx, "erosita_flux"] = float(flux_arr[idx_cat[i]])
            except (ValueError, TypeError):
                pass
            matched += 1

    print(f"eROSITA: {matched} X-ray matches")
    return _sync_xray_aggregate_fields(df)


# =============================================================================
# CHANDRA CSC X-RAY CATALOG
# =============================================================================


def crossmatch_chandra_csc(
    df: pd.DataFrame,
    radius_arcsec: float = CHANDRA_CSC_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "xmatch"] = "tap",
) -> pd.DataFrame:
    """
    Crossmatch against Chandra Source Catalog 2.1 master sources in VizieR.

    Chandra is an archival pointed-observation catalog, so a non-match means
    no nearby CSC source entry, not a uniform all-sky X-ray non-detection.
    Adds source-specific chandra_* columns and updates aggregate xray_* columns.
    """
    df = df.copy()
    _init_chandra_columns(df)

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return _sync_xray_aggregate_fields(df)

    n_valid = int(valid.sum())
    coords_df = pd.DataFrame({
        "_idx": df.index[valid],
        "ra": df.loc[valid, "ra"].values,
        "dec": df.loc[valid, "dec"].values,
    })
    if method == "xmatch":
        print(f"Chandra CSC: crossmatching {n_valid} candidates via CDS XMatch (radius={radius_arcsec}\")")
        source_table = Table()
        source_table["_idx"] = np.array(coords_df["_idx"])
        source_table["ra"] = coords_df["ra"].values
        source_table["dec"] = coords_df["dec"].values

        try:
            result_tab = XMatch.query(
                cat1=source_table,
                cat2=f"vizier:{CHANDRA_CSC_CATALOG}",
                max_distance=radius_arcsec * u.arcsec,
                colRA1="ra", colDec1="dec",
            )
            result = result_tab.to_pandas() if result_tab is not None and len(result_tab) > 0 else pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"Chandra CSC XMatch lookup failed: {e}") from e

        if not result.empty:
            result = result.rename(columns={
                "2CXO": "chandra_source_id",
                "FPL0.5-7": "chandra_flux_05_7",
                "Favgb": "chandra_flux_broad",
                "signi": "chandra_significance",
                "like": "chandra_likelihood",
                "likeClass": "chandra_likelihood_class",
                "r0": "chandra_pos_err_maj_arcsec",
                "r1": "chandra_pos_err_min_arcsec",
                "PA": "chandra_pos_err_pa_deg",
                "fe": "chandra_extended_flag",
                "fv": "chandra_variable_flag",
            })
        sep_col = "angDist"
    else:
        print(f"Chandra CSC: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")
        result = batch_tap_crossmatch(
            coords_df,
            tap_url=VIZIER_TAP_URL,
            catalog_table=f'"{CHANDRA_CSC_CATALOG}"',
            select_cols=(
                'c."2CXO" AS chandra_source_id, '
                'c."FPL0.5-7" AS chandra_flux_05_7, '
                'c."Favgb" AS chandra_flux_broad, '
                'c."signi" AS chandra_significance, '
                'c."like" AS chandra_likelihood, '
                'c."likeClass" AS chandra_likelihood_class, '
                'c."r0" AS chandra_pos_err_maj_arcsec, '
                'c."r1" AS chandra_pos_err_min_arcsec, '
                'c."PA" AS chandra_pos_err_pa_deg, '
                'c."fe" AS chandra_extended_flag, '
                'c."fv" AS chandra_variable_flag'
            ),
            ra_col='"RAICRS"',
            dec_col='"DEICRS"',
            match_radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            n_workers=4,
            verbose=True,
            desc="Chandra CSC TAP",
            raise_on_all_failed=True,
            raise_on_failed_chunk=True,
        )
        sep_col = "sep_arcsec"
    if not result.empty:
        result = result.sort_values(sep_col).drop_duplicates(subset="_idx", keep="first")

    matched = 0
    if not result.empty:
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx not in df.index:
                continue

            df.loc[idx, "chandra_det"] = True
            source_id = row.get("chandra_source_id", "")
            if pd.notna(source_id):
                df.loc[idx, "chandra_source_id"] = str(source_id)
            for out_col in (
                "chandra_flux_05_7",
                "chandra_flux_broad",
                "chandra_significance",
                "chandra_likelihood",
                "chandra_pos_err_maj_arcsec",
                "chandra_pos_err_min_arcsec",
                "chandra_pos_err_pa_deg",
            ):
                value = _safe_float(row.get(out_col))
                if np.isfinite(value):
                    df.loc[idx, out_col] = value
            like_class = row.get("chandra_likelihood_class", "")
            if pd.notna(like_class):
                df.loc[idx, "chandra_likelihood_class"] = str(like_class)
            df.loc[idx, "chandra_extended_flag"] = _to_bool_flag(row.get("chandra_extended_flag"))
            df.loc[idx, "chandra_variable_flag"] = _to_bool_flag(row.get("chandra_variable_flag"))
            sep = _safe_float(row.get(sep_col, row.get("sep_arcsec")))
            if np.isfinite(sep):
                df.loc[idx, "chandra_sep_arcsec"] = round(float(sep), 3)
            matched += 1

    print(f"Chandra CSC: {matched} X-ray matches")
    return _sync_xray_aggregate_fields(df)


# =============================================================================
# PROPER MOTION CONSISTENCY
# =============================================================================


def check_pm_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check proper motion consistency with cluster membership.

    For candidates that have cluster_name and proper motions (pmra, pmdec),
    compute the offset from the cluster mean PM in sigma units.

    Adds column: pm_cluster_offset_sigma.
    """
    df = df.copy()
    df["pm_cluster_offset_sigma"] = np.nan

    required = {"cluster_name", "pmra", "pmdec", "pmra_error" if "pmra_error" in df.columns else "pmra"}
    has_cluster = "cluster_name" in df.columns
    has_pm = "pmra" in df.columns and "pmdec" in df.columns
    if not has_cluster or not has_pm:
        print("PM consistency: requires cluster_name, pmra, pmdec columns, skipping")
        return df

    # Find candidates with cluster membership
    in_cluster = df["cluster_name"].notna() & (df["cluster_name"] != "")
    if not in_cluster.any():
        print("PM consistency: no candidates with cluster membership")
        return df

    # Compute cluster mean PM from the candidates themselves (grouped by cluster)
    cluster_groups = df.loc[in_cluster].groupby("cluster_name")
    cluster_stats = {}
    for name, group in cluster_groups:
        pm_ra = group["pmra"].dropna()
        pm_dec = group["pmdec"].dropna()
        if len(pm_ra) >= 2 and len(pm_dec) >= 2:
            cluster_stats[name] = {
                "pmra_mean": pm_ra.mean(),
                "pmdec_mean": pm_dec.mean(),
                "pmra_std": max(pm_ra.std(), 0.5),  # floor at 0.5 mas/yr
                "pmdec_std": max(pm_dec.std(), 0.5),
            }

    if not cluster_stats:
        # If only single members per cluster, use PM errors if available
        pmra_err_col = "pmra_error" if "pmra_error" in df.columns else None
        pmdec_err_col = "pmdec_error" if "pmdec_error" in df.columns else None
        if pmra_err_col and pmdec_err_col:
            for idx in df.index[in_cluster]:
                # No cluster mean available — flag as nan
                pass
        print("PM consistency: insufficient cluster members for PM comparison")
        return df

    # Compute offset
    matched = 0
    for idx in df.index[in_cluster]:
        cluster = df.loc[idx, "cluster_name"]
        stats = cluster_stats.get(cluster)
        if stats is None:
            continue
        pmra = df.loc[idx, "pmra"]
        pmdec = df.loc[idx, "pmdec"]
        if pd.isna(pmra) or pd.isna(pmdec):
            continue

        d_ra = (pmra - stats["pmra_mean"]) / stats["pmra_std"]
        d_dec = (pmdec - stats["pmdec_mean"]) / stats["pmdec_std"]
        offset_sigma = np.sqrt(d_ra**2 + d_dec**2)
        df.loc[idx, "pm_cluster_offset_sigma"] = round(float(offset_sigma), 2)
        matched += 1

    print(f"PM consistency: computed for {matched} cluster members")
    return df


# =============================================================================
# NEOWISE LIGHT CURVES
# =============================================================================


def query_neowise_lightcurves(
    df: pd.DataFrame,
    max_sep_arcsec: float = NEOWISE_MAX_SEP_ARCSEC,
    output_dir: Path | None = None,
    workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch full NEOWISE light curves for candidates.

    Stores per-epoch W1/W2 photometry (if output_dir set, saves individual LC parquets).
    Adds columns: neowise_n_epochs, neowise_w1_range, neowise_w2_range.
    """


    df = df.copy()
    df["neowise_n_epochs"] = 0
    df["neowise_w1_range"] = np.nan
    df["neowise_w2_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"NEOWISE LCs: fetching {n_valid} light curves")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["neowise_n_epochs", "neowise_w1_range", "neowise_w2_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="NEOWISE LCs",
        file_prefix="neowise_lc",
        summary_cols=summary_cols,
        match_col="neowise_n_epochs",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, max_sep_arcsec, "neowise"),
        summarize_func=_summarize_neowise_lc,
    )
    if not valid_idx:
        print(f"NEOWISE LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, np.nan, np.nan, None, None)

        cache_key = _coord_lookup_cache_key(df, idx, max_sep_arcsec, "neowise")

        query = f"""
        SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, w1snr, w2snr,
               qual_frame, qi_fact, cc_flags
        FROM neowiser_p1bs_psd
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {max_sep_arcsec / 3600.0})
        ) = 1
        ORDER BY mjd ASC
        """
        try:
            result = Irsa.query_tap(query)
            table = result.to_table()
            if table is None or len(table) == 0:
                return (idx, 0, np.nan, np.nan, None, cache_key)

            lc = table.to_pandas()

            lc = filter_neowise_single_exposure_lc(lc)

            if lc.empty:
                return (idx, 0, np.nan, np.nan, None, cache_key)

            summary = _summarize_neowise_lc(lc)

            # Save individual LC if output_dir set
            if output_dir and not lc.empty:
                _write_external_lc_file(output_dir, "neowise_lc", df, idx, lc)

            return (
                idx,
                summary["neowise_n_epochs"],
                summary["neowise_w1_range"],
                summary["neowise_w2_range"],
                None,
                cache_key,
            )
        except Exception as exc:
            return (idx, 0, np.nan, np.nan, f"{idx}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="NEOWISE LCs"):
            idx, n_epochs, w1_range, w2_range, error, cache_key = fut.result()
            if error is not None:
                failures.append(error)
                continue
            df.loc[idx, "neowise_n_epochs"] = n_epochs
            df.loc[idx, "neowise_w1_range"] = w1_range
            df.loc[idx, "neowise_w2_range"] = w2_range
            summary = {"neowise_n_epochs": n_epochs, "neowise_w1_range": w1_range, "neowise_w2_range": w2_range}
            row = _external_lc_status_row(
                df,
                idx,
                module="NEOWISE LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if n_epochs > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)
            if n_epochs > 0:
                matched += 1

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("NEOWISE LCs", failures, n_valid)
    print(f"NEOWISE LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# TESS LIGHT CURVES
# =============================================================================


def _tess_error_is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "i/o operation on closed file",
        "error in reading data product",
        "failed to read",
        "cannot read",
        "corrupt",
        "fits",
    )
    return any(marker in message for marker in markers)


def _tess_cache_paths_from_error(exc: Exception) -> list[Path]:
    paths: list[Path] = []
    for match in re.finditer(r"(/[^\s\"'<>]+?\.fits(?:\.gz)?)", str(exc)):
        text = match.group(1).rstrip(").,;:")
        path = Path(text).expanduser()
        if path not in paths:
            paths.append(path)
    return paths


def _tess_cache_path_is_safe_to_purge(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        ".lightkurve" in parts
        and "mastdownload" in parts
        and (name.endswith(".fits") or name.endswith(".fits.gz"))
    )


def _purge_tess_bad_cache_files(paths: list[Path]) -> list[Path]:
    purged: list[Path] = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path.expanduser()
        if not _tess_cache_path_is_safe_to_purge(resolved):
            continue
        try:
            if resolved.is_file():
                resolved.unlink()
                purged.append(resolved)
        except Exception:
            continue
    return purged


def fetch_tess_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 2,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch TESS light curves via ``lightkurve``.

    Prefers 2-min cadence SPOC, falls back to QLP/FFI products. Review-mode
    overlays use this output directly, so the search cone should stay narrow.

    Adds columns: tess_n_sectors, tess_total_points, tess_flux_range.
    If *output_dir* is set, saves per-candidate parquet files as
    ``tess_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["tess_n_sectors"] = 0
    df["tess_total_points"] = 0
    df["tess_flux_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    _safe_print(f"TESS LCs: fetching {n_valid} light curves")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["tess_n_sectors", "tess_total_points", "tess_flux_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="TESS LCs",
        file_prefix="tess_lc",
        summary_cols=summary_cols,
        match_col="tess_n_sectors",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, TESS_SEARCH_RADIUS_ARCSEC, "tess"),
        summarize_func=lambda lc: _summarize_flux_lc(lc, "sector", "tess_n_sectors", "tess_total_points", "tess_flux_range"),
    )
    if not valid_idx:
        _safe_print(f"TESS LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, 0, np.nan, None, None)

        cache_key = _coord_lookup_cache_key(df, idx, TESS_SEARCH_RADIUS_ARCSEC, "tess")

        def _attempt_fetch() -> tuple:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            search = lk.search_lightcurve(coord, radius=21, mission="TESS")
            if search is None or len(search) == 0:
                return (idx, 0, 0, np.nan, None, cache_key)

            # Prefer SPOC 2-min, then QLP, then any
            spoc = search[search.author == "SPOC"]
            if len(spoc) > 0:
                lc_collection = spoc.download_all(quality_bitmask="default")
            else:
                qlp = search[search.author == "QLP"]
                if len(qlp) > 0:
                    lc_collection = qlp.download_all(quality_bitmask="default")
                else:
                    lc_collection = search.download_all(quality_bitmask="default")

            if lc_collection is None or len(lc_collection) == 0:
                return (idx, 0, 0, np.nan, None, cache_key)

            rows = []
            sectors = set()
            for lc_obj in lc_collection:
                t = lc_obj.time.value
                f = lc_obj.flux.value
                fe = lc_obj.flux_err.value if lc_obj.flux_err is not None else np.full_like(f, np.nan)
                q = lc_obj.quality.value if hasattr(lc_obj, "quality") and lc_obj.quality is not None else np.zeros(len(t), dtype=int)
                sector = getattr(lc_obj.meta, "SECTOR", None) if hasattr(lc_obj, "meta") else None
                if sector is None:
                    sector = getattr(lc_obj, "SECTOR", 0)
                sectors.add(sector)
                for j in range(len(t)):
                    rows.append({
                        "time": float(t[j]),
                        "flux": float(f[j]),
                        "flux_err": float(fe[j]),
                        "quality": int(q[j]),
                        "sector": int(sector) if sector is not None else 0,
                    })

            if not rows:
                return (idx, 0, 0, np.nan, None, cache_key)

            lc_df = pd.DataFrame(rows)
            lc_df = lc_df[np.isfinite(lc_df["flux"])].copy()

            summary = _summarize_flux_lc(lc_df, "sector", "tess_n_sectors", "tess_total_points", "tess_flux_range")

            if output_dir and not lc_df.empty:
                _write_external_lc_file(output_dir, "tess_lc", df, idx, lc_df)

            return (
                idx,
                summary["tess_n_sectors"],
                summary["tess_total_points"],
                summary["tess_flux_range"],
                None,
                cache_key,
            )

        for attempt in range(2):
            try:
                return _attempt_fetch()
            except Exception as exc:
                if attempt == 0 and _tess_error_is_retryable(exc):
                    _purge_tess_bad_cache_files(_tess_cache_paths_from_error(exc))
                    continue
                return (idx, 0, 0, np.nan, f"{idx}: {_short_error(exc)}", cache_key)

        return (idx, 0, 0, np.nan, f"{idx}: TESS retry exhausted", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []

    # lightkurve queries MAST — use low parallelism
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TESS LCs"):
            idx, n_sectors, total_points, flux_range, error, cache_key = fut.result()
            if error is not None:
                failures.append(error)
                row = _external_lc_status_row(
                    df,
                    idx,
                    module="TESS LCs",
                    cache_key=cache_key,
                    summary={
                        "tess_n_sectors": 0,
                        "tess_total_points": 0,
                        "tess_flux_range": np.nan,
                        "error_message": error,
                    },
                    status="error",
                )
                if row is not None:
                    status_rows.append(row)
                continue
            df.loc[idx, "tess_n_sectors"] = n_sectors
            df.loc[idx, "tess_total_points"] = total_points
            df.loc[idx, "tess_flux_range"] = flux_range
            summary = {"tess_n_sectors": n_sectors, "tess_total_points": total_points, "tess_flux_range": flux_range}
            row = _external_lc_status_row(
                df,
                idx,
                module="TESS LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if n_sectors > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)
            if n_sectors > 0:
                matched += 1

    _write_external_lc_status(output_dir, status_rows)
    if failures:
        _safe_print(f"TESS LCs: {matched}/{n_valid} with data; {len(failures)} lookup error(s) recorded")
    else:
        _safe_print(f"TESS LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# KEPLER/K2 LIGHT CURVES
# =============================================================================


def fetch_kepler_k2_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 2,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch Kepler/K2 light curves via ``lightkurve``.

    Adds columns: kepler_n_quarters, kepler_total_points, kepler_flux_range.
    If *output_dir* is set, saves per-candidate parquet files as
    ``kepler_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["kepler_n_quarters"] = 0
    df["kepler_total_points"] = 0
    df["kepler_flux_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"Kepler/K2 LCs: fetching {n_valid} light curves")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["kepler_n_quarters", "kepler_total_points", "kepler_flux_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="Kepler LCs",
        file_prefix="kepler_lc",
        summary_cols=summary_cols,
        match_col="kepler_n_quarters",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, 21.0, "kepler_k2"),
        summarize_func=lambda lc: _summarize_flux_lc(lc, "quarter", "kepler_n_quarters", "kepler_total_points", "kepler_flux_range"),
    )
    if not valid_idx:
        print(f"Kepler/K2 LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, 0, np.nan, None, None)

        cache_key = _coord_lookup_cache_key(df, idx, 21.0, "kepler_k2")

        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            search = lk.search_lightcurve(coord, radius=21, mission=("Kepler", "K2"))
            if search is None or len(search) == 0:
                return (idx, 0, 0, np.nan, None, cache_key)

            lc_collection = search.download_all()
            if lc_collection is None or len(lc_collection) == 0:
                return (idx, 0, 0, np.nan, None, cache_key)

            rows = []
            quarters = set()
            for lc_obj in lc_collection:
                t = lc_obj.time.value
                f = lc_obj.flux.value
                fe = lc_obj.flux_err.value if lc_obj.flux_err is not None else np.full_like(f, np.nan)
                q = lc_obj.quality.value if hasattr(lc_obj, "quality") and lc_obj.quality is not None else np.zeros(len(t), dtype=int)
                quarter = getattr(lc_obj.meta, "QUARTER", getattr(lc_obj.meta, "CAMPAIGN", None)) if hasattr(lc_obj, "meta") else None
                if quarter is None:
                    quarter = getattr(lc_obj, "QUARTER", getattr(lc_obj, "CAMPAIGN", 0))
                quarters.add(quarter)
                for j in range(len(t)):
                    rows.append({
                        "time": float(t[j]),
                        "flux": float(f[j]),
                        "flux_err": float(fe[j]),
                        "quality": int(q[j]),
                        "quarter": int(quarter) if quarter is not None else 0,
                    })

            if not rows:
                return (idx, 0, 0, np.nan, None, cache_key)

            lc_df = pd.DataFrame(rows)
            lc_df = lc_df[np.isfinite(lc_df["flux"])].copy()

            summary = _summarize_flux_lc(lc_df, "quarter", "kepler_n_quarters", "kepler_total_points", "kepler_flux_range")

            if output_dir and not lc_df.empty:
                _write_external_lc_file(output_dir, "kepler_lc", df, idx, lc_df)

            return (
                idx,
                summary["kepler_n_quarters"],
                summary["kepler_total_points"],
                summary["kepler_flux_range"],
                None,
                cache_key,
            )
        except Exception as exc:
            return (idx, 0, 0, np.nan, f"{idx}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Kepler LCs"):
            idx, n_quarters, total_points, flux_range, error, cache_key = fut.result()
            if error is not None:
                failures.append(error)
                continue
            df.loc[idx, "kepler_n_quarters"] = n_quarters
            df.loc[idx, "kepler_total_points"] = total_points
            df.loc[idx, "kepler_flux_range"] = flux_range
            summary = {"kepler_n_quarters": n_quarters, "kepler_total_points": total_points, "kepler_flux_range": flux_range}
            row = _external_lc_status_row(
                df,
                idx,
                module="Kepler LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if n_quarters > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)
            if n_quarters > 0:
                matched += 1

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("Kepler/K2 LCs", failures, n_valid)
    print(f"Kepler/K2 LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# AAVSO LIGHT CURVES
# =============================================================================


def _finite_jd(value: object) -> float | None:
    try:
        jd = float(value)
    except Exception:
        return None
    if not np.isfinite(jd):
        return None
    if 40_000.0 < jd < 100_000.0:
        jd += MJD_TO_JD
    if jd < 2_300_000.0:
        return None
    return jd


def _aavso_jd_window(df: pd.DataFrame, idx) -> tuple[float, float]:
    starts = [
        _finite_jd(df.loc[idx, col])
        for col in ("jd_first", "stats_jd_start")
        if col in df.columns
    ]
    ends = [
        _finite_jd(df.loc[idx, col])
        for col in ("jd_last", "stats_jd_end")
        if col in df.columns
    ]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    now_jd = time.time() / 86400.0 + 2440587.5
    from_jd = min(starts) - 365.0 if starts else AAVSO_DEFAULT_FROM_JD
    to_jd = max(ends) + 365.0 if ends else now_jd
    if to_jd < from_jd:
        to_jd = now_jd
    return max(2_300_000.0, from_jd), min(now_jd + 7.0, to_jd)


def _normalize_aavso_identifier(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "[]", "{}"}:
        return ""
    text = re.sub(r"^(v\*|var)\s+", "", text, flags=re.I).strip()
    if re.fullmatch(r"\d+", text):
        return ""
    if re.match(r"^(gaia|tic)\b", text, flags=re.I):
        return ""
    return text[:96]


def _best_aavso_identifier(row: pd.Series, name_cols: tuple[str, ...] = AAVSO_NAME_COLUMNS) -> str:
    for col in name_cols:
        if col not in row.index:
            continue
        ident = _normalize_aavso_identifier(row[col])
        if ident:
            return ident
    return ""


def _parse_aavso_vsx_response(text: str) -> pd.DataFrame:
    raw = str(text or "")
    raw_lower = raw[:4096].lower()
    if "<html" in raw_lower or "human verification" in raw_lower or "awswaf" in raw_lower:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band"])
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise RuntimeError(f"AAVSO XML parse failed: {_short_error(exc)}") from exc
    data_node = root.find("Data")
    csv_text = data_node.text if data_node is not None else ""
    if not csv_text or not csv_text.strip():
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band"])
    lc_df = pd.read_csv(io.StringIO(csv_text))
    if lc_df.empty or "JD" not in lc_df.columns or "mag" not in lc_df.columns:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band"])
    lc_df["JD"] = pd.to_numeric(lc_df["JD"], errors="coerce")
    lc_df["mag"] = pd.to_numeric(lc_df["mag"].astype(str).str.replace("<", "", regex=False), errors="coerce")
    if "uncert" in lc_df.columns:
        lc_df["uncert"] = pd.to_numeric(lc_df["uncert"], errors="coerce")
    else:
        lc_df["uncert"] = np.nan
    lc_df = lc_df.dropna(subset=["JD", "mag"]).copy()
    if lc_df.empty:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band"])
    band = lc_df["band"] if "band" in lc_df.columns else pd.Series("", index=lc_df.index)

    out = pd.DataFrame(
        {
            "mjd": lc_df["JD"] - MJD_TO_JD,
            "mag": lc_df["mag"],
            "mag_err": lc_df["uncert"],
            "band": band.astype(str).str.strip(),
        }
    )
    if "by" in lc_df.columns:
        out["observer"] = lc_df["by"].astype(str).str.strip()
    if "starName" in lc_df.columns:
        out["aavso_name"] = lc_df["starName"].astype(str).str.strip()
    if "obsID" in lc_df.columns:
        out["obs_id"] = lc_df["obsID"]
    if "obsType" in lc_df.columns:
        out["obs_type"] = lc_df["obsType"].astype(str).str.strip()
    if "mtype" in lc_df.columns:
        out["mtype"] = lc_df["mtype"].astype(str).str.strip()
    if "fainterThan" in lc_df.columns:
        out["fainter_than"] = pd.to_numeric(lc_df["fainterThan"], errors="coerce").fillna(0).astype(int)
    auid_node = root.find("AUID")
    name_node = root.find("Name")
    if auid_node is not None and auid_node.text:
        out["auid"] = auid_node.text.strip()
    if name_node is not None and name_node.text:
        out["vsx_name"] = name_node.text.strip()
    return out.sort_values("mjd").reset_index(drop=True)


def _query_aavso_vsx_lightcurve(identifier: str, from_jd: float, to_jd: float, max_points: int = AAVSO_MAX_POINTS) -> pd.DataFrame:
    params = {
        "view": "api.object",
        "ident": identifier,
        "data": int(max_points),
        "fromjd": f"{from_jd:.5f}",
        "tojd": f"{to_jd:.5f}",
        "csv": "",
        "mtype": "std",
    }
    headers = {"User-Agent": "malca-external-lcs/1.0 (+https://github.com)"}
    last_error: Exception | None = None
    for url in AAVSO_VSX_API_URLS:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=VETTING_HTTP_TIMEOUT)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                status = getattr(response, "status_code", None)
                if status in {403, 404, 405}:
                    last_error = exc
                    continue
                raise
            return _parse_aavso_vsx_response(response.text)
        except requests.HTTPError as exc:
            last_error = exc
            continue
    if last_error is not None:
        status = getattr(getattr(last_error, "response", None), "status_code", None)
        if status in {403, 404, 405} or "403" in str(last_error) or "404" in str(last_error) or "405" in str(last_error):
            return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band"])
        raise last_error
    return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band"])


def fetch_aavso_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch AAVSO light curves via the public VSX light-curve API.

    Adds columns: aavso_lc_n_points.
    If *output_dir* is set, saves per-candidate parquet files as
    ``aavso_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["aavso_lc_n_points"] = 0

    name_cols = tuple(c for c in AAVSO_NAME_COLUMNS if c in df.columns)
    if not name_cols:
        return df
    best_names = df.apply(lambda row: _best_aavso_identifier(row, name_cols), axis=1)
    valid = best_names.astype(str).str.len() > 0
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"AAVSO LCs: fetching {n_valid} light curves by name")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["aavso_lc_n_points"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="AAVSO LCs",
        file_prefix="aavso_lc",
        summary_cols=summary_cols,
        match_col="aavso_lc_n_points",
        cache_key_func=lambda idx: _source_lookup_cache_key(
            f"{best_names.loc[idx]}|{_aavso_jd_window(df, idx)[0]:.1f}|{_aavso_jd_window(df, idx)[1]:.1f}",
            "aavso_vsx",
        ),
        summarize_func=lambda lc: _summarize_count_lc(lc, "aavso_lc_n_points"),
    )
    if not valid_idx:
        print(f"AAVSO LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx: int) -> tuple:
        star_name = str(best_names.loc[idx])
        from_jd, to_jd = _aavso_jd_window(df, idx)
        cache_key = _source_lookup_cache_key(f"{star_name}|{from_jd:.1f}|{to_jd:.1f}", "aavso_vsx")
        try:
            lc_df = _query_aavso_vsx_lightcurve(star_name, from_jd, to_jd)
            n_points = len(lc_df)
            if output_dir and n_points > 0:
                _write_external_lc_file(output_dir, "aavso_lc", df, idx, lc_df)

            return (idx, n_points, None, cache_key)
        except Exception as exc:
            return (idx, 0, f"{idx}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="AAVSO LCs"):
            idx, n_points, error, cache_key = fut.result()
            if error is not None:
                failures.append(error)
                continue
            df.loc[idx, "aavso_lc_n_points"] = n_points
            summary = {"aavso_lc_n_points": n_points}
            row = _external_lc_status_row(
                df,
                idx,
                module="AAVSO LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if n_points > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)
            if n_points > 0:
                matched += 1

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("AAVSO LCs", failures, n_valid)
    print(f"AAVSO LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# OGLE LIGHT CURVES
# =============================================================================


def _normalize_ogle_source_name(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return ""
    return re.sub(r"\s+", "-", text)


def _ogle_ocvs_path_parts(source_name: str) -> tuple[str, str] | None:
    parts = _normalize_ogle_source_name(source_name).split("-")
    if len(parts) < 3 or parts[0].upper() != "OGLE":
        return None
    region = parts[1].lower()
    cls = parts[2].lower()
    class_map = {
        "rrlyr": "rrlyr",
        "rrl": "rrlyr",
        "cep": "cep",
        "t2cep": "t2cep",
        "acep": "acep",
        "ecl": "ecl",
        "lpv": "lpv",
        "dsct": "dsct",
        "rot": "rot",
        "mira": "lpv",
    }
    return region, class_map.get(cls, cls)


def _ogle_candidate_urls(source_name: str, band: str) -> list[str]:
    name = _normalize_ogle_source_name(source_name)
    parts = _ogle_ocvs_path_parts(name)
    if not name or parts is None:
        return []
    region, cls = parts
    band_dir = str(band).upper()
    urls = []
    for base in OGLE_OCVS_BASE_URLS:
        urls.append(f"{base}/{region}/{cls}/phot/{band_dir}/{name}.dat")
    return urls


def _parse_ogle_dat(text: str, *, source_name: str, band: str, url: str = "") -> pd.DataFrame:
    rows: list[tuple[float, float, float]] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band", "ogle_name", "source_url"])
    out = pd.DataFrame(rows, columns=["mjd", "mag", "mag_err"])
    finite = out["mjd"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        median = float(np.nanmedian(finite))
        if median > 2_400_000.0:
            out["mjd"] = out["mjd"] - MJD_TO_JD
        elif median < 20_000.0:
            out["mjd"] = out["mjd"] + 50_000.0 - 0.5
    out["band"] = str(band).upper()
    out["ogle_name"] = _normalize_ogle_source_name(source_name)
    out["source_url"] = str(url)
    return out.dropna(subset=["mjd", "mag"]).reset_index(drop=True)


def _download_ogle_band(source_name: str, band: str) -> pd.DataFrame:
    for url in _ogle_candidate_urls(source_name, band):
        response = requests.get(url, timeout=VETTING_HTTP_TIMEOUT)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        lc = _parse_ogle_dat(response.text, source_name=source_name, band=band, url=url)
        if not lc.empty:
            return lc
    return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band", "ogle_name", "source_url"])


def _match_ogle_names_by_coord(df: pd.DataFrame, candidate_indices: list) -> pd.Series:
    names = pd.Series("", index=df.index, dtype=object)
    if not candidate_indices:
        return names
    try:
        from malca.catalogs.periodic_catalogs import fetch_ogle_periodic_catalog

        cat = fetch_ogle_periodic_catalog(show_tqdm=False)
    except Exception:
        return names
    if cat.empty or not {"ra", "dec", "source_name"}.issubset(cat.columns):
        return names
    cand = df.loc[candidate_indices]
    valid = cand["ra"].notna() & cand["dec"].notna()
    if not valid.any():
        return names
    cat_ra = pd.to_numeric(cat["ra"], errors="coerce")
    cat_dec = pd.to_numeric(cat["dec"], errors="coerce")
    valid_cat = cat_ra.notna() & cat_dec.notna()
    if not valid_cat.any():
        return names
    cand_coords = SkyCoord(ra=cand.loc[valid, "ra"].astype(float).to_numpy() * u.deg, dec=cand.loc[valid, "dec"].astype(float).to_numpy() * u.deg)
    cat_valid = cat.loc[valid_cat].reset_index(drop=True)
    cat_coords = SkyCoord(ra=cat_valid["ra"].astype(float).to_numpy() * u.deg, dec=cat_valid["dec"].astype(float).to_numpy() * u.deg)
    idx_cat, sep, _ = cand_coords.match_to_catalog_sky(cat_coords)
    for row_idx, cat_idx, sep_arcsec in zip(cand.loc[valid].index, idx_cat, sep.arcsec):
        if float(sep_arcsec) <= OGLE_LC_MAX_SEP_ARCSEC:
            names.loc[row_idx] = _normalize_ogle_source_name(cat_valid.iloc[int(cat_idx)]["source_name"])
    return names


def fetch_ogle_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch OGLE OCVS I/V light curves for candidates with an OGLE source name."""
    df = df.copy()
    for col in ("ogle_lc_n_points", "ogle_lc_i_range", "ogle_lc_v_range"):
        df[col] = 0 if col == "ogle_lc_n_points" else np.nan

    name_cols = [c for c in ("period_ogle_name", "ogle_name", "source_name") if c in df.columns]
    source_names = pd.Series("", index=df.index, dtype=object)
    for idx in df.index:
        for col in name_cols:
            name = _normalize_ogle_source_name(df.loc[idx, col])
            if name:
                source_names.loc[idx] = name
                break

    missing_name_idx = [idx for idx in df.index if not source_names.loc[idx]]
    if missing_name_idx and {"ra", "dec"}.issubset(df.columns):
        matched_names = _match_ogle_names_by_coord(df, missing_name_idx)
        for idx, name in matched_names.items():
            if name and not source_names.loc[idx]:
                source_names.loc[idx] = name

    valid = source_names.astype(str).str.len() > 0
    if not valid.any():
        return df
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"OGLE LCs: fetching {n_valid} light curves")
    valid_idx = df.index[valid].tolist()
    summary_cols = ["ogle_lc_n_points", "ogle_lc_i_range", "ogle_lc_v_range"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="OGLE LCs",
        file_prefix="ogle_lc",
        summary_cols=summary_cols,
        match_col="ogle_lc_n_points",
        cache_key_func=lambda idx: _source_lookup_cache_key(source_names.loc[idx], "ogle_ocvs"),
        summarize_func=_summarize_ogle_lc,
    )
    if not valid_idx:
        print(f"OGLE LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx) -> tuple:
        name = source_names.loc[idx]
        cache_key = _source_lookup_cache_key(name, "ogle_ocvs")
        try:
            parts = [_download_ogle_band(name, band) for band in ("I", "V")]
            lc_df = pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not part.empty for part in parts) else pd.DataFrame()
            if lc_df.empty:
                summary = {"ogle_lc_n_points": 0, "ogle_lc_i_range": np.nan, "ogle_lc_v_range": np.nan}
                return (idx, summary, None, cache_key, None)
            lc_df = lc_df.sort_values(["mjd", "band"]).reset_index(drop=True)
            summary = _summarize_ogle_lc(lc_df)
            if output_dir:
                _write_external_lc_file(output_dir, "ogle_lc", df, idx, lc_df)
            return (idx, summary, None, cache_key, lc_df)
        except Exception as exc:
            return (idx, {"ogle_lc_n_points": 0, "ogle_lc_i_range": np.nan, "ogle_lc_v_range": np.nan}, f"{_candidate_cache_id(df, idx)}: {_short_error(exc)}", cache_key, None)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="OGLE LCs"):
            idx, summary, error, cache_key, _lc_df = fut.result()
            if error is not None:
                failures.append(error)
                summary = dict(summary)
                summary["error_message"] = error
                row = _external_lc_status_row(df, idx, module="OGLE LCs", cache_key=cache_key, summary=summary, status="error")
                if row is not None:
                    status_rows.append(row)
                continue
            for col in summary_cols:
                df.loc[idx, col] = summary.get(col, np.nan)
            if int(summary.get("ogle_lc_n_points") or 0) > 0:
                matched += 1
            row = _external_lc_status_row(
                df,
                idx,
                module="OGLE LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if int(summary.get("ogle_lc_n_points") or 0) > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("OGLE LCs", failures, n_valid)
    print(f"OGLE LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# SDSS STRIPE 82 LIGHT CURVES
# =============================================================================


def _stripe82_in_footprint(ra: float, dec: float) -> bool:
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return False
    return abs(float(dec)) <= 1.35 and (float(ra) >= 308.5 or float(ra) <= 60.0)


def _stripe82_link_candidates() -> tuple[list[str], list[str]]:
    master_urls = list(STRIPE82_MASTER_FALLBACK_URLS)
    archive_urls = list(STRIPE82_LC_ARCHIVE_FALLBACK_URLS)
    try:
        response = requests.get(STRIPE82_VARIABLES_URL, timeout=VETTING_HTTP_TIMEOUT)
        response.raise_for_status()
        hrefs = re.findall(r"""href=["']([^"']+)["']""", response.text, flags=re.I)
        for href in hrefs:
            url = urljoin(STRIPE82_VARIABLES_URL, href)
            lower = url.lower()
            if "alllc" in lower or lower.endswith(".tar.gz"):
                archive_urls.append(url)
            elif lower.endswith(".gz"):
                master_urls.append(url)
    except Exception:
        pass
    return list(dict.fromkeys(master_urls)), list(dict.fromkeys(archive_urls))


def _load_stripe82_master() -> pd.DataFrame:
    cache_dir = _external_catalog_cache_dir("stripe82")
    cache_file = cache_dir / "master.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    master_urls, _archive_urls = _stripe82_link_candidates()
    last_error: Exception | None = None
    names = [
        "stripe82_id",
        "ra",
        "dec",
        "period",
        "r_mag",
        "ug",
        "gr",
        "ri",
        "iz",
        "g_n",
        "g_ampl",
        "r_n",
        "r_ampl",
        "i_n",
        "i_ampl",
        "z_qso",
        "mi_qso",
    ]
    for i, url in enumerate(master_urls):
        try:
            path = _download_to_cache(url, cache_dir / f"master_{i}.dat.gz")
            table = pd.read_csv(path, comment="#", sep=r"\s+", names=names, engine="python")
            table = table.dropna(subset=["stripe82_id", "ra", "dec"])
            table["stripe82_id"] = table["stripe82_id"].astype(str)
            table.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
            return table
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Stripe 82 master catalog download/parse failed: {_short_error(last_error or RuntimeError('no URLs'))}")


def _stripe82_archive_path() -> Path:
    cache_dir = _external_catalog_cache_dir("stripe82")
    _master_urls, archive_urls = _stripe82_link_candidates()
    last_error: Exception | None = None
    for i, url in enumerate(archive_urls):
        try:
            return _download_to_cache(url, cache_dir / f"AllLCs_{i}.tar.gz", timeout=300.0)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Stripe 82 LC archive download failed: {_short_error(last_error or RuntimeError('no URLs'))}")


def _read_stripe82_lc_member(tar: tarfile.TarFile, member_name: str, stripe82_id: str) -> pd.DataFrame:
    fh = tar.extractfile(member_name)
    if fh is None:
        return pd.DataFrame(columns=["mjd", "band", "mag", "mag_err", "stripe82_id"])
    text = fh.read().decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            rows.append((float(parts[0]), str(parts[1]).lower(), float(parts[2]), float(parts[3])))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["mjd", "band", "mag", "mag_err", "stripe82_id"])
    out = pd.DataFrame(rows, columns=["mjd", "band", "mag", "mag_err"])
    out["stripe82_id"] = str(stripe82_id)
    out = out.dropna(subset=["mjd", "mag"])
    return out.sort_values("mjd").reset_index(drop=True)


def fetch_stripe82_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch UW SDSS Stripe 82 variable-source ugriz light curves."""
    df = df.copy()
    summary_cols = [
        "stripe82_lc_n_points",
        "stripe82_lc_u_range",
        "stripe82_lc_g_range",
        "stripe82_lc_r_range",
        "stripe82_lc_i_range",
        "stripe82_lc_z_range",
    ]
    for col in summary_cols:
        df[col] = 0 if col == "stripe82_lc_n_points" else np.nan
    if not {"ra", "dec"}.issubset(df.columns):
        return df

    valid = df.apply(lambda row: _stripe82_in_footprint(float(row["ra"]) if pd.notna(row["ra"]) else np.nan, float(row["dec"]) if pd.notna(row["dec"]) else np.nan), axis=1)
    if not valid.any():
        return df
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_valid = int(valid.sum())
    print(f"Stripe 82 LCs: fetching {n_valid} footprint candidates")

    valid_idx = df.index[valid].tolist()
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="Stripe 82 LCs",
        file_prefix="stripe82_lc",
        summary_cols=summary_cols,
        match_col="stripe82_lc_n_points",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, STRIPE82_MAX_SEP_ARCSEC, "stripe82"),
        summarize_func=_summarize_stripe82_lc,
    )
    if not valid_idx:
        print(f"Stripe 82 LCs: {cached_matched}/{n_valid} with data")
        return df

    failures: list[str] = []
    status_rows: list[dict] = []
    matched = cached_matched
    try:
        master = _load_stripe82_master()
        archive_path = _stripe82_archive_path()
    except Exception as exc:
        raise RuntimeError(f"Stripe 82 setup failed: {_short_error(exc)}") from exc

    cand = df.loc[valid_idx]
    cand_coords = SkyCoord(ra=cand["ra"].astype(float).to_numpy() * u.deg, dec=cand["dec"].astype(float).to_numpy() * u.deg)
    master_coords = SkyCoord(ra=master["ra"].astype(float).to_numpy() * u.deg, dec=master["dec"].astype(float).to_numpy() * u.deg)
    idx_master, sep, _ = cand_coords.match_to_catalog_sky(master_coords)

    with tarfile.open(archive_path, "r:gz") as tar:
        members_by_base = {Path(member.name).name: member.name for member in tar.getmembers() if member.isfile()}
        for row_idx, master_idx, sep_arcsec in tqdm(list(zip(valid_idx, idx_master, sep.arcsec)), desc="Stripe 82 LCs"):
            cache_key = _coord_lookup_cache_key(df, row_idx, STRIPE82_MAX_SEP_ARCSEC, "stripe82")
            summary = {col: (0 if col == "stripe82_lc_n_points" else np.nan) for col in summary_cols}
            if float(sep_arcsec) > STRIPE82_MAX_SEP_ARCSEC:
                row = _external_lc_status_row(df, row_idx, module="Stripe 82 LCs", cache_key=cache_key, summary=summary, status="no_data")
                if row is not None:
                    status_rows.append(row)
                continue
            stripe82_id = str(master.iloc[int(master_idx)]["stripe82_id"])
            basename = f"LC_{stripe82_id}.dat"
            member_name = members_by_base.get(basename)
            if member_name is None:
                row = _external_lc_status_row(df, row_idx, module="Stripe 82 LCs", cache_key=cache_key, summary=summary, status="no_data")
                if row is not None:
                    status_rows.append(row)
                continue
            try:
                lc_df = _read_stripe82_lc_member(tar, member_name, stripe82_id)
                if not lc_df.empty and output_dir:
                    _write_external_lc_file(output_dir, "stripe82_lc", df, row_idx, lc_df)
                summary = _summarize_stripe82_lc(lc_df) if not lc_df.empty else summary
                for col in summary_cols:
                    df.loc[row_idx, col] = summary.get(col, np.nan)
                if int(summary.get("stripe82_lc_n_points") or 0) > 0:
                    matched += 1
                row = _external_lc_status_row(df, row_idx, module="Stripe 82 LCs", cache_key=cache_key, summary=summary, status="fetched" if int(summary.get("stripe82_lc_n_points") or 0) > 0 else "no_data")
                if row is not None:
                    status_rows.append(row)
            except Exception as exc:
                short = _short_error(exc)
                failures.append(f"{_candidate_cache_id(df, row_idx)}: {short}")
                summary["error_message"] = short
                row = _external_lc_status_row(df, row_idx, module="Stripe 82 LCs", cache_key=cache_key, summary=summary, status="error")
                if row is not None:
                    status_rows.append(row)

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("Stripe 82 LCs", failures, n_valid)
    print(f"Stripe 82 LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# ALLWISE MULTIEPOCH LIGHT CURVES
# =============================================================================


def _normalize_allwise_mep_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    rename = {
        "w1mpro_ep": "w1mpro",
        "w1sigmpro_ep": "w1sigmpro",
        "w2mpro_ep": "w2mpro",
        "w2sigmpro_ep": "w2sigmpro",
        "w3mpro_ep": "w3mpro",
        "w3sigmpro_ep": "w3sigmpro",
        "w4mpro_ep": "w4mpro",
        "w4sigmpro_ep": "w4sigmpro",
        "mjd_ep": "mjd",
    }
    lower_to_actual = {str(col).lower(): col for col in df.columns}
    for old, new in rename.items():
        actual = lower_to_actual.get(old)
        if actual is not None and new not in df.columns:
            df = df.rename(columns={actual: new})
    if "mjd" not in df.columns:
        time_col = next((lower_to_actual.get(name) for name in ("mjd", "mjd_ep", "jd") if lower_to_actual.get(name)), None)
        if time_col:
            df = df.rename(columns={time_col: "mjd"})
    for col in ("mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro", "w3mpro", "w3sigmpro", "w4mpro", "w4sigmpro"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mjd" in df.columns:
        finite = df["mjd"].dropna()
        if len(finite) and float(finite.median()) > 1_000_000.0:
            df["mjd"] = df["mjd"] - MJD_TO_JD
    mag_cols = [col for col in ("w1mpro", "w2mpro", "w3mpro", "w4mpro") if col in df.columns]
    if "mjd" in df.columns and mag_cols:
        df = df.dropna(subset=["mjd"], how="any")
        df = df[df[mag_cols].notna().any(axis=1)]
    return df.reset_index(drop=True)


def _query_allwise_mep_one(ra: float, dec: float, max_sep_arcsec: float = ALLWISE_MEP_MAX_SEP_ARCSEC) -> pd.DataFrame:
    radius_deg = float(max_sep_arcsec) / 3600.0
    source_query = f"""
    SELECT TOP 1 source_id, cntr, ra, dec
    FROM allwise_p3as_psd
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_deg:.10f})
    ) = 1
    """
    source_result = Irsa.query_tap(source_query).to_table().to_pandas()
    if source_result.empty:
        return pd.DataFrame()
    source_row = source_result.iloc[0]
    source_id = str(source_row.get("source_id", "")).strip()
    cntr = source_row.get("cntr", None)
    where = f"source_id_mf = '{source_id}'" if source_id else ""
    if not where and pd.notna(cntr):
        where = f"cntr_mf = {int(cntr)}"
    if not where:
        return pd.DataFrame()
    mep_query = f"""
    SELECT mjd, source_id_mf, cntr_mf,
           w1mpro_ep, w1sigmpro_ep, w2mpro_ep, w2sigmpro_ep,
           w3mpro_ep, w3sigmpro_ep, w4mpro_ep, w4sigmpro_ep,
           qi_fact, saa_sep, moon_masked
    FROM allwise_p3as_mep
    WHERE {where}
    ORDER BY mjd ASC
    """
    mep = Irsa.query_tap(mep_query).to_table().to_pandas()
    return _normalize_allwise_mep_table(mep)


def _allwise_mep_error_is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ALLWISE_MEP_RETRY_MARKERS)


def _query_allwise_mep_one_with_retry(
    ra: float,
    dec: float,
    max_sep_arcsec: float = ALLWISE_MEP_MAX_SEP_ARCSEC,
    *,
    max_attempts: int = ALLWISE_MEP_MAX_ATTEMPTS,
) -> pd.DataFrame:
    attempts = max(1, int(max_attempts))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return _query_allwise_mep_one(ra, dec, max_sep_arcsec=max_sep_arcsec)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _allwise_mep_error_is_retryable(exc):
                raise
            time.sleep(min(8.0, ALLWISE_MEP_RETRY_BASE_DELAY * (2 ** attempt)))
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def fetch_allwise_mep_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch historical AllWISE Multiepoch W1-W4 photometry."""
    df = df.copy()
    summary_cols = ["allwise_mep_n_epochs", "allwise_mep_w1_range", "allwise_mep_w2_range", "allwise_mep_w3_range", "allwise_mep_w4_range"]
    for col in summary_cols:
        df[col] = 0 if col == "allwise_mep_n_epochs" else np.nan
    valid = df["ra"].notna() & df["dec"].notna() if {"ra", "dec"}.issubset(df.columns) else pd.Series(False, index=df.index)
    if not valid.any():
        return df
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_valid = int(valid.sum())
    print(f"AllWISE MEP LCs: fetching {n_valid} light curves")
    valid_idx = df.index[valid].tolist()
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="AllWISE MEP LCs",
        file_prefix="allwise_mep_lc",
        summary_cols=summary_cols,
        match_col="allwise_mep_n_epochs",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, ALLWISE_MEP_MAX_SEP_ARCSEC, "allwise_mep"),
        summarize_func=_summarize_allwise_mep_lc,
    )
    if not valid_idx:
        print(f"AllWISE MEP LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx) -> tuple:
        cache_key = _coord_lookup_cache_key(df, idx, ALLWISE_MEP_MAX_SEP_ARCSEC, "allwise_mep")
        try:
            lc_df = _query_allwise_mep_one_with_retry(float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"]))
            if lc_df.empty:
                summary = {col: (0 if col == "allwise_mep_n_epochs" else np.nan) for col in summary_cols}
                return (idx, summary, None, cache_key)
            summary = _summarize_allwise_mep_lc(lc_df)
            if output_dir:
                _write_external_lc_file(output_dir, "allwise_mep_lc", df, idx, lc_df)
            return (idx, summary, None, cache_key)
        except Exception as exc:
            return (idx, {col: (0 if col == "allwise_mep_n_epochs" else np.nan) for col in summary_cols}, f"{_candidate_cache_id(df, idx)}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="AllWISE MEP LCs"):
            idx, summary, error, cache_key = fut.result()
            status = "fetched" if int(summary.get("allwise_mep_n_epochs") or 0) > 0 else "no_data"
            if error is not None:
                failures.append(error)
                summary = dict(summary)
                summary["error_message"] = error
                status = "error"
            for col in summary_cols:
                df.loc[idx, col] = summary.get(col, np.nan)
            if status == "fetched":
                matched += 1
            row = _external_lc_status_row(df, idx, module="AllWISE MEP LCs", cache_key=cache_key, summary=summary, status=status)
            if row is not None:
                status_rows.append(row)

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("AllWISE MEP LCs", failures, n_valid)
    print(f"AllWISE MEP LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# VVVX/VIRAC2 LIGHT CURVES
# =============================================================================


def _first_column_by_lower(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _normalize_vvvx_virac_table(raw: pd.DataFrame, source_id: object = None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band", "virac_source_id"])
    df = raw.copy()
    time_col = _first_column_by_lower(df, ("mjd", "mjdobs", "mjd_obs", "hmjd", "hjd", "jd", "epoch_mjd"))
    if time_col is None:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band", "virac_source_id"])
    band_col = _first_column_by_lower(df, ("band", "filter", "filter_name", "filterid"))
    mag_col = _first_column_by_lower(df, ("mag", "m", "aper_mag", "psf_mag", "mag_auto"))
    err_col = _first_column_by_lower(df, ("mag_err", "magerr", "emag", "e_mag", "mag_error", "err"))
    id_col = _first_column_by_lower(df, ("sourceid", "source_id", "sourceid_vvv", "virac_id", "srcid"))
    rows: list[dict] = []
    if band_col and mag_col:
        for _, row in df.iterrows():
            rows.append(
                {
                    "mjd": row.get(time_col),
                    "mag": row.get(mag_col),
                    "mag_err": row.get(err_col) if err_col else np.nan,
                    "band": str(row.get(band_col, "")).strip().lower().replace("k_s", "ks"),
                    "virac_source_id": row.get(id_col) if id_col else source_id,
                }
            )
    else:
        lower_cols = {str(col).lower(): col for col in df.columns}
        band_aliases = {
            "z": ("zmag", "z_mag", "mag_z", "z"),
            "y": ("ymag", "y_mag", "mag_y", "y"),
            "j": ("jmag", "j_mag", "mag_j", "j"),
            "h": ("hmag", "h_mag", "mag_h", "h"),
            "ks": ("ksmag", "ks_mag", "mag_ks", "ks", "k_s"),
        }
        err_aliases = {
            "z": ("ezmag", "zerr", "z_mag_err", "e_zmag", "e_z"),
            "y": ("eymag", "yerr", "y_mag_err", "e_ymag", "e_y"),
            "j": ("ejmag", "jerr", "j_mag_err", "e_jmag", "e_j"),
            "h": ("ehmag", "herr", "h_mag_err", "e_hmag", "e_h"),
            "ks": ("eksmag", "kserr", "ks_mag_err", "e_ksmag", "e_ks"),
        }
        for band, aliases in band_aliases.items():
            b_col = next((lower_cols[a] for a in aliases if a in lower_cols), None)
            if b_col is None:
                continue
            e_col = next((lower_cols[a] for a in err_aliases[band] if a in lower_cols), None)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "mjd": row.get(time_col),
                        "mag": row.get(b_col),
                        "mag_err": row.get(e_col) if e_col else np.nan,
                        "band": band,
                        "virac_source_id": row.get(id_col) if id_col else source_id,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["mjd", "mag", "mag_err", "band", "virac_source_id"])
    out["mjd"] = pd.to_numeric(out["mjd"], errors="coerce")
    out["mag"] = pd.to_numeric(out["mag"], errors="coerce")
    out["mag_err"] = pd.to_numeric(out["mag_err"], errors="coerce")
    finite = out["mjd"].dropna()
    if len(finite) and float(finite.median()) > 1_000_000.0:
        out["mjd"] = out["mjd"] - MJD_TO_JD
    out["band"] = out["band"].astype(str).str.lower().str.replace("k_s", "ks")
    out = out.dropna(subset=["mjd", "mag"])
    out = out[out["band"].isin(["z", "y", "j", "h", "ks"])]
    return out.sort_values(["mjd", "band"]).reset_index(drop=True)


def _query_vvvx_virac_one(ra: float, dec: float, max_sep_arcsec: float = VVVX_VIRAC_MAX_SEP_ARCSEC) -> pd.DataFrame:
    service = pyvo.dal.TAPService(ESO_TAP_CAT_URL)
    radius_deg = float(max_sep_arcsec) / 3600.0
    # VIRAC2 uses ``de`` rather than the more common ``dec`` column label.
    source_query = f"""
    SELECT TOP 1 *
    FROM VVVX_VIRAC_V2_SOURCES
    WHERE CONTAINS(
        POINT('ICRS', ra, de),
        CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_deg:.10f})
    ) = 1
    """
    source_table = service.search(source_query).to_table().to_pandas()
    if source_table.empty:
        return pd.DataFrame()
    source_row = source_table.iloc[0]
    id_col = _first_column_by_lower(source_table, ("sourceid", "source_id", "virac_id", "srcid"))
    source_id = source_row.get(id_col) if id_col else None
    queries = []
    if source_id is not None and pd.notna(source_id):
        id_value = str(source_id).strip()
        id_expr = id_value if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", id_value) else f"'{id_value}'"
        queries.append(f"SELECT TOP 5000 * FROM VVVX_VIRAC_V2_LC WHERE sourceid = {id_expr}")
    queries.append(
        f"""
        SELECT TOP 5000 *
        FROM VVVX_VIRAC_V2_LC
        WHERE CONTAINS(
            POINT('ICRS', ra, de),
            CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_deg:.10f})
        ) = 1
        """
    )
    last_error: Exception | None = None
    for query in queries:
        try:
            lc = service.search(query).to_table().to_pandas()
            norm = _normalize_vvvx_virac_table(lc, source_id=source_id)
            if not norm.empty:
                return norm
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def fetch_vvvx_virac_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 2,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch VVV/VVVX VIRAC2 near-IR time-series photometry from ESO TAP."""
    df = df.copy()
    summary_cols = ["vvvx_virac_n_epochs", "vvvx_virac_z_range", "vvvx_virac_y_range", "vvvx_virac_j_range", "vvvx_virac_h_range", "vvvx_virac_ks_range"]
    for col in summary_cols:
        df[col] = 0 if col == "vvvx_virac_n_epochs" else np.nan
    valid = df["ra"].notna() & df["dec"].notna() if {"ra", "dec"}.issubset(df.columns) else pd.Series(False, index=df.index)
    if not valid.any():
        return df
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_valid = int(valid.sum())
    print(f"VVVX/VIRAC2 LCs: fetching {n_valid} light curves")
    valid_idx = df.index[valid].tolist()
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="VVVX/VIRAC2 LCs",
        file_prefix="vvvx_virac_lc",
        summary_cols=summary_cols,
        match_col="vvvx_virac_n_epochs",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, VVVX_VIRAC_MAX_SEP_ARCSEC, "vvvx_virac2"),
        summarize_func=_summarize_vvvx_virac_lc,
    )
    if not valid_idx:
        print(f"VVVX/VIRAC2 LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx) -> tuple:
        cache_key = _coord_lookup_cache_key(df, idx, VVVX_VIRAC_MAX_SEP_ARCSEC, "vvvx_virac2")
        try:
            lc_df = _query_vvvx_virac_one(float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"]))
            if lc_df.empty:
                summary = {col: (0 if col == "vvvx_virac_n_epochs" else np.nan) for col in summary_cols}
                return (idx, summary, None, cache_key)
            summary = _summarize_vvvx_virac_lc(lc_df)
            if output_dir:
                _write_external_lc_file(output_dir, "vvvx_virac_lc", df, idx, lc_df)
            return (idx, summary, None, cache_key)
        except Exception as exc:
            return (idx, {col: (0 if col == "vvvx_virac_n_epochs" else np.nan) for col in summary_cols}, f"{_candidate_cache_id(df, idx)}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="VVVX/VIRAC2 LCs"):
            idx, summary, error, cache_key = fut.result()
            status = "fetched" if int(summary.get("vvvx_virac_n_epochs") or 0) > 0 else "no_data"
            if error is not None:
                failures.append(error)
                summary = dict(summary)
                summary["error_message"] = error
                status = "error"
            for col in summary_cols:
                df.loc[idx, col] = summary.get(col, np.nan)
            if status == "fetched":
                matched += 1
            row = _external_lc_status_row(df, idx, module="VVVX/VIRAC2 LCs", cache_key=cache_key, summary=summary, status=status)
            if row is not None:
                status_rows.append(row)

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("VVVX/VIRAC2 LCs", failures, n_valid)
    print(f"VVVX/VIRAC2 LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# PAN-STARRS LIGHT CURVES
# =============================================================================


def _panstarrs_lc_retry_delay(response: object, attempt: int) -> float:
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    if retry_after is not None:
        try:
            delay = float(retry_after)
            if np.isfinite(delay) and delay >= 0:
                return min(delay, VETTING_BACKOFF_CAP)
        except (TypeError, ValueError):
            pass
    return min(2.0 ** max(0, attempt), VETTING_BACKOFF_CAP)


def _request_panstarrs_detection_csv(url: str) -> str:
    last_status: int | None = None
    for attempt in range(PANSTARRS_LC_MAX_ATTEMPTS):
        res = requests.get(url, timeout=VETTING_HTTP_TIMEOUT)
        last_status = int(getattr(res, "status_code", 0) or 0)
        if last_status == 200:
            return getattr(res, "text", "") or ""
        if last_status in PANSTARRS_LC_RETRY_STATUSES and attempt < PANSTARRS_LC_MAX_ATTEMPTS - 1:
            time.sleep(_panstarrs_lc_retry_delay(res, attempt))
            continue
        raise RuntimeError(f"Pan-STARRS HTTP {last_status}")
    raise RuntimeError(f"Pan-STARRS HTTP {last_status}")


def fetch_panstarrs_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch Pan-STARRS (PS1 DR2) epoch photometry.

    Adds columns: ps1_lc_n_points.
    If *output_dir* is set, saves per-candidate parquet files as
    ``ps1_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["ps1_lc_n_points"] = 0

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"Pan-STARRS LCs: fetching {n_valid} light curves")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["ps1_lc_n_points"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="Pan-STARRS LCs",
        file_prefix="ps1_lc",
        summary_cols=summary_cols,
        match_col="ps1_lc_n_points",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, PANSTARRS_LC_RADIUS_DEG * 3600.0, "ps1_dr2"),
        summarize_func=lambda lc: _summarize_count_lc(lc, "ps1_lc_n_points"),
    )
    if not valid_idx:
        print(f"Pan-STARRS LCs: {cached_matched}/{n_valid} with data")
        return df

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, None, None)
        cache_key = _coord_lookup_cache_key(df, idx, PANSTARRS_LC_RADIUS_DEG * 3600.0, "ps1_dr2")

        # Skip southern hemisphere queries (-30 limit for PS1)
        if dec < -30.5:
            return (idx, 0, None, cache_key)

        try:
            url = f"https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/detection.csv?ra={ra}&dec={dec}&radius={PANSTARRS_LC_RADIUS_DEG}&pagesize=10000&format=csv"
            text = _request_panstarrs_detection_csv(url)
            if "obsTime" not in text:
                return (idx, 0, None, cache_key)

            lc_df = pd.read_csv(io.StringIO(text))
            
            if lc_df.empty or "obsTime" not in lc_df.columns:
                return (idx, 0, None, cache_key)

            # Filter by infoFlag if present
            if "infoFlag" in lc_df.columns:
                # Keep only detections without DEFECT(2048), SATURATED(4096), FIT_FAIL(8)
                bad_mask = (
                    ((lc_df["infoFlag"] & 2048) != 0) | 
                    ((lc_df["infoFlag"] & 4096) != 0) | 
                    ((lc_df["infoFlag"] & 8) != 0)
                )
                lc_df = lc_df[~bad_mask].copy()

            if lc_df.empty:
                return (idx, 0, None, cache_key)
                
            # Rename for consistency mapping
            lc_df = lc_df.rename(columns={
                "filterID": "filter",
                "obsTime": "mjd",
                "psfFlux": "flux_psf",
                "psfFluxErr": "flux_psf_err"
            })
            
            # Map filters from ID to string (1=g, 2=r, 3=i, 4=z, 5=y)
            filter_map = {1: "g_ps", 2: "r_ps", 3: "i_ps", 4: "z_ps", 5: "y_ps"}
            lc_df["filter"] = lc_df["filter"].map(filter_map)
            
            # Convert AB fluxes to AB magnitudes properly (-2.5*log10(flux) + 8.90) 
            # PS1 fluxes are in Jansky * 10^36... actually MAST API returns 
            # Jy according to MAST schema? No, it's microJanskys or similar.
            # Lightcurvy uses `mag_psf = -2.5*log10(flux_psf) + 8.90` (mJy -> AB_mag)
            
            valid_flux = lc_df["flux_psf"] > 0
            lc_df = lc_df[valid_flux].copy()
            
            lc_df["mag"] = -2.5 * np.log10(lc_df["flux_psf"]) + 8.90
            lc_df["mag_err"] = 1.08 * (lc_df["flux_psf_err"] / lc_df["flux_psf"])
            
            # Cleanup
            lc_df = lc_df.dropna(subset=["mjd", "mag"])
            # MAST's obsTime can arrive as JD; normalize to actual MJD to match our schema.
            if not lc_df.empty and float(pd.to_numeric(lc_df["mjd"], errors="coerce").median()) > 1_000_000.0:
                lc_df["mjd"] = pd.to_numeric(lc_df["mjd"], errors="coerce") - 2400000.5
            n_points = len(lc_df)

            if output_dir and n_points > 0:
                _write_external_lc_file(output_dir, "ps1_lc", df, idx, lc_df)

            return (idx, n_points, None, cache_key)
        except Exception as exc:
            return (idx, 0, f"{idx}: {_short_error(exc)}", cache_key)

    matched = cached_matched
    failures: list[str] = []
    status_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Pan-STARRS LCs"):
            idx, n_points, error, cache_key = fut.result()
            if error is not None:
                failures.append(error)
                continue
            df.loc[idx, "ps1_lc_n_points"] = n_points
            summary = {"ps1_lc_n_points": n_points}
            row = _external_lc_status_row(
                df,
                idx,
                module="Pan-STARRS LCs",
                cache_key=cache_key,
                summary=summary,
                status="fetched" if n_points > 0 else "no_data",
            )
            if row is not None:
                status_rows.append(row)
            if n_points > 0:
                matched += 1

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("Pan-STARRS LCs", failures, n_valid)
    print(f"Pan-STARRS LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# CRTS LIGHT CURVES
# =============================================================================

CRTS_CGI_URL = "http://nunuku.caltech.edu/cgi-bin/getcssconedb_priv.cgi"
CRTS_CGI_BASE_URL = "http://nunuku.caltech.edu"


class _CRTSCSVLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {str(key).lower(): value for key, value in attrs}
        href = attr_map.get("href")
        if href and ".csv" in href.lower():
            self.links.append(href)


def _extract_crts_csv_url(html: str, base_url: str = CRTS_CGI_BASE_URL) -> str | None:
    parser = _CRTSCSVLinkParser()
    parser.feed(str(html or ""))
    for href in parser.links:
        return urljoin(base_url, href)

    match = re.search(r"""(?:https?://[^\s"'<>]+|/[^\s"'<>]+)\.csv(?:\?[^\s"'<>]*)?""", str(html or ""), flags=re.I)
    if match:
        return urljoin(base_url, match.group(0))
    return None


def _crts_cgi_response_is_empty(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        marker in lower
        for marker in (
            "there were 0 lines",
            "there are 0 lines",
            "no object",
            "no match",
            "no rows",
        )
    )


def _read_crts_csv_text(csv_text: str) -> pd.DataFrame:
    text = str(csv_text or "").strip()
    if not text:
        return pd.DataFrame()
    try:
        table = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise RuntimeError(f"CRTS CSV parser failed: {_short_error(exc)}") from exc
    required = {"Mag", "MJD"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"CRTS CSV missing required columns: {', '.join(sorted(missing))}")
    return table


def _fetch_crts_cgi_catalog(ra: float, dec: float, radius_arcsec: float, catalog: str) -> pd.DataFrame:
    params = {
        "RA": f"{float(ra):.7f}",
        "Dec": f"{float(dec):.7f}",
        "Rad": f"{float(radius_arcsec) / 60.0:.5f}",
        "DB": catalog,
        "OUT": "csv",
        "SHORT": "short",
        "PLOT": "plot",
    }
    response = requests.get(CRTS_CGI_URL, params=params, timeout=VETTING_HTTP_TIMEOUT)
    response.raise_for_status()
    text = getattr(response, "text", "") or ""
    if text.lstrip().startswith("MasterID,"):
        return _read_crts_csv_text(text)
    if _crts_cgi_response_is_empty(text):
        return pd.DataFrame()

    csv_url = _extract_crts_csv_url(text, getattr(response, "url", CRTS_CGI_BASE_URL) or CRTS_CGI_BASE_URL)
    if not csv_url:
        return pd.DataFrame()

    csv_response = requests.get(csv_url, timeout=VETTING_HTTP_TIMEOUT)
    csv_response.raise_for_status()
    csv_text = getattr(csv_response, "text", "") or ""
    if _crts_cgi_response_is_empty(csv_text):
        return pd.DataFrame()
    return _read_crts_csv_text(csv_text)


def _normalize_crts_cgi_lightcurve(
    raw: pd.DataFrame,
    *,
    ra: float,
    dec: float,
    radius_arcsec: float,
    catalog: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["crts_id", "mag", "mag_err", "ra", "dec", "mjd", "blend", "catalog", "sep_arcsec"])

    required = {"Mag", "MJD"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"CRTS CSV missing required columns: {', '.join(sorted(missing))}")

    lc = raw.rename(
        columns={
            "MasterID": "crts_id",
            "Mag": "mag",
            "Magerr": "mag_err",
            "RA": "ra",
            "Dec": "dec",
            "MJD": "mjd",
            "Blend": "blend",
        }
    ).copy()
    if "crts_id" not in lc.columns:
        lc["crts_id"] = f"{catalog}:{float(ra):.7f}:{float(dec):.7f}"
    if "mag_err" not in lc.columns:
        lc["mag_err"] = np.nan
    if "ra" not in lc.columns:
        lc["ra"] = float(ra)
    if "dec" not in lc.columns:
        lc["dec"] = float(dec)
    if "blend" not in lc.columns:
        lc["blend"] = 0
    for col in ("mag", "mag_err", "ra", "dec", "mjd", "blend"):
        lc[col] = pd.to_numeric(lc[col], errors="coerce")
    lc = lc.dropna(subset=["mag", "ra", "dec", "mjd"])
    lc["crts_id"] = lc["crts_id"].astype(str)
    if lc.empty:
        return pd.DataFrame(columns=["crts_id", "mag", "mag_err", "ra", "dec", "mjd", "blend", "catalog", "sep_arcsec"])

    target = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    grouped_pos = lc.groupby("crts_id", dropna=False)[["ra", "dec"]].median().dropna()
    if grouped_pos.empty:
        return pd.DataFrame(columns=["crts_id", "mag", "mag_err", "ra", "dec", "mjd", "blend", "catalog", "sep_arcsec"])
    group_coords = SkyCoord(
        ra=grouped_pos["ra"].to_numpy(dtype=float) * u.deg,
        dec=grouped_pos["dec"].to_numpy(dtype=float) * u.deg,
    )
    group_sep = pd.Series(group_coords.separation(target).arcsec, index=grouped_pos.index)
    closest_id = str(group_sep.sort_values().index[0])
    closest_sep = float(group_sep.loc[closest_id])
    if np.isfinite(closest_sep) and closest_sep > float(radius_arcsec):
        return pd.DataFrame(columns=["crts_id", "mag", "mag_err", "ra", "dec", "mjd", "blend", "catalog", "sep_arcsec"])

    lc = lc[lc["crts_id"] == closest_id].copy()
    lc["catalog"] = catalog
    lc["sep_arcsec"] = closest_sep
    cols = ["crts_id", "mag", "mag_err", "ra", "dec", "mjd", "blend", "catalog", "sep_arcsec"]
    return lc[cols].sort_values("mjd").reset_index(drop=True)


def _query_crts_cgi_lightcurve(ra: float, dec: float, radius_arcsec: float) -> pd.DataFrame:
    for catalog in ("photcat", "orphancat"):
        raw = _fetch_crts_cgi_catalog(ra, dec, radius_arcsec, catalog)
        lc = _normalize_crts_cgi_lightcurve(
            raw,
            ra=ra,
            dec=dec,
            radius_arcsec=radius_arcsec,
            catalog=catalog,
        )
        if not lc.empty:
            return lc
    return pd.DataFrame(columns=["crts_id", "mag", "mag_err", "ra", "dec", "mjd", "blend", "catalog", "sep_arcsec"])


def fetch_crts_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch CRTS light curves using the CRTS DR2 CGI endpoint.

    Adds columns: crts_lc_n_points.
    If *output_dir* is set, saves per-candidate parquet files as
    ``crts_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["crts_lc_n_points"] = 0

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"CRTS LCs: fetching {n_valid} light curves via CGI")

    valid_idx = df.index[valid].tolist()
    summary_cols = ["crts_lc_n_points"]
    valid_idx, _, cached_matched = _apply_external_lc_cache_hits(
        df,
        valid_idx,
        output_dir=output_dir,
        refresh_cache=refresh_cache,
        module="CRTS LCs",
        file_prefix="crts_lc",
        summary_cols=summary_cols,
        match_col="crts_lc_n_points",
        cache_key_func=lambda idx: _coord_lookup_cache_key(df, idx, CRTS_MATCH_RADIUS_ARCSEC, "crts"),
        summarize_func=lambda lc: _summarize_count_lc(lc, "crts_lc_n_points"),
    )
    if not valid_idx:
        print(f"CRTS LCs: {cached_matched}/{n_valid} with data")
        return df

    matched = cached_matched
    status_rows: list[dict] = []
    failures: list[str] = []
    for idx in tqdm(valid_idx, desc="CRTS LCs"):
        cache_key = _coord_lookup_cache_key(df, idx, CRTS_MATCH_RADIUS_ARCSEC, "crts")
        try:
            lc_df = _query_crts_cgi_lightcurve(
                float(df.loc[idx, "ra"]),
                float(df.loc[idx, "dec"]),
                CRTS_MATCH_RADIUS_ARCSEC,
            )
        except Exception as exc:
            short = _short_error(exc)
            failures.append(f"{_candidate_cache_id(df, idx)}: {short}")
            row = _external_lc_status_row(
                df,
                idx,
                module="CRTS LCs",
                cache_key=cache_key,
                summary={"crts_lc_n_points": 0, "error_message": short},
                status="error",
            )
            if row is not None:
                status_rows.append(row)
            continue

        n_points = int(len(lc_df))
        df.loc[idx, "crts_lc_n_points"] = n_points
        if output_dir and n_points > 0:
            _write_external_lc_file(output_dir, "crts_lc", df, idx, lc_df)
        if n_points > 0:
            matched += 1

        row = _external_lc_status_row(
            df,
            idx,
            module="CRTS LCs",
            cache_key=cache_key,
            summary={"crts_lc_n_points": n_points},
            status="fetched" if n_points > 0 else "no_data",
        )
        if row is not None:
            status_rows.append(row)

    _write_external_lc_status(output_dir, status_rows)
    _raise_lookup_failures("CRTS LCs", failures, n_valid)
    print(f"CRTS LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# EXTERNAL LC ORCHESTRATOR
# =============================================================================



def fetch_external_lcs(
    df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    run_atlas: bool = False,
    run_ztf: bool = True,
    run_gaia_epoch: bool = True,
    run_tess: bool = True,
    run_neowise: bool = True,
    run_kepler: bool = True,
    run_aavso: bool = True,
    run_ogle: bool = True,
    run_stripe82: bool = True,
    run_allwise_mep: bool = True,
    run_vvvx_virac: bool = True,
    run_ps1: bool = True,
    run_crts: bool = True,
    atlas_token: str | None = None,
    workers: int = 4,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Orchestrator for fetching external light curves from all sources.

    Calls each fetch function in sequence with *output_dir*.
    Supports checkpoint resume (same pattern as ``vet_candidates``).
    NEOWISE full light curves are included as cached precursor data for
    downstream multi-survey feature extraction.
    """
    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            _safe_print(msg)

    # Normalise coordinate column names
    if "ra" not in df.columns and "ra_deg" in df.columns:
        df = df.rename(columns={"ra_deg": "ra"})
    if "dec" not in df.columns and "dec_deg" in df.columns:
        df = df.rename(columns={"dec_deg": "dec"})

    # Resume from checkpoint if available
    _resumed = False
    if checkpoint_path and checkpoint_path.exists():
        try:
            df = pd.read_parquet(checkpoint_path)
            _resumed = True
            _emit(f"Resumed external LCs from checkpoint: {checkpoint_path}")
        except Exception:
            pass

    total_start = time.perf_counter()
    _emit(f"EXTERNAL LIGHT CURVES: {len(df)} candidates")

    _MODULE_MARKERS = {
        "ATLAS LCs": "atlas_has_phot",
        "ZTF LCs": "ztf_lc_n_det",
        "Gaia epoch LCs": "gaia_epoch_lc_n_g",
        "TESS LCs": "tess_n_sectors",
        "NEOWISE LCs": "neowise_n_epochs",
        "Kepler LCs": "kepler_n_quarters",
        "AAVSO LCs": "aavso_lc_n_points",
        "OGLE LCs": "ogle_lc_n_points",
        "Stripe 82 LCs": "stripe82_lc_n_points",
        "AllWISE MEP LCs": "allwise_mep_n_epochs",
        "VVVX/VIRAC2 LCs": "vvvx_virac_n_epochs",
        "Pan-STARRS LCs": "ps1_lc_n_points",
        "CRTS LCs": "crts_lc_n_points",
    }

    def _module_done(name):
        if not _resumed:
            return False
        col = _MODULE_MARKERS.get(name)
        if col is None or col not in df.columns:
            return False
        status_df = _read_external_lc_status(output_dir)
        if not status_df.empty and {"module", "status"}.issubset(status_df.columns):
            failed = (
                (status_df["module"].astype(str) == name)
                & (status_df["status"].astype(str).isin({"error", "failed"}))
            )
            if bool(failed.any()):
                return False
        s = df[col]
        return s.notna().any() and (s != 0).any()

    failures: list[str] = []

    def _run_module(name, func, **kwargs):
        nonlocal df
        if _module_done(name):
            _emit(f"{name} skipped (already in checkpoint)")
            return
        t0 = time.perf_counter()
        fallback_stream = getattr(sys, "__stderr__", None) or getattr(sys, "stderr", None)
        stdout_stream = _SafeOutputStream(sys.stdout, fallback_stream)
        stderr_stream = _SafeOutputStream(sys.stderr, fallback_stream)
        try:
            with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
                df = func(df, **kwargs)
        except Exception as exc:
            msg = f"{name} failed: {_short_error(exc)}"
            failures.append(msg)
            _emit(msg)
        else:
            _emit(f"{name} completed in {time.perf_counter() - t0:.1f}s")
        finally:
            if checkpoint_path:
                df.to_parquet(checkpoint_path, index=False)

    if run_atlas:
        _run_module("ATLAS LCs", query_atlas_forced_phot, token=atlas_token, output_dir=output_dir, refresh_cache=refresh_cache)

    if run_ztf:
        _run_module("ZTF LCs", fetch_ztf_lightcurves, output_dir=output_dir, workers=workers, refresh_cache=refresh_cache)

    if run_gaia_epoch:
        _run_module("Gaia epoch LCs", fetch_gaia_epoch_lcs, output_dir=output_dir, refresh_cache=refresh_cache)

    if run_tess:
        _run_module("TESS LCs", fetch_tess_lightcurves, output_dir=output_dir, workers=min(workers, 2), refresh_cache=refresh_cache)

    if run_neowise:
        _run_module("NEOWISE LCs", query_neowise_lightcurves, output_dir=output_dir, workers=workers, refresh_cache=refresh_cache)

    if run_kepler:
        _run_module("Kepler LCs", fetch_kepler_k2_lightcurves, output_dir=output_dir, workers=min(workers, 2), refresh_cache=refresh_cache)

    if run_aavso:
        _run_module("AAVSO LCs", fetch_aavso_lightcurves, output_dir=output_dir, workers=workers, refresh_cache=refresh_cache)

    if run_ogle:
        _run_module("OGLE LCs", fetch_ogle_lightcurves, output_dir=output_dir, workers=workers, refresh_cache=refresh_cache)

    if run_stripe82:
        _run_module("Stripe 82 LCs", fetch_stripe82_lightcurves, output_dir=output_dir, refresh_cache=refresh_cache)

    if run_allwise_mep:
        _run_module("AllWISE MEP LCs", fetch_allwise_mep_lightcurves, output_dir=output_dir, workers=workers, refresh_cache=refresh_cache)

    if run_vvvx_virac:
        _run_module("VVVX/VIRAC2 LCs", fetch_vvvx_virac_lightcurves, output_dir=output_dir, workers=min(workers, 2), refresh_cache=refresh_cache)

    if run_ps1:
        _run_module("Pan-STARRS LCs", fetch_panstarrs_lightcurves, output_dir=output_dir, workers=workers, refresh_cache=refresh_cache)

    if run_crts:
        _run_module("CRTS LCs", fetch_crts_lightcurves, output_dir=output_dir, refresh_cache=refresh_cache)

    elapsed = time.perf_counter() - total_start
    if failures:
        df.attrs["external_lc_failures"] = list(failures)
        _emit(f"External LCs completed with {len(failures)} module failure(s) in {elapsed:.1f}s")
    else:
        _emit(f"External LCs completed in {elapsed:.1f}s")
    return df


# =============================================================================
# ORCHESTRATION
# =============================================================================


def vet_candidates(
    df: pd.DataFrame,
    *,
    run_simbad: bool = True,
    run_gaia_var: bool = True,
    run_asassn_var: bool = True,
    run_microlens: bool = True,
    run_ztf_var: bool = True,
    run_tns: bool = True,
    run_gaia_eb: bool = True,
    run_alerce: bool = True,
    run_atlas: bool = True,
    run_gaia_epoch: bool = True,
    run_erosita: bool = True,
    run_chandra_csc: bool = True,
    run_pm_check: bool = True,
    run_neowise_lc: bool = False,
    simbad_radius_arcsec: float = SIMBAD_RADIUS_ARCSEC,
    asassn_radius_arcsec: float = ASASSN_VAR_RADIUS_ARCSEC,
    microlens_radius_arcsec: float = OGLE_MICROLENS_RADIUS_ARCSEC,
    ztf_var_radius_arcsec: float = ZTF_VAR_RADIUS_ARCSEC,
    tns_radius_arcsec: float = TNS_RADIUS_ARCSEC,
    alerce_radius_arcsec: float = ALERCE_RADIUS_ARCSEC,
    erosita_radius_arcsec: float = EROSITA_RADIUS_ARCSEC,
    chandra_radius_arcsec: float = CHANDRA_CSC_RADIUS_ARCSEC,
    gaia_var_chunk_size: int = GAIA_VAR_CHUNK_SIZE,
    atlas_token: str | None = None,
    atlas_output_dir: Path | None = None,
    tns_api_key: str | None = None,
    alerce_workers: int = 8,
    neowise_output_dir: Path | None = None,
    neowise_workers: int = 4,
    checkpoint_path: Path | None = None,
    method: Literal["tap", "xmatch"] = "xmatch",
    skip_existing: bool = False,
    cache_dir: Path | str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Run all vetting queries on a candidate DataFrame.

    Parameters
    ----------
    df : DataFrame with at minimum 'ra', 'dec' columns.
         'gaia_id' column needed for Gaia variability queries.
    run_simbad : query SIMBAD for object type, bibliography
    run_gaia_var : query Gaia DR3 variability tables
    run_asassn_var : crossmatch ASAS-SN variable star catalog
    run_microlens : crossmatch known microlensing event catalogs
    run_ztf_var : crossmatch ZTF periodic variables (Chen+ 2020)
    run_tns : crossmatch Transient Name Server
    run_gaia_eb : query Gaia DR3 eclipsing binary parameters (ECL sources only)
    run_alerce : query ALeRCE ZTF broker
    run_atlas : query ATLAS forced photometry (requires token)
    run_gaia_epoch : check Gaia epoch photometry availability
    run_erosita : crossmatch eROSITA X-ray catalog
    run_chandra_csc : crossmatch Chandra CSC 2.1 source catalog
    run_pm_check : proper motion consistency with clusters
    run_neowise_lc : fetch full NEOWISE light curves (opt-in)
    checkpoint_path : if set, save intermediate results after each module
    method : Propagated to non-SIMBAD crossmatch functions such as ZTF vars,
        TNS, eROSITA, and Chandra CSC. SIMBAD always uses CDS XMatch. For ASAS-SN
        variables, ``xmatch`` uses the bundled local CSV.
    skip_existing : Skip modules whose marker columns already contain data in
        the input table. Checkpoints are always skipped this way.
    cache_dir : Directory for persistent vetting caches.
    refresh_cache : Force cache-backed modules to requery remote services.

    Returns
    -------
    DataFrame with vetting columns added.
    """
    # Normalise coordinate column names (pipeline uses ra_deg/dec_deg).
    if "ra" not in df.columns and "ra_deg" in df.columns:
        df = df.rename(columns={"ra_deg": "ra"})
    if "dec" not in df.columns and "dec_deg" in df.columns:
        df = df.rename(columns={"dec_deg": "dec"})
    missing_coord_cols = [col for col in ("ra", "dec") if col not in df.columns]
    if missing_coord_cols:
        print(
            "Warning: vetting input is missing coordinate column(s): "
            f"{', '.join(missing_coord_cols)}; coordinate crossmatches will be skipped"
        )
        for col in missing_coord_cols:
            df[col] = np.nan

    # Resume from checkpoint if available.
    _resumed = False
    if checkpoint_path and checkpoint_path.exists():
        try:
            df = pd.read_parquet(checkpoint_path)
            _resumed = True
            print(f"Resumed from checkpoint: {checkpoint_path}")
        except Exception:
            pass

    total_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"POST-REVIEW VETTING: {len(df)} candidates")
    print(f"{'='*60}\n")

    # Map each module to a marker column — if that column has data, skip.
    _MODULE_MARKERS = {
        "SIMBAD": "simbad_main_id",
        "Gaia variability": "gaia_var_flag",
        "Gaia epoch photometry": "gaia_epoch_available",
        "ASAS-SN variables": "asassn_var_name",
        "Microlensing catalogs": "microlens_match",
        "ZTF variables": "ztf_var_type",
        "TNS": "tns_name",
        "Gaia EB params": "gaia_eb_period",
        "ALeRCE": "alerce_oid",
        "eROSITA": "erosita_det",
        "Chandra CSC": "chandra_det",
        "ATLAS forced phot": "atlas_has_phot",
        "PM consistency": "pm_cluster_offset_sigma",
        "NEOWISE LCs": "neowise_n_epochs",
    }

    def _module_done(name):
        """Check if a module's marker column already has data (from checkpoint)."""
        if not (_resumed or skip_existing):
            return False
        col = _MODULE_MARKERS.get(name)
        if col is None or col not in df.columns:
            return False
        s = df[col]
        if s.dtype == object:
            return (s.fillna("").astype(str).str.strip() != "").any()
        return s.notna().any()

    def _run_module(name, func, **kwargs):
        nonlocal df
        if _module_done(name):
            print(f"  {name} — skipped (already in checkpoint)\n")
            return
        t0 = time.perf_counter()
        df = func(df, **kwargs)
        print(f"  {name} completed in {time.perf_counter() - t0:.1f}s\n")
        if checkpoint_path:
            df.to_parquet(checkpoint_path, index=False)

    if run_simbad:
        _run_module(
            "SIMBAD",
            query_simbad_batch,
            radius_arcsec=simbad_radius_arcsec,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )

    if run_gaia_var:
        _run_module(
            "Gaia variability",
            query_gaia_variability,
            chunk_size=gaia_var_chunk_size,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )

    if run_gaia_epoch:
        _run_module(
            "Gaia epoch photometry",
            query_gaia_epoch_photometry,
            chunk_size=gaia_var_chunk_size,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )

    if run_asassn_var:
        # ASAS-SN II/366 is not on CDS XMatch; use local CSV when method='xmatch'
        _asassn_method = "local" if method == "xmatch" else "tap"
        _run_module("ASAS-SN variables", crossmatch_asassn_variables,
                    radius_arcsec=asassn_radius_arcsec, method=_asassn_method)

    if run_microlens:
        _run_module("Microlensing catalogs", crossmatch_microlensing_catalogs,
                    radius_arcsec=microlens_radius_arcsec, method=method)

    if run_ztf_var:
        _run_module("ZTF variables", crossmatch_ztf_variables, radius_arcsec=ztf_var_radius_arcsec, method=method)

    if run_tns:
        _run_module("TNS", crossmatch_tns, radius_arcsec=tns_radius_arcsec, tns_api_key=tns_api_key)

    if run_gaia_eb:
        _run_module(
            "Gaia EB params",
            query_gaia_eb_params,
            chunk_size=gaia_var_chunk_size,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )

    if run_alerce:
        _run_module(
            "ALeRCE",
            query_alerce,
            radius_arcsec=alerce_radius_arcsec,
            workers=alerce_workers,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )

    if run_erosita:
        # eROSITA: prefer local FITS when method='xmatch' (if file exists)
        if method == "xmatch" and EROSITA_LOCAL_FITS.exists():
            _erosita_method = "local"
        elif method == "xmatch":
            print(f"eROSITA: local FITS not found at {EROSITA_LOCAL_FITS}; falling back to CDS XMatch")
            _erosita_method = method
        else:
            _erosita_method = method
        _run_module("eROSITA", crossmatch_erosita, radius_arcsec=erosita_radius_arcsec, method=_erosita_method)

    if run_chandra_csc:
        _run_module("Chandra CSC", crossmatch_chandra_csc, radius_arcsec=chandra_radius_arcsec, method=method)

    if run_erosita or run_chandra_csc:
        df = _sync_xray_aggregate_fields(df)
        if checkpoint_path:
            df.to_parquet(checkpoint_path, index=False)

    if run_atlas:
        _run_module(
            "ATLAS forced phot",
            query_atlas_forced_phot,
            token=atlas_token,
            output_dir=atlas_output_dir,
            refresh_cache=refresh_cache,
        )

    if run_pm_check:
        _run_module("PM consistency", check_pm_consistency)

    if run_neowise_lc:
        _run_module("NEOWISE LCs", query_neowise_lightcurves,
                    output_dir=neowise_output_dir, workers=neowise_workers,
                    refresh_cache=refresh_cache)

    # Summary
    _print_vetting_summary(df, total_start)
    return df


def _print_vetting_summary(df: pd.DataFrame, total_start: float) -> None:
    """Print comprehensive vetting summary."""
    print(f"\n{'='*60}")
    print("VETTING SUMMARY")
    print(f"{'='*60}")

    def _truthy_series(series: pd.Series) -> pd.Series:
        truthy = {"1", "true", "t", "yes", "y", "variable"}
        values = series.fillna(False)
        if values.dtype == bool:
            return values
        if pd.api.types.is_numeric_dtype(values):
            return values.astype(float) != 0.0
        return values.astype(str).str.strip().str.lower().isin(truthy)

    if "simbad_main_id" in df.columns:
        n = (df["simbad_main_id"] != "").sum()
        print(f"  SIMBAD matches:         {n}/{len(df)}")
        if n > 0:
            print(f"  Median SIMBAD refs:     {df.loc[df['simbad_main_id'] != '', 'simbad_nbref'].median():.0f}")

    if "gaia_var_flag" in df.columns:
        print(f"  Gaia variable flag:     {_truthy_series(df['gaia_var_flag']).sum()}/{len(df)}")
    if "gaia_var_class" in df.columns:
        n = (df["gaia_var_class"] != "").sum()
        print(f"  Gaia classified:        {n}/{len(df)}")
        if n > 0:
            for cls, cnt in df.loc[df["gaia_var_class"] != "", "gaia_var_class"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "gaia_epoch_available" in df.columns:
        print(f"  Gaia epoch available:   {df['gaia_epoch_available'].sum()}/{len(df)}")

    if "asassn_var_type" in df.columns:
        n = (df["asassn_var_type"] != "").sum()
        print(f"  ASAS-SN var matches:    {n}/{len(df)}")

    if "microlens_match" in df.columns:
        n = df["microlens_match"].fillna(False).astype(bool).sum()
        print(f"  Microlens matches:      {n}/{len(df)}")
        if n > 0 and "microlens_catalog" in df.columns:
            for cls, cnt in df.loc[df["microlens_match"].fillna(False).astype(bool), "microlens_catalog"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "ztf_var_type" in df.columns:
        n = (df["ztf_var_type"] != "").sum()
        print(f"  ZTF var matches:        {n}/{len(df)}")
        if n > 0:
            for cls, cnt in df.loc[df["ztf_var_type"] != "", "ztf_var_type"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "tns_name" in df.columns:
        n = (df["tns_name"] != "").sum()
        print(f"  TNS transients:         {n}/{len(df)}")
        if n > 0:
            for cls, cnt in df.loc[df["tns_type"] != "", "tns_type"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "gaia_eb_period" in df.columns:
        n = df["gaia_eb_period"].notna().sum()
        print(f"  Gaia EB params:         {n}/{len(df)}")

    if "alerce_oid" in df.columns:
        n = (df["alerce_oid"] != "").sum()
        print(f"  ALeRCE matches:         {n}/{len(df)}")
        if n > 0:
            lc_cls = df.loc[df["alerce_lc_class"] != "", "alerce_lc_class"].value_counts().head(5)
            if len(lc_cls) > 0:
                print(f"  ALeRCE LC classes:")
                for cls, cnt in lc_cls.items():
                    print(f"    {cls}: {cnt}")

    if "erosita_det" in df.columns:
        n = df["erosita_det"].fillna(False).astype(bool).sum()
        print(f"  eROSITA X-ray det:      {n}/{len(df)}")
    if "chandra_det" in df.columns:
        n = df["chandra_det"].fillna(False).astype(bool).sum()
        print(f"  Chandra CSC det:        {n}/{len(df)}")
    if "xray_det" in df.columns:
        n = df["xray_det"].fillna(False).astype(bool).sum()
        print(f"  Structured X-ray det:   {n}/{len(df)}")

    if "atlas_has_phot" in df.columns:
        n = df["atlas_has_phot"].sum()
        print(f"  ATLAS photometry:       {n}/{len(df)}")

    if "pm_cluster_offset_sigma" in df.columns:
        n = df["pm_cluster_offset_sigma"].notna().sum()
        if n > 0:
            outliers = (df["pm_cluster_offset_sigma"] > 3).sum()
            print(f"  PM consistency:         {n} checked, {outliers} outliers (>3σ)")

    if "neowise_n_epochs" in df.columns:
        n = (df["neowise_n_epochs"] > 0).sum()
        print(f"  NEOWISE LCs:            {n}/{len(df)}")

    # Flag "likely known" vs "potentially new"
    known_mask = pd.Series(False, index=df.index)
    
    # We only want to flag true for variables with a catalog type/class, not
    # generic variable-flag evidence.
    if "gaia_var_class" in df.columns:
        known_mask |= df["gaia_var_class"] != ""
    if "asassn_var_type" in df.columns:
        known_mask |= df["asassn_var_type"] != ""
    if "microlens_match" in df.columns:
        known_mask |= df["microlens_match"].fillna(False).astype(bool)
    if "ztf_var_type" in df.columns:
        known_mask |= df["ztf_var_type"] != ""
    if "tns_name" in df.columns:
        known_mask |= df["tns_name"] != ""
    if "alerce_lc_class" in df.columns:
        known_mask |= df["alerce_lc_class"] != ""
    if "vsx_class" in df.columns:
        known_mask |= df["vsx_class"].fillna("").astype(str).str.strip() != ""

    if "simbad_otype" in df.columns:
        def is_var_otype(x):
            s = str(x).strip()
            if not s: return False
            if 'V*' in s: return True
            matches = {'EB*', 'YSO', 'SN', 'Nova', 'Catac', 'RR*', 'Cepheid', 'Mira', 'BYDra', 'RSCVn', 'Symbiotic', 'ELL', 'Blazar', 'QSO', 'AGN'}
            s_low = s.lower()
            return any(m.lower() in s_low for m in matches)
        known_mask |= df["simbad_otype"].apply(is_var_otype)
    df["vetting_likely_known"] = known_mask

    n_known = known_mask.sum()
    n_new = len(df) - n_known
    print(f"\n  Likely known:           {n_known}")
    print(f"  Potentially new:        {n_new}")
    print(f"\n  Total time: {time.perf_counter() - total_start:.1f}s")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


VETTING_ONLY_MODULES = {
    "simbad": "run_simbad",
    "gaia-var": "run_gaia_var",
    "gaia-epoch": "run_gaia_epoch",
    "asassn-var": "run_asassn_var",
    "microlens": "run_microlens",
    "ztf-var": "run_ztf_var",
    "tns": "run_tns",
    "gaia-eb": "run_gaia_eb",
    "alerce": "run_alerce",
    "erosita": "run_erosita",
    "chandra-csc": "run_chandra_csc",
    "pm-check": "run_pm_check",
    "atlas": "run_atlas",
    "neowise-lc": "run_neowise_lc",
}


def _parse_only_modules(value: str | None) -> set[str] | None:
    if value is None:
        return None
    selected = {part.strip().lower() for part in value.replace(",", " ").split() if part.strip()}
    unknown = selected - set(VETTING_ONLY_MODULES)
    if unknown:
        choices = ", ".join(sorted(VETTING_ONLY_MODULES))
        raise ValueError(f"unknown --only module(s): {', '.join(sorted(unknown))}. Choices: {choices}")
    return selected


def main():
    """CLI for standalone vetting."""


    parser = argparse.ArgumentParser(description="Post-review vetting of MALCA candidates")
    g_io = parser.add_argument_group("Input / output")
    g_radii = parser.add_argument_group("Search radii")
    g_skip = parser.add_argument_group("Skip toggles")
    g_workers = parser.add_argument_group("Workers & options")
    g_general = parser.add_argument_group("General")

    g_io.add_argument("input", type=Path, help="Input Parquet with candidates (needs ra, dec columns)")
    g_io.add_argument("-o", "--output", type=Path, default=None, help="Output Parquet path (default: <input>_vetted.parquet)")
    g_io.add_argument("--min-score", type=float, default=None, help="Only vet candidates with interest_score >= this value")
    g_io.add_argument("--all-candidates", action="store_true", help="Vet all input rows instead of only failed_any=False passers")
    g_radii.add_argument("--simbad-radius", type=float, default=SIMBAD_RADIUS_ARCSEC, help=f"SIMBAD search radius in arcsec (default: {SIMBAD_RADIUS_ARCSEC})")
    g_radii.add_argument("--asassn-radius", type=float, default=ASASSN_VAR_RADIUS_ARCSEC, help=f"ASAS-SN crossmatch radius in arcsec (default: {ASASSN_VAR_RADIUS_ARCSEC})")
    g_radii.add_argument("--microlens-radius", type=float, default=OGLE_MICROLENS_RADIUS_ARCSEC, help=f"Microlensing catalog crossmatch radius in arcsec (default: {OGLE_MICROLENS_RADIUS_ARCSEC})")
    g_radii.add_argument("--alerce-radius", type=float, default=ALERCE_RADIUS_ARCSEC, help=f"ALeRCE search radius in arcsec (default: {ALERCE_RADIUS_ARCSEC})")
    g_radii.add_argument("--erosita-radius", type=float, default=EROSITA_RADIUS_ARCSEC, help=f"eROSITA search radius in arcsec (default: {EROSITA_RADIUS_ARCSEC})")
    g_radii.add_argument("--chandra-radius", type=float, default=CHANDRA_CSC_RADIUS_ARCSEC, help=f"Chandra CSC search radius in arcsec (default: {CHANDRA_CSC_RADIUS_ARCSEC})")
    g_radii.add_argument("--ztf-var-radius", type=float, default=ZTF_VAR_RADIUS_ARCSEC, help=f"ZTF variable crossmatch radius in arcsec (default: {ZTF_VAR_RADIUS_ARCSEC})")
    g_radii.add_argument("--tns-radius", type=float, default=TNS_RADIUS_ARCSEC, help=f"TNS crossmatch radius in arcsec (default: {TNS_RADIUS_ARCSEC})")
    g_skip.add_argument("--no-simbad", action="store_true", help="Skip SIMBAD query")
    g_skip.add_argument("--no-gaia-var", action="store_true", help="Skip Gaia DR3 variability query")
    g_skip.add_argument("--no-gaia-epoch", action="store_true", help="Skip Gaia DR3 epoch photometry check")
    g_skip.add_argument("--no-asassn-var", action="store_true", help="Skip ASAS-SN variable catalog crossmatch")
    g_skip.add_argument("--no-microlens", action="store_true", help="Skip microlensing event catalog crossmatch")
    g_skip.add_argument("--no-ztf-var", action="store_true", help="Skip ZTF periodic variables crossmatch")
    g_skip.add_argument("--no-tns", action="store_true", help="Skip TNS transient crossmatch")
    g_skip.add_argument("--no-gaia-eb", action="store_true", help="Skip Gaia DR3 eclipsing binary parameters")
    g_skip.add_argument("--no-alerce", action="store_true", help="Skip ALeRCE ZTF query")
    g_skip.add_argument("--no-erosita", action="store_true", help="Skip eROSITA X-ray crossmatch")
    g_skip.add_argument("--no-chandra-csc", action="store_true", help="Skip Chandra CSC X-ray crossmatch")
    g_skip.add_argument("--no-pm-check", action="store_true", help="Skip proper motion consistency check")
    g_skip.add_argument("--no-atlas", action="store_true", help="Skip ATLAS forced photometry (default: enabled)")
    g_workers.add_argument("--alerce-workers", type=int, default=8, help="Parallel workers for ALeRCE queries (default: 8)")
    g_workers.add_argument("--atlas-token", type=str, default=None, help="ATLAS forced photometry API token (or set MALCA_ATLAS_TOKEN env var)")
    g_workers.add_argument("--tns-api-key", type=str, default=None, help="TNS API key (ignored; TNS uses local catalog)")
    g_workers.add_argument("--neowise-lc", dest="neowise_lc", action="store_true", help="Fetch full NEOWISE light curves (default: disabled)")
    g_workers.add_argument("--no-neowise-lc", dest="neowise_lc", action="store_false", help="Skip full NEOWISE light curves")
    g_workers.add_argument("--neowise-output-dir", type=Path, default=None, help="Directory to save individual NEOWISE LCs")
    g_workers.add_argument("--neowise-workers", type=int, default=4, help="Parallel workers for NEOWISE queries")
    g_workers.add_argument(
        "--method",
        choices=["tap", "xmatch"],
        default="xmatch",
        help="Catalog crossmatch mode for supported modules; xmatch uses local ASAS-SN variables CSV (default: xmatch)",
    )
    g_general.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only selected modules, comma-separated. Choices: " + ", ".join(sorted(VETTING_ONLY_MODULES)),
    )
    g_general.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Persistent cache directory for slow vetting modules (default: {DEFAULT_CACHE_DIR})",
    )
    g_general.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force cache-backed modules to requery remote services and update cache",
    )
    g_general.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path (default: <input>_vetting_CHECKPOINT.parquet)")
    g_general.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint saving/resume")
    g_general.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip modules whose marker columns already contain data in the input table",
    )

    parser.set_defaults(neowise_lc=False)

    args = parser.parse_args()
    try:
        only_modules = _parse_only_modules(args.only)
    except ValueError as exc:
        parser.error(str(exc))

    # Load input
    path = args.input.expanduser()
    df = read_feature_table(path)

    print(f"Loaded {len(df)} candidates from {path}")

    # Default checkpoint: <input>_vetting_CHECKPOINT.parquet
    if args.no_checkpoint:
        _ckpt_path = None
    elif args.checkpoint:
        _ckpt_path = args.checkpoint
    else:
        _ckpt_path = path.with_name(path.stem + "_vetting_CHECKPOINT.parquet")

    # Filter by score if requested
    if args.min_score is not None and "interest_score" in df.columns:
        before = len(df)
        df = df[df["interest_score"] >= args.min_score].copy()
        print(f"Filtered to {len(df)} candidates with score >= {args.min_score} (from {before})")
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)

    if only_modules is not None:
        print("Running only vetting modules: " + ", ".join(sorted(only_modules)))

    def _enabled(module_name: str, no_flag: bool) -> bool:
        if only_modules is not None:
            return module_name in only_modules
        return not no_flag

    # Run vetting
    df = vet_candidates(
        df,
        run_simbad=_enabled("simbad", args.no_simbad),
        run_gaia_var=_enabled("gaia-var", args.no_gaia_var),
        run_gaia_epoch=_enabled("gaia-epoch", args.no_gaia_epoch),
        run_asassn_var=_enabled("asassn-var", args.no_asassn_var),
        run_microlens=_enabled("microlens", args.no_microlens),
        run_ztf_var=_enabled("ztf-var", args.no_ztf_var),
        run_tns=_enabled("tns", args.no_tns),
        run_gaia_eb=_enabled("gaia-eb", args.no_gaia_eb),
        run_alerce=_enabled("alerce", args.no_alerce),
        run_erosita=_enabled("erosita", args.no_erosita),
        run_chandra_csc=_enabled("chandra-csc", args.no_chandra_csc),
        run_atlas=_enabled("atlas", args.no_atlas),
        run_pm_check=_enabled("pm-check", args.no_pm_check),
        run_neowise_lc=("neowise-lc" in only_modules) if only_modules is not None else args.neowise_lc,
        simbad_radius_arcsec=args.simbad_radius,
        asassn_radius_arcsec=args.asassn_radius,
        microlens_radius_arcsec=args.microlens_radius,
        ztf_var_radius_arcsec=args.ztf_var_radius,
        tns_radius_arcsec=args.tns_radius,
        alerce_radius_arcsec=args.alerce_radius,
        alerce_workers=args.alerce_workers,
        erosita_radius_arcsec=args.erosita_radius,
        chandra_radius_arcsec=args.chandra_radius,
        atlas_token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN"),
        tns_api_key=args.tns_api_key or os.environ.get("MALCA_TNS_API_KEY"),
        neowise_output_dir=args.neowise_output_dir,
        neowise_workers=args.neowise_workers,
        checkpoint_path=_ckpt_path,
        method=args.method,
        skip_existing=args.skip_existing,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )

    # Save output
    out_path = args.output or path.with_name(path.stem + "_vetted.parquet")
    write_feature_table(df, out_path)
    print(f"\nSaved vetted results to {out_path}")

    # Clean up checkpoint on successful completion.
    if _ckpt_path and _ckpt_path.exists():
        _ckpt_path.unlink()
        print(f"Checkpoint removed: {_ckpt_path}")


if __name__ == "__main__":
    main()
