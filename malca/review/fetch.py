"""Orchestrate light-curve fetch → analyse → import for the review widget.

Supports three search modes:
  - ASAS-SN ID  → download_lightcurve_by_id
  - Gaia DR3 ID → download_lightcurve_by_gaia_id
  - RA/Dec      → cone_search (returns catalog rows; caller picks target)

After downloading, we run the full `malca.stats.compute_stats` suite on the
downloaded SkyPatrol-format CSV and then hand the result to import_candidates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from malca.config import RUN_MAX_GAP_POINTS, BASELINE_FUNC
from malca.fetch import (
    cone_search,
    download_lightcurve_by_id,
    download_lightcurve_by_gaia_id,
)
from malca.review.stats_merge import merge_stats_summary_into_payload




def fetch_and_analyze_by_id(
    asas_sn_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
    backend: str = "skypatrol2",
    refresh_cache: bool = False,
    refresh_stats_cache: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Download LC by ASAS-SN ID, compute basic stats, return (1-row DF, lc_path)."""
    lc_path, catalog_info = download_lightcurve_by_id(asas_sn_id, backend=backend, refresh_cache=refresh_cache)
    df = _build_candidate_row(asas_sn_id, lc_path, catalog_info,
                              run_stats=run_stats, run_events=run_events,
                              refresh_stats_cache=refresh_stats_cache)
    return df, lc_path


def fetch_and_analyze_by_gaia_id(
    gaia_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
    backend: str = "skypatrol2",
    refresh_cache: bool = False,
    refresh_stats_cache: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Download LC by Gaia DR3 source_id, compute basic stats."""
    lc_path, catalog_info = download_lightcurve_by_gaia_id(gaia_id, backend=backend, refresh_cache=refresh_cache)
    candidate_id = str(catalog_info.get("asas_sn_id", f"gaia_{gaia_id}"))
    df = _build_candidate_row(candidate_id, lc_path, catalog_info,
                              run_stats=run_stats, run_events=run_events,
                              refresh_stats_cache=refresh_stats_cache)
    return df, lc_path


def fetch_cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    backend: str = "skypatrol2",
) -> pd.DataFrame:
    """Return catalog rows from a cone search (no LC download)."""
    return cone_search(ra, dec, radius_arcsec=radius_arcsec, backend=backend)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_candidate_row(
    candidate_id: str,
    lc_path: Path,
    catalog_info: dict,
    *,
    run_stats: bool = True,
    run_events: bool = False,
    refresh_stats_cache: bool = False,
) -> pd.DataFrame:
    """Create a single-row candidate DataFrame suitable for import_candidates."""
    row: dict = {
        "candidate_id": str(candidate_id),
        "lc_path": str(lc_path),
    }

    # Forward ALL catalog metadata into the candidate row
    if catalog_info:
        # Column name mapping: SkyPatrol → MALCA
        _CATALOG_TO_MALCA = {
            "plx": "parallax",
            "plx_d": "parallax_error",
            "pm_ra": "pmra",
            "pm_dec": "pmdec",
            "gaia_eff_temp": "teff_gspphot",
            "gaia_mag": "phot_g_mean_mag",
        }
        for key, val in catalog_info.items():
            if val is None:
                continue
            malca_key = _CATALOG_TO_MALCA.get(key, key)
            row[malca_key] = val

    # Alias ra/dec for downstream modules that expect these names
    if "ra_deg" in row:
        row["ra"] = row["ra_deg"]
    if "dec_deg" in row:
        row["dec"] = row["dec_deg"]
    # characterize looks for source_id (= Gaia DR3 source_id)
    if "gaia_id" in row and "source_id" not in row:
        row["source_id"] = row["gaia_id"]

    if run_stats:
        stats = _compute_stats_from_skypatrol_csv(lc_path, refresh_stats_cache=refresh_stats_cache)
        print(f"[fetch] Full stats computed: {len(stats)} keys")
        row.update(stats)

    if run_events:
        events = _compute_events_for_csv(lc_path)
        row.update(events)

    return pd.DataFrame([row])


def _stats_cache_path(lc_path: Path) -> Path | None:
    try:
        path = Path(lc_path).expanduser().resolve()
        stat = path.stat()
    except Exception:
        return None
    key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return path.parent / "stats_cache" / f"{path.stem}.{digest}.json"


def _read_stats_cache(cache_path: Path | None) -> dict | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _json_safe_stats(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_stats_cache(cache_path: Path | None, stats: dict) -> None:
    if cache_path is None or not stats:
        return
    try:
        payload = {str(k): _json_safe_stats(v) for k, v in stats.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
    except Exception:
        pass


def _compute_stats_from_skypatrol_csv(lc_path: Path, *, refresh_stats_cache: bool = False) -> dict:
    """Compute full compute_stats() suite from a SkyPatrol-format CSV."""
    cache_path = _stats_cache_path(lc_path)
    if not refresh_stats_cache:
        cached = _read_stats_cache(cache_path)
        if cached is not None:
            return cached

    try:
        from malca.stats import compute_stats

        candidate_id = Path(lc_path).stem
        parent = str(Path(lc_path).parent)
        _df, summary = compute_stats(candidate_id, parent, compute_ls=True)
        out: dict = {}
        merge_stats_summary_into_payload(out, summary)
        _write_stats_cache(cache_path, out)
        return out
    except Exception as e:
        print(f"[fetch] Warning: full stats computation failed: {e}")
        traceback.print_exc()
        return {}


def _compute_events_for_csv(lc_path: Path) -> dict:
    """Run process_lightcurve on a SkyPatrol CSV file and return key columns."""
    try:
        from malca.config import (
            TRIGGER_MODE, LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP,
            SIGNIFICANCE_THRESHOLD, P_POINTS, MAG_POINTS,
            RUN_MIN_POINTS, RUN_MAX_GAP_POINTS, BASELINE_FUNC,
        )
        from malca.stv.events import process_lightcurve

        result = process_lightcurve(
            str(lc_path),
            trigger_mode=TRIGGER_MODE,
            logbf_threshold_dip=LOGBF_THRESHOLD_DIP,
            logbf_threshold_jump=LOGBF_THRESHOLD_JUMP,
            significance_threshold=SIGNIFICANCE_THRESHOLD,
            p_points=P_POINTS,
            p_min_dip=None,
            p_max_dip=None,
            p_min_jump=None,
            p_max_jump=None,
            mag_points=MAG_POINTS,
            run_min_points=RUN_MIN_POINTS,
            max_gap_points=RUN_MAX_GAP_POINTS,
            run_max_gap_days=None,
            run_min_duration_days=0.0,
            baseline_tag=BASELINE_FUNC,
            compute_event_prob=True,
        )
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[fetch] Warning: process_lightcurve failed: {e}")
        return {}
