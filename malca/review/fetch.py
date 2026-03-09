"""Orchestrate light-curve fetch → analyse → import for the review widget.

Supports three search modes:
  - ASAS-SN ID  → download_lightcurve_by_id
  - Gaia DR3 ID → download_lightcurve_by_gaia_id
  - RA/Dec      → cone_search (returns catalog rows; caller picks target)

After downloading, we run the full `malca.stats.compute_stats` suite on the
downloaded SkyPatrol-format CSV and then hand the result to import_candidates.
"""
from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

from malca.config.config_pipeline import (
    TRIGGER_MODE, LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP,
    SIGNIFICANCE_THRESHOLD, P_POINTS, MAG_POINTS,
    RUN_MIN_POINTS, RUN_MAX_GAP_POINTS, BASELINE_FUNC,
)
from malca.events import process_lightcurve
from malca.fetch import (
    cone_search,
    download_lightcurve_by_id,
    download_lightcurve_by_gaia_id,
)
from malca.review.stats_merge import merge_stats_summary_into_payload
from malca.stats import compute_stats


P_MIN_DIP = None
P_MAX_DIP = None
P_MIN_JUMP = None
P_MAX_JUMP = None
MAX_GAP_POINTS = RUN_MAX_GAP_POINTS
RUN_MAX_GAP_DAYS = None
RUN_MIN_DURATION_DAYS = 0.0
BASELINE_TAG = BASELINE_FUNC








def fetch_and_analyze_by_id(
    asas_sn_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
    backend: str = "skypatrol2",
) -> tuple[pd.DataFrame, Path]:
    """Download LC by ASAS-SN ID, compute basic stats, return (1-row DF, lc_path)."""
    lc_path, catalog_info = download_lightcurve_by_id(asas_sn_id, backend=backend)
    df = _build_candidate_row(asas_sn_id, lc_path, catalog_info,
                              run_stats=run_stats, run_events=run_events)
    return df, lc_path


def fetch_and_analyze_by_gaia_id(
    gaia_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
    backend: str = "skypatrol2",
) -> tuple[pd.DataFrame, Path]:
    """Download LC by Gaia DR3 source_id, compute basic stats."""
    lc_path, catalog_info = download_lightcurve_by_gaia_id(gaia_id, backend=backend)
    candidate_id = str(catalog_info.get("asas_sn_id", f"gaia_{gaia_id}"))
    df = _build_candidate_row(candidate_id, lc_path, catalog_info,
                              run_stats=run_stats, run_events=run_events)
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
        stats = _compute_stats_from_skypatrol_csv(lc_path)
        print(f"[fetch] Full stats computed: {len(stats)} keys")
        row.update(stats)

    if run_events:
        events = _compute_events_for_csv(lc_path)
        row.update(events)

    return pd.DataFrame([row])


def _compute_stats_from_skypatrol_csv(lc_path: Path) -> dict:
    """Compute full compute_stats() suite from a SkyPatrol-format CSV."""
    try:
        candidate_id = Path(lc_path).stem
        parent = str(Path(lc_path).parent)
        _df, summary = compute_stats(candidate_id, parent, compute_ls=True)
        out: dict = {}
        merge_stats_summary_into_payload(out, summary)
        return out
    except Exception as e:
        print(f"[fetch] Warning: full stats computation failed: {e}")
        traceback.print_exc()
        return {}


def _compute_events_for_csv(lc_path: Path) -> dict:
    """Run process_lightcurve on a SkyPatrol CSV file and return key columns."""
    try:
        result = process_lightcurve(
            str(lc_path),
            trigger_mode=TRIGGER_MODE,
            logbf_threshold_dip=LOGBF_THRESHOLD_DIP,
            logbf_threshold_jump=LOGBF_THRESHOLD_JUMP,
            significance_threshold=SIGNIFICANCE_THRESHOLD,
            p_points=P_POINTS,
            p_min_dip=P_MIN_DIP,
            p_max_dip=P_MAX_DIP,
            p_min_jump=P_MIN_JUMP,
            p_max_jump=P_MAX_JUMP,
            mag_points=MAG_POINTS,
            run_min_points=RUN_MIN_POINTS,
            max_gap_points=MAX_GAP_POINTS,
            run_max_gap_days=RUN_MAX_GAP_DAYS,
            run_min_duration_days=RUN_MIN_DURATION_DAYS,
            baseline_tag=BASELINE_TAG,
            compute_event_prob=True,
        )
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[fetch] Warning: process_lightcurve failed: {e}")
        return {}
