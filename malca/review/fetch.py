"""Orchestrate light-curve fetch → analyse → import for the review widget.

Supports three search modes:
  - ASAS-SN ID  → download_lightcurve_by_id
  - Gaia DR3 ID → download_lightcurve_by_gaia_id
  - RA/Dec      → cone_search (returns catalog rows; caller picks target)

After downloading, we compute stats from the SkyPatrol CSV directly,
then hand the result to import_candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.fetch import (
    cone_search,
    download_lightcurve_by_id,
    download_lightcurve_by_gaia_id,
)


def fetch_and_analyze_by_id(
    asas_sn_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Download LC by ASAS-SN ID, compute basic stats, return (1-row DF, lc_path)."""
    lc_path, catalog_info = download_lightcurve_by_id(asas_sn_id)
    df = _build_candidate_row(asas_sn_id, lc_path, catalog_info,
                              run_stats=run_stats, run_events=run_events)
    return df, lc_path


def fetch_and_analyze_by_gaia_id(
    gaia_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Download LC by Gaia DR3 source_id, compute basic stats."""
    lc_path, catalog_info = download_lightcurve_by_gaia_id(gaia_id)
    candidate_id = str(catalog_info.get("asas_sn_id", f"gaia_{gaia_id}"))
    df = _build_candidate_row(candidate_id, lc_path, catalog_info,
                              run_stats=run_stats, run_events=run_events)
    return df, lc_path


def fetch_cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
) -> pd.DataFrame:
    """Return catalog rows from a cone search (no LC download)."""
    return cone_search(ra, dec, radius_arcsec=radius_arcsec)


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

    # Populate metadata from catalog query
    if catalog_info:
        for key in ("asas_sn_id", "ra_deg", "dec_deg", "gaia_id", "source_id",
                     "mean_vmag", "phot_g_mean_mag", "catalog_sources"):
            if key in catalog_info and catalog_info[key] is not None:
                row[key] = catalog_info[key]

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
        row.update(stats)

    if run_events:
        events = _compute_events_for_csv(lc_path)
        row.update(events)

    return pd.DataFrame([row])


def _compute_stats_from_skypatrol_csv(lc_path: Path) -> dict:
    """Compute basic stats directly from a SkyPatrol-format CSV.

    Uses ``read_skypatrol_csv`` which handles the column remapping
    (JD, Mag, Mag Error, Filter, Quality, Camera → internal names).
    """
    try:
        from malca.utils import read_skypatrol_csv

        df = read_skypatrol_csv(lc_path)
        if df.empty:
            return {}

        # Use g-band preferentially
        df_g = df[df["v_g_band"] == 0]
        if df_g.empty:
            df_g = df  # fall back to all data

        # Filter to good data
        good = df_g[(df_g["good_bad"] == 1)].copy()
        if good.empty:
            good = df_g.copy()

        jd = good["JD"].values
        mag = good["mag"].values
        err = good["error"].values

        n_points = len(good)
        jd_start = float(np.nanmin(jd)) if n_points else np.nan
        jd_end = float(np.nanmax(jd)) if n_points else np.nan
        span = jd_end - jd_start if n_points else 0.0

        dt = np.diff(np.sort(jd))
        cadence_median = float(np.nanmedian(dt)) if len(dt) else np.nan

        mean_mag = float(np.nanmean(mag))
        median_mag = float(np.nanmedian(mag))
        std_mag = float(np.nanstd(mag, ddof=1)) if n_points > 1 else 0.0

        n_cameras = int(good["camera#"].nunique()) if "camera#" in good.columns else 0

        return {
            "n_points": n_points,
            "n_cameras": n_cameras,
            "baseline_mag": median_mag,
            "cadence_median_days": cadence_median,
            "stats_jd_start": jd_start,
            "stats_jd_end": jd_end,
            "stats_time_span_days": span,
            "stats_photometry_mean_mag": mean_mag,
            "stats_photometry_median_mag": median_mag,
            "stats_photometry_std_mag": std_mag,
            "stats_file_points_total": len(df),
            "stats_file_points_kept_after_filter": n_points,
        }
    except Exception as e:
        print(f"[fetch] Warning: stats computation failed: {e}")
        return {}


def _compute_events_for_csv(lc_path: Path) -> dict:
    """Run process_lightcurve on a SkyPatrol CSV file and return key columns."""
    try:
        from malca.events import process_lightcurve
        from malca.config.config_events import (
            TRIGGER_MODE, LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP,
            SIGNIFICANCE_THRESHOLD, P_POINTS,
            P_MIN_DIP, P_MAX_DIP, P_MIN_JUMP, P_MAX_JUMP,
            MAG_POINTS, RUN_MIN_POINTS, MAX_GAP_POINTS,
            RUN_MAX_GAP_DAYS, RUN_MIN_DURATION_DAYS,
            BASELINE_TAG,
        )
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
