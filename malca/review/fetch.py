"""Orchestrate light-curve fetch → analyse → import for the review widget.

Supports three search modes:
  - ASAS-SN ID  → download_lightcurve_by_id
  - Gaia DR3 ID → download_lightcurve_by_gaia_id
  - RA/Dec      → cone_search (returns catalog rows; caller picks target)

After downloading, we run compute_stats and optionally events detection,
then hand the result to import_candidates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.fetch import (
    cone_search,
    download_lightcurve_by_id,
    download_lightcurve_by_gaia_id,
    download_lightcurve_for_target,
)


def fetch_and_analyze_by_id(
    asas_sn_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Download LC by ASAS-SN ID, compute basic stats, return (1-row DF, lc_path).

    The returned DataFrame has at minimum ``candidate_id`` and ``lc_path``
    columns so it can be passed directly to ``import_candidates``.
    """
    lc_path = download_lightcurve_by_id(asas_sn_id)
    df = _build_candidate_row(asas_sn_id, lc_path, run_stats=run_stats, run_events=run_events)
    return df, lc_path


def fetch_and_analyze_by_gaia_id(
    gaia_id: str,
    *,
    run_stats: bool = True,
    run_events: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Download LC by Gaia DR3 source_id, compute basic stats."""
    lc_path = download_lightcurve_by_gaia_id(gaia_id)
    candidate_id = f"gaia_{gaia_id}"
    df = _build_candidate_row(candidate_id, lc_path, run_stats=run_stats, run_events=run_events)
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
    *,
    run_stats: bool = True,
    run_events: bool = False,
) -> pd.DataFrame:
    """Create a single-row candidate DataFrame suitable for import_candidates."""
    row: dict = {
        "candidate_id": str(candidate_id),
        "lc_path": str(lc_path),
    }

    if run_stats:
        stats = _compute_stats_for_csv(lc_path)
        row.update(stats)

    if run_events:
        events = _compute_events_for_csv(lc_path)
        row.update(events)

    return pd.DataFrame([row])


def _compute_stats_for_csv(lc_path: Path) -> dict:
    """Run compute_stats on a SkyPatrol CSV file and return a flat dict."""
    try:
        from malca.stats import compute_stats
        candidate_id = lc_path.stem
        parent = str(lc_path.parent)
        _df, summary = compute_stats(candidate_id, parent)
        return summary if isinstance(summary, dict) else {}
    except Exception as e:
        print(f"[fetch] Warning: compute_stats failed: {e}")
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
