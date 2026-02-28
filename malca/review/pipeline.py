"""Pipeline stage detection and on-demand runner for the review widget.

Detects which analysis stages have been run (by checking for signature
columns in the candidate payload) and can run missing stages on demand.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage signatures: columns whose presence indicates a stage has run
# ---------------------------------------------------------------------------
STAGE_SIGNATURES: dict[str, list[str]] = {
    "stats": [
        "n_points",
        "cadence_median_days",
        "baseline_mag",
        "stats_photometry_mean_mag",
    ],
    "events": [
        "dip_significant",
        "dip_best_morph",
        "jump_significant",
    ],
    "characterize": [
        "parallax",
        "tmass_j",
        "gal_l",
    ],
    "vetting": [
        "simbad_main_id",
        "gaia_var_flag",
        "vetting_likely_known",
    ],
    "external_lcs": [
        "atlas_has_phot",
        "ztf_lc_n_det",
        "gaia_epoch_lc_n_g",
        "ps1_lc_n_points",
        "crts_lc_n_points",
    ],
}


def _is_missing_value(value: object) -> bool:
    """Return True when a payload value should be treated as absent."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _merge_stats_summary_into_payload(payload: dict, summary: dict) -> None:
    """Map compute_stats() output into the review app's expected stats schema."""
    if not isinstance(summary, dict):
        return

    scalar_map = {
        "file_points_total": "stats_file_points_total",
        "file_points_kept_after_filter": "stats_file_points_kept_after_filter",
        "jd_start": "stats_jd_start",
        "jd_end": "stats_jd_end",
        "time_span_days": "stats_time_span_days",
        "n_unique_nights": "stats_n_unique_nights",
        "duty_cycle_fraction": "stats_duty_cycle_fraction",
        "cadence_mean_dt_days": "stats_cadence_mean_dt_days",
        "cadence_median_dt_days": "stats_cadence_median_dt_days",
        "cadence_p05_dt_days": "stats_cadence_p05_dt_days",
        "cadence_p95_dt_days": "stats_cadence_p95_dt_days",
        "photometry_mean_mag": "stats_photometry_mean_mag",
        "photometry_median_mag": "stats_photometry_median_mag",
        "photometry_weighted_mean_mag": "stats_photometry_weighted_mean_mag",
        "photometry_weighted_mean_sem": "stats_photometry_weighted_mean_sem",
        "photometry_std_mag": "stats_photometry_std_mag",
        "photometry_robust_sigma_mag": "stats_photometry_robust_sigma_mag",
        "photometry_IQR_mag": "stats_photometry_IQR_mag",
        "photometry_p05_mag": "stats_photometry_p05_mag",
        "photometry_p16_mag": "stats_photometry_p16_mag",
        "photometry_p84_mag": "stats_photometry_p84_mag",
        "photometry_p95_mag": "stats_photometry_p95_mag",
        "clipped_mean_mag_3sigma_about_median": "stats_clipped_mean_mag_3sigma_about_median",
        "clipped_std_mag_3sigma_about_median": "stats_clipped_std_mag_3sigma_about_median",
        "n_outliers_removed_robust_3sigma": "stats_n_outliers_removed_robust_3sigma",
        "variability_reduced_chi2_vs_constant": "stats_variability_reduced_chi2_vs_constant",
        "variability_von_neumann_ratio": "stats_variability_von_neumann_ratio",
        "variability_lag1_autocorr": "stats_variability_lag1_autocorr",
        "variability_stetson_I": "stats_variability_stetson_I",
        "variability_stetson_J": "stats_variability_stetson_J",
        "variability_stetson_K": "stats_variability_stetson_K",
        "variability_lomb_scargle_best_period_days": "stats_variability_lomb_scargle_best_period_days",
        "variability_lomb_scargle_peak_power": "stats_variability_lomb_scargle_peak_power",
        "variability_lomb_scargle_fap": "stats_variability_lomb_scargle_fap",
        "trend_slope_mag_per_day": "stats_trend_slope_mag_per_day",
        "trend_slope_mag_per_year": "stats_trend_slope_mag_per_year",
        "trend_r2": "stats_trend_r2",
        # ALeRCE-style features
        "amplitude": "stats_amplitude",
        "beyond_1_std": "stats_beyond_1_std",
        "con": "stats_con",
        "delta_mag_fid": "stats_delta_mag_fid",
        "excess_var": "stats_excess_var",
        "first_mag": "stats_first_mag",
        "gskew": "stats_gskew",
        "max_slope": "stats_max_slope",
        "meanvariance": "stats_meanvariance",
        "median_abs_dev": "stats_median_abs_dev",
        "median_brp": "stats_median_brp",
        "percent_amplitude": "stats_percent_amplitude",
        "q31": "stats_q31",
        "skew": "stats_skew",
        "small_kurtosis": "stats_small_kurtosis",
        "pvar": "stats_pvar",
        "anderson_darling": "stats_anderson_darling",
        "pair_slope_trend": "stats_pair_slope_trend",
        "rcs": "stats_rcs",
        "autocor_length": "stats_autocor_length",
        "sf_ml_amplitude": "stats_sf_ml_amplitude",
        "sf_ml_gamma": "stats_sf_ml_gamma",
        # period-dependent features
        "harmonics_mag_1": "stats_harmonics_mag_1",
        "harmonics_mag_2": "stats_harmonics_mag_2",
        "harmonics_mag_3": "stats_harmonics_mag_3",
        "harmonics_mag_4": "stats_harmonics_mag_4",
        "harmonics_mag_5": "stats_harmonics_mag_5",
        "harmonics_mag_6": "stats_harmonics_mag_6",
        "harmonics_mag_7": "stats_harmonics_mag_7",
        "harmonics_phase_2": "stats_harmonics_phase_2",
        "harmonics_phase_3": "stats_harmonics_phase_3",
        "harmonics_phase_4": "stats_harmonics_phase_4",
        "harmonics_phase_5": "stats_harmonics_phase_5",
        "harmonics_phase_6": "stats_harmonics_phase_6",
        "harmonics_phase_7": "stats_harmonics_phase_7",
        "harmonics_mse": "stats_harmonics_mse",
        "psi_cs": "stats_psi_cs",
        "psi_eta": "stats_psi_eta",
        # stochastic model features
        "gp_drw_sigma": "stats_gp_drw_sigma",
        "gp_drw_tau": "stats_gp_drw_tau",
        "iar_phi": "stats_iar_phi",
        "mhps_high": "stats_mhps_high",
        "mhps_low": "stats_mhps_low",
        "mhps_non_zero": "stats_mhps_non_zero",
        "mhps_pn_flag": "stats_mhps_pn_flag",
        "mhps_ratio": "stats_mhps_ratio",
    }

    for source_key, target_key in scalar_map.items():
        value = summary.get(source_key)
        if _is_missing_value(value):
            continue
        if isinstance(value, (pd.DataFrame, pd.Series, dict, list, tuple, set)):
            continue
        payload[target_key] = value

    err_stats = summary.get("error_and_snr_stats")
    if isinstance(err_stats, dict):
        for subkey, value in err_stats.items():
            if _is_missing_value(value):
                continue
            if isinstance(value, (pd.DataFrame, pd.Series, dict, list, tuple, set)):
                continue
            payload[f"stats_error_and_snr_stats_{subkey}"] = value

    # Populate the core summary fields the review UI uses for the Stats chip.
    kept_points = summary.get("file_points_kept_after_filter")
    if not _is_missing_value(kept_points):
        payload["n_points"] = kept_points

    cadence_median = summary.get("cadence_median_dt_days")
    if not _is_missing_value(cadence_median):
        payload["cadence_median_days"] = cadence_median

    if _is_missing_value(payload.get("baseline_mag")):
        baseline_mag = summary.get("photometry_median_mag")
        if not _is_missing_value(baseline_mag):
            payload["baseline_mag"] = baseline_mag

    by_camera = summary.get("by_camera")
    if isinstance(by_camera, pd.DataFrame) and not by_camera.empty:
        payload["n_cameras"] = int(len(by_camera))


def detect_pipeline_status(payload: dict) -> dict[str, str]:
    """Determine which pipeline stages have completed for a candidate.

    Returns a dict mapping stage name → status:
      "complete"  = all signature columns present and non-null
      "partial"   = some signature columns present
      "missing"   = no signature columns present
    """
    result = {}
    for stage, sig_cols in STAGE_SIGNATURES.items():
        present = sum(
            1 for c in sig_cols
            if c in payload and payload[c] is not None
            and not (isinstance(payload[c], float) and np.isnan(payload[c]))
        )
        if present == 0:
            result[stage] = "missing"
        elif present == len(sig_cols):
            result[stage] = "complete"
        else:
            result[stage] = "partial"
    return result


def run_missing_stages(
    conn: sqlite3.Connection,
    candidate_id: str,
    progress_callback: Callable[[str], None] | None = None,
    force_stages: list[str] | None = None,
) -> list[str]:
    """Detect and run missing pipeline stages for a candidate.

    Returns a list of stage names that were executed.
    """
    # 1. Load payload from DB
    row = conn.execute(
        "SELECT payload_json, lc_path FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    conn.commit()  # Release structural read lock during heavy API wait times
    
    if row is None:
        raise ValueError(f"Candidate {candidate_id} not found in DB")

    payload = json.loads(row[0]) if row[0] else {}
    lc_path = row[1] or payload.get("lc_path")

    # 2. Detect current status
    status = detect_pipeline_status(payload)
    stages_run: list[str] = []

    def p(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"[pipeline] {msg}")

    # 3. Run missing stages in order
    force = force_stages or []

    if status.get("stats") in ("missing", "partial") or "stats" in force:
        if lc_path and Path(lc_path).exists():
            p("Computing LC stats...")
            _run_stats_stage(payload, lc_path, p)
            stages_run.append("stats")

    if status.get("events") in ("missing", "partial") or "events" in force:
        if lc_path and Path(lc_path).exists():
            p("Running event detection...")
            _run_events_stage(payload, lc_path, p)
            stages_run.append("events")

    if status.get("characterize") in ("missing", "partial") or "characterize" in force:
        ra = payload.get("ra_deg")
        dec = payload.get("dec_deg")
        if ra is not None and dec is not None:
            p("Characterizing...")
            _run_characterize_stage(payload, p)
            stages_run.append("characterize")

    if status.get("vetting") in ("missing", "partial") or "vetting" in force:
        ra = payload.get("ra_deg")
        dec = payload.get("dec_deg")
        if ra is not None and dec is not None:
            p("Vetting crossmatches...")
            _run_vetting_stage(payload, p)
            stages_run.append("vetting")

    if status.get("external_lcs") in ("missing", "partial") or "external_lcs" in force:
        ra = payload.get("ra_deg")
        dec = payload.get("dec_deg")
        if ra is not None and dec is not None:
            p("Fetching external LCs...")
            _run_external_lcs_stage(payload, output_dir=_resolve_output_dir(conn, candidate_id), p=p)
            stages_run.append("external_lcs")

    # 4. Write updated payload back to DB
    if stages_run:
        update_candidate_payload(conn, candidate_id, payload)

    return stages_run


def update_candidate_payload(
    conn: sqlite3.Connection,
    candidate_id: str,
    updates: dict,
) -> None:
    """Merge *updates* into the existing payload_json for a candidate."""
    row = conn.execute(
        "SELECT payload_json FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return
    existing = json.loads(row[0]) if row[0] else {}
    existing.update(updates)
    conn.execute(
        "UPDATE candidates SET payload_json = ? WHERE candidate_id = ?",
        (json.dumps(existing, default=str), candidate_id),
    )

    # Also update extracted columns if they match _CANDIDATE_COLUMNS
    from malca.review.store import _CANDIDATE_COLUMNS, _as_bool, _to_float
    col_updates = []
    params = []
    for col, _dtype, etype in _CANDIDATE_COLUMNS:
        if col in updates:
            col_updates.append(f"{col} = ?")
            raw = updates[col]
            if etype == "bool":
                params.append(int(_as_bool(raw)) if raw is not None else None)
            elif etype == "float":
                params.append(_to_float(raw))
            else:
                params.append(str(raw) if raw is not None else None)
    if col_updates:
        params.append(candidate_id)
        conn.execute(
            f"UPDATE candidates SET {', '.join(col_updates)} WHERE candidate_id = ?",
            params,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_stats_stage(payload: dict, lc_path: str, p: Callable | None = None) -> None:
    """Run compute_stats and merge results into payload."""
    try:
        from malca.stats import compute_stats
        candidate_id = Path(lc_path).stem
        parent = str(Path(lc_path).parent)
        _df, summary = compute_stats(candidate_id, parent)
        _merge_stats_summary_into_payload(payload, summary)
    except Exception as e:
        if p: p(f"Stats stage failed: {e}")
        else: print(f"[pipeline] Stats stage failed: {e}")


def _run_events_stage(payload: dict, lc_path: str, p: Callable | None = None) -> None:
    """Run process_lightcurve and merge results into payload."""
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
        if isinstance(result, dict):
            payload.update(result)
    except Exception as e:
        if p: p(f"Events stage failed: {e}")
        else: print(f"[pipeline] Events stage failed: {e}")


def _run_characterize_stage(payload: dict, p: Callable | None = None) -> None:
    """Run characterize_candidates_df on a 1-row DataFrame."""
    try:
        from malca.characterize import characterize_candidates_df
        df = pd.DataFrame([payload])
        df_out = characterize_candidates_df(df)
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
    except Exception as e:
        if p: p(f"Characterize stage failed: {e}")
        else: print(f"[pipeline] Characterize stage failed: {e}")


def _run_vetting_stage(payload: dict, p: Callable | None = None) -> None:
    """Run vet_candidates on a 1-row DataFrame."""
    try:
        from malca.vetting import vet_candidates
        df = pd.DataFrame([payload])
        df_out = vet_candidates(
            df,
            run_atlas=False,
            # (other vetting happens unconditionally in vet_candidates if columns missing)
        )
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
    except Exception as e:
        if p: p(f"Vetting stage failed: {e}")
        else: print(f"[pipeline] Vetting stage failed: {e}")


def _resolve_output_dir(conn: sqlite3.Connection, candidate_id: str) -> Path:
    """Resolve the results output directory for a candidate's run."""
    row = conn.execute(
        "SELECT source_path FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row and row[0]:
        src = Path(str(row[0]))
        # If source_path is under a run directory, use its results/ dir
        if src.parent.name == "results":
            return src.parent
        if (src.parent / "results").is_dir():
            return src.parent / "results"
    # Fallback: use the default output directory
    default = Path(__file__).resolve().parents[2] / "output" / "results"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _run_external_lcs_stage(payload: dict, output_dir: Path, p: Callable | None = None) -> None:
    """Run fetch_external_lcs on a 1-row DataFrame."""
    try:
        from malca.vetting import fetch_external_lcs
        df = pd.DataFrame([payload])
        df_out = fetch_external_lcs(
            df,
            output_dir=output_dir,
            run_atlas=True,
            run_ztf=True,
            run_gaia_epoch=True,
            run_tess=False,
            run_kepler=False,
            run_aavso=False,
            run_ps1=True,
            run_crts=True,
            progress_callback=p,
        )
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
    except Exception as e:
        if p: p(f"External LCs stage failed: {e}")
        else: print(f"[pipeline] External LCs stage failed: {e}")
