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
        print(f"[fetch] Stats computed: {len(stats)} keys: {sorted(stats.keys())}")
        row.update(stats)

    if run_events:
        events = _compute_events_for_csv(lc_path)
        row.update(events)

    return pd.DataFrame([row])


def _compute_stats_from_skypatrol_csv(lc_path: Path) -> dict:
    """Compute full variability stats from a SkyPatrol-format CSV."""
    try:
        from malca.utils import read_skypatrol_csv

        df = read_skypatrol_csv(lc_path)
        if df.empty:
            return {}

        # Use g-band preferentially
        df_g = df[df["v_g_band"] == 0]
        if df_g.empty:
            df_g = df

        # Filter to good data
        good = df_g[(df_g["good_bad"] == 1)].copy()
        if good.empty:
            good = df_g.copy()

        sort_idx = np.argsort(good["JD"].values)
        jd = good["JD"].values[sort_idx]
        mag = good["mag"].values[sort_idx]
        err = good["error"].values[sort_idx]

        n = len(jd)
        if n == 0:
            return {}

        jd_start = float(jd[0])
        jd_end = float(jd[-1])
        span = jd_end - jd_start

        dt = np.diff(jd)
        cadence_median = float(np.nanmedian(dt)) if len(dt) else np.nan

        mean_mag = float(np.nanmean(mag))
        median_mag = float(np.nanmedian(mag))
        std_mag = float(np.nanstd(mag, ddof=1)) if n > 1 else 0.0

        # Robust sigma (MAD-based)
        mad = np.nanmedian(np.abs(mag - median_mag))
        robust_sigma = float(1.4826 * mad) if np.isfinite(mad) else std_mag

        # IQR
        q75, q25 = np.nanpercentile(mag, [75, 25])
        iqr = float(q75 - q25)

        # Reduced chi-squared vs constant (mean)
        finite_err = err[np.isfinite(err) & (err > 0)]
        if len(finite_err) > 1:
            resid = mag[np.isfinite(err) & (err > 0)] - mean_mag
            chi2 = np.sum((resid / finite_err) ** 2)
            reduced_chi2 = float(chi2 / (len(finite_err) - 1))
        else:
            reduced_chi2 = np.nan

        # Von Neumann ratio
        if n > 1 and std_mag > 0:
            delta_sq = np.sum(np.diff(mag) ** 2) / (n - 1)
            von_neumann = float(delta_sq / (std_mag ** 2))
        else:
            von_neumann = np.nan

        # Lag-1 autocorrelation
        if n > 2 and std_mag > 0:
            m = mag - mean_mag
            lag1 = float(np.sum(m[:-1] * m[1:]) / np.sum(m ** 2))
        else:
            lag1 = np.nan

        # Stetson J and K indices
        if n > 1 and len(finite_err) > 1:
            valid = np.isfinite(err) & (err > 0)
            m_v = mag[valid]
            e_v = err[valid]
            delta = (m_v - np.nanmean(m_v)) / e_v
            # Stetson J: sum of sign(product of consecutive residuals) * sqrt(abs(product))
            if len(delta) > 1:
                p = delta[:-1] * delta[1:]
                stetson_j = float(np.sum(np.sign(p) * np.sqrt(np.abs(p))) / len(p))
            else:
                stetson_j = np.nan
            # Stetson K: kurtosis-like
            stetson_k = float(np.mean(np.abs(delta)) / np.sqrt(np.mean(delta ** 2))) if np.mean(delta ** 2) > 0 else np.nan
        else:
            stetson_j = np.nan
            stetson_k = np.nan

        # SNR median
        valid_snr = np.isfinite(err) & (err > 0) & np.isfinite(mag)
        if np.any(valid_snr):
            snr = np.abs(mag[valid_snr]) / err[valid_snr]
            snr_median = float(np.nanmedian(snr))
        else:
            snr_median = np.nan

        # Duty cycle
        nights = np.unique(np.floor(jd).astype(int))
        total_nights = int(np.floor(jd[-1]) - np.floor(jd[0]) + 1) if n > 0 else 0
        duty_cycle = float(len(nights) / total_nights) if total_nights > 0 else np.nan

        # Linear trend
        if n > 2:
            t = jd - jd[0]
            t_years = t / 365.25
            coeffs = np.polyfit(t_years, mag, 1)
            slope = float(coeffs[0])  # mag/year
            predicted = np.polyval(coeffs, t_years)
            ss_res = np.sum((mag - predicted) ** 2)
            ss_tot = np.sum((mag - mean_mag) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            slope = np.nan
            r2 = np.nan

        n_cameras = int(good["camera#"].nunique()) if "camera#" in good.columns else 0

        return {
            "n_points": n,
            "n_cameras": n_cameras,
            "baseline_mag": median_mag,
            "cadence_median_days": cadence_median,
            "stats_jd_start": jd_start,
            "stats_jd_end": jd_end,
            "stats_time_span_days": span,
            "stats_photometry_mean_mag": mean_mag,
            "stats_photometry_median_mag": median_mag,
            "stats_photometry_std_mag": std_mag,
            "stats_photometry_robust_sigma_mag": robust_sigma,
            "stats_photometry_IQR_mag": iqr,
            "stats_variability_reduced_chi2_vs_constant": reduced_chi2,
            "stats_variability_von_neumann_ratio": von_neumann,
            "stats_variability_lag1_autocorr": lag1,
            "stats_variability_stetson_J": stetson_j,
            "stats_variability_stetson_K": stetson_k,
            "stats_error_and_snr_stats_snr_median": snr_median,
            "stats_duty_cycle_fraction": duty_cycle,
            "stats_cadence_median_dt_days": cadence_median,
            "stats_trend_slope_mag_per_year": slope,
            "stats_trend_r2": r2,
            "stats_file_points_total": len(df),
            "stats_file_points_kept_after_filter": n,
        }
    except Exception as e:
        print(f"[fetch] Warning: stats computation failed: {e}")
        import traceback; traceback.print_exc()
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
