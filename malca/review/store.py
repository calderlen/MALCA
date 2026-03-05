from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import math
import shutil
import sqlite3

import numpy as np
import pandas as pd

from malca.characterize import characterize_candidates_df
from malca.config.config_characterize import GAIA_CHUNK_SIZE
from malca.config.config_paths import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE
from malca.review.metadata import normalize_vsx_record
from malca.vetting import vet_candidates







DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "output" / "review" / "review.db"
DEFAULT_STANDALONE_DB_PATH = Path(__file__).resolve().parents[2] / "output" / "review" / "standalone.db"
STATUS_OPTIONS = ["unreviewed", "reviewed", "needs_followup"]
EVENT_CLASS_OPTIONS = [
    "unclassified",
    "dipper",
    "yso",
    "microlensing",
    "flare",
    "instrumental",
    "unknown_interesting",
    "other",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _to_float(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        x = float(v)
        if np.isnan(x):
            return None
        return x
    except Exception:
        return None


def _normalize_large_integer_like_id(v) -> str | None:
    """Normalize large integer-like identifiers to plain strings.

    Converts values like 4.272990850383009e+17 -> "427299085038300900".
    Returns None for missing values.
    """
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s
    if d == d.to_integral_value():
        try:
            return format(d.to_integral_value(), "f")
        except Exception:
            return s
    return s


def infer_candidate_id(df: pd.DataFrame) -> pd.Series:
    if "candidate_id" not in df.columns:
        raise ValueError("Input must include a 'candidate_id' column.")

    vals = df["candidate_id"].astype(str).str.strip()
    if not vals.nunique(dropna=True) == len(df):
        raise ValueError("'candidate_id' values must be unique.")
    return vals


def load_candidates_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file type. Use CSV or Parquet.")


def detect_run_directory_files(run_dir: Path) -> dict[str, Path | None]:
    """
    Auto-detect MALCA review files from a run directory.

    Returns dict with keys:
    - 'candidates': Path to best candidates file found (or None)
    - 'plot_dir': Path to plots directory (or None)
    - 'gaia_cache': Path to gaia cache (or None)
    - 'run_params': Path to run_params.json (or None)
    - 'warnings': List of warning messages
    """
    results = {
        'candidates': None,
        'plot_dir': None,
        'gaia_cache': None,
        'run_params': None,
        'warnings': []
    }

    # Validate directory exists
    if not run_dir.exists():
        results['warnings'].append(f"Directory does not exist: {run_dir}")
        return results

    if not run_dir.is_dir():
        results['warnings'].append(f"Path is not a directory: {run_dir}")
        return results

    # Check for run_params.json (validates it's a run directory)
    run_params = run_dir / "run_params.json"
    if run_params.exists():
        results['run_params'] = run_params
    else:
        results['warnings'].append("run_params.json not found - may not be a MALCA run directory")

    # Detect candidates file.
    # Priority: vetted products first, then non-vetted products.
    candidates_priority: list[Path] = []

    vetted_exact = run_dir / "results" / "lc_events_vetted.parquet"
    if vetted_exact.exists():
        candidates_priority.append(vetted_exact)

    vetted_pattern = sorted(
        (run_dir / "results").glob("lc_events_vetted_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates_priority.extend(vetted_pattern)

    for rel_path in (
        "results/lc_events_spectra.parquet",
        "results/lc_events_neighbors.parquet",
        "results/lc_events_classified.parquet",
        "results/lc_events_enriched.parquet",
        "results/lc_events_characterized.parquet",
        "results/lc_events_filtered.parquet",
    ):
        candidate_file = run_dir / rel_path
        if candidate_file.exists():
            candidates_priority.append(candidate_file)

    for candidate_file in candidates_priority:
        if candidate_file.exists():
            results['candidates'] = candidate_file
            break

    if results['candidates'] is None:
        results['warnings'].append("No candidates file found in results/ directory")

    # Detect plot directory
    plot_dir = run_dir / "plots"
    if plot_dir.exists() and plot_dir.is_dir():
        results['plot_dir'] = plot_dir
    else:
        results['warnings'].append("plots/ directory not found")

    # Detect gaia cache (optional, no warning if missing)
    gaia_cache = run_dir / "gaia_cache" / "gaia_cache.parquet"
    if gaia_cache.exists():
        results['gaia_cache'] = gaia_cache

    return results


# ---------------------------------------------------------------------------
# Single source of truth for all extracted candidate columns.
#
# Each entry: (column_name, sql_type, extract_type)
#   extract_type: 'bool' | 'float' | 'text'
#
# The order here determines column order in the DB table and INSERT.
# 'candidate_id', 'source_path', 'payload_json', 'imported_at' are handled
# separately (they aren't payload fields).
# ---------------------------------------------------------------------------
_CANDIDATE_COLUMNS: list[tuple[str, str, str]] = [
    # -- identification --
    ("asas_sn_id",               "TEXT",    "text"),
    ("lc_path",                  "TEXT",    "text"),
    # -- top-level filter flags --
    ("failed_any",               "INTEGER", "bool"),
    ("periodic_flag",            "INTEGER", "bool"),
    ("catalog_match",            "INTEGER", "bool"),
    ("catalog_source",           "TEXT",    "text"),
    ("period_sources",           "TEXT",    "text"),
    ("period_n_sources",         "REAL",    "float"),
    ("period_consensus_days",    "REAL",    "float"),
    ("period_consensus_agree",   "INTEGER", "bool"),
    ("period_conflict_flag",     "INTEGER", "bool"),
    ("period_consensus_support", "REAL",    "float"),
    ("period_primary_source",    "TEXT",    "text"),
    ("period_source_periods",    "TEXT",    "text"),
    ("period_ogle_match",        "INTEGER", "bool"),
    ("period_ogle_days",         "REAL",    "float"),
    ("period_ogle_class",        "TEXT",    "text"),
    ("period_ogle_sep_arcsec",   "REAL",    "float"),
    ("high_ruwe_flag",           "INTEGER", "bool"),
    # -- periodicity --
    ("periodicity_score",        "REAL",    "float"),
    ("lsp_bootstrap_sig",        "REAL",    "float"),
    ("lsp_power",                "REAL",    "float"),
    ("lsp_period",               "REAL",    "float"),
    ("lsp_is_alias",             "INTEGER", "bool"),
    ("lsp_is_significant",       "INTEGER", "bool"),
    ("phase_plot_ready",         "INTEGER", "bool"),
    ("phase_period_days",        "REAL",    "float"),
    ("phase_source",             "TEXT",    "text"),
    ("phase_quality_score",      "REAL",    "float"),
    # -- dip detection --
    ("dip_significant",          "INTEGER", "bool"),
    ("dip_best_morph",           "TEXT",    "text"),
    ("dip_best_log_bf",          "REAL",    "float"),
    ("dip_best_delta_bic",       "REAL",    "float"),
    ("dip_best_width_param",     "REAL",    "float"),
    ("dip_symmetry_score",       "REAL",    "float"),
    ("dip_best_amp",             "REAL",    "float"),
    ("dip_best_t0",              "REAL",    "float"),
    ("dip_best_alpha",           "REAL",    "float"),
    ("dip_best_tau",             "REAL",    "float"),
    ("dip_bayes_factor",         "REAL",    "float"),
    ("dip_best_p",               "REAL",    "float"),
    ("dip_best_mag_event",       "REAL",    "float"),
    ("dip_trigger_max",          "REAL",    "float"),
    ("dip_max_event_prob",       "REAL",    "float"),
    ("dip_trigger_threshold",    "REAL",    "float"),
    # -- dip runs --
    ("dip_count",                "REAL",    "float"),
    ("dip_run_count",            "REAL",    "float"),
    ("dip_max_run_points",       "REAL",    "float"),
    ("dip_max_run_duration",     "REAL",    "float"),
    ("dip_max_run_sum",          "REAL",    "float"),
    ("dip_max_run_max",          "REAL",    "float"),
    ("dip_max_run_cameras",      "REAL",    "float"),
    ("dip_max_log_bf_local",     "REAL",    "float"),
    # -- jump detection --
    ("jump_significant",         "INTEGER", "bool"),
    ("jump_best_morph",          "TEXT",    "text"),
    ("jump_best_log_bf",         "REAL",    "float"),
    ("jump_best_delta_bic",      "REAL",    "float"),
    ("jump_best_width_param",    "REAL",    "float"),
    ("jump_best_amp",            "REAL",    "float"),
    ("jump_best_t0",             "REAL",    "float"),
    ("jump_best_alpha",          "REAL",    "float"),
    ("jump_best_tau",            "REAL",    "float"),
    ("jump_bayes_factor",        "REAL",    "float"),
    ("jump_best_p",              "REAL",    "float"),
    ("jump_best_mag_event",      "REAL",    "float"),
    ("jump_trigger_max",         "REAL",    "float"),
    ("jump_max_event_prob",      "REAL",    "float"),
    ("jump_trigger_threshold",   "REAL",    "float"),
    # -- jump runs --
    ("jump_count",               "REAL",    "float"),
    ("jump_run_count",           "REAL",    "float"),
    ("jump_max_run_points",      "REAL",    "float"),
    ("jump_max_run_duration",    "REAL",    "float"),
    ("jump_max_run_sum",         "REAL",    "float"),
    ("jump_max_run_max",         "REAL",    "float"),
    ("jump_max_run_cameras",     "REAL",    "float"),
    ("jump_max_log_bf_local",    "REAL",    "float"),
    # -- dip recurrence --
    ("dip_is_single_event",              "INTEGER", "bool"),
    ("dip_inter_event_spacing_median",   "REAL",    "float"),
    ("dip_inter_event_spacing_std",      "REAL",    "float"),
    ("dip_amplitude_consistency",        "REAL",    "float"),
    ("dip_duration_consistency",         "REAL",    "float"),
    # -- jump recurrence --
    ("jump_is_single_event",             "INTEGER", "bool"),
    ("jump_inter_event_spacing_median",  "REAL",    "float"),
    ("jump_inter_event_spacing_std",     "REAL",    "float"),
    ("jump_amplitude_consistency",       "REAL",    "float"),
    ("jump_duration_consistency",        "REAL",    "float"),
    # -- event scoring --
    ("dipper_score",             "REAL",    "float"),
    ("dipper_n_dips",            "REAL",    "float"),
    ("dipper_n_valid_dips",      "REAL",    "float"),
    ("jumper_score",             "REAL",    "float"),
    ("jumper_n_jumps",           "REAL",    "float"),
    ("jumper_n_valid_jumps",     "REAL",    "float"),
    # -- stellar parameters --
    ("ruwe",                     "REAL",    "float"),
    ("radial_velocity",          "REAL",    "float"),
    ("rv_amplitude_robust",      "REAL",    "float"),
    ("teff_gspphot",             "REAL",    "float"),
    ("logg_gspphot",             "REAL",    "float"),
    ("mh_gspphot",               "REAL",    "float"),
    ("distance_gspphot",         "REAL",    "float"),
    ("parallax",                 "REAL",    "float"),
    ("parallax_error",           "REAL",    "float"),
    ("pmra",                     "REAL",    "float"),
    ("pmdec",                    "REAL",    "float"),
    # -- photometry --
    ("tmass_j",                  "REAL",    "float"),
    ("tmass_j_err",              "REAL",    "float"),
    ("tmass_h",                  "REAL",    "float"),
    ("tmass_h_err",              "REAL",    "float"),
    ("tmass_k",                  "REAL",    "float"),
    ("tmass_k_err",              "REAL",    "float"),
    ("unwise_w1",                "REAL",    "float"),
    ("unwise_w1_err",            "REAL",    "float"),
    ("unwise_w2",                "REAL",    "float"),
    ("unwise_w2_err",            "REAL",    "float"),
    ("allwise_w3",               "REAL",    "float"),
    ("allwise_w3_err",           "REAL",    "float"),
    ("allwise_w4",               "REAL",    "float"),
    ("allwise_w4_err",           "REAL",    "float"),
    ("apass_v",                  "REAL",    "float"),
    ("apass_v_err",              "REAL",    "float"),
    ("apass_b",                  "REAL",    "float"),
    ("apass_b_err",              "REAL",    "float"),
    ("apass_g",                  "REAL",    "float"),
    ("apass_g_err",              "REAL",    "float"),
    ("apass_r",                  "REAL",    "float"),
    ("apass_r_err",              "REAL",    "float"),
    ("apass_i",                  "REAL",    "float"),
    ("apass_i_err",              "REAL",    "float"),
    ("galex_fuv",                "REAL",    "float"),
    ("galex_fuv_err",            "REAL",    "float"),
    ("galex_nuv",                "REAL",    "float"),
    ("galex_nuv_err",            "REAL",    "float"),
    ("H_K",                      "REAL",    "float"),
    ("W1_W2",                    "REAL",    "float"),
    ("iphas_ha_mag",             "REAL",    "float"),
    ("iphas_r_ha",               "REAL",    "float"),
    ("unwise_w1_zscore",         "REAL",    "float"),
    ("unwise_w2_zscore",         "REAL",    "float"),
    ("unwise_w1_var",            "INTEGER", "bool"),
    # -- galactic coordinates --
    ("gal_l",                    "REAL",    "float"),
    ("gal_b",                    "REAL",    "float"),
    # -- extinction & environment --
    ("A_v_3d",                   "REAL",    "float"),
    ("ebv_3d",                   "REAL",    "float"),
    ("dust_sigma",               "REAL",    "float"),
    ("population",               "TEXT",    "text"),
    ("age50",                    "REAL",    "float"),
    ("mass50",                   "REAL",    "float"),
    ("banyan_field_prob",        "REAL",    "float"),
    ("banyan_best_assoc",        "TEXT",    "text"),
    # -- crossmatch details --
    ("vsx_class",                "TEXT",    "select"),
    ("vsx_sep_arcsec",           "REAL",    "float"),
    ("vsx_period",               "REAL",    "float"),
    ("sfr_name",                 "TEXT",    "text"),
    ("sfr_sep_arcmin",           "REAL",    "float"),
    ("cluster_name",             "TEXT",    "text"),
    ("cluster_membership_prob",  "REAL",    "float"),
    # -- vetting classification --
    ("vetting_likely_known",     "INTEGER", "bool"),
    ("asassn_var_type",          "TEXT",    "select"),
    ("gaia_var_class",           "TEXT",    "select"),
    ("simbad_otype",             "TEXT",    "select"),
    ("ztf_var_type",             "TEXT",    "select"),
    # -- vetting details: SIMBAD --
    ("simbad_main_id",           "TEXT",    "text"),
    ("simbad_nbref",             "REAL",    "float"),
    ("simbad_sep_arcsec",        "REAL",    "float"),
    # -- vetting details: Gaia variability --
    ("gaia_var_flag",            "TEXT",    "text"),
    ("gaia_var_score",           "REAL",    "float"),
    # -- vetting details: Gaia EB --
    ("gaia_eb_period",           "REAL",    "float"),
    ("gaia_eb_morph",            "TEXT",    "text"),
    ("gaia_eb_global_ranking",   "REAL",    "float"),
    # -- vetting details: Gaia epoch --
    ("gaia_epoch_available",     "INTEGER", "bool"),
    ("gaia_epoch_n_obs",         "REAL",    "float"),
    ("gaia_epoch_g_range",       "REAL",    "float"),
    # -- vetting details: ASAS-SN --
    ("asassn_var_name",          "TEXT",    "text"),
    ("asassn_var_period",        "REAL",    "float"),
    # -- vetting details: ZTF --
    ("ztf_var_period",           "REAL",    "float"),
    ("ztf_var_amp",              "REAL",    "float"),
    # -- vetting details: TNS --
    ("tns_name",                 "TEXT",    "text"),
    ("tns_type",                 "TEXT",    "select"),
    ("tns_redshift",             "REAL",    "float"),
    ("tns_disc_date",            "TEXT",    "text"),
    # -- vetting details: ALeRCE --
    ("alerce_oid",               "TEXT",    "text"),
    ("alerce_ndet",              "REAL",    "float"),
    ("alerce_lc_class",          "TEXT",    "select"),
    ("alerce_lc_prob",           "REAL",    "float"),
    ("alerce_stamp_class",       "TEXT",    "text"),
    ("alerce_stamp_prob",        "REAL",    "float"),
    # -- vetting details: X-ray --
    ("xray_det",                 "INTEGER", "bool"),
    ("xray_flux",                "REAL",    "float"),
    ("xray_sep_arcsec",          "REAL",    "float"),
    # -- vetting details: proper motion --
    ("pm_cluster_offset_sigma",  "REAL",    "float"),
    # -- vetting details: ATLAS --
    ("atlas_has_phot",           "INTEGER", "bool"),
    ("atlas_n_det_cyan",         "REAL",    "float"),
    ("atlas_n_det_orange",       "REAL",    "float"),
    ("atlas_cyan_range",         "REAL",    "float"),
    ("atlas_orange_range",       "REAL",    "float"),
    # -- vetting details: NEOWISE --
    ("neowise_n_epochs",         "REAL",    "float"),
    ("neowise_w1_range",         "REAL",    "float"),
    ("neowise_w2_range",         "REAL",    "float"),
    # -- external light curves: ZTF --
    ("ztf_lc_n_det",             "INTEGER", "float"),
    ("ztf_lc_g_range",           "REAL",    "float"),
    ("ztf_lc_r_range",           "REAL",    "float"),
    # -- external light curves: Gaia epoch --
    ("gaia_epoch_lc_n_g",        "INTEGER", "float"),
    ("gaia_epoch_lc_g_range",    "REAL",    "float"),
    # -- external light curves: TESS --
    ("tess_n_sectors",           "INTEGER", "float"),
    ("tess_total_points",        "INTEGER", "float"),
    ("tess_flux_range",          "REAL",    "float"),
    # -- external light curves: Kepler --
    ("kepler_n_quarters",        "INTEGER", "float"),
    ("kepler_total_points",      "INTEGER", "float"),
    ("kepler_flux_range",        "REAL",    "float"),
    # -- external light curves: AAVSO --
    ("aavso_lc_n_points",        "INTEGER", "float"),
    # -- external light curves: Pan-STARRS --
    ("ps1_lc_n_points",          "INTEGER", "float"),
    # -- external light curves: CRTS --
    ("crts_lc_n_points",         "INTEGER", "float"),
    # -- vetting details: other --
    ("cluster_dist_pc",          "REAL",    "float"),
    ("iphas_ha_excess",          "REAL",    "float"),
    # -- light curve basics --
    ("n_points",                 "REAL",    "float"),
    ("n_cameras",                "REAL",    "float"),
    ("baseline_mag",             "REAL",    "float"),
    ("baseline_source",          "TEXT",    "text"),
    ("cadence_median_days",      "REAL",    "float"),
    ("trigger_mode",             "TEXT",    "text"),
    # -- YSO / classification --
    ("trigger_type",             "TEXT",    "text"),
    ("yso_class",                "TEXT",    "select"),
    ("final_class",              "TEXT",    "text"),
    ("P_eb",                     "REAL",    "float"),
    ("P_cv",                     "REAL",    "float"),
    ("P_starspot",               "REAL",    "float"),
    ("P_disk",                   "REAL",    "float"),
    ("a_circ_au",                "REAL",    "float"),
    ("transit_prob",             "REAL",    "float"),
    ("hill_radius_rsun",         "REAL",    "float"),
    # -- individual fail flags --
    ("failed_posterior_strength", "INTEGER", "bool"),
    ("failed_run_robustness",    "INTEGER", "bool"),
    ("failed_morphology",        "INTEGER", "bool"),
    ("failed_score",             "INTEGER", "bool"),
    ("failed_periodicity",       "INTEGER", "bool"),
    ("failed_gaia_ruwe",         "INTEGER", "bool"),
    ("failed_periodic_catalog",  "INTEGER", "bool"),
    ("failed_signal_amplitude",  "INTEGER", "bool"),
    ("bad_cameras_filtered",     "INTEGER", "bool"),
    # -- light curve statistics (from stats.py / enrichment) --
    ("stats_file_points_total",                    "REAL", "float"),
    ("stats_file_points_kept_after_filter",         "REAL", "float"),
    ("stats_jd_start",                             "REAL", "float"),
    ("stats_jd_end",                               "REAL", "float"),
    ("stats_time_span_days",                       "REAL", "float"),
    ("stats_n_unique_nights",                      "REAL", "float"),
    ("stats_duty_cycle_fraction",                  "REAL", "float"),
    ("stats_cadence_mean_dt_days",                 "REAL", "float"),
    ("stats_cadence_median_dt_days",               "REAL", "float"),
    ("stats_cadence_p05_dt_days",                  "REAL", "float"),
    ("stats_cadence_p95_dt_days",                  "REAL", "float"),
    ("stats_photometry_mean_mag",                  "REAL", "float"),
    ("stats_photometry_median_mag",                "REAL", "float"),
    ("stats_photometry_weighted_mean_mag",         "REAL", "float"),
    ("stats_photometry_weighted_mean_sem",         "REAL", "float"),
    ("stats_photometry_std_mag",                   "REAL", "float"),
    ("stats_photometry_robust_sigma_mag",          "REAL", "float"),
    ("stats_photometry_IQR_mag",                   "REAL", "float"),
    ("stats_photometry_p05_mag",                   "REAL", "float"),
    ("stats_photometry_p16_mag",                   "REAL", "float"),
    ("stats_photometry_p84_mag",                   "REAL", "float"),
    ("stats_photometry_p95_mag",                   "REAL", "float"),
    ("stats_clipped_mean_mag_3sigma_about_median", "REAL", "float"),
    ("stats_clipped_std_mag_3sigma_about_median",  "REAL", "float"),
    ("stats_n_outliers_removed_robust_3sigma",     "REAL", "float"),
    ("stats_error_and_snr_stats_error_mean",       "REAL", "float"),
    ("stats_error_and_snr_stats_error_median",     "REAL", "float"),
    ("stats_error_and_snr_stats_error_p05",        "REAL", "float"),
    ("stats_error_and_snr_stats_error_p95",        "REAL", "float"),
    ("stats_error_and_snr_stats_snr_median",       "REAL", "float"),
    ("stats_error_and_snr_stats_snr_p05",          "REAL", "float"),
    ("stats_error_and_snr_stats_snr_p95",          "REAL", "float"),
    ("stats_variability_reduced_chi2_vs_constant", "REAL", "float"),
    ("stats_variability_von_neumann_ratio",        "REAL", "float"),
    ("stats_variability_lag1_autocorr",             "REAL", "float"),
    ("stats_variability_stetson_I",                "REAL", "float"),
    ("stats_variability_stetson_J",                "REAL", "float"),
    ("stats_variability_stetson_K",                "REAL", "float"),
    ("stats_variability_lomb_scargle_best_period_days", "REAL", "float"),
    ("stats_variability_lomb_scargle_peak_power",  "REAL", "float"),
    ("stats_variability_lomb_scargle_fap",         "REAL", "float"),
    ("stats_trend_slope_mag_per_day",              "REAL", "float"),
    ("stats_trend_slope_mag_per_year",             "REAL", "float"),
    ("stats_trend_r2",                             "REAL", "float"),
    # -- ALeRCE-style features --
    ("stats_amplitude",                            "REAL", "float"),
    ("stats_beyond_1_std",                         "REAL", "float"),
    ("stats_con",                                  "REAL", "float"),
    ("stats_delta_mag_fid",                        "REAL", "float"),
    ("stats_excess_var",                           "REAL", "float"),
    ("stats_first_mag",                            "REAL", "float"),
    ("stats_gskew",                                "REAL", "float"),
    ("stats_max_slope",                            "REAL", "float"),
    ("stats_meanvariance",                         "REAL", "float"),
    ("stats_median_abs_dev",                       "REAL", "float"),
    ("stats_median_brp",                           "REAL", "float"),
    ("stats_percent_amplitude",                    "REAL", "float"),
    ("stats_q31",                                  "REAL", "float"),
    ("stats_skew",                                 "REAL", "float"),
    ("stats_small_kurtosis",                       "REAL", "float"),
    ("stats_pvar",                                 "REAL", "float"),
    ("stats_anderson_darling",                     "REAL", "float"),
    ("stats_pair_slope_trend",                     "REAL", "float"),
    ("stats_rcs",                                  "REAL", "float"),
    ("stats_autocor_length",                       "REAL", "float"),
    ("stats_sf_ml_amplitude",                      "REAL", "float"),
    ("stats_sf_ml_gamma",                          "REAL", "float"),
    # -- period-dependent features --
    ("stats_harmonics_mag_1",                      "REAL", "float"),
    ("stats_harmonics_mag_2",                      "REAL", "float"),
    ("stats_harmonics_mag_3",                      "REAL", "float"),
    ("stats_harmonics_mag_4",                      "REAL", "float"),
    ("stats_harmonics_mag_5",                      "REAL", "float"),
    ("stats_harmonics_mag_6",                      "REAL", "float"),
    ("stats_harmonics_mag_7",                      "REAL", "float"),
    ("stats_harmonics_phase_2",                    "REAL", "float"),
    ("stats_harmonics_phase_3",                    "REAL", "float"),
    ("stats_harmonics_phase_4",                    "REAL", "float"),
    ("stats_harmonics_phase_5",                    "REAL", "float"),
    ("stats_harmonics_phase_6",                    "REAL", "float"),
    ("stats_harmonics_phase_7",                    "REAL", "float"),
    ("stats_harmonics_mse",                        "REAL", "float"),
    ("stats_psi_cs",                               "REAL", "float"),
    ("stats_psi_eta",                              "REAL", "float"),
    # -- stochastic model features --
    ("stats_gp_drw_sigma",                         "REAL", "float"),
    ("stats_gp_drw_tau",                           "REAL", "float"),
    ("stats_iar_phi",                              "REAL", "float"),
    ("stats_mhps_high",                            "REAL", "float"),
    ("stats_mhps_low",                             "REAL", "float"),
    ("stats_mhps_non_zero",                        "REAL", "float"),
    ("stats_mhps_pn_flag",                         "REAL", "float"),
    ("stats_mhps_ratio",                           "REAL", "float"),
    # -- LTV: long-term variability core metrics --
    ("ltv_slope",                    "REAL",    "float"),  # mag/year linear slope
    ("ltv_slope_quad",               "REAL",    "float"),  # quadratic term (mag/yr^2)
    ("ltv_max_diff",                 "REAL",    "float"),  # max seasonal difference (mag)
    ("ltv_dispersion",               "REAL",    "float"),  # peak-to-peak dispersion (mag)
    ("ltv_median",                   "REAL",    "float"),  # median magnitude
    ("ltv_n_seasons",                "INTEGER", "float"),  # number of non-empty seasons
    ("ltv_ls_period",                "REAL",    "float"),  # best LS period (days)
    ("ltv_ls_power",                 "REAL",    "float"),  # LS power at best period
    ("ltv_ls_fap",                   "REAL",    "float"),  # LS false alarm probability
    # -- LTV: filter flags --
    ("ltv_passed_filters",           "INTEGER", "bool"),   # passed all false positive filters
    ("ltv_dust_candidate",           "INTEGER", "bool"),   # dust-driven variability flag
    ("ltv_dust_excess",              "INTEGER", "bool"),   # mid-IR excess flag
    # -- LTV: crossmatch --
    ("ltv_vsx_match",                "INTEGER", "bool"),
    ("ltv_vsx_name",                 "TEXT",    "text"),
    ("ltv_milliquas_match",          "INTEGER", "bool"),   # AGN/QSO flag
    ("ltv_gaia_alert_match",         "INTEGER", "bool"),   # Gaia photometric alert
    # -- LTV: NEOWISE time-series --
    ("ltv_neowise_w1_slope",         "REAL",    "float"),  # NEOWISE W1 trend slope (mag/yr)
    ("ltv_neowise_w1_w2_slope",      "REAL",    "float"),  # NEOWISE W1-W2 color trend slope
    ("ltv_neowise_n_epochs",         "INTEGER", "float"),  # number of NEOWISE epochs
]

# Derived helpers
_COL_NAMES = [c[0] for c in _CANDIDATE_COLUMNS]
_BOOL_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "bool"}
_FLOAT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "float"}
_TEXT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "text"}
_SELECT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "select"}


def get_distinct_values(conn: sqlite3.Connection, column: str) -> list[str]:
    """Return sorted distinct non-empty values for a select-filter column."""
    if column not in _SELECT_COLS and column not in _TEXT_COLS:
        return []
    rows = conn.execute(
        f"SELECT DISTINCT {column} FROM candidates "
        f"WHERE {column} IS NOT NULL AND {column} != '' "
        f"ORDER BY {column}"
    ).fetchall()
    cleaned = set()
    for r in rows:
        val = r[0]
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8", errors="replace")
            except Exception:
                val = str(val)
        val_str = str(val).strip()
        if val_str and val_str not in ("None", "NaN", "nan"):
            cleaned.add(val_str)
    return sorted(list(cleaned))


def init_db(conn: sqlite3.Connection) -> None:
    col_defs = ",\n            ".join(
        f"{col} {dtype}" for col, dtype, _ in _CANDIDATE_COLUMNS
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            source_path TEXT,
            {col_defs},
            payload_json TEXT NOT NULL,
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            candidate_id TEXT PRIMARY KEY,
            interest_score INTEGER,
            event_class TEXT DEFAULT 'unclassified',
            review_pass INTEGER,
            notes TEXT,
            status TEXT,
            reviewer TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            reviewer TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Migrate: add any columns missing from older DBs.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    for col, dtype, _ in _CANDIDATE_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE candidates ADD COLUMN {col} {dtype}")
    conn.commit()


def db_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    init_db(conn)
    return conn


def save_app_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO app_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (key, value, _utc_now()),
    )
    conn.commit()


def load_app_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return default if row is None else str(row[0])


def import_candidates(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source_path: str,
    *,
    characterize_before_import: bool = True,
    characterize_crossmatch: Path = VSX_CROSSMATCH_PATH,
    characterize_chunk_size: int = GAIA_CHUNK_SIZE,
    characterize_cache: Path = GAIA_CACHE_FILE,
    characterize_dust: bool = True,
    characterize_starhorse: str | None = "tap",
    vet_before_import: bool = True,
) -> tuple[int, int]:
    if df.empty:
        return 0, 0

    df_use = df
    if characterize_before_import:
        try:
            df_use = characterize_candidates_df(
                df,
                crossmatch=characterize_crossmatch,
                chunk_size=characterize_chunk_size,
                cache=characterize_cache,
                dust=characterize_dust,
                starhorse=characterize_starhorse,
            )
            if not isinstance(df_use, pd.DataFrame) or df_use.empty:
                df_use = df
        except Exception as e:
            print(f"Warning: characterization before import failed: {e}")
            df_use = df

    if vet_before_import:
        # Auto-detect: skip if vetting columns already populated (e.g. from malca vetting).
        _VET_DETECT_COLS = {"simbad_main_id", "gaia_var_flag", "alerce_oid"}
        _has_vetting = any(
            col in df_use.columns and df_use[col].notna().any()
            for col in _VET_DETECT_COLS
        )
        if _has_vetting:
            print("Vetting: columns already present in input, skipping re-vetting")
            vet_before_import = False

    if vet_before_import:
        try:
            # --- vetting cache: skip candidates already vetted ----
            # Use file-based cache for real paths or fetch:// sources
            if source_path and source_path.startswith("fetch://"):
                _vetting_cache_dir = Path("output") / "cache" / "vetting_cache"
                _vetting_cache_dir.mkdir(parents=True, exist_ok=True)
                _cache_name = source_path.replace("fetch://", "").replace("/", "_") + ".parquet"
                _vetting_cache_path = _vetting_cache_dir / _cache_name
                _use_cache = True
            elif source_path:
                _vetting_cache_path = Path(source_path + ".vetting_cache.parquet")
                _use_cache = True
            else:
                _vetting_cache_path = None
                _use_cache = False
            _cache_df = None
            _id_col = "candidate_id" if "candidate_id" in df_use.columns else None
            n_new = len(df_use)  # default: vet everything

            if _id_col and _vetting_cache_path is not None and _vetting_cache_path.exists():
                try:
                    _cache_df = pd.read_parquet(_vetting_cache_path)
                    cached_ids = set(_cache_df[_id_col])
                    mask_new = ~df_use[_id_col].isin(cached_ids)
                    n_cached = (~mask_new).sum()
                    n_new = mask_new.sum()
                    print(f"Vetting cache: {n_cached} cached, {n_new} to vet")
                except Exception:
                    _cache_df = None

            if _cache_df is not None and n_new == 0:
                # All candidates cached — merge vetting columns from cache
                cache_cols = [c for c in VETTING_COLUMNS if c in _cache_df.columns]
                df_use = df_use.merge(
                    _cache_df[[_id_col] + cache_cols],
                    on=_id_col, how="left", suffixes=("", "_cached"),
                )
                df_use = df_use[[c for c in df_use.columns if not c.endswith("_cached")]]
                print("Vetting: all candidates served from cache")
            else:
                if _cache_df is not None and n_new > 0:
                    # Vet only the new candidates
                    _run_tns = not (source_path and source_path.startswith("fetch://"))
                    df_new = vet_candidates(
                        df_use.loc[mask_new],
                        run_pm_check=False,
                        run_tns=_run_tns,
                        method="xmatch",
                    )
                    # Merge cached vetting columns onto cached rows
                    cache_cols = [c for c in VETTING_COLUMNS if c in _cache_df.columns]
                    df_old = df_use.loc[~mask_new].merge(
                        _cache_df[[_id_col] + cache_cols],
                        on=_id_col, how="left", suffixes=("", "_cached"),
                    )
                    df_old = df_old[[c for c in df_old.columns if not c.endswith("_cached")]]
                    df_use = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    # No cache or no candidate_id — vet everything
                    _run_tns = not (source_path and source_path.startswith("fetch://"))
                    df_use = vet_candidates(
                        df_use,
                        run_pm_check=False,
                        run_tns=_run_tns,
                        method="xmatch",
                    )

                # Update cache
                if _id_col and _vetting_cache_path is not None:
                    try:
                        vet_cols = [c for c in VETTING_COLUMNS if c in df_use.columns]
                        new_cache = df_use[[_id_col] + vet_cols].copy()
                        if _cache_df is not None:
                            new_cache = pd.concat([
                                _cache_df[~_cache_df[_id_col].isin(new_cache[_id_col])],
                                new_cache,
                            ], ignore_index=True)
                        new_cache.to_parquet(_vetting_cache_path, index=False)
                        print(f"Vetting cache saved: {len(new_cache)} entries → {_vetting_cache_path}")
                    except Exception as e:
                        print(f"Warning: failed to save vetting cache: {e}")
        except Exception as e:
            print(f"Warning: vetting before import failed: {e}")

    df_use = df_use.copy()
    df_use["candidate_id"] = infer_candidate_id(df_use)
    imported_at = _utc_now()

    def _opt_str(d, key):
        v = d.get(key)
        return str(v) if v is not None else None

    def _opt_bool(d, key):
        v = d.get(key)
        return int(_as_bool(v)) if v is not None else None

    rows = []
    for _, row in df_use.iterrows():
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        row_dict = normalize_vsx_record(row_dict)

        # Preserve large integer identifiers as non-scientific strings.
        if "gaia_id" in row_dict:
            row_dict["gaia_id"] = _normalize_large_integer_like_id(row_dict.get("gaia_id"))
        if "source_id" in row_dict and row_dict.get("source_id") is not None:
            row_dict["source_id"] = _normalize_large_integer_like_id(row_dict.get("source_id"))

        vals: list = [str(row_dict.get("candidate_id")), source_path]
        for col, _dtype, etype in _CANDIDATE_COLUMNS:
            payload_key = col
            raw = row_dict.get(payload_key)
            if etype == "bool":
                vals.append(_opt_bool(row_dict, payload_key))
            elif etype == "float":
                vals.append(_to_float(raw))
            else:
                vals.append(_opt_str(row_dict, payload_key))
        vals.append(json.dumps(row_dict, default=str))
        vals.append(imported_at)
        rows.append(tuple(vals))

    _all_col_names = ["candidate_id", "source_path"] + _COL_NAMES + ["payload_json", "imported_at"]
    _candidate_cols = ", ".join(_all_col_names)
    _placeholders = ", ".join(["?"] * len(_all_col_names))
    _update_cols = [c for c in _all_col_names if c != "candidate_id"]
    _conflict_set = ", ".join(f"{c}=excluded.{c}" for c in _update_cols)

    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO candidates ({_candidate_cols})
        VALUES ({_placeholders})
        ON CONFLICT(candidate_id) DO UPDATE SET {_conflict_set}
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return len(rows), int(after - before)


def query_queue(conn: sqlite3.Connection, *, filters: dict | None = None) -> pd.DataFrame:
    """Query the candidate queue using filter parameters."""
    if filters is None:
        filters = {}

    where: list[str] = []
    params: list = []

    # --- review status ---
    if filters.get('only_unreviewed'):
        where.append("(r.status IS NULL OR r.status='unreviewed')")

    # --- failed_any shortcut ---
    if filters.get('require_failed_any_false'):
        where.append("(c.failed_any IS NULL OR c.failed_any = 0)")

    # --- optional source-path scoping (exact path) ---
    source_path = filters.get('source_path')
    if source_path:
        where.append("(c.source_path = ?)")
        params.append(str(source_path))

    # --- optional source-path scope token (bundle-like substring) ---
    source_path_like = filters.get('source_path_like')
    if source_path_like:
        where.append("(c.source_path LIKE ?)")
        params.append(f"%{str(source_path_like)}%")

    # --- Any / True / False bool-mode filters (auto-generated) ---
    mode_map = {"Any": None, "True": 1, "False": 0}
    for col in _BOOL_COLS:
        key = f"{col}_mode"
        mode = filters.get(key, "Any")
        val = mode_map.get(mode)
        if val is not None:
            where.append(f"(c.{col} = ?)")
            params.append(val)

    # --- numeric range filters (auto-generated) ---
    # Convention: "min_<col>" → >=, "max_<col>" → <=
    for col in sorted(_FLOAT_COLS):
        for prefix, op in [("min_", ">="), ("max_", "<=")]:
            key = f"{prefix}{col}"
            val = filters.get(key)
            if val is not None:
                where.append(f"(c.{col} IS NOT NULL AND c.{col} {op} ?)")
                params.append(float(val))

    # --- string filters (auto-generated; exact match) ---
    for col in sorted(_TEXT_COLS):
        val = filters.get(col)
        if val and val != "Any":
            val = str(val).strip()
            if val and val != "Any":
                where.append(f"(c.{col} IS NOT NULL AND c.{col} = ?)")
                params.append(val)

    # --- select-exclude filters (multi-value dropdown) ---
    for col in sorted(_SELECT_COLS):
        exc = filters.get(f"exclude_{col}")
        if exc:
            placeholders = ",".join(["?"] * len(exc))
            where.append(f"(c.{col} IS NULL OR c.{col} NOT IN ({placeholders}))")
            params.extend(exc)

    # --- sorting (any float column + review columns, multi-column) ---
    _sortable = {c: f"c.{c}" for c in _FLOAT_COLS}
    _sortable["candidate_id"] = "c.candidate_id"
    _sortable.update({"updated_at": "r.updated_at", "interest_score": "r.interest_score",
                       "review_pass": "r.review_pass"})
    sort_cols = filters.get('sort_cols') or [filters.get('sort_col', 'candidate_id')]
    direction = "DESC" if filters.get('sort_desc') else "ASC"
    order_parts = []
    for sc in sort_cols:
        col_expr = _sortable.get(sc)
        if col_expr:
            order_parts.append(f"{col_expr} {direction}")
    if not order_parts:
        order_parts.append(f"c.candidate_id {direction}")

    query = f"""
        SELECT
            c.candidate_id,
            c.asas_sn_id,
            c.lc_path,
            c.failed_any,
            c.periodic_flag,
            c.catalog_match,
            c.high_ruwe_flag,
            c.periodicity_score,
            c.lsp_bootstrap_sig,
            c.lsp_power,
            c.lsp_period,
            c.dip_best_log_bf,
            c.jump_best_log_bf,
            r.interest_score,
            r.review_pass,
            r.status,
            r.notes,
            r.reviewer,
            r.updated_at
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if where:
        query += " WHERE " + " AND ".join(where)
    order_clause = ", ".join(order_parts)
    if "c.candidate_id" not in order_clause:
        order_clause += ", c.candidate_id ASC"
    query += f" ORDER BY {order_clause}"
    return pd.read_sql_query(query, conn, params=params)


def get_candidate_payload(conn: sqlite3.Connection, candidate_id: str) -> dict:
    row = conn.execute("SELECT payload_json FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


def get_diagnostic_background(conn: sqlite3.Connection) -> dict:
    """Load background arrays for diagnostic plots.

    Returns a dict with keys: cmd_bprp0, cmd_mg0, kiel_teff, kiel_logg,
    ir_hk, ir_w1w2, rpm_bprp, rpm_hg, uv_bprp, uv_nuv_g.
    Values are numpy arrays (may be empty).
    """
    result: dict = {}

    # Kiel: prefer StarHorse teff50/logg50 from payload, fall back to GSP-Phot columns
    rows = conn.execute(
        "SELECT teff_gspphot, logg_gspphot, "
        "       json_extract(payload_json, '$.teff50'), "
        "       json_extract(payload_json, '$.logg50') "
        "FROM candidates"
    ).fetchall()
    teff_list, logg_list = [], []
    for gsp_t, gsp_g, sh_t, sh_g in rows:
        t = sh_t if sh_t is not None else gsp_t
        g = sh_g if sh_g is not None else gsp_g
        if t is not None and g is not None:
            try:
                tf, gf = float(t), float(g)
                if math.isfinite(tf) and math.isfinite(gf):
                    teff_list.append(tf)
                    logg_list.append(gf)
            except (TypeError, ValueError):
                pass
    result["kiel_teff"] = np.array(teff_list, dtype=np.float64)
    result["kiel_logg"] = np.array(logg_list, dtype=np.float64)

    # CMD: mg0 and bprp0 are in payload_json
    rows = conn.execute(
        "SELECT json_extract(payload_json, '$.mg0'), "
        "       json_extract(payload_json, '$.bprp0') "
        "FROM candidates "
        "WHERE json_extract(payload_json, '$.mg0') IS NOT NULL "
        "  AND json_extract(payload_json, '$.bprp0') IS NOT NULL"
    ).fetchall()
    if rows:
        arr = np.array(rows, dtype=np.float64)
        mask = np.isfinite(arr).all(axis=1)
        result["cmd_bprp0"] = arr[mask, 1]
        result["cmd_mg0"] = arr[mask, 0]
    else:
        result["cmd_bprp0"] = np.empty(0)
        result["cmd_mg0"] = np.empty(0)

    # IR color-color: prefer dereddened from payload, fall back to observed
    rows = conn.execute(
        "SELECT tmass_h - tmass_k, unwise_w1 - unwise_w2, "
        "       json_extract(payload_json, '$.H_K_dered'), "
        "       json_extract(payload_json, '$.W1_W2_dered') "
        "FROM candidates "
        "WHERE tmass_h IS NOT NULL AND tmass_k IS NOT NULL "
        "  AND unwise_w1 IS NOT NULL AND unwise_w2 IS NOT NULL"
    ).fetchall()
    hk_list, w1w2_list = [], []
    for hk_obs, w1w2_obs, hk_d, w1w2_d in rows:
        hk = hk_d if hk_d is not None else hk_obs
        w1w2 = w1w2_d if w1w2_d is not None else w1w2_obs
        if hk is not None and w1w2 is not None:
            try:
                hkf, wf = float(hk), float(w1w2)
                if math.isfinite(hkf) and math.isfinite(wf):
                    hk_list.append(hkf)
                    w1w2_list.append(wf)
            except (TypeError, ValueError):
                pass
    result["ir_hk"] = np.array(hk_list, dtype=np.float64)
    result["ir_w1w2"] = np.array(w1w2_list, dtype=np.float64)

    # RPM: H_G = G + 5*log10(pm_arcsec) + 5
    rows = conn.execute(
        "SELECT json_extract(payload_json, '$.phot_g_mean_mag'), "
        "       json_extract(payload_json, '$.bp_rp'), pmra, pmdec "
        "FROM candidates "
        "WHERE json_extract(payload_json, '$.phot_g_mean_mag') IS NOT NULL "
        "  AND json_extract(payload_json, '$.bp_rp') IS NOT NULL "
        "  AND pmra IS NOT NULL AND pmdec IS NOT NULL"
    ).fetchall()
    rpm_bprp_list, rpm_hg_list = [], []
    for g_mag, bprp, pmra, pmdec in rows:
        try:
            g_f, bprp_f = float(g_mag), float(bprp)
            pm_total = math.sqrt(float(pmra) ** 2 + float(pmdec) ** 2)
            if pm_total > 0 and math.isfinite(g_f) and math.isfinite(bprp_f):
                pm_arcsec = pm_total / 1000.0
                h_g = g_f + 5.0 * math.log10(pm_arcsec) + 5.0
                rpm_bprp_list.append(bprp_f)
                rpm_hg_list.append(h_g)
        except (TypeError, ValueError):
            pass
    result["rpm_bprp"] = np.array(rpm_bprp_list, dtype=np.float64)
    result["rpm_hg"] = np.array(rpm_hg_list, dtype=np.float64)

    # UV-Optical: NUV - G vs BP-RP
    rows = conn.execute(
        "SELECT galex_nuv - json_extract(payload_json, '$.phot_g_mean_mag'), "
        "       json_extract(payload_json, '$.bp_rp') "
        "FROM candidates "
        "WHERE galex_nuv IS NOT NULL "
        "  AND json_extract(payload_json, '$.phot_g_mean_mag') IS NOT NULL "
        "  AND json_extract(payload_json, '$.bp_rp') IS NOT NULL"
    ).fetchall()
    if rows:
        arr = np.array(rows, dtype=np.float64)
        mask = np.isfinite(arr).all(axis=1)
        result["uv_nuv_g"] = arr[mask, 0]
        result["uv_bprp"] = arr[mask, 1]
    else:
        result["uv_nuv_g"] = np.empty(0)
        result["uv_bprp"] = np.empty(0)

    return result


VETTING_COLUMNS = [
    "vetting_likely_known",
    "simbad_main_id", "simbad_otype", "simbad_nbref", "simbad_sep_arcsec",
    "gaia_var_flag", "gaia_var_class", "gaia_var_score",
    "gaia_eb_period", "gaia_eb_morph", "gaia_eb_global_ranking",
    "gaia_epoch_available", "gaia_epoch_n_obs", "gaia_epoch_g_range",
    "asassn_var_name", "asassn_var_type", "asassn_var_period",
    "ztf_var_type", "ztf_var_period", "ztf_var_amp",
    "tns_name", "tns_type", "tns_redshift", "tns_disc_date",
    "alerce_oid", "alerce_ndet", "alerce_lc_class", "alerce_lc_prob",
    "alerce_stamp_class", "alerce_stamp_prob",
    "xray_det", "xray_flux", "xray_sep_arcsec",
    "vsx_class", "vsx_sep_arcsec",
    "sfr_name", "sfr_sep_arcmin",
    "cluster_name", "cluster_dist_pc",
    "banyan_best_assoc", "banyan_field_prob",
    "yso_class",
    "iphas_ha_excess",
    "pm_cluster_offset_sigma",
    "atlas_has_phot", "atlas_n_det_cyan", "atlas_n_det_orange",
    "atlas_cyan_range", "atlas_orange_range",
    "neowise_n_epochs", "neowise_w1_range", "neowise_w2_range",
]


def merge_vetting_results(
    conn: sqlite3.Connection,
    vetting_df: pd.DataFrame,
    id_column: str | None = None,
) -> int:
    """Merge vetting results into existing candidate payload_json.

    Matches candidates by candidate_id or asas_sn_id. Updates only
    vetting-related columns in the payload, preserving all other data.

    Returns number of candidates updated.
    """
    if vetting_df.empty:
        return 0

    # Determine ID column
    if id_column is None:
        for col in ("candidate_id", "asas_sn_id"):
            if col in vetting_df.columns:
                id_column = col
                break
    if id_column is None:
        raise ValueError("Vetting DataFrame must have 'candidate_id' or 'asas_sn_id' column")

    # Build lookup: id -> vetting dict
    vetting_cols = [c for c in VETTING_COLUMNS if c in vetting_df.columns]
    if not vetting_cols:
        print("Warning: no vetting columns found in DataFrame")
        return 0

    vetting_df = vetting_df.copy()
    vetting_df[id_column] = vetting_df[id_column].astype(str).str.strip()
    lookup = {}
    for _, row in vetting_df.iterrows():
        cid = row[id_column]
        d = {}
        for col in vetting_cols:
            val = row[col]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                d[col] = val
        if d:
            lookup[cid] = d

    if not lookup:
        return 0

    # Fetch all candidates and update payloads
    rows = conn.execute("SELECT candidate_id, payload_json FROM candidates").fetchall()
    updated = 0
    for cid, payload_json in rows:
        cid_str = str(cid).strip()
        vetting_data = lookup.get(cid_str)
        if vetting_data is None:
            continue

        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}

        payload.update(vetting_data)

        # Update payload JSON
        conn.execute(
            "UPDATE candidates SET payload_json=? WHERE candidate_id=?",
            (json.dumps(payload, default=str), cid),
        )
        # Also update real columns for SQL-filterable vetting fields
        _REAL_VETTING_COLS = {"vetting_likely_known", "asassn_var_type",
                              "gaia_var_class", "simbad_otype", "ztf_var_type"}
        for col in _REAL_VETTING_COLS:
            if col in vetting_data:
                val = vetting_data[col]
                if col == "vetting_likely_known":
                    val = int(bool(val))
                conn.execute(
                    f"UPDATE candidates SET {col}=? WHERE candidate_id=?",
                    (val, cid),
                )
        updated += 1

    conn.commit()
    print(f"Merged vetting data for {updated}/{len(rows)} candidates ({len(vetting_cols)} columns)")
    return updated


def get_review(conn: sqlite3.Connection, candidate_id: str) -> dict:
    row = conn.execute(
        """
        SELECT interest_score, review_pass, notes, status, reviewer, updated_at, event_class
        FROM reviews WHERE candidate_id=?
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        return {
            "interest_score": None,
            "event_class": "unclassified",
            "review_pass": 1,
            "notes": "",
            "status": "unreviewed",
            "reviewer": "",
            "updated_at": None,
        }
    score = None if row[0] is None else int(row[0])
    if score is not None:
        score = int(np.clip(score, 1, 4))
    return {
        "interest_score": score,
        "event_class": str(row[6]) if row[6] else "unclassified",
        "review_pass": 1 if row[1] is None else max(1, int(row[1])),
        "notes": "" if row[2] is None else str(row[2]),
        "status": "unreviewed" if row[3] is None else str(row[3]),
        "reviewer": "" if row[4] is None else str(row[4]),
        "updated_at": row[5],
    }


def save_review(
    conn: sqlite3.Connection,
    *,
    candidate_id: str,
    interest_score: int | None,
    event_class: str = "unclassified",
    review_pass: int,
    notes: str,
    status: str,
    reviewer: str,
    event_type: str = "save",
) -> None:
    ts = _utc_now()
    if interest_score is None:
        score_int = None
    else:
        score_int = int(np.clip(int(interest_score), 1, 4))
    pass_int = max(1, int(review_pass))
    ec = str(event_class) if event_class else "unclassified"
    conn.execute(
        """
        INSERT INTO reviews (candidate_id, interest_score, event_class, review_pass, notes, status, reviewer, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            interest_score=excluded.interest_score,
            event_class=excluded.event_class,
            review_pass=excluded.review_pass,
            notes=excluded.notes,
            status=excluded.status,
            reviewer=excluded.reviewer,
            updated_at=excluded.updated_at
        """,
        (candidate_id, score_int, ec, pass_int, notes, status, reviewer, ts),
    )
    payload = {
        "interest_score": score_int,
        "event_class": ec,
        "review_pass": pass_int,
        "notes": notes,
        "status": status,
        "reviewer": reviewer,
        "updated_at": ts,
    }
    conn.execute(
        """
        INSERT INTO review_history (candidate_id, event_type, payload_json, reviewer, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (candidate_id, event_type, json.dumps(payload, default=str), reviewer, ts),
    )
    conn.commit()


def recent_history(conn: sqlite3.Connection, limit: int = 5) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT candidate_id, event_type, reviewer, created_at
        FROM review_history
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=[int(limit)],
    )


def count_progress(conn: sqlite3.Connection) -> tuple[int, int]:
    total = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    reviewed = conn.execute("SELECT COUNT(*) FROM reviews WHERE status IS NOT NULL AND status != 'unreviewed'").fetchone()[0]
    return int(reviewed), int(total)


def find_plot_image(payload: dict, plot_dir: Path) -> Path | None:
    if not plot_dir.exists():
        return None
    keys = []
    for k in ("candidate_id", "asas_sn_id"):
        if k in payload and payload[k] is not None:
            keys.append(str(payload[k]))
    lc_path = payload.get("path")
    if lc_path:
        keys.append(Path(str(lc_path)).stem)
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    for key in keys:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.pdf"):
            matches = sorted(plot_dir.rglob(f"*{key}*{ext[1:]}"), key=lambda p: str(p))
            if not matches:
                continue
            non_phase = [p for p in matches if "phase" not in p.stem.lower()]
            return non_phase[0] if non_phase else matches[0]
    return None


def find_phase_plot_image(payload: dict, plot_dir: Path) -> Path | None:
    """Locate a phase-folded plot image for a candidate."""
    if not plot_dir.exists():
        return None
    keys = []
    for k in ("candidate_id", "asas_sn_id"):
        if k in payload and payload[k] is not None:
            keys.append(str(payload[k]))
    lc_path = payload.get("path")
    if lc_path:
        keys.append(Path(str(lc_path)).stem)
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    for key in keys:
        phase_patterns = (
            f"*{key}*candidate_phase*.png",
            f"*{key}*phase*.png",
            f"*{key}*candidate_phase*.jpg",
            f"*{key}*phase*.jpg",
            f"*{key}*candidate_phase*.jpeg",
            f"*{key}*phase*.jpeg",
            f"*{key}*candidate_phase*.pdf",
            f"*{key}*phase*.pdf",
        )
        for pattern in phase_patterns:
            matches = sorted(plot_dir.rglob(pattern), key=lambda p: str(p))
            if matches:
                return matches[0]
    return None


def export_reviews(conn: sqlite3.Connection, out_path: Path, only_reviewed: bool = True) -> None:
    candidate_cols = ["candidate_id", "source_path"] + _COL_NAMES
    review_cols = [
        "interest_score",
        "event_class",
        "review_pass",
        "notes",
        "status",
        "reviewer",
        "updated_at",
    ]
    select_cols = [
        *[f"c.{col}" for col in candidate_cols],
        *[f"r.{col}" for col in review_cols],
    ]
    select_clause = ",\n            ".join(select_cols)
    query = f"""
        SELECT
            {select_clause}
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if only_reviewed:
        query += " WHERE r.status IS NOT NULL AND r.status != 'unreviewed'"
    df = pd.read_sql_query(query, conn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False, compression="zstd")
    else:
        df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Raw light-curve file import
# ---------------------------------------------------------------------------
_LC_CACHE_DIR = Path("~/.malca/cache/imported").expanduser()


def import_lightcurve_files(
    conn: sqlite3.Connection,
    file_path: Path,
    *,
    characterize: bool = False,
    vet: bool = False,
) -> tuple[int, int]:
    """Import raw light-curve CSV/parquet files into the review DB.

    If the file has an ``asas_sn_id`` column with multiple unique values,
    each source is split into its own cached CSV.  Otherwise the file is
    treated as a single source and the filename stem is used as candidate_id.

    Returns (n_rows, n_new) like ``import_candidates``.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = load_candidates_file(file_path)
    if df.empty:
        return 0, 0

    _LC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Detect multi-source files
    id_col = None
    for col in ("asas_sn_id", "source_id", "candidate_id"):
        if col in df.columns and df[col].nunique() > 1:
            id_col = col
            break

    if id_col and df[id_col].nunique() > 1:
        # Multi-source: split into individual LC files
        rows = []
        for src_id, sub in df.groupby(id_col):
            cache_file = _LC_CACHE_DIR / f"{src_id}.csv"
            sub.to_csv(cache_file, index=False)
            rows.append({
                "candidate_id": str(src_id),
                "lc_path": str(cache_file),
            })
        candidate_df = pd.DataFrame(rows)
    else:
        # Single-source: copy to cache
        candidate_id = file_path.stem
        cache_file = _LC_CACHE_DIR / file_path.name
        if cache_file != file_path:

            shutil.copy2(file_path, cache_file)
        candidate_df = pd.DataFrame([{
            "candidate_id": candidate_id,
            "lc_path": str(cache_file),
        }])

    return import_candidates(
        conn,
        candidate_df,
        source_path=str(file_path),
        characterize_before_import=characterize,
        vet_before_import=vet,
    )
