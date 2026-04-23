from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import numbers
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote, quote_plus

import pandas as pd


# Grouped metadata fields: list of (group_name, fields) where each field is
# a (label, key) tuple.  Groups are rendered as collapsible sections in the
# Dash app and as headed blocks in the TUI.
REVIEW_METADATA_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Triage Summary", [
        ("Known?", "vetting_likely_known"),
        ("Final class", "final_class"),
        ("YSO class", "yso_class"),
        ("Dipper score", "dipper_score"),
        ("Jumper score", "jumper_score"),
        ("Period consensus (d)", "period_consensus_days"),
        ("Plot ready", "phase_plot_ready"),
        ("Failed any", "failed_any"),
    ]),
    ("LTV Summary", [
        ("Slope (mag/yr)", "ltv_slope"),
        ("Quadratic slope", "ltv_slope_quad"),
        ("Dispersion (mag)", "ltv_dispersion"),
        ("Median (mag)", "ltv_median"),
        ("Median err proxy (mag)", "ltv_median_err"),
        ("Max diff (mag)", "ltv_max_diff"),
        ("N seasons", "ltv_n_seasons"),
        ("Time span (days)", "ltv_time_span_days"),
        ("Unique nights", "ltv_n_unique_nights"),
        ("LS period (d)", "ltv_ls_period"),
        ("LS power", "ltv_ls_power"),
        ("LS FAP", "ltv_ls_fap"),
        ("Has V band", "ltv_vg_has_v"),
        ("V/g overlap (days)", "ltv_vg_overlap_days"),
        ("V/g overlap fraction", "ltv_vg_overlap_fraction"),
        ("Passed filters", "ltv_passed_filters"),
        ("Dust candidate", "ltv_dust_candidate"),
        ("Dust excess", "ltv_dust_excess"),
        ("VSX match", "ltv_vsx_match"),
        ("MILLIQUAS match", "ltv_milliquas_match"),
        ("Gaia alert match", "ltv_gaia_alert_match"),
        ("NEOWISE W1 slope", "ltv_neowise_w1_slope"),
        ("NEOWISE W1-W2 slope", "ltv_neowise_w1_w2_slope"),
        ("NEOWISE epochs", "ltv_neowise_n_epochs"),
    ]),
    ("LTV Season Diagnostics", [
        ("Season points min", "ltv_season_points_min"),
        ("Season points median", "ltv_season_points_median"),
        ("Season points max", "ltv_season_points_max"),
        ("Season span mean (days)", "ltv_season_span_days_mean"),
        ("Season span median (days)", "ltv_season_span_days_median"),
        ("Season span max (days)", "ltv_season_span_days_max"),
        ("Season step max (mag)", "ltv_season_step_max_mag"),
        ("Season step mean abs (mag)", "ltv_season_step_mean_abs_mag"),
        ("Season step max fraction", "ltv_season_step_max_fraction"),
        ("Season monotonicity fraction", "ltv_season_monotonicity_fraction"),
        ("Season Spearman rho", "ltv_season_spearman_rho"),
        ("Season Kendall tau", "ltv_season_kendall_tau"),
        ("Leave-1-out slope std", "ltv_leave1out_slope_std"),
        ("Leave-1-out slope range", "ltv_leave1out_slope_range"),
    ]),
    ("LTV Trend Diagnostics", [
        ("Quadratic coeff1", "ltv_coeff1"),
        ("Quadratic coeff2", "ltv_coeff2"),
        ("Trend slope (mag/yr)", "ltv_trend_slope_mag_per_year"),
        ("Trend quad term", "ltv_trend_quad_mag_per_year2"),
        ("Trend slope err (mag/yr)", "ltv_trend_slope_err_mag_per_year"),
        ("Trend slope SNR", "ltv_trend_slope_snr"),
        ("Trend R^2", "ltv_trend_r2"),
        ("Delta BIC linear", "ltv_trend_delta_bic_linear"),
        ("Delta BIC quadratic", "ltv_trend_delta_bic_quadratic"),
    ]),
    ("LTV Stochastic", [
        ("SF amplitude (mag)", "ltv_stoch_sf_ml_amplitude"),
        ("SF gamma", "ltv_stoch_sf_ml_gamma"),
        ("IAR phi", "ltv_stoch_iar_phi"),
        ("MHPS high", "ltv_stoch_mhps_high"),
        ("MHPS low", "ltv_stoch_mhps_low"),
        ("MHPS non-zero", "ltv_stoch_mhps_non_zero"),
        ("MHPS PN flag", "ltv_stoch_mhps_pn_flag"),
        ("MHPS ratio", "ltv_stoch_mhps_ratio"),
        ("GP-DRW sigma", "ltv_stoch_gp_drw_sigma"),
        ("GP-DRW tau", "ltv_stoch_gp_drw_tau"),
    ]),
    ("Identification", [
        ("ASAS-SN ID", "asas_sn_id"),
        ("Gaia ID", "gaia_id"),
        ("RA", "ra"),
        ("Dec", "dec"),
        ("2MASS ID", "tmass_id"),
        ("AllWISE ID", "allwise_id"),
    ]),
    ("Light Curve Basics", [
        ("N points", "n_points"),
        ("JD first", "jd_first"),
        ("JD last", "jd_last"),
        ("Cadence median (d)", "cadence_median_days"),
        ("N cameras", "n_cameras"),
        ("Camera IDs", "camera_ids"),
        ("Camera min points", "camera_min_points"),
        ("Camera max points", "camera_max_points"),
        ("Baseline mag", "baseline_mag"),
        ("Baseline source", "baseline_source"),
        ("Pre-periodicity label", "pre_periodicity_label"),
        ("Pre-periodic flag", "pre_periodic_flag"),
        ("Selected period (d)", "pre_periodicity_selected_period"),
        ("Selected method", "pre_periodicity_method"),
        ("Phase-peak SNR", "pre_periodicity_phase_peak_snr"),
        ("Phase-peak width", "pre_periodicity_phase_peak_width"),
        ("Phase-peak regions", "pre_periodicity_phase_peak_regions"),
        ("Phase-peak flag", "pre_periodicity_phase_peak_flag"),
    ]),
    ("Event Scoring", [
        ("Dipper N dips", "dipper_n_dips"),
        ("Dipper N valid dips", "dipper_n_valid_dips"),
        ("Jumper N jumps", "jumper_n_jumps"),
        ("Jumper N valid jumps", "jumper_n_valid_jumps"),
    ]),
    ("Dip Detection", [
        ("Dip significant", "dip_significant"),
        ("Dip best morph", "dip_best_morph"),
        ("Dip best log BF", "dip_best_log_bf"),
        ("Dip best delta BIC", "dip_best_delta_bic"),
        ("Dip best width param", "dip_best_width_param"),
        ("Dip symmetry score", "dip_symmetry_score"),
        ("Dip best amp", "dip_best_amp"),
        ("Dip best t0", "dip_best_t0"),
        ("Dip best alpha", "dip_best_alpha"),
        ("Dip best tau", "dip_best_tau"),
        ("Dip Bayes factor", "dip_bayes_factor"),
        ("Dip best p", "dip_best_p"),
        ("Dip best mag event", "dip_best_mag_event"),
        ("Dip trigger max", "dip_trigger_max"),
        ("Dip max event prob", "dip_max_event_prob"),
        ("Dip trigger threshold", "dip_trigger_threshold"),
    ]),
    ("Dip Runs", [
        ("Dip count", "dip_count"),
        ("Dip run count", "dip_run_count"),
        ("Dip max run points", "dip_max_run_points"),
        ("Dip max run duration", "dip_max_run_duration"),
        ("Dip max run sum", "dip_max_run_sum"),
        ("Dip max run max", "dip_max_run_max"),
        ("Dip max run cameras", "dip_max_run_cameras"),
        ("Dip max log BF local", "dip_max_log_bf_local"),
    ]),
    ("Dip Recurrence", [
        ("Dip single event", "dip_is_single_event"),
        ("Dip spacing median", "dip_inter_event_spacing_median"),
        ("Dip spacing std", "dip_inter_event_spacing_std"),
        ("Dip amplitude consistency", "dip_amplitude_consistency"),
        ("Dip duration consistency", "dip_duration_consistency"),
    ]),
    ("Jump Detection", [
        ("Jump significant", "jump_significant"),
        ("Jump best morph", "jump_best_morph"),
        ("Jump best log BF", "jump_best_log_bf"),
        ("Jump best delta BIC", "jump_best_delta_bic"),
        ("Jump best width param", "jump_best_width_param"),
        ("Jump best amp", "jump_best_amp"),
        ("Jump best t0", "jump_best_t0"),
        ("Jump best alpha", "jump_best_alpha"),
        ("Jump best tau", "jump_best_tau"),
        ("Jump Bayes factor", "jump_bayes_factor"),
        ("Jump best p", "jump_best_p"),
        ("Jump best mag event", "jump_best_mag_event"),
        ("Jump trigger max", "jump_trigger_max"),
        ("Jump max event prob", "jump_max_event_prob"),
        ("Jump trigger threshold", "jump_trigger_threshold"),
    ]),
    ("Jump Runs", [
        ("Jump count", "jump_count"),
        ("Jump run count", "jump_run_count"),
        ("Jump max run points", "jump_max_run_points"),
        ("Jump max run duration", "jump_max_run_duration"),
        ("Jump max run sum", "jump_max_run_sum"),
        ("Jump max run max", "jump_max_run_max"),
        ("Jump max run cameras", "jump_max_run_cameras"),
        ("Jump max log BF local", "jump_max_log_bf_local"),
    ]),
    ("Jump Recurrence", [
        ("Jump single event", "jump_is_single_event"),
        ("Jump spacing median", "jump_inter_event_spacing_median"),
        ("Jump spacing std", "jump_inter_event_spacing_std"),
        ("Jump amplitude consistency", "jump_amplitude_consistency"),
        ("Jump duration consistency", "jump_duration_consistency"),
    ]),
    ("Period Consensus", [
        ("Catalog match", "catalog_match"),
        ("Catalog source", "catalog_source"),
        ("Period sources", "period_sources"),
        ("Period N sources", "period_n_sources"),
        ("Period consensus agree", "period_consensus_agree"),
    ]),
    ("Periodicity", [
        ("Periodicity score", "periodicity_score"),
        ("LSP power", "lsp_power"),
        ("LSP period (d)", "lsp_period"),
        ("LSP bootstrap sig", "lsp_bootstrap_sig"),
        ("LSP is alias", "lsp_is_alias"),
        ("LSP is significant", "lsp_is_significant"),
        ("PDM period (d)", "pdm_period"),
        ("PDM theta", "pdm_theta"),
        ("PDM SNR", "pdm_snr"),
        ("CE period (d)", "ce_period"),
        ("CE entropy", "ce_entropy"),
        ("CE SNR", "ce_snr"),
    ]),
    ("Phase Folding", [
        ("Period (d)", "phase_period_days"),
        ("Source", "phase_source"),
        ("Quality score", "phase_quality_score"),
    ]),
    ("Vetting", [
        ("VSX class", "vsx_class"),
        ("VSX period (d)", "vsx_period"),
        ("VSX sep (arcsec)", "vsx_sep_arcsec"),
        ("SIMBAD ID", "simbad_main_id"),
        ("SIMBAD type", "simbad_otype"),
        ("SIMBAD refs", "simbad_nbref"),
        ("SIMBAD sep (\")", "simbad_sep_arcsec"),
        ("Gaia variable", "gaia_var_flag"),
        ("Gaia var class", "gaia_var_class"),
        ("Gaia var score", "gaia_var_score"),
        ("Gaia EB period (d)", "gaia_eb_period"),
        ("Gaia EB morph", "gaia_eb_morph"),
        ("Gaia EB ranking", "gaia_eb_global_ranking"),
        ("Gaia epoch obs", "gaia_epoch_n_obs"),
        ("Gaia G range", "gaia_epoch_g_range"),
        ("ASAS-SN var name", "asassn_var_name"),
        ("ASAS-SN var type", "asassn_var_type"),
        ("ASAS-SN period", "asassn_var_period"),
        ("Microlens match", "microlens_match"),
        ("Microlens catalog", "microlens_catalog"),
        ("Microlens name", "microlens_name"),
        ("Microlens alt name", "microlens_alt_name"),
        ("Microlens t_E (d)", "microlens_te_days"),
        ("Microlens sep (\")", "microlens_sep_arcsec"),
        ("ZTF var type", "ztf_var_type"),
        ("ZTF var period (d)", "ztf_var_period"),
        ("ZTF var amp", "ztf_var_amp"),
        ("TNS name", "tns_name"),
        ("TNS type", "tns_type"),
        ("TNS redshift", "tns_redshift"),
        ("TNS disc date", "tns_disc_date"),
        ("ALeRCE OID", "alerce_oid"),
        ("ALeRCE ndet", "alerce_ndet"),
        ("ALeRCE LC class", "alerce_lc_class"),
        ("ALeRCE LC prob", "alerce_lc_prob"),
        ("ALeRCE stamp class", "alerce_stamp_class"),
        ("ALeRCE stamp prob", "alerce_stamp_prob"),
        ("Gaia epoch avail", "gaia_epoch_available"),
        ("X-ray detected", "xray_det"),
        ("X-ray flux", "xray_flux"),
        ("X-ray sep (\")", "xray_sep_arcsec"),
        ("PM cluster offset", "pm_cluster_offset_sigma"),
    ]),
    ("Stellar Parameters", [
        # -- Gaia DR3 --
        ("G mag", "phot_g_mean_mag"),
        ("BP-RP", "bp_rp"),
        ("BP-RP excess", "bp_rp_excess_corr"),
        ("RUWE", "ruwe"),
        ("High RUWE", "high_ruwe_flag"),
        ("Fidelity", "fidelity"),
        ("RV (km/s)", "radial_velocity"),
        ("RV amp robust (km/s)", "rv_amplitude_robust"),
        ("Teff (GSP-Phot)", "teff_gspphot"),
        ("log g (GSP-Phot)", "logg_gspphot"),
        ("[M/H] (GSP-Phot)", "mh_gspphot"),
        ("Distance (GSP-Phot)", "distance_gspphot"),
        ("Parallax (mas)", "parallax"),
        ("PM RA (mas/yr)", "pmra"),
        ("PM Dec (mas/yr)", "pmdec"),
        ("Total PM (mas/yr)", "pm_total"),
        ("High PM", "high_pm_flag"),
        # -- StarHorse --
        ("SH Distance (kpc)", "dist50"),
        ("SH Teff", "teff50"),
        ("SH log g", "logg50"),
        ("SH [M/H]", "met50"),
        ("SH Mass", "mass50"),
        ("SH Age (Gyr)", "age50"),
        ("SH A_V", "av50"),
        ("SH A_G", "ag50"),
        ("SH A_BP", "abp50"),
        ("SH A_RP", "arp50"),
        ("SH M_G (dered)", "mg0"),
    ]),
    ("Photometry", [
        ("2MASS J", "tmass_j"),
        ("2MASS H", "tmass_h"),
        ("2MASS K", "tmass_k"),
        ("unWISE W1", "unwise_w1"),
        ("unWISE W2", "unwise_w2"),
        ("AllWISE W3", "allwise_w3"),
        ("AllWISE W4", "allwise_w4"),
        ("APASS V", "apass_v"),
        ("APASS B", "apass_b"),
        ("APASS g'", "apass_g"),
        ("APASS r'", "apass_r"),
        ("APASS i'", "apass_i"),
        ("GALEX FUV", "galex_fuv"),
        ("GALEX NUV", "galex_nuv"),
        ("H-K", "H_K"),
        ("H-K (dered)", "H_K_dered"),
        ("W1-W2", "W1_W2"),
        ("W1-W2 (dered)", "W1_W2_dered"),
        ("BP-RP (dered)", "bprp0"),
        ("IPHAS H-alpha", "iphas_ha_mag"),
        ("IPHAS r-Halpha", "iphas_r_ha"),
        ("IPHAS R-I", "iphas_r_i"),
        ("IPHAS H-alpha excess", "iphas_ha_excess"),
        ("unWISE W1 z-score", "unwise_w1_zscore"),
        ("unWISE W2 z-score", "unwise_w2_zscore"),
        ("unWISE W1 var", "unwise_w1_var"),
    ]),
    ("External Follow-up", [
        ("Has spectrum", "has_spectrum"),
        ("Spectrum sources", "spectrum_sources"),
        ("Spectrum links", "spectrum_links"),
        ("ATLAS photometry", "atlas_has_phot"),
        ("ATLAS cyan n", "atlas_n_det_cyan"),
        ("ATLAS orange n", "atlas_n_det_orange"),
        ("ATLAS cyan range", "atlas_cyan_range"),
        ("ATLAS orange range", "atlas_orange_range"),
        ("NEOWISE epochs", "neowise_n_epochs"),
        ("NEOWISE W1 range", "neowise_w1_range"),
        ("NEOWISE W2 range", "neowise_w2_range"),
    ]),
    ("Environment", [
        # -- Galactic position --
        ("Galactic l", "gal_l"),
        ("Galactic b", "gal_b"),
        ("R_gal (kpc)", "rgal"),
        ("X_gal", "xgal"),
        ("Y_gal", "ygal"),
        ("Z_gal", "zgal"),
        # -- Extinction --
        ("A_V (3D)", "A_v_3d"),
        ("E(B-V) (3D)", "ebv_3d"),
        ("Dust sigma", "dust_sigma"),
        ("Dust max dist (kpc)", "dust_max_dist_kpc"),
        # -- Spatial associations --
        ("Population", "population"),
        ("BANYAN field prob", "banyan_field_prob"),
        ("BANYAN best assoc", "banyan_best_assoc"),
        ("SFR name", "sfr_name"),
        ("SFR sep (arcmin)", "sfr_sep_arcmin"),
        ("Near SFR", "near_sfr"),
        ("Cluster name", "cluster_name"),
        ("Cluster dist (pc)", "cluster_dist_pc"),
        ("Cluster age (Myr)", "cluster_age_myr"),
    ]),
    ("YSO / Classification", [
        ("Trigger", "trigger_type"),
        ("P_eb", "P_eb"),
        ("P_cv", "P_cv"),
        ("P_starspot", "P_starspot"),
        ("P_disk", "P_disk"),
        ("a_circ (AU)", "a_circ_au"),
        ("Transit prob", "transit_prob"),
        ("Hill radius (Rsun)", "hill_radius_rsun"),
    ]),
    ("Filter Flags", [
        ("Posterior strength", "failed_posterior_strength"),
        ("Run robustness", "failed_run_robustness"),
        ("Morphology", "failed_morphology"),
        ("Score", "failed_score"),
        ("Periodicity", "failed_periodicity"),
        ("Gaia PM", "failed_gaia_pm"),
        ("Signal amplitude", "failed_signal_amplitude"),
        ("Bad cameras filtered", "bad_cameras_filtered"),
    ]),
]

# Flat list derived from grouped metadata fields.
REVIEW_METADATA_FIELDS: list[tuple[str, str]] = [
    (label, key)
    for _group_name, fields in REVIEW_METADATA_GROUPS
    for label, key in fields
]

# Groups that start expanded in the Dash GUI.
_DEFAULT_OPEN_GROUPS: set[str] = {
    group_name
    for group_name, _fields in REVIEW_METADATA_GROUPS
}


# Maps a value key to its symmetric error key (displayed as "value ± err").
_ERROR_KEYS: dict[str, str] = {
    "parallax": "parallax_error",
    "pmra": "e_PMRA",  # Will need to make sure this is available/renamed if needed
    "pmdec": "e_PMDEC",
    "tmass_j": "tmass_j_err",
    "tmass_h": "tmass_h_err",
    "tmass_k": "tmass_k_err",
    "unwise_w1": "unwise_w1_err",
    "unwise_w2": "unwise_w2_err",
    "allwise_w3": "allwise_w3_err",
    "allwise_w4": "allwise_w4_err",
    "apass_v": "apass_v_err",
    "apass_b": "apass_b_err",
    "apass_g": "apass_g_err",
    "apass_r": "apass_r_err",
    "apass_i": "apass_i_err",
    "galex_fuv": "galex_fuv_err",
    "galex_nuv": "galex_nuv_err",
}

# Maps a value key to its (lo_percentile, hi_percentile) keys
# (displayed as "value (+hi/-lo)" where lo/hi are absolute bounds).
_RANGE_KEYS: dict[str, tuple[str, str]] = {
    "dist50": ("dist16", "dist84"),
    "teff50": ("teff16", "teff84"),
    "logg50": ("logg16", "logg84"),
    "met50": ("met16", "met84"),
    "mass50": ("mass16", "mass84"),
    "age50": ("age16", "age84"),
    "av50": ("av16", "av84"),
}


def normalize_vsx_record(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record)


def normalize_vsx_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "vsx_class" not in out.columns:
        out["vsx_class"] = pd.NA

    if "vsx_sep_arcsec" not in out.columns:
        out["vsx_sep_arcsec"] = pd.NA
    return out


def _is_present(val: Any) -> bool:
    """Return True if *val* should be shown (not None/NaN/empty)."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _format_large_integer_like(val: Any) -> Any:
    """Return non-scientific integer string when *val* is integer-like.

    Gaia IDs often arrive as float/scientific-notation strings (e.g. "4.27e+17").
    For display we normalize those to full integer strings.
    """
    if val is None:
        return val
    s = str(val).strip()
    if not s:
        return val
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return val
    if d != d.to_integral_value():
        return val
    try:
        return format(d.to_integral_value(), "f")
    except Exception:
        return val


def _format_value_for_display(key: str, val: Any) -> Any:
    if key == "gaia_id":
        return _format_large_integer_like(val)
    return val


def _display_label_for_group(group_name: str, label: str) -> str:
    if group_name.startswith("Dip ") and label.startswith("Dip "):
        return label[4:]
    if group_name.startswith("Jump ") and label.startswith("Jump "):
        return label[5:]
    return label


_SIGFIG_DEFAULT = 4
_TINY_DISPLAY_THRESHOLD = 1e-6

_FIXED_DECIMAL_KEYS: dict[str, int] = {
    "ra": 6,
    "dec": 6,
    "gal_l": 5,
    "gal_b": 5,
    "parallax": 4,
    "pmra": 3,
    "pmdec": 3,
    "pm_total": 3,
    "baseline_mag": 3,
    "phot_g_mean_mag": 3,
    "vsx_sep_arcsec": 2,
    "simbad_sep_arcsec": 2,
    "xray_sep_arcsec": 2,
    "sfr_sep_arcmin": 2,
    "phase_period_days": 4,
    "period_consensus_days": 4,
    "vsx_period": 4,
    "gaia_eb_period": 4,
    "asassn_var_period": 4,
    "ztf_var_period": 4,
    "lsp_period": 4,
    "pdm_period": 4,
    "ce_period": 4,
    "period_ogle_days": 4,
}


def _trim_fixed(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _rounded_display_text(key: str, val: Any) -> str:
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, numbers.Integral):
        return str(val)
    if not isinstance(val, numbers.Real):
        return str(val)

    f = float(val)
    if not math.isfinite(f):
        return str(val)
    if abs(f) < _TINY_DISPLAY_THRESHOLD:
        return "0"

    decimals = _FIXED_DECIMAL_KEYS.get(key)
    if decimals is None:
        if ("bayes_factor" in key) or ("log_bf" in key) or (key in {"dipper_score", "jumper_score"}):
            decimals = 1
        elif (
            key.startswith("P_")
            or ("prob" in key)
            or key.endswith("_score")
            or key.endswith("_fraction")
            or key.endswith("_threshold")
            or key.endswith("_fap")
        ):
            return _round_sigfigs(f, n=3)
        elif any(token in key for token in ("cadence", "duration", "spacing", "width", "range", "slope")):
            decimals = 3

    if decimals is not None:
        return _trim_fixed(f, decimals)
    return _round_sigfigs(f, n=_SIGFIG_DEFAULT)


def _round_sigfigs(val: Any, n: int = _SIGFIG_DEFAULT) -> str:
    """Round a numeric value to *n* significant figures for display.

    Only rounds real non-integer values; strings, booleans, and integers are
    returned as-is to avoid mangling IDs or text that happen to look numeric.
    """
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, numbers.Integral):
        return str(val)
    if not isinstance(val, numbers.Real):
        return str(val)
    f = float(val)
    if not math.isfinite(f):
        return str(val)
    if f == 0.0:
        return "0"
    return f"{f:.{n}g}"


def _format_with_uncertainty(
    key: str, val: Any, payload: dict[str, Any], *, round_sf: bool = False,
) -> str:
    """Format a value with its error/range if available."""
    displayed = _format_value_for_display(key, val)

    def _fmt(v: Any) -> str:
        if round_sf:
            return _rounded_display_text(key, v)
        return str(v)

    val_str = _fmt(displayed)

    # Symmetric error: value ± err
    err_key = _ERROR_KEYS.get(key)
    if err_key:
        err_val = payload.get(err_key)
        if _is_present(err_val):
            return f"{val_str} \u00b1 {_fmt(err_val)}"

    # Percentile range: value (+hi/-lo)
    range_keys = _RANGE_KEYS.get(key)
    if range_keys:
        lo_val = payload.get(range_keys[0])
        hi_val = payload.get(range_keys[1])
        if _is_present(lo_val) and _is_present(hi_val):
            try:
                v = float(val)
                lo = float(lo_val)
                hi = float(hi_val)
                plus = hi - v
                minus = v - lo
                if round_sf:
                    return f"{val_str} (+{_fmt(plus)}/-{_fmt(minus)})"
                return f"{val_str} (+{plus:.4g}/-{minus:.4g})"
            except (TypeError, ValueError):
                pass

    return val_str


def extract_review_metadata(
    payload: dict[str, Any],
    *,
    round_sigfigs: bool = False,
) -> list[tuple[str, Any]]:
    """Return a flat list of ``(label, value)`` pairs for display.

    This is the original flat interface used by callers that don't need
    group structure (e.g. ``plot_batch.py``).
    """
    p = normalize_vsx_record(payload)
    rows: list[tuple[str, Any]] = []
    for label, key in REVIEW_METADATA_FIELDS:
        val = p.get(key)
        if not _is_present(val):
            continue
        rows.append((label, _format_with_uncertainty(key, val, p, round_sf=round_sigfigs)))
    return rows


def extract_review_metadata_grouped(
    payload: dict[str, Any],
    *,
    round_sigfigs: bool = False,
) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Return metadata organised by group.

    Returns a list of ``(group_name, items)`` tuples where *items* is a
    list of ``(label, value)`` pairs.  Groups with no present values are
    omitted.
    """
    p = normalize_vsx_record(payload)
    groups: list[tuple[str, list[tuple[str, Any]]]] = []
    for group_name, fields in REVIEW_METADATA_GROUPS:
        items: list[tuple[str, Any]] = []
        for label, key in fields:
            val = p.get(key)
            if not _is_present(val):
                continue
            items.append((
                _display_label_for_group(group_name, label),
                _format_with_uncertainty(key, val, p, round_sf=round_sigfigs),
            ))
        if items:
            groups.append((group_name, items))
    return groups


def is_group_default_open(group_name: str) -> bool:
    """Whether *group_name* should start expanded in the Dash GUI."""
    return group_name in _DEFAULT_OPEN_GROUPS


# ---------------------------------------------------------------------------
# External lookup URL construction
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Return *val* as a float if finite, else ``None``."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _safe_str(val: Any) -> str | None:
    """Return *val* as a non-empty stripped string, or ``None``."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _jd_to_mpc_iso(jd: float) -> str | None:
    """Convert JD to ISO format YYYY-MM-DD HH:MM:SS (UTC) for MPChecker."""
    if not math.isfinite(jd):
        return None
    try:
        # JD -> MJD -> datetime
        # JD starts at noon, so subtract 0.5 to get to midnight-based MJD?
        # Standard: MJD = JD - 2400000.5
        mjd = jd - 2400000.5
        epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
        dt = epoch + timedelta(days=mjd)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def build_external_lookup_links(
    payload: dict[str, Any],
    radius_arcsec: float = 10.0,
) -> list[tuple[str, str]]:
    """Build ``(label, url)`` pairs for external astronomical services.

    Uses identifier-based URLs when a catalogue ID is available, falling
    back to coordinate-based (RA/Dec) cone searches otherwise.  Links
    whose required fields are missing are silently omitted.
    """
    links: list[tuple[str, str]] = []

    ra = _safe_float(payload.get("ra"))
    if ra is None:
        ra = _safe_float(payload.get("ra_deg"))

    dec = _safe_float(payload.get("dec"))
    if dec is None:
        dec = _safe_float(payload.get("dec_deg"))

    has_coords = ra is not None and dec is not None

    # --- Aggregators ---
    # -- SIMBAD ---------------------------------------------------------------
    simbad_id = _safe_str(payload.get("simbad_main_id"))
    if simbad_id:
        links.append((
            "SIMBAD",
            f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={quote_plus(simbad_id)}",
        ))
    elif has_coords:
        links.append((
            "SIMBAD",
            f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}+{dec}&CooFrame=FK5&CooEpoch=2000&Radius={radius_arcsec}&Radius.unit=arcsec",
        ))

    # -- NED (IPAC) -----------------------------------------------------------
    if has_coords:
        radius_arcmin = radius_arcsec / 60.0
        links.append((
            "NED",
            f"https://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search&in_csys=Equatorial&in_equinox=J2000.0&lon={ra}d&lat={dec}d&radius={radius_arcmin}",
        ))

    # -- ADS ------------------------------------------------------------------
    if simbad_id:
        links.append((
            "ADS",
            f"https://ui.adsabs.harvard.edu/search/q=object%3A%22{quote(simbad_id)}%22",
        ))
    elif has_coords:
        links.append((
            "ADS",
            f"https://ui.adsabs.harvard.edu/search/q=pos(%22{ra}+{dec}%22%2C+{radius_arcsec}s)",
        ))

    # -- TNS ------------------------------------------------------------------
    tns_name = _safe_str(payload.get("tns_name"))
    if tns_name:
        # TNS names are like "AT2021abc" or "SN2020xyz"; strip leading
        # prefixes that the URL path doesn't expect.
        tns_slug = tns_name.strip()
        links.append((
            "TNS",
            f"https://www.wis-tns.org/object/{quote(tns_slug)}",
        ))
    elif has_coords:
        links.append((
            "TNS",
            f"https://www.wis-tns.org/search?ra={ra}&decl={dec}&radius={radius_arcsec}&coords_unit=arcsec",
        ))

    # -- AAVSO VSX ------------------------------------------------------------
    if has_coords:
        links.append((
            "VSX",
            f"https://www.aavso.org/vsx/index.php?view=results.get&coords={ra}+{dec}&format=d&num=1",
        ))

    # -- ALeRCE Explorer ------------------------------------------------------
    alerce_oid = _safe_str(payload.get("alerce_oid"))
    if alerce_oid:
        links.append((
            "ALeRCE",
            f"https://alerce.online/object/{quote(alerce_oid)}",
        ))


    # --- Catalogs ---
    # -- VizieR ---------------------------------------------------------------
    if has_coords:
        links.append((
            "VizieR",
            f"https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-c={ra}+{dec}&-c.rs={radius_arcsec}",
        ))

    # -- MPChecker (MPC) ------------------------------------------------------
    if has_coords:
        # Prefer dip/jump event time, then first observation
        event_time = _safe_float(payload.get("dip_best_t0"))
        if event_time is None:
            event_time = _safe_float(payload.get("jump_best_t0"))
        if event_time is None:
            event_time = _safe_float(payload.get("jd_first"))

        iso_date = _jd_to_mpc_iso(event_time) if event_time else None
        if iso_date:
            links.append((
                "MPChecker",
                f"https://minorplanetcenter.net/cgi-bin/checkmp.cgi?ra={ra}&decl={dec}&radius={radius_arcsec}&iso={quote(iso_date)}",
            ))


    # --- Visualizers / Surveys ---
    # -- ASAS-SN (Variable Stars Database) ------------------------------------
    if has_coords:
        radius_arcmin = radius_arcsec / 60.0
        links.append((
            "ASAS-SN",
            f"https://asas-sn.osu.edu/variables?ra={ra}&dec={dec}&radius={radius_arcmin}",
        ))

    # -- ZTF / IRSA -----------------------------------------------------------
    if has_coords:
        links.append((
            "ZTF",
            f"https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?catalog=ztf_objects_dr22&objstr={ra}+{dec}&radius={radius_arcsec}&radunits=arcsec",
        ))

    # -- DSS (STScI) ----------------------------------------------------------
    if has_coords:
        radius_arcmin = radius_arcsec / 60.0
        # h/w are in arcminutes
        links.append((
            "DSS",
            f"https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r={ra}&d={dec}&e=J2000&h={radius_arcmin * 2}&w={radius_arcmin * 2}&f=gif&c=none",
        ))

    # -- DECaLS (Legacy Survey) -----------------------------------------------
    if has_coords:
        links.append((
            "DECaLS",
            f"https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&zoom=14&layer=ls-dr10",
        ))

    # -- Aladin Sky Atlas -----------------------------------------------------
    if has_coords:
        # Use a modest default field of view around the target while scaling
        # with the requested search radius.
        fov_deg = max(0.03, min(2.0, (radius_arcsec * 2.0) / 3600.0))
        links.append((
            "Aladin",
            f"https://aladin.cds.unistra.fr/AladinLite/?target={quote_plus(f'{ra} {dec}')}&fov={fov_deg:.4f}&survey=P%2FDSS2%2Fcolor",
        ))

    # -- SDSS (SkyServer) -----------------------------------------------------
    if has_coords:
        # scale is roughly 0.396 sec/pixel for SDSS. Width/height of 500 max.
        width_pixels = min(1000, max(200, int((radius_arcsec * 2) / 0.396)))
        links.append((
            "SDSS",
            f"http://skyserver.sdss.org/dr17/en/tools/chart/navi.aspx?ra={ra}&dec={dec}&scale=0.396&width={width_pixels}&height={width_pixels}",
        ))

    return links
