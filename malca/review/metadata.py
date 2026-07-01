from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import numbers
import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote, quote_plus

import pandas as pd

from malca.ltv.multi_survey import LTV_MS_FEATURE_COLUMN_SPECS
from malca.review.filter_schema import is_known_variable_type_value


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
    ("Nuclear Summary", [
        ("Gaia star veto", "gaia_stellar_veto_score"),
        ("Gaia extragalactic prior", "gaia_extragalactic_prior_score"),
        ("Host nuclear score", "host_nuclear_score"),
        ("Nuclear offset (arcsec)", "nuclear_offset_arcsec"),
        ("AGN prior score", "agn_prior_score"),
        ("AGN reasons", "agn_prior_reasons"),
        ("TDE score", "tde_candidate_score"),
        ("TDE reasons", "tde_candidate_reasons"),
        ("CLAGN score", "clagn_photometric_score"),
        ("CLAGN reasons", "clagn_reasons"),
        ("Redshift", "redshift"),
        ("Redshift source", "redshift_source"),
        ("Spectral class", "host_spectral_class"),
        ("Broad-line flag", "broad_line_flag"),
        ("WISE AGN score", "wise_agn_score"),
        ("NEOWISE variability", "neowise_variability_score"),
        ("X-ray prior", "xray_agn_prior_score"),
        ("Radio prior", "radio_agn_prior_score"),
        ("UV/TDE support", "uv_tde_score"),
        ("Known CLAGN", "known_clagn_match"),
    ]),
    ("LTV Summary", [
        ("Slope (mag/yr)", "ltv_slope"),
        ("Quadratic slope", "ltv_slope_quad"),
        ("Dispersion (mag)", "ltv_dispersion"),
        ("Median (mag)", "ltv_median"),
        ("Median err proxy (mag)", "ltv_median_err"),
        ("Max diff (mag)", "ltv_max_diff"),
        ("N seasons", "ltv_n_seasons"),
        ("Time span (days)", "time_span_days"),
        ("Unique nights", "n_unique_nights"),
        ("LS period (d)", "ltv_ls_period"),
        ("LS power", "ltv_ls_power"),
        ("LS FAP", "ltv_ls_fap"),
        ("Has V band", "ltv_vg_has_v"),
        ("V/g overlap (days)", "ltv_vg_overlap_days"),
        ("V/g overlap fraction", "ltv_vg_overlap_fraction"),
        ("Failed any", "failed_any"),
        ("Filter reason", "filter_reason"),
        ("LTV class", "ltv_class"),
        ("LTV class reason", "ltv_class_reason"),
        ("LTV interest score", "ltv_interest_score"),
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
    ("LTV Long-Term Features", [
        ("Smooth P95-P5 (mag)", "ltv_smooth_p95_p5"),
        ("Smooth variance", "ltv_smooth_var"),
        ("Residual variance", "ltv_resid_var"),
        ("Long/short variance ratio", "ltv_long_short_var_ratio"),
        ("Smooth N points", "ltv_smooth_n_points"),
        ("Smooth 100d P95-P5 (mag)", "ltv_smooth_100d_p95_p5"),
        ("Smooth 100d variance", "ltv_smooth_100d_smooth_var"),
        ("Residual 100d variance", "ltv_smooth_100d_resid_var"),
        ("Long/short variance ratio 100d", "ltv_smooth_100d_long_short_var_ratio"),
        ("Smooth 100d N points", "ltv_smooth_100d_n_points"),
        ("Smooth 300d P95-P5 (mag)", "ltv_smooth_300d_p95_p5"),
        ("Smooth 300d variance", "ltv_smooth_300d_smooth_var"),
        ("Residual 300d variance", "ltv_smooth_300d_resid_var"),
        ("Long/short variance ratio 300d", "ltv_smooth_300d_long_short_var_ratio"),
        ("Smooth 300d N points", "ltv_smooth_300d_n_points"),
        ("Smooth 1000d P95-P5 (mag)", "ltv_smooth_1000d_p95_p5"),
        ("Smooth 1000d variance", "ltv_smooth_1000d_smooth_var"),
        ("Residual 1000d variance", "ltv_smooth_1000d_resid_var"),
        ("Long/short variance ratio 1000d", "ltv_smooth_1000d_long_short_var_ratio"),
        ("Smooth 1000d N points", "ltv_smooth_1000d_n_points"),
        ("Binned SF N bins", "ltv_binned_sf_n_bins"),
        ("Binned SF 30d", "ltv_binned_sf_30d_mag2"),
        ("Binned SF 100d", "ltv_binned_sf_100d_mag2"),
        ("Binned SF 300d", "ltv_binned_sf_300d_mag2"),
        ("Binned SF 1000d", "ltv_binned_sf_1000d_mag2"),
        ("Binned SF 3000d", "ltv_binned_sf_3000d_mag2"),
        ("Binned SF 300d/30d", "ltv_binned_sf_300d_30d_ratio"),
        ("Binned SF 1000d/30d", "ltv_binned_sf_1000d_30d_ratio"),
        ("Binned SF 3000d/30d", "ltv_binned_sf_3000d_30d_ratio"),
        ("Binned SF slope", "ltv_binned_sf_slope"),
        ("Theil-Sen slope (mag/yr)", "ltv_theil_sen_slope_mag_per_year"),
        ("Theil-Sen intercept", "ltv_theil_sen_intercept_mag"),
        ("Theil-Sen low slope", "ltv_theil_sen_low_slope_mag_per_year"),
        ("Theil-Sen high slope", "ltv_theil_sen_high_slope_mag_per_year"),
        ("Bayesian Blocks N blocks", "ltv_bb_n_blocks"),
        ("Bayesian Blocks change points", "ltv_bb_n_change_points"),
        ("Bayesian Blocks range (mag)", "ltv_bb_range_mag"),
        ("Bayesian Blocks largest jump (mag)", "ltv_bb_largest_jump_mag"),
        ("Bayesian Blocks max offset (mag)", "ltv_bb_max_block_offset_mag"),
        ("LOWESS P95-P5 (mag)", "ltv_lowess_p95_p5"),
        ("LOWESS residual std", "ltv_lowess_resid_std"),
        ("LOWESS max abs residual", "ltv_lowess_max_abs_resid"),
        ("Variogram short lag", "ltv_variogram_short_mag2"),
        ("Variogram mid lag", "ltv_variogram_mid_mag2"),
        ("Variogram long lag", "ltv_variogram_long_mag2"),
        ("Variogram long/short ratio", "ltv_variogram_long_short_ratio"),
        ("Variogram slope", "ltv_variogram_slope"),
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
    ("LTV Multi-Survey", [
        (col.replace("ltv_ms_", "").replace("_", " "), col)
        for col, _sql, _kind in LTV_MS_FEATURE_COLUMN_SPECS
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
        ("ASAS-SN field", "asassn_field_key"),
        ("ASAS-SN fields", "asassn_fields"),
        ("ASAS-SN field count", "asassn_field_count"),
        ("ASAS-SN field fraction", "asassn_field_key_fraction"),
        ("Camera name", "camera_name_key"),
        ("Camera names", "camera_names"),
        ("Camera name count", "camera_name_count"),
        ("Camera name fraction", "camera_name_key_fraction"),
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
        ("Phase lag g-V", "pre_periodicity_phase_lag_g_v_cycles"),
        ("Phase lag g-V abs", "pre_periodicity_phase_lag_g_v_abs_cycles"),
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
    ("Microlensing Fits", [
        ("Fit status", "parallax_status"),
        ("Warning", "parallax_warning"),
        ("Attempted", "parallax_attempted"),
        ("Preferred", "parallax_preferred"),
        ("Delta BIC", "parallax_delta_bic"),
        ("Best branch", "parallax_best_branch"),
        ("t_E (d)", "parallax_best_tE_days"),
        ("u_0", "parallax_best_u0"),
        ("pi_E", "parallax_best_piE"),
        ("pi_E North", "parallax_best_piE_N"),
        ("pi_E East", "parallax_best_piE_E"),
        ("f_s (blend)", "parallax_best_fs"),
        ("f_b (bg)", "parallax_best_fb"),
        ("Chi^2", "parallax_best_chi2"),
        ("Red. Chi^2", "parallax_best_reduced_chi2"),
        ("BIC", "parallax_best_bic"),
        ("PSPL Chi^2", "parallax_pspl_flux_chi2"),
        ("PSPL BIC", "parallax_pspl_flux_bic"),
    ]),
    ("Period Consensus", [
        ("Catalog match", "catalog_match"),
        ("Catalog source", "catalog_source"),
        ("Period sources", "period_sources"),
        ("Period N sources", "period_n_sources"),
        ("Period consensus agree", "period_consensus_agree"),
        ("OGLE name", "period_ogle_name"),
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
        ("eROSITA detected", "erosita_det"),
        ("eROSITA flux", "erosita_flux"),
        ("eROSITA sep (\")", "erosita_sep_arcsec"),
        ("Chandra detected", "chandra_det"),
        ("Chandra source", "chandra_source_id"),
        ("Chandra flux 0.5-7", "chandra_flux_05_7"),
        ("Chandra broad flux", "chandra_flux_broad"),
        ("Chandra significance", "chandra_significance"),
        ("Chandra likelihood", "chandra_likelihood"),
        ("Chandra likelihood class", "chandra_likelihood_class"),
        ("Chandra pos err major (\")", "chandra_pos_err_maj_arcsec"),
        ("Chandra pos err minor (\")", "chandra_pos_err_min_arcsec"),
        ("Chandra pos err PA", "chandra_pos_err_pa_deg"),
        ("Chandra extended", "chandra_extended_flag"),
        ("Chandra variable", "chandra_variable_flag"),
        ("Chandra sep (\")", "chandra_sep_arcsec"),
        ("X-ray detected", "xray_det"),
        ("X-ray flux", "xray_flux"),
        ("X-ray sep (\")", "xray_sep_arcsec"),
        ("X-ray catalogs", "xray_source_catalogs"),
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
        ("M_G0 (dustmaps3d)", "mg0"),
    ]),
    ("Photometry", [
        ("2MASS J", "tmass_j"),
        ("2MASS H", "tmass_h"),
        ("2MASS K", "tmass_k"),
        ("WISE W1", "w1"),
        ("WISE W2", "w2"),
        ("WISE W3", "w3"),
        ("WISE W4", "w4"),
        ("APASS V", "apass_v"),
        ("APASS B", "apass_b"),
        ("APASS g'", "apass_g"),
        ("APASS r'", "apass_r"),
        ("APASS i'", "apass_i"),
        ("GALEX FUV", "galex_fuv"),
        ("GALEX NUV", "galex_nuv"),
        ("H-K", "H_K"),
        ("H-K (dered)", "H_K_dered"),
        ("W1-W2", "w1_w2"),
        ("W1-W2 (dered)", "w1_w2_dered"),
        ("W1-W3", "w1_w3"),
        ("W1-W4", "w1_w4"),
        ("W2-W3", "w2_w3"),
        ("W2-W4", "w2_w4"),
        ("W3-W4", "w3_w4"),
        ("SED alpha", "sed_alpha"),
        ("SED alpha class", "sed_alpha_class"),
        ("SED alpha points", "sed_alpha_n_points"),
        ("SED alpha lambda min", "sed_alpha_lambda_min_micron"),
        ("SED alpha lambda max", "sed_alpha_lambda_max_micron"),
        ("SED alpha status", "sed_alpha_status"),
        ("BP-RP0 (dustmaps3d)", "bprp0"),
        ("IPHAS r", "iphas_r_mag"),
        ("IPHAS r err", "iphas_r_err"),
        ("IPHAS i", "iphas_i_mag"),
        ("IPHAS i err", "iphas_i_err"),
        ("IPHAS H-alpha", "iphas_ha_mag"),
        ("IPHAS H-alpha err", "iphas_ha_err"),
        ("IPHAS r-Halpha", "iphas_r_ha"),
        ("IPHAS r-Halpha err", "iphas_r_ha_err"),
        ("IPHAS r-i", "iphas_r_i"),
        ("IPHAS r-i err", "iphas_r_i_err"),
        ("IPHAS sep", "iphas_sep_arcsec"),
        ("IPHAS source", "iphas_source_catalog"),
        ("IPHAS H-alpha excess", "iphas_ha_excess"),
        ("VPHAS+ r", "vphas_r_mag"),
        ("VPHAS+ r err", "vphas_r_err"),
        ("VPHAS+ i", "vphas_i_mag"),
        ("VPHAS+ i err", "vphas_i_err"),
        ("VPHAS+ H-alpha", "vphas_ha_mag"),
        ("VPHAS+ H-alpha err", "vphas_ha_err"),
        ("VPHAS+ r-Halpha", "vphas_r_ha"),
        ("VPHAS+ r-Halpha err", "vphas_r_ha_err"),
        ("VPHAS+ r-i", "vphas_r_i"),
        ("VPHAS+ r-i err", "vphas_r_i_err"),
        ("VPHAS+ sep", "vphas_sep_arcsec"),
        ("VPHAS+ source", "vphas_source_catalog"),
        ("VPHAS+ H-alpha excess", "vphas_ha_excess"),
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
        ("Kepler quarters", "kepler_n_quarters"),
        ("Kepler points", "kepler_total_points"),
        ("Kepler flux range", "kepler_flux_range"),
        ("AAVSO points", "aavso_lc_n_points"),
        ("OGLE points", "ogle_lc_n_points"),
        ("OGLE I range", "ogle_lc_i_range"),
        ("OGLE V range", "ogle_lc_v_range"),
        ("Stripe 82 points", "stripe82_lc_n_points"),
        ("Stripe 82 u range", "stripe82_lc_u_range"),
        ("Stripe 82 g range", "stripe82_lc_g_range"),
        ("Stripe 82 r range", "stripe82_lc_r_range"),
        ("Stripe 82 i range", "stripe82_lc_i_range"),
        ("Stripe 82 z range", "stripe82_lc_z_range"),
        ("AllWISE MEP epochs", "allwise_mep_n_epochs"),
        ("AllWISE W1 range", "allwise_mep_w1_range"),
        ("AllWISE W2 range", "allwise_mep_w2_range"),
        ("AllWISE W3 range", "allwise_mep_w3_range"),
        ("AllWISE W4 range", "allwise_mep_w4_range"),
        ("VVVX/VIRAC epochs", "vvvx_virac_n_epochs"),
        ("VVVX Z range", "vvvx_virac_z_range"),
        ("VVVX Y range", "vvvx_virac_y_range"),
        ("VVVX J range", "vvvx_virac_j_range"),
        ("VVVX H range", "vvvx_virac_h_range"),
        ("VVVX Ks range", "vvvx_virac_ks_range"),
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

_PRESENTATION_SECTION_BY_GROUP: dict[str, str] = {
    "Triage Summary": "Review Summary",
    "Nuclear Summary": "Catalog & Vetting",
    "LTV Summary": "Coverage & Photometry",
    "LTV Season Diagnostics": "Advanced Metadata",
    "LTV Trend Diagnostics": "Advanced Metadata",
    "LTV Long-Term Features": "Advanced Metadata",
    "LTV Stochastic": "Advanced Metadata",
    "LTV Multi-Survey": "Advanced Metadata",
    "Identification": "Coverage & Photometry",
    "Light Curve Basics": "Coverage & Photometry",
    "Event Scoring": "Review Summary",
    "Dip Detection": "Dip Evidence",
    "Dip Runs": "Dip Evidence",
    "Dip Recurrence": "Dip Evidence",
    "Jump Detection": "Jump Evidence",
    "Jump Runs": "Jump Evidence",
    "Jump Recurrence": "Jump Evidence",
    "Microlensing Fits": "Catalog & Vetting",
    "Period Consensus": "Catalog & Vetting",
    "Periodicity": "Catalog & Vetting",
    "Phase Folding": "Catalog & Vetting",
    "Vetting": "Catalog & Vetting",
    "Stellar Parameters": "Classification & Environment",
    "Photometry": "Classification & Environment",
    "External Follow-up": "Catalog & Vetting",
    "Environment": "Classification & Environment",
    "YSO / Classification": "Classification & Environment",
    "Filter Flags": "Review Summary",
}

_PRESENTATION_ROLE_BY_KEY: dict[str, str] = {
    "vetting_likely_known": "summary",
    "final_class": "summary",
    "yso_class": "summary",
    "dipper_score": "summary",
    "jumper_score": "summary",
    "phase_plot_ready": "summary",
    "failed_any": "summary",
    "catalog_match": "summary",
    "period_consensus_agree": "summary",
    "asas_sn_id": "summary",
    "gaia_id": "summary",
    "n_points": "summary",
    "time_span_days": "summary",
    "cadence_median_days": "summary",
    "n_cameras": "summary",
    "baseline_mag": "summary",
    "baseline_source": "summary",
    "gaia_var_flag": "summary",
    "xray_det": "summary",
    "high_ruwe_flag": "summary",
    "bad_cameras_filtered": "summary",
    "failed_periodicity": "summary",
}

for _key in {
    "dipper_n_dips",
    "dipper_n_valid_dips",
    "dip_significant",
    "dip_best_morph",
    "dip_best_log_bf",
    "dip_best_delta_bic",
    "dip_bayes_factor",
    "dip_best_p",
    "dip_best_mag_event",
    "dip_trigger_max",
    "dip_max_event_prob",
    "dip_trigger_threshold",
    "dip_count",
    "dip_run_count",
    "dip_max_run_points",
    "dip_max_run_duration",
    "dip_max_run_sum",
    "dip_max_run_max",
    "dip_max_run_cameras",
    "dip_max_log_bf_local",
    "dip_is_single_event",
    "dip_inter_event_spacing_median",
    "dip_inter_event_spacing_std",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
}:
    _PRESENTATION_ROLE_BY_KEY.setdefault(_key, "dip")

for _key in {
    "jumper_n_jumps",
    "jumper_n_valid_jumps",
    "jump_significant",
    "jump_best_morph",
    "jump_best_log_bf",
    "jump_best_delta_bic",
    "jump_bayes_factor",
    "jump_best_p",
    "jump_best_mag_event",
    "jump_trigger_max",
    "jump_max_event_prob",
    "jump_trigger_threshold",
    "jump_count",
    "jump_run_count",
    "jump_max_run_points",
    "jump_max_run_duration",
    "jump_max_run_sum",
    "jump_max_run_max",
    "jump_max_run_cameras",
    "jump_max_log_bf_local",
    "jump_is_single_event",
    "jump_inter_event_spacing_median",
    "jump_inter_event_spacing_std",
    "jump_amplitude_consistency",
    "jump_duration_consistency",
}:
    _PRESENTATION_ROLE_BY_KEY.setdefault(_key, "jump")

del _key

_DISPLAY_UNIT_LABELS = {
    '"': "arcsec",
    "AU": "AU",
    "Gyr": "Gyr",
    "K": "K",
    "Myr": "Myr",
    "Rsun": "Rsun",
    r"$\mathrm{mag}^2$": r"$\mathrm{mag}^2$",
    "arcmin": "arcmin",
    "arcsec": "arcsec",
    "cgs": "cgs",
    "d": "d",
    "day": "day",
    "days": "days",
    "kpc": "kpc",
    "km/s": "km/s",
    "mag": "mag",
    "mag/day": "mag/day",
    "mag/year": "mag/year",
    "mag/yr": "mag/yr",
    "mag/yr^2": "mag/yr^2",
    "mas": "mas",
    "mas/yr": "mas/yr",
    "pc": "pc",
    "rad": "rad",
}


def bracket_unit_label(label: str) -> str:
    """Use square brackets for unit parentheticals in display labels."""
    text = str(label)

    def _replace(match: re.Match[str]) -> str:
        unit = match.group(1).strip()
        display = _DISPLAY_UNIT_LABELS.get(unit)
        if display is None:
            return match.group(0)
        return f"[{display}]"

    return re.sub(r"\(([^()]*)\)", _replace, text)


def markdown_literal_unit_label(label: str) -> str:
    """Return a unit-bracketed label safe for Dash Markdown rendering."""
    text = bracket_unit_label(label)
    return text.replace("[", r"\[").replace("]", r"\]")

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
    "w1": "w1_err",
    "w2": "w2_err",
    "w3": "w3_err",
    "w4": "w4_err",
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


def _known_text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "<na>", "none", "null"}:
        return ""
    return text


def _truthy_catalog_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Number):
        return float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _finite_catalog_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _is_variable_simbad_otype(value: Any) -> bool:
    return is_known_variable_type_value("simbad_otype", value)


def has_known_catalog_evidence(record: dict[str, Any] | None) -> bool:
    """Return whether payload/catalog fields identify an already known object."""
    if not isinstance(record, dict):
        return False
    if _truthy_catalog_value(record.get("vetting_likely_known")):
        return True
    for column in (
        "vsx_class",
        "asassn_var_type",
        "gaia_var_class",
        "ztf_var_type",
        "tns_type",
        "alerce_lc_class",
        "microlens_catalog",
    ):
        if is_known_variable_type_value(column, record.get(column)):
            return True
    if _truthy_catalog_value(record.get("microlens_match")):
        return True
    if _known_text_value(record.get("tns_name")):
        return True
    if _finite_catalog_number(record.get("gaia_eb_period")):
        return True
    return _is_variable_simbad_otype(record.get("simbad_otype"))


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
    "erosita_sep_arcsec": 2,
    "chandra_sep_arcsec": 2,
    "chandra_pos_err_maj_arcsec": 2,
    "chandra_pos_err_min_arcsec": 2,
    "chandra_pos_err_pa_deg": 2,
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
        rows.append((bracket_unit_label(label), _format_with_uncertainty(key, val, p, round_sf=round_sigfigs)))
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
                bracket_unit_label(_display_label_for_group(group_name, label)),
                _format_with_uncertainty(key, val, p, round_sf=round_sigfigs),
            ))
        if items:
            groups.append((group_name, items))
    return groups


def metadata_presentation_section(group_name: str, key: str) -> str:
    """Return the review-GUI presentation section for a metadata field."""
    role = _PRESENTATION_ROLE_BY_KEY.get(str(key))
    if role == "dip":
        return "Dip Evidence"
    if role == "jump":
        return "Jump Evidence"
    return _PRESENTATION_SECTION_BY_GROUP.get(str(group_name), "Advanced Metadata")


def metadata_presentation_role(group_name: str, key: str) -> str:
    """Return the review-GUI presentation role for a metadata field."""
    explicit = _PRESENTATION_ROLE_BY_KEY.get(str(key))
    if explicit:
        return explicit
    section = metadata_presentation_section(group_name, key)
    if section == "Coverage & Photometry":
        return "coverage"
    if section == "Catalog & Vetting":
        return "catalog"
    if section == "Classification & Environment":
        return "classification"
    return "advanced"


def extract_review_metadata_feature_rows(
    payload: dict[str, Any],
    *,
    round_sigfigs: bool = False,
) -> list[dict[str, str]]:
    """Return formatted metadata rows with keys and presentation sections.

    This complements ``extract_review_metadata_grouped`` for the Dash GUI: the
    grouped API stays stable for existing callers, while the GUI can build a
    streamlined review surface and an exhaustive feature table without losing
    the original key provenance.
    """
    p = normalize_vsx_record(payload)
    rows: list[dict[str, str]] = []
    for group_name, fields in REVIEW_METADATA_GROUPS:
        for label, key in fields:
            val = p.get(key)
            if not _is_present(val):
                continue
            display_label = bracket_unit_label(_display_label_for_group(group_name, label))
            rows.append({
                "section": metadata_presentation_section(group_name, key),
                "source_group": str(group_name),
                "label": display_label,
                "key": str(key),
                "value": str(_format_with_uncertainty(key, val, p, round_sf=round_sigfigs)),
                "role": metadata_presentation_role(group_name, key),
            })
    return rows


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

    # -- ALeRCE ---------------------------------------------------------------
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
