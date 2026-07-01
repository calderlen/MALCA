"""Three-layer feature organization for MALCA candidate tables.

MALCA feature products use three explicit feature layers:

``lc_stats``
    Measurements computed directly from native MALCA/ASAS-SN light curves.
``external_stats``
    Values retrieved from external catalogs or external light-curve services.
``derived_stats``
    Deterministic combinations, flags, classifications, and ratios derived from
    light-curve or external raw columns.

The helpers in this module build and inspect canonical layer-first products.
Old flat product schemas are handled only by the migration package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from malca.core.derived_stats import append_derived_features
from malca.io.table_io import read_parquet_table, write_parquet_table


FEATURE_LAYER_VERSION = "1"
FEATURE_LAYER_VERSION_COLUMN = "feature_layer_version"
LC_STATS_LAYER = "lc_stats"
EXTERNAL_STATS_LAYER = "external_stats"
DERIVED_STATS_LAYER = "derived_stats"
FEATURE_LAYER_COLUMNS: tuple[str, str, str] = (
    LC_STATS_LAYER,
    EXTERNAL_STATS_LAYER,
    DERIVED_STATS_LAYER,
)
ALL_FEATURE_LAYER_COLUMNS: tuple[str, ...] = (
    FEATURE_LAYER_VERSION_COLUMN,
    *FEATURE_LAYER_COLUMNS,
)


def is_layer_first_frame(df: pd.DataFrame) -> bool:
    """Return True when a frame has the canonical three feature-layer columns."""
    return isinstance(df, pd.DataFrame) and set(FEATURE_LAYER_COLUMNS).issubset(set(map(str, df.columns)))


_IDENTITY_AND_BOOKKEEPING_COLUMNS = {
    "candidate_id",
    "target_id",
    "flare_id",
    "timescale",
    "asas_sn_id",
    "lc_path",
    "path",
    "dat_path",
    "local_lightcurve_path",
    "source_path",
    "payload_json",
    "imported_at",
    "filter_reason",
    "trigger_type",
    "review_pass",
    "interest_score",
    "workflow_status",
    "event_class",
    "disposition",
    "reviewer",
    "duplicate_of",
    "known_object_id",
    "known_object_source",
    "taxonomy_version",
    "classification_confidence",
    "morphology_primary",
    "morphology_secondary",
    "morphology_polarity",
    "morphology_recurrence",
    "baseline_behavior",
    "physical_primary",
    "physical_secondary",
    "comments",
    "review_notes",
    "nuclear_target_id",
}
IDENTITY_AND_BOOKKEEPING_COLUMNS = frozenset(_IDENTITY_AND_BOOKKEEPING_COLUMNS)

_LC_EXACT_COLUMNS = {
    "periodicity_score",
    "lsp_bootstrap_sig",
    "lsp_power",
    "lsp_period",
    "lsp_is_alias",
    "lsp_is_significant",
    "pdm_period",
    "pdm_theta",
    "pdm_snr",
    "pdm_bootstrap_sig",
    "pdm_is_significant",
    "ce_period",
    "ce_entropy",
    "ce_snr",
    "ce_bootstrap_sig",
    "ce_is_significant",
    "periodicity_bootstrap_sig",
    "periodicity_is_significant",
    "phase_plot_ready",
    "phase_period_days",
    "phase_source",
    "phase_quality_score",
    "lc_feature_status",
    "n_points",
    "jd_first",
    "jd_last",
    "time_span_days",
    "n_unique_nights",
    "n_cameras",
    "camera_ids",
    "camera_min_points",
    "camera_max_points",
    "excluded_cameras",
    "raw_median_suspect_cameras",
    "baseline_mag",
    "baseline_source",
    "pre_periodicity_label",
    "pre_periodic_flag",
    "pre_periodicity_selected_period",
    "pre_periodicity_method",
    "cadence_median_days",
    "trigger_mode",
    "bands",
    "t_start_jd",
    "t_peak_jd",
    "t_centroid_jd",
    "t_end_jd",
    "duration_days",
    "n_nights",
    "peak_snr",
    "peak_flux_resid",
    "fluence_flux_days",
    "amp_flux_rel",
    "width_days",
    "asymmetry",
    "flare_quality",
    "primary_band",
    "primary_band_used",
    "n_total_points",
    "n_g_points",
    "n_v_points",
    "n_primary_points",
    "has_primary_band",
    "passes_min_points",
    "baseline_days",
    "median_mag",
    "median_mag_err",
    "reference_mag",
    "quality_status",
    "quality_error",
    "nuc_n_points_raw",
    "nuc_n_points_clean",
    "nuc_n_points_g",
    "nuc_n_points_v",
    "nuc_jd_first",
    "nuc_jd_last",
    "nuc_peak_snr_max",
    "nuc_flare_status",
    "nuc_n_flares",
    "nuc_first_flare_jd",
    "nuc_last_flare_jd",
    "nuc_best_period_days",
    "nuc_best_t0_jd",
    "nuc_period_score",
    "nuc_repeating_score",
    "nuc_phase_rms_days",
    "nuc_phase_tolerance_days",
    "nuc_matched_flare_count",
    "nuc_matched_cycle_count",
    "nuc_observable_cycle_count",
    "nuc_missed_cycle_count",
    "nuc_observed_cycle_fraction",
    "nuc_amp_consistency",
    "nuc_width_consistency",
    "nuc_alias_flag",
    "nuc_repeating_candidate",
    "nuc_repeating_status",
}

_LC_PREFIXES = (
    "stats_",
    "dip_",
    "jump_",
    "dipper_",
    "jumper_",
)

_LTV_EXTERNAL_EXACT_COLUMNS = {
    "ltv_vsx_match",
    "ltv_vsx_name",
    "ltv_milliquas_match",
    "ltv_gaia_alert_match",
    "ltv_neowise_w1_slope",
    "ltv_neowise_w1_w2_slope",
    "ltv_neowise_n_epochs",
}

_LTV_DERIVED_EXACT_COLUMNS = {
    "ltv_class",
    "ltv_class_reason",
    "ltv_interest_score",
    "ltv_dust_candidate",
    "ltv_dust_excess",
    "ltv_failed_slope",
    "ltv_failed_max_diff",
    "ltv_failed_dec",
    "ltv_failed_refcat_offset",
    "ltv_failed_photometric_scatter",
    "ltv_failed_high_pm",
    "ltv_failed_neighbor_high_pm",
    "ltv_failed_crowding",
}

_EXTERNAL_EXACT_COLUMNS = {
    "source_id",
    "gaia_id",
    "gaia_dr2_id",
    "gaia_id_release",
    "gaia_id_mapping_status",
    "dr2_dr3_angular_distance_mas",
    "dr2_dr3_magnitude_difference",
    "ra",
    "dec",
    "asassn_field_key",
    "asassn_fields",
    "asassn_field_count",
    "asassn_field_key_fraction",
    "camera_name_key",
    "camera_names",
    "camera_name_count",
    "camera_name_key_fraction",
    "catalog_match",
    "catalog_source",
    "periodic_flag",
    "period_ogle_name",
    "period_ogle_match",
    "period_ogle_days",
    "period_ogle_class",
    "period_ogle_sep_arcsec",
    "ruwe",
    "radial_velocity",
    "rv_amplitude_robust",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "distance_gspphot",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "tmass_j",
    "tmass_j_err",
    "tmass_h",
    "tmass_h_err",
    "tmass_k",
    "tmass_k_err",
    "w1",
    "w1_err",
    "w2",
    "w2_err",
    "w3",
    "w3_err",
    "w4",
    "w4_err",
    "A_v_3d",
    "ebv_3d",
    "dust_sigma",
    "dust_max_dist_kpc",
    "population",
    "age50",
    "mass50",
    "banyan_field_prob",
    "banyan_best_assoc",
    "redshift",
    "redshift_source",
    "spectral_type",
    "spectral_type_source",
    "host_spectral_class",
    "spectrum_sources",
    "spectrum_links",
    "source_catalog",
    "source_name",
    "agn_type",
    "asassn_id",
    "g_mag",
    "v_mag",
    "r_mag",
    "i_mag",
    "wise_w1",
    "wise_w2",
    "black_hole_mass",
    "bol_luminosity",
    "eddington_ratio",
    "qso_prob",
    "broad_type",
    "fetch_status",
    "fetch_error",
    "cone_sep_arcsec",
    "cone_n_matches",
    "cone_catalog_sources",
    "local_lc_path",
    "local_lc_suffix",
    "local_lc_dir",
    "local_lc_filename",
    "local_lc_status",
    "catalog_lc_path",
    "gal_l",
    "gal_b",
    "pm_cluster_offset_sigma",
    "name",
    "host_name",
    "target_ra",
    "target_dec",
    "match_ra",
    "match_dec",
    "expected_period_days",
    "target_type",
    "validation_mode",
    "expected_min_flares",
    "period_tolerance_days",
    "min_repeating_flares",
    "min_repeating_score",
    "period_max_days",
    "literature_note",
    "ra_deg",
    "dec_deg",
    "refcat_id",
    "tic_id",
    "plx",
    "plx_d",
    "pm_ra",
    "pm_ra_d",
    "pm_dec",
    "pm_dec_d",
    "gaia_mag",
    "gaia_mag_d",
    "gaia_b_mag",
    "gaia_b_mag_d",
    "gaia_r_mag",
    "gaia_r_mag_d",
    "gaia_eff_temp",
    "gaia_g_extinc",
    "gaia_var",
    "sfd_g_extinc",
    "rp_00_1",
    "rp_01",
    "rp_10",
    "nstat",
    "mean_vmag",
    "epochs",
    "nuc_target_name",
    "nuc_host_name",
    "nuc_redshift",
    "nuc_target_ra",
    "nuc_target_dec",
    "nuc_expected_period_days",
    "nuc_validation_mode",
    "nuc_expected_min_flares",
    "nuc_period_tolerance_days",
    "nuc_min_repeating_score",
    "nuc_literature_note",
    "nuc_fetch_backend",
    "nuc_fetch_status",
    "nuc_fetch_error",
    "nuc_cone_sep_arcsec",
    "nuc_cone_n_matches",
    "nuc_fetch_attempts_json",
    "nuc_fetch_backend_used",
    "nuc_fetch_id_kind_used",
    "nuc_fetch_id_used",
}

_EXTERNAL_PREFIXES = (
    "vsx_",
    "simbad_",
    "gaia_var_",
    "gaia_eb_",
    "gaia_epoch_",
    "asassn_var_",
    "microlens_",
    "ztf_var_",
    "tns_",
    "alerce_",
    "xray_",
    "atlas_",
    "neowise_",
    "unwise_",
    "ztf_lc_",
    "tess_",
    "kepler_",
    "aavso_",
    "ogle_lc_",
    "stripe82_lc_",
    "allwise_mep_",
    "vvvx_virac_",
    "ps1_",
    "crts_",
    "apass_",
    "galex_",
    "iphas_",
    "vphas_",
    "sfr_",
    "cluster_",
    "banyan_",
    "swift_",
    "radio_",
    "known_clagn_",
    "ms_",
    "ltv_ms_",
    "pstarrs_",
    "sp1_",
)

_DERIVED_EXACT_COLUMNS = {
    "pm_total",
    "high_pm_flag",
    "high_ruwe_flag",
    "bp_rp",
    "mg",
    "mg0",
    "bprp0",
    "H_K",
    "w1_w2",
    "w1_w3",
    "w1_w4",
    "w2_w3",
    "w2_w4",
    "w3_w4",
    "iphas_ha_excess",
    "unwise_w1_var",
    "period_sources",
    "period_n_sources",
    "period_consensus_days",
    "period_consensus_agree",
    "period_conflict_flag",
    "period_consensus_support",
    "period_primary_source",
    "period_source_periods",
    "gaia_parallax_snr",
    "gaia_pm_snr",
    "gaia_stellar_veto_score",
    "gaia_extragalactic_prior_score",
    "host_match",
    "host_source",
    "host_sep_arcsec",
    "nuclear_offset_arcsec",
    "host_nuclear_score",
    "host_assoc_status",
    "radio_det",
    "radio_source_catalogs",
    "radio_sep_arcsec",
    "radio_flux_mjy",
    "radio_agn_prior_score",
    "wise_agn_score",
    "neowise_variability_score",
    "xray_agn_prior_score",
    "uv_tde_score",
    "agn_prior_score",
    "agn_prior_reasons",
    "tde_candidate_score",
    "tde_candidate_reasons",
    "clagn_photometric_score",
    "clagn_reasons",
    "prior_agn_spectrum_flag",
    "broad_line_flag",
    "swift_uvot_obs",
    "swift_uvot_det",
    "swift_xrt_det",
    "swift_uvot_sep_arcsec",
    "swift_xrt_sep_arcsec",
    "swift_source_catalogs",
    "swift_status",
    "known_clagn_training_label",
    "nuc_n_points",
    "nuc_time_span_days",
    "nuc_flux_frac_amp_p95_p05",
    "nuc_flux_slope_snr",
    "n_flare_events",
    "recurrence_count",
    "preflare_rms",
    "tde_single_flare_score",
    "tde_quiet_baseline_score",
    "tde_no_recurrence_score",
    "tde_smooth_decline_score",
    "fallback_fit_r2",
    "clagn_state_change_mag",
    "clagn_monotonicity_score",
    "clagn_plateau_score",
    "vetting_likely_known",
    "yso_class",
    "final_class",
    "P_eb",
    "P_cv",
    "P_starspot",
    "P_disk",
    "a_circ_au",
    "transit_prob",
    "hill_radius_rsun",
    "failed_any",
    "failed_posterior_strength",
    "failed_run_robustness",
    "failed_morphology",
    "failed_score",
    "failed_periodicity",
    "failed_gaia_ruwe",
    "failed_periodic_catalog",
    "failed_signal_amplitude",
    "bad_cameras_filtered",
    "nuc_bad_cameras_filtered",
    "nuc_clean_status",
    "nuc_failed_no_lc",
    "nuc_failed_low_quality_lc",
    "nuc_failed_no_flares",
    "nuc_failed_nonrepeating",
    "nuc_failed_foreground_star",
    "nuc_class",
    "nuc_class_reason",
    "nuc_interest_score",
}

_DERIVED_PREFIXES = (
    "derived_",
)


def feature_layer_for_column(column: str) -> str | None:
    """Return the three-layer feature bucket for a flat column name."""
    col = str(column)
    if col in ALL_FEATURE_LAYER_COLUMNS or col in _IDENTITY_AND_BOOKKEEPING_COLUMNS:
        return None
    if col.startswith(_DERIVED_PREFIXES) or col in _DERIVED_EXACT_COLUMNS:
        return DERIVED_STATS_LAYER
    if col in _LTV_DERIVED_EXACT_COLUMNS:
        return DERIVED_STATS_LAYER
    if col.startswith(("nuc_failed_",)):
        return DERIVED_STATS_LAYER
    if col.startswith(
        (
            "nuc_outlier",
            "nuc_stochastic_outlier",
            "nuc_state_change",
            "nuc_transient_like",
            "nuc_drw_interest",
        )
    ) or col in {"nuc_rank", "nuc_feature_status"}:
        return DERIVED_STATS_LAYER
    if col.startswith(("nuc_target_", "nuc_fetch_", "nuc_cone_")):
        return EXTERNAL_STATS_LAYER
    if col.startswith(_EXTERNAL_PREFIXES) or col in _EXTERNAL_EXACT_COLUMNS:
        return EXTERNAL_STATS_LAYER
    if col.startswith("nuc_"):
        return LC_STATS_LAYER
    if col in _LTV_EXTERNAL_EXACT_COLUMNS or col.startswith("ltv_ms_"):
        return EXTERNAL_STATS_LAYER
    if col.startswith("ltv_"):
        return LC_STATS_LAYER
    if col.startswith(_LC_PREFIXES) or col in _LC_EXACT_COLUMNS:
        return LC_STATS_LAYER
    return None


def feature_columns_by_layer(columns: Iterable[str]) -> dict[str, list[str]]:
    """Group known feature columns by layer while preserving input order."""
    out = {layer: [] for layer in FEATURE_LAYER_COLUMNS}
    for col in columns:
        layer = feature_layer_for_column(str(col))
        if layer is not None:
            out[layer].append(str(col))
    return out


def non_layer_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return flat columns that are recognized as layer-backed features."""
    return [str(col) for col in columns if feature_layer_for_column(str(col)) is not None]


def unclassified_non_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return columns that are neither layer-backed nor known bookkeeping."""
    out: list[str] = []
    for col in columns:
        name = str(col)
        if name in ALL_FEATURE_LAYER_COLUMNS or name in IDENTITY_AND_BOOKKEEPING_COLUMNS:
            continue
        if feature_layer_for_column(name) is None:
            out.append(name)
    return out


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _json_ready(value: Any) -> Any:
    if _is_missing_value(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _parse_layer_value(value: Any) -> dict[str, Any]:
    if _is_missing_value(value):
        return {}
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, Mapping):
            return {str(k): _json_ready(v) for k, v in parsed.items()}
    return {}


def parse_layer_value(value: Any) -> dict[str, Any]:
    """Return a normalized dict for a layer JSON/dict value."""
    return _parse_layer_value(value)


def layer_path_for_column(column: str) -> str | None:
    """Return ``layer.column`` for a registered flat feature column."""
    layer = feature_layer_for_column(column)
    return f"{layer}.{column}" if layer is not None else None


def split_layer_path(path: str) -> tuple[str, str]:
    """Split a canonical ``layer.key`` feature path."""
    text = str(path)
    if "." not in text:
        raise ValueError(f"Feature path must be '<layer>.<key>': {path}")
    layer, key = text.split(".", 1)
    if layer not in FEATURE_LAYER_COLUMNS:
        raise ValueError(f"Unknown feature layer in path '{path}'")
    if not key:
        raise ValueError(f"Feature path is missing a key: {path}")
    return layer, key


def is_layer_path(path: str) -> bool:
    try:
        split_layer_path(path)
        return True
    except ValueError:
        return False


def feature_value_series(
    df: pd.DataFrame,
    path: str,
    *,
    default: Any = pd.NA,
) -> pd.Series:
    """Extract one canonical layer-path feature as a Series."""
    layer, key = split_layer_path(path)
    if layer not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[layer].map(lambda value: _parse_layer_value(value).get(key, default))


def feature_mapping_get(mapping: Mapping[str, Any], column_or_path: str, default: Any = None) -> Any:
    """Read a value from a layer-first mapping by identity name, flat feature name, or layer path."""
    if not isinstance(mapping, Mapping):
        return default
    key = str(column_or_path)
    if key in mapping:
        return mapping.get(key, default)
    if "." in key:
        try:
            layer, layer_key = split_layer_path(key)
        except ValueError:
            return default
        return _parse_layer_value(mapping.get(layer)).get(layer_key, default)
    layer = feature_layer_for_column(key)
    if layer is None:
        return default
    return _parse_layer_value(mapping.get(layer)).get(key, default)


def select_feature_values(
    df: pd.DataFrame,
    paths: Iterable[str],
    *,
    include_identity: Iterable[str] = (),
) -> pd.DataFrame:
    """Build a temporary view from identity columns and canonical feature paths."""
    out = pd.DataFrame(index=df.index)
    for col in include_identity:
        name = str(col)
        if name not in df.columns:
            raise KeyError(f"Identity column not found in layer-first table: {name}")
        out[name] = df[name]
    for path in paths:
        name = str(path)
        out[name] = feature_value_series(df, name)
    return out.reset_index(drop=True)


def with_feature_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected flat feature columns populated from layers when absent."""
    out = df.copy()
    for column in columns:
        name = str(column)
        if name in out.columns:
            continue
        path = name if is_layer_path(name) else layer_path_for_column(name)
        if path is None:
            continue
        out[name] = feature_value_series(out, path)
    return out


def row_feature_layers(row: Mapping[str, Any], *, include_missing: bool = False) -> dict[str, dict[str, Any]]:
    """Split a row-like mapping into the three feature layers."""
    layers: dict[str, dict[str, Any]] = {layer: {} for layer in FEATURE_LAYER_COLUMNS}
    for layer in FEATURE_LAYER_COLUMNS:
        layers[layer].update(_parse_layer_value(row.get(layer)))

    for key, raw_value in row.items():
        layer = feature_layer_for_column(str(key))
        if layer is None:
            continue
        if not include_missing and _is_missing_value(raw_value):
            continue
        value = _json_ready(raw_value)
        if value is None and not include_missing:
            continue
        layers[layer][str(key)] = value
    return layers


def _mapping_with_derived_features(row: Mapping[str, Any]) -> dict[str, Any]:
    frame = append_derived_features(pd.DataFrame([dict(row)]), overwrite=False)
    if frame.empty:
        return dict(row)
    return frame.iloc[0].to_dict()


def to_layer_first_mapping(
    row: Mapping[str, Any],
    *,
    include_missing: bool = False,
    layer_values_as_json: bool = False,
    run_derived: bool = True,
) -> dict[str, Any]:
    """Return one row with recognized feature values moved into layer fields."""
    source = _mapping_with_derived_features(row) if run_derived else dict(row)
    layers = row_feature_layers(source, include_missing=include_missing)
    out: dict[str, Any] = {}
    for key, raw_value in source.items():
        name = str(key)
        if name in ALL_FEATURE_LAYER_COLUMNS or feature_layer_for_column(name) is not None:
            continue
        if not include_missing and _is_missing_value(raw_value):
            continue
        value = _json_ready(raw_value)
        if value is None and not include_missing:
            continue
        out[name] = value

    out[FEATURE_LAYER_VERSION_COLUMN] = FEATURE_LAYER_VERSION
    for layer in FEATURE_LAYER_COLUMNS:
        if layer_values_as_json:
            out[layer] = json.dumps(layers[layer], sort_keys=True, separators=(",", ":"), default=str)
        else:
            out[layer] = layers[layer]
    return out


def split_feature_frame(
    df: pd.DataFrame,
    *,
    run_derived: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return one DataFrame per feature layer, preserving the original index."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {layer: pd.DataFrame(index=getattr(df, "index", None)) for layer in FEATURE_LAYER_COLUMNS}

    out = append_derived_features(df) if run_derived else df.copy()
    grouped = feature_columns_by_layer(out.columns)
    return {layer: out[cols].copy() if cols else pd.DataFrame(index=out.index) for layer, cols in grouped.items()}


def to_layer_first_frame(
    df: pd.DataFrame,
    *,
    include_missing: bool = False,
    run_derived: bool = True,
) -> pd.DataFrame:
    """Return a canonical table with flat feature columns moved into layers."""
    if not isinstance(df, pd.DataFrame):
        return df
    if df.empty:
        out = df.copy()
        for layer in FEATURE_LAYER_COLUMNS:
            if layer not in out.columns:
                out[layer] = pd.Series(dtype="object")
        if FEATURE_LAYER_VERSION_COLUMN not in out.columns:
            out[FEATURE_LAYER_VERSION_COLUMN] = pd.Series(dtype="object")
        return out

    enriched = append_derived_features(df) if run_derived else df.copy()
    layer_records = [
        row_feature_layers(row, include_missing=include_missing)
        for row in enriched.to_dict("records")
    ]
    feature_cols = set(non_layer_feature_columns(enriched.columns))
    keep_cols = [
        col
        for col in enriched.columns
        if col not in feature_cols and col not in ALL_FEATURE_LAYER_COLUMNS
    ]
    out = enriched[keep_cols].copy()
    for layer in FEATURE_LAYER_COLUMNS:
        out[layer] = [
            json.dumps(layer_map[layer], sort_keys=True, separators=(",", ":"), default=str)
            for layer_map in layer_records
        ]
    out[FEATURE_LAYER_VERSION_COLUMN] = FEATURE_LAYER_VERSION
    return out


def layer_frames(
    df: pd.DataFrame,
    *,
    id_columns: Iterable[str] = ("candidate_id", "asas_sn_id", "source_id", "gaia_id"),
) -> dict[str, pd.DataFrame]:
    """Return one DataFrame per feature layer, retaining available ID columns."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {layer: pd.DataFrame() for layer in FEATURE_LAYER_COLUMNS}

    grouped: dict[str, list[str]] = {layer: [] for layer in FEATURE_LAYER_COLUMNS}
    rows_by_layer: dict[str, list[dict[str, Any]]] = {layer: [] for layer in FEATURE_LAYER_COLUMNS}
    for _, row in df.iterrows():
        for layer in FEATURE_LAYER_COLUMNS:
            parsed = _parse_layer_value(row.get(layer))
            rows_by_layer[layer].append(parsed)
            for key in parsed:
                if key not in grouped[layer]:
                    grouped[layer].append(key)
    ids = [col for col in id_columns if col in df.columns]
    return {
        layer: pd.concat(
            [
                df[ids].reset_index(drop=True) if ids else pd.DataFrame(index=df.index).reset_index(drop=True),
                pd.DataFrame(rows_by_layer[layer], columns=cols).reset_index(drop=True),
            ],
            axis=1,
        )
        for layer, cols in grouped.items()
    }


def write_layer_frames(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write separate parquet files for the three layer views."""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for layer, frame in layer_frames(df).items():
        path = out_dir / f"{layer}.parquet"
        write_parquet_table(frame, path)
        paths[layer] = path
    return paths


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return read_parquet_table(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    else:
        write_parquet_table(df, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize MALCA's lc/external/derived feature layers for an existing candidate table.",
    )
    parser.add_argument("input", type=Path, help="Input parquet or CSV table.")
    parser.add_argument("--output", type=Path, default=None, help="Output parquet/CSV as a canonical layer-first table.")
    parser.add_argument("--separate-dir", type=Path, default=None, help="Also write lc_stats/external_stats/derived_stats parquet files here.")
    parser.add_argument("--include-missing", action="store_true", help="Keep null-valued keys in layer JSON blobs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = _read_table(args.input.expanduser())
    out = to_layer_first_frame(df, include_missing=bool(args.include_missing))

    if args.output is not None:
        _write_table(out, args.output.expanduser())
        print(f"Wrote layered table: {args.output.expanduser()}")
    if args.separate_dir is not None:
        paths = write_layer_frames(out, args.separate_dir.expanduser())
        for layer, path in paths.items():
            print(f"Wrote {layer}: {path}")
    if args.output is None and args.separate_dir is None:
        counts = {layer: len(cols) for layer, cols in feature_columns_by_layer(out.columns).items()}
        print(json.dumps({"rows": int(len(out)), "feature_columns": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
