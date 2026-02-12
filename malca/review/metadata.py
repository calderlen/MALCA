from __future__ import annotations

from typing import Any

import pandas as pd


# Grouped metadata fields: list of (group_name, fields) where each field is
# a (label, key) tuple.  Groups are rendered as collapsible sections in the
# Dash app and as headed blocks in the TUI.
REVIEW_METADATA_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Identification", [
        ("ASAS-SN ID", "asas_sn_id"),
        ("Path", "path"),
        ("Gaia ID", "gaia_id"),
        ("RA", "ra"),
        ("Dec", "dec"),
        ("2MASS ID", "tmass_id"),
        ("AllWISE ID", "allwise_id"),
    ]),
    ("Crossmatch", [
        ("VSX class", "vsx_class"),
        ("VSX sep (arcsec)", "vsx_sep_arcsec"),
        ("catalog_match", "catalog_match"),
        ("SFR name", "sfr_name"),
        ("SFR sep (arcmin)", "sfr_sep_arcmin"),
        ("Cluster name", "cluster_name"),
        ("Cluster dist (pc)", "cluster_dist_pc"),
    ]),
    ("Light Curve Basics", [
        ("n_points", "n_points"),
        ("JD first", "jd_first"),
        ("JD last", "jd_last"),
        ("Cadence median (days)", "cadence_median_days"),
        ("n_cameras", "n_cameras"),
        ("Camera IDs", "camera_ids"),
        ("Camera min points", "camera_min_points"),
        ("Camera max points", "camera_max_points"),
        ("Baseline mag", "baseline_mag"),
        ("Baseline source", "baseline_source"),
        ("Trigger mode", "trigger_mode"),
    ]),
    ("Periodicity", [
        ("periodic_flag", "periodic_flag"),
        ("periodicity_score", "periodicity_score"),
        ("lsp_power", "lsp_power"),
        ("lsp_period", "lsp_period"),
        ("lsp_bootstrap_sig", "lsp_bootstrap_sig"),
        ("lsp_is_alias", "lsp_is_alias"),
        ("lsp_is_significant", "lsp_is_significant"),
    ]),
    ("Dip Detection", [
        ("dip_significant", "dip_significant"),
        ("dip_best_morph", "dip_best_morph"),
        ("dip_best_log_bf", "dip_best_log_bf"),
        ("dip_best_delta_bic", "dip_best_delta_bic"),
        ("dip_best_width_param", "dip_best_width_param"),
        ("dip_symmetry_score", "dip_symmetry_score"),
        ("dip_best_amp", "dip_best_amp"),
        ("dip_best_t0", "dip_best_t0"),
        ("dip_best_alpha", "dip_best_alpha"),
        ("dip_best_tau", "dip_best_tau"),
        ("dip_bayes_factor", "dip_bayes_factor"),
        ("dip_best_p", "dip_best_p"),
        ("dip_best_mag_event", "dip_best_mag_event"),
        ("dip_trigger_max", "dip_trigger_max"),
        ("dip_max_event_prob", "dip_max_event_prob"),
        ("dip_trigger_threshold", "dip_trigger_threshold"),
    ]),
    ("Dip Runs", [
        ("dip_count", "dip_count"),
        ("dip_run_count", "dip_run_count"),
        ("dip_max_run_points", "dip_max_run_points"),
        ("dip_max_run_duration", "dip_max_run_duration"),
        ("dip_max_run_sum", "dip_max_run_sum"),
        ("dip_max_run_max", "dip_max_run_max"),
        ("dip_max_run_cameras", "dip_max_run_cameras"),
        ("dip_max_log_bf_local", "dip_max_log_bf_local"),
    ]),
    ("Jump Detection", [
        ("jump_significant", "jump_significant"),
        ("jump_best_morph", "jump_best_morph"),
        ("jump_best_log_bf", "jump_best_log_bf"),
        ("jump_best_delta_bic", "jump_best_delta_bic"),
        ("jump_best_width_param", "jump_best_width_param"),
        ("jump_best_amp", "jump_best_amp"),
        ("jump_best_t0", "jump_best_t0"),
        ("jump_best_alpha", "jump_best_alpha"),
        ("jump_best_tau", "jump_best_tau"),
        ("jump_bayes_factor", "jump_bayes_factor"),
        ("jump_best_p", "jump_best_p"),
        ("jump_best_mag_event", "jump_best_mag_event"),
        ("jump_trigger_max", "jump_trigger_max"),
        ("jump_max_event_prob", "jump_max_event_prob"),
        ("jump_trigger_threshold", "jump_trigger_threshold"),
    ]),
    ("Jump Runs", [
        ("jump_count", "jump_count"),
        ("jump_run_count", "jump_run_count"),
        ("jump_max_run_points", "jump_max_run_points"),
        ("jump_max_run_duration", "jump_max_run_duration"),
        ("jump_max_run_sum", "jump_max_run_sum"),
        ("jump_max_run_max", "jump_max_run_max"),
        ("jump_max_run_cameras", "jump_max_run_cameras"),
        ("jump_max_log_bf_local", "jump_max_log_bf_local"),
    ]),
    ("Dip Recurrence", [
        ("dip_is_single_event", "dip_is_single_event"),
        ("dip_inter_event_spacing_median", "dip_inter_event_spacing_median"),
        ("dip_inter_event_spacing_std", "dip_inter_event_spacing_std"),
        ("dip_amplitude_consistency", "dip_amplitude_consistency"),
        ("dip_duration_consistency", "dip_duration_consistency"),
    ]),
    ("Jump Recurrence", [
        ("jump_is_single_event", "jump_is_single_event"),
        ("jump_inter_event_spacing_median", "jump_inter_event_spacing_median"),
        ("jump_inter_event_spacing_std", "jump_inter_event_spacing_std"),
        ("jump_amplitude_consistency", "jump_amplitude_consistency"),
        ("jump_duration_consistency", "jump_duration_consistency"),
    ]),
    ("Event Scoring", [
        ("dipper_score", "dipper_score"),
        ("dipper_n_dips", "dipper_n_dips"),
        ("dipper_n_valid_dips", "dipper_n_valid_dips"),
        ("jumper_score", "jumper_score"),
        ("jumper_n_jumps", "jumper_n_jumps"),
        ("jumper_n_valid_jumps", "jumper_n_valid_jumps"),
    ]),
    ("Stellar Parameters", [
        ("ruwe", "ruwe"),
        ("high_ruwe_flag", "high_ruwe_flag"),
        ("teff_gspphot", "teff_gspphot"),
        ("logg_gspphot", "logg_gspphot"),
        ("mh_gspphot", "mh_gspphot"),
        ("distance_gspphot", "distance_gspphot"),
        ("Parallax (mas)", "parallax"),
        ("PM RA (mas/yr)", "pmra"),
        ("PM Dec (mas/yr)", "pmdec"),
    ]),
    ("Photometry", [
        ("2MASS J", "tmass_j"),
        ("2MASS H", "tmass_h"),
        ("2MASS K", "tmass_k"),
        ("unWISE W1", "unwise_w1"),
        ("unWISE W2", "unwise_w2"),
        ("H-K", "H_K"),
        ("W1-W2", "W1_W2"),
        ("IPHAS H-alpha", "iphas_ha_mag"),
        ("unWISE W1 z-score", "unwise_w1_zscore"),
        ("unWISE W2 z-score", "unwise_w2_zscore"),
    ]),
    ("Galactic Coordinates", [
        ("Galactic l", "gal_l"),
        ("Galactic b", "gal_b"),
    ]),
    ("Extinction & Environment", [
        ("A_v_3d", "A_v_3d"),
        ("ebv_3d", "ebv_3d"),
        ("population", "population"),
        ("age50", "age50"),
        ("mass50", "mass50"),
        ("banyan_field_prob", "banyan_field_prob"),
        ("banyan_best_assoc", "banyan_best_assoc"),
    ]),
    ("YSO / Classification", [
        ("yso_class", "yso_class"),
        ("Trigger", "trigger_type"),
        ("final_class", "final_class"),
        ("P_eb", "P_eb"),
        ("P_cv", "P_cv"),
        ("P_starspot", "P_starspot"),
        ("P_disk", "P_disk"),
        ("a_circ_au", "a_circ_au"),
        ("transit_prob", "transit_prob"),
        ("hill_radius_rsun", "hill_radius_rsun"),
    ]),
    ("Filter Flags", [
        ("failed_any", "failed_any"),
        ("failed_sparse", "failed_sparse"),
        ("failed_multi_camera", "failed_multi_camera"),
        ("failed_vsx", "failed_vsx"),
        ("failed_evidence_strength", "failed_evidence_strength"),
        ("failed_run_robustness", "failed_run_robustness"),
        ("failed_morphology", "failed_morphology"),
        ("failed_score", "failed_score"),
        ("failed_periodicity", "failed_periodicity"),
        ("failed_gaia_ruwe", "failed_gaia_ruwe"),
        ("failed_periodic_catalog", "failed_periodic_catalog"),
        ("failed_signal_amplitude", "failed_signal_amplitude"),
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
    "Identification",
    "Crossmatch",
    "Periodicity",
    "Dip Detection",
    "Jump Detection",
    "Stellar Parameters",
    "Extinction & Environment",
    "YSO / Classification",
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


def extract_review_metadata(
    payload: dict[str, Any],
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
        rows.append((label, val))
    return rows


def extract_review_metadata_grouped(
    payload: dict[str, Any],
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
            items.append((label, val))
        if items:
            groups.append((group_name, items))
    return groups


def is_group_default_open(group_name: str) -> bool:
    """Whether *group_name* should start expanded in the Dash GUI."""
    return group_name in _DEFAULT_OPEN_GROUPS
