"""Training-data feature guide for MALCA ML classification.

This module defines which columns to *include* and which to *exclude*
when building feature matrices for event-class classification
(circumstellar dust, microlensing, flare, etc.).

Usage
-----
>>> from malca.ml.features import ML_FEATURE_COLUMNS, ML_DROP_COLUMNS
>>> X = df[ML_FEATURE_COLUMNS].copy()  # only curated features

Or, for a safer approach that handles missing columns::

    >>> from malca.ml.features import select_ml_features
    >>> X = select_ml_features(df)
"""

from __future__ import annotations

import pandas as pd

# ---- TARGET VARIABLE --------------------------------------------------------
# The supervised label the model trains on.  Set via the review system.
ML_LABEL_COLUMN = "event_class"

# ---- COLUMNS TO DROP -------------------------------------------------------
# These should **never** be used as ML features.  They are pipeline
# bookkeeping, redundant, or rule-based outputs that would bake in
# assumptions the model should learn on its own.
ML_DROP_COLUMNS: set[str] = {
    # Identifiers / paths (not physics)
    "candidate_id",
    "path",
    "lc_path",
    "source_path",
    "asas_sn_id",
    "gaia_id",
    "tmass_id",
    "allwise_id",
    "camera_ids",

    # Pipeline configuration (not physics)
    "trigger_mode",
    "trigger_type",
    "dip_trigger_threshold",
    "jump_trigger_threshold",
    "baseline_source",

    # Rule-based classification outputs
    # These encode hardcoded heuristics; the ML model should learn its
    # own decision boundaries rather than inheriting pipeline biases.
    "P_eb",
    "P_cv",
    "P_starspot",
    "P_disk",
    "final_class",
    "yso_class",

    # Filter flags — pipeline gating decisions, not physical features.
    # The ML model should learn its own gating from the raw data.
    "failed_any",
    "failed_sparse",
    "failed_multi_camera",
    "failed_vsx",
    "failed_evidence_strength",
    "failed_run_robustness",
    "failed_morphology",
    "failed_score",
    "failed_periodicity",
    "failed_gaia_ruwe",
    "failed_periodic_catalog",
    "failed_signal_amplitude",
    "bad_cameras_filtered",

    # Redundant pair — A_v = R_v * E(B-V); keep A_v_3d only
    "ebv_3d",

    # Review metadata (not features)
    "interest_score",
    "review_pass",
    "notes",
    "status",
    "reviewer",
    "updated_at",
    "event_class",  # this is the label, not a feature

    # Timestamps (not useful as features directly)
    "jd_first",
    "jd_last",
    "dip_best_t0",
    "jump_best_t0",

    # Characterization module status columns
    "char_status_population",
    "char_error_population",
    "char_status_starhorse",
    "char_error_starhorse",
    "char_status_dust",
    "char_error_dust",
    "char_status_yso",
    "char_error_yso",
    "char_status_banyan",
    "char_error_banyan",
    "char_status_iphas",
    "char_error_iphas",
    "char_status_sfr",
    "char_error_sfr",
    "char_status_clusters",
    "char_error_clusters",
    "char_status_unwise",
    "char_error_unwise",

    # Payload JSON blob
    "payload_json",
    "imported_at",
    "source_id",
}

# ---- RECOMMENDED FEATURE COLUMNS -------------------------------------------
# These are the physics-driven features the classifier should use.
# Grouped by category for readability.  Order does not matter for the model.
ML_FEATURE_COLUMNS: list[str] = [
    # -- Periodicity --
    "periodic_flag",
    "periodicity_score",
    "lsp_power",
    "lsp_period",
    "lsp_bootstrap_sig",
    "lsp_is_alias",
    "lsp_is_significant",

    # -- Dip detection --
    "dip_significant",
    "dip_best_log_bf",
    "dip_best_delta_bic",
    "dip_best_width_param",
    "dip_symmetry_score",
    "dip_best_amp",
    "dip_best_alpha",
    "dip_best_tau",
    "dip_bayes_factor",
    "dip_best_p",
    "dip_best_mag_event",
    "dip_trigger_max",
    "dip_max_event_prob",

    # -- Dip runs --
    "dip_count",
    "dip_run_count",
    "dip_max_run_points",
    "dip_max_run_duration",
    "dip_max_run_sum",
    "dip_max_run_max",
    "dip_max_run_cameras",
    "dip_max_log_bf_local",

    # -- Dip recurrence --
    "dip_is_single_event",
    "dip_inter_event_spacing_median",
    "dip_inter_event_spacing_std",
    "dip_amplitude_consistency",
    "dip_duration_consistency",

    # -- Jump detection --
    "jump_significant",
    "jump_best_log_bf",
    "jump_best_delta_bic",
    "jump_best_width_param",
    "jump_best_amp",
    "jump_best_alpha",
    "jump_best_tau",
    "jump_bayes_factor",
    "jump_best_p",
    "jump_best_mag_event",
    "jump_trigger_max",
    "jump_max_event_prob",

    # -- Jump runs --
    "jump_count",
    "jump_run_count",
    "jump_max_run_points",
    "jump_max_run_duration",
    "jump_max_run_sum",
    "jump_max_run_max",
    "jump_max_run_cameras",
    "jump_max_log_bf_local",

    # -- Jump recurrence --
    "jump_is_single_event",
    "jump_inter_event_spacing_median",
    "jump_inter_event_spacing_std",
    "jump_amplitude_consistency",
    "jump_duration_consistency",

    # -- Event scoring --
    "dipper_score",
    "dipper_n_dips",
    "dipper_n_valid_dips",
    "jumper_score",
    "jumper_n_jumps",
    "jumper_n_valid_jumps",

    # -- Light curve basics --
    "n_points",
    "cadence_median_days",
    "n_cameras",
    "baseline_mag",

    # -- Stellar parameters (Gaia DR3) --
    "ruwe",
    "high_ruwe_flag",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "distance_gspphot",
    "parallax",
    "pmra",
    "pmdec",

    # -- Photometry --
    "tmass_j",
    "tmass_h",
    "tmass_k",
    "unwise_w1",
    "unwise_w2",
    "H_K",
    "W1_W2",
    "iphas_ha_mag",
    "unwise_w1_zscore",
    "unwise_w2_zscore",

    # -- Galactic coordinates (microlensing prior) --
    "gal_l",
    "gal_b",

    # -- Extinction & environment --
    "A_v_3d",
    "population",
    "age50",
    "mass50",
    "banyan_field_prob",
    "banyan_best_assoc",

    # -- Crossmatch context --
    "catalog_match",
    "vsx_class",
    "vsx_sep_arcsec",
    "sfr_name",
    "sfr_sep_arcmin",
    "cluster_name",
    "cluster_membership_prob",

    # -- Orbital / transit context --
    "a_circ_au",
    "transit_prob",
    "hill_radius_rsun",
]

# ---- MORPHOLOGY FEATURE (categorical) --------------------------------------
# Best-fit morphology model name.  Encode as category codes for tree models.
ML_MORPH_COLUMNS: list[str] = [
    "dip_best_morph",
    "jump_best_morph",
]


def select_ml_features(
    df: pd.DataFrame,
    *,
    include_morph: bool = True,
) -> pd.DataFrame:
    """Select and prepare ML-ready feature columns from *df*.

    - Keeps only columns listed in ``ML_FEATURE_COLUMNS`` (+ morph).
    - Encodes object/string columns as category codes.
    - Replaces inf with NaN, then fills NaN with 0.0.
    - Returns a copy; never mutates the input.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (e.g. from ``export_reviews``).
    include_morph : bool
        If True (default), include ``dip_best_morph`` / ``jump_best_morph``
        as integer-coded categorical features.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for model training.
    """
    import numpy as np

    cols = [c for c in ML_FEATURE_COLUMNS if c in df.columns]
    if include_morph:
        cols += [c for c in ML_MORPH_COLUMNS if c in df.columns]

    X = df[cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X
