"""
Long-Term Variability (LTV) detection pipeline.

Implements the methodology from the paper for detecting slowly varying sources
in ASAS-SN data.

Modules:
    core: Seasonal trend computation (main processing)
    filter: False positive filtering (slope, PM, bright stars, etc.)
    crossmatch: Catalog crossmatches (Gaia, VSX, MILLIQUAS, SIMBAD)
    neowise: NEOWISE IR light curve extraction
    pipeline: Full pipeline integration

Usage:
    malca ltv-pipeline --mag-bin 13_13.5
"""

from importlib import import_module


_EXPORT_MODULES = {
    # Core
    "process_one_lc": "malca.ltv.core",
    "compute_trend_metrics": "malca.ltv.core",
    "compute_lomb_scargle": "malca.ltv.core",
    "seasonal_midpoints_from_ra": "malca.ltv.core",
    "assign_seasons_strict": "malca.ltv.core",
    "season_medians_with_gap_indices": "malca.ltv.core",
    # Filter
    "filter_slope_threshold": "malca.ltv.filter",
    "filter_max_diff_threshold": "malca.ltv.filter",
    "filter_south_pole": "malca.ltv.filter",
    "filter_refcat_offset": "malca.ltv.filter",
    "filter_photometric_scatter": "malca.ltv.filter",
    "filter_crowding": "malca.ltv.filter",
    "filter_high_proper_motion": "malca.ltv.filter",
    "filter_neighbor_high_pm": "malca.ltv.filter",
    "apply_all_filters": "malca.ltv.filter",
    "apply_all_filters_audit": "malca.ltv.filter",
    # Crossmatch (API only; Gaia DR3 and VSX use local catalog)
    "load_local_catalog": "malca.ltv.crossmatch",
    "merge_local_catalog": "malca.ltv.crossmatch",
    "crossmatch_from_local": "malca.ltv.crossmatch",
    "crossmatch_gaia_alerts": "malca.ltv.crossmatch",
    "crossmatch_milliquas": "malca.ltv.crossmatch",
    "query_simbad_classification": "malca.ltv.crossmatch",
    "crossmatch_all_catalogs": "malca.ltv.crossmatch",
    "crossmatch_tap_catalog": "malca.ltv.crossmatch",
    # NEOWISE
    "query_neowise_lc": "malca.ltv.neowise",
    "combine_epochs": "malca.ltv.neowise",
    "fit_neowise_trends": "malca.ltv.neowise",
    "extract_neowise_trends": "malca.ltv.neowise",
    # Dust flags
    "apply_dust_flags": "malca.ltv.dust",
    # CMD scaffolding
    "load_mist_grid": "malca.ltv.cmd",
    "compute_cmd_features": "malca.ltv.cmd",
    "assign_cmd_groups": "malca.ltv.cmd",
    "fetch_bailer_jones_distances": "malca.ltv.cmd",
    # Gaia epoch photometry
    "query_gaia_epoch_photometry_batch": "malca.ltv.gaia_epoch",
    "apply_gaia_epoch_flags": "malca.ltv.gaia_epoch",
    # Stochastic post-filter features
    "add_stochastic_postfilter_features": "malca.ltv.stochastic",
    # LTV external-survey summaries
    "compute_ltv_multi_survey_features": "malca.ltv.multi_survey",
    "write_ltv_multi_survey_features": "malca.ltv.multi_survey",
    # Pipeline
    "run_full_pipeline": "malca.ltv.pipeline",
    # Injection
    "inject_trend": "malca.ltv.injection",
    "run_ltv_injection_recovery": "malca.ltv.injection",
    "compute_rejection_summary": "malca.ltv.injection",
    "generate_ltv_injection_plots": "malca.ltv.injection",
    # Optimization
    "check_optimizations": "malca.ltv.optim",
    "cached": "malca.ltv.optim",
    "clear_cache": "malca.ltv.optim",
    "get_pooled_session": "malca.ltv.optim",
    # Review DB ingest
    "map_ltv_columns": "malca.ltv.review",
    "ingest_ltv_results": "malca.ltv.review",
}


_EXPORT_ALIASES = {
    "run_ltv_injection_recovery": ("malca.ltv.injection", "run_injection_recovery"),
    "generate_ltv_injection_plots": ("malca.ltv.injection", "generate_plots"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_ALIASES.get(name, (_EXPORT_MODULES[name], name))
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
