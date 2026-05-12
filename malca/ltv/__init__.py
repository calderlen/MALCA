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
    # Core processing (compute seasonal trends)
    malca ltv.core --mag-bin 13_13.5 --output ltv_output.parquet
    
    # Full pipeline (filtering + crossmatch + NEOWISE + extinction)
    from malca.ltv import run_full_pipeline
    df_candidates = run_full_pipeline(df_ltv, verbose=True)
"""

from malca.ltv.core import (
    process_one_lc,
    compute_trend_metrics,
    compute_lomb_scargle,
    seasonal_midpoints_from_ra,
    assign_seasons_strict,
    season_medians_with_gap_indices,
)

from malca.ltv.filter import (
    filter_slope_threshold,
    filter_max_diff_threshold,
    filter_south_pole,
    filter_refcat_offset,
    filter_photometric_scatter,
    filter_crowding,
    filter_high_proper_motion,
    filter_neighbor_high_pm,
    apply_all_filters,
)

from malca.ltv.crossmatch import (
    load_local_catalog,
    merge_local_catalog,
    crossmatch_from_local,
    crossmatch_gaia_alerts,
    crossmatch_milliquas,
    query_simbad_classification,
    crossmatch_all_catalogs,
    crossmatch_tap_catalog,
)

from malca.ltv.neowise import (
    query_neowise_lc,
    combine_epochs,
    fit_neowise_trends,
    extract_neowise_trends,
)

from malca.ltv.dust import (
    apply_dust_flags,
)

from malca.ltv.cmd import (
    load_mist_grid,
    compute_cmd_features,
    assign_cmd_groups,
    fetch_bailer_jones_distances,
)

from malca.ltv.gaia_epoch import (
    query_gaia_epoch_photometry_batch,
    apply_gaia_epoch_flags,
)
from malca.ltv.stochastic import (
    add_stochastic_postfilter_features,
)
from malca.ltv.pipeline import (
    run_full_pipeline,
)
from malca.ltv.injection import (
    inject_trend,
    run_injection_recovery as run_ltv_injection_recovery,
    compute_rejection_summary,
    generate_plots as generate_ltv_injection_plots,
)

from malca.ltv.optim import (
    check_optimizations,
    cached,
    clear_cache,
    get_pooled_session,
)

from malca.ltv.review import (
    map_ltv_columns,
    ingest_ltv_results,
)


__all__ = [
    # Core
    "process_one_lc",
    "compute_trend_metrics",
    "compute_lomb_scargle",
    "seasonal_midpoints_from_ra",
    "assign_seasons_strict",
    "season_medians_with_gap_indices",
    # Filter
    "filter_slope_threshold",
    "filter_max_diff_threshold",
    "filter_south_pole",
    "filter_refcat_offset",
    "filter_photometric_scatter",
    "filter_crowding",
    "filter_high_proper_motion",
    "filter_neighbor_high_pm",
    "apply_all_filters",
    # Crossmatch (API only — Gaia DR3 and VSX use local catalog)
    "crossmatch_gaia_alerts",
    "crossmatch_milliquas",
    "query_simbad_classification",
    "crossmatch_all_catalogs",
    "crossmatch_tap_catalog",
    # NEOWISE
    "query_neowise_lc",
    "combine_epochs",
    "fit_neowise_trends",
    "extract_neowise_trends",
    # Dust flags
    "apply_dust_flags",
    # CMD scaffolding
    "load_mist_grid",
    "compute_cmd_features",
    "assign_cmd_groups",
    # Gaia epoch photometry
    "query_gaia_epoch_photometry_batch",
    "apply_gaia_epoch_flags",
    # Stochastic post-filter features
    "add_stochastic_postfilter_features",
    # Pipeline
    "run_full_pipeline",
    "inject_trend",
    "run_ltv_injection_recovery",
    "compute_rejection_summary",
    "generate_ltv_injection_plots",
    # Optimization
    "check_optimizations",
    # Local catalog
    "load_local_catalog",
    "merge_local_catalog",
    "crossmatch_from_local",
    # Review DB ingest
    "map_ltv_columns",
    "ingest_ltv_results",
]
