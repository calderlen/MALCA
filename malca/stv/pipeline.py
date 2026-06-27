#!/usr/bin/env python3
"""
Wrapper script to run events.py on tagged light curves.

Workflow:
1. Build/load manifest (source_id → lc_dir mapping)
2. Apply tags (sparse, periodic, multi-camera)
3. Construct file paths for kept sources
4. Pass to events.py
5. [Optional] Apply filters (posterior strength, run robustness, etc.)
6. [Optional] Generate postprocess plots for passing candidates
7. [Optional] Run characterization (Gaia DR3 + dust extinction)
8. [Optional] Run classification (EB/CV/starspot rejection, YSO classification)
9. [Optional] Enrich passing candidates with comprehensive light curve stats

Usage:
    malca stv-pipeline --mag-bin 13_13.5 [options...]
    malca stv-pipeline --mag-bin 13_13.5 14_14.5 [options...]  # Process multiple bins together
    malca stv-pipeline --mag-bin all [options...]  # Process all 6 magnitude bins together
    malca stv-pipeline --mag-bin 13_13.5 --run-filter --run-classify --run-enrich
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from tqdm.auto import tqdm
import pandas as pd

from malca.enrichment.characterize import characterize_candidates_df
from malca.enrichment.classify import compute_all_classifications
from malca.config import (
    GAIA_CHUNK_SIZE, NEIGHBOR_RADIUS_ARCSEC, NEIGHBOR_CHUNK_SIZE,
    SPECTRA_RADIUS_ARCSEC, SPECTRA_CHUNK_SIZE,
    UNWISE_CHECKPOINT_EVERY,
)
from malca.products.candidates import ensure_candidate_id, merge_candidate_columns, select_passing_candidates
from malca.config import (
    MIN_TIME_SPAN, MIN_POINTS_PER_DAY, MIN_CAMERAS,
    VSX_MAX_SEP_ARCSEC, CAMERA_MEDIAN_TOLERANCE, STATS_CHUNK_SIZE,
    MIN_BAYES_FACTOR, POST_FILTER_MIN_RUN_CAMERAS, POST_FILTER_MIN_RUN_POINTS,
    CLEAN_LC_MAX_ERROR_ABSOLUTE, CLEAN_LC_MAX_ERROR_SIGMA,
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    PRE_PERIODICITY_CE_SNR_THRESHOLD, PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_POINTS, PRE_PERIODICITY_MIN_PERIOD,
    PRE_PERIODICITY_N_PERIODS, PRE_PERIODICITY_SCATTER_RATIO_MAX,
    POST_FILTER_PDM_METHOD,
)
from malca.cli_config import add_config_args, apply_config, namespace_keys
from malca.config import EVENTS_OUTPUT_CHUNK_SIZE
from malca.config import PARQUET_OUTPUT_COMPRESSION, PARQUET_CACHE_COMPRESSION
from malca.config import ASASSN_INDEX_PATH, LCV2_ROOT, VSX_CROSSMATCH_PATH, GAIA_LOCAL_CATALOG, DEFAULT_OUTPUT_DIR
from malca.config import (
    WORKERS, BATCH_SIZE, TRIGGER_MODE, P_POINTS, MAG_POINTS,
    LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP, SIGNIFICANCE_THRESHOLD,
    MIN_MAG_OFFSET, RUN_MIN_POINTS, RUN_MAX_GAP_POINTS,
    BASELINE_FUNC, BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
    JD_OFFSET, MAG_BINS,
)
from malca.config import PDM_METHOD_CHOICES
from malca.enrich.neighbor import run_neighbor_enrichment
from malca.enrich.spectra import run_spectra_availability
from malca.catalogs.gaia_fetch import _extract_gaia_ids, fetch_gaia_catalog
from malca.catalogs.gaia_ids import canonicalize_gaia_ids_in_frame, normalize_gaia_source_id_series
from malca.io.manifest import build_manifest
from malca.products.product_schema import add_stv_identity, assert_stv_product_schema
from malca.stv.periodicity_gate import apply_pre_periodicity_gate, PREGATE_ROUTER_MODE
from malca.stv.plot import plot_passing_candidates
from malca.stv.filter import apply_filters
from malca.products.run_bundle import collect_candidate_lightcurve_files, export_run_bundle, import_bundle_zip
from malca.products.run_context import (
    init_pipeline_run_context,
    maybe_sync_review_bundle,
    run_dir_from_bundle,
    write_run_log,
    write_run_params,
    write_run_summary,
)
from malca.products.run_metadata import (
    build_fingerprint,
    build_run_summary,
    fingerprint_digest,
    json_stable,
    load_summary_state,
    preserve_imported_run_snapshots,
)
from malca.review.store import db_connect, import_candidates
from concurrent.futures import ProcessPoolExecutor
from malca.core.stats import compute_stats, _enrich_row_worker
from malca.stv.tag import RAW_MEDIAN_SUSPECT_COL, apply_tags, filter_camera_medians
from malca.products.feature_layers import to_layer_first_frame
from malca.io.table_io import (
    read_feature_table,
    read_passing_feature_table,
    require_parquet_path,
    write_feature_table,
    write_parquet_table,
)
from malca.core.utils import log as _log
from malca.enrichment.vetting import vet_candidates


RUN_REUSE_FINGERPRINT_VERSION = 1
RUN_REUSE_PARAM_ATTRS = (
    # Input selection
    "mag_bin",
    "index_root",
    "lc_root",
    "flat_lc_dir",
    "index_file",
    "manifest_file",
    "filtered_file",
    "output",
    "import_bundle",
    "extension",
    "test_run",
    "test_run_n",
    # Tag parameters
    "min_time_span",
    "min_points_per_day",
    "min_cameras",
    "mag_lo",
    "mag_hi",
    "skip_sparse",
    "skip_multi_camera",
    "skip_mag_range",
    "skip_vsx",
    "vsx_max_sep",
    "vsx_crossmatch",
    "pass_all_tags",
    "enforce_tags",
    "skip_camera_median",
    "camera_median_tolerance",
    # Pre-events periodicity gate
    "apply_pre_periodicity_gate",
    "pre_periodicity_min_period",
    "pre_periodicity_max_period",
    "pre_periodicity_n_periods",
    "pre_periodicity_ce_snr_threshold",
    "pre_periodicity_min_points",
    "pre_periodicity_scatter_ratio_max",
    "pre_periodicity_checkpoint",
    # Event detection
    "trigger_mode",
    "logbf_threshold_dip",
    "logbf_threshold_jump",
    "significance_threshold",
    "p_points",
    "p_min_dip",
    "p_max_dip",
    "p_min_jump",
    "p_max_jump",
    "mag_points",
    "mag_min_dip",
    "mag_max_dip",
    "mag_min_jump",
    "mag_max_jump",
    "baseline_func",
    "baseline_s0",
    "baseline_w0",
    "baseline_q",
    "baseline_jitter",
    "baseline_sigma_floor",
    "run_min_points",
    "run_max_gap_points",
    "run_max_gap_days",
    "run_min_duration_days",
    "no_event_prob",
    "min_mag_offset",
    # Filter and validation
    "run_filter",
    "skip_evidence_strength",
    "min_bayes_factor",
    "allow_infinite_local_bf",
    "skip_significant_detection",
    "significant_no_require_flag",
    "significant_min_peak_count",
    "significant_min_run_count",
    "skip_run_robustness",
    "min_run_count",
    "max_run_count",
    "filter_min_run_cameras",
    "filter_min_run_points",
    "apply_morphology",
    "dip_morphology",
    "jump_morphology",
    "min_delta_bic",
    "apply_score_filter",
    "min_dip_score",
    "min_jump_score",
    "min_score",
    "apply_periodicity_validation",
    "periodicity_n_bootstrap",
    "periodicity_significance",
    "periodicity_pdm_method",
    "periodicity_no_exclude_aliases",
    "periodicity_reject",
    "periodicity_checkpoint_dir",
    "periodicity_all_candidates",
    "phase_plot_max_sig",
    "phase_plot_min_power",
    "phase_plot_allow_alias",
    "skip_gaia_ruwe_validation",
    "gaia_max_ruwe",
    "gaia_reject",
    "skip_gaia_pm_validation",
    "gaia_max_pm",
    "gaia_pm_reject",
    "auto_fetch_gaia_cache",
    "gaia_fetch_passers_only",
    "external_validations_passers_only",
    "skip_periodic_catalog_validation",
    "periodic_catalog_max_sep",
    "periodic_catalog_reject",
    # Downstream science products
    "run_characterize",
    "run_dust",
    "gaia_cache",
    "gaia_fetch_chunk_size",
    "characterize_crossmatch",
    "characterize_chunk_size",
    "characterize_starhorse",
    "characterize_starhorse_cache",
    "characterize_unwise_checkpoint_every",
    "characterize_banyan",
    "characterize_iphas",
    "characterize_sfr",
    "characterize_clusters",
    "characterize_unwise",
    "run_sed_photometry",
    "sed_sources",
    "fit_atmosphere",
    "run_classify",
    "run_enrich",
    "enrich_workers",
    "enrich_compute_ls",
    "run_neighbor_enrich",
    "neighbor_radius_arcsec",
    "neighbor_cache",
    "run_spectra_enrich",
    "spectra_radius_arcsec",
    "spectra_cache",
    "run_vetting",
    "vetting_min_score",
    "vetting_simbad_radius",
    "vetting_asassn_radius",
    "no_vetting_simbad",
    "no_vetting_gaia_var",
    "no_vetting_gaia_epoch",
    "no_vetting_asassn_var",
    "no_vetting_alerce",
    "no_vetting_erosita",
    "no_vetting_pm_check",
    "vetting_atlas",
    "vetting_atlas_token",
    "vetting_neowise_lc",
    "vetting_input",
    "run_external_lcs",
    "run_multi_survey_features",
    "external_lc_workers",
    "external_lc_refresh_cache",
    "external_lc_atlas",
    "external_lc_atlas_token",
)

RUN_REUSE_CODE_FILES = (
    "config.py",
    "cli_config.py",
    "manifest.py",
    "stv/pipeline.py",
    "stv/tag.py",
    "stv/events.py",
    "baseline.py",
    "stv/triggering.py",
    "stv/score.py",
    "stv/filter.py",
    "stv/periodicity_gate.py",
    "utils.py",
    "characterize.py",
    "classify.py",
    "stats.py",
    "enrich/neighbor.py",
    "enrich/spectra.py",
    "review/sed.py",
    "sed_model.py",
    "sed_photometry.py",
    "vetting.py",
)

PIPELINE_STAGE_CHOICES = ("full", "cluster", "home", "full-extended")


def _stage_runs_upstream(stage: str) -> bool:
    return str(stage) in {"full", "cluster", "full-extended"}


def _stage_runs_downstream(stage: str) -> bool:
    return str(stage) in {"full", "home", "full-extended"}


def _stage_defaults_to_extended_enrichment(stage: str) -> bool:
    return str(stage) == "full-extended"


# Set threading environment variables before importing numpy/pandas/numba
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")


PIPELINE_CONFIG_DEFAULTS: dict[str, Any] = {
    "manifest_file": None,
    "filtered_file": None,
    "force_manifest": False,
    "force_tag": False,
    "extension": None,
    "min_time_span": MIN_TIME_SPAN,
    "min_points_per_day": MIN_POINTS_PER_DAY,
    "min_cameras": MIN_CAMERAS,
    "mag_lo": 10.0,
    "mag_hi": 18.0,
    "skip_sparse": False,
    "skip_multi_camera": False,
    "skip_mag_range": False,
    "skip_vsx": False,
    "skip_camera_median": False,
    "camera_median_tolerance": CAMERA_MEDIAN_TOLERANCE,
    "vsx_max_sep": VSX_MAX_SEP_ARCSEC,
    "vsx_crossmatch": VSX_CROSSMATCH_PATH,
    "pass_all_tags": False,
    "enforce_tags": None,
    "stats_chunk_size": STATS_CHUNK_SIZE,
    "batch_size": BATCH_SIZE,
    "apply_pre_periodicity_gate": False,
    "pre_periodicity_min_period": PRE_PERIODICITY_MIN_PERIOD,
    "pre_periodicity_max_period": PRE_PERIODICITY_MAX_PERIOD,
    "pre_periodicity_n_periods": PRE_PERIODICITY_N_PERIODS,
    "pre_periodicity_ce_snr_threshold": PRE_PERIODICITY_CE_SNR_THRESHOLD,
    "pre_periodicity_min_points": PRE_PERIODICITY_MIN_POINTS,
    "pre_periodicity_scatter_ratio_max": PRE_PERIODICITY_SCATTER_RATIO_MAX,
    "pre_periodicity_checkpoint": None,
    "pre_periodicity_workers": WORKERS,
    "trigger_mode": TRIGGER_MODE,
    "logbf_threshold_dip": LOGBF_THRESHOLD_DIP,
    "logbf_threshold_jump": LOGBF_THRESHOLD_JUMP,
    "significance_threshold": SIGNIFICANCE_THRESHOLD,
    "p_points": P_POINTS,
    "mag_points": MAG_POINTS,
    "run_min_points": RUN_MIN_POINTS,
    "run_max_gap_points": RUN_MAX_GAP_POINTS,
    "run_max_gap_days": None,
    "run_min_duration_days": 0.0,
    "no_event_prob": False,
    "p_min_dip": None,
    "p_max_dip": None,
    "p_min_jump": None,
    "p_max_jump": None,
    "baseline_func": BASELINE_FUNC,
    "baseline_s0": BASELINE_S0,
    "baseline_w0": BASELINE_W0,
    "baseline_q": BASELINE_Q,
    "baseline_jitter": BASELINE_JITTER,
    "baseline_sigma_floor": None,
    "mag_min_dip": None,
    "mag_max_dip": None,
    "mag_min_jump": None,
    "mag_max_jump": None,
    "min_mag_offset": MIN_MAG_OFFSET,
    "output": None,
    "chunk_size": EVENTS_OUTPUT_CHUNK_SIZE,
    "import_bundle": None,
    "export_bundle": None,
    "export_bundle_enabled": True,
    "full_bundle": False,
    "review_sync_enabled": True,
    "review_sync_dir": Path("reviews"),
    "review_sync_hash_assets": False,
    "run_filter": True,
    "skip_evidence_strength": False,
    "min_bayes_factor": MIN_BAYES_FACTOR,
    "allow_infinite_local_bf": False,
    "skip_significant_detection": False,
    "significant_no_require_flag": False,
    "significant_min_peak_count": 1,
    "significant_min_run_count": 1,
    "skip_run_robustness": False,
    "min_run_count": 1,
    "max_run_count": None,
    "filter_min_run_cameras": POST_FILTER_MIN_RUN_CAMERAS,
    "filter_min_run_points": POST_FILTER_MIN_RUN_POINTS,
    "apply_morphology": False,
    "dip_morphology": "gaussian",
    "jump_morphology": "paczynski",
    "min_delta_bic": 10.0,
    "apply_score_filter": True,
    "min_score": 0.0,
    "min_dip_score": None,
    "min_jump_score": None,
    "apply_periodicity_validation": False,
    "periodicity_n_bootstrap": 1000,
    "periodicity_significance": 0.01,
    "periodicity_pdm_method": POST_FILTER_PDM_METHOD,
    "periodicity_no_exclude_aliases": False,
    "periodicity_reject": False,
    "periodicity_all_candidates": False,
    "periodicity_workers": WORKERS,
    "periodicity_checkpoint_dir": None,
    "phase_plot_max_sig": 0.01,
    "phase_plot_min_power": 0.3,
    "phase_plot_allow_alias": False,
    "skip_gaia_ruwe_validation": False,
    "gaia_max_ruwe": 1.4,
    "gaia_reject": False,
    "skip_gaia_pm_validation": False,
    "gaia_max_pm": 100.0,
    "gaia_pm_reject": False,
    "auto_fetch_gaia_cache": True,
    "gaia_fetch_passers_only": True,
    "external_validations_passers_only": True,
    "skip_periodic_catalog_validation": False,
    "periodic_catalog_max_sep": 3.0,
    "periodic_catalog_reject": False,
    "run_postprocess": False,
    "max_plots": None,
    "plot_format": "png",
    "run_characterize": True,
    "gaia_cache": None,
    "gaia_fetch_chunk_size": GAIA_CHUNK_SIZE,
    "characterize_crossmatch": VSX_CROSSMATCH_PATH,
    "characterize_chunk_size": GAIA_CHUNK_SIZE,
    "characterize_starhorse": "tap",
    "characterize_starhorse_cache": None,
    "characterize_unwise_checkpoint_every": UNWISE_CHECKPOINT_EVERY,
    "characterize_banyan": True,
    "characterize_iphas": True,
    "characterize_sfr": True,
    "characterize_clusters": True,
    "characterize_unwise": False,
    "run_dust": True,
    "run_sed_photometry": True,
    "sed_sources": "default",
    "fit_atmosphere": True,
    "run_classify": True,
    "run_enrich": True,
    "enrich_workers": None,
    "enrich_compute_ls": False,
    "run_neighbor_enrich": True,
    "neighbor_radius_arcsec": NEIGHBOR_RADIUS_ARCSEC,
    "neighbor_chunk_size": NEIGHBOR_CHUNK_SIZE,
    "neighbor_cache": None,
    "run_spectra_enrich": True,
    "spectra_radius_arcsec": SPECTRA_RADIUS_ARCSEC,
    "spectra_chunk_size": SPECTRA_CHUNK_SIZE,
    "spectra_cache": None,
    "run_vetting": True,
    "vetting_min_score": None,
    "vetting_simbad_radius": 5.0,
    "vetting_asassn_radius": 5.0,
    "no_vetting_simbad": False,
    "no_vetting_gaia_var": False,
    "no_vetting_gaia_epoch": False,
    "no_vetting_asassn_var": False,
    "no_vetting_alerce": False,
    "no_vetting_erosita": False,
    "no_vetting_pm_check": False,
    "vetting_atlas": False,
    "vetting_atlas_token": None,
    "vetting_neowise_lc": False,
    "vetting_input": None,
    "run_external_lcs": False,
    "run_multi_survey_features": False,
    "external_lc_workers": 4,
    "external_lc_refresh_cache": False,
    "external_lc_atlas": False,
    "external_lc_atlas_token": None,
    "test_run": False,
    "test_run_n": 10000,
}

PIPELINE_CONFIG_PATH_KEYS = {
    "flat_lc_dir",
    "index_file",
    "manifest_file",
    "filtered_file",
    "vsx_crossmatch",
    "pre_periodicity_checkpoint",
    "output",
    "import_bundle",
    "export_bundle",
    "review_sync_dir",
    "periodicity_checkpoint_dir",
    "gaia_cache",
    "characterize_crossmatch",
    "characterize_starhorse_cache",
    "neighbor_cache",
    "spectra_cache",
    "vetting_input",
}



def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically to avoid corruption on interruption."""
    write_parquet_table(df, path, compression=PARQUET_OUTPUT_COMPRESSION)


def load_table(path: Path) -> pd.DataFrame:
    return read_feature_table(path)


def load_passing_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    return _select_passing_candidates(read_passing_feature_table(path, columns=columns))


def _effective_enrich_workers(args: argparse.Namespace) -> tuple[int, str | None]:
    """Return worker count for compute_stats enrichment and optional log note."""
    explicit_workers = getattr(args, "enrich_workers", None)
    if explicit_workers is not None:
        return max(1, int(explicit_workers)), None

    general_workers = max(1, int(getattr(args, "workers", WORKERS)))
    capped_workers = min(general_workers, 8)
    if capped_workers < general_workers:
        return (
            capped_workers,
            f"--enrich-workers unset; capped from --workers={general_workers}",
        )
    return capped_workers, None


def build_run_reuse_fingerprint(
    args: argparse.Namespace,
    *,
    stage: str,
    is_auto_all_mode: bool,
) -> dict[str, Any]:
    """Build the science-product fingerprint required for implicit run reuse."""
    params = {
        name: json_stable(getattr(args, name, None))
        for name in RUN_REUSE_PARAM_ATTRS
    }
    params.update({
        "stage": stage,
        "is_auto_all_mode": bool(is_auto_all_mode),
        "pre_periodicity_router_mode": PREGATE_ROUTER_MODE,
        "clean_max_error_absolute": CLEAN_LC_MAX_ERROR_ABSOLUTE,
        "clean_max_error_sigma": CLEAN_LC_MAX_ERROR_SIGMA,
        "bad_camera_scatter_ratio": BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    })
    return build_fingerprint(
        version=RUN_REUSE_FINGERPRINT_VERSION,
        params=params,
        code_base=Path(__file__).resolve().parent.parent,
        code_paths=RUN_REUSE_CODE_FILES,
    )


def run_reuse_fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    return fingerprint_digest(fingerprint)


def _stored_run_reuse_fingerprint_matches(params: dict[str, Any], current_fingerprint: dict[str, Any]) -> bool:
    stored_fingerprint = params.get("run_reuse_fingerprint")
    if not isinstance(stored_fingerprint, dict):
        return False
    return json_stable(stored_fingerprint) == json_stable(current_fingerprint)


def _score_filter_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "apply_score_filter", getattr(args, "apply_score", False)))


def _gaia_cache_arg(args: argparse.Namespace) -> Path:
    return getattr(args, "gaia_cache", None) or GAIA_LOCAL_CATALOG


def _config_arg(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None:
        return PIPELINE_CONFIG_DEFAULTS[name]
    return value


def save_table(df: pd.DataFrame, path: Path) -> None:
    if Path(path).name.startswith("lc_events_"):
        df = to_layer_first_frame(add_stv_identity(df))
        assert_stv_product_schema(df, stage=Path(path).stem)
        write_feature_table(df, require_parquet_path(path), compression=PARQUET_OUTPUT_COMPRESSION)
    else:
        safe_write_parquet(df, require_parquet_path(path))


def _run_external_lcs_enrichment(
    df_input: pd.DataFrame,
    *,
    results_dir: Path,
    atlas: bool,
    atlas_token: str | None,
    workers: int,
    refresh_cache: bool,
    overwrite: bool = False,
) -> tuple[Path, Path, pd.DataFrame]:
    """Fetch safe-default external light curves and write the enriched table."""
    from malca.enrichment.vetting import fetch_external_lcs

    external_lc_dir = results_dir / "external_lcs"
    external_lc_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = external_lc_dir / "external_lcs_CHECKPOINT.parquet"
    if overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()
    output_path = results_dir / "lc_events_external_lcs.parquet"

    df_run = _ensure_candidate_id_column(_select_passing_candidates(df_input))
    df_out = fetch_external_lcs(
        df_run,
        output_dir=external_lc_dir,
        run_atlas=bool(atlas),
        run_ztf=True,
        run_gaia_epoch=True,
        run_tess=True,
        run_neowise=True,
        run_kepler=True,
        run_aavso=True,
        run_ogle=True,
        run_stripe82=True,
        run_allwise_mep=True,
        run_vvvx_virac=True,
        run_ps1=True,
        run_crts=True,
        atlas_token=atlas_token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN"),
        workers=int(workers or 4),
        checkpoint_path=checkpoint_path,
        refresh_cache=bool(refresh_cache),
    )
    save_table(df_out, output_path)
    return output_path, external_lc_dir, df_out


def _run_multi_survey_features_enrichment(
    df_input: pd.DataFrame,
    *,
    results_dir: Path,
    external_lc_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    """Compute event-relative multi-survey features and write the enriched table."""
    from malca.enrichment.multi_survey_features import compute_multi_survey_features

    output_path = results_dir / "lc_events_multi_survey_features.parquet"
    df_run = _ensure_candidate_id_column(_select_passing_candidates(df_input))
    df_out = compute_multi_survey_features(df_run, external_lc_dir=external_lc_dir)
    save_table(df_out, output_path)
    return output_path, df_out


def _select_passing_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with failed_any == False when that column exists."""
    return select_passing_candidates(df)


def _candidate_result_priority(results_dir: Path, *, include_extended: bool = True) -> list[Path]:
    """Return candidate result files from most to least enriched."""
    paths: list[Path] = []
    if include_extended:
        paths.extend([
            results_dir / "lc_events_multi_survey_features.parquet",
            results_dir / "lc_events_external_lcs.parquet",
        ])
    paths.extend([
        results_dir / "lc_events_vetted.parquet",
        results_dir / "lc_events_spectra.parquet",
        results_dir / "lc_events_neighbors.parquet",
        results_dir / "lc_events_classified.parquet",
        results_dir / "lc_events_characterized.parquet",
        results_dir / "lc_events_filtered.parquet",
    ])
    return paths


def _first_existing_candidate_result(results_dir: Path, *, include_extended: bool = True) -> Path | None:
    for candidate_path in _candidate_result_priority(results_dir, include_extended=include_extended):
        if candidate_path.exists():
            return candidate_path
    return None


def _copy_single_tagged_table_output(tagged_outputs: list[Path], merged_path: Path) -> bool:
    """Copy a single tagged Parquet table to its canonical path without loading it."""
    if len(tagged_outputs) != 1:
        return False
    source = tagged_outputs[0]
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != merged_path.resolve():
        shutil.copy2(source, merged_path)
    return True


def _ensure_candidate_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure downstream enrichment has a candidate_id key when one can be inferred."""
    return add_stv_identity(df)


def _branch_events_attempted_this_run(branch_detection_stats: dict[str, object] | None) -> int | None:
    """Return how many event inputs were attempted in this wrapper invocation."""
    if not isinstance(branch_detection_stats, dict):
        return None

    total = 0
    found = False
    for branch_stats in branch_detection_stats.values():
        if not isinstance(branch_stats, dict) or "attempted_this_run" not in branch_stats:
            continue
        try:
            total += int(branch_stats.get("attempted_this_run") or 0)
            found = True
        except (TypeError, ValueError):
            return None
    return total if found else None


def _should_skip_filter_stage(
    *,
    output_path: Path,
    overwrite: bool,
    branch_detection_stats: dict[str, object] | None,
) -> bool:
    """Skip filtering only when the existing filtered product is still current."""
    if overwrite or not output_path.exists():
        return False

    attempted_this_run = _branch_events_attempted_this_run(branch_detection_stats)
    return attempted_this_run == 0


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            key = str(path.resolve(strict=False))
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique



def _candidate_asassn_index_paths(out_dir: Path, index_override: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir).expanduser()
    default_index = ASASSN_INDEX_PATH.expanduser()
    output_root = _output_root_from_run_dir(out_dir)

    candidates: list[Path] = []
    if index_override is not None:
        candidates.append(Path(index_override).expanduser())

    candidates.extend([
        out_dir / "bundle_assets" / "asassn_index_full.parquet",
        out_dir / "bundle_assets" / default_index.name,
        out_dir / default_index.name,
        out_dir / "input" / default_index.name,
        output_root / default_index.name,
        default_index,
    ])

    search_dirs = [
        out_dir / "bundle_assets",
        out_dir / "input",
        out_dir,
        output_root,
        Path("input"),
        DEFAULT_OUTPUT_DIR,
    ]
    for search_dir in _unique_paths(search_dirs):
        if not search_dir.exists() or (not search_dir.is_dir()):
            continue
        for pattern in ("asassn_index*.parquet",):
            candidates.extend(sorted(search_dir.glob(pattern)))

    return _unique_paths(candidates)


def _output_root_from_run_dir(out_dir: Path) -> Path:
    path = Path(out_dir).expanduser()
    if path.parent.name in {"stv", "ltv"} and path.parent.parent.name == "runs":
        return path.parent.parent.parent
    if path.parent.name == "runs":
        return path.parent.parent
    return path.parent


def _resolve_asassn_index_path(out_dir: Path, index_override: Path | None = None) -> tuple[Path | None, list[Path]]:
    candidates = _candidate_asassn_index_paths(out_dir, index_override=index_override)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate, candidates
    return None, candidates


def default_run_dir(base_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_root / "runs" / "stv" / timestamp


def find_latest_run_dir(
    base_root: Path,
    mag_bin: list[str],
    reuse_fingerprint: dict[str, Any] | None = None,
) -> Path | None:
    """Find the newest run directory safe to reuse for the current configuration."""
    runs_dir = base_root / "runs" / "stv"
    if not runs_dir.is_dir():
        return None
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        params_file = d / "run_params.json"
        if not params_file.exists():
            continue
        try:
            with open(params_file) as f:
                params = json.load(f)
        except Exception:
            continue
        if params.get("mag_bin") == mag_bin:
            if reuse_fingerprint is not None and not _stored_run_reuse_fingerprint_matches(params, reuse_fingerprint):
                continue
            return d  # sorted reverse by timestamp, first match is latest
    return None


def _normalize_mag_bins(raw_value: Any) -> list[str] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else None
    if isinstance(raw_value, (list, tuple)):
        values = [str(v).strip() for v in raw_value if str(v).strip()]
        return values or None
    return None


def _read_mag_bins_from_params_file(params_file: Path) -> list[str] | None:
    if not params_file.exists():
        return None
    try:
        with params_file.open() as f:
            params = json.load(f)
    except Exception:
        return None
    return _normalize_mag_bins(params.get("mag_bin"))


_BUNDLE_MAG_BIN_RE = re.compile(
    r"^results/lc_events_(?:results|filtered|enriched)_([0-9.]+_[0-9.]+)\.parquet$"
)


def _infer_mag_bins_from_bundle_contents(zf: zipfile.ZipFile) -> list[str] | None:
    tags: set[str] = set()
    for name in zf.namelist():
        m = _BUNDLE_MAG_BIN_RE.match(str(name))
        if not m:
            continue
        tag = str(m.group(1))
        if tag and tag != "multi":
            tags.add(tag)
    if not tags:
        return None
    return sorted(tags)


def _read_mag_bins_from_bundle(bundle_zip: Path) -> list[str] | None:
    bundle_zip = Path(bundle_zip).expanduser()
    if not bundle_zip.exists() or (not zipfile.is_zipfile(bundle_zip)):
        return None
    try:
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            inferred = _infer_mag_bins_from_bundle_contents(zf)
            params = None
            try:
                with zf.open("run_params.json") as f:
                    params = json.load(f)
            except Exception:
                params = None
    except Exception:
        return None

    from_params = _normalize_mag_bins(params.get("mag_bin")) if isinstance(params, dict) else None

    # Prefer inference from bundled per-mag-bin result filenames, since run_params.json can be
    # overwritten when multiple mag bins share one out_dir (concurrent runs).
    if inferred:
        return inferred
    return from_params


def _assert_mag_bin_match(expected: list[str], observed: list[str], source: str) -> None:
    if expected != observed:
        raise SystemExit(
            f"Provided --mag-bin ({observed}) does not match {source} ({expected})."
        )


def get_out_dir_from_bundle(bundle_path: Path, base_root: Path, *, overwrite: bool = False) -> Path:
    """Extract run directory name from bundle filename.

    If the derived run directory already exists:
      - return it directly when ``overwrite`` is True
      - otherwise return a ``_home``-suffixed directory for safety
    """
    runs_dir = base_root / "runs" / "stv"
    return run_dir_from_bundle(bundle_path, runs_dir, collision_suffix="_home", overwrite=overwrite)


def clear_existing_output(path: Path | None, fmt: str) -> None:
    if path is None or (not path.exists()):
        return
    if fmt == "parquet_chunk" and path.is_dir():
        removed_any = False
        for child in path.glob("chunk_*.parquet*"):
            child.unlink()
            removed_any = True
        if removed_any:
            print(f"Overwriting existing output chunks in {path}")
        return

    path.unlink()
    print(f"Overwriting existing output file: {path}")


def _collect_bundle_lightcurve_files(out_dir: Path, mag_bin_tag: str | None = None, include_all: bool = False) -> list[tuple[Path, str]]:
    """Collect candidate .dat2/.raw2 files to include in bundle assets.

    By default only includes light curves for candidates that passed all
    filters (failed_any=False). Pass include_all=True to bundle every
    candidate regardless of filter outcome. Source files are read directly from
    their original location and are never modified in place.
    """
    if mag_bin_tag:
        filtered_candidates = out_dir / "results" / f"lc_events_filtered_{mag_bin_tag}.parquet"
    else:
        filtered_candidates = out_dir / "results" / "lc_events_filtered.parquet"
    if not filtered_candidates.exists():
        return []

    try:
        if include_all:
            df_candidates = load_table(filtered_candidates)
        else:
            df_candidates = load_passing_table(filtered_candidates)
    except Exception as exc:
        print(f"Warning: could not read {filtered_candidates} for light curve bundling: {exc}")
        return []

    if "lc_path" not in df_candidates.columns:
        return []

    if not include_all:
        print(f"Bundling light curves for {len(df_candidates)} passing candidates (failed_any=False)")

    return collect_candidate_lightcurve_files(
        df_candidates,
        path_cols=("lc_path",),
        arc_prefix="bundle_assets/lightcurves",
        allowed_suffix_prefixes=("dat",),
        sidecar_suffixes=(".raw2",),
    ).files


def export_bundle_zip(bundle_zip: Path, out_dir: Path, include_all: bool = False, mag_bin_tag: str | None = None) -> list[str]:
    """Create transfer bundle zip from a pipeline out_dir."""
    include_rel_paths = [
        "run_params.json",
        "run_summary.json",
        "run.log",
        "results/lc_events_filtered.parquet",
        "results/lc_events_enriched.parquet",
        "results/lc_events_characterized.parquet",
        "results/lc_events_classified.parquet",
        "results/lc_events_neighbors.parquet",
        "results/lc_events_spectra.parquet",
        "results/lc_events_vetted.parquet",
        "results/lc_events_external_lcs.parquet",
        "results/lc_events_multi_survey_features.parquet",
        "results/sed_photometry.parquet",
        "results/sed_model_fits.parquet",
        "results/sed_model_curves.parquet",
    ]
    if mag_bin_tag:
        include_rel_paths.extend([
            f"run_params_{mag_bin_tag}.json",
            f"run_{mag_bin_tag}.log",
            f"results/lc_events_filtered_{mag_bin_tag}.parquet",
            f"results/lc_events_enriched_{mag_bin_tag}.parquet",
        ])
    if include_all:
        include_rel_paths.append("bundle_assets/asassn_index_full.parquet")
    include_globs = [
        f"results/lc_events_results_{mag_bin_tag}*" if mag_bin_tag else "results/lc_events_results*",
    ]
    include_dirs = [
        "plots",
        "results/external_lcs",
    ]
    if include_all:
        include_dirs.extend(["manifests", "tags", "paths", "gaia_cache"])
    lightcurve_files = _collect_bundle_lightcurve_files(out_dir, mag_bin_tag=mag_bin_tag, include_all=include_all)
    bundled_paths = export_run_bundle(
        bundle_zip,
        out_dir,
        include_files=include_rel_paths,
        include_globs=include_globs,
        include_dirs=include_dirs,
        external_files=lightcurve_files,
        description="STV run",
    )
    print(f"Bundled {len(bundled_paths)} files with ZIP_DEFLATED compression.")
    return bundled_paths


def _add_gaia_ids_from_index(df_events: pd.DataFrame, index_path) -> pd.DataFrame:
    """
    Merge gaia_id and asas_sn_id from the ASASSN index into the events DataFrame.

    The ASASSN index covers all ~17M ASAS-SN sources and carries a Gaia ID for
    each one, so almost every candidate should receive a gaia_id after this merge.
    The VSX crossmatch only covers ~99K known variables and must not be used here.

    Parameters
    ----------
    df_events : pd.DataFrame
        Events DataFrame from events.py (must have 'lc_path' column).
    index_path : Path or str
        Path to the ASASSN index parquet (or CSV) file.

    Returns
    -------
    pd.DataFrame
        Events DataFrame with gaia_id and asas_sn_id columns added
        (NaN for the rare unmatched sources).
    """
    if "lc_path" not in df_events.columns:
        _log("Warning: Cannot add gaia_id - 'lc_path' column not found")
        return df_events

    if not Path(index_path).exists():
        _log(f"Warning: ASASSN index not found at {index_path}")
        return df_events

    try:
        df = df_events.copy()

        # Derive asas_sn_id from the LC filename stem (e.g. "498216332934.dat3" → 498216332934)
        def _extract_id(path_str):
            if pd.isna(path_str):
                return None
            try:
                return int(Path(str(path_str)).stem.split(".")[0])
            except Exception:
                return None

        df["asas_sn_id"] = df["lc_path"].apply(_extract_id)

        # Load only the columns we need from the index
        index_path = Path(index_path)
        _log(f"Loading ASASSN index from {index_path.name}...")
        df_index = pd.read_parquet(require_parquet_path(index_path), columns=["asas_sn_id", "gaia_id"])

        df_index["asas_sn_id"] = pd.to_numeric(df_index["asas_sn_id"], errors="coerce")
        df_index = df_index.dropna(subset=["asas_sn_id"])
        df_index["asas_sn_id"] = df_index["asas_sn_id"].astype("int64")
        df_index["gaia_id"] = normalize_gaia_source_id_series(df_index["gaia_id"])

        df_merged = df.merge(
            df_index[["asas_sn_id", "gaia_id"]].drop_duplicates(subset=["asas_sn_id"]),
            on="asas_sn_id",
            how="left",
        )

        n_with_gaia = df_merged["gaia_id"].notna().sum()
        n_total = len(df_merged)
        pct = 100.0 * n_with_gaia / n_total if n_total > 0 else 0.0
        _log(f"[gaia_id merge] Added gaia_id for {n_with_gaia}/{n_total} events ({pct:.2f}%)")
        df_merged = canonicalize_gaia_ids_in_frame(df_merged)
        if "gaia_id_mapping_status" in df_merged.columns:
            n_translated = int(df_merged["gaia_id_mapping_status"].astype(str).eq("dr2_translated").sum())
            if n_translated:
                _log(f"[gaia_id merge] Translated {n_translated} Gaia DR2 ID(s) to DR3")

        return df_merged
    except Exception as e:
        _log(f"Warning: Failed to merge gaia_id from index: {e}")
        return df_events


def _build_filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build apply_filters kwargs from detect CLI arguments."""
    gaia_cache = _gaia_cache_arg(args)
    auto_fetch_gaia_cache = bool(_config_arg(args, "auto_fetch_gaia_cache"))
    gaia_fetch_chunk_size = int(_config_arg(args, "gaia_fetch_chunk_size"))
    gaia_fetch_passers_only = bool(_config_arg(args, "gaia_fetch_passers_only"))
    external_validations_passers_only = bool(_config_arg(args, "external_validations_passers_only"))
    return {
        # Core filters
        "apply_evidence_strength": not args.skip_evidence_strength,
        "min_bayes_factor": args.min_bayes_factor,
        "require_finite_local_bf": not args.allow_infinite_local_bf,
        "apply_significant_detection": not args.skip_significant_detection,
        "significant_require_flag": not args.significant_no_require_flag,
        "significant_min_peak_count": args.significant_min_peak_count,
        "significant_min_run_count": args.significant_min_run_count,
        "apply_run_robustness": not args.skip_run_robustness,
        "min_run_count": args.min_run_count,
        "max_run_count": args.max_run_count,
        "min_run_points": args.filter_min_run_points,
        "min_run_cameras": args.filter_min_run_cameras,
        # Optional filters
        "apply_morphology": args.apply_morphology,
        "dip_morphology": args.dip_morphology,
        "jump_morphology": args.jump_morphology,
        "min_delta_bic": args.min_delta_bic,
        "apply_score": _score_filter_enabled(args),
        "min_dip_score": args.min_dip_score,
        "min_jump_score": args.min_jump_score,
        "min_score": args.min_score,
        # Validation filters
        "apply_periodicity_validation": args.apply_periodicity_validation,
        "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
        "periodicity_significance": args.periodicity_significance,
        "periodicity_pdm_method": args.periodicity_pdm_method,
        "periodicity_exclude_aliases": not args.periodicity_no_exclude_aliases,
        "periodicity_flag_only": not args.periodicity_reject,
        "periodicity_workers": args.periodicity_workers,
        "periodicity_checkpoint_dir": args.periodicity_checkpoint_dir,
        "periodicity_all_candidates": args.periodicity_all_candidates,
        "phase_plot_max_sig": args.phase_plot_max_sig,
        "phase_plot_min_power": args.phase_plot_min_power,
        "phase_plot_allow_alias": args.phase_plot_allow_alias,
        "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
        "gaia_max_ruwe": args.gaia_max_ruwe,
        "gaia_flag_only": not args.gaia_reject,
        "apply_gaia_pm_validation": not args.skip_gaia_pm_validation,
        "gaia_max_pm": args.gaia_max_pm,
        "gaia_pm_flag_only": not args.gaia_pm_reject,
        "gaia_catalog_path": gaia_cache,
        "auto_fetch_gaia_cache": auto_fetch_gaia_cache,
        "gaia_fetch_chunk_size": gaia_fetch_chunk_size,
        "gaia_fetch_passers_only": gaia_fetch_passers_only,
        "external_validations_passers_only": external_validations_passers_only,
        "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
        "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
        "periodic_catalog_flag_only": not args.periodic_catalog_reject,
        # Progress/logging
        "show_tqdm": args.verbose,
        "verbose": args.verbose,
    }


def _build_home_external_validation_cmd(
    args: argparse.Namespace,
    *,
    post_filter_output: Path,
    index_file: Path,
) -> list[str]:
    """Build the home-stage external validation subprocess command."""
    gaia_cache = _gaia_cache_arg(args)
    auto_fetch_gaia_cache = bool(_config_arg(args, "auto_fetch_gaia_cache"))
    gaia_fetch_chunk_size = int(_config_arg(args, "gaia_fetch_chunk_size"))
    gaia_fetch_passers_only = bool(_config_arg(args, "gaia_fetch_passers_only"))
    external_validations_passers_only = bool(_config_arg(args, "external_validations_passers_only"))
    cmd = [
        sys.executable,
        "-m",
        "malca.stv.filter",
        "--input",
        str(post_filter_output),
        "--output",
        str(post_filter_output),
        "--index-file",
        str(index_file),
        "--skip-evidence-strength",
        "--skip-significant-detection",
        "--skip-run-robustness",
        "--gaia-max-ruwe",
        str(args.gaia_max_ruwe),
        "--gaia-max-pm",
        str(args.gaia_max_pm),
        "--gaia-cache",
        str(gaia_cache),
        "--periodic-catalog-max-sep",
        str(args.periodic_catalog_max_sep),
    ]
    if not auto_fetch_gaia_cache:
        cmd.append("--no-auto-fetch-gaia-cache")
    if not gaia_fetch_passers_only:
        cmd.append("--gaia-fetch-all-candidates")
    if external_validations_passers_only:
        cmd.append("--external-validations-passers-only")
    else:
        cmd.append("--external-validations-all-candidates")
    cmd.extend(["--gaia-fetch-chunk-size", str(gaia_fetch_chunk_size)])

    if args.apply_periodicity_validation:
        cmd.extend(
            [
                "--apply-periodicity-validation",
                "--periodicity-n-bootstrap",
                str(args.periodicity_n_bootstrap),
                "--periodicity-significance",
                str(args.periodicity_significance),
                "--periodicity-pdm-method",
                str(args.periodicity_pdm_method),
                "--workers",
                str(args.periodicity_workers),
                "--phase-plot-max-sig",
                str(args.phase_plot_max_sig),
                "--phase-plot-min-power",
                str(args.phase_plot_min_power),
            ]
        )
        if args.periodicity_no_exclude_aliases:
            cmd.append("--periodicity-no-exclude-aliases")
        if args.periodicity_reject:
            cmd.append("--periodicity-reject")
        if args.periodicity_all_candidates:
            cmd.append("--periodicity-all-candidates")
        if args.phase_plot_allow_alias:
            cmd.append("--phase-plot-allow-alias")
        if args.periodicity_checkpoint_dir:
            cmd.extend(["--checkpoint-dir", str(args.periodicity_checkpoint_dir)])

    if args.gaia_reject:
        cmd.append("--gaia-reject")
    if args.gaia_pm_reject:
        cmd.append("--gaia-pm-reject")
    if args.periodic_catalog_reject:
        cmd.append("--periodic-catalog-reject")
    if args.skip_gaia_ruwe_validation:
        cmd.append("--skip-gaia-ruwe-validation")
    if args.skip_gaia_pm_validation:
        cmd.append("--skip-gaia-pm-validation")
    if args.skip_periodic_catalog_validation:
        cmd.append("--skip-periodic-catalog-validation")
    if not args.verbose:
        cmd.append("--no-progress")
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run events.py on tagged light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --config/--profile for advanced detection, filtering, and catalog settings.",
    )
    g_manifest = parser.add_argument_group("Manifest & index")
    g_tag = parser.add_argument_group("Tag")
    g_pregate = parser.add_argument_group("Pre-periodicity gate")
    g_events = parser.add_argument_group("Event detection")
    g_output = parser.add_argument_group("Output & bundle")
    g_filter = parser.add_argument_group("Filter")
    g_postprocess = parser.add_argument_group("Postprocess")
    g_characterize = parser.add_argument_group("Characterize")
    g_sed = parser.add_argument_group("SED photometry")
    g_classify = parser.add_argument_group("Classify")
    g_enrich = parser.add_argument_group("Enrich")
    g_neighbor = parser.add_argument_group("Neighbor enrichment")
    g_spectra = parser.add_argument_group("Spectra enrichment")
    g_vetting = parser.add_argument_group("Vetting")
    g_external_lcs = parser.add_argument_group("External light-curve enrichment")
    g_general = parser.add_argument_group("General")

    g_manifest.add_argument("--mag-bin", nargs="+", help="Magnitude bin(s) to process. Use 'all' to process all bins automatically.")
    g_manifest.add_argument("--index-root", type=Path, default=LCV2_ROOT,
                        help="Index root directory (contains mag_bin/index*.csv)")
    g_manifest.add_argument("--lc-root", type=Path, default=LCV2_ROOT,
                        help="Light curve root directory (contains mag_bin/lc*_cal/)")
    g_manifest.add_argument("--flat-lc-dir", type=Path, default=None,
                        help="Flat directory of <source_id>.<extension> light curves, such as bundle_assets/lightcurves")
    g_manifest.add_argument("--index-file", type=Path, default=None,
                        help="Optional ASAS-SN index/metadata file")
    g_manifest.add_argument("--extension", "-e", type=str, default=None,
                        help="Light curve file extension to process, e.g. dat2 or dat3. Default: dat3 from config.")
    g_events.add_argument(
        "--trigger-mode",
        choices=["posterior_prob", "logbf"],
        help="Event trigger mode. Default: posterior_prob from config.",
    )
    g_events.add_argument(
        "--baseline-func",
        choices=["gp", "gp_masked", "global_median", "per_camera_median", "phase_template"],
        help="Baseline model for event detection. Default: gp_masked from config.",
    )
    g_events.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Event result rows per output chunk. Default: {EVENTS_OUTPUT_CHUNK_SIZE} from config.",
    )

    g_output.add_argument("--output-dir", dest="out_dir", type=str, default=None,
                        help=f"Directory for all outputs (default: {DEFAULT_OUTPUT_DIR / 'runs' / 'stv'}/<timestamp>)")
    g_output.add_argument(
        "--import-bundle",
        type=Path,
        default=None,
        help="Import a transfer bundle ZIP before running the selected stage.",
    )
    bundle_group = g_output.add_mutually_exclusive_group()
    bundle_group.add_argument(
        "--export-bundle",
        type=Path,
        default=None,
        help="Write a transfer bundle ZIP to this path and enable bundle export.",
    )
    bundle_group.add_argument(
        "--no-export-bundle",
        dest="export_bundle_enabled",
        action="store_false",
        help="Disable transfer bundle export.",
    )
    g_output.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=PIPELINE_STAGE_CHOICES,
        help=(
            "Pipeline stage: full=standard discovery pipeline, "
            "full-extended=full plus external LCs and multi-survey features, "
            "cluster=raw-dependent upstream, home=downstream only"
        ),
    )
    g_filter.add_argument(
        "--gaia-cache",
        type=Path,
        default=None,
        help=f"Shared Gaia DR3 cache for RUWE/PM validation and characterization (default: {GAIA_LOCAL_CATALOG})",
    )
    g_filter.add_argument(
        "--no-auto-fetch-gaia-cache",
        dest="auto_fetch_gaia_cache",
        action="store_false",
        help="Disable automatic Gaia cache fetch before RUWE/PM filtering.",
    )
    g_filter.add_argument(
        "--gaia-fetch-all-candidates",
        dest="gaia_fetch_passers_only",
        action="store_false",
        help="Fetch Gaia rows for all event rows before RUWE/PM instead of only rows still passing prior filters.",
    )
    g_filter.add_argument(
        "--external-validations-passers-only",
        dest="external_validations_passers_only",
        action="store_true",
        help="Run filter-stage external validations only on rows still passing prior filters (default).",
    )
    g_filter.add_argument(
        "--external-validations-all-candidates",
        dest="external_validations_passers_only",
        action="store_false",
        help="Run filter-stage external validations on all event rows.",
    )
    g_filter.add_argument(
        "--gaia-fetch-chunk-size",
        type=int,
        default=None,
        help=f"Number of Gaia source IDs per TAP chunk when auto-fetching (default: {GAIA_CHUNK_SIZE})",
    )
    g_sed.add_argument(
        "--run-sed-photometry",
        dest="run_sed_photometry",
        action="store_true",
        help="Enable the SED photometry enrichment stage after characterization.",
    )
    g_sed.add_argument(
        "--no-sed-photometry",
        dest="run_sed_photometry",
        action="store_false",
        help="Disable the SED photometry enrichment stage after characterization.",
    )
    g_sed.add_argument(
        "--sed-sources",
        type=str,
        default="default",
        help=(
            "SED source keys: 'default' for payload/PS1/SkyMapper/SDSS broad-classification "
            "photometry, 'all' for every registered source, 'far-ir' for AKARI/IRAS/Herschel, "
            "or a comma-separated source list."
        ),
    )
    g_sed.add_argument(
        "--fit-atmosphere",
        dest="fit_atmosphere",
        action="store_true",
        help="Enable mandatory pystellibs Castelli/Kurucz atmosphere fitting after SED photometry.",
    )
    g_sed.add_argument(
        "--no-fit-atmosphere",
        dest="fit_atmosphere",
        action="store_false",
        help="Disable Castelli/Kurucz atmosphere fitting after SED photometry.",
    )
    g_enrich.add_argument(
        "--run-enrich",
        dest="run_enrich",
        action="store_true",
        help="Enable compute_stats enrichment after filtering.",
    )
    g_enrich.add_argument(
        "--no-enrich",
        dest="run_enrich",
        action="store_false",
        help="Disable compute_stats enrichment after filtering.",
    )
    g_enrich.add_argument(
        "--enrich-workers",
        type=int,
        default=None,
        help="Workers for compute_stats enrichment (default: min(--workers, 8)).",
    )
    g_enrich.add_argument(
        "--enrich-compute-ls",
        dest="enrich_compute_ls",
        action="store_true",
        help="Also compute Lomb-Scargle features during compute_stats enrichment.",
    )
    g_enrich.add_argument(
        "--no-enrich-compute-ls",
        dest="enrich_compute_ls",
        action="store_false",
        help="Skip Lomb-Scargle features during compute_stats enrichment.",
    )
    g_external_lcs.add_argument(
        "--run-external-lcs",
        dest="run_external_lcs",
        action="store_true",
        help="Run external light-curve enrichment after vetting.",
    )
    g_external_lcs.add_argument(
        "--no-external-lcs",
        dest="run_external_lcs",
        action="store_false",
        help="Disable external light-curve enrichment.",
    )
    g_external_lcs.add_argument(
        "--run-multi-survey-features",
        dest="run_multi_survey_features",
        action="store_true",
        help="Compute event-relative multi-survey features after external LC enrichment.",
    )
    g_external_lcs.add_argument(
        "--no-multi-survey-features",
        dest="run_multi_survey_features",
        action="store_false",
        help="Disable event-relative multi-survey feature extraction.",
    )
    g_external_lcs.add_argument(
        "--external-lc-workers",
        type=int,
        default=None,
        help="Workers for supported external light-curve fetchers (default: 4).",
    )
    g_external_lcs.add_argument(
        "--external-lc-refresh-cache",
        action="store_true",
        help="Ignore cached external light-curve products.",
    )
    g_external_lcs.add_argument(
        "--external-lc-atlas",
        action="store_true",
        help="Enable ATLAS forced photometry in external light-curve enrichment.",
    )
    g_external_lcs.add_argument(
        "--external-lc-atlas-token",
        type=str,
        default=None,
        help="ATLAS forced-photometry token, or set MALCA_ATLAS_TOKEN/ATLAS_API_TOKEN.",
    )

    add_config_args(g_general)
    g_general.add_argument("--workers", type=int, default=WORKERS, help="Workers for parallel processing")
    g_general.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    g_general.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.set_defaults(**PIPELINE_CONFIG_DEFAULTS)

    args = parser.parse_args()

    def cli_has_option(*option_names: str) -> bool:
        argv = sys.argv[1:]
        return any(
            token == name or token.startswith(f"{name}=")
            for token in argv
            for name in option_names
        )

    cli_overrides = {}
    if cli_has_option("--extension", "-e"):
        cli_overrides["extension"] = args.extension
    if cli_has_option("--trigger-mode"):
        cli_overrides["trigger_mode"] = args.trigger_mode
    if cli_has_option("--baseline-func"):
        cli_overrides["baseline_func"] = args.baseline_func
    if cli_has_option("--chunk-size"):
        cli_overrides["chunk_size"] = args.chunk_size
    if cli_has_option("--export-bundle"):
        cli_overrides["export_bundle"] = args.export_bundle
        cli_overrides["export_bundle_enabled"] = True
    if cli_has_option("--no-export-bundle"):
        cli_overrides["export_bundle_enabled"] = False
    if cli_has_option("--gaia-cache"):
        cli_overrides["gaia_cache"] = args.gaia_cache
    if cli_has_option("--no-auto-fetch-gaia-cache"):
        cli_overrides["auto_fetch_gaia_cache"] = args.auto_fetch_gaia_cache
    if cli_has_option("--gaia-fetch-all-candidates"):
        cli_overrides["gaia_fetch_passers_only"] = args.gaia_fetch_passers_only
    if cli_has_option("--external-validations-passers-only", "--external-validations-all-candidates"):
        cli_overrides["external_validations_passers_only"] = args.external_validations_passers_only
    if cli_has_option("--gaia-fetch-chunk-size"):
        cli_overrides["gaia_fetch_chunk_size"] = args.gaia_fetch_chunk_size
    if cli_has_option("--run-sed-photometry", "--no-sed-photometry"):
        cli_overrides["run_sed_photometry"] = args.run_sed_photometry
    if cli_has_option("--sed-sources"):
        cli_overrides["sed_sources"] = args.sed_sources
    if cli_has_option("--fit-atmosphere", "--no-fit-atmosphere"):
        cli_overrides["fit_atmosphere"] = args.fit_atmosphere
    if cli_has_option("--run-enrich", "--no-enrich"):
        cli_overrides["run_enrich"] = args.run_enrich
    if cli_has_option("--enrich-workers"):
        cli_overrides["enrich_workers"] = args.enrich_workers
    if cli_has_option("--enrich-compute-ls", "--no-enrich-compute-ls"):
        cli_overrides["enrich_compute_ls"] = args.enrich_compute_ls
    if cli_has_option("--run-external-lcs", "--no-external-lcs"):
        cli_overrides["run_external_lcs"] = args.run_external_lcs
    if cli_has_option("--run-multi-survey-features", "--no-multi-survey-features"):
        cli_overrides["run_multi_survey_features"] = args.run_multi_survey_features
    if cli_has_option("--external-lc-workers"):
        cli_overrides["external_lc_workers"] = args.external_lc_workers
    if cli_has_option("--external-lc-refresh-cache"):
        cli_overrides["external_lc_refresh_cache"] = args.external_lc_refresh_cache
    if cli_has_option("--external-lc-atlas"):
        cli_overrides["external_lc_atlas"] = args.external_lc_atlas
    if cli_has_option("--external-lc-atlas-token"):
        cli_overrides["external_lc_atlas_token"] = args.external_lc_atlas_token

    apply_config(
        args,
        command="pipeline",
        valid_keys=namespace_keys(parser, PIPELINE_CONFIG_DEFAULTS),
        path_keys=PIPELINE_CONFIG_PATH_KEYS,
    )
    for key, value in cli_overrides.items():
        setattr(args, key, value)

    # Handle --mag-bin all: expand to all bins in reverse order
    is_auto_all_mode = False
    if args.mag_bin and "all" in args.mag_bin:
        if len(args.mag_bin) > 1:
            parser.error("Cannot mix 'all' with specific magnitude bins. Use '--mag-bin all' alone or specify individual bins.")
        
        # Expand "all" to full list of magnitude bins in reverse order
        is_auto_all_mode = True
        args.mag_bin = list(reversed(MAG_BINS))

    stage = str(args.stage)
    if _stage_defaults_to_extended_enrichment(stage):
        if not cli_has_option("--no-external-lcs"):
            args.run_external_lcs = True
        if not cli_has_option("--no-multi-survey-features"):
            args.run_multi_survey_features = True
    run_upstream = _stage_runs_upstream(stage)
    run_downstream = _stage_runs_downstream(stage)

    if stage != "home" and not args.mag_bin:
        parser.error("--mag-bin is required unless --stage home is used.")

    if stage == "home":
        if args.mag_bin:
            if args.import_bundle is not None:
                bundle_mag_bins = _read_mag_bins_from_bundle(args.import_bundle)
                if bundle_mag_bins is not None:
                    _assert_mag_bin_match(bundle_mag_bins, args.mag_bin, f"{Path(args.import_bundle).expanduser()}/run_params.json")
            if args.out_dir is not None:
                out_dir_params = Path(args.out_dir).expanduser() / "run_params.json"
                out_dir_mag_bins = _read_mag_bins_from_params_file(out_dir_params)
                if out_dir_mag_bins is not None:
                    _assert_mag_bin_match(out_dir_mag_bins, args.mag_bin, str(out_dir_params))
        else:
            if args.import_bundle is None and args.out_dir is None:
                parser.error("--stage home without --mag-bin requires import_bundle in config or --output-dir.")

            detected_mag_bins = None
            detected_source = None

            if args.import_bundle is not None:
                detected_mag_bins = _read_mag_bins_from_bundle(args.import_bundle)
                detected_source = f"{Path(args.import_bundle).expanduser()}/run_params.json"

            if detected_mag_bins is None and args.out_dir is not None:
                out_dir_params = Path(args.out_dir).expanduser() / "run_params.json"
                detected_mag_bins = _read_mag_bins_from_params_file(out_dir_params)
                detected_source = str(out_dir_params)

            if not detected_mag_bins:
                parser.error(
                    "Could not auto-detect --mag-bin for --stage home. "
                    "Expected mag_bin in run_params.json from import_bundle config or --output-dir."
                )

            args.mag_bin = detected_mag_bins
            print(f"Info: auto-detected --mag-bin={args.mag_bin} from {detected_source}")

    if stage == "cluster" and (args.run_characterize or args.run_dust or args.run_sed_photometry or args.run_classify or args.run_neighbor_enrich or args.run_spectra_enrich):
        print("Info: --stage cluster runs upstream only (steps 1-6 plus enrich). Downstream steps are skipped.")
    if stage == "home" and (args.force_manifest or args.force_tag):
        print("Info: --stage home skips manifest/tag/events regardless of force flags.")

    run_reuse_fingerprint = build_run_reuse_fingerprint(
        args,
        stage=stage,
        is_auto_all_mode=is_auto_all_mode,
    )
    run_reuse_fingerprint_hash = run_reuse_fingerprint_digest(run_reuse_fingerprint)

    # Build events.py args from parsed arguments
    events_args = []
    if args.verbose:
        events_args.append("--verbose")
    events_args.extend(["--workers", str(args.workers)])
    events_args.extend(["--trigger-mode", args.trigger_mode])
    events_args.extend(["--logbf-threshold-dip", str(args.logbf_threshold_dip)])
    events_args.extend(["--logbf-threshold-jump", str(args.logbf_threshold_jump)])
    events_args.extend(["--significance-threshold", str(args.significance_threshold)])
    events_args.extend(["--p-points", str(args.p_points)])
    events_args.extend(["--mag-points", str(args.mag_points)])
    events_args.extend(["--run-min-points", str(args.run_min_points)])
    events_args.extend(["--run-max-gap-points", str(args.run_max_gap_points)])
    if args.run_max_gap_days is not None:
        events_args.extend(["--run-max-gap-days", str(args.run_max_gap_days)])
    if args.run_min_duration_days is not None:
        events_args.extend(["--run-min-duration-days", str(args.run_min_duration_days)])
    if args.no_event_prob:
        events_args.append("--no-event-prob")
    if args.p_min_dip is not None:
        events_args.extend(["--p-min-dip", str(args.p_min_dip)])
    if args.p_max_dip is not None:
        events_args.extend(["--p-max-dip", str(args.p_max_dip)])
    if args.p_min_jump is not None:
        events_args.extend(["--p-min-jump", str(args.p_min_jump)])
    if args.p_max_jump is not None:
        events_args.extend(["--p-max-jump", str(args.p_max_jump)])
    events_args.extend(["--baseline-func", args.baseline_func])
    # Baseline kwargs
    events_args.extend(["--baseline-s0", str(args.baseline_s0)])
    events_args.extend(["--baseline-w0", str(args.baseline_w0)])
    events_args.extend(["--baseline-q", str(args.baseline_q)])
    events_args.extend(["--baseline-jitter", str(args.baseline_jitter)])
    if args.baseline_sigma_floor is not None:
        events_args.extend(["--baseline-sigma-floor", str(args.baseline_sigma_floor)])
    # Magnitude grid bounds
    if args.mag_min_dip is not None:
        events_args.extend(["--mag-min-dip", str(args.mag_min_dip)])
    if args.mag_max_dip is not None:
        events_args.extend(["--mag-max-dip", str(args.mag_max_dip)])
    if args.mag_min_jump is not None:
        events_args.extend(["--mag-min-jump", str(args.mag_min_jump)])
    if args.mag_max_jump is not None:
        events_args.extend(["--mag-max-jump", str(args.mag_max_jump)])
    events_format = "parquet_chunk"

    events_args.extend(["--min-mag-offset", str(args.min_mag_offset)])
    events_args.extend(["--output-format", events_format])
    events_args.extend(["--chunk-size", str(args.chunk_size)])

    quiet = not bool(args.verbose)

    def log(message: str) -> None:
        _log(message, quiet=quiet)

    # Determine file names
    mag_bin_tag = "all" if is_auto_all_mode else (args.mag_bin[0] if len(args.mag_bin) == 1 else "multi")

    # IMPORTANT: never write to filesystem root (/output). Default to a writable directory.
    base_output_root = DEFAULT_OUTPUT_DIR.resolve()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
    elif args.import_bundle is not None:
        # Auto-derive out_dir from bundle name
        bundle_path = Path(args.import_bundle).expanduser()
        out_dir = get_out_dir_from_bundle(bundle_path, base_output_root, overwrite=bool(args.overwrite))
        log(f"Using output directory from bundle name: {out_dir}")
    elif args.filtered_file is not None:
        out_dir = Path(args.filtered_file).expanduser().parent
    elif args.manifest_file is not None:
        out_dir = Path(args.manifest_file).expanduser().parent
    elif args.output is not None:
        out_dir = Path(args.output).expanduser().parent
    else:
        out_dir = find_latest_run_dir(base_output_root, args.mag_bin, run_reuse_fingerprint)
        if out_dir is not None:
            log(f"Reusing existing run directory with matching run parameters: {out_dir}")
        else:
            out_dir = default_run_dir(base_output_root)
            log(
                "No existing run directory has matching run parameters; "
                f"starting a fresh run directory: {out_dir}"
            )
    if args.import_bundle is not None and args.overwrite and out_dir.exists():
        log(f"Overwriting existing imported run directory: {out_dir}")
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.import_bundle is not None:
        import_started = time.perf_counter()
        log(f"Importing bundle from {Path(args.import_bundle).expanduser()} to {out_dir}...")
        import_bundle_zip(args.import_bundle, out_dir, show_progress=args.verbose)
        log(f"Bundle import completed in {time.perf_counter() - import_started:.1f}s")
        imported_flat_lc_dir = out_dir / "bundle_assets" / "lightcurves"
        if args.flat_lc_dir is None and imported_flat_lc_dir.is_dir():
            args.flat_lc_dir = imported_flat_lc_dir
            log(f"Using imported flat light-curve directory: {args.flat_lc_dir}")

    if args.gaia_cache is None:
        args.gaia_cache = GAIA_LOCAL_CATALOG

    ctx = init_pipeline_run_context("stv", out_dir)
    manifests_dir = out_dir / "manifests"
    tags_dir = out_dir / "tags"
    paths_dir = out_dir / "paths"
    results_dir = ctx.results_dir
    for d in (manifests_dir, tags_dir, paths_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    if stage == "home":
        required_filtered = results_dir / "lc_events_filtered.parquet"
        if not required_filtered.exists():
            source_hint = f" from bundle {args.import_bundle}" if args.import_bundle else ""
            raise FileNotFoundError(
                f"Home stage requires {required_filtered}{source_hint}. "
                "Run cluster/full stage first and transfer results/lc_events_filtered.parquet."
            )

    if args.output is None:
        events_output = results_dir / f"lc_events_results_{mag_bin_tag}"
    else:
        events_output = Path(args.output).expanduser()
        if args.out_dir is not None and not events_output.is_absolute():
            events_output = out_dir / events_output
        elif args.out_dir is None and not events_output.is_absolute():
            events_output = out_dir / events_output
        if events_output.suffix.lower() == ".parquet":
            events_output = events_output.with_suffix("")

    manifest_file = Path(args.manifest_file).expanduser() if args.manifest_file else (manifests_dir / f"lc_manifest_{mag_bin_tag}.parquet")
    filtered_file = Path(args.filtered_file).expanduser() if args.filtered_file else (tags_dir / f"lc_filtered_{mag_bin_tag}.parquet")
    stats_checkpoint_file = tags_dir / f"lc_stats_checkpoint_{mag_bin_tag}.parquet"
    pre_periodicity_file = tags_dir / f"pre_periodicity_{mag_bin_tag}.parquet"
    periodic_branch_file = manifests_dir / f"lc_periodic_branch_{mag_bin_tag}.parquet"
    stochastic_branch_file = manifests_dir / f"lc_stochastic_branch_{mag_bin_tag}.parquet"
    periodic_paths_file = paths_dir / f"periodic_paths_{mag_bin_tag}.txt"
    branch_cache_dir = results_dir / "_branch_events"
    branch_cache_dir.mkdir(parents=True, exist_ok=True)
    periodic_branch_events_output = branch_cache_dir / f"lc_events_periodic_branch_{mag_bin_tag}"
    stochastic_branch_events_output = branch_cache_dir / f"lc_events_stochastic_branch_{mag_bin_tag}"
    if args.pre_periodicity_checkpoint is None:
        pre_periodicity_checkpoint = tags_dir / f"pre_periodicity_checkpoint_{mag_bin_tag}.parquet"
    else:
        pre_periodicity_checkpoint = Path(args.pre_periodicity_checkpoint).expanduser()

    # Save run parameters to JSON for full reproducibility
    run_start_time = ctx.started_at

    # Build a compact fingerprint of filtering/characterization behavior.
    if args.pass_all_tags:
        enforced_tags = []
    elif args.enforce_tags:
        enforced_tags = [f.strip() for f in args.enforce_tags.split(",") if f.strip()]
    else:
        enforced_tags = []
        if not args.skip_sparse:
            enforced_tags.append("sparse")
        if not args.skip_multi_camera:
            enforced_tags.append("multi_camera")
        if not args.skip_mag_range:
            enforced_tags.append("mag_range")

    config_fingerprint = {
        "skip_vsx": args.skip_vsx,
        "pass_all_tags": args.pass_all_tags,
        "enforced_tags": enforced_tags,
        "pre_periodicity_gate": {
            "enabled": args.apply_pre_periodicity_gate,
            "router_mode": PREGATE_ROUTER_MODE,
            "min_period": args.pre_periodicity_min_period,
            "max_period": args.pre_periodicity_max_period,
            "n_periods": args.pre_periodicity_n_periods,
            "ce_snr_threshold": args.pre_periodicity_ce_snr_threshold,
            "min_points": args.pre_periodicity_min_points,
            "scatter_ratio_max": args.pre_periodicity_scatter_ratio_max,
            "workers": args.pre_periodicity_workers,
            "checkpoint": str(pre_periodicity_checkpoint),
        },
        "periodic_branch": {
            "enabled": args.apply_pre_periodicity_gate,
            "mode": "events_phase_template_residual",
            "baseline_func": "phase_template",
            "workers": args.workers,
            "cache_dir": str(branch_cache_dir),
        },
        "filter": {
            "apply_evidence_strength": not args.skip_evidence_strength,
            "min_bayes_factor": args.min_bayes_factor,
            "require_finite_local_bf": not args.allow_infinite_local_bf,
            "apply_significant_detection": not args.skip_significant_detection,
            "significant_require_flag": not args.significant_no_require_flag,
            "significant_min_peak_count": args.significant_min_peak_count,
            "significant_min_run_count": args.significant_min_run_count,
            "apply_run_robustness": not args.skip_run_robustness,
            "min_run_count": args.min_run_count,
            "max_run_count": args.max_run_count,
            "min_run_cameras": args.filter_min_run_cameras,
            "min_run_points": args.filter_min_run_points,
            "apply_morphology": args.apply_morphology,
            "dip_morphology": args.dip_morphology,
            "jump_morphology": args.jump_morphology,
            "min_delta_bic": args.min_delta_bic,
            "apply_score": _score_filter_enabled(args),
            "min_dip_score": args.min_dip_score,
            "min_jump_score": args.min_jump_score,
            "min_score": args.min_score,
            "apply_periodicity_validation": args.apply_periodicity_validation,
            "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
            "periodicity_significance": args.periodicity_significance,
            "periodicity_pdm_method": args.periodicity_pdm_method,
            "periodicity_exclude_aliases": not args.periodicity_no_exclude_aliases,
            "periodicity_flag_only": not args.periodicity_reject,
            "periodicity_workers": args.periodicity_workers,
            "periodicity_checkpoint_dir": str(args.periodicity_checkpoint_dir) if args.periodicity_checkpoint_dir else None,
            "periodicity_all_candidates": args.periodicity_all_candidates,
            "phase_plot_max_sig": args.phase_plot_max_sig,
            "phase_plot_min_power": args.phase_plot_min_power,
            "phase_plot_allow_alias": args.phase_plot_allow_alias,
            "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
            "gaia_max_ruwe": args.gaia_max_ruwe,
            "gaia_flag_only": not args.gaia_reject,
            "apply_gaia_pm_validation": not args.skip_gaia_pm_validation,
            "gaia_max_pm": args.gaia_max_pm,
            "gaia_pm_flag_only": not args.gaia_pm_reject,
            "external_validations_passers_only": args.external_validations_passers_only,
            "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
            "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
            "periodic_catalog_flag_only": not args.periodic_catalog_reject,
        },
        "characterize": {
            "run_characterize": args.run_characterize,
            "run_dust": args.run_dust,
            "starhorse": args.characterize_starhorse,
            "starhorse_cache": str(args.characterize_starhorse_cache) if args.characterize_starhorse_cache else None,
            "unwise_checkpoint_every": args.characterize_unwise_checkpoint_every,
            "banyan": args.characterize_banyan,
            "iphas": args.characterize_iphas,
            "sfr": args.characterize_sfr,
            "clusters": args.characterize_clusters,
            "unwise": args.characterize_unwise,
        },
        "sed_photometry": {
            "run_sed_photometry": args.run_sed_photometry,
            "sources": args.sed_sources,
            "fit_atmosphere": args.fit_atmosphere,
        },
        "enrich": {
            "run_enrich": args.run_enrich,
            "enrich_workers": args.enrich_workers,
            "enrich_compute_ls": args.enrich_compute_ls,
        },
        "extended_enrichment": {
            "run_external_lcs": args.run_external_lcs,
            "run_multi_survey_features": args.run_multi_survey_features,
            "external_lc_workers": args.external_lc_workers,
            "external_lc_refresh_cache": args.external_lc_refresh_cache,
            "external_lc_atlas": args.external_lc_atlas,
        },
        "downstream_pass_logic": "external validations and downstream products run on filter passers (failed_any == False) by default",
    }

    run_params_file = ctx.run_params_file
    run_params_tagged_file = out_dir / f"run_params_{mag_bin_tag}.json"
    run_summary_file = ctx.run_summary_file
    imported_run_params_snapshot, imported_run_summary_snapshot = preserve_imported_run_snapshots(
        stage=stage,
        import_bundle=args.import_bundle,
        out_dir=out_dir,
        run_params_file=run_params_file,
        run_summary_file=run_summary_file,
    )

    bundle_lightcurve_dir = out_dir / "bundle_assets" / "lightcurves"
    bundle_lightcurve_count = (
        sum(1 for p in bundle_lightcurve_dir.iterdir() if p.is_file())
        if bundle_lightcurve_dir.is_dir()
        else 0
    )
    manifests_file_count = sum(1 for p in manifests_dir.rglob("*") if p.is_file()) if manifests_dir.exists() else 0
    tags_file_count = sum(1 for p in tags_dir.rglob("*") if p.is_file()) if tags_dir.exists() else 0
    paths_file_count = sum(1 for p in paths_dir.rglob("*") if p.is_file()) if paths_dir.exists() else 0

    summary_state = load_summary_state(
        run_summary_file=run_summary_file,
        run_start_time=run_start_time,
        stage=stage,
    )

    results_files: list[Path] = []
    cmd = ctx.command
    try:
        run_params = {
            "timestamp": run_start_time.isoformat(),
            "command": cmd,
            "stage": stage,
            "run_reuse_fingerprint": run_reuse_fingerprint,
            "run_reuse_fingerprint_hash": run_reuse_fingerprint_hash,
            "import_bundle": str(args.import_bundle) if args.import_bundle else None,
            "export_bundle": str(args.export_bundle) if args.export_bundle else None,
            "export_bundle_enabled": args.export_bundle_enabled,
            "review_sync_enabled": args.review_sync_enabled,
            "review_sync_dir": str(args.review_sync_dir),
            "review_sync_hash_assets": bool(args.review_sync_hash_assets),
            "imported_run_params_snapshot": str(imported_run_params_snapshot) if imported_run_params_snapshot else None,
            "imported_run_summary_snapshot": str(imported_run_summary_snapshot) if imported_run_summary_snapshot else None,
            "mag_bin": args.mag_bin,
            "extension": args.extension,
            "test_run": args.test_run,
            "test_run_n": args.test_run_n,
            # Tag parameters
            "min_time_span": args.min_time_span,
            "min_points_per_day": args.min_points_per_day,
            "min_cameras": args.min_cameras,
            "mag_lo": args.mag_lo,
            "mag_hi": args.mag_hi,
            "skip_sparse": args.skip_sparse,
            "skip_multi_camera": args.skip_multi_camera,
            "skip_mag_range": args.skip_mag_range,
            "skip_vsx": args.skip_vsx,
            "vsx_max_sep": args.vsx_max_sep,
            "vsx_crossmatch": str(args.vsx_crossmatch),
            # Detection parameters
            "trigger_mode": args.trigger_mode,
            "logbf_threshold_dip": args.logbf_threshold_dip,
            "logbf_threshold_jump": args.logbf_threshold_jump,
            "significance_threshold": args.significance_threshold,
            "p_points": args.p_points,
            "p_min_dip": args.p_min_dip,
            "p_max_dip": args.p_max_dip,
            "p_min_jump": args.p_min_jump,
            "p_max_jump": args.p_max_jump,
            "mag_points": args.mag_points,
            "mag_min_dip": args.mag_min_dip,
            "mag_max_dip": args.mag_max_dip,
            "mag_min_jump": args.mag_min_jump,
            "mag_max_jump": args.mag_max_jump,
            # Baseline parameters
            "baseline_func": args.baseline_func,
            "baseline_s0": args.baseline_s0,
            "baseline_w0": args.baseline_w0,
            "baseline_q": args.baseline_q,
            "baseline_jitter": args.baseline_jitter,
            "baseline_sigma_floor": args.baseline_sigma_floor,
            # Run parameters
            "run_min_points": args.run_min_points,
            "run_max_gap_points": args.run_max_gap_points,
            "run_max_gap_days": args.run_max_gap_days,
            "run_min_duration_days": args.run_min_duration_days,
            "no_event_prob": args.no_event_prob,
            "min_mag_offset": args.min_mag_offset,
            # System parameters
            "workers": args.workers,
            "batch_size": args.batch_size,
            # Cleaning thresholds (hardcoded, not CLI args)
            "clean_max_error_absolute": CLEAN_LC_MAX_ERROR_ABSOLUTE,
            "clean_max_error_sigma": CLEAN_LC_MAX_ERROR_SIGMA,
            "bad_camera_scatter_ratio": BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
            # Tag stage (camera median)
            "skip_camera_median": args.skip_camera_median,
            "camera_median_tolerance": args.camera_median_tolerance,
            # Pre-events periodicity gate
            "apply_pre_periodicity_gate": args.apply_pre_periodicity_gate,
            "pre_periodicity_min_period": args.pre_periodicity_min_period,
            "pre_periodicity_max_period": args.pre_periodicity_max_period,
            "pre_periodicity_n_periods": args.pre_periodicity_n_periods,
            "pre_periodicity_router_mode": PREGATE_ROUTER_MODE,
            "pre_periodicity_ce_snr_threshold": args.pre_periodicity_ce_snr_threshold,
            "pre_periodicity_min_points": args.pre_periodicity_min_points,
            "pre_periodicity_scatter_ratio_max": args.pre_periodicity_scatter_ratio_max,
            "pre_periodicity_workers": args.pre_periodicity_workers,
            "pre_periodicity_checkpoint": str(pre_periodicity_checkpoint),
            "periodic_branch_events_output": str(periodic_branch_events_output),
            "stochastic_branch_events_output": str(stochastic_branch_events_output),
            # Step 5: Filter
            "run_filter": args.run_filter,
            "skip_evidence_strength": args.skip_evidence_strength,
            "min_bayes_factor": args.min_bayes_factor,
            "allow_infinite_local_bf": args.allow_infinite_local_bf,
            "skip_significant_detection": args.skip_significant_detection,
            "significant_no_require_flag": args.significant_no_require_flag,
            "significant_min_peak_count": args.significant_min_peak_count,
            "significant_min_run_count": args.significant_min_run_count,
            "skip_run_robustness": args.skip_run_robustness,
            "min_run_count": args.min_run_count,
            "max_run_count": args.max_run_count,
            "filter_min_run_cameras": args.filter_min_run_cameras,
            "filter_min_run_points": args.filter_min_run_points,
            "apply_morphology": args.apply_morphology,
            "dip_morphology": args.dip_morphology,
            "jump_morphology": args.jump_morphology,
            "min_delta_bic": args.min_delta_bic,
            "apply_score_filter": _score_filter_enabled(args),
            "min_dip_score": args.min_dip_score,
            "min_jump_score": args.min_jump_score,
            "min_score": args.min_score,
            "apply_periodicity_validation": args.apply_periodicity_validation,
            "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
            "periodicity_significance": args.periodicity_significance,
            "periodicity_no_exclude_aliases": args.periodicity_no_exclude_aliases,
            "periodicity_reject": args.periodicity_reject,
            "periodicity_workers": args.periodicity_workers,
            "periodicity_checkpoint_dir": str(args.periodicity_checkpoint_dir) if args.periodicity_checkpoint_dir else None,
            "phase_plot_max_sig": args.phase_plot_max_sig,
            "phase_plot_min_power": args.phase_plot_min_power,
            "phase_plot_allow_alias": args.phase_plot_allow_alias,
            "skip_gaia_ruwe_validation": args.skip_gaia_ruwe_validation,
            "gaia_max_ruwe": args.gaia_max_ruwe,
            "gaia_reject": args.gaia_reject,
            "skip_gaia_pm_validation": args.skip_gaia_pm_validation,
            "gaia_max_pm": args.gaia_max_pm,
            "gaia_pm_reject": args.gaia_pm_reject,
            "auto_fetch_gaia_cache": args.auto_fetch_gaia_cache,
            "gaia_fetch_passers_only": args.gaia_fetch_passers_only,
            "external_validations_passers_only": args.external_validations_passers_only,
            "skip_periodic_catalog_validation": args.skip_periodic_catalog_validation,
            "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
            "periodic_catalog_reject": args.periodic_catalog_reject,
            # Step 7: Postprocess
            "run_postprocess": args.run_postprocess,
            "max_plots": args.max_plots,
            "plot_format": args.plot_format,
            # Step 8: Characterization
            "run_characterize": args.run_characterize,
            "run_dust": args.run_dust,
            "gaia_cache": str(args.gaia_cache),
            "gaia_fetch_chunk_size": args.gaia_fetch_chunk_size,
            "characterize_crossmatch": str(args.characterize_crossmatch),
            "characterize_chunk_size": args.characterize_chunk_size,
            "characterize_starhorse": args.characterize_starhorse,
            "characterize_starhorse_cache": str(args.characterize_starhorse_cache) if args.characterize_starhorse_cache else None,
            "characterize_unwise_checkpoint_every": args.characterize_unwise_checkpoint_every,
            "characterize_banyan": args.characterize_banyan,
            "characterize_iphas": args.characterize_iphas,
            "characterize_sfr": args.characterize_sfr,
            "characterize_clusters": args.characterize_clusters,
            "characterize_unwise": args.characterize_unwise,
            # Step 9: SED photometry
            "run_sed_photometry": args.run_sed_photometry,
            "sed_sources": args.sed_sources,
            "fit_atmosphere": args.fit_atmosphere,
            # Step 9: Classify
            "run_classify": args.run_classify,
            # Step 6: Enrich
            "run_enrich": args.run_enrich,
            "enrich_workers": args.enrich_workers,
            "enrich_compute_ls": args.enrich_compute_ls,
            # Step 10: Neighbor enrichment
            "run_neighbor_enrich": args.run_neighbor_enrich,
            "neighbor_radius_arcsec": args.neighbor_radius_arcsec,
            "neighbor_chunk_size": args.neighbor_chunk_size,
            "neighbor_cache": str(args.neighbor_cache) if args.neighbor_cache else None,
            # Step 11: Spectra enrichment
            "run_spectra_enrich": args.run_spectra_enrich,
            "spectra_radius_arcsec": args.spectra_radius_arcsec,
            "spectra_chunk_size": args.spectra_chunk_size,
            "spectra_cache": str(args.spectra_cache) if args.spectra_cache else None,
            # Step 12: Vetting
            "run_vetting": args.run_vetting,
            "vetting_min_score": args.vetting_min_score,
            "vetting_simbad_radius": args.vetting_simbad_radius,
            "vetting_asassn_radius": args.vetting_asassn_radius,
            # Step 13b/13c: Extended external enrichment
            "run_external_lcs": args.run_external_lcs,
            "run_multi_survey_features": args.run_multi_survey_features,
            "external_lc_workers": args.external_lc_workers,
            "external_lc_refresh_cache": args.external_lc_refresh_cache,
            "external_lc_atlas": args.external_lc_atlas,
            # File paths
            "index_root": str(args.index_root),
            "lc_root": str(args.lc_root),
            "flat_lc_dir": str(args.flat_lc_dir.expanduser()) if args.flat_lc_dir else None,
            "index_file": str(args.index_file.expanduser()) if args.index_file else None,
            "out_dir": str(out_dir),
            "manifest_file": str(manifest_file),
            "filtered_file": str(filtered_file),
            "pre_periodicity_file": str(pre_periodicity_file),
            "periodic_branch_file": str(periodic_branch_file),
            "stochastic_branch_file": str(stochastic_branch_file),
            "periodic_paths_file": str(periodic_paths_file),
            "branch_cache_dir": str(branch_cache_dir),
            "events_output": str(events_output),
            "bundle_lightcurve_count": bundle_lightcurve_count,
            "manifests_file_count": manifests_file_count,
            "tags_file_count": tags_file_count,
            "paths_file_count": paths_file_count,
        }

        write_run_params(ctx, run_params, extra_paths=[run_params_tagged_file])

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run_params.json: {e}")

    # Write a simple run log with the command and key paths.
    run_log_tagged = out_dir / f"run_{mag_bin_tag}.log"
    try:
        events_cmd_preview = shlex.join([sys.executable, "-m", "malca.stv.events", *events_args, "--", "<paths_file>"])
        run_log_lines = [
                f"timestamp: {run_start_time.isoformat()}",
                f"command: {cmd}",
                f"events_cmd: {events_cmd_preview}",
                f"out_dir: {out_dir}",
                f"run_params: {run_params_file}",
                f"manifests_dir: {manifests_dir}",
                f"tags_dir: {tags_dir}",
                f"paths_dir: {paths_dir}",
                f"results_dir: {results_dir}",
                f"results_output: {events_output}",
                f"branch_cache_dir: {branch_cache_dir}",
                f"manifest_file: {manifest_file}",
                f"filtered_file: {filtered_file}",
                f"stats_checkpoint: {stats_checkpoint_file}",
                f"rejected_tag: {tags_dir / f'rejected_tag_{mag_bin_tag}.parquet'}",
            ]
        write_run_log(ctx, run_log_lines)
        run_log_tagged.write_text("\n".join(run_log_lines) + "\n", encoding="ascii")
    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run log: {e}")

    df_manifest = pd.DataFrame()
    df_filtered = pd.DataFrame()
    df_periodic_candidates = pd.DataFrame()
    pre_periodicity_stats: dict[str, object] | None = None
    branch_detection_stats: dict[str, object] | None = None

    def _normalized_output_path(path: Path, fmt: str) -> Path:
        if fmt == "parquet_chunk":
            return path.with_suffix("") if path.suffix.lower() == ".parquet" else path
        return path if path.suffix.lower() == ".parquet" else path.with_suffix(".parquet")

    def _output_files_for_path(path: Path, fmt: str) -> list[Path]:
        path = _normalized_output_path(path, fmt)
        if fmt == "parquet_chunk":
            return sorted(path.glob("chunk_*.parquet")) if path.exists() and path.is_dir() else []
        return [path] if path.exists() and path.is_file() else []

    def _load_events_output(path: Path, fmt: str) -> pd.DataFrame:
        path = _normalized_output_path(path, fmt)
        files = _output_files_for_path(path, fmt)
        if not files:
            return pd.DataFrame()
        return pd.concat([read_feature_table(f) for f in files], ignore_index=True)

    def _write_events_output(df: pd.DataFrame, path: Path, fmt: str) -> list[Path]:
        path = _normalized_output_path(path, fmt)
        df = add_stv_identity(df)
        assert_stv_product_schema(df, stage="events")
        if fmt == "parquet_chunk":
            if path.exists():
                clear_existing_output(path, fmt)
            path.mkdir(parents=True, exist_ok=True)
            chunk_rows = max(1, int(args.chunk_size or len(df) or 1))
            files: list[Path] = []
            for idx, start in enumerate(range(0, len(df), chunk_rows)):
                chunk_path = path / f"chunk_{idx:06d}.parquet"
                write_feature_table(
                    df.iloc[start : start + chunk_rows],
                    chunk_path,
                    compression=PARQUET_OUTPUT_COMPRESSION,
                )
                files.append(chunk_path)
            return files
        write_feature_table(df, path, compression=PARQUET_OUTPUT_COMPRESSION)
        return [path]

    def _run_events_branch(
        file_paths: list[str],
        *,
        branch_name: str,
        branch_output: Path,
        baseline_func_override: str | None = None,
        metadata_df: pd.DataFrame | None = None,
        branch_paths_file: Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        branch_output = _normalized_output_path(Path(branch_output), events_format)
        checkpoint_log = branch_output.with_name(f"{branch_output.stem}_PROCESSED.txt")
        error_log = branch_output.with_name(f"{branch_output.stem}_ERRORS.parquet")
        metadata_path: Path | None = None

        if args.overwrite:
            clear_existing_output(branch_output, events_format)
            checkpoint_log.unlink(missing_ok=True)
            error_log.unlink(missing_ok=True)

        if metadata_df is not None and not metadata_df.empty:
            metadata_dir = tags_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"metadata_{branch_name}_{mag_bin_tag}.parquet"
            metadata_df.to_parquet(metadata_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
            log(
                f"Wrote {branch_name} metadata Parquet with columns: "
                f"{', '.join(col for col in metadata_df.columns if col != 'lc_path')}"
            )

        if branch_paths_file is not None:
            branch_paths_file.parent.mkdir(parents=True, exist_ok=True)
            with open(branch_paths_file, "w") as f:
                for path_value in file_paths:
                    f.write(f"{path_value}\n")

        processed_paths: set[str] = set()
        if checkpoint_log.exists() and not args.overwrite:
            try:
                with open(checkpoint_log, "r") as f:
                    processed_paths = {line.strip() for line in f if line.strip()}
                log(
                    f"{branch_name.title()} branch checkpoint detected, "
                    f"skipping {len(processed_paths)} already-processed paths"
                )
            except Exception as e:
                log(f"Warning: could not read checkpoint log {checkpoint_log}: {e}")

        if not args.overwrite:
            try:
                df_existing_branch = _load_events_output(branch_output, events_format)
                if "lc_path" in df_existing_branch.columns:
                    output_paths = set(df_existing_branch["lc_path"].dropna().astype(str))
                    new_output_paths = output_paths - processed_paths
                    if new_output_paths:
                        processed_paths |= new_output_paths
                        log(
                            f"{branch_name.title()} branch output contains "
                            f"{len(new_output_paths)} additional processed paths"
                        )
            except Exception as e:
                log(f"Warning: could not inspect existing {branch_name} branch output: {e}")

            try:
                if error_log.exists():
                    df_existing_errors = pd.read_parquet(error_log, columns=["lc_path"])
                    error_paths = set(df_existing_errors["lc_path"].dropna().astype(str))
                    new_error_paths = error_paths - processed_paths
                    if new_error_paths:
                        processed_paths |= new_error_paths
                        log(
                            f"{branch_name.title()} branch error log contains "
                            f"{len(new_error_paths)} additional failed paths; "
                            "skipping them on resume"
                        )
            except Exception as e:
                log(f"Warning: could not inspect existing {branch_name} branch error log: {e}")

        remaining = [path_value for path_value in file_paths if str(path_value) not in processed_paths]
        if file_paths and (not remaining):
            log(f"All {branch_name} branch paths already processed according to checkpoint/output.")

        batch_size = max(1, args.batch_size)
        total_batches = (len(remaining) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(len(remaining), start + batch_size)
            batch_paths = remaining[start:end]
            log(
                f"\nRunning {branch_name} branch batch {batch_idx + 1}/{total_batches} "
                f"({len(batch_paths)} LCs)..."
            )

            batch_paths_file = paths_dir / f"batch_paths_{branch_name}_{mag_bin_tag}_{batch_idx}.txt"
            with open(batch_paths_file, "w") as f:
                for path_value in batch_paths:
                    f.write(f"{path_value}\n")

            branch_events_args = list(events_args)
            if baseline_func_override is not None:
                branch_events_args.extend(["--baseline-func", baseline_func_override])
            branch_events_args.extend([
                "--output",
                str(branch_output),
                "--error-output",
                str(error_log),
            ])
            if metadata_path is not None:
                branch_events_args.extend(["--metadata", str(metadata_path)])

            events_cmd = [
                sys.executable,
                "-m",
                "malca.stv.events",
                *branch_events_args,
                "--input-file",
                str(batch_paths_file),
            ]

            try:
                result = subprocess.run(events_cmd, check=False)
                if result.returncode != 0:
                    print(f"events.py returned non-zero exit ({result.returncode}); stopping.")
                    sys.exit(result.returncode)
            except Exception as e:
                print(f"\nError running events.py for {branch_name} branch: {e}")
                if branch_paths_file is not None:
                    print(f"\nBranch paths saved to: {branch_paths_file}")
                sys.exit(1)

        df_branch = _load_events_output(branch_output, events_format)
        stats = {
            "branch": branch_name,
            "baseline_func": baseline_func_override or args.baseline_func,
            "total_input": int(len(file_paths)),
            "attempted_this_run": int(len(remaining)),
            "skipped_by_checkpoint": int(len(file_paths) - len(remaining)),
            "total_results": int(len(df_branch)),
            "dip_significant": int(df_branch["dip_significant"].fillna(False).sum()) if "dip_significant" in df_branch.columns else 0,
            "jump_significant": int(df_branch["jump_significant"].fillna(False).sum()) if "jump_significant" in df_branch.columns else 0,
            "output_file": str(branch_output),
            "metadata_file": str(metadata_path) if metadata_path is not None else None,
            "paths_file": str(branch_paths_file) if branch_paths_file is not None else None,
        }
        return df_branch, stats

    # Step 1: Build or load manifest
    if run_upstream:
        if args.overwrite or args.force_manifest or not manifest_file.exists():
            if args.flat_lc_dir:
                log(f"Building flat-directory manifest for mag_bin={args.mag_bin} from {Path(args.flat_lc_dir).expanduser()}...")
            else:
                log(f"Building manifest for mag_bin={args.mag_bin}...")
            df_manifest = build_manifest(
                args.index_root,
                args.lc_root,
                mag_bins=args.mag_bin,
                id_column="asas_sn_id",
                file_ext=args.extension,
                show_progress=args.verbose,
                n_workers=args.workers,
                flat_lc_dir=args.flat_lc_dir.expanduser() if args.flat_lc_dir else None,
                index_file=args.index_file.expanduser() if args.index_file else None,
            )

            # Only keep sources where light curve files exist
            df_manifest = df_manifest[df_manifest["dat_exists"]].reset_index(drop=True)

            log(f"Saving manifest to {manifest_file} ({len(df_manifest)} sources)")
            safe_write_parquet(df_manifest, manifest_file)
        else:
            log(f"Loading existing manifest from {manifest_file}")
            df_manifest = pd.read_parquet(manifest_file)
            log(f"Loaded {len(df_manifest)} sources")

        # Step 2: Apply tags
        if args.overwrite or args.force_tag or not filtered_file.exists():
            log(f"\nApplying tags with {args.workers} workers...")

            # Use lc_dir as the directory path for tag input (path/<id>.dat2)
            df_to_filter = df_manifest.rename(columns={"lc_dir": "path"}).copy()

            df_filtered = apply_tags(
                df_to_filter,
                apply_sparse=not args.skip_sparse,
                min_time_span=args.min_time_span,
                min_points_per_day=args.min_points_per_day,
                apply_vsx=not args.skip_vsx,
                vsx_max_sep_arcsec=args.vsx_max_sep,
                vsx_crossmatch_csv=args.vsx_crossmatch,
                apply_multi_camera=not args.skip_multi_camera,
                min_cameras=args.min_cameras,
                apply_mag_range=not args.skip_mag_range,
                mag_lo=args.mag_lo,
                mag_hi=args.mag_hi,
                n_workers=args.workers,
                show_tqdm=args.verbose,
                rejected_log_csv=str(tags_dir / f"rejected_tag_{mag_bin_tag}.parquet"),
                stats_checkpoint=str(stats_checkpoint_file),
                stats_chunk_size=args.stats_chunk_size,
                file_ext=args.extension,
            )

            # Exclude rows based on tag results
            if not args.pass_all_tags:
                failed_cols = [c for c in df_filtered.columns if c.startswith("failed_") and c != "failed_any"]

                if args.enforce_tags:
                    # Only enforce specified filters
                    enforce_set = {f"failed_{f.strip()}" for f in args.enforce_tags.split(",")}
                    enforce_cols = [c for c in failed_cols if c in enforce_set]
                else:
                    enforce_cols = failed_cols

                if enforce_cols:
                    exclude_mask = df_filtered[enforce_cols].any(axis=1)
                    df_filtered = df_filtered[~exclude_mask].reset_index(drop=True)

            log(f"\nKept {len(df_filtered)}/{len(df_manifest)} sources after tagging")
            log(f"Saving filtered manifest to {filtered_file}")
            safe_write_parquet(df_filtered, filtered_file)
        else:
            log(f"\nLoading existing filtered manifest from {filtered_file}")
            df_filtered = pd.read_parquet(filtered_file)
            log(f"Loaded {len(df_filtered)} filtered sources")

    # Test-run sampling: cap sources to limit expensive downstream steps
    if run_upstream and args.test_run and len(df_filtered) > args.test_run_n:
        log(f"\n[TEST RUN] Sampling {args.test_run_n}/{len(df_filtered)} sources")
        df_filtered = df_filtered.sample(n=args.test_run_n, random_state=42).reset_index(drop=True)

    # Step 2.5: Flag raw-space camera medians for audit. These are not hard
    # exclusions; event detection rechecks cameras in residual space.
    camera_median_file = tags_dir / f"camera_medians_{mag_bin_tag}.parquet"
    if run_upstream and (not args.skip_camera_median) and ("mag_bin" in df_filtered.columns):
        rerun_camera_median = bool(args.overwrite or args.force_tag or not camera_median_file.exists())
        cam_cache = None
        if not rerun_camera_median:
            log(f"\nLoading cached camera median results from {camera_median_file}")
            cam_cache = pd.read_parquet(camera_median_file)
            if RAW_MEDIAN_SUSPECT_COL not in cam_cache.columns:
                rerun_camera_median = True
                cam_cache = None
                log("Cached camera median results use old hard-exclusion schema; recomputing.")

        if rerun_camera_median:
            log(f"\nApplying camera median filter (tolerance={args.camera_median_tolerance} mag)...")
            # Camera median validation needs per-source file paths (.dat2 -> .raw2).
            # Keep the original path column unchanged for downstream code.
            camera_median_df = df_filtered.copy()
            if "dat_path" in camera_median_df.columns:
                camera_median_df["path"] = camera_median_df["dat_path"]
            camera_median_checkpoint = tags_dir / f"camera_medians_{mag_bin_tag}_CHECKPOINT.parquet"
            df_camera = filter_camera_medians(
                camera_median_df,
                mag_tolerance=args.camera_median_tolerance,
                show_tqdm=args.verbose,
                n_workers=args.workers,
                checkpoint_path=str(camera_median_checkpoint),
            )
            df_filtered[RAW_MEDIAN_SUSPECT_COL] = df_camera[RAW_MEDIAN_SUSPECT_COL]
            safe_write_parquet(df_filtered[["source_id", RAW_MEDIAN_SUSPECT_COL]], camera_median_file)
        else:
            df_filtered = df_filtered.merge(cam_cache, on="source_id", how="left")
        if RAW_MEDIAN_SUSPECT_COL not in df_filtered.columns:
            df_filtered[RAW_MEDIAN_SUSPECT_COL] = ""
        n_with_suspects = (df_filtered[RAW_MEDIAN_SUSPECT_COL].fillna("") != "").sum()
        log(f"Found {n_with_suspects}/{len(df_filtered)} sources with raw median suspect cameras")

    # Step 2.75: Pre-events periodicity gate and branch split
    if run_upstream and args.apply_pre_periodicity_gate and not df_filtered.empty:
        rerun_pre_periodicity = bool(args.overwrite or args.force_tag or not pre_periodicity_file.exists())
        if not rerun_pre_periodicity:
            log(f"\nLoading cached pre-periodicity gate results from {pre_periodicity_file}")
            df_gate = pd.read_parquet(pre_periodicity_file)
            cached_router_ok = False
            if "pre_periodicity_router_mode" in df_gate.columns:
                cached_router_modes = (
                    df_gate["pre_periodicity_router_mode"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                    .tolist()
                )
                cached_router_ok = cached_router_modes == [PREGATE_ROUTER_MODE.lower()]
            if "pre_periodic_flag" not in df_gate.columns or not cached_router_ok:
                rerun_pre_periodicity = True
                log(
                    "Cached pre-periodicity gate output is incompatible with the requested "
                    f"router mode '{PREGATE_ROUTER_MODE}'; recomputing."
                )

        if rerun_pre_periodicity:
            log(
                "\nRunning pre-events periodicity gate "
                f"({PREGATE_ROUTER_MODE}, workers={args.pre_periodicity_workers})..."
            )
            df_gate = apply_pre_periodicity_gate(
                df_filtered,
                path_col="dat_path" if "dat_path" in df_filtered.columns else "path",
                excluded_cameras_col=None,
                bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
                clean_max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
                clean_max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
                min_period=args.pre_periodicity_min_period,
                max_period=args.pre_periodicity_max_period,
                n_periods=args.pre_periodicity_n_periods,
                ce_snr_threshold=args.pre_periodicity_ce_snr_threshold,
                min_points=args.pre_periodicity_min_points,
                scatter_ratio_max=args.pre_periodicity_scatter_ratio_max,
                workers=args.pre_periodicity_workers,
                checkpoint_path=pre_periodicity_checkpoint,
                show_tqdm=args.verbose,
            )
            safe_write_parquet(df_gate, pre_periodicity_file)

        if "pre_periodic_flag" not in df_gate.columns:
            raise ValueError(
                f"Pre-periodicity gate output at {pre_periodicity_file} is missing 'pre_periodic_flag'"
            )

        df_periodic_candidates = df_gate[df_gate["pre_periodic_flag"].fillna(False)].reset_index(drop=True)
        df_filtered = df_gate[~df_gate["pre_periodic_flag"].fillna(False)].reset_index(drop=True)

        safe_write_parquet(df_periodic_candidates, periodic_branch_file)
        safe_write_parquet(df_filtered, stochastic_branch_file)

        periodic_paths: list[str] = []
        if not df_periodic_candidates.empty:
            periodic_path_col = "dat_path" if "dat_path" in df_periodic_candidates.columns else "path"
            periodic_paths = [str(value) for value in df_periodic_candidates[periodic_path_col].tolist()]
        with open(periodic_paths_file, "w") as handle:
            for item in periodic_paths:
                handle.write(f"{item}\n")

        label_counts = (
            df_gate["pre_periodicity_label"].value_counts(dropna=False).to_dict()
            if "pre_periodicity_label" in df_gate.columns else {}
        )
        pre_periodicity_stats = {
            "total_input": int(len(df_gate)),
            "periodic_branch": int(len(df_periodic_candidates)),
            "stochastic_branch": int(len(df_filtered)),
            "label_counts": {str(key): int(value) for key, value in label_counts.items()},
            "pre_periodicity_file": str(pre_periodicity_file),
            "periodic_branch_file": str(periodic_branch_file),
            "stochastic_branch_file": str(stochastic_branch_file),
            "periodic_paths_file": str(periodic_paths_file),
        }
        log(
            f"Pre-periodicity gate routed {len(df_periodic_candidates)}/{len(df_gate)} "
            "candidates to the periodic branch"
        )

    periodic_audit_cols = [
        "pre_periodicity_label",
        "pre_periodic_flag",
        "pre_periodicity_selected_period",
        "pre_periodicity_method",
    ]

    def _build_branch_metadata_df(df_branch: pd.DataFrame, path_col: str) -> pd.DataFrame | None:
        meta_cols = [path_col]
        if not args.skip_vsx and "vsx_sep_arcsec" in df_branch.columns and "vsx_class" in df_branch.columns:
            meta_cols.extend(["vsx_sep_arcsec", "vsx_class"])
        if RAW_MEDIAN_SUSPECT_COL in df_branch.columns:
            meta_cols.append(RAW_MEDIAN_SUSPECT_COL)
        for col in periodic_audit_cols:
            if col in df_branch.columns and col not in meta_cols:
                meta_cols.append(col)
        if len(meta_cols) == 1:
            return None
        return df_branch[meta_cols].rename(columns={path_col: "lc_path"}).copy()

    paths_file = paths_dir / f"filtered_paths_{mag_bin_tag}.txt"
    if run_upstream:
        if ctx.run_log_file.exists():
            try:
                with ctx.run_log_file.open("a") as f:
                    f.write(f"paths_file: {paths_file}\n")
                    f.write(f"periodic_paths_file: {periodic_paths_file}\n")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not update run log with branch paths files: {e}")

        stochastic_file_col = "dat_path" if "dat_path" in df_filtered.columns else "path"
        stochastic_paths = [str(value) for value in df_filtered[stochastic_file_col].tolist()] if not df_filtered.empty else []
        stochastic_metadata_df = _build_branch_metadata_df(df_filtered, stochastic_file_col) if not df_filtered.empty else None

        periodic_file_col = "dat_path" if "dat_path" in df_periodic_candidates.columns else "path"
        periodic_paths = [str(value) for value in df_periodic_candidates[periodic_file_col].tolist()] if not df_periodic_candidates.empty else []
        periodic_metadata_df = _build_branch_metadata_df(df_periodic_candidates, periodic_file_col) if not df_periodic_candidates.empty else None

        if stochastic_paths:
            log(f"\nPreparing to run stochastic branch events on {len(stochastic_paths)} light curves...")
        else:
            log("\nNo stochastic-branch sources to process after filtering.")

        if periodic_paths:
            log(
                "\nPreparing to run periodic-branch residual events on "
                f"{len(periodic_paths)} light curves..."
            )
        elif args.apply_pre_periodicity_gate:
            log("\nNo periodic-branch sources to process after the pre-periodicity gate.")

        df_stochastic_events, stochastic_stats = _run_events_branch(
            stochastic_paths,
            branch_name="stochastic",
            branch_output=stochastic_branch_events_output,
            metadata_df=stochastic_metadata_df,
            branch_paths_file=paths_file,
        )

        df_periodic_events = pd.DataFrame()
        periodic_stats = {
            "branch": "periodic",
            "baseline_func": "phase_template",
            "total_input": 0,
            "attempted_this_run": 0,
            "skipped_by_checkpoint": 0,
            "total_results": 0,
            "dip_significant": 0,
            "jump_significant": 0,
            "output_file": str(periodic_branch_events_output),
            "metadata_file": None,
            "paths_file": str(periodic_paths_file),
        }
        if args.apply_pre_periodicity_gate:
            df_periodic_events, periodic_stats = _run_events_branch(
                periodic_paths,
                branch_name="periodic",
                branch_output=periodic_branch_events_output,
                baseline_func_override="phase_template",
                metadata_df=periodic_metadata_df,
                branch_paths_file=periodic_paths_file,
            )

        branch_frames = [df for df in (df_stochastic_events, df_periodic_events) if not df.empty]
        if branch_frames:
            df_events_merged = pd.concat(branch_frames, ignore_index=True)
            if "lc_path" in df_events_merged.columns:
                df_events_merged = df_events_merged.drop_duplicates(subset=["lc_path"], keep="last")
            results_files = _write_events_output(df_events_merged, events_output, events_format)
            log(f"\nMerged branch outputs into canonical events product at {events_output}")
        else:
            if args.overwrite:
                clear_existing_output(_normalized_output_path(events_output, events_format), events_format)
            results_files = []
            log("\nNo event-branch results were produced.")
        branch_detection_stats = {
            "stochastic": stochastic_stats,
            "periodic": periodic_stats if args.apply_pre_periodicity_gate else None,
        }
    else:
        results_files = _output_files_for_path(events_output, events_format)

    # Generate run summary with results statistics
    run_end_time = datetime.now()
    try:
        summary = build_run_summary(
            previous_summary=summary_state if isinstance(summary_state, dict) else {},
            run_start_time=run_start_time,
            run_end_time=run_end_time,
            config_fingerprint=config_fingerprint,
            run_upstream=run_upstream,
            manifest_total_sources=(int(len(df_manifest)) if run_upstream else None),
            manifest_filtered_sources=(int(len(df_filtered) + len(df_periodic_candidates)) if run_upstream else None),
            artifact_context={
                "stage": stage,
                "bundle_lightcurve_count": int(bundle_lightcurve_count),
                "manifests_file_count": int(manifests_file_count),
                "tags_file_count": int(tags_file_count),
                "paths_file_count": int(paths_file_count),
                "imported_run_params_snapshot": str(imported_run_params_snapshot) if imported_run_params_snapshot else None,
                "imported_run_summary_snapshot": str(imported_run_summary_snapshot) if imported_run_summary_snapshot else None,
            },
        )
        if pre_periodicity_stats is not None:
            summary["pre_periodicity_gate"] = pre_periodicity_stats
        if branch_detection_stats is not None:
            summary["events_branches"] = branch_detection_stats

        # Tag rejection breakdown
        rejected_log = tags_dir / f"rejected_tag_{mag_bin_tag}.parquet"
        if rejected_log.exists():
            try:
                df_rejected = pd.read_parquet(rejected_log)
                if "reason" in df_rejected.columns:
                    rejection_counts = df_rejected["reason"].value_counts().to_dict()
                    summary["tag_rejections"] = {
                        "total_rejected": len(df_rejected),
                        "by_reason": rejection_counts,
                    }
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse rejection log: {e}")

        if results_files:
            try:
                df_results = _load_events_output(events_output, events_format)

                detection_stats = {
                    "total_detections": len(df_results),
                    "unique_sources": df_results["lc_path"].nunique() if "lc_path" in df_results.columns else None,
                }

                # Count significant detections
                if "dip_significant" in df_results.columns:
                    detection_stats["dip_significant"] = int(df_results["dip_significant"].sum())
                if "jump_significant" in df_results.columns:
                    detection_stats["jump_significant"] = int(df_results["jump_significant"].sum())

                # Event type counts
                if "event_type" in df_results.columns:
                    detection_stats["by_event_type"] = df_results["event_type"].value_counts().to_dict()

                summary["detection_stats"] = detection_stats

            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse detection results: {e}")

        # Write summary (will be updated again if filter/postprocess run)
        write_run_summary(ctx, summary)

        log(f"\nRun summary saved to {run_summary_file}")

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run summary: {e}")

    # Step 5: Apply filters (optional)
    if run_upstream and args.run_filter and results_files:
        post_filter_output = results_dir / f"lc_events_filtered_{mag_bin_tag}.parquet"
        if _should_skip_filter_stage(
            output_path=post_filter_output,
            overwrite=bool(args.overwrite),
            branch_detection_stats=branch_detection_stats,
        ):
            log(
                "\n=== Step 5: Filtered output exists and no new events were "
                f"processed, skipping: {post_filter_output} ==="
            )
        else:
            log("\n=== Step 5: Applying filters ===")
            try:
                # Load events results
                if events_format == "parquet_chunk":
                    df_events = pd.concat([read_feature_table(f) for f in results_files], ignore_index=True)
                else:
                    df_events = load_table(results_files[0])

                # Apply filters
                filter_kwargs = _build_filter_kwargs(args)
                if stage == "cluster":
                    # Cluster stage must avoid internet catalog lookups.
                    filter_kwargs["apply_gaia_ruwe_validation"] = False
                    filter_kwargs["apply_gaia_pm_validation"] = False
                    filter_kwargs["apply_periodic_catalog_validation"] = False

                # Add gaia_id from ASASSN index (needed for validate_gaia_ruwe/pm filters)
                if filter_kwargs.get("apply_gaia_ruwe_validation", True) or filter_kwargs.get("apply_gaia_pm_validation", True):
                    _gaia_index_path, _ = _resolve_asassn_index_path(out_dir, index_override=getattr(args, "index_file", None))
                    if _gaia_index_path:
                        df_events = _add_gaia_ids_from_index(df_events, _gaia_index_path)
                    else:
                        _log("Warning: ASASSN index not found; gaia_id will be missing — RUWE/PM filters will have no matches")

                df_post_filtered = apply_filters(df_events, **filter_kwargs)

                # Save filtered results
                save_table(df_post_filtered, post_filter_output)
                log(f"Filtered results saved to {post_filter_output}")

                # Update summary with filter stats
                n_passed = int((~df_post_filtered["failed_any"]).sum()) if "failed_any" in df_post_filtered.columns else len(df_post_filtered)
                n_failed = int(df_post_filtered["failed_any"].sum()) if "failed_any" in df_post_filtered.columns else 0
                summary["filter_stats"] = {
                    "total_input": len(df_events),
                    "passed": n_passed,
                    "failed": n_failed,
                    "pass_rate": n_passed / len(df_events) if len(df_events) > 0 else 0.0,
                }
                summary["post_filter_stats"] = summary["filter_stats"]

                # Overwrite summary with updated stats
                with open(run_summary_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

                log(f"Filter: {n_passed}/{len(df_events)} passed")

            except Exception as e:
                print(f"Error in filter step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 6: Enrich with compute_stats (optional, runs immediately after filter)
    if run_upstream and args.run_enrich:
        if not args.run_filter:
            print("Warning: --run-enrich requires --run-filter. Skipping enrichment.")
        else:
            log("\n=== Step 6: Enriching with light curve stats ===")
            try:
                # Enrichment now runs directly from filter output
                post_filter_output = results_dir / f"lc_events_filtered_{mag_bin_tag}.parquet"

                if post_filter_output.exists():
                    df_to_enrich = load_passing_table(post_filter_output)
                else:
                    print(f"Warning: No filter output found at {post_filter_output}")
                    df_to_enrich = None

                if df_to_enrich is not None:
                    df_passed = df_to_enrich

                    if len(df_passed) > 0:
                        log(f"Enriching {len(df_passed)} candidates with compute_stats...")

                        # Checkpoint support
                        enrich_checkpoint = results_dir / f"lc_events_enriched_{mag_bin_tag}_CHECKPOINT.parquet"
                        if args.overwrite and enrich_checkpoint.exists():
                            enrich_checkpoint.unlink()

                        already_enriched: set[str] = set()
                        enriched_rows: list[dict] = []
                        if enrich_checkpoint.exists():
                            try:
                                df_ckpt = pd.read_parquet(enrich_checkpoint)
                                enriched_rows = df_ckpt.to_dict("records")
                                already_enriched = set(df_ckpt["lc_path"].astype(str))
                                log(f"Loaded enrichment checkpoint: {len(already_enriched)} already enriched")
                            except Exception as e:
                                log(f"Warning: could not load enrichment checkpoint: {e}")

                        ENRICH_SAVE_INTERVAL = 10000
                        n_enrich_workers, worker_note = _effective_enrich_workers(args)
                        if worker_note:
                            log(f"Using {n_enrich_workers} workers for compute_stats enrichment ({worker_note})")
                        else:
                            log(f"Using {n_enrich_workers} workers for compute_stats enrichment")

                        # Generate tasks lazily so large post-filter tables do not
                        # get duplicated into a second full list of row dicts.
                        df_passed_columns = list(df_passed.columns)
                        path_col_idx = df_passed_columns.index("lc_path")

                        def _iter_enrichment_tasks():
                            for values in df_passed.itertuples(index=False, name=None):
                                raw_path = values[path_col_idx]
                                lc_path = Path(raw_path)
                                path_key = str(lc_path)
                                if path_key in already_enriched:
                                    continue
                                row_dict = dict(zip(df_passed_columns, values))
                                if not lc_path.exists():
                                    enriched_rows.append(row_dict)
                                    continue
                                asassn_id = lc_path.stem.split("-")[0]
                                dir_path = str(lc_path.parent)
                                yield (row_dict, asassn_id, dir_path, args.enrich_compute_ls, args.extension)

                        new_count = 0
                        with ProcessPoolExecutor(max_workers=n_enrich_workers) as executor:
                            for result in tqdm(
                                executor.map(
                                    _enrich_row_worker,
                                    _iter_enrichment_tasks(),
                                    chunksize=64,
                                ),
                                desc="compute_stats",
                                disable=not args.verbose,
                            ):
                                enriched_rows.append(result)
                                new_count += 1
                                if new_count % ENRICH_SAVE_INTERVAL == 0:
                                    pd.DataFrame(enriched_rows).to_parquet(
                                        enrich_checkpoint, index=False,
                                        compression=PARQUET_CACHE_COMPRESSION,
                                    )

                        df_enriched = pd.DataFrame(enriched_rows)
                        if not df_enriched.empty:
                            df_enriched = df_enriched.drop_duplicates(subset=["lc_path"], keep="last")

                        # Save enriched results
                        enrich_output = results_dir / f"lc_events_enriched_{mag_bin_tag}.parquet"
                        save_table(df_enriched, enrich_output)
                        log(f"Enriched results saved to {enrich_output}")

                        # Clean up checkpoint
                        if enrich_checkpoint.exists():
                            enrich_checkpoint.unlink()

                        # Update summary
                        n_stats_cols = len([c for c in df_enriched.columns if c.startswith("stats_")])
                        summary["enrichment_stats"] = {
                            "total_enriched": len(df_enriched),
                            "stats_columns_added": n_stats_cols,
                        }

                        with open(run_summary_file, "w") as f:
                            json.dump(summary, f, indent=2, default=str)

                        log(f"Enrichment: {len(df_enriched)} candidates, {n_stats_cols} stats columns added")
                    else:
                        log("No passing candidates to enrich.")

            except Exception as e:
                print(f"Error in enrichment step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 7: Generate review plots (optional)
    if run_upstream and args.run_postprocess:
        if not args.run_filter:
            print("Warning: --run-postprocess requires --run-filter. Skipping postprocess plots.")
        else:
            log("\n=== Step 7: Generating candidate plots ===")
            try:
                post_filter_output = results_dir / f"lc_events_filtered_{mag_bin_tag}.parquet"
                if not post_filter_output.exists():
                    print(f"Warning: No filter output found at {post_filter_output}; skipping postprocess plots.")
                else:
                    plots_out = out_dir / "plots" / "candidates"
                    baseline_for_plots = {
                        "gp": "per_camera_gp",
                        "gp_masked": "per_camera_gp",
                        "per_camera_median": "per_camera_median",
                        "global_median": "global_median",
                    }.get(str(args.baseline_func), "per_camera_gp")

                    plot_summary = plot_passing_candidates(
                        post_filter_output,
                        plots_out,
                        require_failed_any_false=True,
                        max_plots=args.max_plots,
                        baseline=baseline_for_plots,
                        baseline_kwargs={},
                        skip_events=False,
                        plot_fits=False,
                        format=args.plot_format,
                        show=False,
                        verbose=args.verbose,
                        workers=max(1, int(args.workers)),
                        logbf_threshold_dip=float(args.logbf_threshold_dip),
                        logbf_threshold_jump=float(args.logbf_threshold_jump),
                        jd_offset=JD_OFFSET,
                        clean_max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
                        clean_max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
                        run_params=run_params if 'run_params' in locals() else None,
                        filter_bad_cameras=True,
                        bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
                        show_tqdm=args.verbose,
                    )

                    summary["postprocess_stats"] = {
                        "output_dir": str(plots_out),
                        "total_selected": int(plot_summary.get("total_selected", 0)),
                        "plotted": int(plot_summary.get("plotted", 0)),
                        "failed": int(plot_summary.get("failed", 0)),
                        "phase_plotted": int(plot_summary.get("phase_plotted", 0)),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Postprocess plots written to {plots_out}")
            except Exception as e:
                print(f"Error in postprocess plotting step: {e}")
                if args.verbose:

                    traceback.print_exc()



    # Merge per-mag-bin outputs into canonical (untagged) files for downstream stages.
    # Only merge when entering downstream/home phase — NOT during concurrent cluster runs.
    if run_downstream:
        log("\n=== Merging per-mag-bin outputs ===")
        merge_started = time.perf_counter()
        for merge_prefix in ("lc_events_results", "lc_events_filtered", "lc_events_enriched"):
            pattern = f"{merge_prefix}_*" if merge_prefix == "lc_events_results" else f"{merge_prefix}_*.parquet"
            tagged_outputs = sorted(results_dir.glob(pattern))
            # Exclude checkpoint and temp files from merging
            tagged_outputs = [
                f for f in tagged_outputs
                if "_CHECKPOINT" not in f.name and "_PROCESSED" not in f.name and not f.name.endswith(".tmp")
            ]
            merged_path = results_dir / merge_prefix if merge_prefix == "lc_events_results" else results_dir / f"{merge_prefix}.parquet"
            if tagged_outputs:
                try:
                    if merge_prefix != "lc_events_results" and _copy_single_tagged_table_output(tagged_outputs, merged_path):
                        log(f"Merged 1 output into {merged_path} by copying {tagged_outputs[0].name}")
                        continue
                    elif merge_prefix == "lc_events_results":
                        dfs = [
                            _load_events_output(path, events_format) if path.is_dir() else read_feature_table(path)
                            for path in tagged_outputs
                        ]
                    else:
                        dfs = [read_feature_table(f) for f in tagged_outputs]
                    merged = pd.concat(dfs, ignore_index=True)
                    if "lc_path" in merged.columns:
                        merged = merged.drop_duplicates(subset=["lc_path"], keep="last")
                    if merge_prefix == "lc_events_results":
                        _write_events_output(merged, merged_path, events_format)
                    else:
                        save_table(merged, merged_path)
                    log(f"Merged {len(tagged_outputs)} outputs into {merged_path} ({len(merged)} rows)")
                except Exception as e:
                    log(f"Warning: could not merge {merge_prefix} files: {e}")
        log(f"Merge step completed in {time.perf_counter() - merge_started:.1f}s")

    post_filter_output = results_dir / "lc_events_filtered.parquet"
    has_post_filter_output = post_filter_output.exists()

    # Home-only external catalog validations (Gaia RUWE + periodic catalog)
    if stage == "home" and args.run_filter and has_post_filter_output:
        home_validation_steps = ["Gaia RUWE", "periodic catalog"]
        if args.apply_periodicity_validation:
            home_validation_steps.append("periodicity")
        log(f"\n=== Home External Validation: {' + '.join(home_validation_steps)} ===")
        validation_started = time.perf_counter()
        try:
            index_file, index_candidates = _resolve_asassn_index_path(out_dir, index_override=args.index_file)
            if index_file is None:
                tried_paths = ", ".join(str(p) for p in index_candidates[:6])
                if len(index_candidates) > 6:
                    tried_paths += ", ..."
                if not tried_paths:
                    tried_paths = "(no candidate paths)"
                raise FileNotFoundError(
                    "Index file not found for home external validation. "
                    f"Tried: {tried_paths}. "
                    "Expected bundle_assets/asassn_index_full.parquet from a --full-bundle export, "
                    "or pass --index-file explicitly."
                )
            log(f"Using index file for home external validation: {index_file}")

            external_validation_cmd = _build_home_external_validation_cmd(
                args,
                post_filter_output=post_filter_output,
                index_file=index_file,
            )

            result = subprocess.run(external_validation_cmd, check=False)
            if result.returncode != 0:
                print(f"Home external validation failed with exit code {result.returncode}")
                sys.exit(result.returncode)

            has_post_filter_output = post_filter_output.exists()
            log(f"Home external validation wrote updated filtered results to {post_filter_output}")
            log(f"Home external validation completed in {time.perf_counter() - validation_started:.1f}s")
        except Exception as e:
            print(f"Error in home external validation step: {e}")
            if args.verbose:

                traceback.print_exc()
            sys.exit(1)



    if run_downstream and (args.run_characterize or args.run_dust) and (not has_post_filter_output):
        print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping characterization.")

    # Step 7b: Auto-fetch Gaia data for characterization (incremental)
    if run_downstream and (args.run_characterize or args.run_dust) and has_post_filter_output:
        log("\n=== Ensuring local Gaia catalog is up to date ===")
        gaia_fetch_started = time.perf_counter()
        try:
            gaia_catalog_path = args.gaia_cache.expanduser() if args.gaia_cache else GAIA_LOCAL_CATALOG
            gaia_ids = _extract_gaia_ids(
                post_filter_output,
                args.characterize_crossmatch.expanduser(),
                only_passers=True,
            )
            if gaia_ids:
                fetch_gaia_catalog(gaia_ids, output_path=gaia_catalog_path, chunk_size=args.gaia_fetch_chunk_size)
            else:
                log("No Gaia IDs found; skipping Gaia fetch.")
            log(f"Gaia catalog check completed in {time.perf_counter() - gaia_fetch_started:.1f}s")
        except Exception as e:
            print(f"Warning: Gaia auto-fetch failed: {e}")
            if args.verbose:

                traceback.print_exc()

    # Step 8: Characterization + dust (optional)
    if run_downstream and (args.run_characterize or args.run_dust) and has_post_filter_output:
        log("\n=== Step 8: Characterizing candidates ===")
        characterize_started = time.perf_counter()
        try:
            df_char = load_passing_table(post_filter_output)

            if "lc_path" in df_char.columns and "asas_sn_id" not in df_char.columns:
                def _extract_id(path_str: str) -> str:
                    name = Path(path_str).name
                    return Path(name).stem.split("-")[0]

                df_char["asas_sn_id"] = df_char["lc_path"].astype(str).map(_extract_id)

            # Use full characterize pipeline (single source of truth)
            char_checkpoint = results_dir / "lc_events_characterized_CHECKPOINT.parquet"
            if args.overwrite and char_checkpoint.exists():
                char_checkpoint.unlink()

            starhorse_arg = args.characterize_starhorse if args.run_characterize else None
            df_char = characterize_candidates_df(
                df_char,
                crossmatch=args.characterize_crossmatch.expanduser(),
                chunk_size=args.characterize_chunk_size,
                cache=args.gaia_cache.expanduser() if args.gaia_cache else (out_dir / "gaia_cache" / "gaia_cache.parquet"),
                dust=args.run_dust,
                starhorse=starhorse_arg,
                starhorse_cache=args.characterize_starhorse_cache.expanduser() if args.characterize_starhorse_cache else None,
                run_banyan=args.run_characterize and args.characterize_banyan,
                run_iphas=args.run_characterize and args.characterize_iphas,
                run_sfr=args.run_characterize and args.characterize_sfr,
                run_clusters=args.run_characterize and args.characterize_clusters,
                run_unwise=args.run_characterize and args.characterize_unwise,
                unwise_checkpoint_every=args.characterize_unwise_checkpoint_every,
                checkpoint_path=char_checkpoint,
            )

            characterize_output = results_dir / "lc_events_characterized.parquet"
            save_table(df_char, characterize_output)
            log(f"Characterization results saved to {characterize_output}")
            log(f"Step 8 completed in {time.perf_counter() - characterize_started:.1f}s")

        except Exception as e:
            print(f"Error in characterization step: {e}")
            if args.verbose:

                traceback.print_exc()

    # Step 9: SED photometry (enabled by default)
    sed_photometry_output = results_dir / "sed_photometry.parquet"
    sed_model_fits_output = results_dir / "sed_model_fits.parquet"
    sed_model_curves_output = results_dir / "sed_model_curves.parquet"
    if run_downstream and args.run_sed_photometry:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping SED photometry.")
        elif sed_photometry_output.exists() and not args.overwrite:
            log(f"\n=== Step 9: SED photometry output exists, skipping: {sed_photometry_output} ===")
        else:
            log("\n=== Step 9: Fetching SED photometry ===")
            sed_started = time.perf_counter()
            try:
                from malca.review.sed import SED_COLUMNS, fetch_sed_photometry, resolve_sed_sources

                characterize_output = results_dir / "lc_events_characterized.parquet"
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if characterize_output.exists():
                    df_sed_in = load_passing_table(characterize_output)
                elif post_filter_output.exists():
                    df_sed_in = load_passing_table(post_filter_output)
                else:
                    df_sed_in = None

                if df_sed_in is None:
                    log("Warning: no suitable input found for SED photometry, skipping")
                else:
                    df_sed_in = ensure_candidate_id(df_sed_in, prefix="stv")
                    sources = resolve_sed_sources(args.sed_sources)
                    log(f"SED input: {len(df_sed_in)} passing candidates; sources={','.join(sources)}")
                    sed_rows = fetch_sed_photometry(df_sed_in, sources=sources, progress_callback=log)
                    for col in SED_COLUMNS:
                        if col not in sed_rows.columns:
                            sed_rows[col] = None
                    sed_rows = sed_rows[SED_COLUMNS]
                    save_table(sed_rows, sed_photometry_output)

                    by_source = (
                        sed_rows.groupby("source", dropna=False).size().to_dict()
                        if (not sed_rows.empty and "source" in sed_rows.columns)
                        else {}
                    )
                    summary["sed_photometry_stats"] = {
                        "rows_input": int(len(df_sed_in)),
                        "rows_output": int(len(sed_rows)),
                        "sources_requested": list(sources),
                        "rows_by_source": {str(k): int(v) for k, v in by_source.items()},
                        "output": str(sed_photometry_output),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)

                    log(f"SED photometry saved to {sed_photometry_output} ({len(sed_rows)} rows)")
                    log(f"Step 9 completed in {time.perf_counter() - sed_started:.1f}s")
            except Exception as e:
                print(f"Error in SED photometry step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 9b: Castelli/Kurucz SED atmosphere fitting
    if run_downstream and args.run_sed_photometry and args.fit_atmosphere:
        if not sed_photometry_output.exists():
            log(f"Warning: SED model fitting requires {sed_photometry_output}. Skipping atmosphere fit.")
        elif sed_model_fits_output.exists() and sed_model_curves_output.exists() and not args.overwrite:
            log(f"\n=== Step 9b: SED model outputs exist, skipping: {sed_model_fits_output}, {sed_model_curves_output} ===")
        else:
            log("\n=== Step 9b: Fitting Castelli/Kurucz SED atmosphere models ===")
            sed_model_started = time.perf_counter()
            try:
                from malca.enrichment.sed_model import SED_MODEL_CURVE_COLUMNS, SED_MODEL_FIT_COLUMNS, fit_sed_models

                characterize_output = results_dir / "lc_events_characterized.parquet"
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if characterize_output.exists():
                    df_model_in = load_passing_table(characterize_output)
                elif post_filter_output.exists():
                    df_model_in = load_passing_table(post_filter_output)
                else:
                    df_model_in = None

                if df_model_in is None:
                    log("Warning: no suitable input found for SED model fitting, skipping")
                else:
                    sed_rows_for_model = load_table(sed_photometry_output)
                    log(f"SED model input: {len(df_model_in)} passing candidates; {len(sed_rows_for_model)} SED rows")
                    sed_model_fits, sed_model_curves = fit_sed_models(
                        df_model_in,
                        sed_rows_for_model,
                        progress_callback=log,
                    )
                    for col in SED_MODEL_FIT_COLUMNS:
                        if col not in sed_model_fits.columns:
                            sed_model_fits[col] = None
                    for col in SED_MODEL_CURVE_COLUMNS:
                        if col not in sed_model_curves.columns:
                            sed_model_curves[col] = None
                    sed_model_fits = sed_model_fits[SED_MODEL_FIT_COLUMNS]
                    sed_model_curves = sed_model_curves[SED_MODEL_CURVE_COLUMNS]
                    save_table(sed_model_fits, sed_model_fits_output)
                    save_table(sed_model_curves, sed_model_curves_output)
                    n_ok = int((sed_model_fits["status"].astype(str) == "ok").sum()) if "status" in sed_model_fits.columns else 0
                    summary["sed_model_fit_stats"] = {
                        "rows_input": int(len(df_model_in)),
                        "fit_rows_output": int(len(sed_model_fits)),
                        "curve_rows_output": int(len(sed_model_curves)),
                        "fits_ok": int(n_ok),
                        "fits_output": str(sed_model_fits_output),
                        "curves_output": str(sed_model_curves_output),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)

                    log(
                        f"SED model fits saved to {sed_model_fits_output} "
                        f"({len(sed_model_fits)} rows, {n_ok} ok)"
                    )
                    log(f"SED model curves saved to {sed_model_curves_output} ({len(sed_model_curves)} rows)")
                    log(f"Step 9b completed in {time.perf_counter() - sed_model_started:.1f}s")
            except Exception as e:
                print(f"Error in SED model fitting step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 10: Run classification (optional)
    if run_downstream and args.run_classify:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping classification.")
        else:
            classify_output = results_dir / "lc_events_classified.parquet"
            if classify_output.exists() and not args.overwrite:
                log(f"\n=== Step 10: Classification output exists, skipping: {classify_output} ===")
            else:
                log("\n=== Step 10: Running classification ===")
                classify_started = time.perf_counter()
                try:
                    characterize_output = results_dir / "lc_events_characterized.parquet"
                    post_filter_output = results_dir / "lc_events_filtered.parquet"

                    if characterize_output.exists():
                        df_post_filtered = load_passing_table(characterize_output)
                    elif post_filter_output.exists():
                        df_post_filtered = load_passing_table(post_filter_output)
                    else:
                        df_post_filtered = None
                        print(f"Warning: filter output not found at {post_filter_output}")

                    if df_post_filtered is not None:
                        # Run classification on passing candidates
                        df_passed = df_post_filtered

                        if len(df_passed) > 0:
                            df_classified = compute_all_classifications(df_passed)

                            # Save classified results
                            save_table(df_classified, classify_output)
                            log(f"Classification results saved to {classify_output}")

                            # Update summary with classification stats
                            class_counts = df_classified["final_class"].value_counts().to_dict() if "final_class" in df_classified.columns else {}
                            summary["classification_stats"] = {
                                "total_classified": len(df_classified),
                                "by_class": class_counts,
                            }

                            # Overwrite summary with updated stats
                            with open(run_summary_file, "w") as f:
                                json.dump(summary, f, indent=2, default=str)

                            log(f"Classification: {len(df_classified)} candidates classified")
                            log(f"Step 10 completed in {time.perf_counter() - classify_started:.1f}s")
                        else:
                            log("No passing candidates to classify.")
                            log(f"Step 10 completed in {time.perf_counter() - classify_started:.1f}s")

                except Exception as e:
                    print(f"Error in classification step: {e}")
                    if args.verbose:

                        traceback.print_exc()

    # Step 11: Neighbor enrichment (optional)
    if run_downstream and args.run_neighbor_enrich:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping neighbor enrichment.")
        else:
            log("\n=== Step 11: Bulk neighbor enrichment ===")
            neighbor_started = time.perf_counter()
            try:
                enrich_output = results_dir / "lc_events_enriched.parquet"
                classify_output = results_dir / "lc_events_classified.parquet"
                characterize_output = results_dir / "lc_events_characterized.parquet"
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if classify_output.exists():
                    df_neighbors_in = load_passing_table(classify_output)
                elif characterize_output.exists():
                    df_neighbors_in = load_passing_table(characterize_output)
                elif enrich_output.exists():
                    df_neighbors_in = load_passing_table(enrich_output)
                elif post_filter_output.exists():
                    df_neighbors_in = load_passing_table(post_filter_output)
                else:
                    df_neighbors_in = None

                if df_neighbors_in is not None:
                    df_neighbors_in = ensure_candidate_id(df_neighbors_in, prefix="stv")
                    neighbor_dir = results_dir / "neighbor_enrichment"
                    neighbor_cache = args.neighbor_cache.expanduser() if args.neighbor_cache else (neighbor_dir / "neighbors_cache.parquet")
                    neighbor_checkpoint = neighbor_dir / "neighbors_CHECKPOINT.parquet"
                    if args.overwrite and neighbor_checkpoint.exists():
                        neighbor_checkpoint.unlink()
                    _, df_neighbor_summary = run_neighbor_enrichment(
                        df_neighbors_in,
                        out_dir=neighbor_dir,
                        radius_arcsec=args.neighbor_radius_arcsec,
                        chunk_size=args.neighbor_chunk_size,
                        cache_file=neighbor_cache,
                        checkpoint_path=neighbor_checkpoint,
                        show_progress=args.verbose,
                    )

                    if not df_neighbor_summary.empty:
                        merged = merge_candidate_columns(
                            df_neighbors_in,
                            ensure_candidate_id(df_neighbor_summary, prefix="stv"),
                            [col for col in df_neighbor_summary.columns if col != "candidate_id"],
                        )
                        save_table(merged, results_dir / "lc_events_neighbors.parquet")

                    summary["neighbor_enrichment_stats"] = {
                        "rows_input": int(len(df_neighbors_in)),
                        "radius_arcsec": float(args.neighbor_radius_arcsec),
                        "chunk_size": int(args.neighbor_chunk_size),
                        "output_dir": str(neighbor_dir),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Neighbor enrichment outputs written to {neighbor_dir}")
                    log(f"Step 11 completed in {time.perf_counter() - neighbor_started:.1f}s")

            except Exception as e:
                print(f"Error in neighbor enrichment step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 12: Spectra availability enrichment (optional)
    if run_downstream and args.run_spectra_enrich:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping spectra enrichment.")
        else:
            log("\n=== Step 12: Spectra availability enrichment ===")
            spectra_started = time.perf_counter()
            try:
                neighbor_output = results_dir / "lc_events_neighbors.parquet"
                enrich_output = results_dir / "lc_events_enriched.parquet"
                classify_output = results_dir / "lc_events_classified.parquet"
                characterize_output = results_dir / "lc_events_characterized.parquet"
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if neighbor_output.exists():
                    df_spectra_in = load_passing_table(neighbor_output)
                elif enrich_output.exists():
                    df_spectra_in = load_passing_table(enrich_output)
                elif classify_output.exists():
                    df_spectra_in = load_passing_table(classify_output)
                elif characterize_output.exists():
                    df_spectra_in = load_passing_table(characterize_output)
                elif post_filter_output.exists():
                    df_spectra_in = load_passing_table(post_filter_output)
                else:
                    df_spectra_in = None

                if df_spectra_in is not None:
                    df_spectra_in = ensure_candidate_id(df_spectra_in, prefix="stv")
                    spectra_dir = results_dir / "spectra_enrichment"
                    spectra_cache = args.spectra_cache.expanduser() if args.spectra_cache else (spectra_dir / "spectra_cache.parquet")
                    spectra_checkpoint = spectra_dir / "spectra_CHECKPOINT.parquet"
                    if args.overwrite and spectra_checkpoint.exists():
                        spectra_checkpoint.unlink()
                    _, spectra_summary = run_spectra_availability(
                        df_spectra_in,
                        out_dir=spectra_dir,
                        radius_arcsec=args.spectra_radius_arcsec,
                        chunk_size=args.spectra_chunk_size,
                        cache_file=spectra_cache,
                        checkpoint_path=spectra_checkpoint,
                        show_progress=args.verbose,
                    )

                    if not spectra_summary.empty:
                        merged = merge_candidate_columns(
                            df_spectra_in,
                            ensure_candidate_id(spectra_summary, prefix="stv"),
                            [col for col in spectra_summary.columns if col != "candidate_id"],
                        )
                        save_table(merged, results_dir / "lc_events_spectra.parquet")

                    summary["spectra_enrichment_stats"] = {
                        "rows_input": int(len(df_spectra_in)),
                        "radius_arcsec": float(args.spectra_radius_arcsec),
                        "chunk_size": int(args.spectra_chunk_size),
                        "output_dir": str(spectra_dir),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Spectra enrichment outputs written to {spectra_dir}")
                    log(f"Step 12 completed in {time.perf_counter() - spectra_started:.1f}s")

            except Exception as e:
                print(f"Error in spectra enrichment step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 13: Post-review vetting (enabled by default)
    if run_downstream and args.run_vetting:
        log("\n=== Step 13: Post-review vetting ===")
        vetting_started = time.perf_counter()
        try:
            # Find the best input file for vetting
            vetting_input = args.vetting_input
            if vetting_input is None:
                for candidate_file in [
                    results_dir / "lc_events_spectra.parquet",
                    results_dir / "lc_events_neighbors.parquet",
                    results_dir / "lc_events_characterized.parquet",
                    post_filter_output,
                ]:
                    if candidate_file.exists():
                        vetting_input = candidate_file
                        break

            if vetting_input is None or not Path(vetting_input).exists():
                log("Warning: no suitable input found for vetting, skipping")
            else:
                df_vet = load_passing_table(Path(vetting_input))
                log(f"Vetting input: {vetting_input} ({len(df_vet)} passing candidates)")

                if args.vetting_min_score is not None and "interest_score" in df_vet.columns:
                    before = len(df_vet)
                    df_vet = df_vet[df_vet["interest_score"] >= args.vetting_min_score].copy()
                    log(f"Filtered to {len(df_vet)} candidates with score >= {args.vetting_min_score} (from {before})")

                vetting_checkpoint = results_dir / "lc_events_vetting_CHECKPOINT.parquet"
                df_vet = vet_candidates(
                    df_vet,
                    run_simbad=not args.no_vetting_simbad,
                    run_gaia_var=not args.no_vetting_gaia_var,
                    run_gaia_epoch=not args.no_vetting_gaia_epoch,
                    run_asassn_var=not args.no_vetting_asassn_var,
                    run_alerce=not args.no_vetting_alerce,
                    run_erosita=not args.no_vetting_erosita,
                    run_atlas=args.vetting_atlas,
                    run_pm_check=not args.no_vetting_pm_check,
                    run_neowise_lc=args.vetting_neowise_lc,
                    simbad_radius_arcsec=args.vetting_simbad_radius,
                    asassn_radius_arcsec=args.vetting_asassn_radius,
                    atlas_token=args.vetting_atlas_token,
                    checkpoint_path=vetting_checkpoint,
                )

                vetting_output = results_dir / "lc_events_vetted.parquet"
                save_table(df_vet, vetting_output)
                log(f"Vetting output: {vetting_output}")

                def _count_col(col, empty=""):
                    s = df_vet.get(col, pd.Series(dtype=str))
                    return int((s != empty).sum()) if not s.empty else 0

                summary["vetting_stats"] = {
                    "rows_input": int(len(df_vet)),
                    "simbad_matches": _count_col("simbad_main_id"),
                    "gaia_var_flagged": int(df_vet.get("gaia_var_flag", pd.Series(dtype=bool)).sum()),
                    "gaia_epoch_available": int(df_vet.get("gaia_epoch_available", pd.Series(dtype=bool)).sum()),
                    "asassn_var_matches": _count_col("asassn_var_type"),
                    "alerce_matches": _count_col("alerce_oid"),
                    "erosita_xray_det": int(df_vet.get("xray_det", pd.Series(dtype=bool)).sum()),
                    "likely_known": int(df_vet.get("vetting_likely_known", pd.Series(dtype=bool)).sum()),
                }
                with open(run_summary_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                log(f"Step 13 completed in {time.perf_counter() - vetting_started:.1f}s")

        except Exception as e:
            print(f"Error in vetting step: {e}")
            if args.verbose:

                traceback.print_exc()

    external_lcs_output = results_dir / "lc_events_external_lcs.parquet"
    multi_survey_output = results_dir / "lc_events_multi_survey_features.parquet"
    external_lc_dir = results_dir / "external_lcs"

    # Step 13b: External light-curve enrichment (explicit/extended only)
    if run_downstream and args.run_external_lcs:
        log("\n=== Step 13b: External light-curve enrichment ===")
        external_started = time.perf_counter()
        try:
            external_input = _first_existing_candidate_result(results_dir, include_extended=False)
            if external_input is None or not external_input.exists():
                log("Warning: no suitable input found for external light-curve enrichment, skipping")
            else:
                df_external_in = load_passing_table(external_input)
                log(f"External LC input: {external_input} ({len(df_external_in)} passing candidates)")
                if df_external_in.empty:
                    log("No passing candidates for external light-curve enrichment")
                else:
                    external_lcs_output, external_lc_dir, df_external = _run_external_lcs_enrichment(
                        df_external_in,
                        results_dir=results_dir,
                        atlas=args.external_lc_atlas,
                        atlas_token=args.external_lc_atlas_token,
                        workers=args.external_lc_workers or 4,
                        refresh_cache=args.external_lc_refresh_cache,
                        overwrite=args.overwrite,
                    )
                    summary["external_lc_stats"] = {
                        "rows_input": int(len(df_external_in)),
                        "rows_output": int(len(df_external)),
                        "output": str(external_lcs_output),
                        "output_dir": str(external_lc_dir),
                        "atlas": bool(args.external_lc_atlas),
                        "workers": int(args.external_lc_workers or 4),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"External light-curve output: {external_lcs_output}")
                    log(f"Step 13b completed in {time.perf_counter() - external_started:.1f}s")

        except Exception as e:
            print(f"Error in external light-curve enrichment step: {e}")
            if args.verbose:
                traceback.print_exc()

    # Step 13c: Multi-survey feature extraction (explicit/extended only)
    if run_downstream and args.run_multi_survey_features:
        log("\n=== Step 13c: Multi-survey feature extraction ===")
        multi_started = time.perf_counter()
        try:
            if external_lcs_output.exists():
                multi_input = external_lcs_output
            else:
                multi_input = _first_existing_candidate_result(results_dir, include_extended=False)

            if multi_input is None or not multi_input.exists():
                log("Warning: no suitable input found for multi-survey features, skipping")
            else:
                df_multi_in = load_passing_table(multi_input)
                log(f"Multi-survey input: {multi_input} ({len(df_multi_in)} passing candidates)")
                if df_multi_in.empty:
                    log("No passing candidates for multi-survey feature extraction")
                else:
                    multi_survey_output, df_multi = _run_multi_survey_features_enrichment(
                        df_multi_in,
                        results_dir=results_dir,
                        external_lc_dir=external_lc_dir,
                    )
                    summary["multi_survey_feature_stats"] = {
                        "rows_input": int(len(df_multi_in)),
                        "rows_output": int(len(df_multi)),
                        "output": str(multi_survey_output),
                        "external_lc_dir": str(external_lc_dir),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Multi-survey feature output: {multi_survey_output}")
                    log(f"Step 13c completed in {time.perf_counter() - multi_started:.1f}s")

        except Exception as e:
            print(f"Error in multi-survey feature extraction step: {e}")
            if args.verbose:
                traceback.print_exc()

    # Step 14: Auto-import into review DB
    if run_downstream and has_post_filter_output:
        log("\n=== Step 14: Importing candidates into review DB ===")
        review_db_path = out_dir / "review" / "review.db"
        review_db_updated = False
        try:


            # Find best available results file
            _import_file = _first_existing_candidate_result(results_dir, include_extended=True)

            if _import_file is not None:
                conn = db_connect(review_db_path)
                df_import = load_passing_table(_import_file)
                if df_import.empty:
                    conn.close()
                    log(f"No passing candidates to import into {review_db_path}")
                else:
                    df_import = add_stv_identity(df_import)
                    assert_stv_product_schema(df_import, stage="review_import")
                    n_total, n_new = import_candidates(
                        conn,
                        df_import,
                        source_path=str(out_dir.resolve()),
                        characterize_before_import=False,
                        vet_before_import=False,
                    )
                    if sed_photometry_output.exists():
                        try:
                            from malca.review.sed import upsert_sed_rows

                            sed_rows_for_review = load_table(sed_photometry_output)
                            n_sed = upsert_sed_rows(conn, sed_rows_for_review)
                            log(f"Imported {n_sed} SED photometry rows into {review_db_path}")
                        except Exception as sed_exc:
                            log(f"Warning: SED photometry review import failed: {sed_exc}")
                    if sed_model_fits_output.exists() or sed_model_curves_output.exists():
                        try:
                            from malca.enrichment.sed_model import upsert_sed_model_results

                            sed_model_fits_for_review = (
                                load_table(sed_model_fits_output)
                                if sed_model_fits_output.exists()
                                else pd.DataFrame()
                            )
                            sed_model_curves_for_review = (
                                load_table(sed_model_curves_output)
                                if sed_model_curves_output.exists()
                                else pd.DataFrame()
                            )
                            n_model_fits, n_model_curves = upsert_sed_model_results(
                                conn,
                                sed_model_fits_for_review,
                                sed_model_curves_for_review,
                            )
                            log(
                                f"Imported {n_model_fits} SED model fit rows and "
                                f"{n_model_curves} curve rows into {review_db_path}"
                            )
                        except Exception as sed_model_exc:
                            log(f"Warning: SED model review import failed: {sed_model_exc}")
                    conn.close()
                    review_db_updated = True
                    log(f"Imported {n_new} new candidates ({n_total} total) into {review_db_path}")
            else:
                log("No results file found for review DB import, skipping")

        except Exception as e:
            print(f"Warning: review DB import failed: {e}")
            if args.verbose:

                traceback.print_exc()
        if review_db_updated:
            if args.review_sync_enabled:
                maybe_sync_review_bundle(
                    True,
                    review_db_path,
                    args.review_sync_dir,
                    hash_assets=bool(args.review_sync_hash_assets),
                    verbose=args.verbose,
                )
            else:
                log("Review Git bundle auto-sync disabled by --no-review-sync")

    if args.export_bundle_enabled:
        export_bundle_path = args.export_bundle if args.export_bundle is not None else out_dir / f"{out_dir.name}_bundle_{mag_bin_tag}.zip"
        log(f"\n=== Exporting bundle to {export_bundle_path} ===")
        try:
            if args.full_bundle:
                source_index_file, index_candidates = _resolve_asassn_index_path(out_dir, index_override=args.index_file)
                if source_index_file is None:
                    tried_paths = ", ".join(str(p) for p in index_candidates[:6])
                    if len(index_candidates) > 6:
                        tried_paths += ", ..."
                    if not tried_paths:
                        tried_paths = "(no candidate paths)"
                    raise FileNotFoundError(
                        "Required index file not found for bundle export. "
                        f"Tried: {tried_paths}. "
                        "Pass --index-file or place the index at input/asassn_index_*.parquet."
                    )

                if source_index_file.suffix.lower() != ".parquet":
                    raise ValueError(
                        f"--full-bundle requires a parquet ASAS-SN index file, got: {source_index_file}"
                    )

                bundle_assets_dir = out_dir / "bundle_assets"
                bundle_assets_dir.mkdir(parents=True, exist_ok=True)
                bundle_index_file = bundle_assets_dir / "asassn_index_full.parquet"
                if (not bundle_index_file.exists()) or (bundle_index_file.stat().st_size != source_index_file.stat().st_size):
                    log(f"Copying full index into bundle assets: {source_index_file} -> {bundle_index_file}")
                    shutil.copy2(source_index_file, bundle_index_file)

            bundled = export_bundle_zip(export_bundle_path, out_dir, include_all=args.full_bundle, mag_bin_tag=mag_bin_tag)
            log(f"Exported bundle to {export_bundle_path.expanduser()} with {len(bundled)} files")
        except Exception as e:
            print(f"Error creating export bundle: {e}")


if __name__ == "__main__":
    main()
