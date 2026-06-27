# This file was mechanically split from malca.review.app; preserve behavior when editing.
"""Dash-based keyboard-driven review app for MALCA candidates."""
import atexit
from contextlib import closing
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread, Timer
import argparse
import glob as globlib
import importlib
import json
import logging
import math
import multiprocessing
import os
import re
import sqlite3
import sys
import time
import traceback
import warnings
import webbrowser

from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL, MATCH, dash_table
from dash import DiskcacheManager
from flask import abort, request, send_from_directory
import dash
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from malca.config import GAIA_CHUNK_SIZE
from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.core.phase import phase_template, template_phase_lag
from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
)
from malca.external_lc_manifest import (
    index_external_lc_paths_from_manifest as shared_index_external_lc_paths_from_manifest,
)
from malca.config import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE, DEFAULT_OUTPUT_DIR
from malca.config import (
    JD_OFFSET, MJD_TO_JD, GAIA_TCB_EPOCH_JD, TESS_BTJD_OFFSET, KEPLER_BKJD_OFFSET,
    REVIEW_RESIDUAL_FRACTION,
)
from malca.review.diagnostic_plots import (
    build_atlas_range_figure,
    build_autocorr_memory_figure,
    build_publication_diagnostic_pdf,
    build_catalog_support_figure,
    build_classifier_plane_figure,
    build_cluster_astrometry_figure,
    build_cmd_figure,
    build_dip_repeatability_figure,
    build_gaia_epoch_figure,
    build_harmonic_quality_figure,
    build_ir_colorcolor_figure,
    build_kiel_figure,
    build_ltv_trend_figure,
    build_neowise_range_figure,
    build_neowise_trend_figure,
    build_periodicity_plane_figure,
    build_recurrence_regularity_figure,
    build_rpm_figure,
    build_score_balance_figure,
    build_shape_impulsiveness_figure,
    build_stetson_scatter_figure,
    build_teff_sed_alpha_figure,
    build_uv_optical_figure,
    build_variability_strength_figure,
    build_ztf_range_figure,
)
from malca.review.dustycult import (
    check_dustycult_available,
    control_defaults_for_candidate,
    load_dustycult_curve,
    load_dustycult_fits,
    normalize_controls,
    run_dustycult_fit,
)
from malca.review.dustycult_display import (
    build_dustycult_fit_figure,
    dustycult_fit_metadata_text,
    dustycult_geometry_rows,
    dustycult_posterior_rows,
    dustycult_status_card_rows,
    format_dustycult_float,
    select_dustycult_display_row,
)
from malca.review.phoebe_fit import (
    PHOEBE_MODEL_KINDS,
    check_phoebe_available,
    infer_period_days,
    load_phoebe_fits,
    parse_phoebe_json,
    run_phoebe_fit,
)
from malca.review.interactive_plot import (
    _baseline_config_from_run_params,
    build_interactive_lightcurve_figure,
    resolve_lightcurve_path,
    warm_caches_for_candidate,
    _load_cleaned_df,
    _compute_baseline_bands,
    _build_stat_rows,
    normalize_external_lc_dataframe,
)
from malca.review.lightcurve_publication import build_review_lightcurve_publication_pdf
from malca.review.lightcurve_sources import clear_external_lc_discovery_caches
from malca.review.keyboard import (
    HELP_TEXT,
    CLASS_KEY_MAP,
)
from malca.review.metadata import (
    extract_review_metadata_feature_rows,
    extract_review_metadata_grouped,
    build_external_lookup_links,
    has_known_catalog_evidence,
    markdown_literal_unit_label,
)
from malca.review.cutouts import (
    DEFAULT_CUTOUT_SURVEY_KEY,
    available_cutout_options,
    cutout_payload_for_candidate,
)
from malca.review.filter_schema import (
    SIDEBAR_GROUPS as REVIEW_FILTER_SIDEBAR_GROUPS,
    VETTING_KNOWN_BOOL_FILTERS,
    VETTING_KNOWN_SELECT_FILTERS,
    is_definite_known_type_value,
)
from malca.review.pipeline import detect_pipeline_status, detect_sed_model_status, detect_sed_photometry_status
from malca.review.pipeline import run_missing_stages
from malca.review.pipeline import update_candidate_payload
from malca.review.period_search import (
    has_external_period as shared_has_external_period,
    resolve_stored_review_period as shared_resolve_stored_review_period,
    run_harmonic_check_for_payload as shared_run_harmonic_check_for_payload,
    run_period_search_for_payload as shared_run_period_search_for_payload,
)
from malca.review.dustycult_visualization import build_dustycult_occulter_figure
from malca.review.publication import graph_config_without_image_export, publication_figure, render_publication_pdf, slugify_token
from malca.review.session import create_queue_data_dict
from malca.review.sync import auto_export_review_bundle
from malca.review.taxonomy import (
    MORPHOLOGY_PRIMARY,
    json_list,
    keyboard_payload,
    label_for,
    selection_from_review,
)
from malca.review.store import (
    DEFAULT_DB_PATH,
    DEFAULT_STANDALONE_DB_PATH,
    db_connect,
    count_progress,
    get_review,
    save_review,
    find_plot_image,
    find_phase_plot_image,
    get_candidate_payload,
    load_app_state,
    save_app_state,
    import_candidates,
    load_candidates_file,
    export_reviews,
    detect_run_directory_files,
    merge_review_databases,
    get_distinct_values,
    get_diagnostic_background,
    get_numeric_bounds,
)
from malca.review.store import import_lightcurve_files
from malca.review.sed import build_sed_figure, load_sed_rows, sed_source_statuses
from malca.review.eda_panel import (
    EDA_TABLE_COLUMNS,
    candidate_ids_from_eda_table_context,
    candidate_ids_from_plotly_selection,
    candidate_index_in_queue,
    eda_metric_options,
    eda_plot_row_counts,
    eda_publication_figure,
    eda_scatter_figure,
    eda_status_figure,
    eda_table_rows,
    load_review_eda_frame,
    queue_eda_frame,
    resolve_eda_metric_values,
    selected_candidate_from_queue,
    selected_candidate_row_style,
    selected_row_style,
)
from malca.enrichment.sed_model import load_sed_model_curves, load_sed_model_fits




# Suppress known multiprocessing/diskcache semaphore leak warning at worker shutdown
