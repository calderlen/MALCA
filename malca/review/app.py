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

from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL, MATCH
from dash import DiskcacheManager
from flask import abort, send_from_directory
import dash
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from malca.config.config_characterize import GAIA_CHUNK_SIZE
from malca.config.config_filters import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
)
from malca.config.config_paths import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE
from malca.config.config_pipeline import (
    JD_OFFSET, MJD_TO_JD, GAIA_TCB_EPOCH_JD, TESS_BTJD_OFFSET, KEPLER_BKJD_OFFSET,
    REVIEW_RESIDUAL_FRACTION,
)
from malca.review.diagnostic_plots import (
    build_atlas_range_figure,
    build_autocorr_memory_figure,
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
    build_uv_optical_figure,
    build_variability_strength_figure,
    build_ztf_range_figure,
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
from malca.review.keyboard import (
    HELP_TEXT,
    CLASS_KEY_MAP,
)
from malca.review.metadata import (
    extract_review_metadata_grouped,
    is_group_default_open,
    build_external_lookup_links,
)
from malca.review.filter_schema import (
    SIDEBAR_GROUPS as REVIEW_FILTER_SIDEBAR_GROUPS,
    VETTING_KNOWN_BOOL_FILTERS,
    VETTING_KNOWN_SELECT_FILTERS,
)
from malca.review.handoff import build_explorer_command, launch_detached
from malca.review.pipeline import detect_pipeline_status
from malca.review.pipeline import run_missing_stages
from malca.review.pipeline import update_candidate_payload
from malca.review.period_search import (
    has_external_period as shared_has_external_period,
    run_period_search_for_payload as shared_run_period_search_for_payload,
)
from malca.review.session import create_queue_data_dict
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
    merge_candidate_results,
    merge_vetting_results,
    get_distinct_values,
    get_diagnostic_background,
    get_numeric_bounds,
)
from malca.review.store import import_lightcurve_files




# Suppress known multiprocessing/diskcache semaphore leak warning at worker shutdown
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be.*leaked semaphore",
    module="multiprocessing.resource_tracker",
)

def _configure_background_start_methods() -> None:
    """Prefer spawn so background workers do not inherit the dev-server socket."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        multiprocess = importlib.import_module("multiprocess")
    except ModuleNotFoundError:
        return

    try:
        methods = set(multiprocess.get_all_start_methods())
    except Exception:
        methods = set()

    if "spawn" not in methods:
        return

    try:
        multiprocess.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


_configure_background_start_methods()



CLASS_BADGE_TAGS = list(CLASS_KEY_MAP.values())

class TrackingDiskcacheManager(DiskcacheManager):
    """Diskcache manager that can clean up outstanding worker processes on exit."""

    def __init__(self, cache=None, cache_by=None, expire=None):
        super().__init__(cache=cache, cache_by=cache_by, expire=expire)
        self._active_jobs: set[int] = set()

    def call_job_fn(self, key, job_fn, args, context):
        job = super().call_job_fn(key, job_fn, args, context)
        if job is not None:
            try:
                self._active_jobs.add(int(job))
            except Exception:
                pass
        return job

    def terminate_job(self, job):
        try:
            return super().terminate_job(job)
        finally:
            try:
                if job is not None:
                    self._active_jobs.discard(int(job))
            except Exception:
                pass

    def get_result(self, key, job):
        try:
            return super().get_result(key, job)
        finally:
            try:
                if job is not None:
                    self._active_jobs.discard(int(job))
            except Exception:
                pass

    def terminate_all_jobs(self) -> None:
        for job in tuple(sorted(self._active_jobs)):
            self.terminate_job(job)


# Background callback manager for long-running fetch/import (DiskCache for local dev)
_bc_cache = diskcache.Cache(Path(__file__).resolve().parents[2] / "output" / "review" / ".dash_cache")
_background_callback_manager = TrackingDiskcacheManager(_bc_cache)
_PRELOAD_DELAY_SEC = 0.4
_PRELOAD_LOOKAHEAD = 1
_preload_generation_lock = Lock()
_preload_generation = 0


def _next_preload_generation() -> int:
    """Return a new monotonic preload generation token."""
    global _preload_generation
    with _preload_generation_lock:
        _preload_generation += 1
        return _preload_generation


def _is_current_preload_generation(generation: int) -> bool:
    """Check whether a queued preload request is still current."""
    with _preload_generation_lock:
        return generation == _preload_generation


def _cleanup_background_resources(*_args) -> None:
    """Terminate outstanding background jobs so Ctrl-C fully releases the port."""
    try:
        if _background_callback_manager is not None:
            _background_callback_manager.terminate_all_jobs()
    except Exception:
        pass

    try:
        _bc_cache.close()
    except Exception:
        pass


atexit.register(_cleanup_background_resources)

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="MALCA Review",
    background_callback_manager=_background_callback_manager,
)

# Custom OLED black CSS
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>MALCA Review</title>
    {%metas%}
    {%css%}
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
        }
        body {
            background-color: #000;
            color: #e0e0e0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }
        .main-container {
            height: 100vh;
            display: flex;
            background-color: #000;
        }
        .sidebar {
            position: fixed;
            left: -280px;
            top: 0;
            width: 280px;
            height: 100vh;
            background-color: #0a0a0a;
            border-right: 1px solid #333;
            transition: left 0.2s ease;
            z-index: 1000;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 12px 14px;
            font-size: 11px;
            color: #bbb;
        }
        .sidebar.expanded {
            left: 0;
        }
        .sidebar .section-title {
            color: #0af;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin: 0 0 6px 0;
        }
        .sidebar hr {
            border: none;
            border-top: 1px solid #222;
            margin: 10px 0;
        }
        .sidebar label {
            display: block;
            color: #777;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .sidebar-field-label {
            color: #8ba4b8;
            font-size: 10px;
            line-height: 1.25;
            margin-bottom: 2px;
            letter-spacing: 0.15px;
        }
        .sidebar-field-label p {
            margin: 0;
        }
        .sidebar details {
            margin-bottom: 2px;
        }
        .sidebar details summary {
            color: #0af;
            font-size: 11px;
            cursor: pointer;
            padding: 3px 0;
            user-select: none;
        }
        .sidebar details summary:hover {
            color: #4cf;
        }
        .sidebar details[open] > summary {
            margin-bottom: 4px;
        }
        .sidebar-toggle {
            appearance: none;
            position: fixed;
            left: 0;
            top: 50px;
            width: 30px;
            height: 60px;
            background-color: #1a1a1a;
            border: 1px solid #555;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1001;
            color: #0af;
            font-size: 20px;
            transition: left 0.2s ease;
            padding: 0;
        }
        .sidebar-toggle.sidebar-expanded {
            left: 280px;
        }
        .sidebar-toggle:hover {
            background-color: #2a2a2a;
        }
        .content-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            min-width: 0;
            padding-left: 0;
            transition: padding-left 0.2s ease;
        }
        /* When sidebar is open, push main content so it doesn't get covered */
        .sidebar.expanded + .content-area {
            padding-left: 280px;
        }
        .workspace-panels {
            flex: 1;
            min-height: 0;
            min-width: 0;
            display: flex;
            overflow: hidden;
            padding: 8px 10px 10px 10px;
            gap: 0;
        }
        .left-info-panel {
            flex: 0 0 340px;
            width: 340px;
            min-width: 260px;
            max-width: 72vw;
            display: flex;
            flex-direction: column;
            gap: 8px;
            height: 100%;
            min-height: 0;
            overflow: hidden;
            padding-right: 8px;
        }
        .left-info-scroll {
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            overscroll-behavior: contain;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-right: 2px;
            padding-bottom: 12px;
        }
        .right-plot-panel {
            flex: 1;
            min-width: 0;
            min-height: 0;
        }
        .header-bar {
            background-color: #0a0a0a;
            border-bottom: 1px solid #555;
            padding: 6px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
            align-items: center;
            gap: 14px;
            font-size: 11px;
        }
        .header-key-info {
            flex: 1;
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-key-info .item {
            color: #8fb1c8;
            font-size: 10px;
            white-space: nowrap;
        }
        .header-key-info .item.path {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .plot-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            background-color: #000;
            overflow: hidden;
            min-height: 260px;
            padding: 0 2px 0 8px;
            gap: 8px;
        }
        .panel-splitter-vertical {
            position: relative;
            width: 12px;
            flex: 0 0 12px;
            height: auto;
            margin: 0 2px;
            cursor: col-resize;
            user-select: none;
            touch-action: none;
        }
        .panel-splitter-vertical::before {
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 1px;
            height: auto;
            transform: translateX(-50%);
            background: rgba(126, 150, 166, 0.45);
        }
        .panel-splitter-vertical::after {
            content: '::';
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            letter-spacing: 1px;
            padding: 6px 3px;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            border-radius: 999px;
            color: #8db0c8;
            font-size: 10px;
            background: rgba(8, 18, 25, 0.9);
            border: 1px solid rgba(86, 114, 132, 0.55);
            line-height: 1;
        }
        .panel-splitter-vertical:hover::after,
        .panel-splitter-vertical.dragging::after {
            color: #b5d4ea;
            border-color: rgba(133, 171, 196, 0.9);
            background: rgba(12, 26, 35, 0.96);
        }
        .plot-toolbar {
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
            padding: 8px 10px;
            border: 1px solid rgba(84, 118, 140, 0.35);
            background: linear-gradient(180deg, rgba(8, 18, 24, 0.9), rgba(3, 8, 12, 0.75));
            border-radius: 8px;
            font-size: 11px;
        }
        .compact-btn {
            background-color: #14212b;
            color: #c6d7e8;
            border: 1px solid rgba(92, 129, 154, 0.6);
            border-radius: 5px;
            padding: 2px 7px;
            font-size: 10px;
            cursor: pointer;
        }
        .compact-btn:hover {
            border-color: #7da8c4;
            background-color: #1a2b38;
        }
        .meta-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 6px 10px;
            border: 1px solid rgba(84, 118, 140, 0.25);
            border-radius: 8px;
            background: rgba(6, 14, 20, 0.7);
        }
        .meta-toolbar .title {
            color: #8fb1c8;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        .plot-toolbar .label-chip {
            color: #85a7bf;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .plot-toolbar .dash-checklist label,
        .plot-toolbar label {
            color: #c9d4df !important;
            margin-right: 8px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .plot-toolbar .dash-checklist label,
        .plot-toolbar .dash-radioitems label {
            padding: 4px 9px;
            border-radius: 4px;
            border: 1px solid rgba(60, 92, 112, 0.55);
            background: rgba(7, 16, 22, 0.9);
        }
        .sidebar .dash-checklist label,
        .sidebar .dash-radioitems label {
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid rgba(60, 92, 112, 0.55);
            background: rgba(7, 16, 22, 0.9);
            margin-bottom: 4px;
        }
        .toolbar-slider-control {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            min-width: 140px;
            max-width: 200px;
        }
        .meta-field-row {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            padding: 2px 0;
            border-bottom: 1px solid #1a1a1a;
        }
        .meta-field-label {
            color: #7fa3bc;
            flex-shrink: 0;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-field-label {
            text-transform: none;
            letter-spacing: 0.2px;
        }
        .meta-field-value {
            color: #e2edf6;
            text-align: right;
            font-weight: 600;
            word-break: break-word;
            white-space: normal;
        }
        .meta-field-label p,
        .meta-field-value p {
            margin: 0;
        }
        .vetting-banner-empty {
            padding: 6px 12px;
            margin: 4px 0;
            border-radius: 4px;
            background: #333;
            color: #999;
            font-size: 0.85em;
            text-align: center;
        }
        .vetting-banner-shell {
            margin: 4px 0;
        }
        .vetting-banner-header {
            padding: 3px 8px;
            border-radius: 4px 4px 0 0;
            font-weight: bold;
            font-size: 11px;
            text-align: center;
        }
        .vetting-banner-header.known {
            background: #4a1111;
            color: #ff6b6b;
            border: 1px solid #ff6b6b;
            border-bottom: none;
        }
        .vetting-banner-header.new {
            background: #114a11;
            color: #6bff6b;
            border: 1px solid #6bff6b;
            border-bottom: none;
        }
        .vetting-banner-grid {
            display: flex;
            flex-direction: column;
            gap: 2px;
            padding: 5px 6px 6px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-top: none;
            border-radius: 0 0 4px 4px;
        }
        .vetting-banner-grid.with-links {
            border-radius: 0;
        }
        .vetting-banner-cell {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            border: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            overflow: hidden;
        }
        .vetting-banner-shell.known .vetting-banner-cell {
            background: #2a1a1a;
        }
        .vetting-banner-shell.new .vetting-banner-cell {
            background: #1a2a1a;
        }
        .vetting-banner-label {
            color: #888;
            font-size: 11px;
            flex-shrink: 0;
        }
        .vetting-banner-value {
            color: #e0e0e0;
            font-weight: bold;
            text-align: right;
            word-break: break-word;
            white-space: normal;
        }
        .vetting-banner-hit.known {
            color: #ff6b6b;
        }
        .vetting-banner-hit.new {
            color: #6bff6b;
        }
        .vetting-banner-links {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            padding: 6px;
            background: #161616;
            border: 1px solid #333;
            border-top: none;
            border-radius: 0 0 4px 4px;
        }
        .vetting-banner-link {
            display: inline-block;
            padding: 2px 6px;
            background: #222;
            border: 1px solid #444;
            border-radius: 3px;
            color: #8af;
            text-decoration: none;
            font-size: 10px;
            white-space: nowrap;
        }
        .vetting-banner-link:hover {
            text-decoration: none;
            border-color: #6a8ca6;
        }
        /* Blue slider theming — global override for all Dash sliders in toolbar */
        .plot-toolbar .rc-slider-rail {
            background-color: #284256 !important;
        }
        .plot-toolbar .rc-slider-track {
            background-color: #0af !important;
        }
        .plot-toolbar .rc-slider-handle {
            border-color: #0af !important;
            background-color: #0b141d !important;
            box-shadow: 0 0 0 3px rgba(0, 170, 255, 0.2) !important;
            outline: none !important;
        }
        .plot-toolbar .rc-slider-handle:hover,
        .plot-toolbar .rc-slider-handle:focus,
        .plot-toolbar .rc-slider-handle:active,
        .plot-toolbar .rc-slider-handle-dragging {
            border-color: #0af !important;
            background-color: #0b141d !important;
            box-shadow: 0 0 0 5px rgba(0, 170, 255, 0.3) !important;
        }
        .plot-toolbar .rc-slider-dot-active {
            border-color: #0af !important;
        }
        .plot-toolbar .rc-slider-tooltip-inner {
            background-color: #0af !important;
            border: 1px solid #0af !important;
            color: #fff !important;
        }
        .plot-toolbar .rc-slider-tooltip-arrow {
            border-top-color: #0af !important;
            border-bottom-color: #0af !important;
        }
        .plot-frame {
            flex: 1;
            min-height: 260px;
            border: 1px solid rgba(84, 118, 140, 0.35);
            border-radius: 10px;
            background: radial-gradient(circle at 20% 0%, rgba(17, 39, 54, 0.22), rgba(0, 0, 0, 0.05) 45%, rgba(0, 0, 0, 0));
            overflow: hidden;
            position: relative;
        }
        .plot-native {
            width: 100%;
            height: 100%;
        }
        .plot-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .plot-stats {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 2px;
        }
        .plot-status {
            border: 1px solid rgba(102, 126, 143, 0.45);
            border-radius: 8px;
            padding: 4px 8px;
            background: rgba(9, 18, 25, 0.82);
            color: #d4dfeb;
            font-size: 10px;
            line-height: 1.2;
            overflow: hidden;
        }
        .plot-status .status-line {
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .plot-status details {
            margin-top: 2px;
        }
        .plot-status summary {
            cursor: pointer;
            color: #8fb1c8;
            font-size: 10px;
            user-select: none;
        }
        .plot-status ul {
            margin: 3px 0 0 14px;
            padding: 0;
        }
        .plot-status li {
            margin: 1px 0;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .plot-status.warn {
            border-color: rgba(186, 144, 44, 0.7);
            background: rgba(41, 29, 6, 0.62);
        }
        .plot-status.error {
            border-color: rgba(192, 72, 72, 0.78);
            background: rgba(48, 12, 12, 0.58);
        }
        .camera-diag {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            font-size: 10px;
            color: #b9cad9;
        }
        .camera-diag .item {
            border: 1px solid rgba(90, 118, 138, 0.55);
            border-radius: 999px;
            padding: 2px 8px;
            background: rgba(11, 23, 31, 0.7);
        }
        .queue-provenance-panel {
            border: 1px solid rgba(77, 106, 127, 0.5);
            border-radius: 8px;
            padding: 6px 8px;
            background: rgba(9, 18, 25, 0.82);
            margin-top: 6px;
        }
        .queue-provenance-list {
            margin: 6px 0 0 16px;
            padding: 0;
            color: #cad9e5;
            font-size: 10px;
            line-height: 1.35;
        }
        .queue-provenance-list li {
            margin: 3px 0;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .queue-provenance-note {
            margin-top: 6px;
            color: #7d91a6;
            font-size: 10px;
            line-height: 1.3;
        }
        .sidebar-camera-actions {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-bottom: 4px;
        }
        .sidebar-camera-actions .action-btn {
            margin-right: 0;
            padding: 1px 6px;
            font-size: 10px;
        }
        .sidebar-camera-checklist label {
            color: #cad9e5 !important;
            margin-right: 8px;
            margin-bottom: 2px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .stat-card {
            padding: 4px 8px;
            border-radius: 5px;
            border: 1px solid rgba(64, 96, 116, 0.45);
            background-color: rgba(8, 17, 24, 0.75);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        .stat-card .label {
            color: #7fa3bc;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-card .value {
            color: #e2edf6;
            font-size: 12px;
            font-weight: 600;
            text-align: right;
        }
        .run-config-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
        }
        .run-config-item {
            border: 1px solid rgba(78, 110, 132, 0.45);
            border-radius: 6px;
            background: rgba(7, 16, 22, 0.68);
            padding: 6px 8px;
            min-height: 52px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .run-config-item.wide {
            grid-column: 1 / -1;
            min-height: 0;
        }
        .run-config-item .k {
            color: #7fa3bc;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .run-config-item .v {
            color: #dce8f2;
            font-size: 11px;
            font-weight: 600;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .run-config-item.wide .v {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            word-break: break-word;
        }
        .run-config-item.warning {
            border-color: rgba(186, 144, 44, 0.75);
            background: rgba(44, 30, 8, 0.2);
        }
        .repro-badge {
            border-radius: 999px;
            border: 1px solid rgba(99, 129, 153, 0.6);
            padding: 2px 8px;
            font-size: 10px;
            color: #9bc1dc;
            background: rgba(12, 25, 33, 0.8);
        }
        .repro-badge.warn {
            border-color: rgba(186, 144, 44, 0.75);
            color: #e4c16d;
            background: rgba(44, 30, 8, 0.62);
        }
        .metadata-bar {
            background-color: #0a0a0a;
            border-top: 1px solid #333;
            padding: 10px 20px;
            font-size: 11px;
            color: #aaa;
            max-height: 80px;
            overflow-y: auto;
        }
        .control-bar {
            background-color: #0a0a0a;
            padding: 6px 15px;
            border-top: 1px solid #555;
            font-size: 11px;
        }
        .review-form {
            background-color: #0a0a0a;
            padding: 4px 12px;
            border-top: 1px solid #555;
            overflow-y: auto;
            font-size: 11px;
        }
        .score-btn {
            background-color: #1a1a1a;
            border: 1px solid #444;
            color: #fff;
            padding: 2px 7px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.1s;
            margin-right: 4px;
        }
        .score-btn:hover {
            border-color: #666;
            background-color: #2a2a2a;
        }
        .score-btn.active {
            border-color: #0af;
            background-color: #003366;
        }
        .badge-btn {
            background-color: transparent;
            border: 1px solid #444;
            color: #888;
            padding: 3px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
            margin-right: 5px;
            transition: all 0.1s;
        }
        .badge-btn:hover {
            border-color: #666;
            color: #bbb;
            background-color: #1a1a1a;
        }
        .badge-btn.active {
            border-color: #0f0;
            color: #0f0;
            background-color: #003300;
        }
        .action-btn {
            background-color: #1a1a1a;
            color: #0af;
            border: 1px solid #0af;
            padding: 2px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
            margin-right: 6px;
        }
        .action-btn.primary {
            background-color: #003366;
        }
        .queue-refresh-btn {
            font-size: 12px !important;
            font-weight: 600;
            min-height: 34px;
            padding: 6px 10px;
        }
        .notification {
            color: #7fd6a8;
            font-size: 11px;
            flex: 1 1 280px;
            min-width: 0;
            max-width: 100%;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            opacity: 0.95;
        }
        .help-link {
            color: #0af;
            cursor: pointer;
            text-decoration: none;
            font-size: 11px;
        }
        .help-link:hover {
            text-decoration: underline;
        }
        .Select-control {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
            min-height: 26px !important;
            height: 26px !important;
            box-shadow: none !important;
        }
        .Select-menu-outer {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
        }
        input[type="text"],
        input[type="number"],
        input[type="search"],
        input[type="email"],
        input[type="password"],
        textarea {
            background-color: #1a1a1a !important;
            border: 1px solid #555 !important;
            color: #e0e0e0 !important;
        }
        /* Dropdown styling */
        .Select-control, .Select-menu-outer, .Select-menu, .Select-option {
            background-color: #0f1418 !important;
            color: #d5e3ef !important;
            border-color: #2f4658 !important;
        }
        .Select-menu-outer {
            overscroll-behavior: contain;
            -webkit-overflow-scrolling: touch;
            /* Fix scroll feel */
            scrollbar-color: #2f4658 #0f1418;
            scrollbar-width: thin;
        }
        /* Custom scrollbar for Webkit */
        .Select-menu-outer::-webkit-scrollbar {
            width: 8px;
            background-color: #0f1418;
        }
        .Select-menu-outer::-webkit-scrollbar-thumb {
            background-color: #2f4658;
            border-radius: 4px;
        }
        .Select-menu-outer::-webkit-scrollbar-thumb:hover {
            background-color: #0af;
        }
        /* NUCLEAR OPTION: Force text color on EVERYTHING inside the dropdown menu */
        body .Select-menu-outer,
        body .Select-menu-outer *,
        body .Select-menu-outer div,
        body .Select-menu-outer span,
        body .Select-menu-outer label,
        body .Select-menu-outer a,
        body .Select-menu-outer button,
        body .Select-menu-outer strong,
        body .Select-menu-outer b,
        body .Select-menu-outer i,
        body .Select-menu-outer em,
        body .Select-menu-outer small,
        body .Select-option,
        body .VirtualizedSelectOption,
        body .Select * {
            color: #dce8f2 !important;
            opacity: 1 !important;
        }

        /* Force dividers (borders) between options */
        body .Select-option,
        body .VirtualizedSelectOption,
        body [class*="Select-option"] {
            border-bottom: 1px solid #2f4658 !important;
            padding: 8px 10px !important; /* Ensure enough space for divider to be seen */
        }
        
        /* Last child should not have a border usually, but for clarity let's keep it or remove it */
        body .Select-option:last-child,
        body .VirtualizedSelectOption:last-child {
            border-bottom: none !important;
        }

        /* Ensure hover state stays readable and distinct */
        body .Select-option.is-focused,
        body .VirtualizedSelectOption.is-focused,
        body .Select-option:hover,
        body .VirtualizedSelectOption:hover {
            background-color: #1d2d3a !important;
            color: #ffffff !important;
            cursor: pointer !important;
        }

        /* Specific fix for VirtualizedSelectOption "Select All" helper text container */
        .VirtualizedSelectOption {
             display: flex !important;
             align-items: center !important;
        }

        /* Force any SVG icons (checkmarks) to be visible */
        .Select-menu-outer svg,
        .Select-menu-outer path {
            fill: #dce8f2 !important;
            stroke: #dce8f2 !important;
        }

        /* Force high specificity on the option text itself */
        .VirtualizedSelectOption, .Select-option {
            color: #dce8f2 !important;
            text-shadow: none !important;
        }
        
        /* Pseudo-elements just in case */
        body .Select-menu-outer *::before,
        body .Select-menu-outer *::after,
        body .Select *::before,
        body .Select *::after {
            color: #dce8f2 !important;
        }
        .Select-placeholder, .Select-value-label {
            color: #9db4c7 !important;
            font-size: 10px !important;
            line-height: 24px !important;
        }
        .Select-input {
            height: 24px !important;
        }
        .Select-arrow-zone {
            padding-right: 5px !important;
        }
        .Select-arrow {
            border-top-color: #9db4c7 !important;
            border-left-color: transparent !important;
            border-right-color: transparent !important;
            opacity: 1 !important;
        }
        .is-open > .Select-control .Select-arrow {
            border-top-color: transparent !important;
            border-bottom-color: #b8cede !important;
        }
        /* Checkbox and label styling */
        label, .form-label {
            color: #aaa !important;
        }
        /* DASH/REACT-SELECT COMPONENT OVERRIDES - THE FINAL HAMMER */
        
        /* 1. The Menu Container */
        .Select-menu-outer {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
        }

        /* 2. The Options (including Select All) */
        .VirtualizedSelectOption, 
        .Select-option,
        .dash-dropdown-option,
        div[role="option"] {
            background-color: #10171d !important;
            color: #dce8f2 !important;
            border-bottom: 1px solid #2f4658 !important; /* Visible divider */
            opacity: 1 !important;
        }

        /* 3. Hover/Focused State */
        .VirtualizedSelectOption.is-focused,
        .Select-option.is-focused,
        .dash-dropdown-option:hover,
        .VirtualizedSelectOption:hover {
            background-color: #1d2d3a !important;
            color: #ffffff !important;
            cursor: pointer !important;
        }

        /* 4. "Select All" / "Deselect All" often lives in a special header or div at the top */
        .Select-menu-outer > div:first-child,
        .Select-menu-outer > div:nth-child(1) {
             color: #dce8f2 !important;
        }
        
        /* 5. Force text color on children (labels, spans) inside options */
        .VirtualizedSelectOption *,
        .Select-option * {
            color: inherit !important;
        }
        .dash-dropdown {
            background-color: #0f1418 !important;
            color: #dce8f2 !important;
            border: 1px solid #2f4658 !important;
            border-radius: 4px !important;
            min-height: 24px !important;
            height: 24px !important;
            padding: 0 6px !important;
            box-shadow: none !important;
        }
        .dash-dropdown-trigger {
            min-height: 24px !important;
            height: 24px !important;
            background-color: #0f1418 !important;
        }
        .dash-dropdown-value,
        .dash-dropdown-value-item {
            color: #dce8f2 !important;
            font-size: 10px !important;
            line-height: 20px !important;
        }
        .dash-dropdown-content,
        .dash-dropdown-options,
        .dash-options-list,
        .dash-dropdown-search-container {
            background-color: #0f1418 !important;
            border: 1px solid #2f4658 !important;
            color: #dce8f2 !important;
        }
        .dash-dropdown-search {
            background-color: #0f1418 !important;
            color: #dce8f2 !important;
            border: 1px solid #2f4658 !important;
            font-size: 10px !important;
            min-height: 22px !important;
        }
        .dash-dropdown-option,
        .dash-options-list-option {
            background-color: #10171d !important;
            color: #dce8f2 !important;
            font-size: 10px !important;
            padding: 3px 8px !important;
            min-height: 22px !important;
            border-bottom: 1px solid #2f4658 !important;
        }
        .dash-dropdown-option:hover,
        .dash-options-list-option:hover,
        .dash-dropdown-option.selected,
        .dash-options-list-option.selected {
            background-color: #1d2d3a !important;
            color: #fff !important;
        }
        /* Checklist items */
        .dash-checklist label {
            color: #e0e0e0 !important;
        }
        .dash-checklist input[type="checkbox"] {
            accent-color: #0af;
        }
        .dash-radioitems input[type="radio"],
        input[type="radio"] {
            accent-color: #0af;
        }
        /* Input placeholders */
        ::placeholder {
            color: #666 !important;
            opacity: 1;
        }
        :-ms-input-placeholder {
            color: #666 !important;
        }
        ::-ms-input-placeholder {
            color: #666 !important;
        }
        /* Collapsible metadata sections */
        .metadata-sections {
            background-color: #0a0a0a;
            border-top: 2px solid #555;
            overflow: visible;
            flex-shrink: 0;
            padding: 0 12px;
            border-radius: 8px;
        }
        .candidate-metadata {
            flex: 1;
            min-height: 100px;
            height: auto;
            max-height: none;
        }
        @media (min-width: 1600px) {
            .candidate-metadata {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                align-items: start;
            }
        }
        .metadata-health {
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(83, 113, 133, 0.5);
            border-radius: 8px;
            padding: 5px 8px;
            margin: 5px 0 6px 0;
            background: rgba(8, 16, 23, 0.72);
            font-size: 10px;
            line-height: 1.3;
        }
        .metadata-health .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(99, 129, 153, 0.6);
            padding: 1px 7px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            white-space: nowrap;
            font-weight: 600;
            color: #9bc1dc;
            background: rgba(12, 25, 33, 0.82);
        }
        .metadata-health .detail {
            color: #adbfce;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .bottom-context-bar {
            display: flex;
            align-items: center;
            gap: 22px;
            flex-wrap: wrap;
            min-width: 0;
        }
        .bottom-context-item {
            display: flex;
            align-items: baseline;
            gap: 8px;
            min-width: 0;
            flex: 1 1 320px;
        }
        .bottom-context-k {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            white-space: nowrap;
            color: #86a7bd;
        }
        .bottom-context-v {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 10px;
            color: #c5d5e1;
        }
        .metadata-health.metadata-health-base .chip {
            color: #e4c16d;
            border-color: rgba(186, 144, 44, 0.75);
            background: rgba(44, 30, 8, 0.62);
        }
        .metadata-health.metadata-health-partial .chip {
            color: #8dc6de;
            border-color: rgba(96, 146, 174, 0.7);
            background: rgba(10, 31, 44, 0.64);
        }
        .metadata-health.metadata-health-enriched .chip {
            color: #9fd4b7;
            border-color: rgba(72, 148, 112, 0.72);
            background: rgba(10, 35, 23, 0.62);
        }
        .metadata-sections details {
            border-bottom: 1px solid #222;
        }
        .metadata-sections summary {
            cursor: pointer;
            padding: 5px 6px;
            color: #0af;
            font-size: 11px;
            font-weight: bold;
            user-select: none;
        }
        .metadata-sections summary:hover {
            color: #4cf;
        }
        .metadata-sections .meta-grid {
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 2px 6px 6px 6px;
            font-size: 11px;
        }

        /* Theme overrides */
        body[data-theme="black"] {
            background-color: #000 !important;
            color: #e0e0e0 !important;
        }
        body[data-theme="gray"] {
            background-color: #2e3440 !important;
            color: #d8dee9 !important;
        }
        body[data-theme="white"] {
            background-color: #eef2f6 !important;
            color: #1c2733 !important;
        }
        body[data-theme="black"] .main-container,
        body[data-theme="gray"] .main-container,
        body[data-theme="white"] .main-container {
            background-color: inherit !important;
        }
        body[data-theme="black"] .sidebar,
        body[data-theme="black"] .header-bar,
        body[data-theme="black"] .metadata-sections,
        body[data-theme="black"] .control-bar,
        body[data-theme="black"] .review-form,
        body[data-theme="black"] .plot-status,
        body[data-theme="black"] .run-config-item,
        body[data-theme="black"] .plot-toolbar { background-color: #0a0a0a !important; border-color: #555 !important; color: #e0e0e0 !important; }
        body[data-theme="gray"] .sidebar,
        body[data-theme="gray"] .header-bar,
        body[data-theme="gray"] .metadata-sections,
        body[data-theme="gray"] .control-bar,
        body[data-theme="gray"] .review-form,
        body[data-theme="gray"] .plot-status,
        body[data-theme="gray"] .run-config-item,
        body[data-theme="gray"] .plot-toolbar { background-color: #3b4252 !important; border-color: #4c566a !important; color: #d8dee9 !important; }
        body[data-theme="white"] .sidebar,
        body[data-theme="white"] .header-bar,
        body[data-theme="white"] .metadata-sections,
        body[data-theme="white"] .control-bar,
        body[data-theme="white"] .review-form,
        body[data-theme="white"] .plot-status,
        body[data-theme="white"] .run-config-item,
        body[data-theme="white"] .plot-toolbar { background-color: #ffffff !important; border-color: #c5d0da !important; color: #1c2733 !important; }
        body[data-theme="black"] .section-title,
        body[data-theme="black"] .help-link,
        body[data-theme="black"] .metadata-sections summary,
        body[data-theme="black"] #progress-text { color: #0af !important; }
        body[data-theme="gray"] .section-title,
        body[data-theme="gray"] .help-link,
        body[data-theme="gray"] .metadata-sections summary,
        body[data-theme="gray"] #progress-text { color: #88c0d0 !important; }
        body[data-theme="white"] .section-title,
        body[data-theme="white"] .help-link,
        body[data-theme="white"] .metadata-sections summary,
        body[data-theme="white"] #progress-text { color: #245f8f !important; }
        body[data-theme="black"] .action-btn.primary { background-color: #0af !important; color: #08131d !important; border-color: #0af !important; }
        body[data-theme="gray"] .action-btn.primary { background-color: #88c0d0 !important; color: #2e3440 !important; border-color: #88c0d0 !important; }
        body[data-theme="white"] .action-btn.primary { background-color: #245f8f !important; color: #f5f7fa !important; border-color: #245f8f !important; }
        body[data-theme="black"] input, body[data-theme="black"] textarea, body[data-theme="black"] select,
        body[data-theme="black"] .dash-dropdown .Select-control,
        body[data-theme="black"] .dash-dropdown .Select-menu-outer { background-color: #0a0a0a !important; color: #e0e0e0 !important; border-color: #555 !important; }
        body[data-theme="gray"] input, body[data-theme="gray"] textarea, body[data-theme="gray"] select,
        body[data-theme="gray"] .dash-dropdown .Select-control,
        body[data-theme="gray"] .dash-dropdown .Select-menu-outer { background-color: #3b4252 !important; color: #eceff4 !important; border-color: #4c566a !important; }
        body[data-theme="white"] input, body[data-theme="white"] textarea, body[data-theme="white"] select,
        body[data-theme="white"] .dash-dropdown .Select-control,
        body[data-theme="white"] .dash-dropdown .Select-menu-outer { background-color: #ffffff !important; color: #1c2733 !important; border-color: #c5d0da !important; }
        body[data-theme="white"] .plot-container,
        body[data-theme="white"] .metadata-bar,
        body[data-theme="white"] #bottom-context-info {
            background-color: #eef2f6 !important;
            border-color: #c5d0da !important;
            color: #4f6273 !important;
        }
        body[data-theme="white"] .plot-toolbar,
        body[data-theme="white"] .meta-toolbar,
        body[data-theme="white"] .camera-diag .item,
        body[data-theme="white"] .run-config-item,
        body[data-theme="white"] .repro-badge,
        body[data-theme="white"] .metadata-health,
        body[data-theme="white"] .compact-btn,
        body[data-theme="white"] .score-btn,
        body[data-theme="white"] .badge-btn,
        body[data-theme="white"] .action-btn:not(.primary),
        body[data-theme="white"] .sidebar-toggle,
        body[data-theme="white"] #help-modal .modal-content {
            background: #ffffff !important;
            background-image: none !important;
            border-color: #c5d0da !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .sidebar-toggle {
            color: #245f8f !important;
        }
        body[data-theme="white"] .sidebar-toggle:hover,
        body[data-theme="white"] .compact-btn:hover,
        body[data-theme="white"] .score-btn:hover,
        body[data-theme="white"] .badge-btn:hover,
        body[data-theme="white"] .action-btn:not(.primary):hover {
            background: #e7edf3 !important;
            border-color: #9fb1bf !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .score-btn.active {
            background: #dbe7f1 !important;
            border-color: #245f8f !important;
            color: #163b57 !important;
        }
        body[data-theme="white"] .badge-btn.active {
            background: #e8f7ec !important;
            border-color: #2f7a57 !important;
            color: #2f7a57 !important;
        }
        body[data-theme="white"] .header-key-info .item,
        body[data-theme="white"] .notification,
        body[data-theme="white"] .camera-diag,
        body[data-theme="white"] .run-config-item .k,
        body[data-theme="white"] .plot-toolbar .label-chip,
        body[data-theme="white"] .plot-toolbar .dash-checklist label,
        body[data-theme="white"] .plot-toolbar label,
        body[data-theme="white"] .meta-toolbar .title,
        body[data-theme="white"] .metadata-health .detail,
        body[data-theme="white"] .plot-status summary,
        body[data-theme="white"] .sidebar label,
        body[data-theme="white"] .sidebar details summary,
        body[data-theme="white"] .dash-checklist label,
        body[data-theme="white"] .sidebar-camera-checklist label,
        body[data-theme="white"] #review-progress-indicator,
        body[data-theme="white"] #pdm-result-label,
        body[data-theme="white"] #bottom-pipeline-status,
        body[data-theme="white"] #pass-indicator,
        body[data-theme="white"] #status-indicator {
            color: #4f6273 !important;
        }
        body[data-theme="white"] .sidebar details summary:hover,
        body[data-theme="white"] .metadata-sections summary:hover,
        body[data-theme="white"] .plot-status summary:hover,
        body[data-theme="white"] .help-link:hover {
            color: #245f8f !important;
        }
        body[data-theme="white"] .run-config-item .v,
        body[data-theme="white"] .plot-status,
        body[data-theme="white"] .plot-status .status-line,
        body[data-theme="white"] .plot-status li,
        body[data-theme="white"] .metadata-health,
        body[data-theme="white"] .bottom-context-v,
        body[data-theme="white"] .meta-field-label,
        body[data-theme="white"] .meta-field-value,
        body[data-theme="white"] .vetting-banner-label,
        body[data-theme="white"] .vetting-banner-value,
        body[data-theme="white"] .vetting-banner-empty,
        body[data-theme="white"] #help-modal .modal-body,
        body[data-theme="white"] #help-modal .modal-footer,
        body[data-theme="white"] #help-modal pre {
            color: #1c2733 !important;
        }
        body[data-theme="white"] .bottom-context-k {
            color: #5f7384 !important;
        }
        body[data-theme="white"] .meta-field-row {
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] .plot-status.warn {
            border-color: rgba(186, 144, 44, 0.45) !important;
            background: rgba(255, 239, 202, 0.88) !important;
        }
        body[data-theme="white"] .plot-status.error {
            border-color: rgba(192, 72, 72, 0.45) !important;
            background: rgba(255, 226, 226, 0.92) !important;
        }
        body[data-theme="white"] .metadata-health .chip,
        body[data-theme="white"] .repro-badge {
            color: #245f8f !important;
            background: #edf4fa !important;
            border-color: #b9c9d7 !important;
        }
        body[data-theme="white"] .metadata-health.metadata-health-base .chip {
            color: #946200 !important;
            border-color: #e0c27b !important;
            background: #fff3d8 !important;
        }
        body[data-theme="white"] .metadata-health.metadata-health-partial .chip {
            color: #1f6485 !important;
            border-color: #a7cad9 !important;
            background: #e8f5fb !important;
        }
        body[data-theme="white"] .metadata-health.metadata-health-enriched .chip {
            color: #2f7a57 !important;
            border-color: #a8d0ba !important;
            background: #e9f7ef !important;
        }
        body[data-theme="white"] .repro-badge.warn {
            color: #946200 !important;
            border-color: #e0c27b !important;
            background: #fff3d8 !important;
        }
        body[data-theme="white"] .sidebar hr,
        body[data-theme="white"] .metadata-sections details {
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] .dash-checklist label,
        body[data-theme="white"] .dash-radioitems label {
            box-shadow: none !important;
        }
        body[data-theme="white"] .panel-splitter-vertical::after {
            color: #4f6273 !important;
            background: rgba(255, 255, 255, 0.96) !important;
            border-color: rgba(159, 177, 191, 0.75) !important;
        }
        body[data-theme="white"] .panel-splitter-vertical::before {
            background: rgba(159, 177, 191, 0.6) !important;
        }
        body[data-theme="white"] .Select-control,
        body[data-theme="white"] .Select-menu-outer,
        body[data-theme="white"] .Select-menu,
        body[data-theme="white"] .Select-option,
        body[data-theme="white"] .VirtualizedSelectOption,
        body[data-theme="white"] .Select-placeholder,
        body[data-theme="white"] .Select-value,
        body[data-theme="white"] .Select-value-label,
        body[data-theme="white"] .Select-input,
        body[data-theme="white"] .Select-clear-zone,
        body[data-theme="white"] .Select-arrow-zone,
        body[data-theme="white"] .Select-menu-outer,
        body[data-theme="white"] .Select-menu-outer *,
        body[data-theme="white"] .Select * {
            background-color: #ffffff !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .Select-option,
        body[data-theme="white"] .VirtualizedSelectOption,
        body[data-theme="white"] [class*="Select-option"] {
            border-bottom: 1px solid #dde6ee !important;
        }
        body[data-theme="white"] .form-select,
        body[data-theme="white"] .form-control,
        body[data-theme="white"] .sidebar .form-select,
        body[data-theme="white"] .sidebar .form-control,
        body[data-theme="white"] .plot-toolbar .form-select,
        body[data-theme="white"] .plot-toolbar .form-control {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
            box-shadow: none !important;
        }
        body[data-theme="white"] .form-select:focus,
        body[data-theme="white"] .form-control:focus,
        body[data-theme="white"] .sidebar .form-select:focus,
        body[data-theme="white"] .sidebar .form-control:focus,
        body[data-theme="white"] .plot-toolbar .form-select:focus,
        body[data-theme="white"] .plot-toolbar .form-control:focus {
            border-color: #7da8c4 !important;
            box-shadow: 0 0 0 2px rgba(36, 95, 143, 0.12) !important;
        }
        body[data-theme="white"] .dash-dropdown,
        body[data-theme="white"] .dash-dropdown-trigger,
        body[data-theme="white"] .dash-dropdown-value,
        body[data-theme="white"] .dash-dropdown-value-item,
        body[data-theme="white"] .dash-dropdown-content,
        body[data-theme="white"] .dash-dropdown-options,
        body[data-theme="white"] .dash-options-list,
        body[data-theme="white"] .dash-dropdown-search-container,
        body[data-theme="white"] .dash-dropdown-search {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .dash-dropdown-option,
        body[data-theme="white"] .dash-options-list-option {
            background-color: #ffffff !important;
            color: #1c2733 !important;
            border-color: #dde6ee !important;
        }
        body[data-theme="white"] .dash-dropdown-option:hover,
        body[data-theme="white"] .dash-options-list-option:hover,
        body[data-theme="white"] .dash-dropdown-option.selected,
        body[data-theme="white"] .dash-options-list-option.selected {
            background-color: #eaf1f6 !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select-control .Select-input > input {
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select.has-value.Select--single > .Select-control .Select-value,
        body[data-theme="white"] .Select.has-value.is-pseudo-focused.Select--single > .Select-control .Select-value {
            background: #ffffff !important;
            background-image: none !important;
            border-color: transparent !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select.has-value.Select--single > .Select-control .Select-value .Select-value-label,
        body[data-theme="white"] .Select.has-value.is-pseudo-focused.Select--single > .Select-control .Select-value .Select-value-label,
        body[data-theme="white"] .has-value.Select--single > .Select-control .Select-value a.Select-value-label,
        body[data-theme="white"] .has-value.is-pseudo-focused.Select--single > .Select-control .Select-value a.Select-value-label {
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select.is-focused:not(.is-open) > .Select-control {
            border-color: #7da8c4 !important;
            box-shadow: 0 0 0 2px rgba(36, 95, 143, 0.12) !important;
        }
        body[data-theme="white"] .sidebar .dash-checklist,
        body[data-theme="white"] .sidebar .dash-radioitems,
        body[data-theme="white"] .meta-toolbar .dash-checklist,
        body[data-theme="white"] .meta-toolbar .dash-radioitems,
        body[data-theme="white"] .plot-toolbar .dash-checklist,
        body[data-theme="white"] .plot-toolbar .dash-radioitems {
            background: transparent !important;
        }
        body[data-theme="white"] .sidebar .dash-checklist label,
        body[data-theme="white"] .sidebar .dash-radioitems label,
        body[data-theme="white"] .sidebar-camera-checklist label {
            background: #f5f8fb !important;
            border: 1px solid #d6e0e8 !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .meta-toolbar .dash-checklist label,
        body[data-theme="white"] .meta-toolbar .dash-radioitems label,
        body[data-theme="white"] .meta-toolbar label,
        body[data-theme="white"] .plot-toolbar .dash-checklist label,
        body[data-theme="white"] .plot-toolbar .dash-radioitems label,
        body[data-theme="white"] .plot-toolbar label {
            background: #f5f8fb !important;
            border: 1px solid #d6e0e8 !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .sidebar .dash-checklist label:hover,
        body[data-theme="white"] .sidebar .dash-radioitems label:hover,
        body[data-theme="white"] .sidebar-camera-checklist label:hover,
        body[data-theme="white"] .meta-toolbar .dash-checklist label:hover,
        body[data-theme="white"] .meta-toolbar .dash-radioitems label:hover,
        body[data-theme="white"] .meta-toolbar label:hover,
        body[data-theme="white"] .plot-toolbar .dash-checklist label:hover,
        body[data-theme="white"] .plot-toolbar .dash-radioitems label:hover,
        body[data-theme="white"] .plot-toolbar label:hover {
            background: #eaf1f6 !important;
            border-color: #b8c8d5 !important;
        }
        body[data-theme="white"] #sidebar-status {
            color: #2f7a57 !important;
        }
        body[data-theme="white"] #help-modal .modal-content,
        body[data-theme="white"] #help-modal .modal-header,
        body[data-theme="white"] #help-modal .modal-body,
        body[data-theme="white"] #help-modal .modal-footer {
            background-color: #ffffff !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .vetting-banner-empty,
        body[data-theme="white"] .vetting-banner-grid,
        body[data-theme="white"] .vetting-banner-links {
            background: #ffffff !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .vetting-banner-cell {
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] .vetting-banner-shell.known .vetting-banner-cell,
        body[data-theme="white"] .vetting-banner-shell.new .vetting-banner-cell {
            background: #f7fafc !important;
        }
        body[data-theme="white"] .vetting-banner-header.known {
            background: #fbe7e7 !important;
            color: #9f2d2d !important;
            border-color: #e4b4b4 !important;
        }
        body[data-theme="white"] .vetting-banner-header.new {
            background: #e8f7ec !important;
            color: #2f7a57 !important;
            border-color: #b7dcbf !important;
        }
        body[data-theme="white"] .vetting-banner-hit.known {
            color: #9f2d2d !important;
        }
        body[data-theme="white"] .vetting-banner-hit.new {
            color: #2f7a57 !important;
        }
        body[data-theme="white"] .vetting-banner-link {
            background: #f5f8fb !important;
            border-color: #c5d0da !important;
            color: #245f8f !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
'''

# Global runtime path env keys (used by background workers on Windows spawn).
_REVIEW_DB_ENV = "MALCA_REVIEW_DB_PATH"
_REVIEW_PLOT_ENV = "MALCA_REVIEW_PLOT_DIR"


def _env_path_or_none(name: str) -> str | None:
    """Return a stripped env path value, or None when unset/empty."""
    value = os.environ.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# Global variables
DB_PATH = _env_path_or_none(_REVIEW_DB_ENV) or str(DEFAULT_DB_PATH)
PLOT_DIR = _env_path_or_none(_REVIEW_PLOT_ENV)
INITIAL_CANDIDATE_QUERY: str | None = None
_PLOT_STATIC_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}
DEFAULT_THEME = "black"
FETCH_BACKEND_OPTIONS = [
    {"label": "SkyPatrol2 API", "value": "skypatrol2"},
    {"label": "SkyPatrol1 Web", "value": "skypatrol1"},
]
DEFAULT_FETCH_BACKEND = str(os.environ.get("MALCA_FETCH_BACKEND", "skypatrol2")).strip().lower()
if DEFAULT_FETCH_BACKEND not in {"skypatrol2", "skypatrol1"}:
    DEFAULT_FETCH_BACKEND = "skypatrol2"
DEFAULT_RESIDUAL_FRACTION = REVIEW_RESIDUAL_FRACTION
DEFAULT_EXTERNAL_SOURCE_VIEW = "asassn"
EXTERNAL_SOURCE_VIEW_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "ASAS-SN Only", "value": "asassn"},
    {"label": "ATLAS", "value": "atlas"},
    {"label": "ZTF", "value": "ztf"},
    {"label": "Gaia Epoch", "value": "gaia_epoch"},
    {"label": "PS1", "value": "ps1"},
    {"label": "CRTS", "value": "crts"},
]


def _review_persistence_token() -> str:
    """Return a persistence scope tied to the active review DB."""
    try:
        return str(Path(DB_PATH).expanduser().resolve())
    except Exception:
        return str(DB_PATH)

PLOT_PRESETS = {
    'Clean': {
        'overlays': ['raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras'],
        'camera_mode': 'all',
    },
    'Diagnostics': {
        'overlays': ['raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras', 'diagnostics'],
        'camera_mode': 'all',
    },
    'Full': {
        'overlays': ['raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras', 'diagnostics', 'confidence'],
        'camera_mode': 'all',
    },
}

NATIVE_BAND_OPTIONS = [
    {'label': ' g', 'value': 'g'},
    {'label': ' V', 'value': 'V'},
]


@lru_cache(maxsize=8)
def _load_run_params_meta_for_plot_dir(plot_dir: str | None) -> tuple[dict | None, str, str]:
    """Load run_params with status/meta from active plot directory."""
    if not plot_dir:
        return None, "missing", "No plot directory set"
    run_params_path = Path(plot_dir).resolve().parent / "run_params.json"
    if not run_params_path.exists():
        return None, "missing", f"Missing {run_params_path}"
    try:
        with open(run_params_path) as f:
            data = json.load(f)
    except Exception as exc:
        return None, "invalid", f"Could not parse run_params.json: {exc}"
    if not isinstance(data, dict):
        return None, "invalid", "run_params.json is not a JSON object"
    return data, "loaded", f"Loaded from {run_params_path}"


def _load_run_params_for_plot_dir(plot_dir: str | None) -> dict:
    """Load run_params.json near the active plot directory."""
    run_params, _status, _msg = _load_run_params_meta_for_plot_dir(plot_dir)
    return run_params or {}


def _render_stat_cards(stat_rows: list[tuple[str, str]]) -> list:
    """Render stats as grouped collapsible sections with readable labels."""
    if not stat_rows:
        return []

    label_overrides = {
        "stats_jd_start": "JD start (day)",
        "stats_jd_end": "JD end (day)",
        "stats_time_span_days": "Time span (days)",
        "stats_n_unique_nights": "Unique nights",
        "stats_duty_cycle_fraction": "Duty cycle",
        "stats_file_points_total": "Points total",
        "stats_file_points_kept_after_filter": "Points kept after filter",
        "stats_cadence_mean_dt_days": r"Cadence mean $\Delta t$ (days)",
        "stats_cadence_median_dt_days": r"Cadence median $\Delta t$ (days)",
        "stats_cadence_p05_dt_days": r"Cadence $P_{05}$ $\Delta t$ (days)",
        "stats_cadence_p95_dt_days": r"Cadence $P_{95}$ $\Delta t$ (days)",
        "stats_photometry_robust_sigma_mag": r"Robust $\sigma_m$ (mag)",
        "stats_photometry_std_mag": r"$\sigma_m$ (mag)",
        "stats_photometry_IQR_mag": r"IQR($m$) (mag)",
        "stats_photometry_mean_mag": r"Mean($m$) (mag)",
        "stats_photometry_median_mag": r"Median($m$) (mag)",
        "stats_photometry_weighted_mean_mag": r"Weighted mean($m$) (mag)",
        "stats_photometry_weighted_mean_sem": r"SEM($\bar{m}_w$) (mag)",
        "stats_photometry_weighted_std_mag": r"Weighted $\sigma_m$ (mag)",
        "stats_photometry_p05_mag": r"$P_{05}(m)$ (mag)",
        "stats_photometry_p16_mag": r"$P_{16}(m)$ (mag)",
        "stats_photometry_p84_mag": r"$P_{84}(m)$ (mag)",
        "stats_photometry_p95_mag": r"$P_{95}(m)$ (mag)",
        "stats_clipped_mean_mag_3sigma_about_median": r"Clipped mean (3$\sigma$ about median) (mag)",
        "stats_clipped_std_mag_3sigma_about_median": r"Clipped std (3$\sigma$ about median) (mag)",
        "stats_n_outliers_removed_robust_3sigma": r"Outliers removed (robust 3$\sigma$)",
        "stats_error_and_snr_stats_error_mean": "Error mean (mag)",
        "stats_error_and_snr_stats_error_median": "Error median (mag)",
        "stats_error_and_snr_stats_error_p05": r"Error $P_{05}$ (mag)",
        "stats_error_and_snr_stats_error_p95": r"Error $P_{95}$ (mag)",
        "stats_error_and_snr_stats_snr_median": "SNR median",
        "stats_error_and_snr_stats_snr_p05": r"SNR $P_{05}$",
        "stats_error_and_snr_stats_snr_p95": r"SNR $P_{95}$",
        "stats_variability_reduced_chi2_vs_constant": r"Reduced $\chi^2$ vs constant",
        "stats_variability_von_neumann_ratio": r"Inverse von Neumann ratio $1/\eta$",
        "stats_variability_roms": "RoMS",
        "stats_variability_lag1_autocorr": r"Lag-1 $\rho$",
        "stats_variability_stetson_I": r"Stetson $I$",
        "stats_variability_stetson_J": r"Stetson $J$",
        "stats_variability_stetson_K": r"Stetson $K$",
        "stats_variability_stetson_L": r"Stetson $L$",
        "stats_variability_stetson_J_time": r"Stetson $J_\mathrm{time}$",
        "stats_variability_stetson_L_time": r"Stetson $L_\mathrm{time}$",
        "stats_variability_string_length_resid_total": "String length total (mag)",
        "stats_variability_string_length_resid_mean_step": "String length mean step (mag)",
        "stats_variability_string_length_resid_n_steps": "String length n steps",
        "stats_variability_lomb_scargle_best_period_days": "Lomb-Scargle best period (days)",
        "stats_variability_lomb_scargle_peak_power": "Lomb-Scargle peak power",
        "stats_variability_lomb_scargle_fap": "Lomb-Scargle FAP",
        "stats_trend_slope_mag_per_day": r"$\mathrm{d}m/\mathrm{d}t$ (mag/day)",
        "stats_trend_slope_mag_per_year": r"$\mathrm{d}m/\mathrm{d}t$ (mag/year)",
        "stats_trend_r2": r"Trend $R^2$",
        "stats_gp_drw_sigma": r"GP-DRW $\sigma$ (mag)",
        "stats_gp_drw_tau": r"GP-DRW $\tau$",
        "stats_iar_phi": r"IAR $\phi$",
        "stats_sf_ml_amplitude": "SF-ML amplitude (mag)",
        "stats_sf_ml_gamma": r"SF-ML $\gamma$",
        "stats_psi_cs": r"Psi CS ($\psi_{\mathrm{CS}}$)",
        "stats_psi_eta": r"Psi eta ($\psi_{\eta}$)",
        "stats_con": "Con statistic",
        "stats_intrinsic_sigma_mag": r"Intrinsic $\sigma$ (mag)",
        "stats_excess_var": r"Intrinsic $\sigma$ (mag)",
        "stats_amplitude": "Amplitude (mag)",
        "stats_first_mag": r"First $m$ (mag)",
        "stats_max_slope": "Max slope (mag/day)",
        "stats_median_abs_dev": r"Median abs dev (mag)",
        "stats_gskew": "g-skew",
        "stats_meanvariance": "Mean/variance",
        "stats_median_brp": "Median BRP",
        "stats_constancy_p_value": "Constancy p-value",
        "stats_pvar": "Constancy p-value",
        "stats_q31": r"$Q_{31}$ (mag)",
        "stats_rcs": "RCS",
        "stats_autocor_length": "Autocorrelation length",
        "stats_delta_mag_fid": r"$\Delta m_{\mathrm{fid}}$ (mag)",
        "stats_beyond_1_std": "Beyond 1 std",
        "stats_small_kurtosis": "Small kurtosis",
        "stats_pair_slope_trend": "Pair slope trend",
        "stats_harmonics_mse": r"$\mathrm{MSE}$ ($\mathrm{mag}^2$)",
        "stats_harmonics_order": "Recommended harmonic order",
        "stats_harmonics_period": "Adopted period (d)",
        "stats_harmonics_a0": r"Zero-point $A_0$ (mag)",
        "stats_harmonics_model_amplitude": "Model amplitude (mag)",
        "stats_harmonics_reduced_chi2": r"Reduced $\chi^2$",
        "stats_mhps_pn_flag": "MHPS PN flag",
        "stats_mhps_non_zero": "MHPS non-zero count",
        "ltv_median": "Median (mag)",
        "ltv_median_err": "Median err proxy (mag)",
        "ltv_time_span_days": "Time span (days)",
        "ltv_n_unique_nights": "Unique nights",
        "ltv_vg_has_v": "Has V band",
        "ltv_vg_overlap_days": "V/g overlap (days)",
        "ltv_vg_overlap_fraction": "V/g overlap fraction",
        "filtered_cams": "Filtered cameras",
    }

    alerce_feature_keys = {
        "stats_amplitude",
        "stats_beyond_1_std",
        "stats_con",
        "stats_delta_mag_fid",
        "stats_intrinsic_sigma_mag",
        "stats_excess_var",
        "stats_first_mag",
        "stats_gskew",
        "stats_max_slope",
        "stats_meanvariance",
        "stats_median_abs_dev",
        "stats_median_brp",
        "stats_percent_amplitude",
        "stats_q31",
        "stats_skew",
        "stats_small_kurtosis",
        "stats_constancy_p_value",
        "stats_pvar",
        "stats_anderson_darling",
        "stats_pair_slope_trend",
        "stats_rcs",
        "stats_autocor_length",
    }

    group_order = [
        "LTV Summary",
        "LTV Trend",
        "LTV Seasons",
        "LTV Stochastic",
        "Coverage & Cadence",
        "Photometry & SNR",
        "Periodicity",
        "Variability",
        "Trend",
        "Harmonics",
        "Stochastic Models",
        "MHPS / Structure Function",
        "ALeRCE Features",
        "Camera Diagnostics",
        "Other",
    ]

    def _stat_group(key: str) -> str:
        if key == "filtered_cams":
            return "Camera Diagnostics"
        if key.startswith("ltv_stoch_"):
            return "LTV Stochastic"
        if key.startswith("ltv_season_") or key.startswith("ltv_leave1out_"):
            return "LTV Seasons"
        if key.startswith("ltv_trend_") or key in {
            "ltv_slope", "ltv_slope_quad", "ltv_max_diff", "ltv_coeff1", "ltv_coeff2"
        }:
            return "LTV Trend"
        if key.startswith("ltv_"):
            return "LTV Summary"
        if key.startswith("stats_file_points_") or key in {
            "stats_jd_start", "stats_jd_end", "stats_time_span_days",
            "stats_n_unique_nights", "stats_duty_cycle_fraction",
        } or key.startswith("stats_cadence_"):
            return "Coverage & Cadence"
        if key.startswith("stats_photometry_") or key.startswith("stats_error_and_snr_stats_") or key.startswith("stats_clipped_") or key == "stats_n_outliers_removed_robust_3sigma":
            return "Photometry & SNR"
        if key.startswith("stats_variability_lomb_scargle_") or key.startswith("stats_psi_"):
            return "Periodicity"
        if key.startswith("stats_variability_"):
            return "Variability"
        if key.startswith("stats_trend_"):
            return "Trend"
        if key.startswith("stats_harmonics_"):
            return "Harmonics"
        if key.startswith("stats_gp_drw_") or key.startswith("stats_iar_"):
            return "Stochastic Models"
        if key.startswith("stats_mhps_") or key.startswith("stats_sf_ml_"):
            return "MHPS / Structure Function"
        if key in alerce_feature_keys:
            return "ALeRCE Features"
        return "Other"

    def _fallback_label(key: str) -> str:
        if key.startswith("stats_"):
            raw = key[6:]
        elif key.startswith("ltv_"):
            raw = key[4:]
        else:
            raw = key
        token_map = {
            "jd": "JD",
            "snr": "SNR",
            "iqr": "IQR",
            "std": "Std",
            "gp": "GP",
            "drw": "DRW",
            "iar": "IAR",
            "mhps": "MHPS",
            "rcs": "RCS",
            "fap": "FAP",
            "bic": "BIC",
            "ls": "LS",
            "vg": "V/g",
            "ml": "ML",
            "chi2": r"$\chi^2$",
            "r2": r"$R^2$",
            "sigma": r"$\sigma$",
            "tau": r"$\tau$",
            "phi": r"$\phi$",
            "rho": r"$\rho$",
            "gamma": r"$\gamma$",
            "eta": r"$\eta$",
            "psi": r"$\psi$",
        }
        parts: list[str] = []
        for tok in [t for t in raw.split("_") if t]:
            lower = tok.lower()
            p_match = re.fullmatch(r"p(\d{2})", lower)
            if p_match:
                parts.append(rf"$P_{{{p_match.group(1)}}}$")
            elif lower in token_map:
                parts.append(token_map[lower])
            elif lower.isdigit():
                parts.append(lower)
            else:
                parts.append(lower.capitalize())
        return " ".join(parts)

    def _stat_label(key: str) -> str:
        if key in label_overrides:
            return label_overrides[key]
        mag_match = re.fullmatch(r"stats_harmonics_mag_(\d+)", key)
        if mag_match:
            n = mag_match.group(1)
            return rf"Amplitude $A_{{{n}}}$ (mag)"
        ratio_match = re.fullmatch(r"stats_harmonics_r(\d+)1", key)
        if ratio_match:
            n = ratio_match.group(1)
            return rf"Amplitude ratio $R_{{{n}1}}$"
        phase_match = re.fullmatch(r"stats_harmonics_phase_(\d+)", key)
        if phase_match:
            n = phase_match.group(1)
            return rf"Phase combination $\phi_{{{n}1}}$ (rad)"
        return _fallback_label(key)

    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in group_order}
    for key_raw, value in stat_rows:
        key = str(key_raw)
        grouped.setdefault(_stat_group(key), []).append((_stat_label(key), str(value)))

    sections = []
    for group_name in group_order:
        rows = grouped.get(group_name, [])
        if not rows:
            continue
        field_divs = [
            html.Div([
                dcc.Markdown(label, className='meta-field-label stat-field-label', mathjax=True),
                dcc.Markdown(value, className='meta-field-value', mathjax=True),
            ], className='meta-field-row')
            for label, value in rows
        ]
        sections.append(
            html.Details(
                [html.Summary(f"{group_name} ({len(rows)})"), html.Div(field_divs, className='meta-grid')],
                open=group_name in {"Coverage & Cadence", "Photometry & SNR"},
            )
        )

    total_rows = sum(len(grouped.get(name, [])) for name in group_order)
    return [html.Details(
        [html.Summary(f"Stats ({total_rows})"), html.Div(sections, className='metadata-sections')],
        open='open',
    )]


def _render_plot_status_panel(status: str, message: str, warnings: list[str] | None) -> html.Div:
    """Render native plot status and warnings panel."""
    warnings = [str(w) for w in (warnings or []) if str(w).strip()]
    cls = 'plot-status'
    if status in {'missing-file', 'missing-columns', 'empty-after-filter', 'empty-camera-selection', 'error'}:
        cls += ' error'
    elif warnings:
        cls += ' warn'

    base_message = str(message or '').strip()
    if not base_message:
        base_message = 'Native interactive plot active.'

    headline = html.Div(base_message, className='status-line')
    if not warnings:
        return html.Div([headline], className=cls)

    warning_items = [html.Li(w) for w in warnings[:8]]
    if len(warnings) > 8:
        warning_items.append(html.Li(f"...and {len(warnings) - 8} more"))

    details = html.Details([
        html.Summary(f"{len(warnings)} warning(s)"),
        html.Ul(warning_items),
    ])
    return html.Div([headline, details], className=cls)


def _render_camera_diag_panel(camera_diagnostics: dict[str, list[str]], filtered_values: list[str] | None) -> list:
    """Render explainable camera filtering tags."""
    if not filtered_values:
        return []

    chips = []
    for cam in sorted(filtered_values):
        reasons = camera_diagnostics.get(str(cam), [])
        label = f"Cam {cam}: {','.join(reasons) if reasons else 'unknown'}"
        chips.append(html.Div(label, className='item'))
    return chips


def _run_params_path_for_plot_dir(plot_dir: str | None) -> Path | None:
    """Get run_params.json path from current plot directory."""
    if not plot_dir:
        return None
    candidate = Path(plot_dir).resolve().parent / "run_params.json"
    return candidate if candidate.exists() else None


def _derive_defaults_from_run_params(run_params: dict | None) -> tuple[str, list[str]]:
    """Derive initial preset and overlays from run config."""
    if not run_params:
        preset = 'Diagnostics'
        return preset, list(PLOT_PRESETS[preset]['overlays'])

    run_filter = bool(run_params.get('run_filter', True))
    run_postprocess = bool(run_params.get('run_postprocess', True))
    min_bf = float(run_params.get('min_bayes_factor', 0.0) or 0.0)
    if (not run_filter) and (not run_postprocess):
        preset = 'Clean'
    elif min_bf >= 12:
        preset = 'Full'
    else:
        preset = 'Diagnostics'

    overlays = set(PLOT_PRESETS[preset]['overlays'])
    if bool(run_params.get('skip_camera_median', False)):
        overlays.discard('filter_bad_cameras')
    if not bool(run_params.get('run_postprocess', True)):
        overlays.discard('diagnostics')
        overlays.discard('confidence')
    return preset, sorted(list(overlays))


def _run_config_rows(run_params: dict) -> list[tuple[str, str]]:
    """Compact rows for run config panel."""
    rows: list[tuple[str, str]] = []
    for label, key in (
        ('Stage', 'stage'),
        ('Baseline', 'baseline_func'),
        ('Baseline S0', 'baseline_s0'),
        ('Baseline w0', 'baseline_w0'),
        ('Baseline q', 'baseline_q'),
        ('Baseline jitter', 'baseline_jitter'),
        ('Baseline sigma floor', 'baseline_sigma_floor'),
        ('Trigger mode', 'trigger_mode'),
        ('Workers', 'workers'),
        ('Batch size', 'batch_size'),
        ('Min Bayes factor', 'min_bayes_factor'),
        ('LogBF dip thr', 'logbf_threshold_dip'),
        ('LogBF jump thr', 'logbf_threshold_jump'),
        ('Significance thr', 'significance_threshold'),
        ('Clean err abs', 'clean_max_error_absolute'),
        ('Clean err sigma', 'clean_max_error_sigma'),
        ('Bad cam scatter', 'bad_camera_scatter_ratio'),
    ):
        val = run_params.get(key)
        if val is None:
            continue
        rows.append((label, str(val)))
    return rows


def _run_config_mismatch_warnings(run_params: dict | None, overlays: set[str]) -> list[str]:
    """Warnings for GUI/view assumptions mismatching run config."""
    if not run_params:
        return ["run_params.json missing; native plot uses fallback defaults."]

    warns: list[str] = []
    expected_filter_bad = not bool(run_params.get('skip_camera_median', False))
    if ('filter_bad_cameras' in overlays) != expected_filter_bad:
        warns.append(
            f"Bad-camera filter toggle differs from run config (expected {'on' if expected_filter_bad else 'off'})."
        )

    if run_params.get('baseline_func') is None:
        warns.append("baseline_func missing in run_params; baseline defaults may differ from original run.")
    if run_params.get('clean_max_error_absolute') is None or run_params.get('clean_max_error_sigma') is None:
        warns.append("cleaning thresholds missing in run_params; fallback cleaning defaults are active.")
    return warns


def _path_is_under(path: Path | None, root: Path | None) -> bool:
    """Return True when *path* is located under *root*."""
    if path is None or root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _baseline_provenance_warning(
    payload: dict,
    *,
    plot_dir: Path | None,
    run_params: dict | None,
    stored_lc_path: object = None,
    source_path: object = None,
) -> str | None:
    """Describe when the displayed baseline is not provenance-matched to a pipeline run."""
    lc_path = resolve_lightcurve_path(payload, plot_dir)
    if lc_path is None:
        return None

    if not run_params:
        return "Baseline is recomputed live in review with fallback defaults; no run_params.json is loaded."

    source_text = str(source_path or '').strip()
    if source_text.startswith('fetch://'):
        return "Baseline is recomputed from the imported SkyPatrol light curve using the current run settings; it is not a saved pipeline baseline."

    plot_run_dir = plot_dir.parent.resolve() if plot_dir is not None else None
    source_run_dir = _run_dir_from_source_path(source_path)

    if _path_is_under(lc_path, plot_run_dir) or _path_is_under(lc_path, source_run_dir):
        return None

    cluster_path = str(payload.get('path') or '').strip()
    if cluster_path:
        try:
            if Path(cluster_path).expanduser().resolve() == lc_path.resolve():
                return None
        except Exception:
            pass

    stored_lc_text = str(stored_lc_path or '').strip()
    if stored_lc_text:
        try:
            if Path(stored_lc_text).expanduser().resolve() == lc_path.resolve():
                return "Baseline is recomputed from the local review copy of this light curve; it may differ from the original pipeline baseline source."
        except Exception:
            pass

    if lc_path.suffix.lower() == '.csv':
        return "Baseline is recomputed from a local CSV light curve in review; it may not match the original pipeline baseline source."

    return "Baseline is recomputed from the currently resolved local light curve; it may differ from the original pipeline baseline source."


def _render_run_config_panel(run_params: dict | None, run_params_path: Path | None, warnings: list[str]) -> list:
    """Render run config as simple meta-field rows (same style as stats/metadata)."""
    rows: list[tuple[str, str]] = [
        ('Status', 'Loaded' if run_params else 'Missing'),
        ('Path', str(run_params_path) if run_params_path else 'not found'),
    ]
    rows.extend(_run_config_rows(run_params or {}))
    if warnings:
        rows.append(('Warnings', ' | '.join(warnings)))

    field_divs = [
        html.Div([
            html.Span(label, className='meta-field-label'),
            html.Span(str(value), className='meta-field-value'),
        ], className='meta-field-row')
        for label, value in rows
    ]
    return [html.Div(field_divs, className='meta-grid')]


def _render_repro_badge(run_params: dict | None, warnings: list[str]) -> html.Span:
    """Render reproducibility status badge."""
    if (run_params is None) or warnings:
        text = 'Repro: fallback/defaults'
        cls = 'repro-badge warn'
    else:
        text = 'Repro: exact run params'
        cls = 'repro-badge'
    return html.Span(text, className=cls)


def _render_explorer_selection_panel(selection_meta: dict | None) -> list:
    """Render compact provenance for queues exported from explorer."""
    meta = dict(selection_meta or {})
    if not meta:
        return [html.Div('This review DB was not exported from explorer.', style={'color': '#7d91a6'})]

    plot_meta = dict(meta.get('plot') or {})
    filter_meta = dict(meta.get('filters') or {})
    rows: list[tuple[str, object]] = [
        ('Created', meta.get('created_at')),
        ('Candidates', meta.get('candidate_count')),
        ('Filtered rows', meta.get('filtered_count')),
        ('Sources', ', '.join(str(v) for v in (meta.get('source_labels') or [])) or '-'),
        ('Query', filter_meta.get('query') or '-'),
        ('X metric', plot_meta.get('x_metric') or '-'),
        ('Y metric', plot_meta.get('y_metric') or '-'),
    ]
    source_files = [str(v) for v in (meta.get('source_files') or []) if str(v).strip()]
    if source_files:
        rows.append(('Source DBs', '; '.join(source_files[:3]) + (' ...' if len(source_files) > 3 else '')))

    return [
        html.Div([
            html.Span(f'{label}: ', className='meta-field-label'),
            html.Span(str(value), className='meta-field-value'),
        ], className='meta-field-row')
        for label, value in rows
        if value not in (None, '')
    ]


def _format_queue_count(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _render_queue_filter_provenance_panel(queue_data: dict | None) -> list:
    """Render queue attrition summary for active sidebar filters."""
    data = dict(queue_data or {})
    scope_size = int(data.get("scope_size") or 0)
    visible_size = int(data.get("queue_size") or 0)
    filtered_out = int(data.get("filtered_out_count") or max(scope_size - visible_size, 0))
    active_filters = list(data.get("filter_provenance") or [])

    summary_rows = [
        ("Scoped queue", _format_queue_count(scope_size)),
        ("Visible now", _format_queue_count(visible_size)),
        ("Filtered out", _format_queue_count(filtered_out)),
        ("Active sidebar filters", _format_queue_count(len(active_filters))),
    ]

    summary = html.Div([
        html.Div([
            html.Span(label, className='meta-field-label'),
            html.Span(str(value), className='meta-field-value'),
        ], className='meta-field-row')
        for label, value in summary_rows
    ], className='meta-grid')

    if active_filters:
        details = html.Ul([
            html.Li(
                f"{item.get('label', 'filter')}: "
                f"{_format_queue_count(item.get('filtered_count', 0))} filtered out "
                f"({_format_queue_count(item.get('remaining_count', 0))} remain)"
            )
            for item in active_filters
        ], className='queue-provenance-list')
        note = html.Div(
            "Per-filter counts are relative to the full scoped queue. "
            "Filters can overlap, so these counts do not sum to the total filtered-out rows.",
            className='queue-provenance-note',
        )
    else:
        details = html.Div(
            "No sidebar filters are active. The visible queue matches the full scoped queue.",
            className='queue-provenance-note',
        )
        note = None

    children: list = [summary, details]
    if note is not None:
        children.append(note)
    return [html.Div(children, className='queue-provenance-panel')]


_METADATA_EXTRA_GROUPS = (
    "Triage Summary",
    "Vetting",
    "Period Consensus",
    "External Follow-up",
    "Stellar Parameters",
    "Photometry",
    "Environment",
    "YSO / Classification",
)

_METADATA_CATALOG_GROUPS = (
    "Vetting",
    "Period Consensus",
    "External Follow-up",
    "Stellar Parameters",
    "Photometry",
    "Environment",
)


def _summarize_group_names(group_names: list[str], max_items: int = 2) -> str:
    names = [str(n) for n in group_names if str(n).strip()]
    if not names:
        return "none"
    if len(names) <= max_items:
        return ", ".join(names)
    return f"{', '.join(names[:max_items])} +{len(names) - max_items}"


def _render_metadata_health(grouped: list[tuple[str, list[tuple[str, object]]]] | None, *, context_msg: str | None = None) -> html.Div:
    """Render compact metadata-enrichment status for current candidate."""
    if context_msg:
        return html.Div([
            html.Span("Base only", className='chip'),
            html.Span(str(context_msg), className='detail'),
        ], className='metadata-health metadata-health-base')

    grouped_map = {name: items for name, items in (grouped or [])}
    extra_present = [name for name in _METADATA_EXTRA_GROUPS if grouped_map.get(name)]
    catalog_present = [name for name in _METADATA_CATALOG_GROUPS if grouped_map.get(name)]
    extra_fields = int(sum(len(grouped_map.get(name) or []) for name in extra_present))

    if not extra_present:
        return html.Div([
            html.Span("Base only", className='chip'),
            html.Span(
                "No crossmatch/classification/catalog metadata fields are present for this candidate.",
                className='detail',
            ),
        ], className='metadata-health metadata-health-base')

    if catalog_present:
        detail = (
            f"Catalog metadata present in {_summarize_group_names(catalog_present)} "
            f"({extra_fields} extra fields total)."
        )
        return html.Div([
            html.Span("Catalog enriched", className='chip'),
            html.Span(detail, className='detail'),
        ], className='metadata-health metadata-health-enriched')

    extra_label = _summarize_group_names(extra_present)
    return html.Div([
        html.Span("Partial", className='chip'),
        html.Span(
            f"Only {extra_label} metadata is present ({extra_fields} extra fields); no catalog stellar/photometric enrichment.",
            className='detail',
        ),
    ], className='metadata-health metadata-health-partial')


def _render_vetting_banner(payload: dict | None, radius_arcsec: float = 10.0) -> html.Div:
    """Render a vetting status panel with source cards above the metadata grid."""
    if not payload or 'vetting_likely_known' not in payload:
        return html.Div("Not vetted", className='vetting-banner-empty')

    known = payload.get('vetting_likely_known')
    banner_state = 'known' if known else 'new'

    # Status header
    header_text = "KNOWN VARIABLE" if known else "POTENTIALLY NEW"

    cards = []

    def _ok(v) -> bool:
        """True if v is a non-empty, non-NaN string."""
        return bool(v) and str(v).strip().lower() not in ('nan', '<na>')

    def _label(text: str) -> html.Span:
        return html.Span(text, className='vetting-banner-label')

    def _value(text: str, *, hit: bool = False, title: str | None = None) -> html.Span:
        cls = 'vetting-banner-value'
        if hit:
            cls += f' vetting-banner-hit {banner_state}'
        return html.Span(text, className=cls, title=title)

    def _cell(left: str, right: str, *, hit: bool = False, title: str | None = None) -> html.Div:
        return html.Div([
            _label(left),
            _value(right, hit=hit, title=title),
        ], className='vetting-banner-cell')

    # SIMBAD cell
    simbad_id = payload.get('simbad_main_id')
    simbad_otype = payload.get('simbad_otype')
    if _ok(simbad_id) or _ok(simbad_otype):
        refs = payload.get('simbad_nbref')
        parts = []
        if _ok(simbad_otype):
            parts.append(str(simbad_otype))
        if _ok(simbad_id):
            parts.append(str(simbad_id))
        if refs:
            parts.append(f"({refs} refs)")
        cards.append(_cell("SIMBAD", " \u00b7 ".join(parts), hit=True, title=str(simbad_id or '')))

    # VSX cell
    vsx_cls = payload.get('vsx_class')
    if _ok(vsx_cls):
        vsx_sep = payload.get('vsx_sep_arcsec')
        sep_str = f" ({vsx_sep:.1f}\")" if vsx_sep and not pd.isna(vsx_sep) else ""
        vsx_p = payload.get('vsx_period')
        p_str = f", P={vsx_p:.4f}d" if vsx_p and not pd.isna(vsx_p) else ""
        cards.append(_cell("VSX", f"{vsx_cls}{p_str}{sep_str}", hit=True))

    # Gaia variability cell
    gaia_cls = payload.get('gaia_var_class')
    if _ok(gaia_cls):
        score = payload.get('gaia_var_score')
        score_str = f" ({score:.2f})" if score and not pd.isna(score) else ""
        cards.append(_cell("Gaia DR3", f"{gaia_cls}{score_str}", hit=True))

    # Gaia EB period cell
    eb_period = payload.get('gaia_eb_period')
    if eb_period and not pd.isna(eb_period):
        cards.append(_cell("Gaia EB", f"P={eb_period:.4f} d", hit=True))

    # ASAS-SN cell
    asassn_type = payload.get('asassn_var_type')
    if _ok(asassn_type):
        period = payload.get('asassn_var_period')
        p_str = f" P={period:.4f}d" if period and not pd.isna(period) else ""
        cards.append(_cell("ASAS-SN", f"{asassn_type}{p_str}", hit=True))

    # Microlensing catalog cell
    if _coerce_bool(payload.get('microlens_match')):
        ml_catalog = payload.get('microlens_catalog')
        ml_name = payload.get('microlens_name')
        ml_te = payload.get('microlens_te_days')
        ml_sep = payload.get('microlens_sep_arcsec')
        display = ml_name if _ok(ml_name) else (ml_catalog if _ok(ml_catalog) else "Match")
        te_str = f" tE={ml_te:.1f}d" if ml_te and not pd.isna(ml_te) else ""
        sep_str = f" ({ml_sep:.1f}\")" if ml_sep and not pd.isna(ml_sep) else ""
        cards.append(_cell("Microlens", f"{display}{te_str}{sep_str}", hit=True))

    # ZTF cell
    ztf_type = payload.get('ztf_var_type')
    if _ok(ztf_type):
        ztf_p = payload.get('ztf_var_period')
        zp_str = f" P={ztf_p:.4f}d" if ztf_p and not pd.isna(ztf_p) else ""
        cards.append(_cell("ZTF", f"{ztf_type}{zp_str}", hit=True))

    # TNS cell
    tns_name = payload.get('tns_name')
    if _ok(tns_name):
        tns_type = payload.get('tns_type', '')
        cards.append(_cell("TNS", f"{tns_name} ({tns_type})" if tns_type else tns_name, hit=True))

    # ALeRCE cell
    alerce_cls = payload.get('alerce_lc_class')
    if _ok(alerce_cls):
        prob = payload.get('alerce_lc_prob')
        prob_str = f" ({prob:.0%})" if prob and not pd.isna(prob) else ""
        cards.append(_cell("ALeRCE", f"{alerce_cls}{prob_str}", hit=True))

    # X-ray cell
    xray = payload.get('xray_det')
    if xray:
        flux = payload.get('xray_flux')
        flux_str = f" {flux:.1e}" if flux and not pd.isna(flux) else ""
        cards.append(_cell("X-ray", f"Detected{flux_str}", hit=True))

    # SFR cell
    sfr_name = payload.get('sfr_name')
    if _ok(sfr_name):
        sfr_sep = payload.get('sfr_sep_arcmin')
        sep_str = f" ({sfr_sep:.1f}')" if sfr_sep and not pd.isna(sfr_sep) else ""
        cards.append(_cell("SFR", f"{sfr_name}{sep_str}", hit=True))

    # Cluster cell
    cluster_name = payload.get('cluster_name')
    if _ok(cluster_name):
        cluster_dist = payload.get('cluster_dist_pc')
        d_str = f" ({cluster_dist:.0f} pc)" if cluster_dist and not pd.isna(cluster_dist) else ""
        cards.append(_cell("Cluster", f"{cluster_name}{d_str}", hit=True))

    # BANYAN cell
    banyan_assoc = payload.get('banyan_best_assoc')
    banyan_fp = payload.get('banyan_field_prob')
    if _ok(banyan_assoc) and str(banyan_assoc).strip().lower() != 'field':
        fp_str = f" (P_field={banyan_fp:.0%})" if banyan_fp and not pd.isna(banyan_fp) else ""
        cards.append(_cell("BANYAN", f"{banyan_assoc}{fp_str}", hit=True))

    # YSO class cell (skip generic classifications from IR color-color)
    yso_cls = payload.get('yso_class')
    if yso_cls and str(yso_cls).strip().lower() not in ('nan', '<na>', '', 'main sequence', 'unknown'):
        cards.append(_cell("YSO", str(yso_cls), hit=True))

    # Spectra cell
    has_spectrum = _coerce_bool(payload.get('has_spectrum'))
    if has_spectrum:
        sources = payload.get('spectrum_sources')
        src_str = f" ({sources})" if sources and not pd.isna(sources) else ""
        cards.append(_cell("Spectra", f"Available{src_str}", hit=True))

    # OGLE cell
    ogle_match = payload.get('period_ogle_match')
    if ogle_match:
        ogle_cls = payload.get('period_ogle_class', '')
        ogle_p = payload.get('period_ogle_days')
        ogle_sep = payload.get('period_ogle_sep_arcsec')
        p_str = f" P={ogle_p:.4f}d" if ogle_p and not pd.isna(ogle_p) else ""
        sep_str = f" ({ogle_sep:.1f}\")" if ogle_sep and not pd.isna(ogle_sep) else ""
        cards.append(_cell("OGLE", f"{ogle_cls}{p_str}{sep_str}" if ogle_cls else f"Match{p_str}{sep_str}", hit=True))

    # unWISE W1 variability cell
    w1_var = payload.get('unwise_w1_var')
    if w1_var:
        w1_z = payload.get('unwise_w1_zscore')
        z_str = f" (z={w1_z:.1f})" if w1_z and not pd.isna(w1_z) else ""
        cards.append(_cell("unWISE W1", f"Variable{z_str}", hit=True))

    # LTV trend cell
    ltv_slope = payload.get('ltv_slope')
    if ltv_slope is not None and not pd.isna(ltv_slope):
        ltv_diff = payload.get('ltv_max_diff')
        ltv_fap = payload.get('ltv_ls_fap')
        direction = "▲" if ltv_slope > 0 else "▼"
        diff_str = f" Δ{ltv_diff:.3f}mag" if ltv_diff and not pd.isna(ltv_diff) else ""
        fap_str = f" FAP={ltv_fap:.2e}" if ltv_fap and not pd.isna(ltv_fap) else ""
        cards.append(_cell("LTV", f"{direction}{ltv_slope:+.4f} mag/yr{diff_str}{fap_str}", hit=True))

    # IPHAS H-alpha excess cell
    ha_excess = payload.get('iphas_ha_excess')
    if ha_excess and not pd.isna(ha_excess) and float(ha_excess) > 0:
        cards.append(_cell("IPHAS Hα", f"excess={float(ha_excess):.2f}", hit=True))

    # Gaia epoch cell (non-hit, informational)
    epoch_n = payload.get('gaia_epoch_n_obs')
    if epoch_n and int(epoch_n) > 0:
        g_range = payload.get('gaia_epoch_g_range')
        r_str = f", dG={g_range:.2f}" if g_range and not pd.isna(g_range) else ""
        cards.append(_cell("Gaia epoch", f"{int(epoch_n)} obs{r_str}"))

    if not cards and not known:
        # No matches at all — emphasize "new"
        cards.append(html.Div([
            _value("No catalog matches found", hit=True),
        ], className='vetting-banner-cell'))

    # External links toolbar
    links = build_external_lookup_links(payload, radius_arcsec=radius_arcsec)
    links_row = None
    if links:
        link_els = []
        for label, url in links:
            link_els.append(html.A(
                label,
                href=url,
                target='_blank',
                rel='noopener noreferrer',
                className='vetting-banner-link',
            ))

        links_row = html.Div(link_els, className='vetting-banner-links')

    children = [
        html.Div(header_text, className=f'vetting-banner-header {banner_state}'),
        html.Div(cards, className='vetting-banner-grid with-links' if links_row else 'vetting-banner-grid'),
    ]
    if links_row:
        children.append(links_row)

    return html.Div(children, className=f'vetting-banner-shell {banner_state}')


def _keyboard_key(key_value: str | None) -> str:
    """Extract raw key token from encoded keyboard input value."""
    if not key_value:
        return ""
    return str(key_value).split("\t", 1)[0].strip()


def _format_large_integer_like_display(value) -> str:
    """Format integer-like values without scientific notation for display."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s
    if d != d.to_integral_value():
        return s
    try:
        return format(d.to_integral_value(), "f")
    except Exception:
        return s


def _truncate_bottom_context_value(value: object, *, max_len: int = 84, tail_parts: int = 4) -> tuple[str, str]:
    """Return (full_text, display_text) for compact bottom-bar display."""
    full = str(value or "-").strip() or "-"
    if len(full) <= max_len:
        return full, full

    normalized = full.replace("\\", "/")
    if "/" in normalized:
        parts = [p for p in normalized.split("/") if p]
        prefix = "/" if normalized.startswith("/") else ""
        if len(parts) > tail_parts:
            display = f"{prefix}.../{'/'.join(parts[-tail_parts:])}"
            if len(display) <= max_len:
                return full, display

    head = max(12, max_len // 2 - 6)
    tail = max(12, max_len - head - 3)
    return full, f"{full[:head]}...{full[-tail:]}"


def _render_bottom_context(label: str, value: object) -> html.Div:
    """Render one labeled bottom-bar context item."""
    full, display = _truncate_bottom_context_value(value)
    return html.Div(
        [
            html.Span(f"{label}:", className='bottom-context-k'),
            html.Span(display, className='bottom-context-v', title=full),
        ],
        className='bottom-context-item',
    )


def _resolve_run_dir_from_plot_dir(plot_dir: str | None) -> Path | None:
    """Infer run directory from plot-dir or run-dir style path."""
    if not plot_dir:
        return None
    p = Path(str(plot_dir)).expanduser().resolve()
    if p.name == "plots":
        return p.parent
    if (p / "plots").is_dir():
        return p
    if (p / "results").is_dir():
        return p
    if (p.parent / "results").is_dir():
        return p.parent
    if (p.parent / "plots").is_dir():
        return p.parent
    return None


def _resolve_run_dir_from_db_path(db_path: str | Path | None) -> Path | None:
    """Infer run directory from a review DB path or standalone bundled DB path."""
    if not db_path:
        return None
    p = Path(str(db_path)).expanduser().resolve()
    if p.suffix.lower() != ".db":
        return None
    if (p.parent / "results").is_dir() or (p.parent / "plots").is_dir() or (p.parent / "bundle_assets" / "lightcurves").is_dir():
        return p.parent
    if p.parent.name != "review":
        return None
    run_dir = p.parent.parent
    if (run_dir / "results").is_dir() or (run_dir / "plots").is_dir() or (run_dir / "bundle_assets" / "lightcurves").is_dir():
        return run_dir
    return None


def _review_db_for_plot_dir(plot_dir: str | None) -> Path | None:
    """Return the sibling run-local review DB for a plot dir, if present."""
    run_dir = _resolve_run_dir_from_plot_dir(plot_dir)
    if run_dir is None:
        return None
    candidate = run_dir / "review" / "review.db"
    if candidate.exists():
        return candidate.resolve()
    return None


def _db_plot_mismatch_warning(db_path: str | Path | None, plot_dir: str | None) -> str:
    """Describe likely DB/plot-dir mismatches that would hide candidates."""
    if not plot_dir or not db_path:
        return ""

    selected = Path(str(db_path)).expanduser().resolve()
    sibling = _review_db_for_plot_dir(plot_dir)
    if sibling is None or sibling == selected:
        return ""

    selected_count = _count_candidates_in_db(selected)
    sibling_count = _count_candidates_in_db(sibling)
    if sibling_count < 0:
        return ""

    if selected_count == 0 and sibling_count > 0:
        return (
            f"Selected DB {selected} has 0 candidates, but the run-local DB for "
            f"{Path(str(plot_dir)).expanduser().resolve()} is {sibling} with {sibling_count} candidates. "
            f"Use --db {sibling} or omit --db to use the run-local DB automatically."
        )

    return ""


def _project_root() -> Path:
    """Repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


def _configured_plot_dir() -> Path | None:
    """Return the configured plot directory, if any."""
    if not PLOT_DIR:
        return None
    return Path(str(PLOT_DIR)).expanduser().resolve()


def _review_db_state_signature(db_path: str | Path | None = None) -> str:
    """Return a cache signature that tracks review DB state, including WAL writes."""
    base = Path(db_path or DB_PATH).expanduser()
    try:
        resolved = base.resolve()
    except Exception:
        resolved = base

    parts = [str(resolved)]
    for suffix, label in (("", "db"), ("-wal", "wal")):
        path = resolved if not suffix else Path(f"{resolved}{suffix}")
        try:
            stat = path.stat()
            parts.append(f"{label}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}")
        except Exception:
            parts.append(f"{label}:missing")
    return "|".join(parts)


def _diagnostic_background_signature(db_path: str | Path | None = None) -> str:
    """Return a stable cache signature for diagnostic background data."""
    try:
        resolved = Path(db_path or DB_PATH).expanduser().resolve()
        stat = resolved.stat()
        return f"{resolved}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}"
    except Exception:
        return str(Path(db_path or DB_PATH).expanduser())


def _diagnostic_background_cache_key(signature: str) -> str:
    """Return diskcache key used for diagnostic background blobs."""
    return f"diagnostic-background:{signature}"


def _get_cached_diagnostic_background(signature: str | None) -> dict | None:
    """Load cached diagnostic background data from diskcache."""
    if not signature:
        return None
    try:
        cached = _bc_cache.get(_diagnostic_background_cache_key(signature))
    except Exception:
        return None
    return cached if isinstance(cached, dict) else None


def _store_cached_diagnostic_background(signature: str, background: dict) -> None:
    """Persist diagnostic background data to diskcache."""
    if not signature:
        return
    _bc_cache.set(_diagnostic_background_cache_key(signature), background)


@lru_cache(maxsize=512)
def _candidate_context_cached(
    db_path_text: str,
    db_signature: str,
    candidate_id: str,
) -> tuple[str, str | None, str | None]:
    """Load one candidate payload + local path context from SQLite, cached by DB state."""
    _ = db_signature
    with closing(db_connect(Path(db_path_text))) as conn:
        payload = get_candidate_payload(conn, candidate_id) or {}
        row = conn.execute(
            "SELECT lc_path, source_path FROM candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()

    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("candidate_id"):
        payload["candidate_id"] = str(candidate_id)

    payload_json = json.dumps(payload, sort_keys=True, default=str)
    stored_lc_path = None
    source_path = None
    if row:
        stored_lc_path = str(row[0]).strip() if row[0] not in (None, "") else None
        source_path = str(row[1]).strip() if row[1] not in (None, "") else None
    return payload_json, stored_lc_path, source_path


def _candidate_context(candidate_id: str | None) -> tuple[dict, str | None, str | None]:
    """Return payload plus stored lc/source paths for the active DB."""
    cid = str(candidate_id or "").strip()
    if not cid:
        return {}, None, None

    payload_json, stored_lc_path, source_path = _candidate_context_cached(
        str(Path(DB_PATH).expanduser()),
        _review_db_state_signature(DB_PATH),
        cid,
    )
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("candidate_id"):
        payload["candidate_id"] = cid
    return payload, stored_lc_path, source_path


@lru_cache(maxsize=16)
def _progress_counts_cached(db_path_text: str, db_signature: str) -> tuple[int, int]:
    """Load reviewed/total progress counts, cached by DB state."""
    _ = db_signature
    with closing(db_connect(Path(db_path_text))) as conn:
        return count_progress(conn)


def _progress_counts() -> tuple[int, int]:
    """Return reviewed/total counts for the active review DB."""
    return _progress_counts_cached(
        str(Path(DB_PATH).expanduser()),
        _review_db_state_signature(DB_PATH),
    )


def _clear_review_state_caches() -> None:
    """Clear in-process caches derived from the review DB."""
    _candidate_context_cached.cache_clear()
    _progress_counts_cached.cache_clear()


def _run_dir_from_source_path(source_path: object = None) -> Path | None:
    """Infer a run directory from a candidate source path when possible."""
    text = str(source_path or "").strip()
    if text:
        try:
            source = Path(text).expanduser().resolve()
        except Exception:
            source = Path(text).expanduser()

        for candidate in (source, source.parent, source.parent.parent):
            if (candidate / "results").is_dir() or (candidate / "bundle_assets" / "lightcurves").is_dir():
                return candidate

    return _resolve_run_dir_from_db_path(DB_PATH)


def _effective_local_lc_path(
    payload: dict | None,
    *,
    stored_lc_path: object = None,
    source_path: object = None,
) -> str | None:
    """Return the best local/bundled LC path for review UI display and lookup."""
    payload_dict = dict(payload) if isinstance(payload, dict) else {}

    explicit_local_paths: list[str] = []
    for raw in (stored_lc_path, payload_dict.get("lc_path")):
        text = str(raw or "").strip()
        if not text or text in explicit_local_paths:
            continue
        explicit_local_paths.append(text)
        try:
            if Path(text).expanduser().exists():
                return text
        except Exception:
            continue

    if explicit_local_paths and not payload_dict.get("lc_path"):
        payload_dict["lc_path"] = explicit_local_paths[0]

    plot_dir = _configured_plot_dir()
    if plot_dir is None:
        run_dir = _run_dir_from_source_path(source_path)
        if run_dir is not None:
            plot_dir = run_dir / "plots"

    try:
        resolved = resolve_lightcurve_path(payload_dict, plot_dir)
    except Exception:
        resolved = None

    cluster_lc_path = str(payload_dict.get("path") or "").strip()
    if resolved is not None:
        resolved_text = str(resolved)
        if resolved_text and resolved_text != cluster_lc_path:
            return resolved_text

    if explicit_local_paths and cluster_lc_path:
        return explicit_local_paths[0]

    return None


def _display_lc_paths(
    payload: dict | None,
    *,
    stored_lc_path: object = None,
    source_path: object = None,
) -> tuple[str | None, str | None]:
    """Return display-friendly (cluster_path, local_path) values for the footer/search UI."""
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    cluster_lc_path = str(payload_dict.get("path") or "").strip() or None
    local_lc_path = _effective_local_lc_path(
        payload_dict,
        stored_lc_path=stored_lc_path,
        source_path=source_path,
    )

    stored_lc_text = str(stored_lc_path or "").strip() or None
    if cluster_lc_path:
        return cluster_lc_path, local_lc_path

    # LTV standalone imports may only have the original raw-lc path in lc_path.
    # If it cannot be resolved locally, show it as the source/cluster path rather
    # than mislabeling it as a usable local review copy.
    if local_lc_path is None and stored_lc_text:
        return stored_lc_text, None

    return None, local_lc_path


def _plot_asset_root() -> Path:
    """Root used for locating and serving static plot files."""
    plot_dir = _configured_plot_dir()
    return plot_dir if plot_dir is not None else _project_root()


def _plot_url_for_path(plot_path: Path) -> str:
    """Return a `/plots/...` URL for a discovered plot path."""
    root = _plot_asset_root().resolve()
    candidate = Path(plot_path).expanduser().resolve()
    try:
        rel_path = candidate.relative_to(root)
    except ValueError:
        return ""
    return f"/plots/{rel_path.as_posix()}"


def _plot_file_from_src(src: str) -> Path | None:
    """Resolve a static plot URL back to an on-disk file path."""
    text = str(src or "")
    if not text.startswith('/plots/'):
        return None

    rel = text[len('/plots/'):]
    suffix = Path(rel).suffix.lower()
    if suffix not in _PLOT_STATIC_EXTENSIONS:
        return None

    root = _plot_asset_root().resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _plot_search_root_for_payload(payload: dict | None) -> Path | None:
    """Return the best plot directory to search for a candidate payload."""
    plot_dir = _configured_plot_dir()
    if plot_dir is not None:
        return plot_dir

    for key in ("plot_path", "png_path", "path", "lc_path"):
        raw_path = (payload or {}).get(key)
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        try:
            candidate = candidate.resolve()
        except Exception:
            pass

        if candidate.suffix.lower() in _PLOT_STATIC_EXTENSIONS and candidate.exists():
            return candidate.parent

        for parent in candidate.parents:
            run_dir = _resolve_run_dir_from_plot_dir(str(parent))
            if run_dir is None:
                continue
            plot_candidate = run_dir / "plots"
            if plot_candidate.is_dir():
                return plot_candidate

    return None


def _candidate_plot_src(payload: dict | None) -> str:
    """Return a static plot URL for *payload*, if one can be located."""
    plot_root = _plot_search_root_for_payload(payload)
    if plot_root is None:
        return ""

    plot_path = find_plot_image(payload or {}, plot_root)
    if plot_path and plot_path.exists():
        return _plot_url_for_path(plot_path)
    return ""


def _count_candidates_in_db(path: Path) -> int:
    """Return number of candidates in DB, or -1 when unavailable."""
    try:
        with closing(sqlite3.connect(str(path), timeout=30.0)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candidates'"
            ).fetchone()
            if not row:
                return -1
            return int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    except Exception:
        return -1


def _resolve_db_cli_path(raw_path: str) -> Path:
    """Resolve --db robustly for both cwd-relative and repo-relative usage."""
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    repo_candidate = (_project_root() / p).resolve()
    existing = [x for x in (cwd_candidate, repo_candidate) if x.exists()]

    if len(existing) == 2:
        ranked = sorted(
            existing,
            key=lambda x: (_count_candidates_in_db(x), x.stat().st_size),
            reverse=True,
        )
        return ranked[0]
    if len(existing) == 1:
        return existing[0]
    return repo_candidate


def _resolve_plot_cli_path(raw_path: str) -> Path:
    """Resolve --plot-dir robustly for both cwd-relative and repo-relative usage."""
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    repo_candidate = (_project_root() / p).resolve()

    if cwd_candidate.exists() and cwd_candidate.is_dir():
        return cwd_candidate
    if repo_candidate.exists() and repo_candidate.is_dir():
        return repo_candidate
    return repo_candidate


def _extract_bundle_scope(path_text: str | None) -> str:
    """Extract output_bundle_* token from a path-like string."""
    if not path_text:
        return ""
    text = str(path_text)
    m = re.search(r"(output_bundle_[^/\\]+)", text)
    return m.group(1) if m else ""


def _split_import_sources(path_text: str | None) -> list[str]:
    """Split import-path text into individual path tokens."""
    if not path_text:
        return []
    tokens: list[str] = []
    for line in str(path_text).replace(";", "\n").splitlines():
        for piece in line.split(","):
            text = piece.strip()
            if text:
                tokens.append(text)
    return tokens


def _candidate_files_for_run_dir(run_dir: Path) -> list[Path]:
    """Return candidate files for a run/results directory, including multi-bin outputs."""
    def _sort_key(path: Path) -> tuple[float, float, str]:
        stem = path.stem
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)$", stem)
        if match:
            return (float(match.group(1)), float(match.group(2)), stem)
        return (float('inf'), float('inf'), stem)

    root = run_dir / "results"
    if run_dir.name == "results":
        root = run_dir
    if not root.exists() or not root.is_dir():
        return []

    exact_vetted = root / "lc_events_vetted.parquet"
    if exact_vetted.exists():
        return [exact_vetted.resolve()]

    tagged_vetted = sorted(root.glob("lc_events_vetted_*.parquet"), key=_sort_key)
    if tagged_vetted:
        return [p.resolve() for p in tagged_vetted]

    fallback_names = (
        "lc_events_spectra.parquet",
        "lc_events_neighbors.parquet",
        "lc_events_classified.parquet",
        "lc_events_enriched.parquet",
        "lc_events_characterized.parquet",
        "lc_events_filtered.parquet",
    )
    matches = [root / name for name in fallback_names if (root / name).exists()]
    return [p.resolve() for p in matches]


def _resolve_import_sources(path_text: str | None, *, allow_run_dirs: bool = True) -> list[Path]:
    """Resolve import-path text into one or more source files."""
    resolved: list[Path] = []
    seen: set[str] = set()
    for token in _split_import_sources(path_text):
        matches: list[Path] = []
        if any(ch in token for ch in "*?[]"):
            matches = [Path(p).expanduser().resolve() for p in sorted(globlib.glob(token, recursive=True))]
        else:
            raw = Path(token).expanduser()
            candidates = [raw]
            if not raw.is_absolute():
                candidates.append((_project_root() / raw).expanduser())
            for candidate in candidates:
                try:
                    if candidate.exists():
                        matches = [candidate.resolve()]
                        break
                except Exception:
                    continue
        if not matches:
            continue
        for match in matches:
            sources = _candidate_files_for_run_dir(match) if (allow_run_dirs and match.is_dir()) else [match]
            for source in sources:
                key = str(source)
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(source)
    return resolved


def _summarize_source_paths(paths: list[Path]) -> str:
    """Return a compact human-readable label for a source list."""
    if not paths:
        return ""
    names = [p.name for p in paths]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, {names[1]}"
    return f"{names[0]}, {names[1]} (+{len(names) - 2} more)"


def _extract_bundle_scopes(path_text: str | None) -> list[str]:
    """Extract all bundle tokens from import text."""
    scopes: list[str] = []
    seen: set[str] = set()
    for token in _split_import_sources(path_text):
        scope = _extract_bundle_scope(token)
        if scope and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    return scopes


def _queue_scope_from_import_text(path_text: str | None) -> object:
    """Build queue scoping metadata from import-path text."""
    paths = _resolve_import_sources(path_text)
    if paths:
        return {
            'source_paths': [str(p) for p in paths],
            'label': _summarize_source_paths(paths),
        }
    scopes = _extract_bundle_scopes(path_text)
    if scopes:
        return {
            'source_path_like_any': scopes,
            'label': ", ".join(scopes),
        }
    return ''


def _source_path_for_queue_filter(path_str: str) -> str:
    """Convert an import path (e.g. run dir or results file) to the value stored in candidates.source_path (run dir)."""
    path_str = str(path_str).strip()
    if not path_str:
        return path_str
    # DB stores run directory; import path may be a file under run_dir/results/
    if "/results/" in path_str:
        return path_str.split("/results/")[0]
    return path_str


def _queue_scope_filter_kwargs(scope_value: object) -> dict[str, object]:
    """Translate queue-source store payload into DB filter kwargs."""
    if isinstance(scope_value, dict):
        if scope_value.get('source_paths'):
            # Normalize so file paths (e.g. .../results/foo.parquet) become run dirs to match candidates.source_path
            normalized = [_source_path_for_queue_filter(p) for p in scope_value['source_paths']]
            return {'source_paths': normalized}
        if scope_value.get('source_path_like_any'):
            return {'source_path_like_any': list(scope_value['source_path_like_any'])}
        return {}
    if scope_value:
        return {'source_path_like': str(scope_value)}
    return {}


def _queue_scope_label(scope_value: object) -> str:
    """Return a short label for queue scoping metadata."""
    if isinstance(scope_value, dict):
        label = scope_value.get('label')
        if label:
            return str(label)
        paths = scope_value.get('source_paths') or []
        if paths:
            return _summarize_source_paths([Path(str(p)) for p in paths])
        likes = scope_value.get('source_path_like_any') or []
        if likes:
            return ", ".join(str(v) for v in likes)
        return ''
    return str(scope_value or '')


def _vetting_mode_for_sources(path_text: str | None) -> str:
    """Summarize vetting-mode status for one or more import sources."""
    paths = _resolve_import_sources(path_text)
    if not paths:
        return _vetting_mode_for_input(path_text)
    modes = [_vetting_mode_for_input(p) for p in paths]
    if len(modes) == 1:
        return modes[0]
    counts: dict[str, int] = {}
    for mode in modes:
        counts[mode] = counts.get(mode, 0) + 1
    return "; ".join(
        f"{count} {mode.lower()}" if count != 1 else mode
        for mode, count in sorted(counts.items(), key=lambda item: item[0])
    )


def _vetting_mode_for_input(input_path: str | Path | None) -> str:
    """Classify how import vetting will be satisfied for a source file."""
    if not input_path:
        return "re-vetting needed"

    p = Path(str(input_path)).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass

    if "vetted" in p.stem.lower():
        return "Using vetted input"

    cache_path = Path(str(p) + ".vetting_cache.parquet")
    if cache_path.exists():
        return "cache hit"

    return "re-vetting needed"


def _load_spectra_rows(candidate_id: str, run_dir: Path | None) -> pd.DataFrame:
    """Load spectra matches for one candidate if local enrichment exists."""
    if run_dir is None:
        return pd.DataFrame()
    spectra_long = run_dir / "results" / "spectra_enrichment" / "spectra_long.parquet"
    if not spectra_long.exists():
        return pd.DataFrame()

    cid = str(candidate_id)
    cols = ["candidate_id", "survey", "catalog", "sep_arcsec"]
    try:
        df = pd.read_parquet(spectra_long, columns=cols, filters=[("candidate_id", "==", cid)])
    except Exception:
        try:
            df = pd.read_parquet(spectra_long, columns=cols)
        except Exception:
            return pd.DataFrame()
        if "candidate_id" in df.columns:
            df = df[df["candidate_id"].astype(str) == cid]
    return df.reset_index(drop=True)


@lru_cache(maxsize=4)
def _index_neowise_paths(run_dir_text: str) -> dict[str, str]:
    """Index candidate->NEOWISE parquet paths once per run directory."""
    root = Path(run_dir_text) / "results"
    mapping: dict[str, str] = {}
    if not root.exists():
        return mapping
    for p in root.rglob("neowise_lc_*.parquet"):
        cid = p.stem.replace("neowise_lc_", "")
        if cid:
            mapping[cid] = str(p)
    return mapping


def _build_neowise_figure(df_neowise: pd.DataFrame) -> go.Figure:
    """Build a compact NEOWISE light-curve panel."""
    return _build_neowise_figure_with_theme(df_neowise, DEFAULT_THEME)


def _external_followup_theme(theme: str | None) -> dict[str, object]:
    """Theme tokens for external follow-up cards and mini plots."""
    mode = str(theme or DEFAULT_THEME).strip().lower()
    if mode == "white":
        return {
            "card_style": {
                'border': '1px solid #c5d0da',
                'borderRadius': '6px',
                'padding': '8px 10px',
                'background': '#ffffff',
                'color': '#1c2733',
            },
            "muted": '#5a6b7b',
            "error": '#a53a3a',
            "paper_bg": '#ffffff',
            "plot_bg": '#ffffff',
            "font": '#1c2733',
            "grid": 'rgba(104, 128, 149, 0.18)',
            "legend_bg": 'rgba(255, 255, 255, 0.92)',
            "legend_border": 'rgba(120, 140, 158, 0.35)',
        }
    if mode == "gray":
        return {
            "card_style": {
                'border': '1px solid #4c566a',
                'borderRadius': '6px',
                'padding': '8px 10px',
                'background': '#3b4252',
                'color': '#d8dee9',
            },
            "muted": '#aab6c7',
            "error": '#f29f9f',
            "paper_bg": '#2e3440',
            "plot_bg": '#2e3440',
            "font": '#d8dee9',
            "grid": 'rgba(129, 161, 193, 0.15)',
            "legend_bg": 'rgba(59, 66, 82, 0.9)',
            "legend_border": 'rgba(129, 161, 193, 0.3)',
        }
    return {
        "card_style": {
            'border': '1px solid #2a2a2a',
            'borderRadius': '6px',
            'padding': '8px 10px',
            'background': '#0d0d0d',
            'color': '#e0e0e0',
        },
        "muted": '#9fb6cb',
        "error": '#dd8080',
        "paper_bg": '#0d0d0d',
        "plot_bg": '#0d0d0d',
        "font": '#dce8f2',
        "grid": 'rgba(96, 116, 130, 0.22)',
        "legend_bg": 'rgba(13, 13, 13, 0.88)',
        "legend_border": 'rgba(113, 140, 160, 0.3)',
    }


def _convert_external_times_to_review_axis(values, jd_system: str = "mjd") -> np.ndarray:
    """Convert source-native time values into the review axis: JD - JD_OFFSET."""
    t = pd.to_numeric(values, errors="coerce").to_numpy()
    finite_t = t[np.isfinite(t)]
    if jd_system == "mjd":
        if finite_t.size and float(np.nanmedian(finite_t)) > 1_000_000.0:
            jd = t
        else:
            jd = t + MJD_TO_JD
    elif jd_system == "bjd_gaia":
        jd = t + GAIA_TCB_EPOCH_JD
    elif jd_system == "btjd":
        jd = t + TESS_BTJD_OFFSET
    elif jd_system == "bkjd":
        jd = t + KEPLER_BKJD_OFFSET
    else:
        jd = t
    return jd - JD_OFFSET


def _apply_external_figure_layout(
    fig: go.Figure,
    *,
    title: str,
    theme: str,
    yaxis_label: str,
    reverse_y: bool,
    height: int = 240,
) -> go.Figure:
    """Apply a consistent themed layout for external mini plots."""
    spec = _external_followup_theme(theme)
    fig.update_layout(
        height=height,
        margin=dict(l=42, r=10, t=34, b=32),
        title=title,
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.1,
            bgcolor=spec["legend_bg"],
            bordercolor=spec["legend_border"],
            borderwidth=1,
            font=dict(color=spec["font"]),
        ),
        paper_bgcolor=spec["paper_bg"],
        plot_bgcolor=spec["plot_bg"],
        font=dict(color=spec["font"]),
    )
    fig.update_xaxes(title="JD - 2458000", gridcolor=spec["grid"], zeroline=False)
    fig.update_yaxes(
        title=yaxis_label,
        autorange="reversed" if reverse_y else True,
        gridcolor=spec["grid"],
        zeroline=False,
    )
    return fig


def _build_neowise_figure_with_theme(df_neowise: pd.DataFrame, theme: str) -> go.Figure:
    """Build a compact NEOWISE light-curve panel."""
    fig = go.Figure()
    if df_neowise is None or df_neowise.empty:
        return _apply_external_figure_layout(
            fig,
            title="NEOWISE",
            theme=theme,
            yaxis_label="mag",
            reverse_y=True,
            height=220,
        )

    time_col = "mjd" if "mjd" in df_neowise.columns else ("MJD" if "MJD" in df_neowise.columns else None)
    if time_col is None:
        return _apply_external_figure_layout(
            fig,
            title="NEOWISE (missing MJD column)",
            theme=theme,
            yaxis_label="mag",
            reverse_y=True,
            height=220,
        )

    x = _convert_external_times_to_review_axis(df_neowise[time_col], "mjd")
    band_specs = [
        ("W1", "w1mpro", "w1sigmpro", "#4fa3ff"),
        ("W2", "w2mpro", "w2sigmpro", "#ff8c42"),
    ]
    added = 0
    for name, mag_col, err_col, color in band_specs:
        if mag_col not in df_neowise.columns:
            continue
        y = pd.to_numeric(df_neowise[mag_col], errors="coerce")
        good = np.isfinite(x) & np.isfinite(y)
        if not bool(good.any()):
            continue
        err_vals = None
        if err_col in df_neowise.columns:
            ev = pd.to_numeric(df_neowise[err_col], errors="coerce")
            if np.isfinite(ev[good]).any():
                err_vals = ev[good]
        fig.add_trace(
            go.Scattergl(
                x=x[good],
                y=y[good],
                mode="markers",
                name=name,
                marker=dict(size=5, color=color, opacity=0.85),
                error_y=dict(type="data", array=err_vals, visible=err_vals is not None, thickness=0.7),
            )
        )
        added += 1

    _apply_external_figure_layout(
        fig,
        title="NEOWISE Light Curve",
        theme=theme,
        yaxis_label="mag",
        reverse_y=True,
    )
    if added == 0:
        fig.add_annotation(text="No finite W1/W2 points", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


@lru_cache(maxsize=32)
def _index_external_lc_paths(run_dir_text: str, prefix: str) -> dict[str, str]:
    """Index candidate -> external LC parquet paths for a given prefix."""
    root = Path(run_dir_text) / "results"
    return _index_external_lc_paths_from_root(str(root), prefix)


@lru_cache(maxsize=64)
def _index_external_lc_paths_from_root(root_text: str, prefix: str) -> dict[str, str]:
    """Index candidate -> external LC parquet paths for a results root."""
    root = Path(root_text)
    mapping: dict[str, str] = {}
    if not root.exists():
        return mapping
    for p in root.rglob(f"{prefix}_lc_*.parquet"):
        cid = p.stem.replace(f"{prefix}_lc_", "")
        if cid:
            mapping[cid] = str(p)
    return mapping


def _build_external_lc_figure(
    df_lc: pd.DataFrame,
    title: str,
    band_specs: list[tuple[str, str, str, str]],
    time_col: str = "mjd",
    yaxis_label: str = "mag",
    reverse_y: bool = True,
    filter_col: str | None = None,
    source_name: str | None = None,
    theme: str | None = None,
    jd_system: str = "mjd",
) -> go.Figure:
    """Build a compact LC panel for any external source.

    *band_specs* is a list of (band_value, mag_col, err_col, color) tuples.
    When *filter_col* is set, only rows where ``df[filter_col] == band_value``
    are plotted for each band.
    """
    fig = go.Figure()
    if df_lc is None or df_lc.empty:
        return _apply_external_figure_layout(
            fig,
            title=title,
            theme=theme,
            yaxis_label=yaxis_label,
            reverse_y=reverse_y,
            height=220,
        )

    if source_name:
        df_lc = normalize_external_lc_dataframe(source_name, df_lc)
        if df_lc is None or df_lc.empty:
            return _apply_external_figure_layout(
                fig,
                title=title,
                theme=theme,
                yaxis_label=yaxis_label,
                reverse_y=reverse_y,
                height=220,
            )

    # Resolve time column (case-insensitive)
    col_lookup = {c.lower(): c for c in df_lc.columns}
    actual_time_col = col_lookup.get(time_col.lower())
    if actual_time_col is None:
        return _apply_external_figure_layout(
            fig,
            title=f"{title} (missing {time_col})",
            theme=theme,
            yaxis_label=yaxis_label,
            reverse_y=reverse_y,
            height=220,
        )

    added = 0
    for band_value, mag_col, err_col, color in band_specs:
        actual_filter_col = col_lookup.get(filter_col.lower()) if filter_col else None
        actual_mag_col = col_lookup.get(mag_col.lower())
        actual_err_col = col_lookup.get(err_col.lower()) if err_col else None
        # Filter rows for this band if filter_col is specified
        if actual_filter_col:
            subset = df_lc[df_lc[actual_filter_col].astype(str) == band_value]
        else:
            subset = df_lc
        if subset.empty or actual_mag_col is None:
            continue

        x = _convert_external_times_to_review_axis(subset[actual_time_col], jd_system)
        y = pd.to_numeric(subset[actual_mag_col], errors="coerce")
        good = np.isfinite(x) & np.isfinite(y)
        if not bool(good.any()):
            continue
        err_vals = None
        if actual_err_col and actual_err_col in subset.columns:
            ev = pd.to_numeric(subset[actual_err_col], errors="coerce")
            if np.isfinite(ev[good]).any():
                err_vals = ev[good]
        fig.add_trace(
            go.Scattergl(
                x=x[good],
                y=y[good],
                mode="markers",
                name=band_value,
                marker=dict(size=5, color=color, opacity=0.85),
                error_y=dict(type="data", array=err_vals, visible=err_vals is not None, thickness=0.7),
            )
        )
        added += 1

    _apply_external_figure_layout(
        fig,
        title=f"{title} Light Curve",
        theme=theme,
        yaxis_label=yaxis_label,
        reverse_y=reverse_y,
    )
    if added == 0:
        fig.add_annotation(text="No finite data points", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def _coerce_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series)):
        return bool(len(value))
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _candidate_lookup_keys(candidate_id: str, payload: dict) -> list[str]:
    keys = [str(candidate_id)]
    for key in ("candidate_id", "asas_sn_id"):
        v = payload.get(key)
        if v is not None:
            keys.append(str(v))
    path_v = payload.get("path")
    if path_v:
        keys.append(Path(str(path_v)).stem)
    seen = set()
    return [k for k in keys if k and not (k in seen or seen.add(k))]


def _render_external_followup(payload: dict, candidate_id: str, theme: str | None = None) -> list:
    theme_spec = _external_followup_theme(theme)
    card_style = theme_spec["card_style"]
    muted_text_style = {'fontSize': '10px', 'color': theme_spec["muted"]}
    error_text_style = {'fontSize': '10px', 'color': theme_spec["error"]}
    run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
    lookup_keys = _candidate_lookup_keys(candidate_id, payload)

    # Spectra
    has_spectrum = _coerce_bool(payload.get('has_spectrum'))
    spectrum_sources = str(payload.get('spectrum_sources') or '').strip()
    spectrum_links_raw = str(payload.get('spectrum_links') or '').strip()
    spectrum_links = [x.strip() for x in spectrum_links_raw.replace(';', ',').split(',') if x.strip()]
    spectra_rows = pd.DataFrame()
    for key in lookup_keys:
        spectra_rows = _load_spectra_rows(key, run_dir)
        if not spectra_rows.empty:
            break

    spectra_children = [
        html.Div(f"Has spectra: {'yes' if has_spectrum else 'no'}", style={'fontSize': '11px'}),
        html.Div(f"Sources: {spectrum_sources or 'none'}", style={'fontSize': '11px', 'color': theme_spec["muted"]}),
    ]
    if not spectra_rows.empty:
        spectra_rows = spectra_rows.head(8)
        hdr = html.Tr([html.Th('survey'), html.Th('catalog'), html.Th('sep\"')])
        body = [
            html.Tr([
                html.Td(str(r.get('survey', ''))),
                html.Td(str(r.get('catalog', ''))),
                html.Td(f"{float(r.get('sep_arcsec')):.2f}" if pd.notna(r.get('sep_arcsec')) else ''),
            ])
            for _, r in spectra_rows.iterrows()
        ]
        spectra_children.append(html.Table([html.Thead(hdr), html.Tbody(body)], style={'width': '100%', 'fontSize': '10px'}))
    if spectrum_links:
        spectra_children.append(
            html.Div([
                html.Div('Links:', style={'fontSize': '11px', 'marginTop': '4px'}),
                html.Div([
                    html.A(
                        link,
                        href=link,
                        target='_blank',
                        rel='noopener noreferrer',
                        style={'display': 'block', 'fontSize': '10px', 'color': theme_spec["muted"]},
                    )
                    for link in spectrum_links
                ]),
            ])
        )

    spectra_card = html.Div([
        html.Div('Spectra', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *spectra_children,
    ], style=card_style)

    # ATLAS summary + optional light curve panel
    atlas_children = [
        html.Div(f"Photometry: {'yes' if _coerce_bool(payload.get('atlas_has_phot')) else 'no'}", style={'fontSize': '11px'}),
        html.Div(f"cyan n/range: {payload.get('atlas_n_det_cyan', 'n/a')} / {payload.get('atlas_cyan_range', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"orange n/range: {payload.get('atlas_n_det_orange', 'n/a')} / {payload.get('atlas_orange_range', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if run_dir is not None:
        atlas_idx = _index_external_lc_paths(str(run_dir.resolve()), "atlas")
        for key in lookup_keys:
            path_str = atlas_idx.get(str(key))
            if path_str:
                atlas_path = Path(path_str)
                if atlas_path.exists():
                    try:
                        atlas_lc = pd.read_parquet(atlas_path)
                        atlas_fig = _build_external_lc_figure(
                            atlas_lc, "ATLAS",
                            [("c", "mag", "mag_err", "#00ccff"),
                             ("o", "mag", "mag_err", "#ff8c42")],
                            time_col="mjd",
                            filter_col="filter",
                            source_name="atlas",
                            theme=theme,
                            jd_system="mjd",
                        )
                        atlas_children.append(dcc.Graph(figure=atlas_fig, mathjax=True, config={'displayModeBar': False}, style={'height': '250px'}))
                    except Exception:
                        pass
                break

    atlas_card = html.Div([
        html.Div('ATLAS', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *atlas_children,
    ], style=card_style)

    # NEOWISE summary + optional light curve panel
    neowise_epochs = payload.get('neowise_n_epochs', 0)
    neowise_rows = pd.DataFrame()
    neowise_path = None
    if run_dir is not None:
        idx_map = _index_neowise_paths(str(run_dir.resolve()))
        for key in lookup_keys:
            path_str = idx_map.get(str(key))
            if path_str:
                neowise_path = Path(path_str)
                break
    neowise_plot = None
    if neowise_path and neowise_path.exists():
        try:
            neowise_rows = pd.read_parquet(neowise_path)
            neowise_plot = dcc.Graph(
                figure=_build_neowise_figure_with_theme(neowise_rows, theme),
                mathjax=True,
                config={'displayModeBar': False},
                style={'height': '250px'},
            )
        except Exception:
            neowise_plot = html.Div(f"Could not load NEOWISE parquet: {neowise_path}", style=error_text_style)

    neowise_children = [
        html.Div(f"Epochs: {neowise_epochs}", style={'fontSize': '11px'}),
        html.Div(f"W1 range: {payload.get('neowise_w1_range', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"W2 range: {payload.get('neowise_w2_range', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if neowise_path:
        neowise_children.append(html.Div(f"File: {neowise_path.name}", style=muted_text_style))
    if neowise_plot is not None:
        neowise_children.append(neowise_plot)

    neowise_card = html.Div([
        html.Div('NEOWISE', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *neowise_children,
    ], style=card_style)

    # ZTF LC card
    ztf_children = [
        html.Div(f"Detections: {payload.get('ztf_lc_n_det', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"g range: {payload.get('ztf_lc_g_range', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"r range: {payload.get('ztf_lc_r_range', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if run_dir is not None:
        ztf_idx = _index_external_lc_paths(str(run_dir.resolve()), "ztf")
        for key in lookup_keys:
            path_str = ztf_idx.get(str(key))
            if path_str:
                ztf_path = Path(path_str)
                if ztf_path.exists():
                    try:
                        ztf_lc = pd.read_parquet(ztf_path)
                        ztf_fig = _build_external_lc_figure(
                            ztf_lc, "ZTF",
                            [("zg", "mag", "mag_err", "#44aa44"),
                             ("zr", "mag", "mag_err", "#dd4444"),
                             ("zi", "mag", "mag_err", "#8844cc")],
                            time_col="mjd",
                            filter_col="band",
                            source_name="ztf",
                            theme=theme,
                            jd_system="mjd",
                        )
                        ztf_children.append(dcc.Graph(figure=ztf_fig, mathjax=True, config={'displayModeBar': False}, style={'height': '250px'}))
                    except Exception:
                        pass
                break

    ztf_card = html.Div([
        html.Div('ZTF', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *ztf_children,
    ], style=card_style)

    # Gaia epoch LC card
    gaia_epoch_children = [
        html.Div(f"G points: {payload.get('gaia_epoch_lc_n_g', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"G range: {payload.get('gaia_epoch_lc_g_range', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if run_dir is not None:
        gaia_idx = _index_external_lc_paths(str(run_dir.resolve()), "gaia_epoch")
        for key in lookup_keys:
            path_str = gaia_idx.get(str(key))
            if path_str:
                gaia_path = Path(path_str)
                if gaia_path.exists():
                    try:
                        gaia_lc = pd.read_parquet(gaia_path)
                        gaia_fig = _build_external_lc_figure(
                            gaia_lc, "Gaia Epoch",
                            [("G", "mag", "mag_err", "#e8c547")],
                            time_col="time",
                            yaxis_label="G mag",
                            source_name="gaia_epoch",
                            theme=theme,
                            jd_system="bjd_gaia",
                        )
                        gaia_epoch_children.append(dcc.Graph(figure=gaia_fig, mathjax=True, config={'displayModeBar': False}, style={'height': '250px'}))
                    except Exception:
                        pass
                break

    gaia_epoch_card = html.Div([
        html.Div('Gaia Epoch', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *gaia_epoch_children,
    ], style=card_style)

    # Pan-STARRS LC card
    ps1_children = [
        html.Div(f"Points: {payload.get('ps1_lc_n_points', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if run_dir is not None:
        ps1_idx = _index_external_lc_paths(str(run_dir.resolve()), "ps1")
        for key in lookup_keys:
            path_str = ps1_idx.get(str(key))
            if path_str:
                ps1_path = Path(path_str)
                if ps1_path.exists():
                    try:
                        ps1_lc = pd.read_parquet(ps1_path)
                        ps1_fig = _build_external_lc_figure(
                            ps1_lc, "Pan-STARRS",
                            [("g_ps", "mag", "mag_err", "#44aa44"),
                             ("r_ps", "mag", "mag_err", "#dd4444"),
                             ("i_ps", "mag", "mag_err", "#8844cc"),
                             ("z_ps", "mag", "mag_err", "#ccaa44"),
                             ("y_ps", "mag", "mag_err", "#aaaa33")],
                            time_col="mjd",
                            filter_col="filter",
                            source_name="ps1",
                            theme=theme,
                            jd_system="mjd",
                        )
                        ps1_children.append(dcc.Graph(figure=ps1_fig, mathjax=True, config={'displayModeBar': False}, style={'height': '250px'}))
                    except Exception:
                        pass
                break

    ps1_card = html.Div([
        html.Div('Pan-STARRS', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *ps1_children,
    ], style=card_style)

    # CRTS LC card
    crts_children = [
        html.Div(f"Points: {payload.get('crts_lc_n_points', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if run_dir is not None:
        crts_idx = _index_external_lc_paths(str(run_dir.resolve()), "crts")
        for key in lookup_keys:
            path_str = crts_idx.get(str(key))
            if path_str:
                crts_path = Path(path_str)
                if crts_path.exists():
                    try:
                        crts_lc = pd.read_parquet(crts_path)
                        crts_fig = _build_external_lc_figure(
                            crts_lc, "CRTS",
                            [("CV", "mag", "mag_err", "#bbbbbb")],
                            time_col="mjd",
                            source_name="crts",
                            theme=theme,
                            jd_system="mjd",
                        )
                        crts_children.append(dcc.Graph(figure=crts_fig, mathjax=True, config={'displayModeBar': False}, style={'height': '250px'}))
                    except Exception:
                        pass
                break

    crts_card = html.Div([
        html.Div('CRTS', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *crts_children,
    ], style=card_style)

    return [spectra_card, atlas_card, neowise_card, ztf_card, gaia_epoch_card, ps1_card, crts_card]


# ---- sidebar filter helpers ------------------------------------------------
_ATF_OPTS = [
    {'label': 'Any', 'value': 'Any'},
    {'label': 'True', 'value': 'True'},
    {'label': 'False', 'value': 'False'},
    {'label': 'Unset', 'value': 'Unset'},
]
_inp_style = {'width': '100%', 'margin-bottom': '4px', 'font-size': '11px'}


def _bool_mode_filter(label: str, component_id: str):
    """Return a (Label, Dropdown) pair for Any/True/False/Unset bool filter."""
    return [
        html.Label(f'{label}:'),
        dcc.Dropdown(
            id=component_id,
            options=_ATF_OPTS,
            value='Any',
            clearable=False,
            style={'margin-bottom': '4px', 'font-size': '11px'},
            persistence=_review_persistence_token(),
            persistence_type='local',
        ),
    ]


def _col_id(col: str) -> str:
    """snake_case → dash-case for Dash component IDs."""
    return col.replace('_', '-')


def _select_all_dropdown_values(options: list[dict[str, object]] | None) -> list[str]:
    """Return all distinct option values for a multi-select dropdown."""
    values: list[str] = []
    seen: set[str] = set()
    for option in options or []:
        if not isinstance(option, dict):
            continue
        raw_value = option.get('value')
        if raw_value is None:
            continue
        value = str(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _vetting_known_filter_preset(
    select_options: dict[str, list[dict[str, object]] | None],
) -> tuple[list[str], list[list[str]]]:
    """Exclude definite known-type vetting matches while leaving uncertainty flags untouched."""
    bool_values = ['False'] * len(VETTING_KNOWN_BOOL_FILTERS)
    select_values = [
        _select_all_dropdown_values(select_options.get(col))
        for col in VETTING_KNOWN_SELECT_FILTERS
    ]
    return bool_values, select_values


def _num_range_filter(col: str):
    """Numeric filter with min/max inputs and a slider."""
    return html.Div([
        html.Label(f'{col}:'),
        html.Div([
            dcc.Input(
                id={'type': 'num-filter-min-input', 'col': col},
                type='number',
                placeholder='min',
                debounce=True,
                persistence=_review_persistence_token(),
                persistence_type='local',
                style={'width': '72px', 'font-size': '11px', 'flex': '0 0 72px'},
            ),
            dcc.RangeSlider(
                id={'type': 'num-filter-range', 'col': col},
                min=0,
                max=1,
                value=[0, 1],
                step=0.01,
                allowCross=False,
                marks=None,
                tooltip={'placement': 'bottom', 'always_visible': False},
                updatemode='mouseup',
                disabled=True,
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            dcc.Input(
                id={'type': 'num-filter-max-input', 'col': col},
                type='number',
                placeholder='max',
                debounce=True,
                persistence=_review_persistence_token(),
                persistence_type='local',
                style={'width': '72px', 'font-size': '11px', 'flex': '0 0 72px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'margin-bottom': '4px'}),
    ])


def _text_filter(col: str):
    """Dropdown for exact-match string filter (options hydrated lazily)."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col}:'),
        dcc.Dropdown(
            id=f'filter-{cid}',
            options=[{'label': 'Any', 'value': 'Any'}],
            value='Any',
            clearable=False,
            placeholder='Open sidebar to load options',
            style={'margin-bottom': '4px', 'font-size': '11px'},
            persistence=_review_persistence_token(), persistence_type='local'
        ),
    ])


def _select_filter(col: str):
    """Multi-select dropdown for exclude filtering (options hydrated lazily)."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col} (exclude):'),
        dcc.Dropdown(
            id=f'exclude-{cid}', options=[], multi=True,
            placeholder='None excluded',
            style={'margin-bottom': '4px', 'font-size': '11px'},
            maxHeight=400,
            optionHeight=28,
            persistence=_review_persistence_token(),
            persistence_type='local',
        ),
    ])


def _make_filter_group(name: str, items: list, *, default_open: bool = False):
    """Build a collapsible html.Details for a group of filters."""
    children = []
    if name == 'Vetting':
        children.append(
            html.Div([
                html.Button(
                    'Exclude Known Types',
                    id='vetting-known-types-btn',
                    n_clicks=0,
                    className='compact-btn',
                    title='Turn on definite known-type vetting filters and leave uncertainty-style flags untouched.',
                ),
            ], style={'display': 'flex', 'margin-bottom': '6px'})
        )
    for ftype, col in items:
        if ftype == 'bool':
            children.extend(_bool_mode_filter(col, f'{_col_id(col)}-mode'))
        elif ftype == 'num':
            children.append(_num_range_filter(col))
        elif ftype == 'text':
            children.append(_text_filter(col))
        elif ftype == 'select':
            children.append(_select_filter(col))
    block = [
        html.Summary(name),
        html.Div(children, style={'padding-left': '6px'}),
    ]
    if default_open:
        return html.Details(block, open='open')
    return html.Details(block)


# ---------------------------------------------------------------------------
# Sidebar filter groups — single source of truth for filter UI + state lists
# Each item: ('bool', col_name) | ('num', col_name) | ('text', col_name)
# ---------------------------------------------------------------------------
_SIDEBAR_GROUPS = list(REVIEW_FILTER_SIDEBAR_GROUPS)


def create_layout():
    """Create app layout."""
    return html.Div([
        # Hidden keyboard input
        dcc.Input(
            id='keyboard-input',
            type='text',
            autoFocus=True,
            autoComplete='off',
            style={
                'position': 'absolute',
                'left': '-9999px',      # Off-screen instead of opacity
                'width': '1px',
                'height': '1px',
                'opacity': 0,
                'zIndex': -1,           # Behind everything
                'pointerEvents': 'none' # Don't capture clicks
            }
        ),

        # Data stores
        dcc.Store(id='queue-data'),
        dcc.Store(id='review-db-scope', data=_review_persistence_token()),
        dcc.Store(id='current-index', data=0),
        dcc.Store(id='preload-trigger', data=0),
        dcc.Store(id='review-save-request', data={'nonce': 0}),
        dcc.Store(id='current-candidate-id', data=None),
        dcc.Store(id='queue-size-store', data=0),
        dcc.Store(id='queue-filter-hash-store', data=''),
        dcc.Store(id='current-score', data=None),
        dcc.Store(id='event-class-store', data='unclassified'),
        dcc.Store(id='pending-prefix', data=''),  # kept for callback compatibility
        dcc.Store(id='needs-followup-store', data=False),
        dcc.Store(id='review-pass-store', data=1),
        dcc.Store(id='sidebar-state', data=False),  # collapsed by default
        dcc.Store(id='filter-params', data={}),
        dcc.Store(id='restored-filter-applied', data=0),
        dcc.Store(id='saved-review-gui-state', data=None),
        dcc.Store(id='import-trigger', data=0),  # triggers queue refresh after import
        dcc.Store(id='auto-run-pipeline-trigger', data=None),
        dcc.Store(id='pending-auto-run', data=None),
        dcc.Store(id='pipeline-progress-trigger', data=0),
        dcc.Store(id='pipeline-module-log', data={'lines': []}),
        dcc.Store(id='cone-results-data', data=None),  # cone search catalog rows
        dcc.Store(id='numeric-filter-bounds', data={}),
        dcc.Store(id='diagnostic-background-state', data={'signature': '', 'ready': False, 'cached': False, 'token': 0}),
        dcc.Store(id='auto-period-cache', data={}, storage_type='session'),
        dcc.Store(id='auto-period-request', data={'nonce': 0}),
        dcc.Store(id='plot-render-request', data={'nonce': 1, 'ts': 0.0, 'state': {'idx': 0, 'candidate_id': None, 'plot_mode': 'native', 'overlay_values': list(PLOT_PRESETS['Diagnostics']['overlays']), 'selected_cameras': [], 'selected_bands': ['g', 'V'], 'preset': 'Diagnostics', 'theme': DEFAULT_THEME, 'residual_height': DEFAULT_RESIDUAL_FRACTION, 'baseline_opacity': 0.5, 'external_source_view': DEFAULT_EXTERNAL_SOURCE_VIEW}}),
        dcc.Store(id='plot-render-applied', data=0),
        dcc.Store(id='plot-defaults-initialized', data=False),
        dcc.Store(id='queue-source-path', data=''),
        dcc.Store(id='run-config-json-store', data=''),
        dcc.Store(id='theme-mode-store', data=DEFAULT_THEME),
        dcc.Store(id='review-session-start', data=None, storage_type='session'),
        dcc.Store(id='metadata-resize-init', data=0),
        dcc.Store(id='status-resize-init', data=0),
        dcc.Store(id='sidebar-plot-saved', data=0),  # dummy sink for plot prefs save callback
        dcc.Store(id='candidate-start-time', data=0),
        dcc.Store(id='review-progress-state', data={'reviewed': 0, 'total': 0}),
        dcc.Store(id='startup-selection-applied', data=False),
        dcc.Store(id='last-candidate-saved', data=0),
        dcc.Download(id='plot-export-download'),
        dcc.Download(id='run-config-download'),
        dcc.Interval(id='keyboard-init', interval=200, n_intervals=0, max_intervals=1),
        dcc.Interval(id='review-metrics-interval', interval=1000, n_intervals=0),

        # Sidebar toggle button
        html.Button('☰', id='sidebar-toggle', className='sidebar-toggle', title='Toggle sidebar [Esc]', n_clicks=0),

        # Collapsible sidebar
        html.Div([
            html.Div('Filters', className='section-title'),
            html.Div([
                html.Button('Refresh Slider Bounds', id='refresh-filter-bounds-btn', n_clicks=0, className='compact-btn'),
                html.Button('Reset Numeric Filters', id='reset-numeric-filters-btn', n_clicks=0, className='compact-btn'),
                html.Span('Sliders load on startup; use refresh to rebuild bounds.', id='numeric-bounds-status', style={'fontSize': '10px', 'color': '#7d91a6'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '6px'}),
            html.Div('Dropdown options load when the sidebar opens.', id='filter-load-status', style={'fontSize': '10px', 'color': '#7d91a6', 'marginBottom': '6px'}),

            dcc.Checklist(
                id='filter-unreviewed',
                options=[{'label': ' Only unreviewed', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '3px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            dcc.Checklist(
                id='filter-failed',
                options=[{'label': ' Require failed_any=False', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),

            # -- All filter groups (auto-generated from _SIDEBAR_GROUPS) --
            *[_make_filter_group(name, items, default_open=(name == 'General Flags'))
              for name, items in _SIDEBAR_GROUPS],

            # -- Sorting --
            html.Label('Sort by:'),
            dcc.Dropdown(
                id='sort-col',
                options=(
                    [{'label': 'Candidate ID', 'value': 'candidate_id'}]
                    + [{'label': col, 'value': col}
                       for _, items in _SIDEBAR_GROUPS
                       for ftype, col in items if ftype == 'num']
                     + [{'label': 'Confidence', 'value': 'interest_score'},
                        {'label': 'Review Pass', 'value': 'review_pass'},
                        {'label': 'Updated At', 'value': 'updated_at'}]
                ),
                value=['candidate_id'],
                multi=True,
                clearable=False,
                style={'margin-bottom': '4px', 'font-size': '11px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            dcc.Checklist(
                id='sort-desc',
                options=[{'label': ' Descending', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),

            html.Button('Refresh Queue [R]', id='refresh-btn', n_clicks=0,
                       style={'width': '100%'}, className='action-btn queue-refresh-btn'),

            html.Button('↻ Reset to Beginning', id='reset-queue-btn', n_clicks=0,
                       style={'width': '100%', 'font-size': '11px', 'marginTop': '4px'}, className='action-btn'),
            html.Button('Save GUI State', id='save-review-gui-state-btn', n_clicks=0,
                       style={'width': '100%', 'font-size': '11px', 'marginTop': '4px'}, className='action-btn'),
            html.Div(id='save-review-gui-state-status', style={'fontSize': '10px', 'color': '#7da8c4', 'marginTop': '4px'}),

            html.Div('Open Existing', className='section-title', style={'margin-top': '8px'}),
            dcc.Input(
                id='candidate-search-query',
                placeholder='candidate / ASAS-SN / Gaia / LC stem',
                type='text',
                style={**_inp_style, 'marginBottom': '4px'},
            ),
            html.Button(
                'Search / View',
                id='candidate-search-btn',
                n_clicks=0,
                className='action-btn',
                style={'width': '100%'},
            ),

            html.Div('Explore', className='section-title', style={'margin-top': '8px'}),
            html.Div([
                html.Button('Open Candidate In Explorer', id='open-candidate-in-explorer-btn', n_clicks=0,
                            className='action-btn', style={'flex': '1 1 0'}),
                html.Button('Open DB In Explorer', id='open-db-in-explorer-btn', n_clicks=0,
                            className='action-btn', style={'flex': '1 1 0'}),
            ], style={'display': 'flex', 'gap': '6px', 'marginBottom': '6px'}),
            html.Div(id='explorer-launch-status', style={'fontSize': '10px', 'color': '#7da8c4', 'marginBottom': '6px'}),
            html.Details([
                html.Summary('Explorer Selection'),
                html.Div(id='explorer-selection-panel', style={'fontSize': '10px', 'marginTop': '4px'}),
            ], open=False),

            html.Hr(),

            html.Div('Native Cameras', className='section-title'),
            html.Div([
                html.Button('All', id='cams-all-btn', n_clicks=0, className='action-btn'),
                html.Button('Clear', id='cams-clear-btn', n_clicks=0, className='action-btn'),
                html.Button('Invert', id='cams-invert-btn', n_clicks=0, className='action-btn'),
            ], className='sidebar-camera-actions'),
            dcc.Checklist(
                id='camera-checklist',
                options=[],
                value=[],
                className='sidebar-camera-checklist',
                style={'margin-bottom': '6px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),

            html.Div('Native Bands', className='section-title'),
            dcc.Checklist(
                id='band-checklist',
                options=NATIVE_BAND_OPTIONS,
                value=['g', 'V'],
                className='sidebar-camera-checklist',
                style={'margin-bottom': '6px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),

            html.Hr(),

            # -- Fetch Candidate --
            html.Div('Fetch Candidate', className='section-title'),
            dcc.Dropdown(
                id='fetch-backend',
                options=FETCH_BACKEND_OPTIONS,
                value=DEFAULT_FETCH_BACKEND,
                clearable=False,
                style={'margin-bottom': '4px', 'font-size': '11px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            dcc.Dropdown(
                id='fetch-type',
                options=[
                    {'label': 'ASAS-SN ID', 'value': 'asassn'},
                    {'label': 'Gaia DR3 ID', 'value': 'gaia'},
                    {'label': 'Coords (RA Dec)', 'value': 'coords'},
                ],
                value='asassn',
                clearable=False,
                style={'margin-bottom': '4px', 'font-size': '11px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            dcc.Input(id='fetch-query', placeholder='ID or RA Dec [radius_arcsec]', type='text',
                      style={**_inp_style, 'marginBottom': '4px'},
                      persistence=_review_persistence_token(), persistence_type='local'),
            dcc.Dropdown(
                id='fetch-mode',
                options=[
                    {'label': 'Quick (LC only)', 'value': 'quick'},
                    {'label': 'Full (analyze + vet)', 'value': 'full'},
                    {'label': 'Full + External LCs', 'value': 'full_ext'},
                ],
                value='quick',
                clearable=False,
                style={'margin-bottom': '4px', 'font-size': '11px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            html.Button('Fetch', id='fetch-btn', n_clicks=0,
                        className='action-btn',
                        style={'width': '100%'}),
            dcc.Loading(
                id='loading-fetch', type='dot',
                children=html.Div(id='fetch-status',
                                  style={'fontSize': '11px', 'marginTop': '4px', 'color': '#7da8c4'}),
            ),
            html.Div(id='cone-results-container', style={'fontSize': '10px', 'marginTop': '4px'}),

            html.Hr(),

            # -- Import --
            html.Div('Import', className='section-title'),
            dcc.Input(id='import-path', placeholder='Candidates file path', type='text',
                     style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
            html.Div('Tip: use one path, a run directory, a glob, or multiple newline-separated files.',
                     style={'fontSize': '10px', 'color': '#7d91a6', 'margin': '4px 0 6px'}),

            dcc.Checklist(
                id='import-lc-mode',
                options=[{'label': ' Raw light curve file (CSV/parquet with JD, mag columns)', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '4px', 'fontSize': '10px'},
                persistence=_review_persistence_token(), persistence_type='local',
            ),

            html.Label('Characterize on import:'),
            dcc.Checklist(
                id='characterize-on-import',
                options=[{'label': ' Enable', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(), persistence_type='local',
            ),

            dcc.Input(id='characterize-crossmatch', placeholder='Crossmatch CSV', type='text',
                     value=str(VSX_CROSSMATCH_PATH), style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
            dcc.Input(id='characterize-gaia-cache', placeholder='Gaia cache', type='text',
                     value=str(GAIA_CACHE_FILE), style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
            dcc.Input(id='characterize-chunk-size', placeholder='Chunk size', type='number',
                     value=GAIA_CHUNK_SIZE, style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
            dcc.Checklist(
                id='characterize-dust',
                options=[{'label': ' Enable dustmaps3d', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(), persistence_type='local',
            ),
            dcc.Input(id='characterize-starhorse', placeholder='StarHorse (tap or path)', type='text',
                     value='tap', style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),

            html.Label('Vet on import:'),
            dcc.Checklist(
                id='vet-on-import',
                options=[{'label': ' Enable', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(), persistence_type='local',
            ),

            html.Button('Import', id='import-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%'}),

            html.Hr(),

            html.Div('Export', className='section-title'),
            dcc.Input(id='export-path', placeholder='Export file path', type='text',
                     value='output/review/reviewed_candidates.parquet', style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
            dcc.Checklist(
                id='export-only-reviewed',
                options=[{'label': ' Only reviewed', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            html.Button('Export Reviews', id='export-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%'}),

            html.Hr(),

            html.Div('Merge Reviews', className='section-title'),
            dcc.Input(
                id='merge-target-db-path',
                placeholder='Target review DB path',
                type='text',
                value='output/review/review.db',
                style=_inp_style,
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            html.Div(
                'Merge this DB into the target DB. If both DBs reviewed the same candidate, the newer updated_at wins.',
                style={'fontSize': '10px', 'color': '#7d91a6', 'margin': '4px 0 6px'},
            ),
            html.Button('Merge Into Target DB', id='merge-review-db-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%'}),

            html.Hr(),
            
            html.Div('External Links', className='section-title'),
            html.Label('Search Radius (arcsec):'),
            dcc.Input(id='link-radius-arcsec', placeholder='Radius (arcsec)', type='number',
                     value=10.0, min=0.1, step=1.0, style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
                     
            html.Hr(),

            html.Div('Pace Timer', className='section-title'),
            dcc.Checklist(
                id='pace-timer-toggle',
                options=[{'label': ' Show Timer', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),

            html.Div('Theme', className='section-title'),
            dcc.RadioItems(
                id='theme-mode',
                options=[
                    {'label': ' Black', 'value': 'black'},
                    {'label': ' Gray', 'value': 'gray'},
                    {'label': ' White', 'value': 'white'},
                ],
                value=DEFAULT_THEME,
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),

            html.Div(id='sidebar-status', style={'margin-top': '10px', 'color': '#0f0', 'font-size': '11px'}),
            html.Div('Queue Filter Provenance', className='section-title', style={'margin-top': '10px'}),
            html.Div(id='queue-filter-provenance-panel'),

        ], id='sidebar', className='sidebar'),

        # Main content
        html.Div([
            # Header bar
            html.Div([
                html.Span(id='progress-text', style={'color': '#0af', 'font-size': '11px'}),
                html.Span(id='review-progress-indicator', style={'color': '#9fb6cb', 'font-size': '11px', 'margin-left': '10px'}),
                html.Span(id='pace-timer-display', style={'display': 'none'}),  # hidden placeholder
                html.Div([
                    html.Span(id='header-asas-sn-id', className='item'),
                    html.Span(id='header-gaia-id', className='item'),
                ], className='header-key-info'),
                html.Span(id='notification', className='notification'),
                html.A('[?] Shortcuts', id='help-link', className='help-link'),
            ], className='header-bar'),

            # Split workspace: info left, light curve right
            html.Div([
                html.Div([
                    html.Div([
                        html.Details([
                            html.Summary('Plot Controls'),
                            html.Div([
                                dcc.Dropdown(
                                    id='plot-preset',
                                    options=[{'label': p, 'value': p} for p in ('Clean', 'Diagnostics', 'Full')],
                                    value='Diagnostics',
                                    clearable=False,
                                    style={'minWidth': '140px', 'font-size': '10px'},
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                dcc.RadioItems(
                                    id='plot-mode',
                                    options=[
                                        {'label': ' Native', 'value': 'native'},
                                        {'label': ' PNG', 'value': 'png'},
                                    ],
                                    value='native',
                                    inline=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                dcc.Checklist(
                                    id='plot-overlays',
                                    options=[
                                        {'label': ' Raw lightcurve', 'value': 'raw'},
                                        {'label': ' Dip/Jump markers', 'value': 'markers'},
                                        {'label': ' Residual panel', 'value': 'residuals'},
                                        {'label': ' Phase-fold panel', 'value': 'phase'},
                                        {'label': ' Filter bad cameras', 'value': 'filter_bad_cameras'},
                                        {'label': ' Event diagnostics', 'value': 'diagnostics'},
                                        {'label': ' Confidence colors', 'value': 'confidence'},
                                    ],
                                    value=list(PLOT_PRESETS['Diagnostics']['overlays']),
                                    inline=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                dcc.RadioItems(
                                    id='yaxis-mode',
                                    options=[
                                        {'label': ' Mag', 'value': 'mag'},
                                        {'label': ' Flux', 'value': 'flux'},
                                    ],
                                    value='mag',
                                    inline=True,
                                    style={'font-size': '10px', 'margin-left': '8px'},
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                html.Div([
                                    html.Span('Baseline', style={'color': '#9fb6cb', 'font-size': '10px',
                                                                  'white-space': 'nowrap'}),
                                    dcc.Slider(
                                        id='baseline-opacity-slider',
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=0.5,
                                        marks=None,
                                        tooltip={'placement': 'bottom', 'always_visible': False},
                                        updatemode='drag',
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                ], className='toolbar-slider-control'),
                                html.Div([
                                    html.Span('Residual', style={'color': '#9fb6cb', 'font-size': '10px',
                                                                  'white-space': 'nowrap'}),
                                    dcc.Slider(
                                        id='residual-height-slider',
                                        min=0.15,
                                        max=0.85,
                                        step=0.01,
                                        value=DEFAULT_RESIDUAL_FRACTION,
                                        marks=None,
                                        tooltip={'placement': 'bottom', 'always_visible': False},
                                        updatemode='drag',
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                ], className='toolbar-slider-control'),
                                 html.Button('Reset', id='plot-reset-btn', n_clicks=0, className='compact-btn'),
                                 html.Button('Export', id='export-plot', n_clicks=0, className='compact-btn'),
                                 html.Span(id='plot-render-indicator', style={'color': '#7da8c4', 'font-size': '10px', 'margin-left': '6px'}),
                                 html.Span(id='repro-badge', className='label-chip', style={'margin-left': '6px'}),
                                html.Div([
                                    html.Span('LC Source', style={'color': '#9fb6cb', 'font-size': '10px',
                                                                  'white-space': 'nowrap', 'margin-right': '4px'}),
                                    dbc.Select(
                                        id='external-source-view',
                                         options=EXTERNAL_SOURCE_VIEW_OPTIONS,
                                         value=DEFAULT_EXTERNAL_SOURCE_VIEW,
                                         size='sm',
                                         style={'width': '140px', 'font-size': '10px', 'minWidth': '140px'},
                                         persistence=_review_persistence_token(),
                                         persistence_type='local',
                                     ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '2px',
                                          'margin-left': '10px', 'border-left': '1px solid #444',
                                          'padding-left': '10px'}),
                                html.Div([
                                    html.Span('Period', style={'color': '#9fb6cb', 'font-size': '10px',
                                                               'white-space': 'nowrap', 'margin-right': '4px'}),
                                    dcc.Dropdown(
                                        id='period-method',
                                        options=[
                                            {'label': 'LSP', 'value': 'lsp'},
                                            {'label': 'PDM', 'value': 'pdm'},
                                            {'label': 'CE', 'value': 'ce'},
                                        ],
                                        value='pdm',
                                        clearable=False,
                                        style={'width': '85px', 'font-size': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    dcc.Input(id='pdm-min-period', type='number', value=0.1, min=0.001,
                                              step='any', debounce=True, placeholder='Min P',
                                              style={'width': '72px', 'font-size': '10px'},
                                              persistence=_review_persistence_token(),
                                              persistence_type='local'),
                                    html.Span('–', style={'color': '#9fb6cb', 'margin': '0 2px', 'font-size': '10px'}),
                                    dcc.Input(id='pdm-max-period', type='number', value=10, min=0.001,
                                              step='any', debounce=True, placeholder='Max P',
                                              style={'width': '72px', 'font-size': '10px'},
                                              persistence=_review_persistence_token(),
                                              persistence_type='local'),
                                    html.Span('d', style={'color': '#9fb6cb', 'font-size': '10px', 'margin-right': '4px'}),
                                     html.Button('Find Period', id='pdm-run-btn', n_clicks=0, className='compact-btn'),
                                     dcc.Input(id='pdm-manual-period', type='number', min=0.001,
                                               step=0.001, placeholder='Manual P (d)',
                                               style={'width': '90px', 'font-size': '10px', 'margin-left': '4px'},
                                               persistence=_review_persistence_token(),
                                               persistence_type='local'),
                                     html.Span(id='pdm-result-label', style={'color': '#7da8c4', 'font-size': '10px',
                                                                              'margin-left': '4px'}),
                                     html.Span(id='period-search-indicator', style={'color': '#9fb6cb', 'font-size': '10px',
                                                                                    'margin-left': '4px'}),
                                 ], style={'display': 'flex', 'alignItems': 'center', 'gap': '2px',
                                           'margin-left': '10px', 'border-left': '1px solid #444',
                                           'padding-left': '10px'}),
                                dcc.Store(id='pdm-result-store', data=None),
                            ], className='plot-toolbar'),
                        ], open=True),
                        html.Div(id='plot-stats-cards', className='plot-stats', style={'display': 'none'}),
                        html.Div([
                            html.Div(id='plot-status-panel', className='plot-status'),
                            html.Div(id='camera-filter-panel', className='camera-diag'),
                            html.Div(id='metadata-health-indicator'),
                            # Pipeline status chips
                            html.Div([
                                html.Div(id='pipeline-status-chips',
                                         style={'display': 'flex', 'gap': '6px', 'marginTop': '6px',
                                                'flexWrap': 'wrap', 'fontSize': '10px'}),
                                html.Div([
                                    html.Button('Run All Missing', id='run-pipeline-btn', n_clicks=0,
                                                className='compact-btn',
                                                style={'fontSize': '10px'}),
                                    html.Button('Re-run Current', id='rerun-pipeline-btn', n_clicks=0,
                                                className='compact-btn',
                                                style={'fontSize': '10px'}),
                                ], style={'display': 'flex', 'gap': '6px', 'marginTop': '4px'}),
                                html.Div([
                                    html.Button('Recompute Stats', id='rerun-stage-stats-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute Events', id='rerun-stage-events-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute Characterize', id='rerun-stage-characterize-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute Vetting', id='rerun-stage-vetting-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute External LCs', id='rerun-stage-external-lcs-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                ], style={'display': 'flex', 'gap': '6px', 'marginTop': '4px', 'flexWrap': 'wrap'}),
                                dcc.Loading(
                                    id='loading-pipeline', type='dot',
                                    children=html.Div(id='pipeline-run-status',
                                                      style={'fontSize': '10px', 'marginTop': '2px',
                                                             'color': '#7da8c4'}),
                                ),
                                html.Details([
                                    html.Summary('Log', style={'cursor': 'pointer', 'marginTop': '4px'}),
                                    html.Pre(
                                        id='pipeline-module-log-panel',
                                        style={
                                            'fontSize': '10px',
                                            'lineHeight': '1.35',
                                            'marginTop': '6px',
                                            'maxHeight': '220px',
                                            'overflowY': 'auto',
                                            'padding': '8px',
                                            'background': 'rgba(8, 16, 24, 0.75)',
                                            'border': '1px solid #284059',
                                            'borderRadius': '6px',
                                            'whiteSpace': 'pre-wrap',
                                            'wordBreak': 'break-word',
                                            'color': '#9fc6df',
                                        },
                                    ),
                                ], open=True, style={'marginTop': '4px'}),
                            ], style={'marginTop': '6px'}),
                        ], id='diagnostics-section'),
                        # Grouped candidate metadata sections (collapsible, includes stats)
                        html.Div([
                            html.Div([
                                html.Span('Candidate Panels', className='title'),
                                dcc.Checklist(
                                    id='round-sigfigs',
                                    options=[{'label': ' Round', 'value': 'yes'}],
                                    value=['yes'],
                                    style={'display': 'inline-block', 'font-size': '11px', 'margin-right': '6px'},
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                html.Button('Collapse all', id='toggle-meta-all', n_clicks=0, className='compact-btn'),
                            ], className='meta-toolbar'),
                            html.Div([
                                html.Div(id='vetting-banner'),
                                html.Div(id='candidate-info-grid', className='candidate-metadata'),
                            ], className='metadata-sections'),
                        ]),
                        html.Details([
                            html.Summary('External Data', style={'cursor': 'pointer'}),
                            dcc.Loading(
                                html.Div(
                                    id='external-followup-panel',
                                    style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                ),
                                type='default',
                            ),
                        ], id='external-followup-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('Diagnostic Plots', style={'cursor': 'pointer'}),
                            html.Div(
                                id='diagnostic-plots-status',
                                style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                            ),
                            dcc.Loading(
                                html.Div(
                                    id='diagnostic-plots-panel',
                                    style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                ),
                                type='default',
                            ),
                        ], id='diagnostic-plots-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        # Run config / reproducibility
                        html.Details([
                            html.Summary('Run Config', style={'cursor': 'pointer'}),
                            html.Div([
                                html.Div([
                                    html.Button('Copy Config JSON', id='copy-run-config-btn', n_clicks=0, className='compact-btn', style={'margin-right': '6px'}),
                                    html.Button('Download Config JSON', id='download-run-config-btn', n_clicks=0, className='compact-btn'),
                                ], style={'margin-bottom': '6px'}),
                                html.Div(id='run-config-panel'),
                            ], style={'padding': '8px 10px'}),
                        ], id='run-config-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                    ], className='left-info-scroll'),
                ], id='left-info-panel', className='left-info-panel'),

                html.Div(
                    id='metadata-splitter',
                    className='panel-splitter-vertical',
                    title='Drag to resize information panel',
                ),

                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='interactive-plot',
                            className='plot-native',
                            mathjax=True,
                            config={
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                'responsive': True,
                            },
                            style={'display': 'block', 'width': '100%', 'height': '100%', 'min-height': '600px'},
                        ),
                        html.Img(
                            id='plot-image',
                            src='',
                            alt='Light curve plot',
                            style={'display': 'none', 'width': '100%', 'height': '100%'},
                        ),
                    ], className='plot-frame'),
                ], className='plot-container right-plot-panel'),
            ], className='workspace-panels'),

            # Control bar
            html.Div([
                # Score row
                html.Div([
                    html.Span('Confidence: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                ] + [
                    html.Button(str(i), id=f'score-{i}', n_clicks=0, className='score-btn')
                    for i in range(1, 5)
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '6px'}),

                # Event class row (clickable buttons, single-key shortcuts)
                html.Div([
                    html.Span('Class: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                    html.Span(id='prefix-indicator', style={'margin-right': '6px', 'font-size': '11px'}),
                ] + [
                    html.Button(
                        f'[{key.upper()}] {tag.replace("_", " ")}',
                        id=f'class-badge-{tag}',
                        n_clicks=0,
                        className='badge-btn',
                    )
                    for key, tag in CLASS_KEY_MAP.items()
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'margin-bottom': '6px'}),

                # Action row: Save, Done, Followup, Pass, Status, Notification
                html.Div([
                    html.Button('← Back', id='back-btn', n_clicks=0, className='action-btn'),
                    html.Button('Save [.]', id='save-btn', n_clicks=0, className='action-btn'),
                    html.Button('Done [Enter]', id='done-btn', n_clicks=0, className='action-btn primary'),
                    html.Span('', style={'width': '20px', 'display': 'inline-block'}),
                    html.Span(id='followup-indicator', style={'margin-right': '10px', 'font-size': '11px'}),
                    html.Span(id='pass-indicator', style={'color': '#888', 'margin-right': '10px', 'font-size': '11px'}),
                    html.Span(id='status-indicator', style={'color': '#888', 'margin-right': '10px', 'font-size': '11px'}),
                    html.Span(' | ', style={'color': '#444', 'margin-right': '10px'}),
                    dcc.Loading(
                        id='loading-pipeline-bottom', type='dot',
                        children=html.Span(id='bottom-pipeline-status', style={'color': '#7da8c4', 'font-size': '11px'}),
                    ),
                ], style={'display': 'flex', 'align-items': 'center'}),
            ], className='control-bar'),

            # Notes (M to enter, Esc to exit)
            html.Div([
                dcc.Input(
                    id='notes',
                    type='text',
                    placeholder='Notes - click to type, Esc to exit',
                    style={'width': '100%', 'font-size': '11px', 'height': '26px'},
                ),
            ], className='review-form'),

            html.Div([
                html.Div(id='bottom-context-info', style={
                    'padding': '5px 12px',
                    'background-color': '#0a0a0a',
                    'border-top': '1px solid #555',
                    'color': '#9fb6cb',
                    'font-size': '10px',
                    'line-height': '1.35',
                    'white-space': 'normal',
                    'word-break': 'break-all',
                }),
            ]),

        ], className='content-area'),

        # Help modal
        dbc.Modal([
            dbc.ModalBody(html.Pre(HELP_TEXT, style={'color': '#e0e0e0', 'margin': '0', 'font-size': '11px'})),
            dbc.ModalFooter(dbc.Button("Close", id="close-help", className="action-btn")),
        ], id="help-modal", is_open=False),
    ], className='main-container')


app.layout = create_layout


# --- Pace Timer Callbacks ---

app.clientside_callback(
    """
    function(text) {
        return text;
    }
    """,
    Output('bottom-pipeline-status', 'children'),
    Input('pipeline-run-status', 'children'),
    prevent_initial_call=False
)

app.clientside_callback(
    """
    function(idx) {
        return Date.now();
    }
    """,
    Output('candidate-start-time', 'data'),
    Input('current-index', 'data')
)

app.clientside_callback(
    """
    function(idx, queueData) {
        if (!queueData || !Array.isArray(queueData.candidate_ids)) {
            return [null, 0, ''];
        }
        var ids = queueData.candidate_ids || [];
        var size = (typeof queueData.queue_size === 'number') ? queueData.queue_size : ids.length;
        var filterHash = (typeof queueData.filter_hash === 'string') ? queueData.filter_hash : '';
        var i = parseInt(idx == null ? 0 : idx, 10);
        if (!Number.isFinite(i) || i < 0 || i >= ids.length) {
            return [null, size, filterHash];
        }
        return [String(ids[i]), size, filterHash];
    }
    """,
    [Output('current-candidate-id', 'data'),
     Output('queue-size-store', 'data'),
     Output('queue-filter-hash-store', 'data')],
    [Input('current-index', 'data'),
     Input('queue-data', 'data')],
    prevent_initial_call=False
)

app.clientside_callback(
    """
    function(n_intervals, progressState, queueSize, sessionStart, toggle) {
        function formatHms(totalSeconds) {
            var seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
            var hours = Math.floor(seconds / 3600);
            var minutes = Math.floor((seconds % 3600) / 60);
            var secs = seconds % 60;
            var pad = function(value) {
                return String(value).padStart(2, '0');
            };
            return pad(hours) + ':' + pad(minutes) + ':' + pad(secs);
        }

        // Toggle the review-progress-indicator visibility
        var el = document.getElementById('review-progress-indicator');
        if (el) {
            if (toggle && toggle.indexOf('yes') !== -1) {
                el.style.display = '';
            } else {
                el.style.display = 'none';
            }
        }

        var reviewed = 0;
        var total = 0;
        if (progressState && typeof progressState === 'object') {
            reviewed = parseInt(progressState.reviewed == null ? 0 : progressState.reviewed, 10);
            total = parseInt(progressState.total == null ? 0 : progressState.total, 10);
        }
        if (!Number.isFinite(reviewed) || reviewed < 0) {
            reviewed = 0;
        }
        if (!Number.isFinite(total) || total < 0) {
            total = 0;
        }

        var queueTotal = parseInt(queueSize == null ? 0 : queueSize, 10);
        if ((!Number.isFinite(total) || total <= 0) && Number.isFinite(queueTotal) && queueTotal > 0) {
            total = queueTotal;
        }

        var pct = total > 0 ? (100.0 * reviewed / total) : 0.0;

        var startTs = null;
        if (sessionStart && typeof sessionStart === 'object' && sessionStart.ts != null) {
            startTs = Number(sessionStart.ts);
        }
        if (!Number.isFinite(startTs)) {
            startTs = Date.now() / 1000.0;
        }

        var elapsedS = Math.max(0.0, (Date.now() / 1000.0) - startTs);
        var elapsedTxt = formatHms(elapsedS);

        var pacePerMin = 0.0;
        if (elapsedS > 0 && reviewed > 0) {
            pacePerMin = reviewed / (elapsedS / 60.0);
        }

        var etaTxt = '--:--:--';
        if (pacePerMin > 0 && total > reviewed) {
            etaTxt = formatHms(((total - reviewed) / pacePerMin) * 60.0);
        }

        var paceTxt = pacePerMin > 0 ? pacePerMin.toFixed(2) + '/min' : '--/min';
        return [
            'Reviewed: ' + reviewed + '/' + total + ' (' + pct.toFixed(1) + '%) | Elapsed: ' + elapsedTxt + ' | Pace: ' + paceTxt + ' | ETA: ' + etaTxt,
            ''
        ];
    }
    """,
    [Output('review-progress-indicator', 'children'),
     Output('pace-timer-display', 'children')],
    Input('review-metrics-interval', 'n_intervals'),
    [State('review-progress-state', 'data'),
     State('queue-size-store', 'data'),
     State('review-session-start', 'data'),
     State('pace-timer-toggle', 'value')]
)


# Global keyboard listener (set up once on page load)
app.clientside_callback(
    """
    function() {
        // This runs once when the app loads
        var keyboardInput = document.getElementById('keyboard-input');

        if (!keyboardInput) {
            console.error('keyboard-input element not found!');
            return window.dash_clientside.no_update;
        }

        var valueDescriptor = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        );
        var nativeInputValueSetter = valueDescriptor && valueDescriptor.set
            ? valueDescriptor.set
            : null;

        var dispatchKeyToDash = function(key) {
            if (!key) {
                return;
            }
            if (nativeInputValueSetter) {
                nativeInputValueSetter.call(
                    keyboardInput, key + '\t' + String(Date.now())
                );
            } else {
                keyboardInput.value = key + '\t' + String(Date.now());
            }
            keyboardInput.dispatchEvent(new Event('input', {bubbles: true}));
        };

        // Register once: global keyboard listener that feeds Dash callbacks.
        if (!window.__malcaKeyboardListenerAttached) {
            document.addEventListener('keydown', function(e) {
                var target = e.target;
                var tag = target && target.tagName ? target.tagName : '';
                var targetId = target && target.id ? target.id : '';
                var inFormField = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') && targetId !== 'keyboard-input';

                // Inside a form field: keep typing behavior unchanged.
                // To trigger shortcuts while focused in inputs, use Alt+<key>.
                if (inFormField) {
                    if (e.key === 'Escape') {
                        target.blur();
                        return;
                    }
                    var allowWithFormModifier = e.altKey && !e.ctrlKey && !e.metaKey;
                    if (!allowWithFormModifier) {
                        return;
                    }
                    e.preventDefault();
                    dispatchKeyToDash(e.key);
                    return;
                }

                // Outside form fields, shortcuts are single-key only.
                if (e.ctrlKey || e.metaKey || e.altKey) {
                    return;
                }

                var key = e.key;
                if (!key || key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta') {
                    return;
                }

                if (!e.shiftKey && (key === 'R' || key === 'r')) {
                    e.preventDefault();
                    dispatchKeyToDash('r');
                    return;
                }

                // Prevent browser defaults for keys we use as shortcuts
                if (key === 'Backspace' || key === 'Tab' || key === 'Enter') {
                    e.preventDefault();
                }

                dispatchKeyToDash(key);
            });
            window.__malcaKeyboardListenerAttached = true;
            console.log('Global keyboard listener initialized');
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('keyboard-input', 'value', allow_duplicate=True),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call='initial_duplicate'
)


app.clientside_callback(
    """
    function(_savedStateTs, savedState, currentTheme, reviewScope) {
        var nu = window.dash_clientside.no_update;
        if (savedState && typeof savedState === 'object') {
            return nu;
        }
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.theme::' + scope;
        try {
            var saved = window.localStorage.getItem(storageKey);
            if (saved && ['black', 'gray', 'white'].includes(saved)) {
                return saved;
            }
        } catch (e) {
            // ignore storage read failures
        }
        return ['black', 'gray', 'white'].includes(currentTheme) ? currentTheme : 'black';
    }
    """,
    Output('theme-mode', 'value'),
    Input('saved-review-gui-state', 'modified_timestamp'),
    [State('saved-review-gui-state', 'data'),
     State('theme-mode', 'value'),
     State('review-db-scope', 'data')],
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(theme, reviewScope) {
        var t = ['black', 'gray', 'white'].includes(theme) ? theme : 'black';
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.theme::' + scope;
        try {
            document.body.setAttribute('data-theme', t);
            window.localStorage.setItem(storageKey, t);
        } catch (e) {
            // ignore storage/document failures
        }
        return t;
    }
    """,
    Output('theme-mode-store', 'data'),
    Input('theme-mode', 'value'),
    State('review-db-scope', 'data'),
    prevent_initial_call=False,
)


# --- Sidebar plot prefs: save to localStorage on change ---
app.clientside_callback(
    """
    function(preset, overlays, mode, opacity, resHeight, externalSource, reviewScope) {
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.sidebar.plot.v2::' + scope;
        try {
            var obj = {
                preset: preset,
                overlays: overlays || [],
                mode: mode,
                opacity: opacity,
                resHeight: resHeight,
                externalSource: externalSource
            };
            window.localStorage.setItem(storageKey, JSON.stringify(obj));
        } catch (e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output('sidebar-plot-saved', 'data'),
    [Input('plot-preset', 'value'),
     Input('plot-overlays', 'value'),
     Input('plot-mode', 'value'),
     Input('baseline-opacity-slider', 'value'),
     Input('residual-height-slider', 'value'),
     Input('external-source-view', 'value')],
    State('review-db-scope', 'data'),
    prevent_initial_call=True,
)


# --- Sidebar plot prefs: load from localStorage on init ---
app.clientside_callback(
    """
    function(_savedStateTs, savedState, curPreset, curOverlays, curMode, curOpacity, curResHeight, curExternalSource, reviewScope) {
        var nu = window.dash_clientside.no_update;
        if (savedState && typeof savedState === 'object') {
            return [nu, nu, nu, nu, nu, nu, true];
        }
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.sidebar.plot.v2::' + scope;
        try {
            var raw = window.localStorage.getItem(storageKey);
            if (!raw) return [nu, nu, nu, nu, nu, nu, false];
            var obj = JSON.parse(raw);
            var preset = (obj.preset && ['Clean', 'Diagnostics', 'Full'].includes(obj.preset))
                ? obj.preset : nu;
            var overlays = Array.isArray(obj.overlays) ? obj.overlays : nu;
            var mode = (obj.mode && ['native', 'png'].includes(obj.mode)) ? obj.mode : nu;
            var opacity = (typeof obj.opacity === 'number') ? obj.opacity : nu;
            var resHeight = (typeof obj.resHeight === 'number') ? obj.resHeight : nu;
            var allowedSources = ['all', 'asassn', 'atlas', 'ztf', 'gaia_epoch', 'ps1', 'crts'];
            var externalSource = (obj.externalSource && allowedSources.includes(obj.externalSource))
                ? obj.externalSource : nu;
            return [preset, overlays, mode, opacity, resHeight, externalSource, true];
        } catch (e) {
            return [nu, nu, nu, nu, nu, nu, false];
        }
    }
    """,
    [Output('plot-preset', 'value', allow_duplicate=True),
     Output('plot-overlays', 'value', allow_duplicate=True),
     Output('plot-mode', 'value', allow_duplicate=True),
     Output('baseline-opacity-slider', 'value', allow_duplicate=True),
     Output('residual-height-slider', 'value', allow_duplicate=True),
     Output('external-source-view', 'value', allow_duplicate=True),
     Output('plot-defaults-initialized', 'data', allow_duplicate=True)],
    Input('saved-review-gui-state', 'modified_timestamp'),
    [State('saved-review-gui-state', 'data'),
     State('plot-preset', 'value'),
     State('plot-overlays', 'value'),
     State('plot-mode', 'value'),
     State('baseline-opacity-slider', 'value'),
     State('residual-height-slider', 'value'),
     State('external-source-view', 'value'),
     State('review-db-scope', 'data')],
    prevent_initial_call='initial_duplicate',
)


app.clientside_callback(
    """
    function(_tick) {
        var splitter = document.getElementById('metadata-splitter');
        var leftPanel = document.getElementById('left-info-panel');
        var workspace = document.querySelector('.workspace-panels');
        if (!splitter || !leftPanel || !workspace) {
            return window.dash_clientside.no_update;
        }

        var storageKey = 'malca.review.left_panel.width.v1';
        var minWidth = 260;
        var defaultWidth = 420;

        var computeMaxWidth = function() {
            var total = workspace.clientWidth || window.innerWidth;
            var cap = Math.floor(total * 0.72);
            var floorCap = Math.max(minWidth + 40, cap);
            return floorCap;
        };

        var clampWidth = function(value) {
            var maxWidth = computeMaxWidth();
            var numeric = Number(value);
            if (!isFinite(numeric)) numeric = defaultWidth;
            if (numeric < minWidth) numeric = minWidth;
            if (numeric > maxWidth) numeric = maxWidth;
            return Math.round(numeric);
        };

        var applyWidth = function(value, persist) {
            var w = clampWidth(value);
            leftPanel.style.width = String(w) + 'px';
            leftPanel.style.flex = '0 0 ' + String(w) + 'px';
            if (persist) {
                try { window.localStorage.setItem(storageKey, String(w)); } catch (e) {}
            }
            return w;
        };

        if (!window.__malcaMetadataSplitterAttached) {
            var drag = { active: false, startX: 0, startWidth: 0, pointerId: null };

            var onPointerMove = function(e) {
                if (!drag.active) return;
                var nextWidth = drag.startWidth + (e.clientX - drag.startX);
                applyWidth(nextWidth, false);
                e.preventDefault();
            };

            var stopDrag = function(e) {
                if (!drag.active) return;
                drag.active = false;
                splitter.classList.remove('dragging');
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', stopDrag);
                window.removeEventListener('pointercancel', stopDrag);
                if (drag.pointerId !== null && splitter.releasePointerCapture) {
                    try { splitter.releasePointerCapture(drag.pointerId); } catch (err) {}
                }
                drag.pointerId = null;
                applyWidth(leftPanel.getBoundingClientRect().width, true);
                if (e) e.preventDefault();
            };

            splitter.addEventListener('pointerdown', function(e) {
                drag.active = true;
                drag.startX = e.clientX;
                drag.startWidth = leftPanel.getBoundingClientRect().width;
                drag.pointerId = (typeof e.pointerId === 'number') ? e.pointerId : null;
                splitter.classList.add('dragging');
                if (drag.pointerId !== null && splitter.setPointerCapture) {
                    try { splitter.setPointerCapture(drag.pointerId); } catch (err) {}
                }
                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', stopDrag);
                window.addEventListener('pointercancel', stopDrag);
                e.preventDefault();
            });

            window.addEventListener('resize', function() {
                applyWidth(leftPanel.getBoundingClientRect().width, false);
            });

            window.__malcaMetadataSplitterAttached = true;
        }

        var saved = null;
        try { saved = window.localStorage.getItem(storageKey); } catch (e) { saved = null; }
        var initialWidth = defaultWidth;
        if (saved !== null && saved !== '') {
            var parsed = parseInt(saved, 10);
            if (!isNaN(parsed)) initialWidth = parsed;
        }
        applyWidth(initialWidth, false);

        return window.dash_clientside.no_update;
    }
    """,
    Output('metadata-resize-init', 'data'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(_tick) {
        var splitter = document.getElementById('status-splitter');
        var statusPanel = document.getElementById('plot-status-panel');
        if (!splitter || !statusPanel) {
            return window.dash_clientside.no_update;
        }

        var storageKey = 'malca.review.plot_status.height.v1';
        var minHeight = 16;
        var defaultHeight = 36;

        var computeMaxHeight = function() {
            return Math.max(64, Math.floor(window.innerHeight * 0.42));
        };

        var clampHeight = function(value) {
            var maxHeight = computeMaxHeight();
            var numeric = Number(value);
            if (!isFinite(numeric)) {
                numeric = defaultHeight;
            }
            if (numeric < minHeight) {
                numeric = minHeight;
            }
            if (numeric > maxHeight) {
                numeric = maxHeight;
            }
            return Math.round(numeric);
        };

        var applyHeight = function(value, persist) {
            var h = clampHeight(value);
            statusPanel.style.height = String(h) + 'px';
            statusPanel.style.flex = '0 0 auto';
            if (persist) {
                try {
                    window.localStorage.setItem(storageKey, String(h));
                } catch (e) {
                    // ignore storage failures
                }
            }
            return h;
        };

        if (!window.__malcaStatusSplitterAttached) {
            var drag = {
                active: false,
                startY: 0,
                startHeight: 0,
                pointerId: null,
            };

            var onPointerMove = function(e) {
                if (!drag.active) {
                    return;
                }
                var nextHeight = drag.startHeight - (e.clientY - drag.startY);
                applyHeight(nextHeight, false);
                e.preventDefault();
            };

            var stopDrag = function(e) {
                if (!drag.active) {
                    return;
                }
                drag.active = false;
                splitter.classList.remove('dragging');
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', stopDrag);
                window.removeEventListener('pointercancel', stopDrag);
                if (drag.pointerId !== null && splitter.releasePointerCapture) {
                    try {
                        splitter.releasePointerCapture(drag.pointerId);
                    } catch (err) {
                        // ignore capture-release failures
                    }
                }
                drag.pointerId = null;
                applyHeight(statusPanel.getBoundingClientRect().height, true);
                if (e) {
                    e.preventDefault();
                }
            };

            splitter.addEventListener('pointerdown', function(e) {
                drag.active = true;
                drag.startY = e.clientY;
                drag.startHeight = statusPanel.getBoundingClientRect().height;
                drag.pointerId = (typeof e.pointerId === 'number') ? e.pointerId : null;
                splitter.classList.add('dragging');
                if (drag.pointerId !== null && splitter.setPointerCapture) {
                    try {
                        splitter.setPointerCapture(drag.pointerId);
                    } catch (err) {
                        // ignore capture failures
                    }
                }
                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', stopDrag);
                window.addEventListener('pointercancel', stopDrag);
                e.preventDefault();
            });

            window.addEventListener('resize', function() {
                applyHeight(statusPanel.getBoundingClientRect().height, false);
            });

            window.__malcaStatusSplitterAttached = true;
        }

        var saved = null;
        try {
            saved = window.localStorage.getItem(storageKey);
        } catch (e) {
            saved = null;
        }
        var initialHeight = defaultHeight;
        if (saved !== null && saved !== '') {
            var parsed = parseInt(saved, 10);
            if (!isNaN(parsed)) {
                initialHeight = parsed;
            }
        }
        applyHeight(initialHeight, false);

        return window.dash_clientside.no_update;
    }
    """,
    Output('status-resize-init', 'data'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)


# Toggle sidebar
@app.callback(
    [Output('sidebar', 'className'),
     Output('sidebar-toggle', 'className'),
     Output('sidebar-state', 'data')],
    [Input('sidebar-toggle', 'n_clicks'),
     Input('keyboard-input', 'value')],
    State('sidebar-state', 'data'),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, key_value, is_expanded):
    """Toggle sidebar visibility."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger = ctx.triggered[0]['prop_id']

    # Check if Escape was pressed
    key = _keyboard_key(key_value)
    if 'keyboard-input' in trigger and key == 'Escape':
        is_expanded = not is_expanded

    # Check if toggle button was clicked
    elif 'sidebar-toggle' in trigger and n_clicks:
        is_expanded = not is_expanded
    else:
        return no_update, no_update, no_update

    sidebar_class = 'sidebar expanded' if is_expanded else 'sidebar'
    toggle_class = 'sidebar-toggle sidebar-expanded' if is_expanded else 'sidebar-toggle'
    return sidebar_class, toggle_class, is_expanded


@app.callback(
    Output('refresh-btn', 'n_clicks', allow_duplicate=True),
    Input('keyboard-input', 'value'),
    State('refresh-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def keyboard_refresh_queue(key_value, current_clicks):
    """Trigger Refresh Queue from plain R."""
    if _keyboard_key(key_value).lower() != 'r':
        raise dash.exceptions.PreventUpdate
    return int(current_clicks or 0) + 1


# --- All filter State components used by load_queue -------------------------
# Auto-generated from _SIDEBAR_GROUPS so the UI and callback always stay in sync.
_BOOL_MODE_STATES: list[tuple[str, str]] = []
_NUM_COLUMNS: list[str] = []
_NUM_INPUT_STATES: list[tuple[dict[str, str], str]] = []
_TEXT_STATES: list[tuple[str, str]] = []
_SELECT_STATES: list[tuple[str, str]] = []
_QUEUE_FILTER_APP_STATE_KEY = "dash_queue_filter_ui_state_v1"
_REVIEW_GUI_STATE_APP_STATE_KEY = "dash_review_gui_state_v1"

for _grp_name, _grp_items in _SIDEBAR_GROUPS:
    for _ftype, _col in _grp_items:
        _cid = _col_id(_col)
        if _ftype == 'bool':
            _BOOL_MODE_STATES.append((f'{_cid}-mode', f'{_col}_mode'))
        elif _ftype == 'num':
            _NUM_COLUMNS.append(_col)
            _NUM_INPUT_STATES.append(({'type': 'num-filter-min-input', 'col': _col}, f'min_{_col}'))
            _NUM_INPUT_STATES.append(({'type': 'num-filter-max-input', 'col': _col}, f'max_{_col}'))
        elif _ftype == 'text':
            _TEXT_STATES.append((f'filter-{_cid}', _col))
        elif _ftype == 'select':
            _SELECT_STATES.append((f'exclude-{_cid}', f'exclude_{_col}'))

# Build the callback I/O lists dynamically.
_FILTER_VALUE_STATES = (
    [State('filter-unreviewed', 'value'),
     State('filter-failed', 'value')]
    + [State(cid, 'value') for cid, _ in _BOOL_MODE_STATES]
    + [State(cid, 'value') for cid, _ in _NUM_INPUT_STATES]
    + [State(cid, 'value') for cid, _ in _TEXT_STATES]
    + [State(cid, 'value') for cid, _ in _SELECT_STATES]
    + [State('sort-col', 'value'),
       State('sort-desc', 'value')]
)

_FILTER_VALUE_INPUTS = (
    [Input('filter-unreviewed', 'value'),
     Input('filter-failed', 'value')]
    + [Input(cid, 'value') for cid, _ in _BOOL_MODE_STATES]
    + [Input(cid, 'value') for cid, _ in _NUM_INPUT_STATES]
    + [Input(cid, 'value') for cid, _ in _TEXT_STATES]
    + [Input(cid, 'value') for cid, _ in _SELECT_STATES]
    + [Input('sort-col', 'value'),
       Input('sort-desc', 'value')]
)

_FILTER_VALUE_OUTPUTS = (
    [Output('filter-unreviewed', 'value', allow_duplicate=True),
     Output('filter-failed', 'value', allow_duplicate=True)]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _BOOL_MODE_STATES]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _NUM_INPUT_STATES]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _TEXT_STATES]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _SELECT_STATES]
    + [Output('sort-col', 'value', allow_duplicate=True),
       Output('sort-desc', 'value', allow_duplicate=True)]
)

_queue_states = _FILTER_VALUE_STATES
_TEXT_OPTION_STATES = [State(cid, 'options') for cid, _ in _TEXT_STATES]
_SELECT_OPTION_STATES = [State(cid, 'options') for cid, _ in _SELECT_STATES]


def _coerce_string_list(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set, np.ndarray, pd.Series)):
        values = list(raw_value)
    else:
        values = [raw_value]

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _coerce_yes_checklist_value(raw_value: object) -> list[str]:
    return ['yes'] if 'yes' in _coerce_string_list(raw_value) else []


def _coerce_bool_mode_value(raw_value: object) -> str:
    valid = {'Any', 'True', 'False', 'Unset'}
    text = str(raw_value).strip() if raw_value is not None else ''
    return text if text in valid else 'Any'


def _coerce_numeric_input_value(raw_value: object) -> float | None:
    if raw_value in ('', None):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _coerce_text_filter_value(raw_value: object) -> str:
    text = str(raw_value).strip() if raw_value is not None else ''
    return text or 'Any'


def _coerce_sort_cols(raw_value: object) -> list[str]:
    cols = _coerce_string_list(raw_value)
    return cols or ['candidate_id']


def _coerce_choice(raw_value: object, allowed: set[str], default: str) -> str:
    text = str(raw_value).strip() if raw_value is not None else ''
    return text if text in allowed else default


def _review_gui_state_from_values(
    *,
    theme_mode: object,
    plot_mode: object,
    plot_overlays: object,
    baseline_opacity: object,
    residual_height: object,
    external_source_view: object,
    camera_values: object,
    band_values: object,
    yaxis_mode: object,
    period_method: object,
    pdm_min_period: object,
    pdm_max_period: object,
    pdm_manual_period: object,
) -> dict[str, object]:
    overlay_allowed = {'raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras', 'diagnostics', 'confidence'}
    external_allowed = {str(opt.get('value')) for opt in EXTERNAL_SOURCE_VIEW_OPTIONS}
    return {
        'theme_mode': _coerce_choice(theme_mode, {'black', 'gray', 'white'}, DEFAULT_THEME),
        'plot_mode': _coerce_choice(plot_mode, {'native', 'png'}, 'native'),
        'plot_overlays': [value for value in _coerce_string_list(plot_overlays) if value in overlay_allowed],
        'baseline_opacity': _coerce_numeric_input_value(baseline_opacity),
        'residual_height': _coerce_numeric_input_value(residual_height),
        'external_source_view': _coerce_choice(external_source_view, external_allowed, DEFAULT_EXTERNAL_SOURCE_VIEW),
        'camera_values': _coerce_string_list(camera_values),
        'band_values': _coerce_string_list(band_values) or ['g', 'V'],
        'yaxis_mode': _coerce_choice(yaxis_mode, {'mag', 'flux'}, 'mag'),
        'period_method': _coerce_choice(period_method, {'lsp', 'pdm', 'ce'}, 'pdm'),
        'pdm_min_period': _coerce_numeric_input_value(pdm_min_period),
        'pdm_max_period': _coerce_numeric_input_value(pdm_max_period),
        'pdm_manual_period': _coerce_numeric_input_value(pdm_manual_period),
    }


def _normalize_review_gui_state(raw_state: object) -> dict[str, object] | None:
    if not isinstance(raw_state, dict) or not raw_state:
        return None
    return _review_gui_state_from_values(
        theme_mode=raw_state.get('theme_mode'),
        plot_mode=raw_state.get('plot_mode'),
        plot_overlays=raw_state.get('plot_overlays'),
        baseline_opacity=raw_state.get('baseline_opacity'),
        residual_height=raw_state.get('residual_height'),
        external_source_view=raw_state.get('external_source_view'),
        camera_values=raw_state.get('camera_values'),
        band_values=raw_state.get('band_values'),
        yaxis_mode=raw_state.get('yaxis_mode'),
        period_method=raw_state.get('period_method'),
        pdm_min_period=raw_state.get('pdm_min_period'),
        pdm_max_period=raw_state.get('pdm_max_period'),
        pdm_manual_period=raw_state.get('pdm_manual_period'),
    )


def _queue_filter_ui_state_from_values(*state_values: object) -> dict[str, object]:
    it = iter(state_values)
    state: dict[str, object] = {
        'filter_unreviewed': _coerce_yes_checklist_value(next(it)),
        'filter_failed': _coerce_yes_checklist_value(next(it)),
    }
    for _, fkey in _BOOL_MODE_STATES:
        state[fkey] = _coerce_bool_mode_value(next(it))
    for _, fkey in _NUM_INPUT_STATES:
        state[fkey] = _coerce_numeric_input_value(next(it))
    for _, fkey in _TEXT_STATES:
        state[fkey] = _coerce_text_filter_value(next(it))
    for _, fkey in _SELECT_STATES:
        state[fkey] = _coerce_string_list(next(it))
    state['sort_cols'] = _coerce_sort_cols(next(it))
    state['sort_desc'] = _coerce_yes_checklist_value(next(it))
    return state


def _normalize_saved_queue_filter_ui_state(raw_state: object) -> dict[str, object] | None:
    if not isinstance(raw_state, dict) or not raw_state:
        return None

    state: dict[str, object] = {
        'filter_unreviewed': _coerce_yes_checklist_value(
            raw_state.get('filter_unreviewed', ['yes'] if _coerce_bool(raw_state.get('only_unreviewed')) else [])
        ),
        'filter_failed': _coerce_yes_checklist_value(
            raw_state.get('filter_failed', ['yes'] if _coerce_bool(raw_state.get('require_failed_any_false')) else [])
        ),
    }
    for _, fkey in _BOOL_MODE_STATES:
        state[fkey] = _coerce_bool_mode_value(raw_state.get(fkey))
    for _, fkey in _NUM_INPUT_STATES:
        state[fkey] = _coerce_numeric_input_value(raw_state.get(fkey))
    for _, fkey in _TEXT_STATES:
        state[fkey] = _coerce_text_filter_value(raw_state.get(fkey))
    for _, fkey in _SELECT_STATES:
        state[fkey] = _coerce_string_list(raw_state.get(fkey))
    state['sort_cols'] = _coerce_sort_cols(raw_state.get('sort_cols', raw_state.get('sort_col')))
    sort_desc_value = raw_state.get('sort_desc')
    if isinstance(sort_desc_value, bool):
        sort_desc_value = ['yes'] if sort_desc_value else []
    state['sort_desc'] = _coerce_yes_checklist_value(sort_desc_value)
    return state


def _queue_filter_ui_values_from_state(raw_state: object) -> tuple[object, ...] | None:
    state = _normalize_saved_queue_filter_ui_state(raw_state)
    if state is None:
        return None

    values: list[object] = [
        state['filter_unreviewed'],
        state['filter_failed'],
    ]
    for _, fkey in _BOOL_MODE_STATES:
        values.append(state[fkey])
    for _, fkey in _NUM_INPUT_STATES:
        values.append(state[fkey])
    for _, fkey in _TEXT_STATES:
        values.append(state[fkey])
    for _, fkey in _SELECT_STATES:
        values.append(state[fkey])
    values.append(state['sort_cols'])
    values.append(state['sort_desc'])
    return tuple(values)


def _merge_unhydrated_saved_queue_filter_ui_state(
    ui_state: dict[str, object],
    restore_state: object,
    text_option_values: list[object] | tuple[object, ...],
    select_option_values: list[object] | tuple[object, ...],
) -> dict[str, object]:
    """Use DB-restored text/select filters while those dropdown options are still unhydrated."""
    if not isinstance(restore_state, dict) or not restore_state.get('restored'):
        return ui_state

    saved_ui_state = _normalize_saved_queue_filter_ui_state(restore_state.get('saved_ui_state'))
    if saved_ui_state is None:
        return ui_state

    merged = dict(ui_state)

    for ((_, fkey), options) in zip(_TEXT_STATES, text_option_values):
        current = _coerce_text_filter_value(merged.get(fkey))
        saved = _coerce_text_filter_value(saved_ui_state.get(fkey))
        option_count = len(options or []) if isinstance(options, (list, tuple)) else 0
        if option_count <= 1 and current == 'Any' and saved != 'Any':
            merged[fkey] = saved

    for ((_, fkey), options) in zip(_SELECT_STATES, select_option_values):
        current = _coerce_string_list(merged.get(fkey))
        saved = _coerce_string_list(saved_ui_state.get(fkey))
        option_count = len(options or []) if isinstance(options, (list, tuple)) else 0
        if option_count == 0 and not current and saved:
            merged[fkey] = saved

    return merged


@app.callback(
    [*[Output(cid, 'value', allow_duplicate=True) for cid, _ in _NUM_INPUT_STATES],
     Output('sidebar-status', 'children', allow_duplicate=True)],
    Input('reset-numeric-filters-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_numeric_filters(n_clicks):
    """Clear numeric filter inputs so sliders reset to current queue bounds."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return (*([None] * len(_NUM_INPUT_STATES)), "Reset numeric filters to the current queue bounds.")


_SIDEBAR_FILTER_OUTPUTS = (
    [Output(cid, 'options') for cid, _ in _TEXT_STATES]
    + [Output(cid, 'options') for cid, _ in _SELECT_STATES]
)

_VETTING_KNOWN_FILTER_OUTPUTS = (
    [Output(f'{_col_id(col)}-mode', 'value') for col in VETTING_KNOWN_BOOL_FILTERS]
    + [Output(f'exclude-{_col_id(col)}', 'value') for col in VETTING_KNOWN_SELECT_FILTERS]
)

_VETTING_KNOWN_FILTER_OPTION_STATES = (
    [State('queue-source-path', 'data')]
    + [State(f'exclude-{_col_id(col)}', 'options') for col in VETTING_KNOWN_SELECT_FILTERS]
)


def _normalize_numeric_filter_value(
    fkey: str,
    raw_value: object,
    bounds_data: dict[str, dict[str, float | None]] | None,
) -> float | None:
    """Treat full-range numeric inputs as unset queue filters."""
    if raw_value in ("", None):
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    if fkey.startswith("min_"):
        col = fkey[4:]
        bound_key = "min"
    elif fkey.startswith("max_"):
        col = fkey[4:]
        bound_key = "max"
    else:
        return value

    info = (bounds_data or {}).get(col) or {}
    bound = info.get(bound_key)
    if bound is None:
        return None

    try:
        bound_value = float(bound)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(bound_value):
        return None

    abs_tol = 1e-9 * max(1.0, abs(bound_value))
    if math.isclose(value, bound_value, rel_tol=1e-9, abs_tol=abs_tol):
        return None

    return value


def _normalize_numeric_filter_inputs(
    bounds_data: dict[str, dict[str, float | None]] | None,
    raw_values: dict[str, object],
) -> dict[str, float | None]:
    """Normalize numeric queue filters against the current slider bounds."""
    return {
        fkey: _normalize_numeric_filter_value(fkey, raw_value, bounds_data)
        for fkey, raw_value in raw_values.items()
    }


def _queue_filter_params_from_ui_state(
    ui_state: dict[str, object],
    numeric_bounds: dict[str, dict[str, float | None]] | None,
    queue_source_scope: object,
) -> dict[str, object]:
    filter_params: dict[str, object] = {
        'only_unreviewed': 'yes' in _coerce_yes_checklist_value(ui_state.get('filter_unreviewed')),
        'require_failed_any_false': 'yes' in _coerce_yes_checklist_value(ui_state.get('filter_failed')),
    }
    for _, fkey in _BOOL_MODE_STATES:
        filter_params[fkey] = _coerce_bool_mode_value(ui_state.get(fkey))
    raw_numeric_filters = {
        fkey: ui_state.get(fkey)
        for _, fkey in _NUM_INPUT_STATES
    }
    filter_params.update(_normalize_numeric_filter_inputs(numeric_bounds, raw_numeric_filters))
    for _, fkey in _TEXT_STATES:
        value = _coerce_text_filter_value(ui_state.get(fkey))
        filter_params[fkey] = value if value != 'Any' else None
    for _, fkey in _SELECT_STATES:
        values = _coerce_string_list(ui_state.get(fkey))
        filter_params[fkey] = values if values else None
    filter_params['sort_cols'] = _coerce_sort_cols(ui_state.get('sort_cols'))
    filter_params['sort_desc'] = 'yes' in _coerce_yes_checklist_value(ui_state.get('sort_desc'))
    filter_params.update(_queue_scope_filter_kwargs(queue_source_scope))
    return filter_params


def _load_numeric_filter_bounds(queue_source_scope):
    """Build numeric slider bounds for the current queue scope."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        kwargs = {'columns': _NUM_COLUMNS}
        kwargs.update(_queue_scope_filter_kwargs(queue_source_scope))
        return get_numeric_bounds(conn, **kwargs)


if _background_callback_manager is not None:
    @app.callback(
        Output('numeric-filter-bounds', 'data'),
        Input('refresh-filter-bounds-btn', 'n_clicks'),
        State('queue-source-path', 'data'),
        background=True,
        running=[
            (Output('refresh-filter-bounds-btn', 'disabled'), True, False),
            (Output('numeric-bounds-status', 'children'), 'Loading slider bounds...', 'Sliders load on startup; use refresh to rebuild bounds.'),
        ],
        prevent_initial_call=False,
    )
    def load_numeric_filter_bounds_callback(_refresh_clicks, queue_source_scope):
        return _load_numeric_filter_bounds(queue_source_scope)
else:
    @app.callback(
        Output('numeric-filter-bounds', 'data'),
        Input('refresh-filter-bounds-btn', 'n_clicks'),
        State('queue-source-path', 'data'),
        prevent_initial_call=False,
    )
    def load_numeric_filter_bounds_callback(_refresh_clicks, queue_source_scope):
        return _load_numeric_filter_bounds(queue_source_scope)


def _load_sidebar_filter_payload(sidebar_open, queue_source_scope):
    """Hydrate expensive sidebar filter controls only when the sidebar is opened."""
    if not sidebar_open:
        return tuple([no_update] * len(_SIDEBAR_FILTER_OUTPUTS))

    scope_kwargs = _queue_scope_filter_kwargs(queue_source_scope)
    with closing(db_connect(Path(DB_PATH))) as conn:
        text_options = []
        for _cid, col in _TEXT_STATES:
            values = get_distinct_values(conn, col, **scope_kwargs)
            text_options.append(
                [{'label': 'Any', 'value': 'Any'}]
                + [
                    {'label': str(v), 'value': str(v)}
                    for v in values
                    if v is not None and str(v).strip() != ''
                ]
            )

        select_options = []
        for _cid, filter_key in _SELECT_STATES:
            col = filter_key.replace('exclude_', '', 1)
            values = get_distinct_values(conn, col, **scope_kwargs)
            select_options.append(
                [
                    {'label': str(v), 'value': str(v)}
                    for v in values
                    if v is not None and str(v).strip() != ''
                ]
            )

    return (*text_options, *select_options)


def _load_vetting_known_select_options(queue_source_scope) -> dict[str, list[dict[str, str]]]:
    """Load distinct option sets for the definite known-type vetting filters."""
    scope_kwargs = _queue_scope_filter_kwargs(queue_source_scope)
    with closing(db_connect(Path(DB_PATH))) as conn:
        return {
            col: [
                {'label': str(v), 'value': str(v)}
                for v in get_distinct_values(conn, col, **scope_kwargs)
                if v is not None and str(v).strip() != ''
            ]
            for col in VETTING_KNOWN_SELECT_FILTERS
        }


if _background_callback_manager is not None:
    @app.callback(
        _SIDEBAR_FILTER_OUTPUTS,
        [Input('sidebar-state', 'data'),
         Input('import-trigger', 'data'),
         Input('queue-source-path', 'data')],
        background=True,
        running=[
            (Output('filter-load-status', 'children'), 'Loading filter options...', 'Dropdown options load when the sidebar opens.'),
        ],
        prevent_initial_call=False,
    )
    def hydrate_sidebar_filters_callback(sidebar_open, _import_trigger, queue_source_scope):
        return _load_sidebar_filter_payload(sidebar_open, queue_source_scope)
else:
    @app.callback(
        _SIDEBAR_FILTER_OUTPUTS,
        [Input('sidebar-state', 'data'),
         Input('import-trigger', 'data'),
         Input('queue-source-path', 'data')],
        prevent_initial_call=False,
    )
    def hydrate_sidebar_filters_callback(sidebar_open, _import_trigger, queue_source_scope):
        return _load_sidebar_filter_payload(sidebar_open, queue_source_scope)


@app.callback(
    _VETTING_KNOWN_FILTER_OUTPUTS,
    Input('vetting-known-types-btn', 'n_clicks'),
    _VETTING_KNOWN_FILTER_OPTION_STATES,
    prevent_initial_call=True,
)
def apply_vetting_known_type_filters(n_clicks, *select_options):
    """Apply the definite known-type vetting preset."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    queue_source_scope, *option_lists = select_options
    select_options_by_col = dict(zip(VETTING_KNOWN_SELECT_FILTERS, option_lists))
    if any(not select_options_by_col.get(col) for col in VETTING_KNOWN_SELECT_FILTERS):
        hydrated_options = _load_vetting_known_select_options(queue_source_scope)
        for col, options in hydrated_options.items():
            if not select_options_by_col.get(col):
                select_options_by_col[col] = options
    bool_values, select_values = _vetting_known_filter_preset(select_options_by_col)
    return (*bool_values, *select_values)


@app.callback(
    [*_FILTER_VALUE_OUTPUTS, Output('restored-filter-applied', 'data', allow_duplicate=True)],
    Input('review-db-scope', 'data'),
    prevent_initial_call='initial_duplicate',
)
def restore_saved_queue_filters(_db_scope):
    """Restore sidebar queue filters from DB-backed app_state when available."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        raw = str(load_app_state(conn, _QUEUE_FILTER_APP_STATE_KEY, '') or '').strip()

    if not raw:
        return (*([no_update] * len(_FILTER_VALUE_OUTPUTS)), {'ts': time.time(), 'ready': True, 'restored': False, 'saved_ui_state': None})

    try:
        saved_state = json.loads(raw)
    except Exception:
        return (*([no_update] * len(_FILTER_VALUE_OUTPUTS)), {'ts': time.time(), 'ready': True, 'restored': False, 'saved_ui_state': None})

    values = _queue_filter_ui_values_from_state(saved_state)
    normalized_state = _normalize_saved_queue_filter_ui_state(saved_state)
    if values is None:
        return (*([no_update] * len(_FILTER_VALUE_OUTPUTS)), {'ts': time.time(), 'ready': True, 'restored': False, 'saved_ui_state': normalized_state})

    return (*values, {'ts': time.time(), 'ready': True, 'restored': True, 'saved_ui_state': normalized_state})


@app.callback(
    Output('filter-params', 'data'),
    _FILTER_VALUE_INPUTS,
    State('restored-filter-applied', 'data'),
    _TEXT_OPTION_STATES,
    _SELECT_OPTION_STATES,
    prevent_initial_call=True,
)
def persist_queue_filters(*callback_values):
    """Persist sidebar queue filters to app_state without touching candidate/review data."""
    n_values = len(_FILTER_VALUE_INPUTS)
    n_text_opts = len(_TEXT_OPTION_STATES)
    value_states = callback_values[:n_values]
    restore_state = callback_values[n_values]
    text_option_values = callback_values[n_values + 1:n_values + 1 + n_text_opts]
    select_option_values = callback_values[n_values + 1 + n_text_opts:]
    if not isinstance(restore_state, dict) or not restore_state.get('ready'):
        raise dash.exceptions.PreventUpdate
    ui_state = _queue_filter_ui_state_from_values(*value_states)
    ui_state = _merge_unhydrated_saved_queue_filter_ui_state(
        ui_state,
        restore_state,
        text_option_values,
        select_option_values,
    )
    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            save_app_state(conn, _QUEUE_FILTER_APP_STATE_KEY, json.dumps(ui_state, default=str))
    except Exception as exc:
        print(f"[filters] Warning: could not persist queue filters: {exc}")
    return ui_state


@app.callback(
    Output('saved-review-gui-state', 'data'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)
def load_saved_review_gui_state(_tick):
    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            raw = str(load_app_state(conn, _REVIEW_GUI_STATE_APP_STATE_KEY, '') or '').strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return _normalize_review_gui_state(json.loads(raw))
    except Exception:
        return None


@app.callback(
    [Output('plot-mode', 'value', allow_duplicate=True),
     Output('plot-overlays', 'value', allow_duplicate=True),
     Output('baseline-opacity-slider', 'value', allow_duplicate=True),
     Output('residual-height-slider', 'value', allow_duplicate=True),
     Output('external-source-view', 'value', allow_duplicate=True),
     Output('camera-checklist', 'value', allow_duplicate=True),
     Output('band-checklist', 'value', allow_duplicate=True),
     Output('yaxis-mode', 'value', allow_duplicate=True),
     Output('period-method', 'value', allow_duplicate=True),
     Output('pdm-min-period', 'value', allow_duplicate=True),
     Output('pdm-max-period', 'value', allow_duplicate=True),
     Output('pdm-manual-period', 'value', allow_duplicate=True),
     Output('theme-mode', 'value', allow_duplicate=True),
     Output('plot-defaults-initialized', 'data', allow_duplicate=True)],
    Input('saved-review-gui-state', 'data'),
    prevent_initial_call='initial_duplicate',
)
def restore_saved_review_gui_state(saved_state):
    state = _normalize_review_gui_state(saved_state)
    if state is None:
        return tuple([no_update] * 14)
    return (
        state['plot_mode'],
        state['plot_overlays'],
        state['baseline_opacity'],
        state['residual_height'],
        state['external_source_view'],
        state['camera_values'],
        state['band_values'],
        state['yaxis_mode'],
        state['period_method'],
        state['pdm_min_period'],
        state['pdm_max_period'],
        state['pdm_manual_period'],
        state['theme_mode'],
        True,
    )


@app.callback(
    [Output('save-review-gui-state-status', 'children'),
     Output('saved-review-gui-state', 'data', allow_duplicate=True)],
    Input('save-review-gui-state-btn', 'n_clicks'),
    _FILTER_VALUE_STATES + [
        State('theme-mode', 'value'),
        State('plot-mode', 'value'),
        State('plot-overlays', 'value'),
        State('baseline-opacity-slider', 'value'),
        State('residual-height-slider', 'value'),
        State('external-source-view', 'value'),
        State('camera-checklist', 'value'),
        State('band-checklist', 'value'),
        State('yaxis-mode', 'value'),
        State('period-method', 'value'),
        State('pdm-min-period', 'value'),
        State('pdm-max-period', 'value'),
        State('pdm-manual-period', 'value'),
    ],
    prevent_initial_call=True,
)
def save_review_gui_state(n_clicks, *state_values):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    queue_values = state_values[:len(_FILTER_VALUE_STATES)]
    extra_values = state_values[len(_FILTER_VALUE_STATES):]
    queue_state = _queue_filter_ui_state_from_values(*queue_values)
    gui_state = _review_gui_state_from_values(
        theme_mode=extra_values[0],
        plot_mode=extra_values[1],
        plot_overlays=extra_values[2],
        baseline_opacity=extra_values[3],
        residual_height=extra_values[4],
        external_source_view=extra_values[5],
        camera_values=extra_values[6],
        band_values=extra_values[7],
        yaxis_mode=extra_values[8],
        period_method=extra_values[9],
        pdm_min_period=extra_values[10],
        pdm_max_period=extra_values[11],
        pdm_manual_period=extra_values[12],
    )
    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            save_app_state(conn, _QUEUE_FILTER_APP_STATE_KEY, json.dumps(queue_state, default=str))
            save_app_state(conn, _REVIEW_GUI_STATE_APP_STATE_KEY, json.dumps(gui_state, default=str))
    except Exception as exc:
        return f'Failed to save GUI state: {exc}', no_update
    return f'Saved GUI state to {Path(DB_PATH).expanduser().resolve()}.', gui_state


app.clientside_callback(
    """
    function(boundsData, minValue, maxValue, rangeValue, rangeId) {
        function toNumber(value) {
            if (value === null || value === undefined || value === '') {
                return null;
            }
            const number = Number(value);
            return Number.isFinite(number) ? number : null;
        }

        function clamp(value, lo, hi, fallback) {
            const number = toNumber(value);
            const base = number === null ? fallback : number;
            return Math.min(Math.max(base, lo), hi);
        }

        const col = rangeId && rangeId.col ? rangeId.col : null;
        if (!col) {
            return [minValue, maxValue, 0, 1, [0, 1], 1, true];
        }

        const info = boundsData && typeof boundsData === 'object' ? boundsData[col] : null;
        let dataLo = info ? toNumber(info.min) : null;
        let dataHi = info ? toNumber(info.max) : null;
        if (dataLo === null || dataHi === null) {
            return [minValue, maxValue, 0, 1, [0, 1], 1, true];
        }
        if (dataHi < dataLo) {
            const temp = dataLo;
            dataLo = dataHi;
            dataHi = temp;
        }

        let sliderLo = dataLo;
        let sliderHi = dataHi;
        let disabled = false;
        if (sliderHi === sliderLo) {
            const pad = Math.max(Math.abs(sliderLo) * 0.01, 1.0);
            sliderLo -= pad;
            sliderHi += pad;
            disabled = true;
        }
        const step = sliderHi > sliderLo ? (sliderHi - sliderLo) / 200.0 : 1;

        const currentMin = toNumber(minValue);
        const currentMax = toNumber(maxValue);
        const currentRange = Array.isArray(rangeValue) ? rangeValue : [];
        let currentRangeMin = toNumber(currentRange[0]);
        let currentRangeMax = toNumber(currentRange[1]);
        const rangeLooksInitialized = (
            currentRangeMin !== null &&
            currentRangeMax !== null &&
            currentRangeMin >= sliderLo &&
            currentRangeMax <= sliderHi
        );
        if (!rangeLooksInitialized) {
            currentRangeMin = null;
            currentRangeMax = null;
        }

        const ctx = dash_clientside.callback_context;
        const triggered = ctx ? ctx.triggered_id : null;
        const triggeredType = triggered && typeof triggered === 'object' ? triggered.type : null;

        let lower = dataLo;
        let upper = dataHi;

        if (triggeredType === 'num-filter-range') {
            lower = clamp(currentRangeMin, sliderLo, sliderHi, currentMin !== null ? currentMin : dataLo);
            upper = clamp(currentRangeMax, sliderLo, sliderHi, currentMax !== null ? currentMax : dataHi);
            if (lower > upper) {
                const temp = lower;
                lower = upper;
                upper = temp;
            }
        } else if (triggeredType === 'num-filter-min-input') {
            lower = currentMin === null ? dataLo : clamp(currentMin, sliderLo, sliderHi, dataLo);
            upper = currentMax !== null
                ? clamp(currentMax, sliderLo, sliderHi, dataHi)
                : (currentRangeMax !== null ? clamp(currentRangeMax, sliderLo, sliderHi, dataHi) : dataHi);
            if (lower > upper) {
                upper = lower;
            }
        } else if (triggeredType === 'num-filter-max-input') {
            lower = currentMin !== null
                ? clamp(currentMin, sliderLo, sliderHi, dataLo)
                : (currentRangeMin !== null ? clamp(currentRangeMin, sliderLo, sliderHi, dataLo) : dataLo);
            upper = currentMax === null ? dataHi : clamp(currentMax, sliderLo, sliderHi, dataHi);
            if (lower > upper) {
                lower = upper;
            }
        } else {
            lower = currentMin !== null
                ? clamp(currentMin, sliderLo, sliderHi, dataLo)
                : dataLo;
            upper = currentMax !== null
                ? clamp(currentMax, sliderLo, sliderHi, dataHi)
                : dataHi;
            if (lower > upper) {
                const temp = lower;
                lower = upper;
                upper = temp;
            }
        }

        return [lower, upper, sliderLo, sliderHi, [lower, upper], step, disabled];
    }
    """,
    [Output({'type': 'num-filter-min-input', 'col': MATCH}, 'value'),
     Output({'type': 'num-filter-max-input', 'col': MATCH}, 'value'),
     Output({'type': 'num-filter-range', 'col': MATCH}, 'min'),
     Output({'type': 'num-filter-range', 'col': MATCH}, 'max'),
     Output({'type': 'num-filter-range', 'col': MATCH}, 'value'),
     Output({'type': 'num-filter-range', 'col': MATCH}, 'step'),
     Output({'type': 'num-filter-range', 'col': MATCH}, 'disabled')],
    [Input('numeric-filter-bounds', 'data'),
     Input({'type': 'num-filter-min-input', 'col': MATCH}, 'value'),
     Input({'type': 'num-filter-max-input', 'col': MATCH}, 'value'),
     Input({'type': 'num-filter-range', 'col': MATCH}, 'value')],
    State({'type': 'num-filter-range', 'col': MATCH}, 'id'),
    prevent_initial_call=False,
)

# Initialize queue
@app.callback(
    Output('queue-data', 'data'),
    [Input('refresh-btn', 'n_clicks'),
     Input('import-trigger', 'data'),
     Input('queue-source-path', 'data'),
     Input('restored-filter-applied', 'data'),
     Input('numeric-filter-bounds', 'data')],
    _queue_states + _TEXT_OPTION_STATES + _SELECT_OPTION_STATES,
    prevent_initial_call=False
)
def load_queue(refresh_clicks, import_trigger, queue_source_scope, restored_filters, numeric_bounds, *callback_states):
    """Load queue data from all sidebar filter states."""
    n_values = len(_queue_states)
    n_text_opts = len(_TEXT_OPTION_STATES)
    state_values = callback_states[:n_values]
    text_option_values = callback_states[n_values:n_values + n_text_opts]
    select_option_values = callback_states[n_values + n_text_opts:]

    with closing(db_connect(Path(DB_PATH))) as conn:
        ui_state = _queue_filter_ui_state_from_values(*state_values)
        ui_state = _merge_unhydrated_saved_queue_filter_ui_state(
            ui_state,
            restored_filters,
            text_option_values,
            select_option_values,
        )
        numeric_bounds = numeric_bounds or {}
        filter_params = _queue_filter_params_from_ui_state(
            ui_state,
            numeric_bounds,
            queue_source_scope,
        )

        queue_data = create_queue_data_dict(conn, filter_params)
        active_filters = {
            k: v for k, v in filter_params.items()
            if v not in (None, 'Any', [])
        }
        print(f"[queue] size={queue_data['queue_size']} active_filters={active_filters}")
        return queue_data


@app.callback(
    Output('queue-filter-provenance-panel', 'children'),
    Input('queue-data', 'data'),
    prevent_initial_call=False,
)
def render_queue_filter_provenance(queue_data):
    """Render queue-filter attrition details in the sidebar."""
    return _render_queue_filter_provenance_panel(queue_data)


@app.callback(
    Output('preload-trigger', 'data'),
    Input('current-index', 'data'),
    Input('queue-data', 'data'),
    State('plot-mode', 'value'),
    prevent_initial_call=False,
)
def preload_next_candidates(idx, queue_data, plot_mode):
    """Warm the next candidate's LC/baseline caches after a short navigation debounce."""
    generation = _next_preload_generation()
    if str(plot_mode or 'native').strip().lower() != 'native':
        return no_update
    if queue_data is None or not isinstance(queue_data, dict):
        return no_update
    candidate_ids = queue_data.get('candidate_ids')
    if not candidate_ids:
        return no_update
    idx = int(idx or 0)
    db_path = DB_PATH

    def do_preload():
        time.sleep(_PRELOAD_DELAY_SEC)
        if not _is_current_preload_generation(generation):
            return
        try:
            with closing(db_connect(Path(db_path))) as conn:
                for i in range(1, _PRELOAD_LOOKAHEAD + 1):
                    if not _is_current_preload_generation(generation):
                        return
                    if idx + i >= len(candidate_ids):
                        break
                    cid = candidate_ids[idx + i]
                    row = conn.execute(
                        "SELECT payload_json, source_path FROM candidates WHERE candidate_id = ?",
                        (str(cid),),
                    ).fetchone()
                    if not row:
                        continue
                    try:
                        payload = json.loads(row[0]) if row[0] else {}
                    except Exception:
                        continue
                    source_path = (row[1] or "").strip()
                    run_dir = _run_dir_from_source_path(source_path) if source_path else None
                    plot_dir = (run_dir / "plots") if run_dir and (run_dir / "plots").exists() else None
                    run_params = None
                    if run_dir and (run_dir / "run_params.json").exists():
                        try:
                            with open(run_dir / "run_params.json") as f:
                                run_params = json.load(f)
                        except Exception:
                            pass
                    try:
                        warm_caches_for_candidate(payload, plot_dir, run_params=run_params)
                    except Exception:
                        pass
        except Exception:
            pass

    t = Thread(target=do_preload, daemon=True)
    t.start()
    return no_update


def _queue_candidate_id(queue_data, idx) -> str | None:
    """Return the active candidate ID from queue-data dict storage."""
    if not isinstance(queue_data, dict):
        return None
    candidate_ids = queue_data.get('candidate_ids') or []
    if idx is None:
        return None
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(candidate_ids):
        return None
    return str(candidate_ids[idx])


def _normalized_queue_index(queue_data, idx) -> int | None:
    """Clamp a queue index to the currently visible queue, if needed."""
    if not isinstance(queue_data, dict):
        return None

    candidate_ids = queue_data.get('candidate_ids') or []
    try:
        current_idx = int(idx or 0)
    except (TypeError, ValueError):
        current_idx = 0

    if not candidate_ids:
        return 0 if current_idx != 0 else None

    clamped_idx = min(max(current_idx, 0), len(candidate_ids) - 1)
    if clamped_idx == current_idx:
        return None
    return clamped_idx


@app.callback(
    Output('current-index', 'data', allow_duplicate=True),
    [Input('queue-data', 'data'),
     Input('current-index', 'data')],
    prevent_initial_call='initial_duplicate',
)
def clamp_current_index_to_queue(queue_data, idx):
    """Keep the active index valid when filters change the visible queue."""
    normalized_idx = _normalized_queue_index(queue_data, idx)
    if normalized_idx is None:
        raise dash.exceptions.PreventUpdate
    return normalized_idx


def _has_external_period(payload: dict | None) -> bool:
    """Whether a payload already has a catalog or validated pipeline period."""
    return shared_has_external_period(payload)


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _phase_template(
    phase: np.ndarray,
    resid: np.ndarray,
    *,
    n_bins: int = 48,
    min_bin_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    template = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    phase = np.asarray(phase, dtype=float)
    resid = np.asarray(resid, dtype=float)
    valid = np.isfinite(phase) & np.isfinite(resid)
    if np.count_nonzero(valid) == 0:
        return template, counts

    phase_valid = np.mod(phase[valid], 1.0)
    resid_valid = resid[valid]
    idx = np.floor(phase_valid * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)

    for b in range(n_bins):
        vals = resid_valid[idx == b]
        if vals.size >= min_bin_points:
            template[b] = float(np.median(vals))
            counts[b] = int(vals.size)

    return template, counts


def _template_phase_lag(template_a: np.ndarray, template_b: np.ndarray) -> float:
    template_a = np.asarray(template_a, dtype=float)
    template_b = np.asarray(template_b, dtype=float)
    if template_a.size == 0 or template_a.size != template_b.size:
        return np.nan

    n = int(template_a.size)
    best_corr = -np.inf
    best_shift = 0
    min_overlap = max(6, n // 4)

    for shift in range(n):
        shifted = np.roll(template_b, shift)
        mask = np.isfinite(template_a) & np.isfinite(shifted)
        if np.count_nonzero(mask) < min_overlap:
            continue
        a = template_a[mask]
        b = shifted[mask]
        a = a - np.mean(a)
        b = b - np.mean(b)
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa <= 0 or sb <= 0:
            continue
        corr = float(np.mean((a / sa) * (b / sb)))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    if not np.isfinite(best_corr):
        return np.nan
    lag_bins = min(best_shift, n - best_shift)
    return float(lag_bins / n)


def _score_period_harmonic_candidate(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    period: float,
    *,
    n_bins: int = 48,
    lag_weight: float = 0.1,
) -> dict[str, float]:
    if not np.isfinite(period) or period <= 0:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}

    all_jd = [jd for jd, _ in band_resid.values() if jd.size > 0]
    if not all_jd:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}
    jd0 = float(min(np.min(jd) for jd in all_jd))

    templates: dict[int, np.ndarray] = {}
    scatter_ratios: list[float] = []

    for band, (jd, resid) in band_resid.items():
        phase = np.mod((jd - jd0) / float(period), 1.0)
        template, _ = _phase_template(phase, resid, n_bins=n_bins)
        templates[band] = template

        bin_idx = np.floor(phase * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        model = template[bin_idx]
        valid = np.isfinite(model) & np.isfinite(resid)
        if np.count_nonzero(valid) < 20:
            continue

        raw_sigma = _robust_sigma(resid[valid])
        folded_sigma = _robust_sigma(resid[valid] - model[valid])
        if np.isfinite(raw_sigma) and raw_sigma > 0 and np.isfinite(folded_sigma):
            scatter_ratios.append(float(folded_sigma / raw_sigma))

    if not scatter_ratios:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}

    scatter_ratio = float(np.mean(scatter_ratios))
    lag_phase = np.nan
    if 0 in templates and 1 in templates:
        lag_phase = _template_phase_lag(templates[0], templates[1])
    lag_term = 0.0 if not np.isfinite(lag_phase) else float(lag_phase)

    objective = float(scatter_ratio + lag_weight * lag_term)
    return {"objective": objective, "scatter_ratio": scatter_ratio, "lag_phase": lag_phase}


def _arbitrate_harmonic_period(
    band_dfs: dict[int, pd.DataFrame],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
) -> tuple[float, float, dict[str, float]]:
    if not np.isfinite(base_period) or base_period <= 0:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    band_resid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for band in (0, 1):
        bdf = band_dfs.get(band)
        if bdf is None or bdf.empty or "resid" not in bdf.columns:
            continue
        jd = bdf["JD"].to_numpy(dtype=float)
        resid = bdf["resid"].to_numpy(dtype=float)
        mask = np.isfinite(jd) & np.isfinite(resid)
        if np.count_nonzero(mask) < 30:
            continue
        band_resid[band] = (jd[mask], resid[mask])

    if not band_resid:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    candidates: list[tuple[float, float, dict[str, float]]] = []
    for factor in (1.0, 2.0, 0.5):
        p = float(base_period) * float(factor)
        if not np.isfinite(p) or p <= 0:
            continue
        if p < float(min_period) or p > float(max_period):
            continue
        if any(abs(p - p_prev) <= 1e-10 * max(1.0, abs(p), abs(p_prev)) for _, p_prev, _ in candidates):
            continue
        score = _score_period_harmonic_candidate(band_resid, p)
        candidates.append((float(factor), p, score))

    if not candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    finite_candidates = [c for c in candidates if np.isfinite(c[2].get("objective", np.nan))]
    if not finite_candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    best_factor, best_period, best_score = min(finite_candidates, key=lambda x: float(x[2]["objective"]))
    base_entry = next((c for c in finite_candidates if abs(c[0] - 1.0) < 1e-12), None)
    base_objective = float(base_entry[2]["objective"]) if base_entry is not None else np.nan

    # Avoid jittery harmonic flips unless improvement is meaningful.
    if base_entry is not None and abs(best_factor - 1.0) > 1e-12:
        improvement = (base_objective - float(best_score["objective"])) / max(abs(base_objective), 1e-9)
        if not np.isfinite(improvement) or improvement < 0.02:
            best_factor, best_period, best_score = base_entry

    diag = {
        "objective": float(best_score.get("objective", np.nan)),
        "scatter_ratio": float(best_score.get("scatter_ratio", np.nan)),
        "lag_phase": float(best_score.get("lag_phase", np.nan)),
        "base_objective": base_objective,
    }
    return float(best_period), float(best_factor), diag


def _run_period_search_for_payload(
    payload: dict,
    *,
    min_period: float,
    max_period: float,
    method: str,
) -> tuple[dict | None, str]:
    """Run a period search against the current candidate payload."""
    plot_dir_path = _configured_plot_dir()
    run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
    filter_bad_cameras = not bool(run_params.get('skip_camera_median', False))
    scatter_ratio = float(run_params.get('bad_camera_scatter_ratio', BAD_CAMERA_SCATTER_RATIO_THRESHOLD)) if run_params else BAD_CAMERA_SCATTER_RATIO_THRESHOLD
    clean_abs = float(run_params.get('clean_max_error_absolute', CLEAN_LC_MAX_ERROR_ABSOLUTE)) if run_params else CLEAN_LC_MAX_ERROR_ABSOLUTE
    clean_sig = float(run_params.get('clean_max_error_sigma', CLEAN_LC_MAX_ERROR_SIGMA)) if run_params else CLEAN_LC_MAX_ERROR_SIGMA
    baseline_name, baseline_kwargs, _baseline_warnings = _baseline_config_from_run_params(run_params)
    return shared_run_period_search_for_payload(
        payload,
        plot_dir=plot_dir_path,
        min_period=min_period,
        max_period=max_period,
        method=method,
        filter_bad_cameras=filter_bad_cameras,
        scatter_ratio=scatter_ratio,
        clean_max_error_absolute=clean_abs,
        clean_max_error_sigma=clean_sig,
        baseline_name=baseline_name,
        baseline_kwargs=baseline_kwargs,
    )



@app.callback(
    [Output('queue-data', 'data', allow_duplicate=True),
     Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input('candidate-search-btn', 'n_clicks'),
     Input('candidate-search-query', 'n_submit')],
    [State('candidate-search-query', 'value'),
     State('queue-data', 'data')],
    prevent_initial_call=True,
)
def _lookup_candidate_id_for_query(conn: sqlite3.Connection, query_text: str) -> tuple[str | None, str | None]:
    """Resolve a search query to a candidate_id and match label."""
    normalized = _format_large_integer_like_display(query_text).strip()
    exact_queries = (
        ("candidate_id", "SELECT candidate_id FROM candidates WHERE candidate_id = ? COLLATE NOCASE LIMIT 1"),
        ("ASAS-SN ID", "SELECT candidate_id FROM candidates WHERE asas_sn_id = ? COLLATE NOCASE LIMIT 1"),
        ("local LC path", "SELECT candidate_id FROM candidates WHERE lc_path = ? COLLATE NOCASE LIMIT 1"),
        ("Gaia ID", "SELECT candidate_id FROM candidates WHERE CAST(json_extract(payload_json, '$.gaia_id') AS TEXT) = ? COLLATE NOCASE LIMIT 1"),
        ("cluster LC path", "SELECT candidate_id FROM candidates WHERE CAST(json_extract(payload_json, '$.path') AS TEXT) = ? COLLATE NOCASE LIMIT 1"),
    )
    for label, query in exact_queries:
        row = conn.execute(query, (normalized,)).fetchone()
        if row is not None:
            return str(row[0]), label

    stem_query = normalized.lower()
    rows = conn.execute(
        "SELECT candidate_id, lc_path, source_path, "
        "CAST(json_extract(payload_json, '$.path') AS TEXT), "
        "CAST(json_extract(payload_json, '$.gaia_id') AS TEXT), "
        "payload_json "
        "FROM candidates"
    ).fetchall()
    for candidate_id, local_lc_path, source_path, cluster_lc_path, gaia_id, payload_json in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if candidate_id is not None and not payload.get("candidate_id"):
            payload["candidate_id"] = str(candidate_id)
        if cluster_lc_path and not payload.get("path"):
            payload["path"] = cluster_lc_path
        display_cluster_lc_path, effective_local_lc_path = _display_lc_paths(
            payload,
            stored_lc_path=local_lc_path,
            source_path=source_path,
        )
        for label, raw in (
            ("local LC stem", effective_local_lc_path),
            ("cluster LC stem", display_cluster_lc_path),
            ("Gaia ID", gaia_id),
        ):
            if not raw:
                continue
            text = _format_large_integer_like_display(raw).strip()
            if not text:
                continue
            if text.lower() == stem_query:
                return str(candidate_id), label
            try:
                path_obj = Path(text)
            except Exception:
                continue
            if path_obj.stem.lower() == stem_query or path_obj.name.lower() == stem_query:
                return str(candidate_id), label
    return None, None


def open_existing_candidate(n_clicks, n_submit, query, queue_data):
    """Jump to an existing candidate in the DB by common identifiers or LC stem."""
    _ = n_clicks, n_submit
    query_text = str(query or '').strip()
    if not query_text:
        raise dash.exceptions.PreventUpdate

    with closing(db_connect(Path(DB_PATH))) as conn:
        candidate_id, match_label = _lookup_candidate_id_for_query(conn, query_text)

    if candidate_id is None:
        return no_update, no_update, f"Candidate not found in DB: {query_text}"

    candidate_ids = list((queue_data or {}).get('candidate_ids') or []) if isinstance(queue_data, dict) else []
    if candidate_id in candidate_ids:
        return no_update, candidate_ids.index(candidate_id), f"Jumped to {candidate_id} in the current queue via {match_label or 'search'}."

    return (
        {'candidate_ids': [candidate_id], 'queue_size': 1, 'filter_hash': f'view:{candidate_id}'},
        0,
        f"Viewing {candidate_id} via {match_label or 'search'}. Refresh Queue to restore the filtered queue.",
    )


@app.callback(
    Output('last-candidate-saved', 'data'),
    Input('current-candidate-id', 'data'),
    State('startup-selection-applied', 'data'),
    prevent_initial_call=True,
)
def persist_last_candidate(current_candidate_id, startup_selection_applied):
    """Persist the most recently viewed candidate for this review DB."""
    if not startup_selection_applied:
        raise dash.exceptions.PreventUpdate
    candidate_id = str(current_candidate_id or '').strip()
    if not candidate_id:
        raise dash.exceptions.PreventUpdate
    with closing(db_connect(Path(DB_PATH))) as conn:
        save_app_state(conn, 'review_last_candidate', candidate_id)
    return int(time.time())


@app.callback(
    [Output('queue-data', 'data', allow_duplicate=True),
     Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('startup-selection-applied', 'data')],
    Input('queue-data', 'data'),
    State('startup-selection-applied', 'data'),
    prevent_initial_call='initial_duplicate',
)
def restore_startup_candidate(queue_data, already_applied):
    """Open CLI-requested candidate or last viewed candidate when the queue is ready."""
    if already_applied:
        raise dash.exceptions.PreventUpdate
    if queue_data is None:
        raise dash.exceptions.PreventUpdate

    explicit_query = str(INITIAL_CANDIDATE_QUERY or '').strip()
    candidate_ids = list((queue_data or {}).get('candidate_ids') or []) if isinstance(queue_data, dict) else []
    queue_size = int((queue_data or {}).get('queue_size') or 0) if isinstance(queue_data, dict) else 0

    # Wait for the real queue to materialize before consuming startup restore.
    # Queue initialization can briefly emit an empty queue before persisted filters
    # are restored, and marking startup selection as applied here would prevent
    # the saved candidate from reopening once the populated queue arrives.
    if not explicit_query and queue_size <= 0 and not candidate_ids:
        return no_update, no_update, no_update, no_update

    with closing(db_connect(Path(DB_PATH))) as conn:
        saved_candidate = str(load_app_state(conn, 'review_last_candidate', '') or '').strip()
        if explicit_query:
            candidate_id, match_label = _lookup_candidate_id_for_query(conn, explicit_query)
            if candidate_id is None:
                return no_update, no_update, f"Candidate not found in DB: {explicit_query}", True
            if candidate_id in candidate_ids:
                return no_update, candidate_ids.index(candidate_id), f"Opened {candidate_id} via {match_label or 'startup selection'}.", True
            return (
                {'candidate_ids': [candidate_id], 'queue_size': 1, 'filter_hash': f'view:{candidate_id}'},
                0,
                f"Opened {candidate_id} via {match_label or 'startup selection'}. Refresh Queue to restore the filtered queue.",
                True,
            )

    if saved_candidate and saved_candidate in candidate_ids:
        return no_update, candidate_ids.index(saved_candidate), f"Restored last candidate {saved_candidate}.", True
    return no_update, no_update, no_update, True


@app.callback(
    Output('explorer-launch-status', 'children'),
    [Input('open-candidate-in-explorer-btn', 'n_clicks'),
     Input('open-db-in-explorer-btn', 'n_clicks')],
    State('current-candidate-id', 'data'),
    prevent_initial_call=True,
)
def open_in_explorer(_open_candidate_clicks, _open_db_clicks, current_candidate_id):
    """Launch explorer on this DB, optionally focused on the current candidate."""
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else ''
    candidate_id = str(current_candidate_id or '').strip()
    if triggered == 'open-candidate-in-explorer-btn' and not candidate_id:
        return 'No candidate selected to open in explorer.'

    command, url = build_explorer_command(
        sources=[DB_PATH],
        candidate=(candidate_id if triggered == 'open-candidate-in-explorer-btn' else None),
        plot_dir=PLOT_DIR,
    )
    launch_detached(command)
    return f"Opened explorer at {url}."


@app.callback(
    Output('explorer-selection-panel', 'children'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)
def load_explorer_selection_panel(_tick):
    """Display selection provenance when this review DB came from explorer."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        raw = str(load_app_state(conn, 'explorer_selection_meta', '') or '').strip()
    if not raw:
        return _render_explorer_selection_panel(None)
    try:
        selection_meta = json.loads(raw)
    except Exception:
        selection_meta = None
    return _render_explorer_selection_panel(selection_meta)


def _do_save(candidate_id, score, event_class, needs_followup, notes, event_type, *, increment_pass=False):
    """Shared save helper.  Auto-sets status; only increments review_pass on Done."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, candidate_id)
        current_pass = max(1, review.get('review_pass', 0))
        new_pass = current_pass + 1 if increment_pass else current_pass
        status = 'needs_followup' if needs_followup else 'reviewed'
        save_review(
            conn,
            candidate_id=candidate_id,
            interest_score=score,
            event_class=event_class or 'unclassified',
            review_pass=new_pass,
            notes=notes or '',
            status=status,
            reviewer='calder',
            event_type=event_type,
        )
        _clear_review_state_caches()
        return new_pass, status


app.clientside_callback(
    """
    function(keyValue, currentIdx, queueSize, currentCandidateId, currentScore,
             eventClass, pendingPrefix, needsFollowup, notes, saveRequest) {
        var no = window.dash_clientside.no_update;
        var key = '';
        if (keyValue) {
            key = String(keyValue).split('\\t', 1)[0].trim();
        }
        if (!key || key === '?' || key === 'Escape' || key.toLowerCase() === 'r') {
            return [no, no, no, no, no, no, no];
        }

        var size = parseInt(queueSize == null ? 0 : queueSize, 10);
        if (!Number.isFinite(size) || size <= 0) {
            return [no, 'Queue is empty', no, no, no, no, no];
        }

        var idx = parseInt(currentIdx == null ? 0 : currentIdx, 10);
        if (!Number.isFinite(idx)) {
            idx = 0;
        }
        var candidateId = currentCandidateId == null ? '' : String(currentCandidateId);
        var currentClass = eventClass ? String(eventClass) : 'unclassified';
        var nextClass = currentClass;
        var nextScore = currentScore;
        var nextFollowup = !!needsFollowup;
        var nextIdx = idx;
        var notice = no;
        var saveReq = no;
        var prefixOut = no;

        var nextNonce = 1;
        if (saveRequest && typeof saveRequest === 'object' && typeof saveRequest.nonce === 'number') {
            nextNonce = saveRequest.nonce + 1;
        }

        var buildSaveRequest = function(scoreValue, incrementPass) {
            return {
                nonce: nextNonce,
                candidate_id: candidateId,
                score: scoreValue,
                event_class: nextClass,
                needs_followup: nextFollowup,
                notes: notes || '',
                increment_pass: !!incrementPass,
                event_type: 'keyboard',
            };
        };

        var lower = key.toLowerCase();
        var classMap = {
            d: 'dipper',
            m: 'microlensing',
            f: 'flare',
            l: 'ltv',
            u: 'unknown_interesting',
            i: 'instrumental',
            o: 'other'
        };
        if (Object.prototype.hasOwnProperty.call(classMap, lower)) {
            var classTag = classMap[lower];
            nextClass = (currentClass === classTag) ? 'unclassified' : classTag;
            notice = 'Class: ' + nextClass;
            return [no, notice, no, no, nextClass, prefixOut, no];
        }

        if (key === ',') {
            nextFollowup = !nextFollowup;
            notice = 'Followup: ' + (nextFollowup ? 'ON' : 'OFF');
            return [no, notice, no, nextFollowup, no, prefixOut, no];
        }

        if (key === 'Backspace') {
            nextIdx = Math.max(0, idx - 1);
            notice = '← Previous';
            return [nextIdx !== idx ? nextIdx : no, notice, no, no, no, prefixOut, no];
        }

        if (key === 'Tab') {
            nextIdx = Math.min(idx + 1, size - 1);
            notice = '→ Next';
            return [nextIdx !== idx ? nextIdx : no, notice, no, no, no, prefixOut, no];
        }

        if (key === '.') {
            if (!candidateId) {
                return [no, 'Queue is empty', no, no, no, prefixOut, no];
            }
            notice = '✓ Saved';
            saveReq = buildSaveRequest(currentScore, false);
            return [no, notice, no, no, no, prefixOut, saveReq];
        }

        if (key === 'Enter') {
            if (currentScore == null || currentScore === '') {
                return [no, '⚠ Confidence required', no, no, no, prefixOut, no];
            }
            if (!currentClass || currentClass === 'unclassified') {
                return [no, '⚠ Class required', no, no, no, prefixOut, no];
            }
            nextIdx = Math.min(idx + 1, size - 1);
            notice = '✓ Saved + Next →';
            saveReq = buildSaveRequest(currentScore, true);
            return [nextIdx !== idx ? nextIdx : no, notice, no, no, no, prefixOut, saveReq];
        }

        if (key === '1' || key === '2' || key === '3' || key === '4') {
            nextScore = parseInt(key, 10);
            notice = '✓ Confidence: ' + String(nextScore);
            saveReq = buildSaveRequest(nextScore, false);
            return [no, notice, nextScore, no, no, prefixOut, saveReq];
        }

        return [no, no, no, no, no, prefixOut, no];
    }
    """,
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('current-score', 'data', allow_duplicate=True),
     Output('needs-followup-store', 'data', allow_duplicate=True),
     Output('event-class-store', 'data', allow_duplicate=True),
     Output('pending-prefix', 'data', allow_duplicate=True),
     Output('review-save-request', 'data', allow_duplicate=True)],
    Input('keyboard-input', 'value'),
    [State('current-index', 'data'),
     State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('pending-prefix', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value'),
     State('review-save-request', 'data')],
    prevent_initial_call=True,
)


@app.callback(
    [Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('review-save-request', 'data'),
    State('current-candidate-id', 'data'),
    prevent_initial_call=True,
)
def persist_review_save_request(save_request, current_candidate_id):
    """Persist clientside-queued review saves without blocking UI feedback."""
    if not isinstance(save_request, dict):
        raise dash.exceptions.PreventUpdate

    try:
        nonce = int(save_request.get('nonce', 0) or 0)
    except Exception:
        nonce = 0
    if nonce <= 0:
        raise dash.exceptions.PreventUpdate

    candidate_id = str(save_request.get('candidate_id') or '').strip()
    if not candidate_id:
        raise dash.exceptions.PreventUpdate

    raw_score = save_request.get('score')
    try:
        score = int(raw_score) if raw_score not in (None, '') else None
    except Exception:
        score = None

    try:
        new_pass, _status = _do_save(
            candidate_id,
            score,
            save_request.get('event_class'),
            bool(save_request.get('needs_followup')),
            save_request.get('notes'),
            str(save_request.get('event_type') or 'keyboard'),
            increment_pass=bool(save_request.get('increment_pass')),
        )
    except Exception as exc:
        traceback.print_exc()
        return f"✗ Save failed: {exc}", no_update

    pass_out = no_update
    if str(current_candidate_id or '') == candidate_id:
        pass_out = new_pass
    return no_update, pass_out


@app.callback(
    Output('plot-render-request', 'data'),
    [Input('current-index', 'data'),
     Input('current-candidate-id', 'data'),
     Input('plot-mode', 'value'),
     Input('plot-overlays', 'value'),
     Input('camera-checklist', 'value'),
     Input('plot-preset', 'value'),
     Input('residual-height-slider', 'value'),
     Input('theme-mode-store', 'data'),
     Input('queue-size-store', 'data'),
     Input('pipeline-progress-trigger', 'data'),
     Input('baseline-opacity-slider', 'value'),
     Input('band-checklist', 'value'),
     Input('round-sigfigs', 'value'),
     Input('link-radius-arcsec', 'value'),
     Input('pdm-result-store', 'data'),
     Input('pdm-manual-period', 'value'),
     Input('yaxis-mode', 'value'),
     Input('external-source-view', 'value')],
     State('plot-render-request', 'data'),
    prevent_initial_call=True,
)
def queue_plot_render_request(idx, current_candidate_id, plot_mode, overlay_values, selected_cameras, preset, residual_height, theme_mode, _queue_size, _pipeline_progress, baseline_opacity, selected_bands, round_sigfigs, link_radius, pdm_result, pdm_manual_period, yaxis_mode, external_source_view, existing_request):
    """Debounced render request queue for native plot UX."""
    req = existing_request or {'nonce': 0, 'ts': 0.0}
    # Determine effective PDM period: manual override > PDM result
    override_period = None
    if pdm_manual_period is not None:
        try:
            p = float(pdm_manual_period)
            if p > 0:
                override_period = p
        except (TypeError, ValueError):
            pass
    if override_period is None and pdm_result and isinstance(pdm_result, dict):
        override_period = pdm_result.get('best_period')
    return {
        'nonce': int(req.get('nonce', 0)) + 1,
        'ts': float(time.time()),
        'state': {
            'idx': idx,
            'candidate_id': str(current_candidate_id) if current_candidate_id is not None else None,
            'plot_mode': plot_mode,
            'overlay_values': list(overlay_values or []),
            'selected_cameras': list(selected_cameras or []),
            'selected_bands': list(selected_bands or ['g', 'V']),
            'preset': preset,
            'residual_height': float(residual_height if residual_height is not None else DEFAULT_RESIDUAL_FRACTION),
            'theme': theme_mode or DEFAULT_THEME,
            'baseline_opacity': float(baseline_opacity if baseline_opacity is not None else 0.5),
            'round_sigfigs': bool(True if round_sigfigs is None else ('yes' in round_sigfigs)),
            'link_radius': float(link_radius) if link_radius is not None else 10.0,
            'override_period': override_period,
            'yaxis_mode': str(yaxis_mode or 'mag'),
            'external_source_view': str(external_source_view or DEFAULT_EXTERNAL_SOURCE_VIEW),
        },
    }


def run_period_search(n_clicks, candidate_id, min_period, max_period, method, auto_period_cache):
    """Run period search (LSP/PDM/CE) on current candidate's light curve."""
    if not n_clicks or not candidate_id:
        raise dash.exceptions.PreventUpdate
    candidate_id = str(candidate_id)
    auto_period_cache = dict(auto_period_cache or {})
    min_p, max_p = _normalize_period_search_bounds(min_period, max_period)
    method = str(method or 'pdm').lower()

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    result, label = _run_period_search_for_payload(payload, min_period=min_p, max_period=max_p, method=method)
    auto_period_cache[candidate_id] = {'result': result, 'label': label}
    return result, label, auto_period_cache


_PERIOD_SEARCH_OUTPUTS = [
    Output('pdm-result-store', 'data'),
    Output('pdm-result-label', 'children'),
    Output('auto-period-cache', 'data', allow_duplicate=True),
]


if _background_callback_manager is not None:
    @app.callback(
        _PERIOD_SEARCH_OUTPUTS,
        Input('pdm-run-btn', 'n_clicks'),
        [State('current-candidate-id', 'data'),
         State('pdm-min-period', 'value'),
         State('pdm-max-period', 'value'),
         State('period-method', 'value'),
         State('auto-period-cache', 'data')],
        background=True,
        running=[
            (Output('pdm-run-btn', 'disabled'), True, False),
            (Output('period-search-indicator', 'children'), 'Searching period...', ''),
        ],
        cancel=[Input('current-candidate-id', 'data')],
        prevent_initial_call=True,
    )
    def run_period_search_callback(n_clicks, candidate_id, min_period, max_period, method, auto_period_cache):
        return run_period_search(n_clicks, candidate_id, min_period, max_period, method, auto_period_cache)
else:
    @app.callback(
        _PERIOD_SEARCH_OUTPUTS,
        Input('pdm-run-btn', 'n_clicks'),
        [State('current-candidate-id', 'data'),
         State('pdm-min-period', 'value'),
         State('pdm-max-period', 'value'),
         State('period-method', 'value'),
         State('auto-period-cache', 'data')],
        prevent_initial_call=True,
    )
    def run_period_search_callback(n_clicks, candidate_id, min_period, max_period, method, auto_period_cache):
        return run_period_search(n_clicks, candidate_id, min_period, max_period, method, auto_period_cache)


def _normalize_period_search_bounds(min_period, max_period) -> tuple[float, float]:
    """Normalize period-search bounds from UI inputs."""
    try:
        min_p = float(min_period) if min_period else 0.1
        max_p = float(max_period) if max_period else 100.0
    except (TypeError, ValueError):
        min_p, max_p = 0.1, 100.0
    if min_p <= 0:
        min_p = 0.01
    if max_p <= min_p:
        max_p = min_p + 1.0
    return min_p, max_p


@app.callback(
    [Output('pdm-result-store', 'data', allow_duplicate=True),
     Output('pdm-result-label', 'children', allow_duplicate=True),
     Output('pdm-manual-period', 'value', allow_duplicate=True),
     Output('auto-period-cache', 'data', allow_duplicate=True),
     Output('auto-period-request', 'data', allow_duplicate=True)],
    Input('current-candidate-id', 'data'),
    [State('pdm-min-period', 'value'),
     State('pdm-max-period', 'value'),
     State('auto-period-cache', 'data'),
     State('auto-period-request', 'data')],
    prevent_initial_call=True,
)
def auto_period_on_navigate(candidate_id, min_period, max_period, auto_period_cache, auto_period_request):
    """Populate cached period status or queue an automatic search on navigation."""
    if candidate_id is None:
        return None, '', None, no_update, {'nonce': 0}
    candidate_id = str(candidate_id)
    auto_period_cache = dict(auto_period_cache or {})
    auto_period_request = dict(auto_period_request or {})

    cached_entry = auto_period_cache.get(candidate_id)
    if isinstance(cached_entry, dict):
        return (
            cached_entry.get('result'),
            str(cached_entry.get('label', '')),
            None,
            no_update,
            no_update,
        )

    min_p, max_p = _normalize_period_search_bounds(min_period, max_period)

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    if _has_external_period(payload):
        return None, 'Catalog/pipeline period', None, no_update, {'nonce': 0}

    request = {
        'nonce': int(auto_period_request.get('nonce', 0) or 0) + 1,
        'candidate_id': candidate_id,
        'min_period': min_p,
        'max_period': max_p,
        'method': 'auto',
    }
    return None, 'Auto-searching period...', None, no_update, request


def run_auto_period_search(auto_period_request, auto_period_cache):
    """Run automatic CE/PDM search for the active candidate request."""
    if not isinstance(auto_period_request, dict):
        raise dash.exceptions.PreventUpdate

    try:
        nonce = int(auto_period_request.get('nonce', 0) or 0)
    except Exception:
        nonce = 0
    if nonce <= 0:
        raise dash.exceptions.PreventUpdate

    candidate_id = str(auto_period_request.get('candidate_id') or '').strip()
    if not candidate_id:
        raise dash.exceptions.PreventUpdate

    auto_period_cache = dict(auto_period_cache or {})
    min_p, max_p = _normalize_period_search_bounds(
        auto_period_request.get('min_period'),
        auto_period_request.get('max_period'),
    )
    method = str(auto_period_request.get('method') or 'auto').strip().lower()

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
    result, label = _run_period_search_for_payload(
        payload,
        min_period=min_p,
        max_period=max_p,
        method=method,
    )
    if method == 'auto' and result is None and not str(label or '').lower().startswith('auto'):
        label = f'Auto search: {label}'
    if isinstance(result, dict):
        result = dict(result)
        result.setdefault('auto', method == 'auto')

    auto_period_cache[candidate_id] = {
        'result': result,
        'label': label,
    }
    return result, label, auto_period_cache


_AUTO_PERIOD_OUTPUTS = [
    Output('pdm-result-store', 'data', allow_duplicate=True),
    Output('pdm-result-label', 'children', allow_duplicate=True),
    Output('auto-period-cache', 'data', allow_duplicate=True),
]


if _background_callback_manager is not None:
    @app.callback(
        _AUTO_PERIOD_OUTPUTS,
        Input('auto-period-request', 'data'),
        State('auto-period-cache', 'data'),
        background=True,
        cancel=[Input('current-candidate-id', 'data'),
                Input('pdm-run-btn', 'n_clicks')],
        prevent_initial_call=True,
    )
    def run_auto_period_search_callback(auto_period_request, auto_period_cache):
        return run_auto_period_search(auto_period_request, auto_period_cache)
else:
    @app.callback(
        _AUTO_PERIOD_OUTPUTS,
        Input('auto-period-request', 'data'),
        State('auto-period-cache', 'data'),
        prevent_initial_call=True,
    )
    def run_auto_period_search_callback(auto_period_request, auto_period_cache):
        return run_auto_period_search(auto_period_request, auto_period_cache)


@app.callback(
    [Output('plot-preset', 'value'),
     Output('plot-overlays', 'value', allow_duplicate=True),
     Output('plot-defaults-initialized', 'data')],
    Input('queue-size-store', 'data'),
    State('plot-defaults-initialized', 'data'),
    prevent_initial_call=True,
)
def initialize_plot_defaults_from_run_params(queue_size, initialized):
    """Initialize native plot defaults from run_params once per session."""
    if initialized:
        raise dash.exceptions.PreventUpdate
    if int(queue_size or 0) <= 0:
        raise dash.exceptions.PreventUpdate

    run_params = _load_run_params_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
    preset, overlays = _derive_defaults_from_run_params(run_params)
    return preset, overlays, True


@app.callback(
    [Output('plot-overlays', 'value'),
     Output('camera-checklist', 'value', allow_duplicate=True),
     Output('band-checklist', 'value', allow_duplicate=True),
     Output('external-source-view', 'value', allow_duplicate=True)],
    [Input('plot-preset', 'value'),
     Input('plot-reset-btn', 'n_clicks'),
     Input('cams-all-btn', 'n_clicks'),
     Input('cams-clear-btn', 'n_clicks'),
     Input('cams-invert-btn', 'n_clicks')],
    [State('camera-checklist', 'options'),
     State('camera-checklist', 'value'),
     State('plot-overlays', 'value')],
    prevent_initial_call=True,
)
def update_plot_controls(preset, n_reset, n_all, n_clear, n_invert, camera_options, camera_values, overlay_values):
    """Preset mapping + camera selection action buttons."""
    _ = n_reset, n_all, n_clear, n_invert
    trig = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else ''
    cams = [str(opt.get('value')) for opt in (camera_options or [])]
    selected = [str(v) for v in (camera_values or []) if str(v) in cams]
    overlays = list(overlay_values or [])

    if trig == 'plot-preset' or trig == 'plot-reset-btn':
        cfg = PLOT_PRESETS.get(preset or 'Diagnostics', PLOT_PRESETS['Diagnostics'])
        new_overlays = list(cfg['overlays'])
        new_cams = list(cams)
        return new_overlays, new_cams, ['g', 'V'], DEFAULT_EXTERNAL_SOURCE_VIEW
    if trig == 'cams-all-btn':
        return overlays, list(cams), no_update, no_update
    if trig == 'cams-clear-btn':
        return overlays, [], no_update, no_update
    if trig == 'cams-invert-btn':
        inv = [c for c in cams if c not in set(selected)]
        return overlays, inv, no_update, no_update
    return no_update, no_update, no_update, no_update


def update_display(render_request, applied_nonce, current_candidate_id, queue_size_data):
    """Render candidate display with debounce and stable uirevision behavior."""
    req = render_request or {'nonce': 0, 'ts': 0.0, 'state': {}}
    nonce = int(req.get('nonce', 0))
    applied = int(applied_nonce or 0)
    if nonce <= applied:
        raise dash.exceptions.PreventUpdate

    state = req.get('state', {}) if isinstance(req.get('state', {}), dict) else {}
    idx = int(state.get('idx', 0) or 0)
    plot_mode = state.get('plot_mode', 'native')
    overlays = set(state.get('overlay_values') or [])
    selected_cameras = list(state.get('selected_cameras') or [])
    selected_bands = list(state.get('selected_bands') or ['g', 'V'])
    theme_mode = str(state.get('theme', DEFAULT_THEME) or DEFAULT_THEME)
    residual_height = float(state.get('residual_height', DEFAULT_RESIDUAL_FRACTION) or DEFAULT_RESIDUAL_FRACTION)
    baseline_opacity = float(state.get('baseline_opacity', 0.5) if state.get('baseline_opacity') is not None else 0.5)
    round_sigfigs = bool(state.get('round_sigfigs', True))
    link_radius = float(state.get('link_radius', 10.0))
    yaxis_mode = str(state.get('yaxis_mode', 'mag') or 'mag')
    external_source_view = str(state.get('external_source_view', DEFAULT_EXTERNAL_SOURCE_VIEW) or DEFAULT_EXTERNAL_SOURCE_VIEW)
    override_period = state.get('override_period')
    if override_period is not None:
        try:
            override_period = float(override_period)
            if override_period <= 0:
                override_period = None
        except (TypeError, ValueError):
            override_period = None

    empty_fig = {
        'data': [],
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'margin': {'l': 40, 'r': 20, 't': 40, 'b': 30},
        },
    }

    queue_size = int(queue_size_data or 0)
    candidate_id = state.get('candidate_id')
    if candidate_id is None and current_candidate_id is not None:
        candidate_id = str(current_candidate_id)
    elif candidate_id is not None:
        candidate_id = str(candidate_id)

    if queue_size <= 0 or not candidate_id:
        return '', 'No candidates in queue', _render_metadata_health(None, context_msg='Queue is empty.'), _render_vetting_banner(None, radius_arcsec=link_radius), '[0/0]', empty_fig, {'display': 'block', 'width': '100%', 'height': '100%'}, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'No candidates in queue.', []), _render_camera_diag_panel({}, []), _render_run_config_panel(None, None, ['Queue is empty']), _render_repro_badge(None, ['Queue is empty']), '', nonce

    if idx < 0 or idx >= queue_size:
        return '', 'Invalid index', _render_metadata_health(None, context_msg='Invalid queue index.'), _render_vetting_banner(None, radius_arcsec=link_radius), f'[{idx}/{queue_size}]', empty_fig, {'display': 'block', 'width': '100%', 'height': '100%'}, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'Invalid queue index.', []), _render_camera_diag_panel({}, []), _render_run_config_panel(None, None, ['Invalid queue index']), _render_repro_badge(None, ['Invalid queue index']), '', nonce

    payload, stored_lc_path, source_path = _candidate_context(candidate_id)

    plot_dir_path = _configured_plot_dir()
    plot_search_root = _plot_search_root_for_payload(payload)

    grouped = extract_review_metadata_grouped(payload, round_sigfigs=round_sigfigs)
    metadata_health = _render_metadata_health(grouped)
    vetting_banner = _render_vetting_banner(payload, radius_arcsec=link_radius)
    grid_items = []
    for group_name, items in grouped:
        field_divs = [
            html.Div([
                html.Span(label, className='meta-field-label'),
                html.Span(str(value), className='meta-field-value'),
            ], className='meta-field-row')
            for label, value in items
        ]

        details_id = {'type': 'meta-details', 'group': str(group_name)}
        if is_group_default_open(group_name):
            grid_items.append(
                html.Details(
                    [html.Summary(f"{group_name} ({len(items)})"), html.Div(field_divs, className='meta-grid')],
                    id=details_id,
                    open=True,
                )
            )
        else:
            grid_items.append(
                html.Details(
                    [html.Summary(f"{group_name} ({len(items)})"), html.Div(field_divs, className='meta-grid')],
                    id=details_id,
                    open=False,
                )
            )

    progress = f"[{idx + 1}/{queue_size}]"

    if plot_mode == 'png':
        run_params, run_params_status, run_params_msg = _load_run_params_meta_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
        run_params_path = _run_params_path_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
        mismatch_warnings = _run_config_mismatch_warnings(run_params if run_params else None, overlays)
        if run_params_status != 'loaded':
            mismatch_warnings.append(run_params_msg)

        stats_group = _render_stat_cards(_build_stat_rows(payload, pd.DataFrame(), set()))
        merged_grid = stats_group + grid_items

        png_src = _candidate_plot_src(payload)
        png_msg = 'PNG view enabled. Switch to Native for interactive hover and diagnostics.'
        if 'phase' in overlays and plot_search_root is not None:
            phase_plot_path = find_phase_plot_image(payload, plot_search_root)
            if phase_plot_path and phase_plot_path.exists():
                png_src = _plot_url_for_path(phase_plot_path)
                png_msg = 'Showing phase-folded PNG view.'
            else:
                mismatch_warnings.append('Phase-fold overlay selected, but no phase PNG was found for this candidate.')
        elif 'phase' in overlays:
            mismatch_warnings.append('Phase-fold overlay selected, but no phase PNG was found for this candidate.')
        panel = _render_run_config_panel(run_params if run_params else None, run_params_path, mismatch_warnings)
        return (
            png_src,
            merged_grid,
            metadata_health,
            vetting_banner,
            progress,
            no_update,
            {'display': 'none'},
            {'display': 'block', 'width': '100%', 'height': '100%'},
            no_update,
            [],
            _render_plot_status_panel('ok', png_msg, mismatch_warnings),
            _render_camera_diag_panel({}, []),
            panel,
            _render_repro_badge(run_params if run_params else None, mismatch_warnings),
            json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
            nonce,
        )

    run_params, run_params_status, run_params_msg = _load_run_params_meta_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
    run_params_path = _run_params_path_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
    mismatch_warnings = _run_config_mismatch_warnings(run_params if run_params else None, overlays)
    if run_params_status != 'loaded':
        mismatch_warnings.append(run_params_msg)
    uirevision_key = f"{candidate_id}|{','.join(sorted(str(c) for c in selected_cameras))}|{','.join(sorted(str(b) for b in selected_bands))}|{theme_mode}|{residual_height:.3f}|{baseline_opacity:.2f}|{yaxis_mode}|{external_source_view}"

    # Discover external LC parquets only when the user explicitly asks for them.
    ext_lcs: dict[str, Path] | None = None
    requested_external_source = str(external_source_view or DEFAULT_EXTERNAL_SOURCE_VIEW).strip().lower()
    if requested_external_source not in {'', 'asassn'}:
        run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
        lk = _candidate_lookup_keys(candidate_id, payload)
        found: dict[str, Path] = {}
        search_roots: list[Path] = []
        if run_dir is not None:
            search_roots.append(run_dir / "results")
        default_results_root = Path(__file__).resolve().parents[2] / "output" / "results"
        if default_results_root not in search_roots:
            search_roots.append(default_results_root)
        for prefix in ("atlas", "ztf", "gaia_epoch", "ps1", "crts"):
            for root in search_roots:
                idx_map = _index_external_lc_paths_from_root(str(root.resolve()), prefix)
                for key in lk:
                    p = idx_map.get(str(key))
                    if p:
                        found[prefix] = Path(p)
                        break
                if prefix in found:
                    break
        if found:
            ext_lcs = found

    try:
        native = build_interactive_lightcurve_figure(
            payload,
            plot_dir=plot_dir_path,
            selected_cameras=selected_cameras,
            selected_bands=selected_bands,
            filter_bad_cameras='filter_bad_cameras' in overlays,
            show_baseline=baseline_opacity > 0,
            show_event_markers='markers' in overlays,
            show_residuals='residuals' in overlays,
            show_phase_fold='phase' in overlays,
            show_raw_mag='raw' in overlays,
            override_period=override_period,
            show_diagnostics='diagnostics' in overlays,
            confidence_colors='confidence' in overlays,
            run_params=run_params or {},
            uirevision_key=uirevision_key,
            theme=theme_mode,
            residual_fraction=residual_height,
            baseline_opacity=baseline_opacity,
            yaxis_mode=yaxis_mode,
            external_lcs=ext_lcs,
            external_source_view=external_source_view,
        )
    except Exception as exc:

        traceback.print_exc()
        panel = _render_run_config_panel(run_params if run_params else None, run_params_path, [str(exc)])
        plot_src = _candidate_plot_src(payload)
        if plot_src:
            return (
                plot_src, grid_items, metadata_health, vetting_banner, progress, no_update,
                {'display': 'none'},
                {'display': 'block', 'width': '100%', 'height': '100%'},
                [], [],
                _render_plot_status_panel('error', f'Native plot error: {exc}', mismatch_warnings),
                _render_camera_diag_panel({}, []),
                panel,
                _render_repro_badge(None, [str(exc)]),
                '', nonce,
            )
        return (
            '', grid_items, metadata_health, vetting_banner, progress, empty_fig,
            {'display': 'block', 'width': '100%', 'height': '100%'},
            {'display': 'none'},
            [], [],
            _render_plot_status_panel('error', f'Native plot error: {exc}', mismatch_warnings),
            _render_camera_diag_panel({}, []),
            panel,
            _render_repro_badge(None, [str(exc)]),
            '', nonce,
        )

    native_status = str(native.get('status', 'ok'))
    native_message = str(native.get('status_message', '') or '')
    native_warnings = list(native.get('warnings', []) or [])
    baseline_warning = _baseline_provenance_warning(
        payload,
        plot_dir=plot_dir_path,
        run_params=run_params if run_params else None,
        stored_lc_path=stored_lc_path,
        source_path=source_path,
    )
    if baseline_warning:
        native_warnings.append(baseline_warning)
    run_config_warnings = list(mismatch_warnings)
    if baseline_warning:
        run_config_warnings.append(baseline_warning)

    plot_src = ''
    if native_status in {"missing-file", "missing-columns", "empty-after-filter", "empty-camera-selection", "empty-band-selection"}:
        plot_src = _candidate_plot_src(payload)

    if native_status in {"missing-file", "missing-columns", "empty-after-filter", "empty-camera-selection", "empty-band-selection"} and plot_src:
        fallback_warnings = native_warnings + mismatch_warnings
        fallback_msg = native_message or "Native plot unavailable; showing PNG fallback."
        return (
            plot_src,
            grid_items,
            metadata_health,
            vetting_banner,
            progress,
            no_update,
            {'display': 'none'},
            {'display': 'block', 'width': '100%', 'height': '100%'},
            [],
            [],
            _render_plot_status_panel('warn', f"{fallback_msg} Showing PNG fallback.", fallback_warnings),
            _render_camera_diag_panel(native.get('camera_diagnostics', {}), []),
            _render_run_config_panel(run_params if run_params else None, run_params_path, run_config_warnings),
            _render_repro_badge(run_params if run_params else None, run_config_warnings),
            json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
            nonce,
        )

    filtered = []
    if 'filtered_cams' in {k for k, _ in native.get('stat_rows', [])}:
        for key, val in native.get('stat_rows', []):
            if key == 'filtered_cams':
                filtered = [x.strip() for x in str(val).split(',') if x.strip()]

    # Merge stats into the metadata grid as the first collapsible group
    stats_group = _render_stat_cards(native['stat_rows'])
    merged_grid = stats_group + grid_items

    return (
        plot_src,
        merged_grid,
        metadata_health,
        vetting_banner,
        progress,
        native['figure'],
        {'display': 'block', 'width': '100%', 'height': '100%'},
        {'display': 'none'},
        native['camera_options'],
        [],  # stats merged into candidate-info-grid
        _render_plot_status_panel(native.get('status', 'ok'), native.get('status_message', ''), (native_warnings + mismatch_warnings)),
        _render_camera_diag_panel(native.get('camera_diagnostics', {}), filtered),
        _render_run_config_panel(run_params if run_params else None, run_params_path, run_config_warnings),
        _render_repro_badge(run_params if run_params else None, run_config_warnings),
        json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
        nonce,
    )


_DISPLAY_OUTPUTS = [Output('plot-image', 'src'),
    Output('candidate-info-grid', 'children'),
    Output('metadata-health-indicator', 'children'),
    Output('vetting-banner', 'children'),
    Output('progress-text', 'children'),
    Output('interactive-plot', 'figure'),
    Output('interactive-plot', 'style'),
    Output('plot-image', 'style'),
    Output('camera-checklist', 'options'),
    Output('plot-stats-cards', 'children'),
    Output('plot-status-panel', 'children'),
    Output('camera-filter-panel', 'children'),
    Output('run-config-panel', 'children'),
    Output('repro-badge', 'children'),
    Output('run-config-json-store', 'data'),
    Output('plot-render-applied', 'data')]


if _background_callback_manager is not None:
    @app.callback(
        _DISPLAY_OUTPUTS,
        Input('plot-render-request', 'data'),
        [State('plot-render-applied', 'data'),
         State('current-candidate-id', 'data'),
         State('queue-size-store', 'data')],
        background=True,
        running=[
            (Output('plot-render-indicator', 'children'), 'Rendering plot...', ''),
        ],
        cancel=[Input('plot-render-request', 'modified_timestamp')],
        prevent_initial_call=False,
    )
    def update_display_callback(render_request, applied_nonce, current_candidate_id, queue_size_data):
        return update_display(render_request, applied_nonce, current_candidate_id, queue_size_data)
else:
    @app.callback(
        _DISPLAY_OUTPUTS,
        Input('plot-render-request', 'data'),
        [State('plot-render-applied', 'data'),
         State('current-candidate-id', 'data'),
         State('queue-size-store', 'data')],
        prevent_initial_call=False,
    )
    def update_display_callback(render_request, applied_nonce, current_candidate_id, queue_size_data):
        return update_display(render_request, applied_nonce, current_candidate_id, queue_size_data)


@app.callback(
    Output('external-followup-panel', 'children'),
    [Input('external-followup-details', 'open'),
     Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data')],
    prevent_initial_call=False,
)
def update_external_followup_panel(is_open, candidate_id, theme_mode):
    """Render external follow-up artifacts for the current candidate."""
    if not is_open:
        return []
    if not candidate_id:
        return html.Div("No candidates loaded.", style={'font-size': '11px', 'color': '#c77'})
    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    return _render_external_followup(payload, str(candidate_id), str(theme_mode or DEFAULT_THEME))


def _render_diagnostic_plots(payload: dict, theme: str, background: dict | None = None) -> list:
    """Build diagnostic plot cards from candidate payload data."""
    theme_tokens = _external_followup_theme(theme)
    card_style = theme_tokens["card_style"]
    cards = []
    for builder in (
        build_cmd_figure,
        build_ir_colorcolor_figure,
        build_kiel_figure,
        build_rpm_figure,
        build_uv_optical_figure,
        build_periodicity_plane_figure,
        build_score_balance_figure,
        build_catalog_support_figure,
        build_recurrence_regularity_figure,
        build_dip_repeatability_figure,
        build_variability_strength_figure,
        build_stetson_scatter_figure,
        build_shape_impulsiveness_figure,
        build_harmonic_quality_figure,
        build_autocorr_memory_figure,
        build_cluster_astrometry_figure,
        build_classifier_plane_figure,
        build_atlas_range_figure,
        build_ztf_range_figure,
        build_neowise_range_figure,
        build_gaia_epoch_figure,
        build_ltv_trend_figure,
        build_neowise_trend_figure,
    ):
        try:
            fig = builder(payload, theme, background=background)
        except Exception:
            fig = None
        if fig is not None:
            cards.append(html.Div(
                dcc.Graph(figure=fig, mathjax=True, config={'displayModeBar': False},
                          style={'height': '280px'}),
                style=card_style,
            ))
    return cards


def _prepare_diagnostic_background(is_open, _import_trigger, _pipeline_progress, existing_state):
    """Load and cache diagnostic plot background data for the current review DB."""
    if not is_open:
        return existing_state if isinstance(existing_state, dict) else {
            'signature': '',
            'ready': False,
            'cached': False,
            'token': 0,
        }

    signature = _diagnostic_background_signature(DB_PATH)
    cached = _get_cached_diagnostic_background(signature)
    if cached is None:
        with closing(db_connect(Path(DB_PATH))) as conn:
            cached = get_diagnostic_background(conn)
        _store_cached_diagnostic_background(signature, cached)
        used_cache = False
    else:
        used_cache = True

    next_token = 1
    if isinstance(existing_state, dict):
        try:
            next_token = int(existing_state.get('token', 0) or 0) + 1
        except Exception:
            next_token = 1

    return {
        'signature': signature,
        'ready': True,
        'cached': used_cache,
        'token': next_token,
    }


if _background_callback_manager is not None:
    @app.callback(
        Output('diagnostic-background-state', 'data'),
        [Input('diagnostic-plots-details', 'open'),
         Input('import-trigger', 'data'),
         Input('pipeline-progress-trigger', 'data')],
        State('diagnostic-background-state', 'data'),
        background=True,
        running=[
            (Output('diagnostic-plots-status', 'children'), 'Loading population background...', ''),
        ],
        prevent_initial_call=False,
    )
    def prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state):
        return _prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state)
else:
    @app.callback(
        Output('diagnostic-background-state', 'data'),
        [Input('diagnostic-plots-details', 'open'),
         Input('import-trigger', 'data'),
         Input('pipeline-progress-trigger', 'data')],
        State('diagnostic-background-state', 'data'),
        prevent_initial_call=False,
    )
    def prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state):
        return _prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state)


@app.callback(
    [Output('diagnostic-plots-panel', 'children'),
     Output('diagnostic-plots-status', 'children', allow_duplicate=True)],
    [Input('diagnostic-plots-details', 'open'),
     Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data'),
     Input('diagnostic-background-state', 'data')],
    prevent_initial_call=True,
)
def update_diagnostic_plots(is_open, candidate_id, theme_mode, background_state):
    """Render diagnostic plots for the current candidate."""
    if not is_open:
        return no_update, ''
    if not candidate_id:
        return [], ''

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    signature = _diagnostic_background_signature(DB_PATH)
    cached_background = None
    if isinstance(background_state, dict) and background_state.get('ready') and background_state.get('signature') == signature:
        cached_background = _get_cached_diagnostic_background(signature)

    panels = _render_diagnostic_plots(payload, str(theme_mode or DEFAULT_THEME), background=cached_background)
    status = '' if cached_background is not None else 'Population background loading...'
    return panels, status


@app.callback(
    [Output('header-asas-sn-id', 'children'),
     Output('header-gaia-id', 'children'),
     Output('bottom-context-info', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('queue-size-store', 'data'),
     Input('queue-filter-hash-store', 'data'),
     Input('import-path', 'value'),
     Input('queue-source-path', 'data')],
    prevent_initial_call=False,
)
def update_header_key_info(candidate_id, queue_size, queue_filter_hash, import_path, queue_source_path):
    """Render short identifiers in the header and long context in the bottom panel."""
    queue_label = str(import_path) if import_path else 'all candidates'
    filter_hash = str(queue_filter_hash or '')
    if filter_hash.startswith('view:') or filter_hash.startswith('fetch:'):
        queue_label = filter_hash.split(':', 1)[1]
    elif queue_source_path:
        queue_label = _queue_scope_label(queue_source_path)

    def _bottom_bar(cluster_path_value: object, local_path_value: object) -> html.Div:
        return html.Div(
            [
                _render_bottom_context("Cluster LC", cluster_path_value if cluster_path_value else "-"),
                _render_bottom_context("Local LC", local_path_value if local_path_value else "-"),
                _render_bottom_context("DB", DB_PATH),
                _render_bottom_context("Queue", queue_label),
            ],
            className='bottom-context-bar',
        )

    if int(queue_size or 0) <= 0 or not candidate_id:
        return 'ASAS-SN ID: -', 'Gaia ID: -', _bottom_bar("-", "-")

    payload, stored_lc_path, source_path = _candidate_context(candidate_id)
    asas_sn_id = payload.get('asas_sn_id')
    gaia_id = payload.get('gaia_id')
    cluster_lc_path, local_lc_path = _display_lc_paths(
        payload,
        stored_lc_path=stored_lc_path,
        source_path=source_path,
    )

    asas_text = f"ASAS-SN ID: {asas_sn_id}" if asas_sn_id else f"ASAS-SN ID: {candidate_id}"
    gaia_fmt = _format_large_integer_like_display(gaia_id)
    gaia_text = f"Gaia ID: {gaia_fmt}" if gaia_fmt else 'Gaia ID: -'
    return asas_text, gaia_text, _bottom_bar(cluster_lc_path, local_lc_path)


@app.callback(
    [Output('plot-export-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    Input('export-plot', 'n_clicks'),
    [State('interactive-plot', 'figure'),
     State('plot-mode', 'value'),
     State('plot-image', 'src'),
     State('current-index', 'data'),
     State('current-candidate-id', 'data')],
    prevent_initial_call=True,
)
def export_active_plot(n_clicks, figure, plot_mode, plot_src, idx, candidate_id):
    """Export the currently shown plot.

    Native mode exports PDF with enhanced metadata and high resolution.
    PNG mode exports the currently displayed PNG file.
    """
    if not n_clicks:
        return no_update, no_update

    ordinal = int(idx) + 1 if idx is not None else 0

    if plot_mode == 'native':
        if not figure:
            return no_update, 'No native plot is available to export.'

        # Get ASAS-SN ID for filename
        asas_sn_id = 'unknown'
        try:
            if candidate_id:
                payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
                asas_sn_id = str(payload.get('asas_sn_id') or payload.get('candidate_id') or 'unknown')
        except Exception:
            pass

        fname = f"malca_plot_{ordinal}_{asas_sn_id}.pdf"
        try:
            export_fig = go.Figure(figure)
            export_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='#111111', family='Monaco, Courier New, monospace', size=11),
                title_font=dict(color='#111111'),
                margin=dict(t=54, l=60, r=20, b=50),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='rgba(40,40,40,0.25)',
                    borderwidth=1,
                    font=dict(color='#111111', size=9),
                ),
            )
            export_fig.update_xaxes(
                color='#111111',
                title_font_color='#111111',
                tickfont_color='#111111',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.12)',
                zeroline=False,
            )
            export_fig.update_yaxes(
                color='#111111',
                title_font_color='#111111',
                tickfont_color='#111111',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.12)',
                zeroline=False,
            )
            image_bytes = pio.to_image(export_fig, format='pdf')
        except Exception as exc:
            return no_update, f'Export failed (PDF). Install/enable kaleido. {exc}'
        return dcc.send_bytes(image_bytes, fname), f'Exported {fname}'

    if plot_mode == 'png':
        if not plot_src:
            return no_update, 'No PNG plot is available to export.'
        src = str(plot_src)
        plot_file: Path | None = None
        if src.startswith('/plots/'):
            plot_file = _plot_file_from_src(src)
        else:
            candidate = Path(src)
            if candidate.exists():
                plot_file = candidate

        if plot_file is None or not plot_file.exists():
            return no_update, 'Current PNG file could not be found on disk.'

        fname = plot_file.name
        return dcc.send_file(str(plot_file)), f'Exported {fname}'

    return no_update, 'Unknown plot mode; nothing exported.'


@app.callback(
    [Output('run-config-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    Input('download-run-config-btn', 'n_clicks'),
    State('run-config-json-store', 'data'),
    prevent_initial_call=True,
)
def download_run_config(n_clicks, run_config_json):
    """Download current run_params JSON shown in GUI."""
    if not n_clicks:
        return no_update, no_update
    if not run_config_json:
        return no_update, 'No run_params.json is loaded for this run.'
    return dcc.send_string(run_config_json, 'run_params.json'), 'Downloaded run_params.json'


app.clientside_callback(
    """
    async function(nClicks, runConfigJson) {
        if (!nClicks) {
            return window.dash_clientside.no_update;
        }
        if (!runConfigJson) {
            return 'No run_params.json is loaded for this run.';
        }
        try {
            await navigator.clipboard.writeText(runConfigJson);
            return 'Copied run_params.json to clipboard';
        } catch (e) {
            return 'Clipboard copy failed; use Download Config JSON.';
        }
    }
    """,
    Output('notification', 'children', allow_duplicate=True),
    Input('copy-run-config-btn', 'n_clicks'),
    State('run-config-json-store', 'data'),
    prevent_initial_call=True,
)


# Load review data for current candidate
@app.callback(
    [Output('event-class-store', 'data'),
     Output('needs-followup-store', 'data'),
     Output('review-pass-store', 'data'),
     Output('notes', 'value'),
     Output('current-score', 'data')],
    Input('current-candidate-id', 'data'),
    State('queue-size-store', 'data'),
    prevent_initial_call=False
)
def load_review_form(candidate_id, queue_size):
    """Load existing review for current candidate into stores."""
    if not candidate_id or int(queue_size or 0) == 0:
        return 'unclassified', False, 1, '', None

    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, str(candidate_id))

    # Coerce legacy/unknown classes into the current tag set.
    allowed_classes = {'unclassified'} | set(CLASS_KEY_MAP.values())
    event_class = (review.get('event_class') or 'unclassified')
    if event_class not in allowed_classes:
        event_class = 'other'

    return (
        event_class,
        review.get('status', 'unreviewed') == 'needs_followup',
        review.get('review_pass', 1),
        review.get('notes', ''),
        review.get('interest_score'),
    )


app.clientside_callback(
    """
    function(n1, n2, n3, n4, queueSize, candidateId, eventClass, needsFollowup, notes, saveRequest) {
        var no = window.dash_clientside.no_update;
        if (!candidateId || parseInt(queueSize == null ? 0 : queueSize, 10) <= 0) {
            var triggered = window.dash_clientside.callback_context.triggered || [];
            if (!triggered.length) {
                return [no, no, no];
            }
            return [no, 'Queue is empty', no];
        }

        var triggered = window.dash_clientside.callback_context.triggered || [];
        if (!triggered.length) {
            return [no, no, no];
        }
        var triggerId = String(triggered[0].prop_id || '').split('.')[0];
        if (!triggerId.startsWith('score-')) {
            return [no, no, no];
        }

        var score = parseInt(triggerId.split('-')[1], 10);
        if (!Number.isFinite(score)) {
            return [no, no, no];
        }

        var nextNonce = 1;
        if (saveRequest && typeof saveRequest === 'object' && typeof saveRequest.nonce === 'number') {
            nextNonce = saveRequest.nonce + 1;
        }

        return [
            score,
            '✓ Confidence: ' + String(score),
            {
                nonce: nextNonce,
                candidate_id: String(candidateId),
                score: score,
                event_class: eventClass || 'unclassified',
                needs_followup: !!needsFollowup,
                notes: notes || '',
                increment_pass: false,
                event_type: 'button',
            }
        ];
    }
    """,
    [Output('current-score', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('review-save-request', 'data', allow_duplicate=True)],
    [Input(f'score-{i}', 'n_clicks') for i in range(1, 5)],
    [State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value'),
     State('review-save-request', 'data')],
    prevent_initial_call=True,
)


# Save button
@app.callback(
    [Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('save-btn', 'n_clicks'),
    [State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def save_review_callback(n_clicks, queue_size, candidate_id, score,
                         event_class, needs_followup, notes):
    """Save review."""
    if not n_clicks or int(queue_size or 0) <= 0 or not candidate_id:
        return no_update, no_update

    new_pass, _ = _do_save(
        str(candidate_id), score, event_class, needs_followup, notes, 'save_button',
    )

    return "✓ Saved", new_pass


# Back button (previous candidate)
@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    Input('back-btn', 'n_clicks'),
    State('current-index', 'data'),
    prevent_initial_call=True
)
def back_callback(n_clicks, idx):
    """Go to previous candidate."""
    if not n_clicks:
        return no_update, no_update
    new_idx = max(0, (idx or 0) - 1)
    if new_idx == idx:
        return no_update, "Already at first candidate"
    return new_idx, "← Previous"


# Done button (save + next)
@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('done-btn', 'n_clicks'),
    [State('current-index', 'data'),
     State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def done_callback(n_clicks, idx, queue_size, candidate_id, score,
                  event_class, needs_followup, notes):
    """Save and go to next."""
    if not n_clicks or int(queue_size or 0) <= 0 or not candidate_id:
        return no_update, no_update, no_update

    if score is None:
        return no_update, "⚠ Confidence required", no_update

    if not event_class or event_class == 'unclassified':
        return no_update, "⚠ Class required", no_update

    new_pass, _ = _do_save(
        str(candidate_id), score, event_class, needs_followup, notes, 'done_button',
        increment_pass=True,
    )

    queue_size = int(queue_size or 0)
    new_idx = min(idx + 1, queue_size - 1)

    return new_idx, "✓ Saved + Next →", new_pass


# --- Display callbacks for stores → visible indicators ---

app.clientside_callback(
    """
    function(currentScore) {
        var score = parseInt(currentScore, 10);
        if (!Number.isFinite(score) || [1, 2, 3, 4].indexOf(score) === -1) {
            score = null;
        }
        var out = [];
        for (var i = 1; i <= 4; i += 1) {
            out.push(i === score ? 'score-btn active' : 'score-btn');
        }
        return out;
    }
    """,
    [Output(f'score-{i}', 'className') for i in range(1, 5)],
    Input('current-score', 'data'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(activeClass) {
        var active = activeClass || 'unclassified';
        return ['dipper', 'microlensing', 'flare', 'ltv', 'unknown_interesting', 'instrumental', 'other']
            .map(function(tag) { return tag === active ? 'badge-btn active' : 'badge-btn'; });
    }
    """,
    [Output(f'class-badge-{tag}', 'className') for tag in CLASS_BADGE_TAGS],
    Input('event-class-store', 'data'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function() {
        var no = window.dash_clientside.no_update;
        var triggered = window.dash_clientside.callback_context.triggered || [];
        if (!triggered.length) {
            return [no, no];
        }
        var triggerId = String(triggered[0].prop_id || '').split('.')[0];
        if (!triggerId.startsWith('class-badge-')) {
            return [no, no];
        }
        var tag = triggerId.replace('class-badge-', '');
        var active = arguments[arguments.length - 1] || 'unclassified';
        if (active === tag) {
            return ['unclassified', 'Class: unclassified'];
        }
        return [tag, 'Class: ' + tag];
    }
    """,
    [Output('event-class-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input(f'class-badge-{tag}', 'n_clicks') for tag in CLASS_BADGE_TAGS],
    State('event-class-store', 'data'),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(prefix) {
        if (!prefix) {
            return '';
        }
        return '[' + String(prefix).toUpperCase() + '] ...';
    }
    """,
    Output('prefix-indicator', 'children'),
    Input('pending-prefix', 'data'),
    prevent_initial_call=False,
)


# Expand/collapse all candidate metadata panels
@app.callback(
    [Output({'type': 'meta-details', 'group': ALL}, 'open'),
     Output('toggle-meta-all', 'children')],
    Input('toggle-meta-all', 'n_clicks'),
    State({'type': 'meta-details', 'group': ALL}, 'open'),
    prevent_initial_call=True,
)
def toggle_all_metadata_panels(n_clicks, open_states):
    _ = n_clicks
    if not open_states:
        return [], no_update

    any_closed = any(not bool(v) for v in open_states)
    new_open = True if any_closed else False
    label = 'Collapse all' if new_open else 'Expand all'
    return [new_open for _ in open_states], label


app.clientside_callback(
    """
    function(needsFollowup) {
        return needsFollowup ? '[,] Followup: ON' : '[,] Followup: off';
    }
    """,
    Output('followup-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    prevent_initial_call=False,
)


def _format_hms(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


@app.callback(
    Output('review-session-start', 'data'),
    Input('queue-filter-hash-store', 'data'),
    State('review-session-start', 'data'),
    prevent_initial_call=False,
)
def sync_review_session_start(queue_hash, session_start):
    """Reset session-timer origin when the active queue changes."""
    now = time.time()
    if not isinstance(session_start, dict):
        return {'ts': now, 'filter_hash': queue_hash}

    if session_start.get('filter_hash') != queue_hash:
        return {'ts': now, 'filter_hash': queue_hash}

    ts = session_start.get('ts')
    if ts is None:
        return {'ts': now, 'filter_hash': queue_hash}

    return session_start


@app.callback(
    Output('review-progress-state', 'data'),
    Input('review-db-scope', 'data'),
    Input('queue-data', 'modified_timestamp'),
    Input('review-pass-store', 'data'),
    Input('import-trigger', 'data'),
    prevent_initial_call=False,
)
def load_review_progress_state(_db_scope, _queue_data_ts, _review_pass, _import_trigger):
    """Load reviewed/total counts for the progress indicator without interval polling."""
    reviewed, total = _progress_counts()
    return {'reviewed': int(reviewed), 'total': int(total)}


app.clientside_callback(
    """
    function(reviewPass) {
        return 'Pass: ' + String(reviewPass || 1);
    }
    """,
    Output('pass-indicator', 'children'),
    Input('review-pass-store', 'data'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(needsFollowup, score, candidateId, queueSize) {
        if (!candidateId || parseInt(queueSize == null ? 0 : queueSize, 10) <= 0) {
            return 'Status: —';
        }
        if (needsFollowup) {
            return 'Status: needs_followup';
        }
        if (score !== null && score !== undefined && score !== '') {
            return 'Status: reviewed';
        }
        return 'Status: unreviewed';
    }
    """,
    Output('status-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    Input('current-score', 'data'),
    [State('current-candidate-id', 'data'),
     State('queue-size-store', 'data')],
    prevent_initial_call=False,
)

# Auto-populate import candidates from run directory inferred via plot directory
@app.callback(
    [Output('import-path', 'value'),
     Output('sidebar-status', 'children', allow_duplicate=True)],
    Input('keyboard-init', 'n_intervals'),
    State('import-path', 'value'),
    prevent_initial_call='initial_duplicate',
)
def auto_populate_detected_files(_n_intervals, current_import_path):
    """Auto-detect linked run files on app startup and fill import path."""
    if current_import_path:
        return no_update, no_update

    restored_path = None
    with closing(db_connect(Path(DB_PATH))) as conn:
        restored_path = str(load_app_state(conn, "last_input_file", "") or "").strip()

    run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
    if run_dir:
        try:
            detected = detect_run_directory_files(run_dir)
            detected_sources = _candidate_files_for_run_dir(run_dir)
        except Exception as e:
            detected = {'candidates': None, 'warnings': [f"Auto-detect failed: {str(e)}"]}
            detected_sources = []

        if detected_sources:
            resolved_candidates = "\n".join(str(p) for p in detected_sources)
            vetting_mode = _vetting_mode_for_sources(resolved_candidates)
            with closing(db_connect(Path(DB_PATH))) as conn:
                save_app_state(conn, "last_input_file", resolved_candidates)
            return resolved_candidates, (
                f"✓ Auto-detected {len(detected_sources)} candidates file(s): {_summarize_source_paths(detected_sources)} | "
                f"Vetting mode: {vetting_mode}"
            )

        warnings = detected.get('warnings') or []
        restored_sources = _resolve_import_sources(restored_path)
        if restored_sources:
            restored_mode = _vetting_mode_for_sources(restored_path)
            restored_text = "\n".join(str(p) for p in restored_sources)
            return restored_text, (
                f"⚠ {warnings[0]} | restored last candidates: {_summarize_source_paths(restored_sources)} | "
                f"Vetting mode: {restored_mode}"
                if warnings else (
                    f"✓ Restored last candidates: {_summarize_source_paths(restored_sources)} | "
                    f"Vetting mode: {restored_mode}"
                )
            )

        if warnings:
            return no_update, f"⚠ {warnings[0]}"

    restored_sources = _resolve_import_sources(restored_path)
    if restored_sources:
        restored_mode = _vetting_mode_for_sources(restored_path)
        restored_text = "\n".join(str(p) for p in restored_sources)
        return restored_text, (
            f"✓ Restored last candidates: {_summarize_source_paths(restored_sources)} | "
            f"Vetting mode: {restored_mode}"
        )

    return no_update, "⚠ Could not auto-detect candidates; set import path manually."


@app.callback(
    Output('queue-source-path', 'data'),
    [Input('keyboard-init', 'n_intervals'),
     Input('import-path', 'value')],
    prevent_initial_call=False,
)
def update_queue_source_scope(_n_intervals, import_path):
    """Scope queue to the active run bundle token when possible."""
    if _resolve_run_dir_from_db_path(DB_PATH) is not None:
        return ''
    # Prefer run-dir scope when --plot-dir is set so the queue loads (diagnostic/external panels need current-candidate-id).
    run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
    if run_dir is not None:
        return {'source_path_like_any': [run_dir.name], 'label': run_dir.name}
    # Standalone mode (no --plot-dir): don't scope queue to a restored import path, so merged DBs show full queue.
    if PLOT_DIR is None:
        return ''
    scope = _queue_scope_from_import_text(import_path)
    if scope:
        return scope
    return ''


@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('import-path', 'value'),
    prevent_initial_call=True,
)
def show_vetting_mode_status(import_path):
    """Show whether import will use vetted input, cache, or re-vetting."""
    if not import_path:
        return no_update
    mode = _vetting_mode_for_sources(import_path)
    return f"Vetting mode: {mode}"


def _import_sources_with_options(
    sources: list[Path],
    *,
    characterize_on,
    crossmatch,
    gaia_cache,
    chunk_size,
    dust_on,
    starhorse,
    vet_on,
    lc_mode,
    current_trigger,
) -> tuple[str, int]:
    """Import one or more candidate/light-curve sources into the review DB."""
    if not sources:
        raise ValueError("No import sources resolved")

    raw_lc_mode = 'yes' in (lc_mode or [])
    with closing(db_connect(Path(DB_PATH))) as conn:
        enable_characterize = 'yes' in (characterize_on or [])
        enable_vetting = 'yes' in (vet_on or [])

        if raw_lc_mode:
            total_rows = 0
            total_new = 0
            for src in sources:
                n_rows, n_new = import_lightcurve_files(
                    conn, src,
                    characterize=enable_characterize,
                    vet=enable_vetting,
                )
                total_rows += int(n_rows)
                total_new += int(n_new)
            save_app_state(conn, "last_input_file", "\n".join(str(p) for p in sources))
            return (
                f"✓ LC import: {total_rows} rows ({total_new} new) from {len(sources)} file(s)",
                (current_trigger or 0) + 1,
            )

        vetting_mode = _vetting_mode_for_sources("\n".join(str(p) for p in sources)) if enable_vetting else "vetting disabled"
        total_rows = 0
        total_new = 0
        for src in sources:
            df = load_candidates_file(src)
            n_rows, n_new = import_candidates(
                conn, df, str(src),
                characterize_before_import=enable_characterize,
                characterize_crossmatch=Path(crossmatch).expanduser() if enable_characterize and crossmatch else None,
                characterize_cache=Path(gaia_cache).expanduser() if enable_characterize and gaia_cache else None,
                characterize_chunk_size=int(chunk_size) if enable_characterize and chunk_size else GAIA_CHUNK_SIZE,
                characterize_dust='yes' in (dust_on or []) if enable_characterize else False,
                characterize_starhorse=starhorse.strip() if enable_characterize and starhorse and starhorse.strip() else None,
                vet_before_import=enable_vetting,
            )
            total_rows += int(n_rows)
            total_new += int(n_new)
        save_app_state(conn, "last_input_file", "\n".join(str(p) for p in sources))
        return (
            f"✓ Imported {total_rows} rows ({total_new} new) from {len(sources)} file(s) | Vetting mode: {vetting_mode}",
            (current_trigger or 0) + 1,
        )


# Import candidates
@app.callback(
    [Output('sidebar-status', 'children', allow_duplicate=True),
     Output('import-trigger', 'data')],
    Input('import-btn', 'n_clicks'),
    [State('import-path', 'value'),
     State('characterize-on-import', 'value'),
     State('characterize-crossmatch', 'value'),
     State('characterize-gaia-cache', 'value'),
     State('characterize-chunk-size', 'value'),
     State('characterize-dust', 'value'),
     State('characterize-starhorse', 'value'),
     State('vet-on-import', 'value'),
     State('import-lc-mode', 'value'),
     State('import-trigger', 'data')],
    prevent_initial_call=True
)
def import_candidates_callback(n_clicks, import_path, characterize_on,
                               crossmatch, gaia_cache, chunk_size, dust_on, starhorse, vet_on,
                               lc_mode, current_trigger):
    """Import candidates from file."""
    if not n_clicks or not import_path:
        return no_update, no_update

    raw_lc_mode = 'yes' in (lc_mode or [])
    sources = _resolve_import_sources(import_path, allow_run_dirs=not raw_lc_mode)
    if not sources:
        sources = [Path(str(import_path)).expanduser()]

    try:
        return _import_sources_with_options(
            sources,
            characterize_on=characterize_on,
            crossmatch=crossmatch,
            gaia_cache=gaia_cache,
            chunk_size=chunk_size,
            dust_on=dust_on,
            starhorse=starhorse,
            vet_on=vet_on,
            lc_mode=lc_mode,
            current_trigger=current_trigger,
        )
    except Exception as e:
        return f"✗ Import failed: {str(e)}", no_update


def _try_float(value) -> float | None:
    """Parse finite float values; return None on invalid input."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_first_nonempty_id(row: dict, keys: tuple[str, ...]) -> str | None:
    """Return first non-empty identifier found in row for candidate key names."""
    for key in keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        try:
            if pd.isna(raw):
                continue
        except Exception:
            pass
        text = _format_large_integer_like_display(raw).strip()
        if text and text.lower() not in {'nan', 'none'}:
            return text
    return None


def _nearest_catalog_match_by_coords(catalog_df: pd.DataFrame, ra_deg: float, dec_deg: float) -> tuple[dict | None, float | None]:
    """Return nearest cone-search row and separation (arcsec) for target coords."""
    if catalog_df is None or catalog_df.empty:
        return None, None

    ra_col = next((c for c in ('ra_deg', 'ra', 'RAJ2000', 'RA', 'raj2000') if c in catalog_df.columns), None)
    dec_col = next((c for c in ('dec_deg', 'dec', 'DEJ2000', 'DEC', 'Dec', 'dej2000') if c in catalog_df.columns), None)
    if ra_col is None or dec_col is None:
        return None, None

    ra_vals = pd.to_numeric(catalog_df[ra_col], errors='coerce').to_numpy(dtype=float)
    dec_vals = pd.to_numeric(catalog_df[dec_col], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(ra_vals) & np.isfinite(dec_vals)
    if not np.any(valid):
        return None, None

    ra0 = float(ra_deg) % 360.0
    dec0 = float(dec_deg)
    ra = np.mod(ra_vals[valid], 360.0)
    dec = dec_vals[valid]

    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    dra = np.deg2rad(((ra - ra0 + 180.0) % 360.0) - 180.0)
    dec_rad = np.deg2rad(dec)

    sin_ddec = np.sin((dec_rad - dec0_rad) / 2.0)
    sin_dra = np.sin(dra / 2.0)
    a = sin_ddec * sin_ddec + np.cos(dec0_rad) * np.cos(dec_rad) * sin_dra * sin_dra
    a = np.clip(a, 0.0, 1.0)
    sep_rad = 2.0 * np.arcsin(np.sqrt(a))
    sep_arcsec = np.rad2deg(sep_rad) * 3600.0

    nearest_local = int(np.argmin(sep_arcsec))
    valid_indices = np.flatnonzero(valid)
    nearest_idx = int(valid_indices[nearest_local])
    return catalog_df.iloc[nearest_idx].to_dict(), float(sep_arcsec[nearest_local])


def _fetch_candidate_impl(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger):
    """Core fetch logic; set_progress is optional for streaming status to GUI."""
    from malca.review.fetch import (
        fetch_and_analyze_by_gaia_id,
        fetch_and_analyze_by_id,
        fetch_cone_search,
    )

    def progress(msg):
        if set_progress and msg:
            try:
                set_progress(msg[:400] if len(msg) > 400 else msg)
            except Exception:
                pass

    if not n_clicks or not fetch_query:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    query = fetch_query.strip()
    if not query:
        return "✗ Query cannot be empty", no_update, no_update, no_update, no_update, no_update, no_update

    try:
        effective_backend = str(fetch_backend or DEFAULT_FETCH_BACKEND).strip().lower()
        if effective_backend not in {'skypatrol2', 'skypatrol1'}:
            effective_backend = DEFAULT_FETCH_BACKEND

        effective_fetch_type = str(fetch_type or 'asassn')
        effective_query = query
        cone_status_text = ''

        if effective_fetch_type == 'coords':
            progress('Searching SkyPatrol cone...')

            parts = [p for p in query.replace(',', ' ').split() if p]
            if len(parts) < 2:
                return "✗ Enter coordinates as 'RA Dec [radius_arcsec]'", no_update, no_update, '', None, no_update, no_update

            ra = _try_float(parts[0])
            dec = _try_float(parts[1])
            if ra is None or dec is None:
                return "✗ Invalid RA/Dec. Use decimal degrees: 'RA Dec [radius_arcsec]'", no_update, no_update, '', None, no_update, no_update
            if dec < -90.0 or dec > 90.0:
                return '✗ Dec must be between -90 and +90 degrees', no_update, no_update, '', None, no_update, no_update

            radius = 5.0
            if len(parts) > 2:
                radius = _try_float(parts[2])
                if radius is None or radius <= 0.0:
                    return '✗ Radius must be a positive number (arcsec)', no_update, no_update, '', None, no_update, no_update

            ra = ra % 360.0
            catalog_df = fetch_cone_search(ra, dec, radius_arcsec=radius, backend=effective_backend)
            if catalog_df.empty:
                return f'✗ No sources found within {radius:.2f}"', no_update, no_update, '', None, no_update, no_update

            nearest_row, sep_arcsec = _nearest_catalog_match_by_coords(catalog_df, ra, dec)
            if nearest_row is None or sep_arcsec is None:
                return '✗ Cone search returned rows without usable coordinates', no_update, no_update, '', None, no_update, no_update

            nearest_asas = _extract_first_nonempty_id(nearest_row, ('asas_sn_id', 'asassn_id'))
            nearest_gaia = _extract_first_nonempty_id(nearest_row, ('gaia_id', 'source_id'))

            if nearest_asas:
                effective_fetch_type = 'asassn'
                effective_query = nearest_asas
            elif nearest_gaia:
                effective_fetch_type = 'gaia'
                effective_query = nearest_gaia
            else:
                return '✗ Nearest cone match has no ASAS-SN ID or Gaia ID to fetch', no_update, no_update, '', None, no_update, no_update

            n_found = int(len(catalog_df))
            match_word = 'match' if n_found == 1 else 'matches'
            cone_status_text = (
                f' | cone nearest {effective_query} at {sep_arcsec:.2f}" '
                f'({n_found} {match_word} in {radius:.2f}")'
            )
            progress(f'SkyPatrol cone resolved: {effective_query} ({sep_arcsec:.2f}")')

        progress("Fetching light curve...")
        if effective_fetch_type == 'gaia':

            df, lc_path = fetch_and_analyze_by_gaia_id(effective_query, run_stats=True, backend=effective_backend)
        else:

            df, lc_path = fetch_and_analyze_by_id(effective_query, run_stats=True, backend=effective_backend)

        if df is None or df.empty:
            return f"✗ No data for {effective_query}", no_update, no_update, '', None, no_update, no_update

        progress("Importing basic light curve...")
        
        # We NEVER run characterization/vetting in the fetch callback anymore,
        # so the GUI can render the light curve IMMEDIATELY.
        with closing(db_connect(Path(DB_PATH))) as conn:
            n_rows, n_new = import_candidates(
                conn, df, source_path=f"fetch://{effective_backend}/{effective_fetch_type}/{effective_query}",
                characterize_before_import=False,
                vet_before_import=False,
            )

        cid = str(df.iloc[0]['candidate_id']) if 'candidate_id' in df.columns else effective_query
        _index_external_lc_paths.cache_clear()
        _index_external_lc_paths_from_root.cache_clear()
        
        auto_run = no_update
        if fetch_mode in ('full', 'full_ext'):
            auto_run = {'candidate_id': cid, 'mode': fetch_mode, 'ts': time.time()}

        status = f"✓ Added {effective_query} ({n_new} new) [{effective_backend}]{cone_status_text}"
        return (
            status,
            no_update,
            auto_run,
            '',
            None,
            {'candidate_ids': [cid], 'queue_size': 1, 'filter_hash': f'fetch:{cid}'},
            0,
        )

    except Exception as e:

        traceback.print_exc()
        return f"✗ Fetch failed: {str(e)}", no_update, no_update, '', None, no_update, no_update


# Fetch and Analyze Candidate
if _background_callback_manager is not None:
    @app.callback(
        [Output('fetch-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('pending-auto-run', 'data', allow_duplicate=True),
         Output('cone-results-container', 'children'),
         Output('cone-results-data', 'data'),
         Output('queue-data', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        Input('fetch-btn', 'n_clicks'),
        [State('fetch-type', 'value'),
         State('fetch-query', 'value'),
         State('fetch-mode', 'value'),
         State('fetch-backend', 'value'),
         State('import-trigger', 'data')],
        background=True,
        running=[
            (Output('fetch-btn', 'disabled'), True, False),
        ],
        progress=[Output('fetch-status', 'children')],
        prevent_initial_call=True,
    )
    def fetch_candidate_callback(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger):
        return _fetch_candidate_impl(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger)
else:
    @app.callback(
        [Output('fetch-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('pending-auto-run', 'data', allow_duplicate=True),
         Output('cone-results-container', 'children'),
         Output('cone-results-data', 'data'),
         Output('queue-data', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        Input('fetch-btn', 'n_clicks'),
        [State('fetch-type', 'value'),
         State('fetch-query', 'value'),
         State('fetch-mode', 'value'),
         State('fetch-backend', 'value'),
         State('import-trigger', 'data')],
        prevent_initial_call=True,
    )
    def fetch_candidate_callback(n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger):
        return _fetch_candidate_impl(None, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger)


def _pipeline_status_chip_elements(candidate_id) -> list:
    """Build pipeline status chips for the active candidate."""
    chips = []
    stage_labels = {
        'external_lcs': 'External LCs',
        'periodicity': 'Periodicity',
    }
    if candidate_id is None:
        return chips
    try:
        payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
        status = detect_pipeline_status(payload)

        periodicity_sig_cols = (
            'periodicity_score',
            'lsp_period',
            'lsp_power',
            'lsp_is_significant',
        )
        periodicity_present = sum(
            1
            for col in periodicity_sig_cols
            if col in payload and payload[col] is not None
            and not (isinstance(payload[col], float) and np.isnan(payload[col]))
        )
        if periodicity_present == 0:
            status['periodicity'] = 'missing'
        elif periodicity_present == len(periodicity_sig_cols):
            status['periodicity'] = 'complete'
        else:
            status['periodicity'] = 'partial'

        color_map = {'complete': '#2d6a2d', 'partial': '#6a5c2d', 'missing': '#444'}
        for stage, state in status.items():
            chips.append(html.Span(
                f"{'●' if state == 'complete' else '○'} {stage_labels.get(stage, stage.capitalize())}",
                style={
                    'padding': '1px 6px',
                    'borderRadius': '8px',
                    'backgroundColor': color_map.get(state, '#444'),
                    'color': '#e0e0e0' if state != 'missing' else '#666',
                    'fontSize': '10px',
                },
            ))
    except Exception:
        return []
    return chips


# Pipeline status chips (updated when candidate changes and stages complete)
@app.callback(
    Output('pipeline-status-chips', 'children'),
    [Input('queue-data', 'modified_timestamp'),
     Input('current-candidate-id', 'data'),
     Input('pipeline-progress-trigger', 'data')],
    prevent_initial_call=True
)
def update_pipeline_status_chips(_queue_data_ts, candidate_id, _pipeline_progress):
    """Show pipeline stage completion status for the current candidate."""
    return _pipeline_status_chip_elements(candidate_id)


# Cascade auto-run once queue updates for fetched candidate
@app.callback(
    Output('auto-run-pipeline-trigger', 'data', allow_duplicate=True),
    [Input('queue-data', 'modified_timestamp'),
     Input('current-candidate-id', 'data')],
    State('pending-auto-run', 'data'),
    prevent_initial_call=True,
)
def maybe_cascade_auto_run(_queue_data_ts, candidate_id, pending_auto_run):
    """Emit pending auto-run token when fetched candidate enters queue."""
    if candidate_id is None or not pending_auto_run:
        return no_update

    triggered_ids = {
        item['prop_id'].split('.')[0]
        for item in (callback_context.triggered or [])
        if item.get('prop_id')
    }
    if 'queue-data' not in triggered_ids:
        return no_update
    if str(pending_auto_run.get('candidate_id')) != str(candidate_id):
        return no_update

    try:
        payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
        status = detect_pipeline_status(payload)
        if any(state in {'missing', 'partial'} for state in status.values()):
            return pending_auto_run
    except Exception:
        return no_update
    return no_update


@app.callback(
    Output('pipeline-module-log-panel', 'children'),
    Input('pipeline-module-log', 'data'),
    prevent_initial_call=False,
)
def render_pipeline_module_log(log_data):
    """Render temporary in-GUI module run log lines."""
    lines = []
    if isinstance(log_data, dict):
        raw = log_data.get('lines')
        if isinstance(raw, list):
            lines = [str(x) for x in raw if x is not None]
    if not lines:
        return "No pipeline run log yet."
    return "\n".join(lines[-300:])


def _run_pipeline_impl(set_progress, run_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick):

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if triggered_id == 'auto-run-pipeline-trigger' and not auto_trigger:
        raise dash.exceptions.PreventUpdate

    queue_size = int((queue_data or {}).get('queue_size') or 0) if isinstance(queue_data, dict) else 0
    print(f"[run_pipeline_callback] Triggered by: {triggered_id}, auto_trigger: {auto_trigger}, queue_size: {queue_size}, idx: {idx}")

    if (
        not run_clicks
        and not rerun_clicks
        and not rerun_stats_clicks
        and not rerun_events_clicks
        and not rerun_characterize_clicks
        and not rerun_vetting_clicks
        and not rerun_external_lcs_clicks
        and not auto_trigger
    ):
        raise dash.exceptions.PreventUpdate
        
    candidate_id = None
    fetch_mode = None
    if auto_trigger:
        candidate_id = auto_trigger.get('candidate_id')
        fetch_mode = auto_trigger.get('mode')
    else:
        candidate_id = _queue_candidate_id(queue_data, idx)
        if candidate_id is None:
            return "No candidate selected", no_update, no_update
        
    if not candidate_id:
        return "No candidate_id", no_update, no_update

    try:
        try:
            progress_tick = int(current_progress_tick or 0)
        except Exception:
            progress_tick = 0
        log_lines: list[str] = []

        def append_log_line(text: str) -> None:
            nonlocal log_lines
            line = str(text or '').strip()
            if not line:
                return
            if log_lines and log_lines[-1] == line:
                return
            log_lines.append(line)
            if len(log_lines) > 300:
                log_lines = log_lines[-300:]

        def emit_progress(msg: str, *, bump_render: bool = False, reset_log: bool = False):
            nonlocal progress_tick
            text = str(msg or "")
            text = text[:300] if len(text) > 300 else text
            if reset_log:
                log_lines.clear()
            append_log_line(text)
            if bump_render:
                progress_tick += 1
            if set_progress:
                try:
                    set_progress((text, progress_tick, {'lines': list(log_lines)}))
                except Exception:
                    pass
            else:
                print(f"[pipeline] {text}")

        def p(msg):
            emit_progress(msg, bump_render=False)

        def on_stage_complete(stage_name: str):
            emit_progress(f"✓ {stage_name} complete", bump_render=True)

        emit_progress(f"Running pipeline for {candidate_id}...", reset_log=True)
            
        with closing(db_connect(Path(DB_PATH))) as conn:
            
            # Determine if this was an explicit deep fetch that should bypass caching
            force_stages = []
            force_only = False
            if triggered_id == 'rerun-pipeline-btn':
                force_stages = ['stats', 'events', 'characterize', 'vetting', 'external_lcs']
            elif triggered_id == 'rerun-stage-stats-btn':
                force_stages = ['stats']
                force_only = True
            elif triggered_id == 'rerun-stage-events-btn':
                force_stages = ['events']
                force_only = True
            elif triggered_id == 'rerun-stage-characterize-btn':
                force_stages = ['characterize']
                force_only = True
            elif triggered_id == 'rerun-stage-vetting-btn':
                force_stages = ['vetting']
                force_only = True
            elif triggered_id == 'rerun-stage-external-lcs-btn':
                force_stages = ['external_lcs']
                force_only = True
            elif fetch_mode == 'full':
                force_stages = ['stats', 'events', 'characterize', 'vetting']
            elif fetch_mode in ('full_ext', 'full_ext_crts'):
                force_stages = ['stats', 'events', 'characterize', 'vetting', 'external_lcs']
                
            stages = run_missing_stages(
                conn,
                candidate_id,
                progress_callback=p,
                stage_complete_callback=on_stage_complete,
                force_stages=force_stages,
                only_force=force_only,
            )
            
            # If triggered by a full_ext fetch, ensure we run external LCs
            if fetch_mode == 'full_ext' and 'external_lcs' not in stages:
                p("Running external LCs...")
                from malca.vetting import fetch_external_lcs



                payload = get_candidate_payload(conn, candidate_id)
                df = pd.DataFrame([payload])
                run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
                ext_output = (
                    run_dir / "results"
                    if run_dir
                    else (Path(__file__).resolve().parents[2] / "output" / "results")
                )
                ext_output.mkdir(parents=True, exist_ok=True)
                df_ext = fetch_external_lcs(df, output_dir=ext_output, progress_callback=p)
                if isinstance(df_ext, pd.DataFrame) and not df_ext.empty:
                    row = df_ext.iloc[0].to_dict()
                    update_candidate_payload(conn, candidate_id, row)
                    stages.append('external_lcs')
                    on_stage_complete('external_lcs')

        refresh_idx = int(idx or 0) if idx is not None else 0
        if stages:
            _index_external_lc_paths.cache_clear()
            _index_external_lc_paths_from_root.cache_clear()
            return f"✓ Ran: {', '.join(stages)}", no_update, refresh_idx
        else:
            if triggered_id in {
                'rerun-pipeline-btn',
                'rerun-stage-stats-btn',
                'rerun-stage-events-btn',
                'rerun-stage-characterize-btn',
                'rerun-stage-vetting-btn',
                'rerun-stage-external-lcs-btn',
            }:
                return "No stages could be rerun (missing requirements)", no_update, no_update
            return "All stages already complete (or missing requirements)", no_update, no_update
    except Exception as e:

        traceback.print_exc()
        return f"✗ Pipeline failed: {str(e)}", no_update, no_update

if _background_callback_manager is not None:
    @app.callback(
        [Output('pipeline-run-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        [Input('run-pipeline-btn', 'n_clicks'),
         Input('rerun-pipeline-btn', 'n_clicks'),
         Input('rerun-stage-stats-btn', 'n_clicks'),
         Input('rerun-stage-events-btn', 'n_clicks'),
         Input('rerun-stage-characterize-btn', 'n_clicks'),
         Input('rerun-stage-vetting-btn', 'n_clicks'),
         Input('rerun-stage-external-lcs-btn', 'n_clicks'),
         Input('auto-run-pipeline-trigger', 'data')],
        [State('queue-data', 'data'),
         State('current-index', 'data'),
         State('import-trigger', 'data'),
         State('pipeline-progress-trigger', 'data')],
        background=True,
        running=[
            (Output('run-pipeline-btn', 'disabled'), True, False),
            (Output('rerun-pipeline-btn', 'disabled'), True, False),
            (Output('rerun-stage-stats-btn', 'disabled'), True, False),
            (Output('rerun-stage-events-btn', 'disabled'), True, False),
            (Output('rerun-stage-characterize-btn', 'disabled'), True, False),
            (Output('rerun-stage-vetting-btn', 'disabled'), True, False),
            (Output('rerun-stage-external-lcs-btn', 'disabled'), True, False),
        ],
        progress=[
            Output('pipeline-run-status', 'children'),
            Output('pipeline-progress-trigger', 'data'),
            Output('pipeline-module-log', 'data'),
        ],
        prevent_initial_call='initial_duplicate'
    )
    def run_pipeline_callback(set_progress, n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick):
        return _run_pipeline_impl(set_progress, n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick)
else:
    @app.callback(
        [Output('pipeline-run-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        [Input('run-pipeline-btn', 'n_clicks'),
         Input('rerun-pipeline-btn', 'n_clicks'),
         Input('rerun-stage-stats-btn', 'n_clicks'),
         Input('rerun-stage-events-btn', 'n_clicks'),
         Input('rerun-stage-characterize-btn', 'n_clicks'),
         Input('rerun-stage-vetting-btn', 'n_clicks'),
         Input('rerun-stage-external-lcs-btn', 'n_clicks'),
         Input('auto-run-pipeline-trigger', 'data')],
        [State('queue-data', 'data'),
         State('current-index', 'data'),
         State('import-trigger', 'data'),
         State('pipeline-progress-trigger', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def run_pipeline_callback(n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick):
        return _run_pipeline_impl(None, n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick)


# Export reviews
@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('export-btn', 'n_clicks'),
    [State('export-path', 'value'),
     State('export-only-reviewed', 'value')],
    prevent_initial_call=True
)
def export_reviews_callback(n_clicks, export_path, only_reviewed):
    """Export reviews to file."""
    if not n_clicks or not export_path:
        return no_update

    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            out_path = Path(export_path).expanduser()
            only_reviewed_flag = 'yes' in (only_reviewed or [])
            export_reviews(conn, out_path, only_reviewed=only_reviewed_flag)
            reviewed_text = " (reviewed only)" if only_reviewed_flag else ""
            return f"✓ Exported to {out_path.name}{reviewed_text}"
    except Exception as e:
        return f"✗ Export failed: {str(e)}"


@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('merge-review-db-btn', 'n_clicks'),
    State('merge-target-db-path', 'value'),
    prevent_initial_call=True,
)
def merge_review_db_callback(n_clicks, target_db_path):
    """Merge the current review DB into another review DB."""
    if not n_clicks or not target_db_path:
        return no_update

    try:
        source_db = Path(DB_PATH).expanduser().resolve()
        target_db = Path(str(target_db_path)).expanduser().resolve()
        result = merge_review_databases(source_db, target_db, only_reviewed=True)
        with closing(db_connect(Path(DB_PATH))) as conn:
            save_app_state(conn, 'last_merge_target_db', str(target_db))
        return (
            f"✓ Merged into {target_db.name} | "
            f"reviews inserted={result['reviews_inserted']}, updated={result['reviews_updated']}, "
            f"skipped={result['reviews_skipped']}"
        )
    except Exception as e:
        return f"✗ Merge failed: {str(e)}"


# Help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-link", "n_clicks"),
     Input("close-help", "n_clicks"),
     Input("keyboard-input", "value")],
    State("help-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_help_modal(n1, n2, key_value, is_open):
    """Toggle help modal."""
    ctx = callback_context
    if not ctx.triggered:
        return is_open

    trigger = ctx.triggered[0]['prop_id']

    if 'keyboard-input' in trigger:
        if _keyboard_key(key_value) == '?':
            return not is_open
        return is_open

    if n1 or n2:
        return not is_open
    return is_open


# Static file server
@app.server.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix not in _PLOT_STATIC_EXTENSIONS:
        abort(404)
    return send_from_directory(str(_plot_asset_root()), filename)


# Reset queue to beginning
@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    Input('reset-queue-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_queue_position(n_clicks):
    """Reset queue position to the beginning (index 0)."""
    if not n_clicks:
        return no_update, no_update
    return 0, "Queue position reset to beginning."


def main():
    """Main entry point."""
    global DB_PATH, PLOT_DIR, INITIAL_CANDIDATE_QUERY

    parser = argparse.ArgumentParser(description="MALCA Dash Review App")
    parser.add_argument('--db', default=None, help="SQLite database path (default: standalone.db without --plot-dir, review.db with --plot-dir)")
    parser.add_argument('--plot-dir', help="Plot directory path (auto-detects ./plots if not specified)")
    parser.add_argument('--host', default='127.0.0.1', help="Host")
    parser.add_argument('--port', default=8050, type=int, help="Port")
    parser.add_argument('--candidate', default=None, help="Candidate ID / ASAS-SN ID / Gaia ID / LC stem to open on startup")
    parser.add_argument('--no-browser', action='store_true', help="Do not auto-open a browser tab/window on startup")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--verbose-http', action='store_true',
                        help="Show Flask/Werkzeug per-request access logs")
    parser.add_argument('--merge-vetting', metavar='PATH',
                        help="Merge vetting results from a parquet file into the review DB and exit")
    parser.add_argument('--merge-candidates', metavar='PATH',
                        help="Merge candidate columns from a CSV/parquet file into the review DB and exit")
    args = parser.parse_args()
    if args.merge_vetting and args.merge_candidates:
        parser.error("--merge-vetting and --merge-candidates are mutually exclusive")
    INITIAL_CANDIDATE_QUERY = str(args.candidate).strip() if args.candidate not in (None, '') else None

    # Auto-detect plot directory if not specified
    if args.plot_dir:
        PLOT_DIR = str(_resolve_plot_cli_path(args.plot_dir))
        if not Path(PLOT_DIR).exists() or not Path(PLOT_DIR).is_dir():
            print(f"Error: plot directory does not exist: {PLOT_DIR}")
            print("Use an existing run bundle plots directory, for example:")
            print("  malca review --plot-dir output/runs/output_bundle_13_13.5/plots")
            sys.exit(1)
    else:
        # Try current directory first
        if Path('./plots').is_dir():
            PLOT_DIR = str(Path('./plots').resolve())
            print(f"Auto-detected plot directory: {Path(PLOT_DIR).resolve()}")
        else:
            # Standalone mode — no plot dir required
            PLOT_DIR = None
            print("Running in standalone mode (no --plot-dir)")

    inferred_plot_db = _review_db_for_plot_dir(PLOT_DIR)

    # Choose DB: explicit --db overrides; otherwise standalone gets its own DB
    if args.db is not None:
        DB_PATH = str(_resolve_db_cli_path(args.db))
    elif PLOT_DIR is None:
        # Standalone mode: use a separate DB so pipeline candidates don't bleed in
        DB_PATH = str(_resolve_db_cli_path(str(DEFAULT_STANDALONE_DB_PATH)))
    elif inferred_plot_db is not None:
        DB_PATH = str(inferred_plot_db)
    else:
        DB_PATH = str(_resolve_db_cli_path(str(DEFAULT_DB_PATH)))

    # Publish runtime paths so spawned background workers inherit the same config.
    os.environ[_REVIEW_DB_ENV] = str(DB_PATH)
    if PLOT_DIR:
        os.environ[_REVIEW_PLOT_ENV] = str(PLOT_DIR)
    else:
        os.environ.pop(_REVIEW_PLOT_ENV, None)

    mismatch_warning = _db_plot_mismatch_warning(DB_PATH, PLOT_DIR)
    if mismatch_warning:
        print(f"Warning: {mismatch_warning}")

    if args.merge_vetting:
        vetting_path = Path(args.merge_vetting).expanduser().resolve()
        if not vetting_path.exists():
            print(f"Error: vetting file not found: {vetting_path}")
            sys.exit(1)
        vetting_df = pd.read_parquet(vetting_path)
        print(f"Merging {len(vetting_df)} vetting results from {vetting_path}")
        print(f"  into review DB: {DB_PATH}")
        with closing(db_connect(Path(DB_PATH))) as conn:
            updated = merge_vetting_results(conn, vetting_df)
        print(f"Updated {updated} candidates with vetting data.")
        sys.exit(0)

    if args.merge_candidates:
        candidate_path = Path(args.merge_candidates).expanduser().resolve()
        if not candidate_path.exists():
            print(f"Error: candidate file not found: {candidate_path}")
            sys.exit(1)
        candidate_df = load_candidates_file(candidate_path)
        print(f"Merging {len(candidate_df)} candidate rows from {candidate_path}")
        print(f"  into review DB: {DB_PATH}")
        with closing(db_connect(Path(DB_PATH))) as conn:
            updated = merge_candidate_results(conn, candidate_df)
        print(f"Updated {updated} candidates with candidate data.")
        sys.exit(0)

    print(f"Starting MALCA Review App...")
    print(f"  Database:  {DB_PATH}")
    print(f"  Plot dir:  {PLOT_DIR}")
    print(f"  Server:    http://{args.host}:{args.port}")
    print(f"\nKeyboard shortcuts:")
    print("  [D]ipper [M]icrolensing [F]lare [Y]so [U]nknown [I]nstrumental [O]ther | [1-4] Confidence | [.] Save | [Enter] Done | [Backspace] Back | [Esc] Sidebar | [R] Refresh | [?] Help")
    print("")

    # Auto-open browser
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        Timer(0.1, lambda: webbrowser.open(url)).start()

    if not args.verbose_http:
        # Keep explicit pipeline/status prints, but hide the noisy per-request
        # development-server access lines so long-running actions are readable.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.server.logger.setLevel(logging.ERROR)

    try:
        app.run(debug=args.debug, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup_background_resources()


if __name__ == '__main__':
    main()
