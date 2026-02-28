"""Dash-based keyboard-driven review app for MALCA candidates."""

import sys
import warnings
import argparse
import logging

# Suppress known multiprocessing/diskcache semaphore leak warning at worker shutdown
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be.*leaked semaphore",
    module="multiprocessing.resource_tracker",
)
import json
import time
import sqlite3
import re
from decimal import Decimal, InvalidOperation
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import webbrowser
from threading import Timer
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc
from flask import send_from_directory
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

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
    merge_vetting_results,
    get_distinct_values,
)
from malca.review.metadata import (
    extract_review_metadata_grouped,
    is_group_default_open,
    build_external_lookup_links,
)
from malca.review.keyboard import (
    handle_key_action, HELP_TEXT,
    CLASS_KEY_MAP,
)

CLASS_BADGE_TAGS = list(CLASS_KEY_MAP.values())
from malca.review.session import create_queue_data_dict
from malca.review.interactive_plot import (
    build_interactive_lightcurve_figure,
    resolve_lightcurve_path,
    _load_cleaned_df,
    _compute_baseline_bands,
    normalize_external_lc_dataframe,
)
from malca.config.config_paths import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE
from malca.config.config_characterize import GAIA_CHUNK_SIZE

# Background callback manager for long-running fetch/import (DiskCache for local dev)
try:
    import diskcache
    try:
        from dash import DiskcacheManager
    except ImportError:
        from dash.long_callback import DiskcacheManager
    _bc_cache = diskcache.Cache(Path(__file__).resolve().parents[2] / "output" / "review" / ".dash_cache")
    _background_callback_manager = DiskcacheManager(_bc_cache)
except Exception:
    _background_callback_manager = None

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
        .meta-field-value {
            color: #e2edf6;
            text-align: right;
            font-weight: 600;
            word-break: break-word;
            white-space: normal;
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

# Global variables
DB_PATH = str(DEFAULT_DB_PATH)
PLOT_DIR = None
DEFAULT_THEME = "black"
DEFAULT_RESIDUAL_FRACTION = 0.33
EXTERNAL_SOURCE_VIEW_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "ASAS-SN Only", "value": "asassn"},
    {"label": "ATLAS", "value": "atlas"},
    {"label": "ZTF", "value": "ztf"},
    {"label": "Gaia Epoch", "value": "gaia_epoch"},
    {"label": "PS1", "value": "ps1"},
    {"label": "CRTS", "value": "crts"},
]

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
    """Render stats as a collapsible Details/Summary group."""
    if not stat_rows:
        return []
    field_divs = [
        html.Div([
            html.Span(label, className='meta-field-label'),
            html.Span(str(value), className='meta-field-value'),
        ], className='meta-field-row')
        for label, value in stat_rows
    ]
    return [html.Details(
        [html.Summary(f"Stats ({len(stat_rows)})"), html.Div(field_divs, className='meta-grid')],
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

    run_post_filter = bool(run_params.get('run_post_filter', True))
    run_postprocess = bool(run_params.get('run_postprocess', True))
    min_bf = float(run_params.get('min_bayes_factor', 0.0) or 0.0)
    if (not run_post_filter) and (not run_postprocess):
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


def _render_run_config_panel(run_params: dict | None, run_params_path: Path | None, warnings: list[str]) -> list:
    """Render compact run config cells with an inline warning row."""
    status = "Loaded" if run_params else "Missing"

    def _card(label: str, value: str, *, title: str | None = None, wide: bool = False, warning: bool = False) -> html.Div:
        classes = ['run-config-item']
        if wide:
            classes.append('wide')
        if warning:
            classes.append('warning')
        return html.Div([
            html.Div(label, className='k'),
            html.Div(value, className='v', title=title),
        ], className=' '.join(classes))

    cards = [
        _card('Status', status),
        _card(
            'Path',
            str(run_params_path) if run_params_path else 'not found',
            title=str(run_params_path) if run_params_path else 'not found',
        ),
    ]
    for label, value in _run_config_rows(run_params or {}):
        cards.append(_card(label, value, title=value))
    if warnings:
        warning_text = ' | '.join(warnings)
        cards.append(_card('Warnings', warning_text, title=warning_text, wide=True, warning=True))
    return cards


def _render_repro_badge(run_params: dict | None, warnings: list[str]) -> html.Span:
    """Render reproducibility status badge."""
    if (run_params is None) or warnings:
        text = 'Repro: fallback/defaults'
        cls = 'repro-badge warn'
    else:
        text = 'Repro: exact run params'
        cls = 'repro-badge'
    return html.Span(text, className=cls)


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
    header_text = "KNOWN OBJECT" if known else "POTENTIALLY NEW"

    cards = []

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
    simbad_id = payload.get('simbad_otype') or payload.get('simbad_main_id')
    if simbad_id:
        refs = payload.get('simbad_nbref')
        ref_str = f" ({refs} refs)" if refs else ""
        cards.append(_cell("SIMBAD", f"{simbad_id}{ref_str}", hit=True, title=str(payload.get('simbad_main_id', ''))))

    # VSX cell
    vsx_cls = payload.get('vsx_class')
    if vsx_cls and str(vsx_cls).strip() and str(vsx_cls).strip().lower() not in ('nan', '<na>'):
        vsx_sep = payload.get('vsx_sep_arcsec')
        sep_str = f" ({vsx_sep:.1f}\")" if vsx_sep and not pd.isna(vsx_sep) else ""
        vsx_p = payload.get('vsx_period')
        p_str = f", P={vsx_p:.4f}d" if vsx_p and not pd.isna(vsx_p) else ""
        cards.append(_cell("VSX", f"{vsx_cls}{p_str}{sep_str}", hit=True))

    # Gaia variability cell
    gaia_cls = payload.get('gaia_var_class')
    if gaia_cls:
        score = payload.get('gaia_var_score')
        score_str = f" ({score:.2f})" if score and not pd.isna(score) else ""
        cards.append(_cell("Gaia DR3", f"{gaia_cls}{score_str}", hit=True))

    # Gaia EB period cell
    eb_period = payload.get('gaia_eb_period')
    if eb_period and not pd.isna(eb_period):
        cards.append(_cell("Gaia EB", f"P={eb_period:.4f} d", hit=True))

    # ASAS-SN cell
    asassn_type = payload.get('asassn_var_type')
    if asassn_type:
        period = payload.get('asassn_var_period')
        p_str = f" P={period:.4f}d" if period and not pd.isna(period) else ""
        cards.append(_cell("ASAS-SN", f"{asassn_type}{p_str}", hit=True))

    # ZTF cell
    ztf_type = payload.get('ztf_var_type')
    if ztf_type:
        ztf_p = payload.get('ztf_var_period')
        zp_str = f" P={ztf_p:.4f}d" if ztf_p and not pd.isna(ztf_p) else ""
        cards.append(_cell("ZTF", f"{ztf_type}{zp_str}", hit=True))

    # TNS cell
    tns_name = payload.get('tns_name')
    if tns_name:
        tns_type = payload.get('tns_type', '')
        cards.append(_cell("TNS", f"{tns_name} ({tns_type})" if tns_type else tns_name, hit=True))

    # ALeRCE cell
    alerce_cls = payload.get('alerce_lc_class')
    if alerce_cls:
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
    if sfr_name and str(sfr_name).strip() and str(sfr_name).strip().lower() not in ('nan', '<na>'):
        sfr_sep = payload.get('sfr_sep_arcmin')
        sep_str = f" ({sfr_sep:.1f}')" if sfr_sep and not pd.isna(sfr_sep) else ""
        cards.append(_cell("SFR", f"{sfr_name}{sep_str}", hit=True))

    # Cluster cell
    cluster_name = payload.get('cluster_name')
    if cluster_name and str(cluster_name).strip() and str(cluster_name).strip().lower() not in ('nan', '<na>'):
        cluster_dist = payload.get('cluster_dist_pc')
        d_str = f" ({cluster_dist:.0f} pc)" if cluster_dist and not pd.isna(cluster_dist) else ""
        cards.append(_cell("Cluster", f"{cluster_name}{d_str}", hit=True))

    # BANYAN cell
    banyan_assoc = payload.get('banyan_best_assoc')
    banyan_fp = payload.get('banyan_field_prob')
    if banyan_assoc and str(banyan_assoc).strip() and str(banyan_assoc).strip().lower() not in ('nan', '<na>', 'field'):
        fp_str = f" (P_field={banyan_fp:.0%})" if banyan_fp and not pd.isna(banyan_fp) else ""
        cards.append(_cell("BANYAN", f"{banyan_assoc}{fp_str}", hit=True))

    # YSO class cell
    yso_cls = payload.get('yso_class')
    if yso_cls and str(yso_cls).strip() and str(yso_cls).strip().lower() not in ('nan', '<na>'):
        cards.append(_cell("YSO", str(yso_cls), hit=True))

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


def _project_root() -> Path:
    """Repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


def _count_candidates_in_db(path: Path) -> int:
    """Return number of candidates in DB, or -1 when unavailable."""
    try:
        with closing(sqlite3.connect(path)) as conn:
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
    """Convert source-native time values into the review axis: JD - 2458000."""
    t = pd.to_numeric(values, errors="coerce").to_numpy()
    finite_t = t[np.isfinite(t)]
    if jd_system == "mjd":
        if finite_t.size and float(np.nanmedian(finite_t)) > 1_000_000.0:
            jd = t
        else:
            jd = t + 2400000.5
    elif jd_system == "bjd_gaia":
        jd = t + 2455197.5
    elif jd_system == "btjd":
        jd = t + 2457000.0
    elif jd_system == "bkjd":
        jd = t + 2454833.0
    else:
        jd = t
    return jd - 2458000.0


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
                        atlas_children.append(dcc.Graph(figure=atlas_fig, config={'displayModeBar': False}, style={'height': '250px'}))
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
                        ztf_children.append(dcc.Graph(figure=ztf_fig, config={'displayModeBar': False}, style={'height': '250px'}))
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
                        gaia_epoch_children.append(dcc.Graph(figure=gaia_fig, config={'displayModeBar': False}, style={'height': '250px'}))
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
                        ps1_children.append(dcc.Graph(figure=ps1_fig, config={'displayModeBar': False}, style={'height': '250px'}))
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
                        crts_children.append(dcc.Graph(figure=crts_fig, config={'displayModeBar': False}, style={'height': '250px'}))
                    except Exception:
                        pass
                break

    crts_card = html.Div([
        html.Div('CRTS', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *crts_children,
    ], style=card_style)

    return [spectra_card, atlas_card, neowise_card, ztf_card, gaia_epoch_card, ps1_card, crts_card]


# ---- sidebar filter helpers ------------------------------------------------
_ATF_OPTS = [{'label': v, 'value': v} for v in ('Any', 'True', 'False')]
_inp_style = {'width': '100%', 'margin-bottom': '4px', 'font-size': '11px'}


def _bool_mode_filter(label: str, component_id: str):
    """Return a (Label, Dropdown) pair for an Any/True/False bool filter."""
    return [
        html.Label(f'{label}:'),
        dcc.Dropdown(
            id=component_id,
            options=_ATF_OPTS,
            value='Any',
            clearable=False,
            style={'margin-bottom': '4px', 'font-size': '11px'},
            persistence=True,
            persistence_type='local',
        ),
    ]


def _col_id(col: str) -> str:
    """snake_case → dash-case for Dash component IDs."""
    return col.replace('_', '-')


def _num_range_filter(col: str):
    """Compact min+max inputs for a numeric column on one line."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col}:'),
        html.Div([
            dcc.Input(id=f'min-{cid}', type='number', placeholder='min',
                      style={'width': '48%', 'font-size': '11px', 'margin-right': '4%'},
                      persistence=True, persistence_type='local'),
            dcc.Input(id=f'max-{cid}', type='number', placeholder='max',
                      style={'width': '48%', 'font-size': '11px'},
                      persistence=True, persistence_type='local'),
        ], style={'display': 'flex', 'margin-bottom': '4px'}),
    ])


def _text_filter(col: str):
    """Text input for exact-match string filter."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col}:'),
        dcc.Input(id=f'filter-{cid}', type='text', placeholder='Any',
                  style=_inp_style,
                  persistence=True, persistence_type='local'),
    ])


def _select_filter(col: str):
    """Multi-select dropdown for exclude filtering (options loaded from DB)."""
    cid = _col_id(col)
    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            values = get_distinct_values(conn, col)
        options = [{'label': v, 'value': v} for v in values]
    except Exception:
        options = []
    return html.Div([
        html.Label(f'{col} (exclude):'),
        dcc.Dropdown(
            id=f'exclude-{cid}', options=options, multi=True,
            placeholder='None excluded',
            style={'margin-bottom': '4px', 'font-size': '11px'},
            maxHeight=400,
            optionHeight=28,
            persistence=True,
            persistence_type='local',
        ),
    ])


def _make_filter_group(name: str, items: list, *, default_open: bool = False):
    """Build a collapsible html.Details for a group of filters."""
    children = []
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
_SIDEBAR_GROUPS = [
    ('Vetting', [
        ('bool', 'vetting_likely_known'),
        ('num', 'pm_cluster_offset_sigma'),
        ('select', 'vsx_class'),
        ('select', 'asassn_var_type'),
        ('select', 'gaia_var_class'),
        ('select', 'simbad_otype'),
        ('select', 'ztf_var_type'),
        ('select', 'tns_type'),
        ('select', 'alerce_lc_class'),
        ('select', 'yso_class'),
    ]),
    ('LTV', [
        ('num', 'ltv_slope'),
        ('num', 'ltv_max_diff'),
        ('num', 'ltv_ls_period'),
        ('num', 'ltv_ls_fap'),
        ('bool', 'ltv_passed_filters'),
        ('bool', 'ltv_dust_candidate'),
        ('bool', 'ltv_vsx_match'),
        ('bool', 'ltv_milliquas_match'),
        ('bool', 'ltv_gaia_alert_match'),
    ]),
    ('External Coverage', [
        ('num', 'neowise_n_epochs'),
        ('num', 'atlas_n_det_cyan'),
        ('num', 'atlas_n_det_orange'),
        ('num', 'gaia_epoch_n_obs'),
        ('num', 'xray_flux'),
    ]),
    ('General Flags', [
        ('bool', 'periodic_flag'),
        ('bool', 'catalog_match'),
        ('bool', 'high_ruwe_flag'),
    ]),
    ('Periodicity', [
        ('bool', 'lsp_is_alias'),
        ('bool', 'lsp_is_significant'),
        ('num', 'periodicity_score'),
        ('num', 'phase_quality_score'),
        ('num', 'lsp_bootstrap_sig'),
        ('num', 'lsp_power'),
        ('num', 'lsp_period'),
    ]),
    ('Dip Detection', [
        ('bool', 'dip_significant'),
        ('text', 'dip_best_morph'),
        ('num', 'dip_best_log_bf'),
        ('num', 'dip_best_delta_bic'),
        ('num', 'dip_best_width_param'),
        ('num', 'dip_symmetry_score'),
        ('num', 'dip_best_amp'),
        ('num', 'dip_best_t0'),
        ('num', 'dip_best_alpha'),
        ('num', 'dip_best_tau'),
        ('num', 'dip_bayes_factor'),
        ('num', 'dip_best_p'),
        ('num', 'dip_best_mag_event'),
        ('num', 'dip_trigger_max'),
        ('num', 'dip_max_event_prob'),
        ('num', 'dip_trigger_threshold'),
    ]),
    ('Dip Runs', [
        ('num', 'dip_count'),
        ('num', 'dip_run_count'),
        ('num', 'dip_max_run_points'),
        ('num', 'dip_max_run_duration'),
        ('num', 'dip_max_run_sum'),
        ('num', 'dip_max_run_max'),
        ('num', 'dip_max_run_cameras'),
        ('num', 'dip_max_log_bf_local'),
    ]),
    ('Jump Detection', [
        ('bool', 'jump_significant'),
        ('text', 'jump_best_morph'),
        ('num', 'jump_best_log_bf'),
        ('num', 'jump_best_delta_bic'),
        ('num', 'jump_best_width_param'),
        ('num', 'jump_best_amp'),
        ('num', 'jump_best_t0'),
        ('num', 'jump_best_alpha'),
        ('num', 'jump_best_tau'),
        ('num', 'jump_bayes_factor'),
        ('num', 'jump_best_p'),
        ('num', 'jump_best_mag_event'),
        ('num', 'jump_trigger_max'),
        ('num', 'jump_max_event_prob'),
        ('num', 'jump_trigger_threshold'),
    ]),
    ('Jump Runs', [
        ('num', 'jump_count'),
        ('num', 'jump_run_count'),
        ('num', 'jump_max_run_points'),
        ('num', 'jump_max_run_duration'),
        ('num', 'jump_max_run_sum'),
        ('num', 'jump_max_run_max'),
        ('num', 'jump_max_run_cameras'),
        ('num', 'jump_max_log_bf_local'),
    ]),
    ('Dip Recurrence', [
        ('bool', 'dip_is_single_event'),
        ('num', 'dip_inter_event_spacing_median'),
        ('num', 'dip_inter_event_spacing_std'),
        ('num', 'dip_amplitude_consistency'),
        ('num', 'dip_duration_consistency'),
    ]),
    ('Jump Recurrence', [
        ('bool', 'jump_is_single_event'),
        ('num', 'jump_inter_event_spacing_median'),
        ('num', 'jump_inter_event_spacing_std'),
        ('num', 'jump_amplitude_consistency'),
        ('num', 'jump_duration_consistency'),
    ]),
    ('Event Scoring', [
        ('num', 'dipper_score'),
        ('num', 'dipper_n_dips'),
        ('num', 'dipper_n_valid_dips'),
        ('num', 'jumper_score'),
        ('num', 'jumper_n_jumps'),
        ('num', 'jumper_n_valid_jumps'),
    ]),
    ('Stellar Parameters', [
        ('num', 'ruwe'),
        ('num', 'teff_gspphot'),
        ('num', 'logg_gspphot'),
        ('num', 'mh_gspphot'),
        ('num', 'distance_gspphot'),
        ('num', 'parallax'),
        ('num', 'pmra'),
        ('num', 'pmdec'),
    ]),
    ('Photometry', [
        ('num', 'tmass_j'),
        ('num', 'tmass_h'),
        ('num', 'tmass_k'),
        ('num', 'unwise_w1'),
        ('num', 'unwise_w2'),
        ('num', 'H_K'),
        ('num', 'W1_W2'),
        ('num', 'iphas_ha_mag'),
        ('num', 'unwise_w1_zscore'),
        ('num', 'unwise_w2_zscore'),
    ]),
    ('Galactic Coordinates', [
        ('num', 'gal_l'),
        ('num', 'gal_b'),
    ]),
    ('Extinction & Environment', [
        ('text', 'population'),
        ('text', 'banyan_best_assoc'),
        ('num', 'A_v_3d'),
        ('num', 'ebv_3d'),
        ('num', 'age50'),
        ('num', 'mass50'),
        ('num', 'banyan_field_prob'),
    ]),
    ('Associations', [
        ('text', 'sfr_name'),
        ('num', 'sfr_sep_arcmin'),
        ('text', 'cluster_name'),
        ('num', 'cluster_membership_prob'),
    ]),
    ('Period Consensus', [
        ('num', 'period_n_sources'),
        ('num', 'period_consensus_days'),
        ('num', 'period_consensus_support'),
        ('bool', 'period_consensus_agree'),
        ('bool', 'period_conflict_flag'),
    ]),
    ('LC Cadence & Coverage', [
        ('num', 'stats_time_span_days'),
        ('num', 'stats_n_unique_nights'),
        ('num', 'stats_duty_cycle_fraction'),
        ('num', 'stats_cadence_mean_dt_days'),
        ('num', 'stats_cadence_median_dt_days'),
        ('num', 'stats_cadence_p05_dt_days'),
        ('num', 'stats_cadence_p95_dt_days'),
        ('num', 'stats_file_points_total'),
        ('num', 'stats_file_points_kept_after_filter'),
    ]),
    ('LC Photometric Scatter', [
        ('num', 'stats_photometry_std_mag'),
        ('num', 'stats_photometry_robust_sigma_mag'),
        ('num', 'stats_photometry_IQR_mag'),
        ('num', 'stats_photometry_mean_mag'),
        ('num', 'stats_photometry_median_mag'),
        ('num', 'stats_photometry_weighted_mean_mag'),
        ('num', 'stats_clipped_std_mag_3sigma_about_median'),
        ('num', 'stats_n_outliers_removed_robust_3sigma'),
    ]),
    ('LC Variability', [
        ('num', 'stats_variability_reduced_chi2_vs_constant'),
        ('num', 'stats_variability_von_neumann_ratio'),
        ('num', 'stats_variability_lag1_autocorr'),
        ('num', 'stats_variability_stetson_I'),
        ('num', 'stats_variability_stetson_J'),
        ('num', 'stats_variability_stetson_K'),
        ('num', 'stats_amplitude'),
        ('num', 'stats_beyond_1_std'),
        ('num', 'stats_con'),
        ('num', 'stats_delta_mag_fid'),
        ('num', 'stats_excess_var'),
        ('num', 'stats_first_mag'),
        ('num', 'stats_gskew'),
        ('num', 'stats_max_slope'),
        ('num', 'stats_meanvariance'),
        ('num', 'stats_median_abs_dev'),
        ('num', 'stats_median_brp'),
        ('num', 'stats_percent_amplitude'),
        ('num', 'stats_q31'),
        ('num', 'stats_skew'),
        ('num', 'stats_small_kurtosis'),
        ('num', 'stats_pvar'),
        ('num', 'stats_anderson_darling'),
        ('num', 'stats_pair_slope_trend'),
        ('num', 'stats_rcs'),
        ('num', 'stats_autocor_length'),
    ]),
    ('LC Structure Function', [
        ('num', 'stats_sf_ml_amplitude'),
        ('num', 'stats_sf_ml_gamma'),
    ]),
    ('LC Harmonics (folded)', [
        ('num', 'stats_harmonics_mag_1'),
        ('num', 'stats_harmonics_mag_2'),
        ('num', 'stats_harmonics_mag_3'),
        ('num', 'stats_harmonics_mag_4'),
        ('num', 'stats_harmonics_mag_5'),
        ('num', 'stats_harmonics_mag_6'),
        ('num', 'stats_harmonics_mag_7'),
        ('num', 'stats_harmonics_phase_2'),
        ('num', 'stats_harmonics_phase_3'),
        ('num', 'stats_harmonics_phase_4'),
        ('num', 'stats_harmonics_phase_5'),
        ('num', 'stats_harmonics_phase_6'),
        ('num', 'stats_harmonics_phase_7'),
        ('num', 'stats_harmonics_mse'),
        ('num', 'stats_psi_cs'),
        ('num', 'stats_psi_eta'),
    ]),
    ('LC Stochastic Models', [
        ('num', 'stats_gp_drw_sigma'),
        ('num', 'stats_gp_drw_tau'),
        ('num', 'stats_iar_phi'),
        ('num', 'stats_mhps_high'),
        ('num', 'stats_mhps_low'),
        ('num', 'stats_mhps_non_zero'),
        ('bool', 'stats_mhps_pn_flag'),
        ('num', 'stats_mhps_ratio'),
    ]),
    ('LC Error / SNR / Trend', [
        ('num', 'stats_error_and_snr_stats_snr_median'),
        ('num', 'stats_error_and_snr_stats_error_median'),
        ('num', 'stats_trend_slope_mag_per_year'),
        ('num', 'stats_trend_r2'),
    ]),
    ('Light Curve Basics', [
        ('text', 'baseline_source'),
        ('text', 'trigger_mode'),
        ('num', 'n_points'),
        ('num', 'n_cameras'),
        ('num', 'baseline_mag'),
        ('num', 'cadence_median_days'),
    ]),
    ('Classification', [
        ('text', 'trigger_type'),
        ('text', 'yso_class'),
        ('text', 'final_class'),
        ('num', 'P_eb'),
        ('num', 'P_cv'),
        ('num', 'P_starspot'),
        ('num', 'P_disk'),
        ('num', 'a_circ_au'),
        ('num', 'transit_prob'),
        ('num', 'hill_radius_rsun'),
    ]),
    ('Fail Flags', [
        ('bool', 'failed_posterior_strength'),
        ('bool', 'failed_run_robustness'),
        ('bool', 'failed_morphology'),
        ('bool', 'failed_score'),
        ('bool', 'failed_periodicity'),
        ('bool', 'failed_gaia_ruwe'),
        ('bool', 'failed_periodic_catalog'),
        ('bool', 'failed_signal_amplitude'),
        ('bool', 'bad_cameras_filtered'),
    ]),
]


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
        dcc.Store(id='current-index', data=0),
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
        dcc.Store(id='import-trigger', data=0),  # triggers queue refresh after import
        dcc.Store(id='auto-run-pipeline-trigger', data=None),
        dcc.Store(id='pending-auto-run', data=None),
        dcc.Store(id='cone-results-data', data=None),  # cone search catalog rows
        dcc.Store(id='auto-period-cache', data={}, storage_type='session'),
        dcc.Store(id='plot-render-request', data={'nonce': 1, 'ts': 0.0, 'state': {'idx': 0, 'plot_mode': 'native', 'overlay_values': list(PLOT_PRESETS['Diagnostics']['overlays']), 'selected_cameras': [], 'preset': 'Diagnostics', 'theme': DEFAULT_THEME, 'residual_height': DEFAULT_RESIDUAL_FRACTION, 'baseline_opacity': 0.5, 'external_source_view': 'all'}}),
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
        dcc.Download(id='plot-export-download'),
        dcc.Download(id='run-config-download'),
        dcc.Interval(id='keyboard-init', interval=200, n_intervals=0, max_intervals=1),
        dcc.Interval(id='review-metrics-interval', interval=1000, n_intervals=0),

        # Sidebar toggle button
        html.Button('☰', id='sidebar-toggle', className='sidebar-toggle', title='Toggle sidebar [Esc]', n_clicks=0),

        # Collapsible sidebar
        html.Div([
            html.Div('Filters', className='section-title'),

            dcc.Checklist(
                id='filter-unreviewed',
                options=[{'label': ' Only unreviewed', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '3px'},
                persistence=True,
                persistence_type='local',
            ),
            dcc.Checklist(
                id='filter-failed',
                options=[{'label': ' Require failed_any=False', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'},
                persistence=True,
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
                persistence=True,
                persistence_type='local',
            ),
            dcc.Checklist(
                id='sort-desc',
                options=[{'label': ' Descending', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'},
                persistence=True,
                persistence_type='local',
            ),

            html.Button('Refresh Queue [Shift+R]', id='refresh-btn', n_clicks=0,
                       style={'width': '100%', 'font-size': '11px'}, className='action-btn'),

            html.Div('Open Existing', className='section-title', style={'margin-top': '8px'}),
            dcc.Input(
                id='candidate-search-query',
                placeholder='candidate_id or ASAS-SN ID',
                type='text',
                style={**_inp_style, 'marginBottom': '4px'},
            ),
            html.Button(
                'View Candidate',
                id='candidate-search-btn',
                n_clicks=0,
                className='action-btn',
                style={'width': '100%'},
            ),

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
                persistence=True,
                persistence_type='local',
            ),

            html.Hr(),

            # -- Fetch Candidate --
            html.Div('Fetch Candidate', className='section-title'),
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
            ),
            dcc.Input(id='fetch-query', placeholder='e.g. 427299085038300900', type='text',
                      style={**_inp_style, 'marginBottom': '4px'}),
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
                     persistence=True, persistence_type='local'),

            dcc.Checklist(
                id='import-lc-mode',
                options=[{'label': ' Raw light curve file (CSV/parquet with JD, mag columns)', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '4px', 'fontSize': '10px'},
                persistence=True, persistence_type='local',
            ),

            html.Label('Characterize on import:'),
            dcc.Checklist(
                id='characterize-on-import',
                options=[{'label': ' Enable', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=True, persistence_type='local',
            ),

            dcc.Input(id='characterize-crossmatch', placeholder='Crossmatch CSV', type='text',
                     value=str(VSX_CROSSMATCH_PATH), style=_inp_style,
                     persistence=True, persistence_type='local'),
            dcc.Input(id='characterize-gaia-cache', placeholder='Gaia cache', type='text',
                     value=str(GAIA_CACHE_FILE), style=_inp_style,
                     persistence=True, persistence_type='local'),
            dcc.Input(id='characterize-chunk-size', placeholder='Chunk size', type='number',
                     value=GAIA_CHUNK_SIZE, style=_inp_style,
                     persistence=True, persistence_type='local'),
            dcc.Checklist(
                id='characterize-dust',
                options=[{'label': ' Enable dustmaps3d', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=True, persistence_type='local',
            ),
            dcc.Input(id='characterize-starhorse', placeholder='StarHorse (tap or path)', type='text',
                     value='tap', style=_inp_style,
                     persistence=True, persistence_type='local'),

            html.Label('Vet on import:'),
            dcc.Checklist(
                id='vet-on-import',
                options=[{'label': ' Enable', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'},
                persistence=True, persistence_type='local',
            ),

            html.Button('Import', id='import-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%'}),

            html.Hr(),

            html.Div('Export', className='section-title'),
            dcc.Input(id='export-path', placeholder='Export file path', type='text',
                     value='output/review/reviewed_candidates.parquet', style=_inp_style),
            dcc.Checklist(
                id='export-only-reviewed',
                options=[{'label': ' Only reviewed', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'}
            ),
            html.Button('Export Reviews', id='export-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%'}),

            html.Hr(),
            
            html.Div('External Links', className='section-title'),
            html.Label('Search Radius (arcsec):'),
            dcc.Input(id='link-radius-arcsec', placeholder='Radius (arcsec)', type='number',
                     value=10.0, min=0.1, step=1.0, style=_inp_style,
                     persistence=True, persistence_type='local'),
                     
            html.Hr(),

            html.Div('Pace Timer', className='section-title'),
            dcc.Checklist(
                id='pace-timer-toggle',
                options=[{'label': ' Show Timer', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '4px'}
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
            ),

            html.Div(id='sidebar-status', style={'margin-top': '10px', 'color': '#0f0', 'font-size': '11px'}),

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
                                ),
                                dcc.RadioItems(
                                    id='plot-mode',
                                    options=[
                                        {'label': ' Native', 'value': 'native'},
                                        {'label': ' PNG', 'value': 'png'},
                                    ],
                                    value='native',
                                    inline=True,
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
                                    persistence=True,
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
                                    ),
                                ], className='toolbar-slider-control'),
                                html.Button('Reset', id='plot-reset-btn', n_clicks=0, className='compact-btn'),
                                html.Button('Export', id='export-plot', n_clicks=0, className='compact-btn'),
                                html.Span(id='repro-badge', className='label-chip', style={'margin-left': '6px'}),
                                html.Div([
                                    html.Span('LC Source', style={'color': '#9fb6cb', 'font-size': '10px',
                                                                  'white-space': 'nowrap', 'margin-right': '4px'}),
                                    dbc.Select(
                                        id='external-source-view',
                                        options=EXTERNAL_SOURCE_VIEW_OPTIONS,
                                        value='all',
                                        size='sm',
                                        style={'width': '140px', 'font-size': '10px', 'minWidth': '140px'},
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
                                    ),
                                    dcc.Input(id='pdm-min-period', type='number', value=0.1, min=0.001,
                                              step='any', debounce=True, placeholder='Min P',
                                              style={'width': '72px', 'font-size': '10px'}),
                                    html.Span('–', style={'color': '#9fb6cb', 'margin': '0 2px', 'font-size': '10px'}),
                                    dcc.Input(id='pdm-max-period', type='number', value=10, min=0.001,
                                              step='any', debounce=True, placeholder='Max P',
                                              style={'width': '72px', 'font-size': '10px'}),
                                    html.Span('d', style={'color': '#9fb6cb', 'font-size': '10px', 'margin-right': '4px'}),
                                    html.Button('Find Period', id='pdm-run-btn', n_clicks=0, className='compact-btn'),
                                    dcc.Input(id='pdm-manual-period', type='number', min=0.001,
                                              step=0.001, placeholder='Manual P (d)',
                                              style={'width': '90px', 'font-size': '10px', 'margin-left': '4px'}),
                                    html.Span(id='pdm-result-label', style={'color': '#7da8c4', 'font-size': '10px',
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
                                dcc.Loading(
                                    id='loading-pipeline', type='dot',
                                    children=html.Div(id='pipeline-run-status',
                                                      style={'fontSize': '10px', 'marginTop': '2px',
                                                             'color': '#7da8c4'}),
                                ),
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
                        # Run config / reproducibility
                        html.Details([
                            html.Summary('Run Config', style={'cursor': 'pointer'}),
                            html.Div([
                                html.Div([
                                    html.Button('Copy Config JSON', id='copy-run-config-btn', n_clicks=0, className='compact-btn', style={'margin-right': '6px'}),
                                    html.Button('Download Config JSON', id='download-run-config-btn', n_clicks=0, className='compact-btn'),
                                ], style={'margin-bottom': '6px'}),
                                html.Div(id='run-config-panel', className='run-config-panel'),
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
    function(n_intervals, startTime, toggle) {
        // Toggle the review-progress-indicator visibility
        var el = document.getElementById('review-progress-indicator');
        if (el) {
            if (toggle && toggle.indexOf('yes') !== -1) {
                el.style.display = '';
            } else {
                el.style.display = 'none';
            }
        }
        return '';
    }
    """,
    Output('pace-timer-display', 'children'),
    Input('review-metrics-interval', 'n_intervals'),
    [State('candidate-start-time', 'data'),
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

                if (e.shiftKey && (key === 'R' || key === 'r')) {
                    e.preventDefault();
                    dispatchKeyToDash('Shift+R');
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
    function(_tick, currentTheme) {
        try {
            var saved = window.localStorage.getItem('malca.review.theme');
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
    Input('keyboard-init', 'n_intervals'),
    State('theme-mode', 'value'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(theme) {
        var t = ['black', 'gray', 'white'].includes(theme) ? theme : 'black';
        try {
            document.body.setAttribute('data-theme', t);
            window.localStorage.setItem('malca.review.theme', t);
        } catch (e) {
            // ignore storage/document failures
        }
        return t;
    }
    """,
    Output('theme-mode-store', 'data'),
    Input('theme-mode', 'value'),
    prevent_initial_call=False,
)


# --- Sidebar plot prefs: save to localStorage on change ---
app.clientside_callback(
    """
    function(preset, overlays, mode, opacity, resHeight, externalSource) {
        try {
            var obj = {
                preset: preset,
                overlays: overlays || [],
                mode: mode,
                opacity: opacity,
                resHeight: resHeight,
                externalSource: externalSource
            };
            window.localStorage.setItem('malca.review.sidebar.plot.v2', JSON.stringify(obj));
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
    prevent_initial_call=True,
)


# --- Sidebar plot prefs: load from localStorage on init ---
app.clientside_callback(
    """
    function(_tick, curPreset, curOverlays, curMode, curOpacity, curResHeight, curExternalSource) {
        var nu = window.dash_clientside.no_update;
        try {
            var raw = window.localStorage.getItem('malca.review.sidebar.plot.v2');
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
    Input('keyboard-init', 'n_intervals'),
    [State('plot-preset', 'value'),
     State('plot-overlays', 'value'),
     State('plot-mode', 'value'),
     State('baseline-opacity-slider', 'value'),
     State('residual-height-slider', 'value'),
     State('external-source-view', 'value')],
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
    """Trigger Refresh Queue from Shift+R."""
    if _keyboard_key(key_value) != 'Shift+R':
        raise dash.exceptions.PreventUpdate
    return int(current_clicks or 0) + 1


# --- All filter State components used by load_queue -------------------------
# Auto-generated from _SIDEBAR_GROUPS so the UI and callback always stay in sync.
_BOOL_MODE_STATES: list[tuple[str, str]] = []
_NUM_STATES: list[tuple[str, str]] = []
_TEXT_STATES: list[tuple[str, str]] = []
_SELECT_STATES: list[tuple[str, str]] = []

for _grp_name, _grp_items in _SIDEBAR_GROUPS:
    for _ftype, _col in _grp_items:
        _cid = _col_id(_col)
        if _ftype == 'bool':
            _BOOL_MODE_STATES.append((f'{_cid}-mode', f'{_col}_mode'))
        elif _ftype == 'num':
            _NUM_STATES.append((f'min-{_cid}', f'min_{_col}'))
            _NUM_STATES.append((f'max-{_cid}', f'max_{_col}'))
        elif _ftype == 'text':
            _TEXT_STATES.append((f'filter-{_cid}', _col))
        elif _ftype == 'select':
            _SELECT_STATES.append((f'exclude-{_cid}', f'exclude_{_col}'))

# Build the State list for the callback decorator dynamically.
_queue_states = (
    [State('filter-unreviewed', 'value'),
     State('filter-failed', 'value')]
    + [State(cid, 'value') for cid, _ in _BOOL_MODE_STATES]
    + [State(cid, 'value') for cid, _ in _NUM_STATES]
    + [State(cid, 'value') for cid, _ in _TEXT_STATES]
    + [State(cid, 'value') for cid, _ in _SELECT_STATES]
    + [State('sort-col', 'value'),
       State('sort-desc', 'value')]
)


# Initialize queue
@app.callback(
    Output('queue-data', 'data'),
    [Input('refresh-btn', 'n_clicks'),
     Input('import-trigger', 'data'),
     Input('queue-source-path', 'data')],
    _queue_states,
    prevent_initial_call=False
)
def load_queue(refresh_clicks, import_trigger, queue_source_scope, *state_values):
    """Load queue data from all sidebar filter states."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        # Unpack state values in the same order as _queue_states
        it = iter(state_values)
        filter_unreviewed = next(it)
        filter_failed = next(it)

        filter_params: dict = {
            'only_unreviewed': 'yes' in (filter_unreviewed or []),
            'require_failed_any_false': 'yes' in (filter_failed or []),
        }
        for _, fkey in _BOOL_MODE_STATES:
            filter_params[fkey] = next(it) or 'Any'
        for _, fkey in _NUM_STATES:
            val = next(it)
            filter_params[fkey] = val if val is not None else None
        for _, fkey in _TEXT_STATES:
            val = next(it)
            filter_params[fkey] = val.strip() if val else None
        for _, fkey in _SELECT_STATES:
            val = next(it)
            filter_params[fkey] = val if val else None

        sort_val = next(it)
        if isinstance(sort_val, list):
            filter_params['sort_cols'] = sort_val or ['candidate_id']
        else:
            filter_params['sort_cols'] = [sort_val] if sort_val else ['candidate_id']
        filter_params['sort_desc'] = 'yes' in (next(it) or [])

        if queue_source_scope:
            filter_params['source_path_like'] = str(queue_source_scope)

        queue_data = create_queue_data_dict(conn, filter_params)
        active_filters = {k: v for k, v in filter_params.items() if v and v != 'Any'}
        print(f"[queue] size={queue_data['queue_size']} active_filters={active_filters}")
        return queue_data


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


def _has_external_period(payload: dict | None) -> bool:
    """Whether a payload already has a catalog or validated pipeline period."""
    payload = payload or {}
    for keys in (
        ("phase_period_days",),
        ("period_consensus_days",),
        ("vsx_period", "period_vsx_days"),
        ("asassn_var_period", "period_asassn_var_days"),
        ("gaia_eb_period", "period_gaia_eb_days"),
        ("ztf_var_period", "period_ztf_periodic_days"),
        ("catalog_period",),
    ):
        for key in keys:
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return True
    return False


def _run_period_search_for_payload(
    payload: dict,
    *,
    min_period: float,
    max_period: float,
    method: str,
) -> tuple[dict | None, str]:
    """Run a period search against the current candidate payload."""
    plot_dir_path = Path(PLOT_DIR) if PLOT_DIR else Path('.')
    lc_path = resolve_lightcurve_path(payload, plot_dir_path)
    if lc_path is None:
        return None, 'No LC file'

    from malca.periodogram import ce_find_period, lsp_find_period, pdm_find_period

    df, _, _ = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=True,
        scatter_ratio=2.5,
        clean_max_error_absolute=1.0,
        clean_max_error_sigma=5.0,
    )
    if df is None or df.empty:
        return None, 'Empty LC'

    baseline_cache_key = (str(lc_path.resolve()), (), True, 2.5, 1.0, 5.0)
    band_dfs = _compute_baseline_bands(df, "per_camera_gp", baseline_cache_key)

    resid_parts = []
    for bdf in band_dfs.values():
        if "resid" not in bdf.columns:
            continue
        mask = np.isfinite(bdf["JD"].to_numpy()) & np.isfinite(bdf["resid"].to_numpy())
        resid_parts.append(bdf[mask][["JD", "resid"]])
    if not resid_parts:
        return None, 'No residuals'

    resid_df = pd.concat(resid_parts, ignore_index=True)
    times = resid_df['JD'].to_numpy()
    values = resid_df['resid'].to_numpy()
    if len(times) < 10:
        return None, 'Too few points'

    method = str(method or 'pdm').lower()
    if method == 'pdm':
        best_period, _, _ = pdm_find_period(times, values, min_period=min_period, max_period=max_period)
        label = 'PDM'
    elif method == 'ce':
        best_period, _, _ = ce_find_period(times, values, min_period=min_period, max_period=max_period)
        label = 'CE'
    else:
        best_period, _, _ = lsp_find_period(times, values, min_period=min_period, max_period=max_period)
        label = 'LSP'

    return {'best_period': float(best_period), 'method': label}, f'{label}: P={best_period:.5f} d'



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
def open_existing_candidate(n_clicks, n_submit, query, queue_data):
    """Jump to an existing candidate in the DB by candidate_id or ASAS-SN ID."""
    _ = n_clicks, n_submit
    query_text = str(query or '').strip()
    if not query_text:
        raise dash.exceptions.PreventUpdate

    with closing(db_connect(Path(DB_PATH))) as conn:
        row = conn.execute(
            "SELECT candidate_id FROM candidates WHERE candidate_id = ? COLLATE NOCASE LIMIT 1",
            (query_text,),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT candidate_id FROM candidates WHERE asas_sn_id = ? COLLATE NOCASE LIMIT 1",
                (query_text,),
            ).fetchone()

    if row is None:
        return no_update, no_update, f"Candidate not found in DB: {query_text}"

    candidate_id = str(row[0])
    candidate_ids = list((queue_data or {}).get('candidate_ids') or []) if isinstance(queue_data, dict) else []
    if candidate_id in candidate_ids:
        return no_update, candidate_ids.index(candidate_id), f"Jumped to {candidate_id} in the current queue."

    return (
        {'candidate_ids': [candidate_id], 'queue_size': 1, 'filter_hash': f'view:{candidate_id}'},
        0,
        f"Viewing {candidate_id}. Refresh Queue to restore the filtered queue.",
    )


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
        return new_pass, status


# Keyboard handler (single-key class shortcuts)
@app.callback(
    [Output('current-index', 'data'),
     Output('notification', 'children'),
     Output('current-score', 'data', allow_duplicate=True),
     Output('needs-followup-store', 'data', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True),
     Output('event-class-store', 'data', allow_duplicate=True),
     Output('pending-prefix', 'data', allow_duplicate=True)],
    Input('keyboard-input', 'value'),
    [State('current-index', 'data'),
     State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('pending-prefix', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def handle_keyboard(key_value, current_idx, queue_size, current_candidate_id, current_score,
                    event_class, pending_prefix, needs_followup, notes):
    """Handle keyboard input."""
    NO = (no_update,) * 7  # shorthand for all-no_update

    key = _keyboard_key(key_value)
    if not key:
        return NO

    # Skip keys handled by other callbacks / keydown listener
    if key in ['?', 'Escape', 'Shift+R']:
        return NO

    queue_size = int(queue_size or 0)
    if queue_size == 0:
        return no_update, "Queue is empty", *([no_update] * 5)

    candidate_id = str(current_candidate_id) if current_candidate_id else None

    # --- Direct class key shortcuts (single key toggles class) ---
    kl = key.lower()
    class_tag = CLASS_KEY_MAP.get(kl)
    if class_tag is not None:
        cur = event_class or 'unclassified'
        if cur == class_tag:
            return no_update, "Class: unclassified", no_update, no_update, no_update, 'unclassified', no_update
        return no_update, f"Class: {class_tag}", no_update, no_update, no_update, class_tag, no_update

    # --- Followup toggle (,) ---
    if key == ',':
        new_state = not bool(needs_followup)
        label = "ON" if new_state else "OFF"
        return no_update, f"Followup: {label}", no_update, new_state, no_update, no_update, no_update

    if key == 'Enter':
        if current_score is None:
            return no_update, "⚠ Confidence required", no_update, no_update, no_update, no_update, no_update
        if not event_class or event_class == 'unclassified':
            return no_update, "⚠ Class required", no_update, no_update, no_update, no_update, no_update

    # --- Navigation / scoring / save via handle_key_action ---
    with closing(db_connect(Path(DB_PATH))) as conn:
        new_idx, notification, should_save = handle_key_action(
            key, current_idx, queue_size, conn, candidate_id
        )

    new_score = no_update
    new_pass = no_update
    if should_save and candidate_id:
        score = int(key) if key in '1234' else current_score
        is_done = (key == 'Enter')
        pass_val, _ = _do_save(
            candidate_id, score, event_class, needs_followup, notes, 'keyboard',
            increment_pass=is_done,
        )
        new_score = score
        new_pass = pass_val

    # Only update current-index when it actually changes (navigation).
    # Returning the same value would re-trigger load_review_form needlessly.
    idx_out = new_idx if new_idx != current_idx else no_update

    return idx_out, notification, new_score, no_update, new_pass, no_update, no_update


@app.callback(
    Output('plot-render-request', 'data'),
    [Input('current-index', 'data'),
     Input('plot-mode', 'value'),
     Input('plot-overlays', 'value'),
     Input('camera-checklist', 'value'),
     Input('plot-preset', 'value'),
     Input('residual-height-slider', 'value'),
     Input('theme-mode-store', 'data'),
     Input('queue-size-store', 'data'),
     Input('baseline-opacity-slider', 'value'),
     Input('round-sigfigs', 'value'),
     Input('link-radius-arcsec', 'value'),
     Input('pdm-result-store', 'data'),
     Input('pdm-manual-period', 'value'),
     Input('yaxis-mode', 'value'),
     Input('external-source-view', 'value')],
    State('plot-render-request', 'data'),
    prevent_initial_call=True,
)
def queue_plot_render_request(idx, plot_mode, overlay_values, selected_cameras, preset, residual_height, theme_mode, _queue_size, baseline_opacity, round_sigfigs, link_radius, pdm_result, pdm_manual_period, yaxis_mode, external_source_view, existing_request):
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
            'plot_mode': plot_mode,
            'overlay_values': list(overlay_values or []),
            'selected_cameras': list(selected_cameras or []),
            'preset': preset,
            'residual_height': float(residual_height if residual_height is not None else DEFAULT_RESIDUAL_FRACTION),
            'theme': theme_mode or DEFAULT_THEME,
            'baseline_opacity': float(baseline_opacity if baseline_opacity is not None else 0.5),
            'round_sigfigs': bool(True if round_sigfigs is None else ('yes' in round_sigfigs)),
            'link_radius': float(link_radius) if link_radius is not None else 10.0,
            'override_period': override_period,
            'yaxis_mode': str(yaxis_mode or 'mag'),
            'external_source_view': str(external_source_view or 'all'),
        },
    }


@app.callback(
    [Output('pdm-result-store', 'data'),
     Output('pdm-result-label', 'children'),
     Output('auto-period-cache', 'data', allow_duplicate=True)],
    Input('pdm-run-btn', 'n_clicks'),
    [State('current-candidate-id', 'data'),
     State('pdm-min-period', 'value'),
     State('pdm-max-period', 'value'),
     State('period-method', 'value'),
     State('auto-period-cache', 'data')],
    prevent_initial_call=True,
)
def run_period_search(n_clicks, candidate_id, min_period, max_period, method, auto_period_cache):
    """Run period search (LSP/PDM/CE) on current candidate's light curve."""
    if not n_clicks or not candidate_id:
        raise dash.exceptions.PreventUpdate
    candidate_id = str(candidate_id)
    auto_period_cache = dict(auto_period_cache or {})
    try:
        min_p = float(min_period) if min_period else 0.1
        max_p = float(max_period) if max_period else 100.0
    except (TypeError, ValueError):
        min_p, max_p = 0.1, 100.0
    if min_p <= 0:
        min_p = 0.01
    if max_p <= min_p:
        max_p = min_p + 1.0
    method = str(method or 'pdm').lower()

    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, candidate_id)

    result, label = _run_period_search_for_payload(payload, min_period=min_p, max_period=max_p, method=method)
    auto_period_cache[candidate_id] = {'result': result, 'label': label}
    return result, label, auto_period_cache


@app.callback(
    [Output('pdm-result-store', 'data', allow_duplicate=True),
     Output('pdm-result-label', 'children', allow_duplicate=True),
     Output('pdm-manual-period', 'value', allow_duplicate=True),
     Output('auto-period-cache', 'data', allow_duplicate=True)],
    Input('current-candidate-id', 'data'),
    [State('pdm-min-period', 'value'),
     State('pdm-max-period', 'value'),
     State('auto-period-cache', 'data')],
    prevent_initial_call=True,
)
def auto_period_on_navigate(candidate_id, min_period, max_period, auto_period_cache):
    """Auto-run a first-pass PDM search for the currently viewed candidate."""
    if candidate_id is None:
        return None, '', None, no_update
    candidate_id = str(candidate_id)
    auto_period_cache = dict(auto_period_cache or {})

    cached_entry = auto_period_cache.get(candidate_id)
    if isinstance(cached_entry, dict):
        return (
            cached_entry.get('result'),
            str(cached_entry.get('label', '')),
            None,
            no_update,
        )

    try:
        min_p = float(min_period) if min_period else 0.1
        max_p = float(max_period) if max_period else 100.0
    except (TypeError, ValueError):
        min_p, max_p = 0.1, 100.0
    if min_p <= 0:
        min_p = 0.01
    if max_p <= min_p:
        max_p = min_p + 1.0

    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, candidate_id)

    if _has_external_period(payload):
        return None, 'Catalog/pipeline period', None, no_update

    result, label = _run_period_search_for_payload(
        payload,
        min_period=min_p,
        max_period=max_p,
        method='pdm',
    )
    display_label = f'Auto {label}' if result is not None else f'Auto search: {label}'
    if result is None:
        auto_period_cache[candidate_id] = {'result': None, 'label': display_label}
        return None, display_label, None, auto_period_cache
    result['auto'] = True
    auto_period_cache[candidate_id] = {'result': result, 'label': display_label}
    return result, display_label, None, auto_period_cache


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
        return new_overlays, new_cams, 'all'
    if trig == 'cams-all-btn':
        return overlays, list(cams), no_update
    if trig == 'cams-clear-btn':
        return overlays, [], no_update
    if trig == 'cams-invert-btn':
        inv = [c for c in cams if c not in set(selected)]
        return overlays, inv, no_update
    return no_update, no_update, no_update


@app.callback(
    [Output('plot-image', 'src'),
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
     Output('plot-render-applied', 'data')],
    Input('plot-render-request', 'data'),
    [State('plot-render-applied', 'data'),
     State('current-candidate-id', 'data'),
     State('queue-size-store', 'data')],
    prevent_initial_call=False,
)
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
    theme_mode = str(state.get('theme', DEFAULT_THEME) or DEFAULT_THEME)
    residual_height = float(state.get('residual_height', DEFAULT_RESIDUAL_FRACTION) or DEFAULT_RESIDUAL_FRACTION)
    baseline_opacity = float(state.get('baseline_opacity', 0.5) if state.get('baseline_opacity') is not None else 0.5)
    round_sigfigs = bool(state.get('round_sigfigs', True))
    link_radius = float(state.get('link_radius', 10.0))
    yaxis_mode = str(state.get('yaxis_mode', 'mag') or 'mag')
    external_source_view = str(state.get('external_source_view', 'all') or 'all')
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
    if queue_size <= 0 or not current_candidate_id:
        return '', 'No candidates in queue', _render_metadata_health(None, context_msg='Queue is empty.'), _render_vetting_banner(None, radius_arcsec=link_radius), '[0/0]', empty_fig, {'display': 'block', 'width': '100%', 'height': '100%'}, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'No candidates in queue.', []), _render_camera_diag_panel({}, []), _render_run_config_panel(None, None, ['Queue is empty']), _render_repro_badge(None, ['Queue is empty']), '', nonce

    if idx < 0 or idx >= queue_size:
        return '', 'Invalid index', _render_metadata_health(None, context_msg='Invalid queue index.'), _render_vetting_banner(None, radius_arcsec=link_radius), f'[{idx}/{queue_size}]', empty_fig, {'display': 'block', 'width': '100%', 'height': '100%'}, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'Invalid queue index.', []), _render_camera_diag_panel({}, []), _render_run_config_panel(None, None, ['Invalid queue index']), _render_repro_badge(None, ['Invalid queue index']), '', nonce

    candidate_id = str(current_candidate_id)
    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, candidate_id)

    plot_src = ''
    plot_dir_path = Path(PLOT_DIR) if PLOT_DIR else Path('.')
    plot_path = find_plot_image(payload, plot_dir_path)
    if plot_path and plot_path.exists():
        try:
            rel_path = plot_path.relative_to(plot_dir_path)
            plot_src = f'/plots/{rel_path}'
        except ValueError:
            plot_src = f'/plots/{plot_path.name}'

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

        png_src = plot_src
        png_msg = 'PNG view enabled. Switch to Native for interactive hover and diagnostics.'
        if 'phase' in overlays:
            phase_plot_path = find_phase_plot_image(payload, plot_dir_path)
            if phase_plot_path and phase_plot_path.exists():
                try:
                    rel_phase = phase_plot_path.relative_to(plot_dir_path)
                    png_src = f'/plots/{rel_phase}'
                except ValueError:
                    png_src = f'/plots/{phase_plot_path.name}'
                png_msg = 'Showing phase-folded PNG view.'
            else:
                mismatch_warnings.append('Phase-fold overlay selected, but no phase PNG was found for this candidate.')
        panel = _render_run_config_panel(run_params if run_params else None, run_params_path, mismatch_warnings)
        return (
            png_src,
            grid_items,
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
    uirevision_key = f"{candidate_id}|{','.join(sorted(str(c) for c in selected_cameras))}|{theme_mode}|{residual_height:.3f}|{baseline_opacity:.2f}|{yaxis_mode}|{external_source_view}"

    # Discover external LC parquets for this candidate
    ext_lcs: dict[str, Path] | None = None
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
        import traceback
        traceback.print_exc()
        panel = _render_run_config_panel(run_params if run_params else None, run_params_path, [str(exc)])
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

    if native_status in {"missing-file", "missing-columns", "empty-after-filter", "empty-camera-selection"} and plot_src:
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
            _render_run_config_panel(run_params if run_params else None, run_params_path, fallback_warnings),
            _render_repro_badge(run_params if run_params else None, fallback_warnings),
            json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
            nonce,
        )

    filtered = []
    if 'Filtered cams' in {k for k, _ in native.get('stat_rows', [])}:
        for key, val in native.get('stat_rows', []):
            if key == 'Filtered cams':
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
        _render_plot_status_panel(native.get('status', 'ok'), native.get('status_message', ''), (native.get('warnings', []) + mismatch_warnings)),
        _render_camera_diag_panel(native.get('camera_diagnostics', {}), filtered),
        _render_run_config_panel(run_params if run_params else None, run_params_path, mismatch_warnings),
        _render_repro_badge(run_params if run_params else None, mismatch_warnings),
        json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
        nonce,
    )


@app.callback(
    Output('external-followup-panel', 'children'),
    [Input('external-followup-details', 'open'),
     Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data')],
    prevent_initial_call=False,
)
def update_external_followup_panel(is_open, candidate_id, theme_mode):
    """Render external follow-up artifacts for the current candidate."""
    _ = is_open
    if not candidate_id:
        return html.Div("No candidates loaded.", style={'font-size': '11px', 'color': '#c77'})
    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, str(candidate_id)) or {}

    return _render_external_followup(payload, str(candidate_id), str(theme_mode or DEFAULT_THEME))


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
        queue_label = str(queue_source_path)

    def _bottom_bar(path_value: object) -> html.Div:
        return html.Div(
            [
                _render_bottom_context("Path", path_value if path_value else "-"),
                _render_bottom_context("DB", DB_PATH),
                _render_bottom_context("Queue", queue_label),
            ],
            className='bottom-context-bar',
        )

    if int(queue_size or 0) <= 0 or not candidate_id:
        return 'ASAS-SN ID: -', 'Gaia ID: -', _bottom_bar("-")

    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, str(candidate_id)) or {}
    asas_sn_id = payload.get('asas_sn_id')
    gaia_id = payload.get('gaia_id')
    lc_path = payload.get('path')

    asas_text = f"ASAS-SN ID: {asas_sn_id}" if asas_sn_id else f"ASAS-SN ID: {candidate_id}"
    gaia_fmt = _format_large_integer_like_display(gaia_id)
    gaia_text = f"Gaia ID: {gaia_fmt}" if gaia_fmt else 'Gaia ID: -'
    return asas_text, gaia_text, _bottom_bar(lc_path)


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
                with closing(db_connect(Path(DB_PATH))) as conn:
                    payload = get_candidate_payload(conn, str(candidate_id))
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
        if src.startswith('/plots/') and PLOT_DIR:
            rel = src[len('/plots/'):]
            plot_file = Path(PLOT_DIR) / rel
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


# Score button clicks
@app.callback(
     [Output('current-score', 'data', allow_duplicate=True),
      Output('notification', 'children', allow_duplicate=True),
      Output('review-pass-store', 'data', allow_duplicate=True)],
     [Input(f'score-{i}', 'n_clicks') for i in range(1, 5)],
     [State('queue-size-store', 'data'),
      State('current-candidate-id', 'data'),
      State('event-class-store', 'data'),
      State('needs-followup-store', 'data'),
      State('notes', 'value')],
     prevent_initial_call=True
)
def handle_score_clicks(*args):
    """Handle score button clicks."""
    queue_size, candidate_id, event_class, needs_followup, notes = args[-5:]

    if int(queue_size or 0) <= 0 or not candidate_id:
        return no_update, "Queue is empty", no_update

    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    score = int(button_id.split('-')[1])

    new_pass, _ = _do_save(
        str(candidate_id), score, event_class, needs_followup, notes, 'button',
    )

    return score, f"✓ Confidence: {score}", new_pass


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

# Update score button highlighting
@app.callback(
    [Output(f'score-{i}', 'className') for i in range(1, 5)],
    Input('current-score', 'data'),
    prevent_initial_call=False
)
def update_score_buttons(current_score):
    """Highlight the active score button."""
    try:
        score = int(current_score)
    except Exception:
        score = None
    if score not in (1, 2, 3, 4):
        score = None
    return ['score-btn active' if i == score else 'score-btn' for i in range(1, 5)]


# Update event class badge styling
@app.callback(
    [Output(f'class-badge-{tag}', 'className') for tag in CLASS_BADGE_TAGS],
    Input('event-class-store', 'data'),
    prevent_initial_call=False
)
def update_class_badges(active_class):
    """Highlight the active event class badge."""
    active = active_class or 'unclassified'
    return ['badge-btn active' if tag == active else 'badge-btn'
            for tag in CLASS_BADGE_TAGS]


# Class badge click handler
@app.callback(
    [Output('event-class-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input(f'class-badge-{tag}', 'n_clicks') for tag in CLASS_BADGE_TAGS],
    State('event-class-store', 'data'),
    prevent_initial_call=True
)
def handle_class_clicks(*args):
    """Set event class when its badge is clicked (click again to clear)."""
    current_class = args[-1]
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    tag = trigger.replace('class-badge-', '')
    cur = current_class or 'unclassified'
    if cur == tag:
        return 'unclassified', "Class: unclassified"
    return tag, f"Class: {tag}"


# Prefix indicator
@app.callback(
    Output('prefix-indicator', 'children'),
    Input('pending-prefix', 'data'),
    prevent_initial_call=False
)
def update_prefix_indicator(prefix):
    """Show pending prefix key."""
    if prefix:
        return html.Span(f'[{prefix.upper()}] ...', style={'color': '#f80', 'font-weight': 'bold'})
    return ''


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


# Followup indicator
@app.callback(
    Output('followup-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    prevent_initial_call=False
)
def update_followup_indicator(needs_followup):
    """Show followup flag status."""
    if needs_followup:
        return html.Span('[,] Followup: ON', style={'color': '#f80'})
    return html.Span('[,] Followup: off', style={'color': '#666'})


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
    Output('review-progress-indicator', 'children'),
    Input('review-metrics-interval', 'n_intervals'),
    Input('queue-size-store', 'data'),
    Input('current-index', 'data'),
    Input('review-pass-store', 'data'),
    State('review-session-start', 'data'),
    prevent_initial_call=False,
)
def update_review_progress_indicator(_tick, queue_size, _idx, _review_pass, session_start):
    """Render reviewed/total progress with session pace + elapsed timer."""
    _ = _tick, _idx, _review_pass

    with closing(db_connect(Path(DB_PATH))) as conn:
        reviewed, total = count_progress(conn)

    queue_size = int(queue_size or 0)

    if total <= 0:
        total = queue_size

    pct = (100.0 * reviewed / total) if total > 0 else 0.0

    start_ts = None
    if isinstance(session_start, dict):
        start_ts = session_start.get('ts')
    try:
        start_ts = float(start_ts) if start_ts is not None else None
    except Exception:
        start_ts = None
    if start_ts is None or (not np.isfinite(start_ts)):
        start_ts = time.time()

    elapsed_s = max(0.0, time.time() - start_ts)
    elapsed_txt = _format_hms(elapsed_s)

    pace_per_min = 0.0
    if elapsed_s > 0 and reviewed > 0:
        pace_per_min = float(reviewed) / (elapsed_s / 60.0)

    if pace_per_min > 0 and total > reviewed:
        remaining = total - reviewed
        eta_s = (remaining / pace_per_min) * 60.0
        eta_txt = _format_hms(eta_s)
    else:
        eta_txt = "--:--:--"

    pace_txt = f"{pace_per_min:.2f}/min" if pace_per_min > 0 else "--/min"
    return (
        f"Reviewed: {reviewed}/{total} ({pct:.1f}%) "
        f"| Elapsed: {elapsed_txt} "
        f"| Pace: {pace_txt} "
        f"| ETA: {eta_txt}"
    )


# Pass indicator
@app.callback(
    Output('pass-indicator', 'children'),
    Input('review-pass-store', 'data'),
    prevent_initial_call=False
)
def update_pass_indicator(review_pass):
    """Show current review pass number."""
    return f"Pass: {review_pass or 1}"


# Status indicator
@app.callback(
    Output('status-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    Input('current-score', 'data'),
    [State('current-candidate-id', 'data'),
     State('queue-size-store', 'data')],
    prevent_initial_call=False
)
def update_status_indicator(needs_followup, score, candidate_id, queue_size):
    """Show current effective status."""
    if not candidate_id or int(queue_size or 0) == 0:
        return "Status: —"
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, str(candidate_id))
    status = review.get('status', 'unreviewed')
    color = '#0f0' if status == 'reviewed' else '#f80' if status == 'needs_followup' else '#888'
    return html.Span(f"Status: {status}", style={'color': color})

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
        except Exception as e:
            detected = {'candidates': None, 'warnings': [f"Auto-detect failed: {str(e)}"]}

        candidates_path = detected.get('candidates')
        if candidates_path:
            resolved_candidates = str(Path(candidates_path).expanduser().resolve())
            vetting_mode = _vetting_mode_for_input(resolved_candidates)
            with closing(db_connect(Path(DB_PATH))) as conn:
                save_app_state(conn, "last_input_file", resolved_candidates)
            return resolved_candidates, (
                f"✓ Auto-detected candidates: {Path(resolved_candidates).name} | "
                f"Vetting mode: {vetting_mode}"
            )

        warnings = detected.get('warnings') or []
        if restored_path and Path(restored_path).exists():
            restored_mode = _vetting_mode_for_input(restored_path)
            return str(Path(restored_path).resolve()), (
                f"⚠ {warnings[0]} | restored last candidates: {Path(restored_path).name} | "
                f"Vetting mode: {restored_mode}"
                if warnings else (
                    f"✓ Restored last candidates: {Path(restored_path).name} | "
                    f"Vetting mode: {restored_mode}"
                )
            )

        if warnings:
            return no_update, f"⚠ {warnings[0]}"

    if restored_path and Path(restored_path).exists():
        restored_mode = _vetting_mode_for_input(restored_path)
        return str(Path(restored_path).resolve()), (
            f"✓ Restored last candidates: {Path(restored_path).name} | "
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
    run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
    if run_dir is not None:
        return run_dir.name

    scope = _extract_bundle_scope(import_path)
    return scope or ''


@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('import-path', 'value'),
    prevent_initial_call=True,
)
def show_vetting_mode_status(import_path):
    """Show whether import will use vetted input, cache, or re-vetting."""
    if not import_path:
        return no_update
    mode = _vetting_mode_for_input(import_path)
    return f"Vetting mode: {mode}"


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

    # Raw light curve mode
    if 'yes' in (lc_mode or []):
        try:
            from malca.review.store import import_lightcurve_files
            with closing(db_connect(Path(DB_PATH))) as conn:
                enable_characterize = 'yes' in (characterize_on or [])
                enable_vetting = 'yes' in (vet_on or [])
                n_rows, n_new = import_lightcurve_files(
                    conn, Path(import_path),
                    characterize=enable_characterize,
                    vet=enable_vetting,
                )
                return (
                    f"✓ LC import: {n_rows} rows ({n_new} new)",
                    (current_trigger or 0) + 1,
                )
        except Exception as e:
            return f"✗ LC import failed: {str(e)}", no_update

    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            src = Path(import_path).expanduser()
            df = load_candidates_file(src)

            enable_characterize = 'yes' in (characterize_on or [])
            enable_vetting = 'yes' in (vet_on or [])
            vetting_mode = _vetting_mode_for_input(src) if enable_vetting else "vetting disabled"

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
            save_app_state(conn, "last_input_file", str(src))
            # Increment trigger to cause queue refresh
            return (
                f"✓ Imported {n_rows} rows ({n_new} new) | Vetting mode: {vetting_mode}",
                (current_trigger or 0) + 1,
            )
    except Exception as e:
        return f"✗ Import failed: {str(e)}", no_update


def _fetch_candidate_impl(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, current_trigger):
    """Core fetch logic; set_progress is optional for streaming status to GUI."""
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
        if fetch_type == 'coords':
            progress("Searching cone...")
            from malca.review.fetch import fetch_cone_search
            parts = query.replace(',', ' ').split()
            if len(parts) < 2:
                return "✗ Enter RA and Dec separated by space", no_update, no_update, no_update, no_update, no_update, no_update
            ra, dec = float(parts[0]), float(parts[1])
            radius = float(parts[2]) if len(parts) > 2 else 5.0
            catalog_df = fetch_cone_search(ra, dec, radius_arcsec=radius)
            if catalog_df.empty:
                return "✗ No sources found in cone search", no_update, no_update, '', no_update, no_update, no_update
            show_cols = [c for c in ['asas_sn_id', 'ra_deg', 'dec_deg', 'gaia_id', 'mean_vmag']
                         if c in catalog_df.columns]
            if not show_cols:
                show_cols = catalog_df.columns[:5].tolist()
            table = html.Table([
                html.Thead(html.Tr([html.Th(c, style={'padding': '2px 6px'}) for c in show_cols])),
                html.Tbody([
                    html.Tr(
                        [html.Td(str(row.get(c, '')), style={'padding': '2px 6px'}) for c in show_cols],
                        id={'type': 'cone-row', 'index': i},
                        style={'cursor': 'pointer'},
                        className='cone-result-row',
                    )
                    for i, row in catalog_df[show_cols].iterrows()
                ]),
            ], style={'borderCollapse': 'collapse', 'width': '100%', 'marginTop': '4px',
                      'border': '1px solid #3c5e75', 'color': '#cad9e5'})
            n_found = len(catalog_df)
            cone_data = catalog_df.to_dict('records')
            progress(f"Found {n_found} source(s)")
            return (
                f"Found {n_found} source(s). Click a row to fetch its LC.",
                no_update,
                no_update,
                table,
                cone_data,
                no_update,
                no_update,
            )

        progress("Fetching light curve...")
        if fetch_type == 'gaia':
            from malca.review.fetch import fetch_and_analyze_by_gaia_id
            df, lc_path = fetch_and_analyze_by_gaia_id(query, run_stats=True)
        else:
            from malca.review.fetch import fetch_and_analyze_by_id
            df, lc_path = fetch_and_analyze_by_id(query, run_stats=True)

        if df is None or df.empty:
            return f"✗ No data for {query}", no_update, no_update, '', no_update, no_update, no_update

        progress("Importing basic light curve...")
        
        # We NEVER run characterization/vetting in the fetch callback anymore,
        # so the GUI can render the light curve IMMEDIATELY.
        with closing(db_connect(Path(DB_PATH))) as conn:
            n_rows, n_new = import_candidates(
                conn, df, source_path=f"fetch://{fetch_type}/{query}",
                characterize_before_import=False,
                vet_before_import=False,
            )

        cid = str(df.iloc[0]['candidate_id']) if 'candidate_id' in df.columns else query
        _index_external_lc_paths.cache_clear()
        _index_external_lc_paths_from_root.cache_clear()
        
        auto_run = no_update
        if fetch_mode in ('full', 'full_ext'):
            auto_run = {'candidate_id': cid, 'mode': fetch_mode, 'ts': time.time()}

        status = f"✓ Added {query} ({n_new} new)"
        return (
            status,
            no_update,
            auto_run,
            '',
            no_update,
            {'candidate_ids': [cid], 'queue_size': 1, 'filter_hash': f'fetch:{cid}'},
            0,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"✗ Fetch failed: {str(e)}", no_update, no_update, '', no_update, no_update, no_update


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
         State('import-trigger', 'data')],
        background=True,
        running=[
            (Output('fetch-btn', 'disabled'), True, False),
        ],
        progress=[Output('fetch-status', 'children')],
        prevent_initial_call=True,
    )
    def fetch_candidate_callback(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, current_trigger):
        return _fetch_candidate_impl(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, current_trigger)
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
         State('import-trigger', 'data')],
        prevent_initial_call=True,
    )
    def fetch_candidate_callback(n_clicks, fetch_type, fetch_query, fetch_mode, current_trigger):
        return _fetch_candidate_impl(None, n_clicks, fetch_type, fetch_query, fetch_mode, current_trigger)


# Pipeline status chips (updated when candidate changes)
@app.callback(
    [Output('pipeline-status-chips', 'children'),
     Output('auto-run-pipeline-trigger', 'data', allow_duplicate=True)],
    [Input('queue-data', 'modified_timestamp'),
     Input('current-candidate-id', 'data')],
    State('pending-auto-run', 'data'),
    prevent_initial_call=True
)
def update_pipeline_status_chips(_queue_data_ts, candidate_id, pending_auto_run):
    """Show pipeline stage completion status for the current candidate, and cascade auto-run if pending."""
    chips = []
    auto_run_out = no_update
    triggered_ids = {
        item['prop_id'].split('.')[0]
        for item in (callback_context.triggered or [])
        if item.get('prop_id')
    }
    
    if candidate_id is None:
        return chips, auto_run_out

    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            payload = get_candidate_payload(conn, str(candidate_id)) or {}

        from malca.review.pipeline import detect_pipeline_status
        status = detect_pipeline_status(payload)
        if (
            'queue-data' in triggered_ids
            and pending_auto_run
            and str(pending_auto_run.get('candidate_id')) == str(candidate_id)
        ):
            if any(state in {'missing', 'partial'} for state in status.values()):
                auto_run_out = pending_auto_run

        color_map = {'complete': '#2d6a2d', 'partial': '#6a5c2d', 'missing': '#444'}
        for stage, state in status.items():
            chips.append(html.Span(
                f"{'●' if state == 'complete' else '○'} {stage.capitalize()}",
                style={
                    'padding': '1px 6px',
                    'borderRadius': '8px',
                    'backgroundColor': color_map.get(state, '#444'),
                    'color': '#e0e0e0' if state != 'missing' else '#666',
                    'fontSize': '10px',
                },
            ))
        return chips, auto_run_out
    except Exception:
        return chips, auto_run_out


def _run_pipeline_impl(set_progress, run_clicks, rerun_clicks, auto_trigger, queue_data, idx, current_trigger):
    import dash
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    queue_size = int((queue_data or {}).get('queue_size') or 0) if isinstance(queue_data, dict) else 0
    
    print(f"[run_pipeline_callback] Triggered by: {triggered_id}, auto_trigger: {auto_trigger}, queue_size: {queue_size}, idx: {idx}")

    if not run_clicks and not rerun_clicks and not auto_trigger:
        print("[run_pipeline_callback] No clicks and no auto_trigger, aborting.")
        return no_update, no_update, no_update
        
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
        from malca.review.pipeline import run_missing_stages
        
        def p(msg):
            if set_progress:
                try:
                    set_progress(msg[:300] if len(msg) > 300 else msg)
                except Exception:
                    pass
            else:
                print(f"[pipeline] {msg}")
            
        with closing(db_connect(Path(DB_PATH))) as conn:
            
            # Determine if this was an explicit deep fetch that should bypass caching
            force_stages = []
            if triggered_id == 'rerun-pipeline-btn':
                force_stages = ['stats', 'events', 'characterize', 'vetting', 'external_lcs']
            elif fetch_mode == 'full':
                force_stages = ['stats', 'events', 'characterize', 'vetting']
            elif fetch_mode in ('full_ext', 'full_ext_crts'):
                force_stages = ['stats', 'events', 'characterize', 'vetting', 'external_lcs']
                
            stages = run_missing_stages(conn, candidate_id, progress_callback=p, force_stages=force_stages)
            
            # If triggered by a full_ext fetch, ensure we run external LCs
            if fetch_mode == 'full_ext' and 'external_lcs' not in stages:
                p("Running external LCs...")
                from malca.vetting import fetch_external_lcs
                from malca.review.store import get_candidate_payload
                from malca.review.pipeline import update_candidate_payload
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

        refresh_idx = int(idx or 0) if idx is not None else 0
        if stages:
            _index_external_lc_paths.cache_clear()
            _index_external_lc_paths_from_root.cache_clear()
            return f"✓ Ran: {', '.join(stages)}", no_update, refresh_idx
        else:
            if triggered_id == 'rerun-pipeline-btn':
                return "No stages could be rerun (missing requirements)", no_update, no_update
            return "All stages already complete (or missing requirements)", no_update, no_update
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"✗ Pipeline failed: {str(e)}", no_update, no_update

if _background_callback_manager is not None:
    @app.callback(
        [Output('pipeline-run-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        [Input('run-pipeline-btn', 'n_clicks'),
         Input('rerun-pipeline-btn', 'n_clicks'),
         Input('auto-run-pipeline-trigger', 'data')],
        [State('queue-data', 'data'),
         State('current-index', 'data'),
         State('import-trigger', 'data')],
        background=True,
        running=[
            (Output('run-pipeline-btn', 'disabled'), True, False),
            (Output('rerun-pipeline-btn', 'disabled'), True, False),
        ],
        progress=[Output('pipeline-run-status', 'children')],
        prevent_initial_call='initial_duplicate'
    )
    def run_pipeline_callback(set_progress, n_clicks, rerun_clicks, auto_trigger, queue_data, idx, current_trigger):
        return _run_pipeline_impl(set_progress, n_clicks, rerun_clicks, auto_trigger, queue_data, idx, current_trigger)
else:
    @app.callback(
        [Output('pipeline-run-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        [Input('run-pipeline-btn', 'n_clicks'),
         Input('rerun-pipeline-btn', 'n_clicks'),
         Input('auto-run-pipeline-trigger', 'data')],
        [State('queue-data', 'data'),
         State('current-index', 'data'),
         State('import-trigger', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def run_pipeline_callback(n_clicks, rerun_clicks, auto_trigger, queue_data, idx, current_trigger):
        return _run_pipeline_impl(None, n_clicks, rerun_clicks, auto_trigger, queue_data, idx, current_trigger)


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
    return send_from_directory(PLOT_DIR, filename)


def main():
    """Main entry point."""
    global DB_PATH, PLOT_DIR

    parser = argparse.ArgumentParser(description="MALCA Dash Review App")
    parser.add_argument('--db', default=None, help="SQLite database path (default: standalone.db without --plot-dir, review.db with --plot-dir)")
    parser.add_argument('--plot-dir', help="Plot directory path (auto-detects ./plots if not specified)")
    parser.add_argument('--host', default='127.0.0.1', help="Host")
    parser.add_argument('--port', default=8050, type=int, help="Port")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--verbose-http', action='store_true',
                        help="Show Flask/Werkzeug per-request access logs")
    parser.add_argument('--merge-vetting', metavar='PATH',
                        help="Merge vetting results from a parquet file into the review DB and exit")
    args = parser.parse_args()

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

    # Choose DB: explicit --db overrides; otherwise standalone gets its own DB
    if args.db is not None:
        DB_PATH = str(_resolve_db_cli_path(args.db))
    elif PLOT_DIR is None:
        # Standalone mode: use a separate DB so pipeline candidates don't bleed in
        DB_PATH = str(_resolve_db_cli_path(str(DEFAULT_STANDALONE_DB_PATH)))
    else:
        DB_PATH = str(_resolve_db_cli_path(str(DEFAULT_DB_PATH)))

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

    print(f"Starting MALCA Review App...")
    print(f"  Database:  {DB_PATH}")
    print(f"  Plot dir:  {PLOT_DIR}")
    print(f"  Server:    http://{args.host}:{args.port}")
    print(f"\nKeyboard shortcuts:")
    print("  [D]ipper [M]icrolensing [F]lare [Y]so [U]nknown [I]nstrumental [O]ther | [1-4] Confidence | [.] Save | [Enter] Done | [Backspace] Back | [Esc] Sidebar | [Shift+R] Refresh | [?] Help")
    print("")

    # Auto-open browser
    url = f"http://{args.host}:{args.port}"
    Timer(0.1, lambda: webbrowser.open(url)).start()

    if not args.verbose_http:
        # Keep explicit pipeline/status prints, but hide the noisy per-request
        # development-server access lines so long-running actions are readable.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.server.logger.setLevel(logging.ERROR)

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
