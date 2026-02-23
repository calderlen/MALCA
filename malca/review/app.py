"""Dash-based keyboard-driven review app for MALCA candidates."""

import sys
import argparse
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
    recent_history,
    export_reviews,
    detect_run_directory_files,
    merge_vetting_results,
    get_distinct_values,
)
from malca.review.metadata import (
    extract_review_metadata_grouped,
    is_group_default_open,
)
from malca.review.keyboard import (
    handle_key_action, HELP_TEXT,
    CLASS_KEY_MAP,
)

CLASS_BADGE_TAGS = list(CLASS_KEY_MAP.values())
from malca.review.session import create_queue_data_dict
from malca.review.interactive_plot import build_interactive_lightcurve_figure
from malca.config.config_paths import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE
from malca.config.config_characterize import GAIA_CHUNK_SIZE

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="MALCA Review"
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
            min-height: 0;
            overflow: hidden;
            padding-right: 8px;
        }
        .left-info-scroll {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-right: 2px;
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
        .panel-splitter {
            position: relative;
            height: 12px;
            flex: 0 0 12px;
            margin: 0 12px;
            cursor: row-resize;
            user-select: none;
            touch-action: none;
        }
        .panel-splitter::before {
            content: '';
            position: absolute;
            left: 0;
            right: 0;
            top: 50%;
            height: 1px;
            transform: translateY(-50%);
            background: rgba(126, 150, 166, 0.45);
        }
        .panel-splitter::after {
            content: ':::';
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            padding: 0 7px;
            border-radius: 999px;
            color: #8db0c8;
            font-size: 10px;
            letter-spacing: 2px;
            background: rgba(8, 18, 25, 0.9);
            border: 1px solid rgba(86, 114, 132, 0.55);
            line-height: 1;
        }
        .panel-splitter:hover::after,
        .panel-splitter.dragging::after {
            color: #b5d4ea;
            border-color: rgba(133, 171, 196, 0.9);
            background: rgba(12, 26, 35, 0.96);
        }
        .status-splitter {
            height: 9px;
            flex: 0 0 9px;
            margin: 0;
        }
        .status-splitter::after {
            font-size: 9px;
            letter-spacing: 1.5px;
            padding: 0 5px;
        }
        .panel-splitter-vertical {
            width: 12px;
            flex: 0 0 12px;
            height: auto;
            margin: 0 2px;
            cursor: col-resize;
        }
        .panel-splitter-vertical::before {
            left: 50%;
            right: auto;
            top: 0;
            bottom: 0;
            width: 1px;
            height: auto;
            transform: translateX(-50%);
        }
        .panel-splitter-vertical::after {
            content: '::';
            letter-spacing: 1px;
            padding: 6px 3px;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: translate(-50%, -50%);
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
        .toolbar-slider-control {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            min-width: 140px;
            max-width: 200px;
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
            grid-template-columns: 1fr;
            gap: 6px;
        }
        .run-config-item {
            border: 1px solid rgba(78, 110, 132, 0.45);
            border-radius: 6px;
            background: rgba(7, 16, 22, 0.68);
            padding: 5px 7px;
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
            word-break: break-word;
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
        input, textarea {
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
        /* Dash dropdown specific */
        .dash-dropdown .Select-value-label,
        .dash-dropdown .Select-input > input {
            color: #e0e0e0 !important;
        }
        .dash-dropdown .Select-control {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
        }
        .dash-dropdown .Select-menu-outer {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
        }
        .dash-dropdown .Select-option {
            background-color: #10171d !important;
            color: #dce8f2 !important;
        }
        .dash-dropdown .Select-option:hover {
            background-color: #1d2d3a !important;
            color: #fff !important;
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
            overflow-y: auto;
            padding: 0 12px;
            border-radius: 8px;
        }
        .candidate-metadata {
            flex: 1;
            min-height: 100px;
            height: auto;
            max-height: none;
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

        /* Theme overrides: dark-only options */
        body[data-theme="dracula"] {
            background-color: #282a36 !important;
            color: #f8f8f2 !important;
        }
        body[data-theme="nord"] {
            background-color: #2e3440 !important;
            color: #d8dee9 !important;
        }
        body[data-theme="gruvbox"] {
            background-color: #282828 !important;
            color: #ebdbb2 !important;
        }
        body[data-theme="dracula"] .main-container,
        body[data-theme="nord"] .main-container,
        body[data-theme="gruvbox"] .main-container {
            background-color: inherit !important;
        }
        body[data-theme="dracula"] .sidebar,
        body[data-theme="dracula"] .header-bar,
        body[data-theme="dracula"] .metadata-sections,
        body[data-theme="dracula"] .control-bar,
        body[data-theme="dracula"] .review-form,
        body[data-theme="dracula"] .plot-status,
        body[data-theme="dracula"] .run-config-item,
        body[data-theme="dracula"] .plot-toolbar { background-color: #44475a !important; border-color: #6272a4 !important; color: #f8f8f2 !important; }
        body[data-theme="nord"] .sidebar,
        body[data-theme="nord"] .header-bar,
        body[data-theme="nord"] .metadata-sections,
        body[data-theme="nord"] .control-bar,
        body[data-theme="nord"] .review-form,
        body[data-theme="nord"] .plot-status,
        body[data-theme="nord"] .run-config-item,
        body[data-theme="nord"] .plot-toolbar { background-color: #3b4252 !important; border-color: #4c566a !important; color: #d8dee9 !important; }
        body[data-theme="gruvbox"] .sidebar,
        body[data-theme="gruvbox"] .header-bar,
        body[data-theme="gruvbox"] .metadata-sections,
        body[data-theme="gruvbox"] .control-bar,
        body[data-theme="gruvbox"] .review-form,
        body[data-theme="gruvbox"] .plot-status,
        body[data-theme="gruvbox"] .run-config-item,
        body[data-theme="gruvbox"] .plot-toolbar { background-color: #3c3836 !important; border-color: #504945 !important; color: #ebdbb2 !important; }
        body[data-theme="dracula"] .section-title,
        body[data-theme="dracula"] .help-link,
        body[data-theme="dracula"] .metadata-sections summary,
        body[data-theme="dracula"] #progress-text { color: #bd93f9 !important; }
        body[data-theme="nord"] .section-title,
        body[data-theme="nord"] .help-link,
        body[data-theme="nord"] .metadata-sections summary,
        body[data-theme="nord"] #progress-text { color: #88c0d0 !important; }
        body[data-theme="gruvbox"] .section-title,
        body[data-theme="gruvbox"] .help-link,
        body[data-theme="gruvbox"] .metadata-sections summary,
        body[data-theme="gruvbox"] #progress-text { color: #fabd2f !important; }
        body[data-theme="dracula"] .action-btn.primary { background-color: #bd93f9 !important; color: #282a36 !important; border-color: #bd93f9 !important; }
        body[data-theme="nord"] .action-btn.primary { background-color: #88c0d0 !important; color: #2e3440 !important; border-color: #88c0d0 !important; }
        body[data-theme="gruvbox"] .action-btn.primary { background-color: #fabd2f !important; color: #282828 !important; border-color: #fabd2f !important; }
        body[data-theme="dracula"] input, body[data-theme="dracula"] textarea, body[data-theme="dracula"] select,
        body[data-theme="dracula"] .dash-dropdown .Select-control,
        body[data-theme="dracula"] .dash-dropdown .Select-menu-outer { background-color: #44475a !important; color: #f8f8f2 !important; border-color: #6272a4 !important; }
        body[data-theme="nord"] input, body[data-theme="nord"] textarea, body[data-theme="nord"] select,
        body[data-theme="nord"] .dash-dropdown .Select-control,
        body[data-theme="nord"] .dash-dropdown .Select-menu-outer { background-color: #3b4252 !important; color: #eceff4 !important; border-color: #4c566a !important; }
        body[data-theme="gruvbox"] input, body[data-theme="gruvbox"] textarea, body[data-theme="gruvbox"] select,
        body[data-theme="gruvbox"] .dash-dropdown .Select-control,
        body[data-theme="gruvbox"] .dash-dropdown .Select-menu-outer { background-color: #3c3836 !important; color: #fbf1c7 !important; border-color: #504945 !important; }
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

PLOT_PRESETS = {
    'Clean': {
        'overlays': ['markers', 'residuals', 'filter_bad_cameras'],
        'camera_mode': 'all',
    },
    'Diagnostics': {
        'overlays': ['markers', 'residuals', 'filter_bad_cameras', 'diagnostics'],
        'camera_mode': 'all',
    },
    'Full': {
        'overlays': ['markers', 'residuals', 'filter_bad_cameras', 'diagnostics', 'confidence'],
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
            html.Span(label, style={'color': '#7fa3bc', 'flex-shrink': '0', 'font-size': '10px',
                                     'text-transform': 'uppercase', 'letter-spacing': '0.5px'}),
            html.Span(str(value), style={'color': '#e2edf6', 'text-align': 'right',
                                          'font-weight': '600'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'gap': '8px',
                  'padding': '2px 0', 'border-bottom': '1px solid #1a1a1a'})
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
    """Render run config info cards with warning summary."""
    status = "Loaded" if run_params else "Missing"
    cards = [
        html.Div([
            html.Div('Status', className='k'),
            html.Div(status, className='v'),
        ], className='run-config-item'),
        html.Div([
            html.Div('Path', className='k'),
            html.Div(str(run_params_path) if run_params_path else 'not found', className='v'),
        ], className='run-config-item'),
    ]
    for label, value in _run_config_rows(run_params or {}):
        cards.append(
            html.Div([
                html.Div(label, className='k'),
                html.Div(value, className='v'),
            ], className='run-config-item')
        )
    if warnings:
        cards.append(
            html.Div([
                html.Div('Warnings', className='k'),
                html.Div(' | '.join(warnings), className='v'),
            ], className='run-config-item')
        )
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
    "Crossmatch",
    "Stellar Parameters",
    "Photometry",
    "Galactic Coordinates",
    "Extinction & Environment",
    "YSO / Classification",
)

_METADATA_CATALOG_GROUPS = (
    "Stellar Parameters",
    "Photometry",
    "Galactic Coordinates",
    "Extinction & Environment",
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


def _render_vetting_banner(payload: dict | None) -> html.Div:
    """Render a vetting status panel with source cards above the metadata grid."""
    if not payload or 'vetting_likely_known' not in payload:
        return html.Div(
            "Not vetted",
            style={
                'padding': '6px 12px', 'margin': '4px 0', 'border-radius': '4px',
                'background': '#333', 'color': '#999', 'font-size': '0.85em',
                'text-align': 'center',
            },
        )

    known = payload.get('vetting_likely_known')

    # Status header
    if known:
        header_style = {
            'padding': '5px 10px', 'border-radius': '4px 4px 0 0',
            'background': '#4a1111', 'color': '#ff6b6b', 'font-weight': 'bold',
            'font-size': '0.82em', 'text-align': 'center',
            'border': '1px solid #ff6b6b', 'border-bottom': 'none',
        }
        header_text = "KNOWN OBJECT"
    else:
        header_style = {
            'padding': '5px 10px', 'border-radius': '4px 4px 0 0',
            'background': '#114a11', 'color': '#6bff6b', 'font-weight': 'bold',
            'font-size': '0.82em', 'text-align': 'center',
            'border': '1px solid #6bff6b', 'border-bottom': 'none',
        }
        header_text = "POTENTIALLY NEW"

    # Build source cards
    cell_style = {
        'padding': '3px 8px', 'border-radius': '3px', 'font-size': '0.78em',
        'background': '#1a2a1a' if not known else '#2a1a1a',
        'border': '1px solid #333',
        'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center',
        'gap': '8px', 'overflow': 'hidden',
    }
    label_style = {'color': '#888', 'font-size': '0.9em', 'flex-shrink': '0'}
    value_style = {'color': '#e0e0e0', 'font-weight': 'bold',
                   'text-align': 'right', 'word-break': 'break-word', 'white-space': 'normal'}
    hit_style = {**value_style, 'color': '#ff6b6b' if known else '#6bff6b'}

    cards = []

    # SIMBAD cell
    simbad_id = payload.get('simbad_otype') or payload.get('simbad_main_id')
    if simbad_id:
        refs = payload.get('simbad_nbref')
        ref_str = f" ({refs} refs)" if refs else ""
        cards.append(html.Div([
            html.Span("SIMBAD", style=label_style),
            html.Span(f"{simbad_id}{ref_str}", style=hit_style, title=str(payload.get('simbad_main_id', ''))),
        ], style=cell_style))

    # VSX cell
    vsx_cls = payload.get('vsx_class')
    if vsx_cls and str(vsx_cls).strip() and str(vsx_cls).strip().lower() not in ('nan', '<na>'):
        vsx_sep = payload.get('vsx_sep_arcsec')
        sep_str = f" ({vsx_sep:.1f}\")" if vsx_sep and not pd.isna(vsx_sep) else ""
        vsx_p = payload.get('vsx_period')
        p_str = f", P={vsx_p:.4f}d" if vsx_p and not pd.isna(vsx_p) else ""
        cards.append(html.Div([
            html.Span("VSX", style=label_style),
            html.Span(f"{vsx_cls}{p_str}{sep_str}", style=hit_style),
        ], style=cell_style))

    # Gaia variability cell
    gaia_cls = payload.get('gaia_var_class')
    if gaia_cls:
        score = payload.get('gaia_var_score')
        score_str = f" ({score:.2f})" if score and not pd.isna(score) else ""
        cards.append(html.Div([
            html.Span("Gaia DR3", style=label_style),
            html.Span(f"{gaia_cls}{score_str}", style=hit_style),
        ], style=cell_style))

    # Gaia EB period cell
    eb_period = payload.get('gaia_eb_period')
    if eb_period and not pd.isna(eb_period):
        cards.append(html.Div([
            html.Span("Gaia EB", style=label_style),
            html.Span(f"P={eb_period:.4f} d", style=hit_style),
        ], style=cell_style))

    # ASAS-SN cell
    asassn_type = payload.get('asassn_var_type')
    if asassn_type:
        period = payload.get('asassn_var_period')
        p_str = f" P={period:.4f}d" if period and not pd.isna(period) else ""
        cards.append(html.Div([
            html.Span("ASAS-SN", style=label_style),
            html.Span(f"{asassn_type}{p_str}", style=hit_style),
        ], style=cell_style))

    # ZTF cell
    ztf_type = payload.get('ztf_var_type')
    if ztf_type:
        ztf_p = payload.get('ztf_var_period')
        zp_str = f" P={ztf_p:.4f}d" if ztf_p and not pd.isna(ztf_p) else ""
        cards.append(html.Div([
            html.Span("ZTF", style=label_style),
            html.Span(f"{ztf_type}{zp_str}", style=hit_style),
        ], style=cell_style))

    # TNS cell
    tns_name = payload.get('tns_name')
    if tns_name:
        tns_type = payload.get('tns_type', '')
        cards.append(html.Div([
            html.Span("TNS", style=label_style),
            html.Span(f"{tns_name} ({tns_type})" if tns_type else tns_name, style=hit_style),
        ], style=cell_style))

    # ALeRCE cell
    alerce_cls = payload.get('alerce_lc_class')
    if alerce_cls:
        prob = payload.get('alerce_lc_prob')
        prob_str = f" ({prob:.0%})" if prob and not pd.isna(prob) else ""
        cards.append(html.Div([
            html.Span("ALeRCE", style=label_style),
            html.Span(f"{alerce_cls}{prob_str}", style=hit_style),
        ], style=cell_style))

    # X-ray cell
    xray = payload.get('xray_det')
    if xray:
        flux = payload.get('xray_flux')
        flux_str = f" {flux:.1e}" if flux and not pd.isna(flux) else ""
        cards.append(html.Div([
            html.Span("X-ray", style=label_style),
            html.Span(f"Detected{flux_str}", style=hit_style),
        ], style=cell_style))

    # SFR cell
    sfr_name = payload.get('sfr_name')
    if sfr_name and str(sfr_name).strip() and str(sfr_name).strip().lower() not in ('nan', '<na>'):
        sfr_sep = payload.get('sfr_sep_arcmin')
        sep_str = f" ({sfr_sep:.1f}')" if sfr_sep and not pd.isna(sfr_sep) else ""
        cards.append(html.Div([
            html.Span("SFR", style=label_style),
            html.Span(f"{sfr_name}{sep_str}", style=hit_style),
        ], style=cell_style))

    # Cluster cell
    cluster_name = payload.get('cluster_name')
    if cluster_name and str(cluster_name).strip() and str(cluster_name).strip().lower() not in ('nan', '<na>'):
        cluster_dist = payload.get('cluster_dist_pc')
        d_str = f" ({cluster_dist:.0f} pc)" if cluster_dist and not pd.isna(cluster_dist) else ""
        cards.append(html.Div([
            html.Span("Cluster", style=label_style),
            html.Span(f"{cluster_name}{d_str}", style=hit_style),
        ], style=cell_style))

    # BANYAN cell
    banyan_assoc = payload.get('banyan_best_assoc')
    banyan_fp = payload.get('banyan_field_prob')
    if banyan_assoc and str(banyan_assoc).strip() and str(banyan_assoc).strip().lower() not in ('nan', '<na>', 'field'):
        fp_str = f" (P_field={banyan_fp:.0%})" if banyan_fp and not pd.isna(banyan_fp) else ""
        cards.append(html.Div([
            html.Span("BANYAN", style=label_style),
            html.Span(f"{banyan_assoc}{fp_str}", style=hit_style),
        ], style=cell_style))

    # YSO class cell
    yso_cls = payload.get('yso_class')
    if yso_cls and str(yso_cls).strip() and str(yso_cls).strip().lower() not in ('nan', '<na>'):
        cards.append(html.Div([
            html.Span("YSO", style=label_style),
            html.Span(str(yso_cls), style=hit_style),
        ], style=cell_style))

    # IPHAS H-alpha excess cell
    ha_excess = payload.get('iphas_ha_excess')
    if ha_excess and not pd.isna(ha_excess) and float(ha_excess) > 0:
        cards.append(html.Div([
            html.Span("IPHAS Hα", style=label_style),
            html.Span(f"excess={float(ha_excess):.2f}", style=hit_style),
        ], style=cell_style))

    # Gaia epoch cell (non-hit, informational)
    epoch_n = payload.get('gaia_epoch_n_obs')
    if epoch_n and int(epoch_n) > 0:
        g_range = payload.get('gaia_epoch_g_range')
        r_str = f", dG={g_range:.2f}" if g_range and not pd.isna(g_range) else ""
        cards.append(html.Div([
            html.Span("Gaia epoch", style=label_style),
            html.Span(f"{int(epoch_n)} obs{r_str}", style=value_style),
        ], style=cell_style))

    if not cards and not known:
        # No matches at all — emphasize "new"
        cards.append(html.Div([
            html.Span("No catalog matches found", style={**value_style, 'color': '#6bff6b'}),
        ], style={**cell_style, 'grid-column': '1 / -1'}))

    grid_style = {
        'display': 'flex', 'flex-direction': 'column',
        'gap': '2px', 'padding': '5px 6px 6px',
        'background': '#1a1a1a',
        'border': '1px solid #333', 'border-top': 'none',
        'border-radius': '0 0 4px 4px',
    }

    return html.Div([
        html.Div(header_text, style=header_style),
        html.Div(cards, style=grid_style),
    ], style={'margin': '4px 0'})


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
    fig = go.Figure()
    if df_neowise is None or df_neowise.empty:
        fig.update_layout(height=220, margin=dict(l=36, r=10, t=30, b=28), title="NEOWISE")
        return fig

    time_col = "mjd" if "mjd" in df_neowise.columns else ("MJD" if "MJD" in df_neowise.columns else None)
    if time_col is None:
        fig.update_layout(height=220, margin=dict(l=36, r=10, t=30, b=28), title="NEOWISE (missing MJD column)")
        return fig

    x = pd.to_numeric(df_neowise[time_col], errors="coerce")
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

    fig.update_layout(
        height=240,
        margin=dict(l=42, r=10, t=34, b=32),
        title="NEOWISE Light Curve",
        legend=dict(orientation="h", x=0.0, y=1.1),
        template="plotly_dark",
    )
    fig.update_xaxes(title="MJD")
    fig.update_yaxes(title="mag", autorange="reversed")
    if added == 0:
        fig.add_annotation(text="No finite W1/W2 points", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
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


def _render_external_followup(payload: dict, candidate_id: str) -> list:
    card_style = {
        'border': '1px solid #2a2a2a',
        'borderRadius': '6px',
        'padding': '8px 10px',
        'background': '#0d0d0d',
    }
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
        html.Div(f"Sources: {spectrum_sources or 'none'}", style={'fontSize': '11px', 'color': '#9fb6cb'}),
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
                    html.A(link, href=link, target='_blank', rel='noopener noreferrer', style={'display': 'block', 'fontSize': '10px'})
                    for link in spectrum_links
                ]),
            ])
        )

    spectra_card = html.Div([
        html.Div('Spectra', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *spectra_children,
    ], style=card_style)

    # ATLAS summary
    atlas_card = html.Div([
        html.Div('ATLAS', style={'fontWeight': '600', 'marginBottom': '4px'}),
        html.Div(f"Photometry: {'yes' if _coerce_bool(payload.get('atlas_has_phot')) else 'no'}", style={'fontSize': '11px'}),
        html.Div(f"cyan n/range: {payload.get('atlas_n_det_cyan', 'n/a')} / {payload.get('atlas_cyan_range', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"orange n/range: {payload.get('atlas_n_det_orange', 'n/a')} / {payload.get('atlas_orange_range', 'n/a')}", style={'fontSize': '11px'}),
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
                figure=_build_neowise_figure(neowise_rows),
                config={'displayModeBar': False},
                style={'height': '250px'},
            )
        except Exception:
            neowise_plot = html.Div(f"Could not load NEOWISE parquet: {neowise_path}", style={'fontSize': '10px', 'color': '#c77'})

    neowise_children = [
        html.Div(f"Epochs: {neowise_epochs}", style={'fontSize': '11px'}),
        html.Div(f"W1 range: {payload.get('neowise_w1_range', 'n/a')}", style={'fontSize': '11px'}),
        html.Div(f"W2 range: {payload.get('neowise_w2_range', 'n/a')}", style={'fontSize': '11px'}),
    ]
    if neowise_path:
        neowise_children.append(html.Div(f"File: {neowise_path.name}", style={'fontSize': '10px', 'color': '#9fb6cb'}))
    if neowise_plot is not None:
        neowise_children.append(neowise_plot)

    neowise_card = html.Div([
        html.Div('NEOWISE', style={'fontWeight': '600', 'marginBottom': '4px'}),
        *neowise_children,
    ], style=card_style)

    return [spectra_card, atlas_card, neowise_card]


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
                      style={'width': '48%', 'font-size': '11px', 'margin-right': '4%'}),
            dcc.Input(id=f'max-{cid}', type='number', placeholder='max',
                      style={'width': '48%', 'font-size': '11px'}),
        ], style={'display': 'flex', 'margin-bottom': '4px'}),
    ])


def _text_filter(col: str):
    """Text input for exact-match string filter."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col}:'),
        dcc.Input(id=f'filter-{cid}', type='text', placeholder='Any',
                  style=_inp_style),
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
        ('select', 'asassn_var_type'),
        ('select', 'gaia_var_class'),
        ('select', 'simbad_otype'),
        ('select', 'ztf_var_type'),
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
    ('Crossmatch', [
        ('text', 'vsx_class'),
        ('num', 'vsx_sep_arcsec'),
        ('text', 'sfr_name'),
        ('num', 'sfr_sep_arcmin'),
        ('text', 'cluster_name'),
        ('num', 'cluster_membership_prob'),
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
        dcc.Store(id='current-score', data=None),
        dcc.Store(id='event-class-store', data='unclassified'),
        dcc.Store(id='pending-prefix', data=''),  # kept for callback compatibility
        dcc.Store(id='needs-followup-store', data=False),
        dcc.Store(id='review-pass-store', data=1),
        dcc.Store(id='sidebar-state', data=False),  # collapsed by default
        dcc.Store(id='filter-params', data={}),
        dcc.Store(id='import-trigger', data=0),  # triggers queue refresh after import
        dcc.Store(id='activity-visible', data=False),  # collapsed by default
        dcc.Store(id='plot-render-request', data={'nonce': 1, 'ts': 0.0, 'state': {'idx': 0, 'plot_mode': 'native', 'overlay_values': ['baseline', 'markers', 'residuals', 'filter_bad_cameras', 'diagnostics'], 'selected_cameras': [], 'preset': 'Diagnostics', 'theme': 'dark', 'residual_height': 0.28, 'baseline_opacity': 0.5}}),
        dcc.Store(id='plot-render-applied', data=0),
        dcc.Store(id='plot-defaults-initialized', data=False),
        dcc.Store(id='queue-source-path', data=''),
        dcc.Store(id='run-config-json-store', data=''),
        dcc.Store(id='theme-mode-store', data='dark'),
        dcc.Store(id='review-session-start', data=None, storage_type='session'),
        dcc.Store(id='metadata-resize-init', data=0),
        dcc.Store(id='status-resize-init', data=0),
        dcc.Store(id='section-splitters-init', data=0),
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
                value=['yes'],
                style={'margin-bottom': '3px'}
            ),
            dcc.Checklist(
                id='filter-failed',
                options=[{'label': ' Require failed_any=False', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '6px'}
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
                style={'margin-bottom': '4px', 'font-size': '11px'}
            ),
            dcc.Checklist(
                id='sort-desc',
                options=[{'label': ' Descending', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'}
            ),

            html.Button('Refresh Queue [Shift+R]', id='refresh-btn', n_clicks=0,
                       style={'width': '100%', 'font-size': '11px'}, className='action-btn'),

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
            ),

            html.Hr(),

            html.Div('Import', className='section-title'),
            dcc.Input(id='import-path', placeholder='Candidates file path', type='text',
                     style=_inp_style),

            html.Label('Characterize on import:'),
            dcc.Checklist(
                id='characterize-on-import',
                options=[{'label': ' Enable', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'}
            ),

            dcc.Input(id='characterize-crossmatch', placeholder='Crossmatch CSV', type='text',
                     value=str(VSX_CROSSMATCH_PATH), style=_inp_style),
            dcc.Input(id='characterize-gaia-cache', placeholder='Gaia cache', type='text',
                     value=str(GAIA_CACHE_FILE), style=_inp_style),
            dcc.Input(id='characterize-chunk-size', placeholder='Chunk size', type='number',
                     value=GAIA_CHUNK_SIZE, style=_inp_style),
            dcc.Checklist(
                id='characterize-dust',
                options=[{'label': ' Enable dustmaps3d', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'}
            ),
            dcc.Input(id='characterize-starhorse', placeholder='StarHorse (tap or path)', type='text',
                     value='tap', style=_inp_style),

            html.Label('Vet on import:'),
            dcc.Checklist(
                id='vet-on-import',
                options=[{'label': ' Enable', 'value': 'yes'}],
                value=['yes'],
                style={'margin-bottom': '4px'}
            ),

            html.Button('Import', id='import-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%', 'font-size': '11px'}),

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
                       style={'width': '100%', 'font-size': '11px'}),

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
                    {'label': ' Dark', 'value': 'dark'},
                    {'label': ' Dracula', 'value': 'dracula'},
                    {'label': ' Nord', 'value': 'nord'},
                    {'label': ' Gruvbox', 'value': 'gruvbox'},
                ],
                value='dark',
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
                html.Span(id='pace-timer-display', style={'color': '#aaa', 'font-size': '11px', 'margin-left': '10px', 'font-family': 'monospace'}),
                html.Div([
                    html.Span(id='header-asas-sn-id', className='item'),
                    html.Span(id='header-path', className='item path'),
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
                                        {'label': ' Dip/Jump markers', 'value': 'markers'},
                                        {'label': ' Residual panel', 'value': 'residuals'},
                                        {'label': ' Phase-fold panel', 'value': 'phase'},
                                        {'label': ' Filter bad cameras', 'value': 'filter_bad_cameras'},
                                        {'label': ' Event diagnostics', 'value': 'diagnostics'},
                                        {'label': ' Confidence colors', 'value': 'confidence'},
                                    ],
                                    value=['markers', 'residuals', 'filter_bad_cameras', 'diagnostics'],
                                    inline=True,
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
                                        value=0.28,
                                        marks=None,
                                        tooltip={'placement': 'bottom', 'always_visible': False},
                                        updatemode='drag',
                                    ),
                                ], className='toolbar-slider-control'),
                                html.Button('Reset', id='plot-reset-btn', n_clicks=0, className='compact-btn'),
                                html.Button('Export', id='export-plot', n_clicks=0, className='compact-btn'),
                                html.Span(id='repro-badge', className='label-chip', style={'margin-left': '6px'}),
                            ], className='plot-toolbar'),
                        ], open=True),
                        html.Div(id='vetting-banner'),
                        html.Div(id='splitter-vetting', className='panel-splitter status-splitter',
                                 title='Drag to resize'),
                        html.Div(id='plot-stats-cards', className='plot-stats', style={'display': 'none'}),
                        html.Div([
                            html.Div(id='plot-status-panel', className='plot-status'),
                            html.Div(id='camera-filter-panel', className='camera-diag'),
                            html.Div(id='metadata-health-indicator'),
                        ], id='diagnostics-section'),
                        html.Div(id='splitter-diagnostics', className='panel-splitter status-splitter',
                                 title='Drag to resize'),
                        # Grouped candidate metadata sections (collapsible, includes stats)
                        html.Div([
                            html.Div([
                                html.Span('Candidate Panels', className='title'),
                                dcc.Checklist(
                                    id='round-sigfigs',
                                    options=[{'label': ' Round', 'value': 'yes'}],
                                    value=[],
                                    style={'display': 'inline-block', 'font-size': '11px', 'margin-right': '6px'},
                                ),
                                html.Button('Expand all', id='toggle-meta-all', n_clicks=0, className='compact-btn'),
                            ], className='meta-toolbar'),
                            html.Div(id='candidate-info-grid', className='metadata-sections candidate-metadata'),
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
                        html.Div(id='splitter-metadata', className='panel-splitter status-splitter',
                                 title='Drag to resize'),
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
                    className='panel-splitter panel-splitter-vertical',
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
                            style={'display': 'block', 'width': '100%', 'height': '100%'},
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

            # Recent activity
            html.Div([
                html.Div([
                    html.Span('Activity', style={'color': '#0af', 'font-size': '10px', 'cursor': 'pointer'}),
                ], id='activity-toggle', style={'padding': '2px 12px', 'background-color': '#0a0a0a', 'border-top': '1px solid #555', 'cursor': 'pointer', 'line-height': '1.2'}),
                html.Div(id='recent-activity', style={'display': 'none'}),  # Hidden by default
            ], style={'border-top': '1px solid #555'}),

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
    function(idx) {
        return Date.now();
    }
    """,
    Output('candidate-start-time', 'data'),
    Input('current-index', 'data')
)

app.clientside_callback(
    """
    function(n_intervals, startTime, toggle) {
        if (!toggle || !toggle.length || toggle.indexOf('yes') === -1 || !startTime) {
            return '';
        }
        var elapsed = (Date.now() - startTime) / 1000;
        var mins = Math.floor(elapsed / 60);
        var secs = Math.floor(elapsed % 60);
        return '⏱ ' + (mins > 0 ? mins + 'm ' : '') + secs + 's';
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
            if (saved && ['dark', 'dracula', 'nord', 'gruvbox'].includes(saved)) {
                return saved;
            }
        } catch (e) {
            // ignore storage read failures
        }
        return currentTheme || 'dark';
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
        var t = ['dark', 'dracula', 'nord', 'gruvbox'].includes(theme) ? theme : 'dark';
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
    function(preset, overlays, mode, opacity, resHeight) {
        try {
            var obj = {
                preset: preset,
                overlays: overlays || [],
                mode: mode,
                opacity: opacity,
                resHeight: resHeight
            };
            window.localStorage.setItem('malca.review.sidebar.plot.v1', JSON.stringify(obj));
        } catch (e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output('sidebar-plot-saved', 'data'),
    [Input('plot-preset', 'value'),
     Input('plot-overlays', 'value'),
     Input('plot-mode', 'value'),
     Input('baseline-opacity-slider', 'value'),
     Input('residual-height-slider', 'value')],
    prevent_initial_call=True,
)


# --- Sidebar plot prefs: load from localStorage on init ---
app.clientside_callback(
    """
    function(_tick, curPreset, curOverlays, curMode, curOpacity, curResHeight) {
        var nu = window.dash_clientside.no_update;
        try {
            var raw = window.localStorage.getItem('malca.review.sidebar.plot.v1');
            if (!raw) return [nu, nu, nu, nu, nu, false];
            var obj = JSON.parse(raw);
            var preset = (obj.preset && ['Clean', 'Diagnostics', 'Full'].includes(obj.preset))
                ? obj.preset : nu;
            var overlays = Array.isArray(obj.overlays) ? obj.overlays : nu;
            var mode = (obj.mode && ['native', 'png'].includes(obj.mode)) ? obj.mode : nu;
            var opacity = (typeof obj.opacity === 'number') ? obj.opacity : nu;
            var resHeight = (typeof obj.resHeight === 'number') ? obj.resHeight : nu;
            return [preset, overlays, mode, opacity, resHeight, true];
        } catch (e) {
            return [nu, nu, nu, nu, nu, false];
        }
    }
    """,
    [Output('plot-preset', 'value', allow_duplicate=True),
     Output('plot-overlays', 'value', allow_duplicate=True),
     Output('plot-mode', 'value', allow_duplicate=True),
     Output('baseline-opacity-slider', 'value', allow_duplicate=True),
     Output('residual-height-slider', 'value', allow_duplicate=True),
     Output('plot-defaults-initialized', 'data', allow_duplicate=True)],
    Input('keyboard-init', 'n_intervals'),
    [State('plot-preset', 'value'),
     State('plot-overlays', 'value'),
     State('plot-mode', 'value'),
     State('baseline-opacity-slider', 'value'),
     State('residual-height-slider', 'value')],
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
            if (!isFinite(numeric)) {
                numeric = defaultWidth;
            }
            if (numeric < minWidth) {
                numeric = minWidth;
            }
            if (numeric > maxWidth) {
                numeric = maxWidth;
            }
            return Math.round(numeric);
        };

        var applyWidth = function(value, persist) {
            var w = clampWidth(value);
            leftPanel.style.width = String(w) + 'px';
            leftPanel.style.flex = '0 0 ' + String(w) + 'px';
            if (persist) {
                try {
                    window.localStorage.setItem(storageKey, String(w));
                } catch (e) {
                    // ignore storage failures
                }
            }
            return w;
        };

        if (!window.__malcaMetadataSplitterAttached) {
            var drag = {
                active: false,
                startX: 0,
                startWidth: 0,
                pointerId: null,
            };

            var onPointerMove = function(e) {
                if (!drag.active) {
                    return;
                }
                var nextWidth = drag.startWidth + (e.clientX - drag.startX);
                applyWidth(nextWidth, false);
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
                applyWidth(leftPanel.getBoundingClientRect().width, true);
                if (e) {
                    e.preventDefault();
                }
            };

            splitter.addEventListener('pointerdown', function(e) {
                drag.active = true;
                drag.startX = e.clientX;
                drag.startWidth = leftPanel.getBoundingClientRect().width;
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
                applyWidth(leftPanel.getBoundingClientRect().width, false);
            });

            window.__malcaMetadataSplitterAttached = true;
        }

        var saved = null;
        try {
            saved = window.localStorage.getItem(storageKey);
        } catch (e) {
            saved = null;
        }
        var initialWidth = defaultWidth;
        if (saved !== null && saved !== '') {
            var parsed = parseInt(saved, 10);
            if (!isNaN(parsed)) {
                initialWidth = parsed;
            }
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


# Section splitters in the left info panel
app.clientside_callback(
    """
    function(_tick) {
        if (window.__malcaSectionSplittersAttached) {
            return window.dash_clientside.no_update;
        }

        var configs = [
            {splitterId: 'splitter-vetting', targetId: 'vetting-banner', storageKey: 'malca.review.vetting.height.v1', defaultHeight: null, minHeight: 20},
            {splitterId: 'splitter-diagnostics', targetId: 'diagnostics-section', storageKey: 'malca.review.diagnostics.height.v1', defaultHeight: null, minHeight: 20},
            {splitterId: 'splitter-metadata', targetId: 'candidate-info-grid', storageKey: 'malca.review.metadata.height.v1', defaultHeight: null, minHeight: 40},
        ];

        configs.forEach(function(cfg) {
            var splitter = document.getElementById(cfg.splitterId);
            var target = document.getElementById(cfg.targetId);
            if (!splitter || !target) return;

            var drag = {active: false, startY: 0, startHeight: 0, pointerId: null};

            var clamp = function(h) {
                return Math.max(cfg.minHeight, Math.round(h));
            };

            var apply = function(h, persist) {
                h = clamp(h);
                target.style.maxHeight = h + 'px';
                target.style.overflow = 'auto';
                if (persist) {
                    try { window.localStorage.setItem(cfg.storageKey, String(h)); } catch(e) {}
                }
            };

            var onMove = function(e) {
                if (!drag.active) return;
                var next = drag.startHeight + (e.clientY - drag.startY);
                apply(next, false);
                e.preventDefault();
            };

            var onUp = function(e) {
                if (!drag.active) return;
                drag.active = false;
                splitter.classList.remove('dragging');
                window.removeEventListener('pointermove', onMove);
                window.removeEventListener('pointerup', onUp);
                window.removeEventListener('pointercancel', onUp);
                if (drag.pointerId !== null && splitter.releasePointerCapture) {
                    try { splitter.releasePointerCapture(drag.pointerId); } catch(e) {}
                }
                drag.pointerId = null;
                apply(target.getBoundingClientRect().height, true);
                if (e) e.preventDefault();
            };

            splitter.addEventListener('pointerdown', function(e) {
                drag.active = true;
                drag.startY = e.clientY;
                drag.startHeight = target.getBoundingClientRect().height;
                drag.pointerId = (typeof e.pointerId === 'number') ? e.pointerId : null;
                splitter.classList.add('dragging');
                if (drag.pointerId !== null && splitter.setPointerCapture) {
                    try { splitter.setPointerCapture(drag.pointerId); } catch(e) {}
                }
                window.addEventListener('pointermove', onMove);
                window.addEventListener('pointerup', onUp);
                window.addEventListener('pointercancel', onUp);
                e.preventDefault();
            });

            // Restore saved height
            if (cfg.defaultHeight === null) {
                // Don't constrain by default — only after user drags
                try {
                    var saved = window.localStorage.getItem(cfg.storageKey);
                    if (saved !== null && saved !== '') {
                        var parsed = parseInt(saved, 10);
                        if (!isNaN(parsed)) apply(parsed, false);
                    }
                } catch(e) {}
            } else {
                var initH = cfg.defaultHeight;
                try {
                    var saved = window.localStorage.getItem(cfg.storageKey);
                    if (saved !== null && saved !== '') {
                        var parsed = parseInt(saved, 10);
                        if (!isNaN(parsed)) initH = parsed;
                    }
                } catch(e) {}
                apply(initH, false);
            }
        });

        window.__malcaSectionSplittersAttached = true;
        return window.dash_clientside.no_update;
    }
    """,
    Output('section-splitters-init', 'data'),
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

    # Check if 'T' key was pressed
    key = _keyboard_key(key_value)
    if 'keyboard-input' in trigger and key.lower() == 't':
        is_expanded = not is_expanded

    # Check if toggle button was clicked
    elif 'sidebar-toggle' in trigger and n_clicks:
        is_expanded = not is_expanded
    else:
        return no_update, no_update, no_update

    sidebar_class = 'sidebar expanded' if is_expanded else 'sidebar'
    toggle_class = 'sidebar-toggle sidebar-expanded' if is_expanded else 'sidebar-toggle'
    return sidebar_class, toggle_class, is_expanded


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
        return queue_data



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
     State('queue-data', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('pending-prefix', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def handle_keyboard(key_value, current_idx, queue_data, current_score,
                    event_class, pending_prefix, needs_followup, notes):
    """Handle keyboard input."""
    NO = (no_update,) * 7  # shorthand for all-no_update

    key = _keyboard_key(key_value)
    if not key or not queue_data:
        return NO

    # Skip keys handled by other callbacks / keydown listener
    if key in ['?']:
        return NO

    queue_size = queue_data['queue_size']
    if queue_size == 0:
        return no_update, "Queue is empty", *([no_update] * 5)

    candidate_id = (queue_data['candidate_ids'][current_idx]
                    if current_idx < queue_size else None)

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


# Update plot and unified candidate info
@app.callback(
    Output('plot-render-request', 'data'),
    [Input('current-index', 'data'),
     Input('plot-mode', 'value'),
     Input('plot-overlays', 'value'),
     Input('camera-checklist', 'value'),
     Input('plot-preset', 'value'),
     Input('residual-height-slider', 'value'),
     Input('theme-mode-store', 'data'),
     Input('queue-data', 'data'),
     Input('baseline-opacity-slider', 'value'),
     Input('round-sigfigs', 'value')],
    State('plot-render-request', 'data'),
    prevent_initial_call=True,
)
def queue_plot_render_request(idx, plot_mode, overlay_values, selected_cameras, preset, residual_height, theme_mode, _queue_data, baseline_opacity, round_sigfigs, existing_request):
    """Debounced render request queue for native plot UX."""
    req = existing_request or {'nonce': 0, 'ts': 0.0}
    return {
        'nonce': int(req.get('nonce', 0)) + 1,
        'ts': float(time.time()),
        'state': {
            'idx': idx,
            'plot_mode': plot_mode,
            'overlay_values': list(overlay_values or []),
            'selected_cameras': list(selected_cameras or []),
            'preset': preset,
            'residual_height': float(residual_height or 0.28),
            'theme': theme_mode or 'dark',
            'baseline_opacity': float(baseline_opacity if baseline_opacity is not None else 0.5),
            'round_sigfigs': bool(round_sigfigs and 'yes' in round_sigfigs),
        },
    }


@app.callback(
    [Output('plot-preset', 'value'),
     Output('plot-overlays', 'value', allow_duplicate=True),
     Output('plot-defaults-initialized', 'data')],
    Input('queue-data', 'data'),
    State('plot-defaults-initialized', 'data'),
    prevent_initial_call=True,
)
def initialize_plot_defaults_from_run_params(queue_data, initialized):
    """Initialize native plot defaults from run_params once per session."""
    if initialized:
        raise dash.exceptions.PreventUpdate
    if not queue_data:
        raise dash.exceptions.PreventUpdate

    run_params = _load_run_params_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
    preset, overlays = _derive_defaults_from_run_params(run_params)
    return preset, overlays, True


@app.callback(
    [Output('plot-overlays', 'value'),
     Output('camera-checklist', 'value', allow_duplicate=True)],
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
        return new_overlays, new_cams
    if trig == 'cams-all-btn':
        return overlays, list(cams)
    if trig == 'cams-clear-btn':
        return overlays, []
    if trig == 'cams-invert-btn':
        inv = [c for c in cams if c not in set(selected)]
        return overlays, inv
    return no_update, no_update


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
     State('queue-data', 'data')],
    prevent_initial_call=False,
)
def update_display(render_request, applied_nonce, queue_data):
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
    theme_mode = str(state.get('theme', 'dark') or 'dark')
    residual_height = float(state.get('residual_height', 0.28) or 0.28)
    baseline_opacity = float(state.get('baseline_opacity', 0.5) if state.get('baseline_opacity') is not None else 0.5)
    round_sigfigs = bool(state.get('round_sigfigs', False))

    empty_fig = {
        'data': [],
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'margin': {'l': 40, 'r': 20, 't': 40, 'b': 30},
        },
    }

    if not queue_data or queue_data['queue_size'] == 0:
        return '', 'No candidates in queue', _render_metadata_health(None, context_msg='Queue is empty.'), _render_vetting_banner(None), '[0/0]', empty_fig, {'display': 'block', 'width': '100%', 'height': '100%'}, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'No candidates in queue.', []), _render_camera_diag_panel({}, []), _render_run_config_panel(None, None, ['Queue is empty']), _render_repro_badge(None, ['Queue is empty']), '', nonce

    queue_size = queue_data['queue_size']
    if idx < 0 or idx >= queue_size:
        return '', 'Invalid index', _render_metadata_health(None, context_msg='Invalid queue index.'), _render_vetting_banner(None), f'[{idx}/{queue_size}]', empty_fig, {'display': 'block', 'width': '100%', 'height': '100%'}, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'Invalid queue index.', []), _render_camera_diag_panel({}, []), _render_run_config_panel(None, None, ['Invalid queue index']), _render_repro_badge(None, ['Invalid queue index']), '', nonce

    candidate_id = queue_data['candidate_ids'][idx]
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
    vetting_banner = _render_vetting_banner(payload)
    label_color = '#888'
    value_color = '#e0e0e0'
    grid_items = []
    for group_name, items in grouped:
        field_divs = [
            html.Div([
                html.Span(label, style={'color': label_color, 'flex-shrink': '0'}),
                html.Span(str(value), style={'color': value_color, 'text-align': 'right',
                                              'word-break': 'break-word', 'white-space': 'normal'}),
            ], style={'display': 'flex', 'justify-content': 'space-between', 'gap': '8px',
                      'padding': '2px 0', 'border-bottom': '1px solid #1a1a1a'})
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
    uirevision_key = f"{candidate_id}|{','.join(sorted(str(c) for c in selected_cameras))}|{theme_mode}|{residual_height:.3f}|{baseline_opacity:.2f}"
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
            show_diagnostics='diagnostics' in overlays,
            confidence_colors='confidence' in overlays,
            run_params=run_params or {},
            uirevision_key=uirevision_key,
            theme=theme_mode,
            residual_fraction=residual_height,
            baseline_opacity=baseline_opacity,
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
     Input('current-index', 'data'),
     Input('queue-data', 'data')],
    prevent_initial_call=False,
)
def update_external_followup_panel(is_open, idx, queue_data):
    """Lazy-load external follow-up artifacts only when panel is open."""
    if not is_open:
        return html.Div(
            "Expand this section to load external spectra/light-curve artifacts.",
            style={'font-size': '11px', 'color': '#8a99a8'}
        )

    if not queue_data or not queue_data.get('candidate_ids'):
        return html.Div("No candidates loaded.", style={'font-size': '11px', 'color': '#c77'})

    candidate_ids = queue_data.get('candidate_ids', [])
    if idx is None or idx < 0 or idx >= len(candidate_ids):
        return html.Div("Invalid candidate index.", style={'font-size': '11px', 'color': '#c77'})

    candidate_id = str(candidate_ids[idx])
    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, candidate_id) or {}

    return _render_external_followup(payload, candidate_id)


@app.callback(
    [Output('header-asas-sn-id', 'children'),
     Output('header-path', 'children'),
     Output('header-gaia-id', 'children')],
    [Input('current-index', 'data'),
     Input('queue-data', 'data')],
    prevent_initial_call=False,
)
def update_header_key_info(idx, queue_data):
    """Render key candidate identifiers in the top header."""
    if not queue_data or not queue_data.get('candidate_ids'):
        return 'ASAS-SN ID: -', 'Path: -', 'Gaia ID: -'

    candidate_ids = queue_data.get('candidate_ids', [])
    if idx is None or idx < 0 or idx >= len(candidate_ids):
        return 'ASAS-SN ID: -', 'Path: -', 'Gaia ID: -'

    candidate_id = candidate_ids[idx]
    with closing(db_connect(Path(DB_PATH))) as conn:
        payload = get_candidate_payload(conn, candidate_id) or {}
    asas_sn_id = payload.get('asas_sn_id')
    gaia_id = payload.get('gaia_id')
    lc_path = payload.get('path')

    asas_text = f"ASAS-SN ID: {asas_sn_id}" if asas_sn_id else f"ASAS-SN ID: {candidate_id}"
    path_text = f"Path: {lc_path}" if lc_path else 'Path: -'
    gaia_fmt = _format_large_integer_like_display(gaia_id)
    gaia_text = f"Gaia ID: {gaia_fmt}" if gaia_fmt else 'Gaia ID: -'
    return asas_text, path_text, gaia_text


@app.callback(
    [Output('plot-export-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    Input('export-plot', 'n_clicks'),
    [State('interactive-plot', 'figure'),
     State('plot-mode', 'value'),
     State('plot-image', 'src'),
     State('current-index', 'data')],
    prevent_initial_call=True,
)
def export_active_plot(n_clicks, figure, plot_mode, plot_src, idx):
    """Export the currently shown plot.

    Native mode exports PDF, PNG mode exports the currently displayed PNG file.
    """
    if not n_clicks:
        return no_update, no_update

    ordinal = int(idx) + 1 if idx is not None else 0

    if plot_mode == 'native':
        if not figure:
            return no_update, 'No native plot is available to export.'
        fname = f"malca_plot_{ordinal}.pdf"
        try:
            export_fig = go.Figure(figure)
            export_fig.update_layout(
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='white',
                font={'color': '#111111', 'family': 'Monaco, Courier New, monospace', 'size': 11},
                title_font={'color': '#111111'},
                legend={
                    'bgcolor': 'rgba(255,255,255,0.95)',
                    'bordercolor': 'rgba(40,40,40,0.25)',
                    'borderwidth': 1,
                    'font': {'color': '#111111', 'size': 10},
                },
            )
            export_fig.update_xaxes(
                color='#111111',
                title_font={'color': '#111111'},
                tickfont={'color': '#111111'},
                showgrid=True,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
            )
            export_fig.update_yaxes(
                color='#111111',
                title_font={'color': '#111111'},
                tickfont={'color': '#111111'},
                showgrid=True,
                gridcolor='rgba(0,0,0,0.15)',
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
    Input('current-index', 'data'),
    State('queue-data', 'data'),
    prevent_initial_call=False
)
def load_review_form(idx, queue_data):
    """Load existing review for current candidate into stores."""
    if not queue_data or queue_data['queue_size'] == 0:
        return 'unclassified', False, 1, '', None

    candidate_id = queue_data['candidate_ids'][idx]
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, candidate_id)

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
     [State('current-index', 'data'),
      State('queue-data', 'data'),
      State('event-class-store', 'data'),
      State('needs-followup-store', 'data'),
      State('notes', 'value')],
     prevent_initial_call=True
)
def handle_score_clicks(*args):
    """Handle score button clicks."""
    idx, queue_data, event_class, needs_followup, notes = args[-5:]

    if not queue_data or not queue_data.get('candidate_ids'):
        return no_update, "Queue is empty", no_update

    if idx >= len(queue_data['candidate_ids']):
        return no_update, "Invalid candidate index", no_update

    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    score = int(button_id.split('-')[1])

    candidate_id = queue_data['candidate_ids'][idx]
    new_pass, _ = _do_save(
        candidate_id, score, event_class, needs_followup, notes, 'button',
    )

    return score, f"✓ Confidence: {score}", new_pass


# Save button
@app.callback(
    [Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('save-btn', 'n_clicks'),
    [State('current-index', 'data'),
     State('queue-data', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def save_review_callback(n_clicks, idx, queue_data, score,
                         event_class, needs_followup, notes):
    """Save review."""
    if not n_clicks or not queue_data or not queue_data.get('candidate_ids'):
        return no_update, no_update

    if idx >= len(queue_data['candidate_ids']):
        return "Invalid candidate index", no_update

    candidate_id = queue_data['candidate_ids'][idx]
    new_pass, _ = _do_save(
        candidate_id, score, event_class, needs_followup, notes, 'save_button',
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
     State('queue-data', 'data'),
     State('current-score', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def done_callback(n_clicks, idx, queue_data, score,
                  event_class, needs_followup, notes):
    """Save and go to next."""
    if not n_clicks or not queue_data or not queue_data.get('candidate_ids'):
        return no_update, no_update, no_update

    if idx >= len(queue_data['candidate_ids']):
        return no_update, "Invalid candidate index", no_update

    if score is None:
        return no_update, "⚠ Confidence required", no_update

    if not event_class or event_class == 'unclassified':
        return no_update, "⚠ Class required", no_update

    candidate_id = queue_data['candidate_ids'][idx]
    new_pass, _ = _do_save(
        candidate_id, score, event_class, needs_followup, notes, 'done_button',
        increment_pass=True,
    )

    queue_size = queue_data['queue_size']
    new_idx = min(idx + 1, queue_size - 1)

    return new_idx, "✓ Saved + Next →", new_pass


# --- Display callbacks for stores → visible indicators ---

_CLASS_ACTIVE_STYLE = {
    'border': '1px solid #0f0', 'color': '#0f0', 'background-color': '#003300',
}
_CLASS_INACTIVE_STYLE = {
    'border': '1px solid #444', 'color': '#888', 'background-color': 'transparent',
}


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
    [Output(f'class-badge-{tag}', 'style') for tag in CLASS_BADGE_TAGS],
    Input('event-class-store', 'data'),
    prevent_initial_call=False
)
def update_class_badges(active_class):
    """Highlight the active event class badge."""
    active = active_class or 'unclassified'
    return [_CLASS_ACTIVE_STYLE if tag == active else _CLASS_INACTIVE_STYLE
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
        return no_update, no_update

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
    Input('queue-data', 'data'),
    State('review-session-start', 'data'),
    prevent_initial_call=False,
)
def sync_review_session_start(queue_data, session_start):
    """Reset session-timer origin when the active queue changes."""
    queue_hash = None
    if isinstance(queue_data, dict):
        queue_hash = queue_data.get('filter_hash')

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
    Input('queue-data', 'data'),
    Input('current-index', 'data'),
    Input('review-pass-store', 'data'),
    State('review-session-start', 'data'),
    prevent_initial_call=False,
)
def update_review_progress_indicator(_tick, queue_data, _idx, _review_pass, session_start):
    """Render reviewed/total progress with session pace + elapsed timer."""
    _ = _tick, _idx, _review_pass

    with closing(db_connect(Path(DB_PATH))) as conn:
        reviewed, total = count_progress(conn)

    queue_size = 0
    if isinstance(queue_data, dict):
        queue_size = int(queue_data.get('queue_size') or 0)

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
    State('current-index', 'data'),
    State('queue-data', 'data'),
    prevent_initial_call=False
)
def update_status_indicator(needs_followup, score, idx, queue_data):
    """Show current effective status."""
    if not queue_data or queue_data['queue_size'] == 0:
        return "Status: —"
    candidate_id = queue_data['candidate_ids'][idx]
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, candidate_id)
    status = review.get('status', 'unreviewed')
    color = '#0f0' if status == 'reviewed' else '#f80' if status == 'needs_followup' else '#888'
    return html.Span(f"Status: {status}", style={'color': color})


# Recent activity
@app.callback(
    Output('recent-activity', 'children'),
    Input('current-index', 'data'),
    prevent_initial_call=False
)
def update_recent_activity(idx):
    """Update recent activity display."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        recent = recent_history(conn, limit=5)

    if recent.empty:
        return "No recent activity"

    lines = []
    for _, row in recent.iterrows():
        lines.append(f"• {row['candidate_id']} - {row['event_type']} ({row['created_at']})")

    return html.Div([html.Div(line, style={'margin': '2px 0'}) for line in lines])


# Toggle recent activity
@app.callback(
    [Output('recent-activity', 'style'),
     Output('activity-visible', 'data')],
    [Input('activity-toggle', 'n_clicks'),
     Input('keyboard-input', 'value')],
    State('activity-visible', 'data'),
    prevent_initial_call=True
)
def toggle_activity(n_clicks, key_value, is_visible):
    """Toggle recent activity visibility."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update

    trigger = ctx.triggered[0]['prop_id']

    # Check if 'A' key was pressed
    key = _keyboard_key(key_value)
    if 'keyboard-input' in trigger and key.lower() == 'a':
        is_visible = not is_visible
    # Check if toggle was clicked
    elif 'activity-toggle' in trigger and n_clicks:
        is_visible = not is_visible
    else:
        return no_update, no_update

    new_state = is_visible
    style = {
        'display': 'block',
        'padding': '5px 12px',
        'background-color': '#0a0a0a',
        'color': '#aaa',
        'font-size': '10px'
    } if new_state else {'display': 'none'}
    return style, new_state


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
     State('import-trigger', 'data')],
    prevent_initial_call=True
)
def import_candidates_callback(n_clicks, import_path, characterize_on,
                               crossmatch, gaia_cache, chunk_size, dust_on, starhorse, vet_on, current_trigger):
    """Import candidates from file."""
    if not n_clicks or not import_path:
        return no_update, no_update

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
    parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument('--plot-dir', help="Plot directory path (auto-detects ./plots if not specified)")
    parser.add_argument('--host', default='127.0.0.1', help="Host")
    parser.add_argument('--port', default=8050, type=int, help="Port")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--merge-vetting', metavar='PATH',
                        help="Merge vetting results from a parquet file into the review DB and exit")
    args = parser.parse_args()

    DB_PATH = str(_resolve_db_cli_path(args.db))

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

    # Auto-detect plot directory if not specified
    if args.plot_dir:
        PLOT_DIR = str(_resolve_plot_cli_path(args.plot_dir))
        if not Path(PLOT_DIR).exists() or not Path(PLOT_DIR).is_dir():
            print(f"❌ Error: plot directory does not exist: {PLOT_DIR}")
            print("Use an existing run bundle plots directory, for example:")
            print("  malca review --plot-dir output/runs/output_bundle_13_13.5/plots")
            sys.exit(1)
    else:
        # Try current directory first
        if Path('./plots').is_dir():
            PLOT_DIR = str(Path('./plots').resolve())
            print(f"📂 Auto-detected plot directory: {Path(PLOT_DIR).resolve()}")
        else:
            print("❌ Error: --plot-dir required (or run from a directory containing ./plots)")
            print("\nUsage:")
            print("  1. From run directory: cd output/runs/YOUR_RUN && malca review")
            print("  2. With explicit path: malca review --plot-dir /path/to/plots")
            sys.exit(1)

    print(f"🚀 Starting MALCA Dash Review App...")
    print(f"💾 Database: {DB_PATH}")
    print(f"🖼️  Plot directory: {PLOT_DIR}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"\n⌨️  Keyboard shortcuts:")
    print("  [D]ipper [M]icrolensing [F]lare [Y]so [U]nknown [I]nstrumental [O]ther | [1-4] Confidence | [.] Save | [Enter] Done | [Backspace] Back | [?] Help")
    print("")

    # Auto-open browser
    url = f"http://{args.host}:{args.port}"
    Timer(0.1, lambda: webbrowser.open(url)).start()

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
