"""Dash-based keyboard-driven review app for MALCA candidates."""

import sys
import argparse
import json
import time
from decimal import Decimal, InvalidOperation
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import webbrowser
from threading import Timer

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
from flask import send_from_directory
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from malca.review.store import (
    DEFAULT_DB_PATH,
    db_connect,
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
)
from malca.review.metadata import (
    extract_review_metadata_grouped,
    is_group_default_open,
)
from malca.review.keyboard import (
    handle_key_action, HELP_TEXT,
    CLASS_KEY_MAP, CLASS_PREFIX_KEY,
    PREFIX_KEYS,
)

CLASS_BADGE_TAGS = list(CLASS_KEY_MAP.values()) + ['not_real']
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
    <style>
        body {
            background-color: #000;
            color: #e0e0e0;
            font-family: 'Monaco', 'Courier New', monospace;
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
            flex: 0 0 420px;
            width: 420px;
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
        .plot-toolbar .compact-btn {
            background-color: #14212b;
            color: #c6d7e8;
            border: 1px solid rgba(92, 129, 154, 0.6);
            border-radius: 5px;
            padding: 2px 7px;
            font-size: 10px;
            cursor: pointer;
        }
        .plot-toolbar .compact-btn:hover {
            border-color: #7da8c4;
            background-color: #1a2b38;
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
        .residual-split-control {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-width: 220px;
            max-width: 320px;
        }
        .residual-split-label {
            color: #9fb6cb;
            font-size: 10px;
            white-space: nowrap;
            letter-spacing: 0.2px;
        }
        .residual-split-control .rc-slider {
            flex: 1;
            margin: 0 0 0 2px;
        }
        .residual-split-control .rc-slider-rail {
            background-color: #284256 !important;
        }
        .residual-split-control .rc-slider-track {
            background-color: #0af !important;
        }
        .residual-split-control .rc-slider-handle {
            border-color: #0af !important;
            background-color: #0b141d !important;
        }
        .residual-split-control .rc-slider-handle:hover,
        .residual-split-control .rc-slider-handle:focus,
        .residual-split-control .rc-slider-handle:active {
            border-color: #0af !important;
            box-shadow: 0 0 0 3px rgba(0, 170, 255, 0.2) !important;
        }
        .residual-split-control .rc-slider-dot-active {
            border-color: #0af !important;
        }
        .residual-split-control .rc-slider-mark-text-active {
            color: #7dd !important;
        }
        #residual-height-slider .rc-slider-rail {
            background-color: #284256 !important;
        }
        #residual-height-slider .rc-slider-track,
        #residual-height-slider .rc-slider-track-1 {
            background-color: #0af !important;
        }
        #residual-height-slider .rc-slider-handle,
        #residual-height-slider .rc-slider-handle-1,
        #residual-height-slider .rc-slider-handle-dragging,
        #residual-height-slider .rc-slider-handle-click-focused {
            border-color: #0af !important;
            background-color: #0b141d !important;
            box-shadow: 0 0 0 3px rgba(0, 170, 255, 0.2) !important;
            outline: none !important;
        }
        #residual-height-slider .rc-slider-dot-active {
            border-color: #0af !important;
        }
        #residual-height-slider .rc-slider-mark-text-active {
            color: #7dd !important;
        }
        #residual-height-slider .rc-slider-tooltip-inner {
            background-color: #0af !important;
            border: 1px solid #0af !important;
            color: #fff !important;
        }
        #residual-height-slider .rc-slider-tooltip-arrow {
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
            flex-wrap: wrap;
            gap: 8px;
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
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
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
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
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
            padding: 6px 8px;
            border-radius: 7px;
            border: 1px solid rgba(64, 96, 116, 0.45);
            background-color: rgba(8, 17, 24, 0.75);
            min-width: 95px;
        }
        .stat-card .label {
            color: #7fa3bc;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-card .value {
            color: #e2edf6;
            font-size: 13px;
            font-weight: 600;
            margin-top: 2px;
        }
        .run-config-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
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
            max-width: 56vw;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
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
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4px 8px;
            padding: 2px 6px 6px 6px;
            font-size: 11px;
        }

        /* ------------------------------------------------------------------ */
        /* Solarized light theme overrides (easy-on-eyes, not pure white)     */
        /* ------------------------------------------------------------------ */
        body[data-theme="solarized"] {
            background-color: #fdf6e3 !important;
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .main-container {
            background-color: #fdf6e3 !important;
        }
        body[data-theme="solarized"] .sidebar {
            background-color: #eee8d5 !important;
            border-right: 1px solid #93a1a1 !important;
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .sidebar .section-title,
        body[data-theme="solarized"] .sidebar details summary,
        body[data-theme="solarized"] .help-link {
            color: #268bd2 !important;
        }
        body[data-theme="solarized"] .sidebar details summary:hover,
        body[data-theme="solarized"] .help-link:hover {
            color: #2aa198 !important;
        }
        body[data-theme="solarized"] .sidebar hr {
            border-top: 1px solid #93a1a1 !important;
        }
        body[data-theme="solarized"] .sidebar label,
        body[data-theme="solarized"] .header-key-info .item,
        body[data-theme="solarized"] .plot-status,
        body[data-theme="solarized"] .camera-diag,
        body[data-theme="solarized"] .plot-stats,
        body[data-theme="solarized"] .run-config-panel,
        body[data-theme="solarized"] .metadata-sections,
        body[data-theme="solarized"] .control-bar,
        body[data-theme="solarized"] .review-form,
        body[data-theme="solarized"] .recent-activity {
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .header-bar {
            background-color: #eee8d5 !important;
            border-bottom: 1px solid #93a1a1 !important;
        }
        body[data-theme="solarized"] #progress-text {
            color: #268bd2 !important;
        }
        body[data-theme="solarized"] .notification {
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .plot-container,
        body[data-theme="solarized"] .plot-frame {
            background-color: #fdf6e3 !important;
        }
        body[data-theme="solarized"] .metadata-sections,
        body[data-theme="solarized"] .plot-status,
        body[data-theme="solarized"] .camera-diag,
        body[data-theme="solarized"] .plot-stats,
        body[data-theme="solarized"] .run-config-panel,
        body[data-theme="solarized"] .control-bar,
        body[data-theme="solarized"] .review-form,
        body[data-theme="solarized"] .recent-activity {
            background-color: #f5efdc !important;
            border-color: #93a1a1 !important;
        }
        body[data-theme="solarized"] .metadata-health {
            background-color: #f5efdc !important;
            border-color: #93a1a1 !important;
        }
        body[data-theme="solarized"] .metadata-health .chip {
            background-color: #eee8d5 !important;
            border-color: #93a1a1 !important;
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .metadata-health .detail {
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .metadata-health.metadata-health-base .chip {
            color: #b58900 !important;
            border-color: rgba(181, 137, 0, 0.55) !important;
            background-color: rgba(238, 232, 213, 0.9) !important;
        }
        body[data-theme="solarized"] .metadata-health.metadata-health-partial .chip {
            color: #268bd2 !important;
            border-color: rgba(38, 139, 210, 0.5) !important;
            background-color: rgba(238, 232, 213, 0.92) !important;
        }
        body[data-theme="solarized"] .metadata-health.metadata-health-enriched .chip {
            color: #2aa198 !important;
            border-color: rgba(42, 161, 152, 0.5) !important;
            background-color: rgba(238, 232, 213, 0.92) !important;
        }
        body[data-theme="solarized"] .metadata-sections summary {
            color: #268bd2 !important;
        }
        body[data-theme="solarized"] .metadata-sections summary:hover {
            color: #2aa198 !important;
        }
        body[data-theme="solarized"] .action-btn,
        body[data-theme="solarized"] .badge-btn,
        body[data-theme="solarized"] .score-btn,
        body[data-theme="solarized"] .compact-btn,
        body[data-theme="solarized"] .label-chip,
        body[data-theme="solarized"] .status-chip {
            background-color: #eee8d5 !important;
            color: #586e75 !important;
            border-color: #93a1a1 !important;
        }
        body[data-theme="solarized"] .action-btn.primary {
            background-color: #268bd2 !important;
            color: #fdf6e3 !important;
            border-color: #268bd2 !important;
        }
        body[data-theme="solarized"] input,
        body[data-theme="solarized"] textarea,
        body[data-theme="solarized"] select {
            background-color: #fdf6e3 !important;
            color: #586e75 !important;
            border-color: #93a1a1 !important;
        }
        body[data-theme="solarized"] ::placeholder {
            color: #93a1a1 !important;
        }
        body[data-theme="solarized"] .dash-checklist label,
        body[data-theme="solarized"] .dash-radioitems label {
            color: #586e75 !important;
        }
        body[data-theme="solarized"] .dash-dropdown,
        body[data-theme="solarized"] .dash-dropdown .Select-control,
        body[data-theme="solarized"] .dash-dropdown .Select-menu-outer,
        body[data-theme="solarized"] .dash-dropdown .Select-input input,
        body[data-theme="solarized"] .dash-dropdown .Select-value-label {
            background-color: #fdf6e3 !important;
            color: #586e75 !important;
            border-color: #93a1a1 !important;
        }
        body[data-theme="solarized"] .residual-split-label {
            color: #657b83 !important;
        }
        body[data-theme="solarized"] .residual-split-control .rc-slider-rail {
            background-color: #d8ccb3 !important;
        }
        body[data-theme="solarized"] .residual-split-control .rc-slider-track {
            background-color: #268bd2 !important;
        }
        body[data-theme="solarized"] .residual-split-control .rc-slider-handle {
            border-color: #268bd2 !important;
            background-color: #fdf6e3 !important;
        }
        body[data-theme="solarized"] .residual-split-control .rc-slider-handle:hover,
        body[data-theme="solarized"] .residual-split-control .rc-slider-handle:focus,
        body[data-theme="solarized"] .residual-split-control .rc-slider-handle:active {
            border-color: #268bd2 !important;
            box-shadow: 0 0 0 3px rgba(38, 139, 210, 0.18) !important;
        }
        body[data-theme="solarized"] .residual-split-control .rc-slider-dot-active {
            border-color: #268bd2 !important;
        }
        body[data-theme="solarized"] .residual-split-control .rc-slider-mark-text-active {
            color: #268bd2 !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-rail {
            background-color: #d8ccb3 !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-track,
        body[data-theme="solarized"] #residual-height-slider .rc-slider-track-1 {
            background-color: #268bd2 !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-handle,
        body[data-theme="solarized"] #residual-height-slider .rc-slider-handle-1,
        body[data-theme="solarized"] #residual-height-slider .rc-slider-handle-dragging,
        body[data-theme="solarized"] #residual-height-slider .rc-slider-handle-click-focused {
            border-color: #268bd2 !important;
            background-color: #fdf6e3 !important;
            box-shadow: 0 0 0 3px rgba(38, 139, 210, 0.18) !important;
            outline: none !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-dot-active {
            border-color: #268bd2 !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-mark-text-active {
            color: #268bd2 !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-tooltip-inner {
            background-color: #268bd2 !important;
            border: 1px solid #268bd2 !important;
            color: #fdf6e3 !important;
        }
        body[data-theme="solarized"] #residual-height-slider .rc-slider-tooltip-arrow {
            border-top-color: #268bd2 !important;
            border-bottom-color: #268bd2 !important;
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

PLOT_PRESETS = {
    'Clean': {
        'overlays': ['markers', 'residuals', 'filter_bad_cameras'],
        'camera_mode': 'all',
    },
    'Diagnostics': {
        'overlays': ['baseline', 'markers', 'residuals', 'filter_bad_cameras', 'diagnostics'],
        'camera_mode': 'all',
    },
    'Full': {
        'overlays': ['baseline', 'markers', 'residuals', 'filter_bad_cameras', 'diagnostics', 'confidence'],
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
    """Render compact stats cards below the native plot."""
    cards = []
    for label, value in stat_rows:
        cards.append(
            html.Div([
                html.Div(label, className='label'),
                html.Div(value, className='value'),
            ], className='stat-card')
        )
    return cards


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
        'padding': '4px 8px', 'border-radius': '3px', 'font-size': '0.78em',
        'background': '#1a2a1a' if not known else '#2a1a1a',
        'border': '1px solid #333', 'text-align': 'center',
        'overflow': 'hidden', 'text-overflow': 'ellipsis',
    }
    label_style = {'color': '#888', 'font-size': '0.9em', 'display': 'block'}
    value_style = {'color': '#e0e0e0', 'font-weight': 'bold', 'display': 'block'}
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
        'display': 'grid', 'grid-template-columns': 'repeat(auto-fill, minmax(110px, 1fr))',
        'gap': '3px', 'padding': '5px 6px 6px',
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
        ('bool', 'failed_sparse'),
        ('bool', 'failed_multi_camera'),
        ('bool', 'failed_vsx'),
        ('bool', 'failed_evidence_strength'),
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
        dcc.Store(id='current-score', data=0),
        dcc.Store(id='event-class-store', data='unclassified'),
        dcc.Store(id='pending-prefix', data=''),
        dcc.Store(id='needs-followup-store', data=False),
        dcc.Store(id='review-pass-store', data=1),
        dcc.Store(id='sidebar-state', data=False),  # collapsed by default
        dcc.Store(id='filter-params', data={}),
        dcc.Store(id='import-trigger', data=0),  # triggers queue refresh after import
        dcc.Store(id='activity-visible', data=False),  # collapsed by default
        dcc.Store(id='plot-render-request', data={'nonce': 1, 'ts': 0.0, 'state': {'idx': 0, 'plot_mode': 'native', 'overlay_values': ['baseline', 'markers', 'residuals', 'filter_bad_cameras', 'diagnostics'], 'selected_cameras': [], 'preset': 'Diagnostics', 'theme': 'dark', 'residual_height': 0.28}}),
        dcc.Store(id='plot-render-applied', data=0),
        dcc.Store(id='plot-defaults-initialized', data=False),
        dcc.Store(id='run-config-json-store', data=''),
        dcc.Store(id='theme-mode-store', data='dark'),
        dcc.Store(id='metadata-resize-init', data=0),
        dcc.Store(id='status-resize-init', data=0),
        dcc.Download(id='plot-export-download'),
        dcc.Download(id='run-config-download'),
        dcc.Interval(id='keyboard-init', interval=200, n_intervals=0, max_intervals=1),

        # Sidebar toggle button
        html.Button('☰', id='sidebar-toggle', className='sidebar-toggle', title='Toggle sidebar [T]', n_clicks=0),

        # Collapsible sidebar
        html.Div([
            html.Div('Run Directory', className='section-title'),
            dcc.Input(id='run-dir-path', placeholder='Run directory path', type='text',
                     style=_inp_style),
            html.Button('Auto-Detect Files', id='auto-detect-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%', 'font-size': '11px', 'margin-bottom': '4px'}),

            html.Hr(),

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
                    + [{'label': 'Interest Score', 'value': 'interest_score'},
                       {'label': 'Review Pass', 'value': 'review_pass'},
                       {'label': 'Updated At', 'value': 'updated_at'}]
                ),
                value='candidate_id',
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

            html.Div('Theme', className='section-title'),
            dcc.RadioItems(
                id='theme-mode',
                options=[
                    {'label': ' Dark', 'value': 'dark'},
                    {'label': ' Solarized Light', 'value': 'solarized'},
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
                                    {'label': ' Baseline', 'value': 'baseline'},
                                    {'label': ' Dip/Jump markers', 'value': 'markers'},
                                    {'label': ' Residual panel', 'value': 'residuals'},
                                    {'label': ' Filter bad cameras', 'value': 'filter_bad_cameras'},
                                    {'label': ' Event diagnostics', 'value': 'diagnostics'},
                                    {'label': ' Confidence colors', 'value': 'confidence'},
                                ],
                                value=['baseline', 'markers', 'residuals', 'filter_bad_cameras', 'diagnostics'],
                                inline=True,
                            ),
                            html.Div([
                                html.Span('Residual size', className='residual-split-label'),
                                dcc.Slider(
                                    id='residual-height-slider',
                                    min=0.15,
                                    max=0.45,
                                    step=0.01,
                                    value=0.28,
                                    marks={0.18: '18%', 0.28: '28%', 0.38: '38%'},
                                    tooltip={'placement': 'bottom', 'always_visible': False},
                                    updatemode='drag',
                                ),
                            ], className='residual-split-control'),
                            html.Button('Reset', id='plot-reset-btn', n_clicks=0, className='compact-btn'),
                            html.Button('Export', id='export-plot', n_clicks=0, className='compact-btn'),
                            html.Span(id='repro-badge', className='label-chip', style={'margin-left': '6px'}),
                        ], className='plot-toolbar'),
                        html.Div(id='plot-status-panel', className='plot-status'),
                        html.Div(id='camera-filter-panel', className='camera-diag'),
                        html.Div(id='plot-stats-cards', className='plot-stats'),
                        html.Div(id='metadata-health-indicator'),
                        html.Div(id='vetting-banner'),
                        # Grouped candidate metadata sections (collapsible)
                        html.Div(id='candidate-info-grid', className='metadata-sections candidate-metadata'),
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
                    html.Span('Score: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                ] + [
                    html.Button(str(i), id=f'score-{i}', n_clicks=0, className='score-btn')
                    for i in range(6)
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '6px'}),

                # Event class row (clickable buttons, [C]+key prefix)
                html.Div([
                    html.Span('Class: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                    html.Span(id='prefix-indicator', style={'margin-right': '6px', 'font-size': '11px'}),
                ] + [
                    html.Button(
                        f'[{CLASS_PREFIX_KEY.upper()}] [{key.upper()}]: {tag.replace("_", " ")}',
                        id=f'class-badge-{tag}',
                        n_clicks=0,
                        className='badge-btn',
                    )
                    for key, tag in CLASS_KEY_MAP.items()
                ] + [
                    html.Button(
                        '[X]: not real',
                        id='class-badge-not_real',
                        n_clicks=0,
                        className='badge-btn',
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'margin-bottom': '6px'}),

                # Action row: Save, Done, Followup, Pass, Status, Notification
                html.Div([
                    html.Button('Save [S]', id='save-btn', n_clicks=0, className='action-btn'),
                    html.Button('Done [D]', id='done-btn', n_clicks=0, className='action-btn primary'),
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
                    placeholder='[M] Notes - press Esc to exit',
                    style={'width': '100%', 'font-size': '11px', 'height': '26px'},
                ),
            ], className='review-form'),

            # Recent activity
            html.Div([
                html.Div([
                    html.Span('[A] Activity', style={'color': '#0af', 'font-size': '10px', 'cursor': 'pointer'}),
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


app.layout = create_layout()


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

                // M/m: focus notes textarea (pure client-side).
                if (key === 'm' || key === 'M') {
                    e.preventDefault();
                    var notesEl = document.getElementById('notes');
                    if (notesEl) {
                        var ta = notesEl.querySelector('textarea') || notesEl;
                        ta.focus();
                    }
                    return;
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
            if (saved === 'dark' || saved === 'solarized') {
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
        var t = (theme === 'solarized') ? 'solarized' : 'dark';
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

# Build the State list for the callback decorator dynamically.
_queue_states = (
    [State('filter-unreviewed', 'value'),
     State('filter-failed', 'value')]
    + [State(cid, 'value') for cid, _ in _BOOL_MODE_STATES]
    + [State(cid, 'value') for cid, _ in _NUM_STATES]
    + [State(cid, 'value') for cid, _ in _TEXT_STATES]
    + [State('sort-col', 'value'),
       State('sort-desc', 'value')]
)


# Initialize queue
@app.callback(
    Output('queue-data', 'data'),
    [Input('refresh-btn', 'n_clicks'),
     Input('import-trigger', 'data')],
    _queue_states,
    prevent_initial_call=False
)
def load_queue(refresh_clicks, import_trigger, *state_values):
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

        filter_params['sort_col'] = next(it) or 'candidate_id'
        filter_params['sort_desc'] = 'yes' in (next(it) or [])

        queue_data = create_queue_data_dict(conn, filter_params)
        return queue_data


def _do_save(candidate_id, score, event_class, needs_followup, notes, event_type):
    """Shared save helper.  Auto-sets status and auto-increments review_pass."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, candidate_id)
        new_pass = max(1, review.get('review_pass', 0)) + 1
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


# Keyboard handler (prefix-state machine for class)
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
    if key.lower() in ['t', 'm', 'a', '?']:
        return NO

    queue_size = queue_data['queue_size']
    if queue_size == 0:
        return no_update, "Queue is empty", *([no_update] * 5)

    candidate_id = (queue_data['candidate_ids'][current_idx]
                    if current_idx < queue_size else None)

    # --- Prefix state machine ([C] -> class) ---
    if pending_prefix:
        # We're waiting for the second key after a leader press
        if key == 'Escape':
            return no_update, "Cancelled", no_update, no_update, no_update, no_update, ''

        if pending_prefix == CLASS_PREFIX_KEY:
            class_tag = CLASS_KEY_MAP.get(key.lower())
            if class_tag is not None:
                cur = event_class or 'unclassified'
                if cur == class_tag:
                    return no_update, "Class: unclassified", no_update, no_update, no_update, 'unclassified', ''
                return no_update, f"Class: {class_tag}", no_update, no_update, no_update, class_tag, ''
            return no_update, f"[{CLASS_PREFIX_KEY.upper()}] {key}: unknown class", no_update, no_update, no_update, no_update, ''

        # Unknown prefix (shouldn't happen) — cancel
        return no_update, "Cancelled", no_update, no_update, no_update, no_update, ''

    # --- Enter prefix mode when a leader key is pressed ---
    kl = key.lower()
    if kl in PREFIX_KEYS:
        label = kl.upper()
        return no_update, f"[{label}] ...", no_update, no_update, no_update, no_update, kl

    # --- Quick reject (X): toggle not_real class ---
    if key.lower() == 'x':
        cur = event_class or 'unclassified'
        if cur == 'not_real':
            return no_update, "Class: unclassified", no_update, no_update, no_update, 'unclassified', no_update
        return no_update, "Class: not_real", no_update, no_update, no_update, 'not_real', no_update

    # --- Followup toggle (F) ---
    if key.lower() == 'f':
        new_state = not bool(needs_followup)
        label = "ON" if new_state else "OFF"
        return no_update, f"Followup: {label}", no_update, new_state, no_update, no_update, no_update

    # --- Navigation / scoring / save via handle_key_action ---
    with closing(db_connect(Path(DB_PATH))) as conn:
        new_idx, notification, should_save = handle_key_action(
            key, current_idx, queue_size, conn, candidate_id
        )

    new_score = no_update
    new_pass = no_update
    if should_save and candidate_id:
        score = int(key) if key in '012345' else current_score
        pass_val, _ = _do_save(
            candidate_id, score, event_class, needs_followup, notes, 'keyboard',
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
     Input('queue-data', 'data')],
    State('plot-render-request', 'data'),
    prevent_initial_call=True,
)
def queue_plot_render_request(idx, plot_mode, overlay_values, selected_cameras, preset, residual_height, theme_mode, _queue_data, existing_request):
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

    grouped = extract_review_metadata_grouped(payload)
    metadata_health = _render_metadata_health(grouped)
    vetting_banner = _render_vetting_banner(payload)
    label_color = '#657b83' if theme_mode == 'solarized' else '#888'
    value_color = '#586e75' if theme_mode == 'solarized' else '#e0e0e0'
    grid_items = []
    for group_name, items in grouped:
        field_divs = [
            html.Div([
                html.Span(f"{label}: ", style={'color': label_color}),
                html.Span(str(value), style={'color': value_color}),
            ], style={'white-space': 'nowrap', 'overflow': 'hidden', 'text-overflow': 'ellipsis'})
            for label, value in items
        ]
        if is_group_default_open(group_name):
            grid_items.append(
                html.Details(
                    [html.Summary(f"{group_name} ({len(items)})"), html.Div(field_divs, className='meta-grid')],
                    open='open',
                )
            )
        else:
            grid_items.append(
                html.Details(
                    [html.Summary(f"{group_name} ({len(items)})"), html.Div(field_divs, className='meta-grid')]
                )
            )

    progress = f"[{idx + 1}/{queue_size}] Queue: {queue_size}"

    if plot_mode == 'png':
        run_params, run_params_status, run_params_msg = _load_run_params_meta_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
        run_params_path = _run_params_path_for_plot_dir(str(PLOT_DIR) if PLOT_DIR else None)
        mismatch_warnings = _run_config_mismatch_warnings(run_params if run_params else None, overlays)
        if run_params_status != 'loaded':
            mismatch_warnings.append(run_params_msg)
        panel = _render_run_config_panel(run_params if run_params else None, run_params_path, mismatch_warnings)
        return (
            plot_src,
            grid_items,
            metadata_health,
            vetting_banner,
            progress,
            no_update,
            {'display': 'none'},
            {'display': 'block', 'width': '100%', 'height': '100%'},
            no_update,
            [],
            _render_plot_status_panel('ok', 'PNG view enabled. Switch to Native for interactive hover and diagnostics.', mismatch_warnings),
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
    uirevision_key = f"{candidate_id}|{','.join(sorted(str(c) for c in selected_cameras))}|{theme_mode}|{residual_height:.3f}"
    try:
        native = build_interactive_lightcurve_figure(
            payload,
            plot_dir=plot_dir_path,
            selected_cameras=selected_cameras,
            filter_bad_cameras='filter_bad_cameras' in overlays,
            show_baseline='baseline' in overlays,
            show_event_markers='markers' in overlays,
            show_residuals='residuals' in overlays,
            show_phase_fold=False,
            show_diagnostics='diagnostics' in overlays,
            confidence_colors='confidence' in overlays,
            run_params=run_params or {},
            uirevision_key=uirevision_key,
            theme=theme_mode,
            residual_fraction=residual_height,
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

    return (
        plot_src,
        grid_items,
        metadata_health,
        vetting_banner,
        progress,
        native['figure'],
        {'display': 'block', 'width': '100%', 'height': '100%'},
        {'display': 'none'},
        native['camera_options'],
        _render_stat_cards(native['stat_rows']),
        _render_plot_status_panel(native.get('status', 'ok'), native.get('status_message', ''), (native.get('warnings', []) + mismatch_warnings)),
        _render_camera_diag_panel(native.get('camera_diagnostics', {}), filtered),
        _render_run_config_panel(run_params if run_params else None, run_params_path, mismatch_warnings),
        _render_repro_badge(run_params if run_params else None, mismatch_warnings),
        json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
        nonce,
    )


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
        return 'unclassified', False, 1, '', 0

    candidate_id = queue_data['candidate_ids'][idx]
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, candidate_id)

    return (
        review.get('event_class', 'unclassified'),
        review.get('status', 'unreviewed') == 'needs_followup',
        review.get('review_pass', 1),
        review.get('notes', ''),
        review.get('interest_score', 0),
    )


# Score button clicks
@app.callback(
    [Output('current-score', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    [Input(f'score-{i}', 'n_clicks') for i in range(6)],
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

    return score, f"✓ Score: {score}", new_pass


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
        candidate_id, score or 0, event_class, needs_followup, notes, 'save_button',
    )

    return "✓ Saved", new_pass


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

    candidate_id = queue_data['candidate_ids'][idx]
    new_pass, _ = _do_save(
        candidate_id, score or 0, event_class, needs_followup, notes, 'done_button',
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
    [Output(f'score-{i}', 'className') for i in range(6)],
    Input('current-score', 'data'),
    prevent_initial_call=False
)
def update_score_buttons(current_score):
    """Highlight the active score button."""
    score = current_score or 0
    return ['score-btn active' if i == score else 'score-btn' for i in range(6)]


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


# Followup indicator
@app.callback(
    Output('followup-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    prevent_initial_call=False
)
def update_followup_indicator(needs_followup):
    """Show followup flag status."""
    if needs_followup:
        return html.Span('[F] Followup: ON', style={'color': '#f80'})
    return html.Span('[F] Followup: off', style={'color': '#666'})


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


# Auto-detect run directory files
@app.callback(
    [Output('import-path', 'value'),
     Output('sidebar-status', 'children', allow_duplicate=True)],
    Input('auto-detect-btn', 'n_clicks'),
    State('run-dir-path', 'value'),
    prevent_initial_call=True
)
def auto_detect_files(n_clicks, run_dir_path):
    """Auto-detect files from run directory."""
    if not n_clicks or not run_dir_path:
        return no_update, no_update

    try:
        run_dir = Path(run_dir_path).expanduser().resolve()
        detected = detect_run_directory_files(run_dir)

        messages = []
        import_path_value = no_update

        with closing(db_connect(Path(DB_PATH))) as conn:
            if detected['candidates']:
                import_path_value = str(detected['candidates'])
                save_app_state(conn, "last_input_file", str(detected['candidates']))
                messages.append(f"✓ Candidates: {detected['candidates'].name}")

            if detected['plot_dir']:
                global PLOT_DIR
                detected_plot_dir = Path(detected['plot_dir']).expanduser().resolve()
                PLOT_DIR = str(detected_plot_dir)
                save_app_state(conn, "last_plot_dir", str(detected_plot_dir))
                messages.append(f"✓ Plots: {detected['plot_dir'].name}/")

            if detected['gaia_cache']:
                save_app_state(conn, "last_gaia_cache", str(detected['gaia_cache']))
                messages.append(f"✓ Gaia cache")

            save_app_state(conn, "last_run_dir", str(run_dir))

        if detected['warnings']:
            for warn in detected['warnings']:
                messages.append(f"⚠ {warn}")

        if not detected['candidates'] and not detected['plot_dir']:
            return no_update, "✗ No files detected"

        return import_path_value, " | ".join(messages)

    except Exception as e:
        return no_update, f"✗ Auto-detect failed: {str(e)}"


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
     State('import-trigger', 'data')],
    prevent_initial_call=True
)
def import_candidates_callback(n_clicks, import_path, characterize_on,
                               crossmatch, gaia_cache, chunk_size, dust_on, starhorse, current_trigger):
    """Import candidates from file."""
    if not n_clicks or not import_path:
        return no_update, no_update

    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            src = Path(import_path).expanduser()
            df = load_candidates_file(src)

            enable_characterize = 'yes' in (characterize_on or [])

            n_rows, n_new = import_candidates(
                conn, df, str(src),
                characterize_before_import=enable_characterize,
                characterize_crossmatch=Path(crossmatch).expanduser() if enable_characterize and crossmatch else None,
                characterize_cache=Path(gaia_cache).expanduser() if enable_characterize and gaia_cache else None,
                characterize_chunk_size=int(chunk_size) if enable_characterize and chunk_size else GAIA_CHUNK_SIZE,
                characterize_dust='yes' in (dust_on or []) if enable_characterize else False,
                characterize_starhorse=starhorse.strip() if enable_characterize and starhorse and starhorse.strip() else None,
            )
            save_app_state(conn, "last_input_file", str(src))
            # Increment trigger to cause queue refresh
            return f"✓ Imported {n_rows} rows ({n_new} new)", (current_trigger or 0) + 1
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

    DB_PATH = str(Path(args.db).expanduser().resolve())

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
        PLOT_DIR = str(Path(args.plot_dir).expanduser().resolve())
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
    print("  [N] Next | [P] Previous | [0-5] Score | [S] Save | [D] Done | [T] Toggle sidebar | [?] Help")
    print("")

    # Auto-open browser
    url = f"http://{args.host}:{args.port}"
    Timer(0.1, lambda: webbrowser.open(url)).start()

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
