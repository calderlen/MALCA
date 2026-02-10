"""Dash-based keyboard-driven review app for MALCA candidates."""

import sys
import argparse
from pathlib import Path
import webbrowser
from threading import Timer

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
from flask import send_from_directory
import pandas as pd

from malca.review.store import (
    DEFAULT_DB_PATH,
    INTEREST_REASON_TAGS,
    db_connect,
    get_review,
    save_review,
    find_plot_image,
    load_app_state,
    save_app_state,
    import_candidates,
    load_candidates_file,
    recent_history,
    export_reviews,
    detect_run_directory_files,
)
from malca.review.metadata import (
    extract_review_metadata_grouped,
    is_group_default_open,
)
from malca.review.keyboard import (
    handle_key_action, HELP_TEXT,
    REASON_KEY_MAP, REASON_PREFIX_KEY,
    CLASS_KEY_MAP, CLASS_PREFIX_KEY,
    PREFIX_KEYS,
)
from malca.review.session import create_queue_data_dict
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
        }
        .header-bar {
            background-color: #0a0a0a;
            border-bottom: 1px solid #555;
            padding: 6px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
        }
        .plot-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #000;
            overflow: hidden;
            max-height: 80vh;
        }
        .plot-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
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
            padding: 6px 15px;
            border-top: 1px solid #555;
            max-height: 200px;
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
            color: #0f0;
            font-size: 11px;
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
            background-color: #1a1a1a !important;
            border-color: #444 !important;
        }
        .Select-menu-outer {
            background-color: #1a1a1a !important;
        }
        input, textarea {
            background-color: #1a1a1a !important;
            border: 1px solid #555 !important;
            color: #e0e0e0 !important;
        }
        /* Dropdown styling */
        .Select-control, .Select-menu-outer, .Select-menu, .Select-option {
            background-color: #1a1a1a !important;
            color: #e0e0e0 !important;
            border-color: #555 !important;
        }
        .Select-placeholder, .Select-value-label {
            color: #aaa !important;
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
            background-color: #1a1a1a !important;
            border-color: #444 !important;
        }
        .dash-dropdown .Select-menu-outer {
            background-color: #1a1a1a !important;
            border-color: #444 !important;
        }
        .dash-dropdown .Select-option {
            background-color: #1a1a1a !important;
            color: #e0e0e0 !important;
        }
        .dash-dropdown .Select-option:hover {
            background-color: #2a2a2a !important;
            color: #fff !important;
        }
        /* Checklist items */
        .dash-checklist label {
            color: #e0e0e0 !important;
        }
        .dash-checklist input[type="checkbox"] {
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
            max-height: 300px;
            overflow-y: auto;
            padding: 0 12px;
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
DB_PATH = None
PLOT_DIR = None


def _keyboard_key(key_value: str | None) -> str:
    """Extract raw key token from encoded keyboard input value."""
    if not key_value:
        return ""
    return str(key_value).split("\t", 1)[0].strip()


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
    attrs = {'open': True} if default_open else {}
    return html.Details([
        html.Summary(name),
        html.Div(children, style={'padding-left': '6px'}),
    ], **attrs)


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
        dcc.Store(id='interest-reasons-store', data=[]),
        dcc.Store(id='event-class-store', data='unclassified'),
        dcc.Store(id='pending-prefix', data=''),
        dcc.Store(id='needs-followup-store', data=False),
        dcc.Store(id='review-pass-store', data=1),
        dcc.Store(id='sidebar-state', data=False),  # collapsed by default
        dcc.Store(id='filter-params', data={}),
        dcc.Store(id='import-trigger', data=0),  # triggers queue refresh after import
        dcc.Store(id='activity-visible', data=False),  # collapsed by default
        dcc.Interval(id='keyboard-init', interval=200, n_intervals=0, max_intervals=1),

        # Sidebar toggle button
        html.Div('☰', id='sidebar-toggle', className='sidebar-toggle', title='Toggle sidebar [T]'),

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

            html.Div(id='sidebar-status', style={'margin-top': '10px', 'color': '#0f0', 'font-size': '11px'}),

        ], id='sidebar', className='sidebar'),

        # Main content
        html.Div([
            # Header bar
            html.Div([
                html.Span(id='progress-text', style={'color': '#0af', 'font-size': '11px'}),
                html.A('[?] Shortcuts', id='help-link', className='help-link'),
            ], className='header-bar'),

            # Plot area
            html.Div([
                html.Img(id='plot-image', src='', alt='Light curve plot')
            ], className='plot-container'),

            # Grouped candidate metadata sections (collapsible)
            html.Div(id='candidate-info-grid', className='metadata-sections'),

            # Control bar
            html.Div([
                # Score row
                html.Div([
                    html.Span('Score: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                ] + [
                    html.Button(str(i), id=f'score-{i}', n_clicks=0, className='score-btn')
                    for i in range(6)
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '6px'}),

                # Reason toggle row (clickable buttons, R+key prefix)
                html.Div([
                    html.Span('Reasons: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                ] + [
                    html.Button(
                        f'R {key.upper()}: {tag.replace("_", " ")}',
                        id=f'reason-badge-{tag}',
                        n_clicks=0,
                        className='badge-btn',
                    )
                    for key, tag in REASON_KEY_MAP.items()
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'margin-bottom': '6px'}),

                # Event class row (clickable buttons, G+key prefix)
                html.Div([
                    html.Span('Class: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                    html.Span(id='prefix-indicator', style={'margin-right': '6px', 'font-size': '11px'}),
                ] + [
                    html.Button(
                        f'G {key.upper()}: {tag.replace("_", " ")}',
                        id=f'class-badge-{tag}',
                        n_clicks=0,
                        className='badge-btn',
                    )
                    for key, tag in CLASS_KEY_MAP.items()
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'margin-bottom': '6px'}),

                # Action row: Save, Done, Followup, Pass, Status, Notification
                html.Div([
                    html.Button('Save [S]', id='save-btn', n_clicks=0, className='action-btn'),
                    html.Button('Done [D]', id='done-btn', n_clicks=0, className='action-btn primary'),
                    html.Span('', style={'width': '20px', 'display': 'inline-block'}),
                    html.Span(id='followup-indicator', style={'margin-right': '10px', 'font-size': '11px'}),
                    html.Span(id='pass-indicator', style={'color': '#888', 'margin-right': '10px', 'font-size': '11px'}),
                    html.Span(id='status-indicator', style={'color': '#888', 'margin-right': '10px', 'font-size': '11px'}),
                    html.Div(id='notification', className='notification', style={'display': 'inline-block', 'margin-left': '10px'}),
                ], style={'display': 'flex', 'align-items': 'center'}),
            ], className='control-bar'),

            # Notes (M to enter, Esc to exit)
            html.Div([
                html.Label('[M] Notes (Esc to exit):', style={'color': '#aaa', 'display': 'block', 'margin-bottom': '3px', 'font-size': '11px'}),
                dcc.Textarea(id='notes', style={'width': '100%', 'height': '50px', 'font-size': '11px'}),
            ], className='review-form'),

            # Recent activity
            html.Div([
                html.Div([
                    html.Span('[A] Activity', style={'color': '#0af', 'font-size': '11px', 'cursor': 'pointer'}),
                ], id='activity-toggle', style={'padding': '5px 20px', 'background-color': '#0a0a0a', 'border-top': '1px solid #555', 'cursor': 'pointer'}),
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

        // Get the keyboard input element
        const keyboardInput = document.getElementById('keyboard-input');

        if (!keyboardInput) {
            console.error('keyboard-input element not found!');
            return;
        }

        // Use the native HTMLInputElement value setter to bypass React's
        // internal value tracking.  React overrides the .value setter so
        // that its tracker stays in sync; if we set .value through that
        // override *before* dispatching the event, React sees no change
        // and never fires onChange → Dash callbacks never trigger.
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        ).set;

        const dispatchKeyToDash = function(key) {
            if (!key) {
                return;
            }
            nativeInputValueSetter.call(
                keyboardInput, key + '\t' + String(Date.now())
            );
            keyboardInput.dispatchEvent(new Event('input', { bubbles: true }));
        };

        // Register once: global keyboard listener that feeds Dash callbacks.
        if (!window.__malcaKeyboardListenerAttached) {
            document.addEventListener('keydown', function(e) {
                if (e.ctrlKey || e.metaKey || e.altKey) {
                    return;
                }

                const target = e.target;
                const tag = target && target.tagName ? target.tagName : '';
                const targetId = target && target.id ? target.id : '';

                // Inside a form field: allow Escape to exit, ignore everything else.
                if ((tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') && targetId !== 'keyboard-input') {
                    if (e.key === 'Escape') {
                        target.blur();
                    }
                    return;
                }

                const key = e.key;
                if (!key || key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta') {
                    return;
                }

                // M / m → focus the notes textarea (pure client-side).
                if (key === 'm' || key === 'M') {
                    e.preventDefault();  // don't type 'm' into the textarea
                    const notesEl = document.getElementById('notes');
                    if (notesEl) {
                        const ta = notesEl.querySelector('textarea') || notesEl;
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
    [Input('queue-data', 'data'),
     Input('refresh-btn', 'n_clicks'),
     Input('import-trigger', 'data')],
    _queue_states,
    prevent_initial_call=False
)
def load_queue(existing_data, refresh_clicks, import_trigger, *state_values):
    """Load queue data from all sidebar filter states."""
    conn = db_connect(Path(DB_PATH))

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


def _do_save(candidate_id, score, interest_reasons, event_class, needs_followup, notes, event_type):
    """Shared save helper.  Auto-sets status and auto-increments review_pass."""
    conn = db_connect(Path(DB_PATH))
    review = get_review(conn, candidate_id)
    new_pass = max(1, review.get('review_pass', 0)) + 1
    status = 'needs_followup' if needs_followup else 'reviewed'
    save_review(
        conn,
        candidate_id=candidate_id,
        interest_score=score,
        interest_reason=interest_reasons or [],
        event_class=event_class or 'unclassified',
        review_pass=new_pass,
        notes=notes or '',
        status=status,
        reviewer='calder',
        event_type=event_type,
    )
    return new_pass, status


# Keyboard handler (with G-prefix state machine for event class)
@app.callback(
    [Output('current-index', 'data'),
     Output('notification', 'children'),
     Output('current-score', 'data', allow_duplicate=True),
     Output('interest-reasons-store', 'data', allow_duplicate=True),
     Output('needs-followup-store', 'data', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True),
     Output('event-class-store', 'data', allow_duplicate=True),
     Output('pending-prefix', 'data', allow_duplicate=True)],
    Input('keyboard-input', 'value'),
    [State('current-index', 'data'),
     State('queue-data', 'data'),
     State('current-score', 'data'),
     State('interest-reasons-store', 'data'),
     State('event-class-store', 'data'),
     State('pending-prefix', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def handle_keyboard(key_value, current_idx, queue_data, current_score,
                    interest_reasons, event_class, pending_prefix, needs_followup, notes):
    """Handle keyboard input."""
    NO = (no_update,) * 8  # shorthand for all-no_update

    key = _keyboard_key(key_value)
    if not key or not queue_data:
        return NO

    # Skip keys handled by other callbacks / keydown listener
    if key.lower() in ['t', 'm', 'a', '?']:
        return NO

    queue_size = queue_data['queue_size']
    if queue_size == 0:
        return no_update, "Queue is empty", *([no_update] * 6)

    candidate_id = (queue_data['candidate_ids'][current_idx]
                    if current_idx < queue_size else None)

    # --- Prefix state machine (G → class, R → reason) ---
    if pending_prefix:
        # We're waiting for the second key after a leader press
        if key == 'Escape':
            return no_update, "Cancelled", no_update, no_update, no_update, no_update, no_update, ''

        if pending_prefix == CLASS_PREFIX_KEY:
            class_tag = CLASS_KEY_MAP.get(key.lower())
            if class_tag is not None:
                cur = event_class or 'unclassified'
                if cur == class_tag:
                    return no_update, "Class: unclassified", no_update, no_update, no_update, no_update, 'unclassified', ''
                return no_update, f"Class: {class_tag}", no_update, no_update, no_update, no_update, class_tag, ''
            return no_update, f"G {key}: unknown class", no_update, no_update, no_update, no_update, no_update, ''

        if pending_prefix == REASON_PREFIX_KEY:
            reason_tag = REASON_KEY_MAP.get(key.lower())
            if reason_tag is not None:
                reasons = list(interest_reasons or [])
                if reason_tag in reasons:
                    reasons.remove(reason_tag)
                    msg = f"- {reason_tag}"
                else:
                    reasons.append(reason_tag)
                    msg = f"+ {reason_tag}"
                return no_update, msg, no_update, reasons, no_update, no_update, no_update, ''
            return no_update, f"R {key}: unknown reason", no_update, no_update, no_update, no_update, no_update, ''

        # Unknown prefix (shouldn't happen) — cancel
        return no_update, "Cancelled", no_update, no_update, no_update, no_update, no_update, ''

    # --- Enter prefix mode when a leader key is pressed ---
    kl = key.lower()
    if kl in PREFIX_KEYS:
        label = kl.upper()
        return no_update, f"{label} ...", no_update, no_update, no_update, no_update, no_update, kl

    # --- Followup toggle (F) ---
    if key.lower() == 'f':
        new_state = not bool(needs_followup)
        label = "ON" if new_state else "OFF"
        return no_update, f"Followup: {label}", no_update, no_update, new_state, no_update, no_update, no_update

    # --- Navigation / scoring / save via handle_key_action ---
    conn = db_connect(Path(DB_PATH))
    new_idx, notification, should_save = handle_key_action(
        key, current_idx, queue_size, conn, candidate_id
    )

    new_score = no_update
    new_pass = no_update
    if should_save and candidate_id:
        score = int(key) if key in '012345' else current_score
        pass_val, _ = _do_save(
            candidate_id, score, interest_reasons, event_class, needs_followup, notes, 'keyboard',
        )
        new_score = score
        new_pass = pass_val

    # Only update current-index when it actually changes (navigation).
    # Returning the same value would re-trigger load_review_form needlessly.
    idx_out = new_idx if new_idx != current_idx else no_update

    return idx_out, notification, new_score, no_update, no_update, new_pass, no_update, no_update


# Update plot and unified candidate info
@app.callback(
    [Output('plot-image', 'src'),
     Output('candidate-info-grid', 'children'),
     Output('progress-text', 'children')],
    Input('current-index', 'data'),
    State('queue-data', 'data'),
    prevent_initial_call=False
)
def update_display(idx, queue_data):
    """Update plot and unified candidate info grid."""
    if not queue_data or queue_data['queue_size'] == 0:
        return '', 'No candidates in queue', '[0/0]'

    queue_size = queue_data['queue_size']
    if idx < 0 or idx >= queue_size:
        return '', 'Invalid index', f'[{idx}/{queue_size}]'

    candidate_id = queue_data['candidate_ids'][idx]
    payload = queue_data['payloads'].get(candidate_id, {})

    # Find plot
    plot_path = find_plot_image(payload, Path(PLOT_DIR))
    if plot_path and plot_path.exists():
        # Get relative path from PLOT_DIR (includes subdirectory like 'jump/')
        try:
            rel_path = plot_path.relative_to(Path(PLOT_DIR))
            plot_src = f'/plots/{rel_path}'
        except ValueError:
            # Fallback if not relative
            plot_src = f'/plots/{plot_path.name}'
    else:
        plot_src = ''

    # Build grouped collapsible metadata sections
    grouped = extract_review_metadata_grouped(payload)

    grid_items = []
    for group_name, items in grouped:
        field_divs = [
            html.Div([
                html.Span(f"{label}: ", style={'color': '#888'}),
                html.Span(str(value), style={'color': '#e0e0e0'}),
            ], style={'white-space': 'nowrap', 'overflow': 'hidden', 'text-overflow': 'ellipsis'})
            for label, value in items
        ]
        details_attrs = {'open': True} if is_group_default_open(group_name) else {}
        grid_items.append(
            html.Details([
                html.Summary(f"{group_name} ({len(items)})"),
                html.Div(field_divs, className='meta-grid'),
            ], **details_attrs)
        )

    # Progress
    progress = f"[{idx + 1}/{queue_size}] Queue: {queue_size}"

    return plot_src, grid_items, progress


# Load review data for current candidate
@app.callback(
    [Output('interest-reasons-store', 'data'),
     Output('event-class-store', 'data'),
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
        return [], 'unclassified', False, 1, '', 0

    candidate_id = queue_data['candidate_ids'][idx]
    conn = db_connect(Path(DB_PATH))
    review = get_review(conn, candidate_id)

    return (
        review.get('interest_reason', []),
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
     State('interest-reasons-store', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def handle_score_clicks(*args):
    """Handle score button clicks."""
    idx, queue_data, interest_reasons, event_class, needs_followup, notes = args[-6:]

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
        candidate_id, score, interest_reasons, event_class, needs_followup, notes, 'button',
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
     State('interest-reasons-store', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def save_review_callback(n_clicks, idx, queue_data, score,
                         interest_reasons, event_class, needs_followup, notes):
    """Save review."""
    if not n_clicks or not queue_data or not queue_data.get('candidate_ids'):
        return no_update, no_update

    if idx >= len(queue_data['candidate_ids']):
        return "Invalid candidate index", no_update

    candidate_id = queue_data['candidate_ids'][idx]
    new_pass, _ = _do_save(
        candidate_id, score or 0, interest_reasons, event_class, needs_followup, notes, 'save_button',
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
     State('interest-reasons-store', 'data'),
     State('event-class-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def done_callback(n_clicks, idx, queue_data, score,
                  interest_reasons, event_class, needs_followup, notes):
    """Save and go to next."""
    if not n_clicks or not queue_data or not queue_data.get('candidate_ids'):
        return no_update, no_update, no_update

    if idx >= len(queue_data['candidate_ids']):
        return no_update, "Invalid candidate index", no_update

    candidate_id = queue_data['candidate_ids'][idx]
    new_pass, _ = _do_save(
        candidate_id, score or 0, interest_reasons, event_class, needs_followup, notes, 'done_button',
    )

    queue_size = queue_data['queue_size']
    new_idx = min(idx + 1, queue_size - 1)

    return new_idx, "✓ Saved + Next →", new_pass


# --- Display callbacks for stores → visible indicators ---

_REASON_ACTIVE_STYLE = {
    'border': '1px solid #0af', 'color': '#0af', 'background-color': '#003366',
}
_REASON_INACTIVE_STYLE = {
    'border': '1px solid #444', 'color': '#888', 'background-color': 'transparent',
}
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


# Update reason badges styling
@app.callback(
    [Output(f'reason-badge-{tag}', 'style') for tag in INTEREST_REASON_TAGS],
    Input('interest-reasons-store', 'data'),
    prevent_initial_call=False
)
def update_reason_badges(active_reasons):
    """Highlight active reason badges."""
    active = set(active_reasons or [])
    return [_REASON_ACTIVE_STYLE if tag in active else _REASON_INACTIVE_STYLE
            for tag in INTEREST_REASON_TAGS]


# Reason badge click handler
@app.callback(
    [Output('interest-reasons-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input(f'reason-badge-{tag}', 'n_clicks') for tag in INTEREST_REASON_TAGS],
    State('interest-reasons-store', 'data'),
    prevent_initial_call=True
)
def handle_reason_clicks(*args):
    """Toggle a reason when its badge is clicked."""
    reasons_state = args[-1]
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    # Extract tag from 'reason-badge-<tag>'
    tag = trigger.replace('reason-badge-', '')
    reasons = list(reasons_state or [])
    if tag in reasons:
        reasons.remove(tag)
        return reasons, f"- {tag}"
    reasons.append(tag)
    return reasons, f"+ {tag}"


# Update event class badge styling
@app.callback(
    [Output(f'class-badge-{tag}', 'style') for tag in CLASS_KEY_MAP.values()],
    Input('event-class-store', 'data'),
    prevent_initial_call=False
)
def update_class_badges(active_class):
    """Highlight the active event class badge."""
    active = active_class or 'unclassified'
    return [_CLASS_ACTIVE_STYLE if tag == active else _CLASS_INACTIVE_STYLE
            for tag in CLASS_KEY_MAP.values()]


# Class badge click handler
@app.callback(
    [Output('event-class-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input(f'class-badge-{tag}', 'n_clicks') for tag in CLASS_KEY_MAP.values()],
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
        return html.Span(f'{prefix.upper()} ...', style={'color': '#f80', 'font-weight': 'bold'})
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
    conn = db_connect(Path(DB_PATH))
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
    conn = db_connect(Path(DB_PATH))
    recent = recent_history(conn, limit=5)

    if recent.empty:
        return "No recent activity"

    lines = []
    for _, row in recent.iterrows():
        lines.append(f"• {row['candidate_id']} - {row['event_type']} ({row['created_at']})")

    return html.Div([html.Div(line, style={'margin': '5px 0'}) for line in lines])


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
        'padding': '10px 20px',
        'background-color': '#0a0a0a',
        'color': '#aaa',
        'font-size': '11px'
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
        run_dir = Path(run_dir_path).expanduser()
        detected = detect_run_directory_files(run_dir)

        messages = []
        import_path_value = no_update

        conn = db_connect(Path(DB_PATH))

        if detected['candidates']:
            import_path_value = str(detected['candidates'])
            save_app_state(conn, "last_input_file", str(detected['candidates']))
            messages.append(f"✓ Candidates: {detected['candidates'].name}")

        if detected['plot_dir']:
            global PLOT_DIR
            PLOT_DIR = str(detected['plot_dir'])
            save_app_state(conn, "last_plot_dir", str(detected['plot_dir']))
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
        conn = db_connect(Path(DB_PATH))
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
        conn = db_connect(Path(DB_PATH))
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
    args = parser.parse_args()

    DB_PATH = args.db

    # Auto-detect plot directory if not specified
    if args.plot_dir:
        PLOT_DIR = args.plot_dir
    else:
        # Try current directory first
        if Path('./plots').is_dir():
            PLOT_DIR = './plots'
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
    print("  N - Next | P - Previous | 0-5 - Score | S - Save | D - Done | T - Toggle sidebar | ? - Help")
    print("")

    # Auto-open browser
    url = f"http://{args.host}:{args.port}"
    Timer(0.1, lambda: webbrowser.open(url)).start()

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
