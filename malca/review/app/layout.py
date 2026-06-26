# This file was mechanically split from malca.review.app; preserve behavior when editing.
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
        dcc.Store(id='active-taxonomy-menu', data=''),
        dcc.Store(id='taxonomy-selection-store', data={}),
        dcc.Store(id='taxonomy-submenu-store', data=''),
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
        dcc.Store(id='dustycult-refresh-token', data=0),
        dcc.Store(id='phoebe-refresh-token', data=0),
        dcc.Store(id='plot-render-request', data={'nonce': 1, 'ts': 0.0, 'state': {'idx': 0, 'candidate_id': None, 'plot_mode': 'native', 'overlay_values': list(PLOT_PRESETS['Diagnostics']['overlays']), 'selected_cameras': [], 'selected_bands': ['g', 'V'], 'native_color_mode': 'camera', 'preset': 'Diagnostics', 'theme': DEFAULT_THEME, 'residual_height': DEFAULT_RESIDUAL_FRACTION, 'baseline_opacity': 0.5, 'external_source_values': list(DEFAULT_EXTERNAL_SOURCE_VALUES), 'external_source_view': DEFAULT_EXTERNAL_SOURCE_VIEW, 'external_source_layout': DEFAULT_EXTERNAL_SOURCE_LAYOUT, 'phase_panel_mode': 'fold'}}),
        dcc.Store(id='plot-render-applied', data=0),
        dcc.Store(id='plot-defaults-initialized', data=False),
        dcc.Store(id='queue-source-path', data=''),
        dcc.Store(id='run-config-json-store', data=''),
        dcc.Store(id='theme-mode-store', data=DEFAULT_THEME),
        dcc.Store(id='cutout-selected-survey', data=DEFAULT_CUTOUT_SURVEY_KEY, storage_type='session'),
        dcc.Store(id='review-session-start', data=None, storage_type='session'),
        dcc.Store(id='metadata-resize-init', data=0),
        dcc.Store(id='status-resize-init', data=0),
        dcc.Store(id='eda-resize-init', data=0),
        dcc.Store(id='eda-selection-candidate-ids', data=[]),
        dcc.Store(id='metadata-copy-init', data=0),
        dcc.Store(id='eda-panel-state', data='open', storage_type='local'),
        dcc.Store(id='sidebar-plot-saved', data=0),  # dummy sink for plot prefs save callback
        dcc.Store(id='candidate-start-time', data=0),
        dcc.Store(id='review-progress-state', data={'reviewed': 0, 'total': 0}),
        dcc.Store(id='startup-selection-applied', data=False),
        dcc.Store(id='last-candidate-saved', data=0),
        dcc.Download(id='plot-export-download'),
        dcc.Download(id='sed-export-download'),
        dcc.Download(id='dustycult-export-download'),
        dcc.Download(id='mini-plot-export-download'),
        dcc.Download(id='eda-plot-export-download'),
        dcc.Download(id='run-config-download'),
        dcc.Download(id='spectrum-export-download'),
        dcc.Interval(id='keyboard-init', interval=200, n_intervals=0, max_intervals=1),
        dcc.Interval(id='startup-lazy-init', interval=2500, n_intervals=0, max_intervals=1),
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
            ),
            dcc.Checklist(
                id='filter-failed',
                options=[{'label': ' Require failed_any=False', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'},
            ),
            dcc.Checklist(
                id='select-filter-mode',
                options=[{'label': ' Include selected categorical values', 'value': 'include'}],
                value=[],
                style={'margin-bottom': '6px'},
            ),
            html.Div([
                dcc.Input(
                    id='review-filter-search-query',
                    placeholder='Find filter...',
                    type='text',
                    style={**_inp_style, 'marginBottom': '0'},
                ),
                html.Div([
                    html.Button('Prev', id='review-filter-search-prev-btn', n_clicks=0, className='compact-btn'),
                    html.Button('Next', id='review-filter-search-next-btn', n_clicks=0, className='compact-btn'),
                ], style={'display': 'flex', 'gap': '6px'}),
                html.Div(
                    'Type to find a filter',
                    id='review-filter-search-status',
                    style={'fontSize': '10px', 'color': '#7d91a6', 'marginTop': '4px'},
                ),
            ], style={'marginBottom': '8px'}),

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
                       for ftype, col in items
                       if ftype == 'num' and col not in {'interest_score', 'review_pass'}]
                     + [{'label': 'Confidence', 'value': 'interest_score'},
                        {'label': 'Review Pass', 'value': 'review_pass'},
                        {'label': 'Updated At', 'value': 'updated_at'}]
                ),
                value=['candidate_id'],
                multi=True,
                clearable=False,
                style={'margin-bottom': '4px', 'font-size': '11px'},
            ),
            dcc.Checklist(
                id='sort-desc',
                options=[{'label': ' Descending', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '6px'},
            ),

            html.Button('Refresh Queue', id='refresh-btn', n_clicks=0,
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
            html.Div('Use the EDA rail on the right to plot and jump within the current queue.',
                     style={'fontSize': '10px', 'color': '#7d91a6', 'marginBottom': '6px'}),

            html.Div('Batch Export', className='section-title', style={'marginTop': '20px'}),
            html.Div([
                dcc.Input(id='batch-export-path', type='text', placeholder='/path/to/export/dir', className='text-input', style={'width': '100%', 'marginBottom': '8px'}),
                dcc.Dropdown(
                    id='batch-export-target',
                    options=[
                        {'label': 'All in Queue', 'value': 'all'},
                        {'label': 'Unreviewed in Queue', 'value': 'unreviewed'}
                    ],
                    value='unreviewed',
                    clearable=False,
                    style={'marginBottom': '8px'}
                ),
                html.Button('Export PDFs', id='batch-export-btn', className='action-btn primary', n_clicks=0, style={'width': '100%'}),
                dcc.Loading(id='batch-export-loading', type='dot', children=[
                    html.Div(id='batch-export-status', style={'fontSize': '11px', 'marginTop': '8px', 'color': '#7d91a6', 'minHeight': '15px'})
                ])
            ], style={'padding': '0 1px', 'marginBottom': '12px'}),

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
                children=[
                    html.Div(id='fetch-status',
                             style={'fontSize': '11px', 'marginTop': '4px', 'color': '#7da8c4'}),
                ],
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
                     value=str(DEFAULT_OUTPUT_DIR / 'review' / 'reviewed_candidates.parquet'), style=_inp_style,
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
            dcc.Input(id='review-sync-dir', placeholder='Review Git bundle directory', type='text',
                     value='reviews', style=_inp_style,
                     persistence=_review_persistence_token(), persistence_type='local'),
            dcc.Checklist(
                id='review-sync-hash-assets',
                options=[{'label': ' Hash resolved assets', 'value': 'yes'}],
                value=[],
                style={'margin-bottom': '4px'},
                persistence=_review_persistence_token(),
                persistence_type='local',
            ),
            html.Button('Export Git Bundle', id='export-review-sync-btn', n_clicks=0, className='action-btn',
                       style={'width': '100%'}),

            html.Hr(),

            html.Div('Merge Reviews', className='section-title'),
            dcc.Input(
                id='merge-target-db-path',
                placeholder='Target review DB path',
                type='text',
                value=str(DEFAULT_OUTPUT_DIR / 'review' / 'review.db'),
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
            html.Label('Search Radius [arcsec]:'),
            dcc.Input(id='link-radius-arcsec', placeholder='Radius [arcsec]', type='number',
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
                                html.Div([
                                    dcc.Dropdown(
                                        id='plot-preset',
                                        options=[{'label': p, 'value': p} for p in ('Fast Review', 'Clean', 'Diagnostics', 'Full')],
                                        value='Diagnostics',
                                        clearable=False,
                                        style={'width': '100%', 'font-size': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                ], className='plot-control-group plot-control-preset'),
                                html.Div([
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
                                ], className='plot-control-group plot-control-radio'),
                                html.Div([
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
                                ], className='plot-control-group plot-control-checks plot-control-full'),
                                html.Div([
                                    dcc.RadioItems(
                                        id='yaxis-mode',
                                        options=[
                                            {'label': ' Mag', 'value': 'mag'},
                                            {'label': ' Flux', 'value': 'flux'},
                                        ],
                                        value='mag',
                                        inline=True,
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                ], className='plot-control-group plot-control-radio'),
                                html.Div([
                                    html.Span('Native colors', className='plot-control-label'),
                                    dcc.RadioItems(
                                        id='native-color-mode',
                                        options=[
                                            {'label': ' Camera colors', 'value': 'camera'},
                                            {'label': ' Band colors', 'value': 'band'},
                                        ],
                                        value='camera',
                                        inline=True,
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                ], className='plot-control-group plot-control-radio'),
                                html.Div([
                                    html.Span('Baseline', className='plot-control-label'),
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
                                ], className='plot-control-group toolbar-slider-control'),
                                html.Div([
                                    html.Span('Residual', className='plot-control-label'),
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
                                ], className='plot-control-group toolbar-slider-control'),
                                html.Div([
                                    html.Button('Reset', id='plot-reset-btn', n_clicks=0, className='compact-btn'),
                                    html.Button('Export', id='export-plot', n_clicks=0, className='compact-btn'),
                                    html.Span(id='plot-render-indicator', className='plot-control-status'),
                                    html.Span(id='repro-badge', className='label-chip plot-control-status'),
                                ], className='plot-control-group plot-actions plot-control-full'),
                                html.Div([
                                    html.Div([
                                        html.Span('LC Sources', className='plot-control-label'),
                                        html.Button('All', id='sources-all-btn', n_clicks=0, className='compact-btn'),
                                        html.Button('ASAS-SN', id='sources-native-btn', n_clicks=0, className='compact-btn'),
                                        html.Button('Clear', id='sources-clear-btn', n_clicks=0, className='compact-btn'),
                                    ], className='plot-source-header'),
                                    dcc.Checklist(
                                        id='external-source-values',
                                        options=EXTERNAL_SOURCE_VIEW_OPTIONS,
                                        value=list(DEFAULT_EXTERNAL_SOURCE_VALUES),
                                        inline=True,
                                        className='plot-source-checklist',
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    dcc.RadioItems(
                                        id='external-source-layout',
                                        options=EXTERNAL_SOURCE_LAYOUT_OPTIONS,
                                        value=DEFAULT_EXTERNAL_SOURCE_LAYOUT,
                                        inline=True,
                                        className='plot-source-layout',
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                ], className='plot-control-group plot-source-control plot-control-full'),
                                html.Div([
                                    html.Span('Phase view', className='plot-control-label'),
                                    dcc.RadioItems(
                                        id='phase-panel-mode',
                                        options=[
                                            {'label': ' Fold', 'value': 'fold'},
                                            {'label': ' Time', 'value': 'time'},
                                        ],
                                        value='fold',
                                        inline=True,
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    html.Span('Period', className='plot-control-label'),
                                    dcc.Dropdown(
                                        id='period-method',
                                        options=[
                                            {'label': 'LSP', 'value': 'lsp'},
                                            {'label': 'PDM', 'value': 'pdm'},
                                            {'label': 'CE', 'value': 'ce'},
                                        ],
                                        value='pdm',
                                        clearable=False,
                                        className='period-method-control',
                                        style={'font-size': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    dcc.Input(
                                        id='pdm-min-period',
                                        type='number',
                                        value=0.1,
                                        min=0.001,
                                        step='any',
                                        debounce=True,
                                        placeholder='Min P',
                                        className='period-number-input',
                                        style={'font-size': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    html.Span('-', className='plot-control-label'),
                                    dcc.Input(
                                        id='pdm-max-period',
                                        type='number',
                                        value=10,
                                        min=0.001,
                                        step='any',
                                        debounce=True,
                                        placeholder='Max P',
                                        className='period-number-input',
                                        style={'font-size': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    html.Span('[d]', className='plot-control-label'),
                                    html.Button('Find Period', id='pdm-run-btn', n_clicks=0, className='compact-btn'),
                                    dcc.Input(
                                        id='pdm-manual-period',
                                        type='number',
                                        min=0.001,
                                        step=0.001,
                                        placeholder='Manual P [d]',
                                        className='period-manual-input',
                                        style={'font-size': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    html.Span(id='pdm-result-label', className='plot-control-status'),
                                    html.Span(id='period-search-indicator', className='plot-control-status'),
                                ], className='plot-control-group plot-period-controls plot-control-full'),
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
                                    html.Button('Recompute SED', id='rerun-stage-sed-photometry-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute Vetting', id='rerun-stage-vetting-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute External LCs', id='rerun-stage-external-lcs-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                    html.Button('Recompute Multi-survey', id='rerun-stage-multi-survey-btn', n_clicks=0,
                                                className='compact-btn', style={'fontSize': '10px'}),
                                ], style={'display': 'flex', 'gap': '6px', 'marginTop': '4px', 'flexWrap': 'wrap'}),
                                dcc.Loading(
                                    id='loading-pipeline', type='dot',
                                    children=[
                                        html.Div(id='pipeline-run-status',
                                                 style={'fontSize': '10px', 'marginTop': '2px',
                                                        'color': '#7da8c4'}),
                                    ],
                                ),
                                html.Details([
                                    html.Summary('Log', style={'cursor': 'pointer', 'marginTop': '4px'}),
                                    html.Pre(
                                        id='pipeline-module-log-panel',
                                        className='pipeline-log-panel',
                                        style={
                                            'marginTop': '6px',
                                        },
                                    ),
                                ], open=True, style={'marginTop': '4px'}),
                            ], style={'marginTop': '6px'}),
                        ], id='diagnostics-section'),
                        html.Details([
                            html.Summary('External Data', id='external-followup-summary', style={'cursor': 'pointer'}),
                            html.Div(
                                'Open panel to load external follow-up artifacts.',
                                id='external-followup-status',
                                style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                            ),
                            dcc.Loading(
                                [
                                    html.Div(
                                        id='external-followup-panel',
                                        children=_lazy_panel_placeholder('Waiting for external data panel to load.'),
                                        style={
                                            'padding': '4px 6px 6px 6px',
                                            'display': 'grid',
                                            'gridTemplateColumns': 'repeat(auto-fit, minmax(190px, 1fr))',
                                            'gap': '4px',
                                            'alignItems': 'start',
                                        },
                                    ),
                                ],
                                type='default',
                            ),
                        ], id='external-followup-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('SED', id='sed-summary', style={'cursor': 'pointer'}),
                            html.Div([
                                html.Div([
                                    html.Span('Extinction', style={'fontSize': '10px', 'color': '#7d91a6', 'marginRight': '8px'}),
                                    dcc.RadioItems(
                                        id='sed-extinction-mode',
                                        options=[
                                            {'label': ' Observed', 'value': 'observed'},
                                            {'label': ' ISM-corrected', 'value': 'corrected'},
                                            {'label': ' Both', 'value': 'both'},
                                        ],
                                        value='corrected',
                                        inline=True,
                                        inputStyle={'marginRight': '3px'},
                                        labelStyle={'fontSize': '10px', 'marginRight': '10px'},
                                        persistence=_review_persistence_token(),
                                        persistence_type='local',
                                    ),
                                    html.Button(
                                        'Export SED PDF',
                                        id='export-sed-plot',
                                        n_clicks=0,
                                        className='compact-btn',
                                        style={'marginLeft': 'auto'},
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'flexWrap': 'wrap', 'padding': '8px 10px 0 10px'}),
                                html.Div(
                                    id='sed-status',
                                    children='Open panel to load SED photometry.',
                                    style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                                ),
                                dcc.Loading(
                                    [
                                        html.Div(
                                            id='sed-plot-panel',
                                            children=_lazy_panel_placeholder('Waiting for SED panel to load.'),
                                            style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                        ),
                                    ],
                                    type='default',
                                ),
                            ]),
                        ], id='sed-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('Spectrum', id='spectrum-summary', style={'cursor': 'pointer'}),
                            html.Div([
                                html.Div([
                                    html.Span('Source', style={'fontSize': '10px', 'color': '#7d91a6', 'marginRight': '8px'}),
                                    dcc.Dropdown(
                                        id='spectrum-source-dropdown',
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        style={'minWidth': '180px', 'fontSize': '11px', 'flex': '1'},
                                    ),
                                    html.Button(
                                        'Export PDF',
                                        id='export-spectrum-plot',
                                        n_clicks=0,
                                        className='compact-btn',
                                        style={'marginLeft': 'auto'},
                                    ),
                                    html.Button(
                                        'CSV',
                                        id='export-spectrum-csv',
                                        n_clicks=0,
                                        className='compact-btn',
                                    ),
                                    html.Button(
                                        'FITS',
                                        id='export-spectrum-fits',
                                        n_clicks=0,
                                        className='compact-btn',
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'flexWrap': 'wrap', 'padding': '8px 10px 0 10px'}),
                                html.Div(
                                    id='spectrum-status',
                                    children='Open panel to load spectrum.',
                                    style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                                ),
                                dcc.Loading(
                                    [
                                        html.Div(
                                            id='spectrum-plot-panel',
                                            children=_lazy_panel_placeholder('Waiting for Spectrum panel to load.'),
                                            style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                        ),
                                    ],
                                    type='default',
                                ),
                            ]),
                        ], id='spectrum-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('DustyCult', id='dustycult-summary', style={'cursor': 'pointer'}),
                            html.Div(
                                id='dustycult-config-status',
                                children='Open panel to load DustyCult fit state.',
                                style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                            ),
                            _dustycult_controls_layout(),
                            dcc.Loading(
                                [
                                    html.Div(
                                        id='dustycult-result-panel',
                                        children=_lazy_panel_placeholder('Waiting for DustyCult panel to load.'),
                                        style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                    ),
                                ],
                                type='default',
                            ),
                        ], id='dustycult-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('PHOEBE', id='phoebe-summary', style={'cursor': 'pointer'}),
                            html.Div(
                                id='phoebe-config-status',
                                children='Open panel to load PHOEBE fit state.',
                                style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                            ),
                            _phoebe_controls_layout(),
                            dcc.Loading(
                                [
                                    html.Div(
                                        id='phoebe-result-panel',
                                        children=_lazy_panel_placeholder('Waiting for PHOEBE panel to load.'),
                                        style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                    ),
                                ],
                                type='default',
                            ),
                        ], id='phoebe-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('Candidate Panels', style={'cursor': 'pointer'}),
                            html.Div([
                                dcc.Checklist(
                                    id='round-sigfigs',
                                    options=[{'label': ' Round', 'value': 'yes'}],
                                    value=['yes'],
                                    style={'display': 'inline-block', 'font-size': '11px', 'marginRight': '6px'},
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                html.Button('Collapse all', id='toggle-meta-all', n_clicks=0, className='compact-btn'),
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'flexWrap': 'wrap'}),
                            html.Div([
                                html.Div(id='vetting-banner'),
                                html.Div(id='candidate-info-grid', className='candidate-metadata'),
                            ], style={'padding': '4px 0 8px'}),
                        ], id='candidate-panels-details', open=True, className='metadata-sections', style={'margin-top': '0'}),
                        html.Details([
                            html.Summary('Diagnostic Plots', id='diagnostic-plots-summary', style={'cursor': 'pointer'}),
                            html.Div(
                                id='diagnostic-plots-status',
                                children='Open panel to prepare diagnostic plots.',
                                style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 0 10px'},
                            ),
                            dcc.Loading(
                                [
                                    html.Div(
                                        id='diagnostic-plots-panel',
                                        children=_lazy_panel_placeholder('Waiting for diagnostic plots to load.'),
                                        style={'padding': '8px 10px', 'display': 'grid', 'gap': '8px'},
                                    ),
                                ],
                                type='default',
                            ),
                        ], id='diagnostic-plots-details', open=False, className='metadata-sections', style={'margin-top': '0'}),
                        # Run config / reproducibility
                        html.Details([
                            html.Summary('Run Config', id='run-config-summary', style={'cursor': 'pointer'}),
                            html.Div([
                                html.Div([
                                    html.Button('Copy Config JSON', id='copy-run-config-btn', n_clicks=0, className='compact-btn', style={'margin-right': '6px'}),
                                    html.Button('Download Config JSON', id='download-run-config-btn', n_clicks=0, className='compact-btn'),
                                ], style={'margin-bottom': '6px'}),
                                html.Div(
                                    'Waiting for current candidate run configuration.',
                                    id='run-config-status',
                                    style={'fontSize': '10px', 'color': '#7d91a6', 'marginBottom': '6px'},
                                ),
                                html.Div(id='run-config-panel', children=_lazy_panel_placeholder('Waiting for run config to load.')),
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
                            config=graph_config_without_image_export({
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                'responsive': True,
                            }),
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

                html.Div(
                    [
                        html.Div(id='eda-drag-handle', className='eda-drag-handle', title='Drag to resize EDA panel'),
                        html.Button('EDA', id='eda-panel-toggle', className='eda-panel-toggle', title='Show EDA panel', n_clicks=0),
                    ],
                    id='eda-splitter',
                    className='eda-splitter panel-splitter-vertical',
                    title='Drag to resize EDA panel',
                ),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div('Queue EDA', className='eda-panel-title'),
                                html.Div(id='eda-status', className='eda-status-line'),
                            ], style={'minWidth': '0'}),
                            html.Div([
                                html.Button('Wide', id='eda-expand-btn', n_clicks=0, className='compact-btn', title='Expand EDA panel'),
                                html.Button('Hide', id='eda-collapse-btn', n_clicks=0, className='compact-btn', title='Collapse EDA panel'),
                            ], className='eda-panel-actions'),
                        ], className='eda-panel-header'),
                        html.Div([
                            html.Div([
                                html.Div('X metric', className='eda-field-label'),
                                dcc.Dropdown(
                                    id='eda-x-metric',
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                            ]),
                            html.Div([
                                html.Div('Y metric', className='eda-field-label'),
                                dcc.Dropdown(
                                    id='eda-y-metric',
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                            ]),
                            html.Div([
                                html.Div('Color', className='eda-field-label'),
                                dcc.Dropdown(
                                    id='eda-color-metric',
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                            ]),
                            html.Div([
                                html.Div('Symbol', className='eda-field-label'),
                                dcc.Dropdown(
                                    id='eda-symbol-metric',
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                            ]),
                            html.Div([
                                dcc.Checklist(
                                    id='eda-log-flags',
                                    options=[
                                        {'label': ' Log X', 'value': 'logx'},
                                        {'label': ' Log Y', 'value': 'logy'},
                                    ],
                                    value=[],
                                    inline=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                            ], className='eda-control-full'),
                            html.Div([
                                dcc.Checklist(
                                    id='eda-selection-mode',
                                    options=[{'label': ' Selection filters table', 'value': 'table'}],
                                    value=[],
                                    inline=True,
                                    persistence=_review_persistence_token(),
                                    persistence_type='local',
                                ),
                                html.Button(
                                    'Clear selection',
                                    id='eda-clear-selection-btn',
                                    n_clicks=0,
                                    className='compact-btn',
                                ),
                            ], className='eda-control-full', style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                        ], className='eda-controls'),
                        html.Div([
                            html.Div([
                                html.Div('Custom Plot', className='eda-panel-title'),
                                html.Div([
                                    html.Button('Export PDF', id='eda-export-pdf-btn', n_clicks=0, className='compact-btn'),
                                    html.Div(id='eda-export-status', className='eda-status-line'),
                                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'minWidth': '0'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'gap': '8px', 'marginBottom': '6px'}),
                            html.Div([
                                dcc.Graph(
                                    id='eda-custom-graph',
                                    mathjax=True,
                                    config=graph_config_without_image_export({
                                        'displaylogo': False,
                                        'scrollZoom': True,
                                        'doubleClick': False,
                                        'responsive': True,
                                    }),
                                    style={'height': '100%', 'width': '100%', 'minWidth': '0'},
                                ),
                            ], className='eda-graph-wrap'),
                        ], className='eda-graph-card'),
                        html.Div([
                            html.Div('Candidates', className='eda-panel-title', style={'marginBottom': '6px'}),
                            dash_table.DataTable(
                                id='eda-candidate-table',
                                columns=EDA_TABLE_COLUMNS,
                                data=[],
                                page_action='native',
                                page_size=12,
                                sort_action='native',
                                sort_mode='multi',
                                filter_action='native',
                                cell_selectable=True,
                                style_table={'overflowX': 'auto', 'minHeight': '220px'},
                                style_cell={
                                    'backgroundColor': 'var(--review-table-cell-bg)',
                                    'color': 'var(--review-table-text)',
                                    'border': '1px solid var(--review-table-border)',
                                    'fontFamily': "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                                    'fontSize': '11px',
                                    'padding': '6px 7px',
                                    'maxWidth': '130px',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                },
                                style_header={
                                    'backgroundColor': 'var(--review-table-header-bg)',
                                    'color': 'var(--review-table-text)',
                                    'fontWeight': '600',
                                    'border': '1px solid var(--review-table-header-border)',
                                },
                                style_data_conditional=[],
                            ),
                        ], className='eda-table-card'),
                    ], className='eda-panel-inner'),
                ], id='eda-panel', className='eda-panel'),
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

                # Taxonomy row (clickable buttons, single-key shortcuts)
                html.Div([
                    html.Span('Morphology: ', style={'color': '#aaa', 'margin-right': '8px', 'font-size': '11px'}),
                ] + [
                    html.Button(
                        f'[{item["key"].upper()}] {item["label"]}',
                        id={'type': 'taxonomy-primary-btn', 'value': item['value']},
                        n_clicks=0,
                        className='badge-btn',
                    )
                    for item in MORPHOLOGY_PRIMARY
                ] + [
                    html.Button('[H] hypothesis', id='taxonomy-hypothesis-btn', n_clicks=0, className='badge-btn'),
                    html.Span(id='prefix-indicator', style={'margin-right': '6px', 'font-size': '11px'}),
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'margin-bottom': '6px'}),

                html.Div(id='taxonomy-submenu-panel', style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'gap': '4px', 'margin-bottom': '6px'}),
                html.Div(id='taxonomy-summary', style={'color': '#9fb6cb', 'font-size': '11px', 'margin-bottom': '6px'}),

                html.Div([
                    html.Button(
                        f'legacy {tag.replace("_", " ")}',
                        id=f'class-badge-{tag}',
                        n_clicks=0,
                        className='badge-btn',
                        style={'display': 'none'},
                    )
                    for tag in CLASS_BADGE_TAGS
                ], style={'display': 'none'}),

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
                        children=[
                            html.Span(id='bottom-pipeline-status', style={'color': '#7da8c4', 'font-size': '11px'}),
                        ],
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
            dbc.ModalBody([
                html.Pre(HELP_TEXT, style={'color': '#e0e0e0', 'margin': '0', 'font-size': '11px'}),
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-help", className="action-btn"),
            ]),
        ], id="help-modal", is_open=False),
    ], className='main-container')


app.layout = create_layout


# --- Pace Timer Callbacks ---
