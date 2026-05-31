# This file was mechanically split from malca.review.app; preserve behavior when editing.
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


if _background_callback_manager is not None and _UI_BACKGROUND_CALLBACKS:
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


if _background_callback_manager is not None and _UI_BACKGROUND_CALLBACKS:
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

    plot_dir_path = _review_plot_dir_for_context()
    run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
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
    phase_panel_mode = _coerce_choice(state.get('phase_panel_mode'), {'fold', 'time'}, 'fold')
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

    plot_dir_path = _review_plot_dir_for_context(source_path)
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
        run_params, run_params_status, run_params_msg = _load_run_params_meta_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
        run_params_path = _run_params_path_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
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

    run_params, run_params_status, run_params_msg = _load_run_params_meta_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
    run_params_path = _run_params_path_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
    mismatch_warnings = _run_config_mismatch_warnings(run_params if run_params else None, overlays)
    if run_params_status != 'loaded':
        mismatch_warnings.append(run_params_msg)
    uirevision_key = f"{candidate_id}|{','.join(sorted(str(c) for c in selected_cameras))}|{','.join(sorted(str(b) for b in selected_bands))}|{theme_mode}|{residual_height:.3f}|{baseline_opacity:.2f}|{yaxis_mode}|{phase_panel_mode}|{external_source_view}"

    # Discover external LC parquets only when the user explicitly asks for them.
    ext_lcs: dict[str, Path] | None = None
    requested_external_source = str(external_source_view or DEFAULT_EXTERNAL_SOURCE_VIEW).strip().lower()
    if requested_external_source not in {'', 'asassn'}:
        run_dir = _resolve_run_dir_from_plot_dir(str(plot_dir_path) if plot_dir_path else None)
        lk = _candidate_lookup_keys(candidate_id, payload)
        found: dict[str, Path] = {}
        search_roots: list[Path] = []
        if run_dir is not None:
            search_roots.append(run_dir / "results")
        default_results_root = _APP_REPO_ROOT / "output" / "results"
        if default_results_root not in search_roots:
            search_roots.append(default_results_root)
        for prefix in ("atlas", "ztf", "gaia_epoch", "tess", "ps1", "crts"):
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
            phase_panel_mode=phase_panel_mode,
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


if _background_callback_manager is not None and _UI_BACKGROUND_CALLBACKS:
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
    [Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data')],
    prevent_initial_call=False,
)
def update_external_followup_panel(candidate_id, theme_mode):
    """Render external follow-up artifacts for the current candidate."""
    if not candidate_id:
        return html.Div("No candidates loaded.", style={'font-size': '11px', 'color': '#c77'})
    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    return _render_external_followup(payload, str(candidate_id), str(theme_mode or DEFAULT_THEME))


