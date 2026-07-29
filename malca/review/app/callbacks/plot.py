# This file was mechanically split from malca.review.app; preserve behavior when editing.
NATIVE_PLOT_STYLE = {'display': 'block', 'width': '100%', 'height': '100%'}
AUTO_FALLBACK_PERIOD_METHOD = 'pdm'


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
    if isinstance(result, dict):
        result = dict(result)
        result.setdefault('candidate_id', candidate_id)
        result.setdefault('search_method', method)
        result.setdefault('source', str(result.get('method') or method.upper()))
        result.setdefault('auto', False)
        result.setdefault('min_period', min_p)
        result.setdefault('max_period', max_p)
        _store_period_cache_entry(
            auto_period_cache,
            candidate_id=candidate_id,
            method=method,
            min_period=min_p,
            max_period=max_p,
            result=result,
            label=label,
        )
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
        max_p = float(max_period) if max_period else 10.0
    except (TypeError, ValueError):
        min_p, max_p = 0.1, 10.0
    if min_p <= 0:
        min_p = 0.01
    if max_p <= min_p:
        max_p = min_p + 1.0
    return min_p, max_p


def _payload_baseline_days(payload: dict | None) -> float | None:
    """Derive baseline span from candidate payload stats when available."""
    payload = payload or {}
    for key in ("stats_time_span_days", "baseline_days", "period_baseline_days"):
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0:
            return float(value)
    try:
        jd_start = float(payload.get("stats_jd_start"))
        jd_end = float(payload.get("stats_jd_end"))
    except (TypeError, ValueError):
        return None
    if np.isfinite(jd_start) and np.isfinite(jd_end) and jd_end > jd_start:
        return float(jd_end - jd_start)
    return None


def _adaptive_review_period_bounds(
    payload: dict | None,
    *,
    long_p: bool = False,
    user_min: float | None = None,
    user_max: float | None = None,
) -> tuple[float, float]:
    """Return review-stage adaptive (min, max) period bounds for a candidate."""
    from malca.core.period_bounds import STAGE_LONG, STAGE_REVIEW, adaptive_period_bounds

    baseline = _payload_baseline_days(payload)
    stage = STAGE_LONG if long_p else STAGE_REVIEW
    bounds = adaptive_period_bounds(
        baseline_days=baseline,
        stage=stage,
        user_min_period=user_min,
        user_max_period=user_max,
    )
    return bounds.as_tuple()


def _long_p_enabled(toggle_value: object) -> bool:
    if isinstance(toggle_value, (list, tuple, set)):
        return "long" in {str(v) for v in toggle_value}
    return bool(toggle_value)

def _period_cache_key(candidate_id: object, method: object, min_period: float, max_period: float, base_period: object = None) -> str:
    method_name = str(method or 'pdm').strip().lower()
    key = f"{str(candidate_id)}|{method_name}|{float(min_period):.12g}|{float(max_period):.12g}"
    if method_name == 'alias_check' or base_period is not None:
        try:
            base = float(base_period)
        except (TypeError, ValueError):
            base = np.nan
        base_text = f"{base:.12g}" if np.isfinite(base) else ""
        key = f"{key}|base={base_text}"
    return key


def _period_cache_entry(
    auto_period_cache: dict | None,
    *,
    candidate_id: object,
    method: object,
    min_period: float,
    max_period: float,
    base_period: object = None,
) -> dict | None:
    cache = dict(auto_period_cache or {})
    key = _period_cache_key(candidate_id, method, min_period, max_period, base_period)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    if str(entry.get('candidate_id') or '') != str(candidate_id):
        return None
    if str(entry.get('method') or '').strip().lower() != str(method or 'pdm').strip().lower():
        return None
    try:
        entry_min = float(entry.get('min_period'))
        entry_max = float(entry.get('max_period'))
    except (TypeError, ValueError):
        return None
    if not (np.isclose(entry_min, float(min_period)) and np.isclose(entry_max, float(max_period))):
        return None
    if base_period is not None:
        try:
            entry_base = float(entry.get('base_period'))
            expected_base = float(base_period)
        except (TypeError, ValueError):
            return None
        if not np.isclose(entry_base, expected_base):
            return None
    return entry


def _store_period_cache_entry(
    auto_period_cache: dict,
    *,
    candidate_id: object,
    method: object,
    min_period: float,
    max_period: float,
    result: dict | None,
    label: str,
    base_period: object = None,
) -> None:
    method_name = str(method or 'pdm').strip().lower()
    key = _period_cache_key(candidate_id, method_name, min_period, max_period, base_period)
    auto_period_cache[key] = {
        'candidate_id': str(candidate_id),
        'method': method_name,
        'min_period': float(min_period),
        'max_period': float(max_period),
        'result': result,
        'label': str(label or ''),
    }
    if base_period is not None:
        try:
            auto_period_cache[key]['base_period'] = float(base_period)
        except (TypeError, ValueError):
            pass


def _pending_auto_harmonic_result(candidate_id: object, min_period: float, max_period: float, base_period: float, base_source: str) -> dict:
    return {
        'pending': True,
        'auto': True,
        'candidate_id': str(candidate_id),
        'method': 'Harmonic check',
        'search_method': 'alias_check',
        'source': 'Auto harmonic check',
        'min_period': float(min_period),
        'max_period': float(max_period),
        'base_period': float(base_period),
        'base_period_source': str(base_source or 'stored period'),
    }


def _failed_auto_harmonic_result(candidate_id: object, min_period: float, max_period: float, base_period: object, label: str) -> dict:
    result = {
        'auto': True,
        'candidate_id': str(candidate_id),
        'method': 'Harmonic check',
        'search_method': 'alias_check',
        'source': 'Auto harmonic check',
        'min_period': float(min_period),
        'max_period': float(max_period),
        'error': str(label or 'No valid harmonic candidate'),
    }
    try:
        result['base_period'] = float(base_period)
    except (TypeError, ValueError):
        pass
    return result


def _pending_auto_period_search_result(candidate_id: object, min_period: float, max_period: float, method: str) -> dict:
    method_name = str(method or AUTO_FALLBACK_PERIOD_METHOD).strip().lower()
    method_label = method_name.upper()
    return {
        'pending': True,
        'auto': True,
        'candidate_id': str(candidate_id),
        'method': method_label,
        'search_method': method_name,
        'source': f"Auto {method_label}",
        'min_period': float(min_period),
        'max_period': float(max_period),
        'reason': 'no stored period',
    }


def _failed_auto_period_search_result(candidate_id: object, min_period: float, max_period: float, method: str, label: str) -> dict:
    method_name = str(method or AUTO_FALLBACK_PERIOD_METHOD).strip().lower()
    method_label = method_name.upper()
    return {
        'auto': True,
        'candidate_id': str(candidate_id),
        'method': method_label,
        'search_method': method_name,
        'source': f"Auto {method_label}",
        'min_period': float(min_period),
        'max_period': float(max_period),
        'error': str(label or 'No valid period'),
        'reason': 'no stored period',
    }


def _auto_period_search_label(method: str, label: str) -> str:
    method_name = str(method or AUTO_FALLBACK_PERIOD_METHOD).strip().lower()
    method_label = method_name.upper()
    clean_label = str(label or '').strip()
    if clean_label.lower().startswith('auto '):
        return clean_label
    if clean_label.startswith(f"{method_label}:"):
        return f"Auto {method_label}:{clean_label[len(method_label) + 1:]}"
    return f"Auto {method_label}: {clean_label}" if clean_label else f"Auto {method_label}: no result"


@app.callback(
    [Output('pdm-min-period', 'value', allow_duplicate=True),
     Output('pdm-max-period', 'value', allow_duplicate=True)],
    [Input('current-candidate-id', 'data'),
     Input('period-long-p-toggle', 'value')],
    prevent_initial_call=True,
)
def sync_adaptive_period_bounds_on_navigate(candidate_id, long_p_toggle):
    """Set review period-search bounds from the candidate baseline (or long-P)."""
    if candidate_id is None:
        raise dash.exceptions.PreventUpdate
    payload, _stored_lc_path, _source_path = _candidate_context(str(candidate_id))
    min_p, max_p = _adaptive_review_period_bounds(
        payload,
        long_p=_long_p_enabled(long_p_toggle),
    )
    return float(min_p), float(max_p)


@app.callback(
    [Output('pdm-result-store', 'data', allow_duplicate=True),
     Output('pdm-result-label', 'children', allow_duplicate=True),
     Output('pdm-manual-period', 'value', allow_duplicate=True),
     Output('auto-period-cache', 'data', allow_duplicate=True),
     Output('auto-period-request', 'data', allow_duplicate=True)],
    [Input('current-candidate-id', 'data'),
     Input('pdm-min-period', 'value'),
     Input('pdm-max-period', 'value'),
     Input('period-long-p-toggle', 'value')],
    [State('auto-period-cache', 'data'),
     State('auto-period-request', 'data')],
    prevent_initial_call=True,
)
def auto_period_on_navigate(candidate_id, min_period, max_period, long_p_toggle, auto_period_cache, auto_period_request):
    """Queue harmonic checks for stored periods, or a fallback search when no period exists."""
    if candidate_id is None:
        return None, '', None, no_update, {'nonce': 0}
    candidate_id = str(candidate_id)
    auto_period_cache = dict(auto_period_cache or {})
    auto_period_request = dict(auto_period_request or {})
    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
    long_p = _long_p_enabled(long_p_toggle)
    if min_period is None or max_period is None:
        min_p, max_p = _adaptive_review_period_bounds(payload, long_p=long_p)
    else:
        min_p, max_p = _normalize_period_search_bounds(min_period, max_period)
        if long_p:
            min_long, max_long = _adaptive_review_period_bounds(payload, long_p=True)
            if max_p < max_long:
                max_p = float(max_long)
            if min_p > min_long:
                min_p = float(min_long)
    base_period, base_source = shared_resolve_stored_review_period(payload)
    if base_period is None:
        method = AUTO_FALLBACK_PERIOD_METHOD
        cached_entry = _period_cache_entry(
            auto_period_cache,
            candidate_id=candidate_id,
            method=method,
            min_period=min_p,
            max_period=max_p,
        )
        if isinstance(cached_entry, dict):
            return (
                cached_entry.get('result'),
                str(cached_entry.get('label', '')),
                None,
                no_update,
                no_update,
            )
        request = {
            'nonce': int(auto_period_request.get('nonce', 0) or 0) + 1,
            'candidate_id': candidate_id,
            'min_period': min_p,
            'max_period': max_p,
            'method': method,
            'reason': 'no stored period',
        }
        return (
            _pending_auto_period_search_result(candidate_id, min_p, max_p, method),
            f"No stored period; running auto {method.upper()}...",
            None,
            no_update,
            request,
        )

    cached_entry = _period_cache_entry(
        auto_period_cache,
        candidate_id=candidate_id,
        method='alias_check',
        min_period=min_p,
        max_period=max_p,
        base_period=base_period,
    )
    if isinstance(cached_entry, dict):
        return (
            cached_entry.get('result'),
            str(cached_entry.get('label', '')),
            None,
            no_update,
            no_update,
        )

    request = {
        'nonce': int(auto_period_request.get('nonce', 0) or 0) + 1,
        'candidate_id': candidate_id,
        'min_period': min_p,
        'max_period': max_p,
        'method': 'alias_check',
        'base_period': float(base_period),
        'base_period_source': str(base_source or 'stored period'),
    }
    return (
        _pending_auto_harmonic_result(candidate_id, min_p, max_p, float(base_period), str(base_source or 'stored period')),
        'Auto harmonic check: checking aliases...',
        None,
        no_update,
        request,
    )


def run_auto_period_search(auto_period_request, auto_period_cache):
    """Run the automatic harmonic check or missing-period fallback search."""
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
    method = str(auto_period_request.get('method') or 'alias_check').strip().lower()
    try:
        base_period = float(auto_period_request.get('base_period'))
    except (TypeError, ValueError):
        base_period = np.nan

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
    if method == 'alias_check':
        result, label = _run_harmonic_check_for_payload(
            payload,
            min_period=min_p,
            max_period=max_p,
        )
        if isinstance(result, dict):
            result = dict(result)
            result['auto'] = True
            result['candidate_id'] = candidate_id
            result['search_method'] = 'alias_check'
            result['source'] = 'Auto harmonic check'
            result['min_period'] = min_p
            result['max_period'] = max_p
            base_period = float(result.get('base_period', base_period))
        else:
            result = _failed_auto_harmonic_result(candidate_id, min_p, max_p, base_period, label)
    else:
        method = AUTO_FALLBACK_PERIOD_METHOD
        result, label = _run_period_search_for_payload(
            payload,
            min_period=min_p,
            max_period=max_p,
            method=method,
        )
        label = _auto_period_search_label(method, label)
        if isinstance(result, dict):
            result = dict(result)
            result['auto'] = True
            result['candidate_id'] = candidate_id
            result['search_method'] = method
            result['source'] = f"Auto {method.upper()}"
            result['min_period'] = min_p
            result['max_period'] = max_p
            result['reason'] = 'no stored period'
        else:
            result = _failed_auto_period_search_result(candidate_id, min_p, max_p, method, label)
    cache_base_period = base_period if method == 'alias_check' else None
    _store_period_cache_entry(
        auto_period_cache,
        candidate_id=candidate_id,
        method=method,
        min_period=min_p,
        max_period=max_p,
        result=result,
        label=label,
        base_period=cache_base_period,
    )
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
     Output('external-source-values', 'value', allow_duplicate=True)],
    [Input('plot-preset', 'value'),
     Input('plot-reset-btn', 'n_clicks'),
     Input('cams-all-btn', 'n_clicks'),
     Input('cams-clear-btn', 'n_clicks'),
     Input('cams-invert-btn', 'n_clicks'),
     Input('sources-all-btn', 'n_clicks'),
     Input('sources-native-btn', 'n_clicks'),
     Input('sources-clear-btn', 'n_clicks')],
    [State('camera-checklist', 'options'),
     State('camera-checklist', 'value'),
     State('plot-overlays', 'value'),
     State('external-source-values', 'value')],
    prevent_initial_call=True,
)
def update_plot_controls(preset, n_reset, n_all, n_clear, n_invert, n_sources_all, n_sources_native, n_sources_clear, camera_options, camera_values, overlay_values, source_values):
    """Preset mapping plus camera/source selection action buttons."""
    _ = n_reset, n_all, n_clear, n_invert, n_sources_all, n_sources_native, n_sources_clear
    trig = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else ''
    cams = [str(opt.get('value')) for opt in (camera_options or [])]
    selected = [str(v) for v in (camera_values or []) if str(v) in cams]
    overlays = list(overlay_values or [])
    _ = source_values

    if trig == 'plot-preset' or trig == 'plot-reset-btn':
        cfg = PLOT_PRESETS.get(preset or 'Diagnostics', PLOT_PRESETS['Diagnostics'])
        new_overlays = list(cfg['overlays'])
        new_cams = list(cams)
        return new_overlays, new_cams, ['g', 'V'], list(DEFAULT_EXTERNAL_SOURCE_VALUES)
    if trig == 'cams-all-btn':
        return overlays, list(cams), no_update, no_update
    if trig == 'cams-clear-btn':
        return overlays, [], no_update, no_update
    if trig == 'cams-invert-btn':
        inv = [c for c in cams if c not in set(selected)]
        return overlays, inv, no_update, no_update
    if trig == 'sources-all-btn':
        return overlays, no_update, no_update, list(EXTERNAL_SOURCE_VALUES)
    if trig == 'sources-native-btn':
        return overlays, no_update, no_update, ['asassn']
    if trig == 'sources-clear-btn':
        return overlays, no_update, no_update, []
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
    link_radius = float(state.get('link_radius', 30.0))
    yaxis_mode = str(state.get('yaxis_mode', 'mag') or 'mag')
    native_color_mode = 'band' if str(state.get('native_color_mode', 'camera') or 'camera') == 'band' else 'camera'
    phase_panel_mode = _coerce_choice(state.get('phase_panel_mode'), {'fold', 'time'}, 'fold')
    external_source_values = normalize_external_source_values(
        state.get('external_source_values', state.get('external_source_view', DEFAULT_EXTERNAL_SOURCE_VIEW)),
        default=list(DEFAULT_EXTERNAL_SOURCE_VALUES),
    )
    external_source_layout = normalize_external_source_layout(state.get('external_source_layout'))
    override_period_source = str(state.get('override_period_source') or 'manual/search')
    phase_period_pending = bool(state.get('phase_period_pending', False))
    phase_period_pending_source = str(state.get('phase_period_pending_source') or '')
    suppress_catalog_phase_period = bool(state.get('suppress_catalog_phase_period', False))
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
        return '', 'No candidates in queue', _render_metadata_health(None, context_msg='Queue is empty.'), _render_vetting_banner(None, radius_arcsec=link_radius), '[0/0]', empty_fig, NATIVE_PLOT_STYLE, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'No candidates in queue.', []), _render_camera_diag_panel({}, []), 'No run configuration: queue is empty.', _render_run_config_panel(None, None, ['Queue is empty']), _render_repro_badge(None, ['Queue is empty']), '', nonce

    if idx < 0 or idx >= queue_size:
        return '', 'Invalid index', _render_metadata_health(None, context_msg='Invalid queue index.'), _render_vetting_banner(None, radius_arcsec=link_radius), f'[{idx}/{queue_size}]', empty_fig, NATIVE_PLOT_STYLE, {'display': 'none'}, [], [], _render_plot_status_panel('error', 'Invalid queue index.', []), _render_camera_diag_panel({}, []), 'No run configuration: invalid queue index.', _render_run_config_panel(None, None, ['Invalid queue index']), _render_repro_badge(None, ['Invalid queue index']), '', nonce

    payload, stored_lc_path, source_path = _candidate_context(candidate_id)

    plot_dir_path = _review_plot_dir_for_context(source_path)
    plot_search_root = _plot_search_root_for_payload(payload)

    grouped = extract_review_metadata_grouped(payload, round_sigfigs=round_sigfigs)
    feature_rows = extract_review_metadata_feature_rows(payload, round_sigfigs=round_sigfigs)
    metadata_health = _render_metadata_health(grouped)
    vetting_banner = _render_vetting_banner(payload, radius_arcsec=link_radius)
    grid_items = _render_metadata_review_layout(payload, grouped, [], feature_rows)

    progress = f"[{idx + 1}/{queue_size}]"

    if plot_mode == 'png':
        run_params, run_params_status, run_params_msg = _load_run_params_meta_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
        run_params_path = _run_params_path_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
        mismatch_warnings = _run_config_mismatch_warnings(run_params if run_params else None, overlays)
        if run_params_status != 'loaded':
            mismatch_warnings.append(run_params_msg)

        stat_rows = _build_stat_rows(payload, pd.DataFrame(), set())
        merged_grid = _render_metadata_review_layout(payload, grouped, stat_rows, feature_rows)

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
        run_config_status = run_params_msg if run_params_status != 'loaded' else f"Loaded run configuration from {run_params_path}"
        return (
            png_src,
            merged_grid,
            metadata_health,
            vetting_banner,
            progress,
            no_update,
            {'display': 'none'},
            NATIVE_PLOT_STYLE,
            no_update,
            [],
            _render_plot_status_panel('ok', png_msg, mismatch_warnings),
            _render_camera_diag_panel({}, []),
            run_config_status,
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
    axis_convention_revision = "mag-dimming-down-v1"
    uirevision_key = f"{candidate_id}|{','.join(sorted(str(c) for c in selected_cameras))}|{','.join(sorted(str(b) for b in selected_bands))}|{native_color_mode}|{theme_mode}|{residual_height:.3f}|{baseline_opacity:.2f}|{yaxis_mode}|{phase_panel_mode}|{','.join(external_source_values)}|{external_source_layout}|{axis_convention_revision}"

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
            override_period_source=override_period_source,
            phase_period_pending=phase_period_pending,
            phase_period_pending_source=phase_period_pending_source,
            suppress_catalog_phase_period=suppress_catalog_phase_period,
            show_diagnostics='diagnostics' in overlays,
            confidence_colors='confidence' in overlays,
            run_params=run_params or {},
            uirevision_key=uirevision_key,
            theme=theme_mode,
            residual_fraction=residual_height,
            baseline_opacity=baseline_opacity,
            yaxis_mode=yaxis_mode,
            native_color_mode=native_color_mode,
            external_source_view=external_source_values,
            external_panel_mode=external_source_layout,
            candidate_id=candidate_id,
        )
    except Exception as exc:

        traceback.print_exc()
        panel = _render_run_config_panel(run_params if run_params else None, run_params_path, [str(exc)])
        run_config_status = run_params_msg if run_params_status != 'loaded' else f"Loaded run configuration from {run_params_path}"
        plot_src = _candidate_plot_src(payload)
        if plot_src:
            return (
                plot_src, grid_items, metadata_health, vetting_banner, progress, no_update,
                {'display': 'none'},
                NATIVE_PLOT_STYLE,
                [], [],
                _render_plot_status_panel('error', f'Native plot error: {exc}', mismatch_warnings),
                _render_camera_diag_panel({}, []),
                run_config_status,
                panel,
                _render_repro_badge(None, [str(exc)]),
                '', nonce,
            )
        return (
            '', grid_items, metadata_health, vetting_banner, progress, empty_fig,
            NATIVE_PLOT_STYLE,
            {'display': 'none'},
            [], [],
            _render_plot_status_panel('error', f'Native plot error: {exc}', mismatch_warnings),
            _render_camera_diag_panel({}, []),
            run_config_status,
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
            NATIVE_PLOT_STYLE,
            [],
            [],
            _render_plot_status_panel('warn', f"{fallback_msg} Showing PNG fallback.", fallback_warnings),
            _render_camera_diag_panel(native.get('camera_diagnostics', {}), []),
            run_params_msg if run_params_status != 'loaded' else f"Loaded run configuration from {run_params_path}",
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

    # Merge stats into the streamlined metadata layout.
    stat_rows = native['stat_rows']
    merged_grid = _render_metadata_review_layout(payload, grouped, stat_rows, feature_rows)

    return (
        '',
        merged_grid,
        metadata_health,
        vetting_banner,
        progress,
        native['figure'],
        NATIVE_PLOT_STYLE,
        {'display': 'none'},
        native['camera_options'],
        [],  # stats merged into candidate-info-grid
        _render_plot_status_panel(native.get('status', 'ok'), native.get('status_message', ''), (native_warnings + mismatch_warnings)),
        _render_camera_diag_panel(native.get('camera_diagnostics', {}), filtered),
        run_params_msg if run_params_status != 'loaded' else f"Loaded run configuration from {run_params_path}",
        _render_run_config_panel(run_params if run_params else None, run_params_path, run_config_warnings),
        _render_repro_badge(run_params if run_params else None, run_config_warnings),
        json.dumps(run_params, indent=2, sort_keys=True) if run_params else '',
        nonce,
    )


update_display = _review_perf_wrapped('update_display', update_display)


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
    Output('run-config-status', 'children'),
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
    [Output('external-followup-panel', 'children'),
     Output('external-followup-status', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data'),
     Input('external-followup-summary', 'n_clicks')],
    [State('cutout-selected-survey', 'data')],
    prevent_initial_call=False,
)
def update_external_followup_panel(candidate_id, theme_mode, panel_requested=True, selected_cutout_survey=None):
    """Render external follow-up artifacts for the current candidate."""
    if not _details_open(panel_requested):
        return no_update, no_update
    if not candidate_id:
        return _lazy_panel_placeholder("No candidates loaded.", "error"), "No candidates loaded."
    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    return (
        _render_external_followup(
            payload,
            str(candidate_id),
            str(theme_mode or DEFAULT_THEME),
            selected_cutout_survey=selected_cutout_survey,
        ),
        f"Loaded external data for {candidate_id}.",
    )


@app.callback(
    [Output('cutout-image', 'src'),
     Output('cutout-source-link', 'href'),
     Output('cutout-source-link', 'title'),
     Output('cutout-status', 'children'),
     Output('cutout-selected-survey', 'data')],
    [Input('cutout-survey-select', 'value')],
    [State('current-candidate-id', 'data'),
     State('cutout-selected-survey', 'data')],
    prevent_initial_call=True,
)
def update_survey_cutout(selected_survey, candidate_id, stored_survey):
    """Update the live survey cutout URL when the selected survey changes."""
    selected_key = selected_survey or stored_survey or DEFAULT_CUTOUT_SURVEY_KEY
    if not candidate_id:
        return "", "#", "", "No candidates loaded.", selected_key

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
    triggered_id = getattr(callback_context, "triggered_id", None)
    cutout_data = cutout_payload_for_candidate(
        payload,
        selected_key=selected_key,
        prefer_compatible=(triggered_id != 'cutout-survey-select'),
    )
    image_url = str(cutout_data.get("image_url") or "")
    source_url = str(cutout_data.get("source_url") or "#")
    return (
        image_url,
        source_url,
        source_url if image_url else "",
        str(cutout_data.get("message") or ""),
        str(cutout_data.get("selected_key") or selected_key),
    )
