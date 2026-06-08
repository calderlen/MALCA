# This file was mechanically split from malca.review.app; preserve behavior when editing.
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


def _coerce_select_filter_mode_value(raw_value: object) -> str:
    return 'include' if 'include' in _coerce_string_list(raw_value) else 'exclude'


def _select_filter_mode_checklist_value(raw_value: object) -> list[str]:
    return ['include'] if _coerce_select_filter_mode_value(raw_value) == 'include' else []


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
    external_source_values: object = None,
    external_source_layout: object = None,
    external_source_view: object = None,
    camera_values: object = None,
    band_values: object = None,
    yaxis_mode: object = None,
    phase_panel_mode: object = None,
    period_method: object = None,
    pdm_min_period: object = None,
    pdm_max_period: object = None,
    pdm_manual_period: object = None,
) -> dict[str, object]:
    overlay_allowed = {'raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras', 'diagnostics', 'confidence'}
    source_values = normalize_external_source_values(
        external_source_values if external_source_values is not None else external_source_view,
        default=list(DEFAULT_EXTERNAL_SOURCE_VALUES),
    )
    source_layout = normalize_external_source_layout(external_source_layout)
    return {
        'theme_mode': _coerce_choice(theme_mode, {'black', 'gray', 'white'}, DEFAULT_THEME),
        'plot_mode': _coerce_choice(plot_mode, {'native', 'png'}, 'native'),
        'plot_overlays': [value for value in _coerce_string_list(plot_overlays) if value in overlay_allowed],
        'baseline_opacity': _coerce_numeric_input_value(baseline_opacity),
        'residual_height': _coerce_numeric_input_value(residual_height),
        'external_source_values': source_values,
        'external_source_layout': source_layout,
        'external_source_view': legacy_external_source_view(source_values),
        'camera_values': _coerce_string_list(camera_values),
        'band_values': _coerce_string_list(band_values) or ['g', 'V'],
        'yaxis_mode': _coerce_choice(yaxis_mode, {'mag', 'flux'}, 'mag'),
        'phase_panel_mode': _coerce_choice(phase_panel_mode, {'fold', 'time'}, 'fold'),
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
        external_source_values=raw_state.get('external_source_values'),
        external_source_layout=raw_state.get('external_source_layout'),
        external_source_view=raw_state.get('external_source_view'),
        camera_values=raw_state.get('camera_values'),
        band_values=raw_state.get('band_values'),
        yaxis_mode=raw_state.get('yaxis_mode'),
        phase_panel_mode=raw_state.get('phase_panel_mode'),
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
        'select_filter_mode': _coerce_select_filter_mode_value(next(it)),
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
        'select_filter_mode': _coerce_select_filter_mode_value(raw_state.get('select_filter_mode')),
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
        _select_filter_mode_checklist_value(state['select_filter_mode']),
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
    [Output('select-filter-mode', 'value', allow_duplicate=True)]
    + [Output(f'{_col_id(col)}-mode', 'value') for col in VETTING_KNOWN_BOOL_FILTERS]
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


def _normalized_queue_filter_ui_state(
    ui_state: dict[str, object],
    bounds_data: dict[str, dict[str, float | None]] | None,
) -> dict[str, object]:
    """Normalize raw queue-filter UI state before persisting it."""
    normalized = dict(ui_state)
    raw_numeric_filters = {
        fkey: normalized.get(fkey)
        for _, fkey in _NUM_INPUT_STATES
    }
    normalized.update(_normalize_numeric_filter_inputs(bounds_data, raw_numeric_filters))
    return normalized


def _queue_filter_params_from_ui_state(
    ui_state: dict[str, object],
    numeric_bounds: dict[str, dict[str, float | None]] | None,
    queue_source_scope: object,
) -> dict[str, object]:
    filter_params: dict[str, object] = {
        'only_unreviewed': 'yes' in _coerce_yes_checklist_value(ui_state.get('filter_unreviewed')),
        'require_failed_any_false': 'yes' in _coerce_yes_checklist_value(ui_state.get('filter_failed')),
        'select_filter_mode': _coerce_select_filter_mode_value(ui_state.get('select_filter_mode')),
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


if _background_callback_manager is not None and _UI_BACKGROUND_CALLBACKS:
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


def _fresh_vetting_known_select_options(
    queue_source_scope,
    current_options: dict[str, list[dict[str, object]] | None],
) -> dict[str, list[dict[str, object]] | None]:
    """Return DB-backed known-type options for the active queue scope."""
    refreshed_options = _load_vetting_known_select_options(queue_source_scope)
    resolved = dict(current_options)
    for col in VETTING_KNOWN_SELECT_FILTERS:
        resolved[col] = refreshed_options.get(col, [])
    return resolved


if _background_callback_manager is not None and _UI_BACKGROUND_CALLBACKS:
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
    [Input('vetting-known-types-btn', 'n_clicks'),
     Input('vetting-definite-known-types-btn', 'n_clicks')],
    _VETTING_KNOWN_FILTER_OPTION_STATES,
    prevent_initial_call=True,
)
def apply_vetting_known_type_filters(known_clicks, definite_clicks, *select_options):
    """Apply the broad or definite-only known-type vetting preset."""
    triggered_id = getattr(callback_context, 'triggered_id', None)
    if triggered_id == 'vetting-known-types-btn':
        if not known_clicks:
            raise dash.exceptions.PreventUpdate
        include_uncertain = True
    elif triggered_id == 'vetting-definite-known-types-btn':
        if not definite_clicks:
            raise dash.exceptions.PreventUpdate
        include_uncertain = False
    else:
        raise dash.exceptions.PreventUpdate

    queue_source_scope, *option_lists = select_options
    select_options_by_col = dict(zip(VETTING_KNOWN_SELECT_FILTERS, option_lists))
    select_options_by_col = _fresh_vetting_known_select_options(
        queue_source_scope,
        select_options_by_col,
    )
    bool_values, select_values = _vetting_known_filter_preset(
        select_options_by_col,
        include_uncertain=include_uncertain,
    )
    return ([], *bool_values, *select_values)


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
    [*[Output(cid, 'value', allow_duplicate=True) for cid, _ in _TEXT_STATES],
     *[Output(cid, 'value', allow_duplicate=True) for cid, _ in _SELECT_STATES]],
    Input('restored-filter-applied', 'data'),
    _TEXT_OPTION_INPUTS,
    _SELECT_OPTION_INPUTS,
    _TEXT_VALUE_STATES,
    _SELECT_VALUE_STATES,
    prevent_initial_call='initial_duplicate',
)
def rehydrate_saved_text_select_filter_values(restore_state, *callback_values):
    """Reapply saved dropdown selections once their option lists finish hydrating."""
    if not isinstance(restore_state, dict) or not restore_state.get('restored'):
        raise dash.exceptions.PreventUpdate

    saved_ui_state = _normalize_saved_queue_filter_ui_state(restore_state.get('saved_ui_state'))
    if saved_ui_state is None:
        raise dash.exceptions.PreventUpdate

    n_text_opts = len(_TEXT_OPTION_INPUTS)
    n_select_opts = len(_SELECT_OPTION_INPUTS)
    n_text_values = len(_TEXT_VALUE_STATES)

    text_option_values = callback_values[:n_text_opts]
    select_option_values = callback_values[n_text_opts:n_text_opts + n_select_opts]
    text_current_values = callback_values[n_text_opts + n_select_opts:n_text_opts + n_select_opts + n_text_values]
    select_current_values = callback_values[n_text_opts + n_select_opts + n_text_values:]

    outputs: list[object] = []
    changed = False

    for ((_, fkey), options, current) in zip(_TEXT_STATES, text_option_values, text_current_values):
        current_value = _coerce_text_filter_value(current)
        saved_value = _coerce_text_filter_value(saved_ui_state.get(fkey))
        option_values = {
            str(option.get('value'))
            for option in (options or [])
            if isinstance(option, dict) and option.get('value') is not None
        }
        if saved_value != 'Any' and current_value == 'Any' and saved_value in option_values:
            outputs.append(saved_value)
            changed = True
        else:
            outputs.append(no_update)

    for ((_, fkey), options, current) in zip(_SELECT_STATES, select_option_values, select_current_values):
        current_value = _coerce_string_list(current)
        saved_value = _coerce_string_list(saved_ui_state.get(fkey))
        option_values = {
            str(option.get('value'))
            for option in (options or [])
            if isinstance(option, dict) and option.get('value') is not None
        }
        restored_values = [value for value in saved_value if value in option_values]
        if restored_values and not current_value:
            outputs.append(restored_values)
            changed = True
        else:
            outputs.append(no_update)

    if not changed:
        raise dash.exceptions.PreventUpdate

    return tuple(outputs)


@app.callback(
    Output('filter-params', 'data'),
    _FILTER_VALUE_INPUTS,
    State('restored-filter-applied', 'data'),
    State('numeric-filter-bounds', 'data'),
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
    numeric_bounds = callback_values[n_values + 1]
    text_option_values = callback_values[n_values + 2:n_values + 2 + n_text_opts]
    select_option_values = callback_values[n_values + 2 + n_text_opts:]
    if not isinstance(restore_state, dict) or not restore_state.get('ready'):
        raise dash.exceptions.PreventUpdate
    ui_state = _queue_filter_ui_state_from_values(*value_states)
    ui_state = _merge_unhydrated_saved_queue_filter_ui_state(
        ui_state,
        restore_state,
        text_option_values,
        select_option_values,
    )
    ui_state = _normalized_queue_filter_ui_state(ui_state, numeric_bounds)
    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            save_app_state(conn, _QUEUE_FILTER_APP_STATE_KEY, json.dumps(ui_state, default=str))
    except Exception as exc:
        print(f"[filters] Warning: could not persist queue filters: {exc}")
    return ui_state


@app.callback(
    Output('saved-review-gui-state', 'data'),
    Input('startup-lazy-init', 'n_intervals'),
    prevent_initial_call=True,
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
     Output('external-source-values', 'value', allow_duplicate=True),
     Output('external-source-layout', 'value', allow_duplicate=True),
     Output('camera-checklist', 'value', allow_duplicate=True),
     Output('band-checklist', 'value', allow_duplicate=True),
     Output('yaxis-mode', 'value', allow_duplicate=True),
     Output('phase-panel-mode', 'value', allow_duplicate=True),
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
        return tuple([no_update] * 16)
    return (
        state['plot_mode'],
        state['plot_overlays'],
        state['baseline_opacity'],
        state['residual_height'],
        state['external_source_values'],
        state['external_source_layout'],
        state['camera_values'],
        state['band_values'],
        state['yaxis_mode'],
        state['phase_panel_mode'],
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
        State('numeric-filter-bounds', 'data'),
        State('theme-mode', 'value'),
        State('plot-mode', 'value'),
        State('plot-overlays', 'value'),
        State('baseline-opacity-slider', 'value'),
        State('residual-height-slider', 'value'),
        State('external-source-values', 'value'),
        State('external-source-layout', 'value'),
        State('camera-checklist', 'value'),
        State('band-checklist', 'value'),
        State('yaxis-mode', 'value'),
        State('phase-panel-mode', 'value'),
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
    numeric_bounds = state_values[len(_FILTER_VALUE_STATES)]
    extra_values = state_values[len(_FILTER_VALUE_STATES) + 1:]
    queue_state = _queue_filter_ui_state_from_values(*queue_values)
    queue_state = _normalized_queue_filter_ui_state(queue_state, numeric_bounds)
    gui_state = _review_gui_state_from_values(
        theme_mode=extra_values[0],
        plot_mode=extra_values[1],
        plot_overlays=extra_values[2],
        baseline_opacity=extra_values[3],
        residual_height=extra_values[4],
        external_source_values=extra_values[5],
        external_source_layout=extra_values[6],
        camera_values=extra_values[7],
        band_values=extra_values[8],
        yaxis_mode=extra_values[9],
        phase_panel_mode=extra_values[10],
        period_method=extra_values[11],
        pdm_min_period=extra_values[12],
        pdm_max_period=extra_values[13],
        pdm_manual_period=extra_values[14],
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
        const placeholderRange = (
            currentMin === null &&
            currentMax === null &&
            currentRangeMin === 0 &&
            currentRangeMax === 1 &&
            !(Math.abs(dataLo) <= 1e-12 && Math.abs(dataHi - 1) <= 1e-12)
        );
        const rangeLooksInitialized = (
            currentRangeMin !== null &&
            currentRangeMax !== null &&
            currentRangeMin >= sliderLo &&
            currentRangeMax <= sliderHi
        );
        if (!rangeLooksInitialized || placeholderRange) {
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
