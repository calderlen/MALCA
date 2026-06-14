# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output('eda-panel', 'className'),
     Output('eda-splitter', 'className'),
     Output('eda-expand-btn', 'children'),
     Output('eda-expand-btn', 'title'),
     Output('eda-panel-toggle', 'children'),
     Output('eda-panel-toggle', 'title'),
     Output('eda-panel-state', 'data')],
    [Input('eda-collapse-btn', 'n_clicks'),
     Input('eda-expand-btn', 'n_clicks'),
     Input('eda-panel-toggle', 'n_clicks')],
    State('eda-panel-state', 'data'),
    prevent_initial_call=False,
)
def toggle_eda_panel(collapse_clicks, expand_clicks, restore_clicks, panel_state):
    """Set embedded EDA rail state: collapsed, restored width, or expanded wide."""
    state = str(panel_state or 'open')
    if state not in {'open', 'collapsed', 'expanded'}:
        state = 'open'

    triggered = callback_context.triggered_id
    if triggered == 'eda-collapse-btn' and collapse_clicks:
        state = 'collapsed'
    elif triggered == 'eda-panel-toggle' and restore_clicks:
        state = 'open'
    elif triggered == 'eda-expand-btn' and expand_clicks:
        state = 'open' if state == 'expanded' else 'expanded'

    panel_class = 'eda-panel'
    splitter_class = 'eda-splitter panel-splitter-vertical'
    if state == 'collapsed':
        panel_class += ' is-collapsed'
        splitter_class += ' collapsed'
    elif state == 'expanded':
        panel_class += ' is-expanded'

    expand_text = 'Restore' if state == 'expanded' else 'Wide'
    expand_title = 'Restore saved EDA width' if state == 'expanded' else 'Expand EDA panel'
    return panel_class, splitter_class, expand_text, expand_title, 'EDA', 'Show EDA panel', state


@app.callback(
    [Output('eda-x-metric', 'options'),
     Output('eda-x-metric', 'value'),
     Output('eda-y-metric', 'options'),
     Output('eda-y-metric', 'value'),
     Output('eda-color-metric', 'options'),
     Output('eda-color-metric', 'value'),
     Output('eda-symbol-metric', 'options'),
     Output('eda-symbol-metric', 'value')],
    [Input('queue-data', 'data'),
     Input('review-db-scope', 'data'),
     Input('last-candidate-saved', 'data'),
     Input('import-trigger', 'data')],
    [State('eda-x-metric', 'value'),
     State('eda-y-metric', 'value'),
     State('eda-color-metric', 'value'),
     State('eda-symbol-metric', 'value')],
    prevent_initial_call=False,
)
def sync_eda_metric_controls(queue_data, _db_scope, _last_saved, _import_trigger, x_metric, y_metric, color_metric, symbol_metric):
    try:
        frame = _current_eda_frame()
        queue_frame = queue_eda_frame(frame, _queue_candidate_ids(queue_data))
        metric_frame = queue_frame if not queue_frame.empty else frame
        options = eda_metric_options(metric_frame)
        x_value, y_value, color_value, symbol_value = resolve_eda_metric_values(
            metric_frame,
            x_metric=x_metric,
            y_metric=y_metric,
            color_metric=color_metric,
            symbol_metric=symbol_metric,
        )
    except Exception:
        options = []
        x_value = y_value = color_value = symbol_value = None
    return options, x_value, options, y_value, options, color_value, options, symbol_value


@app.callback(
    [Output('eda-status', 'children'),
     Output('eda-custom-graph', 'figure'),
     Output('eda-candidate-table', 'data'),
     Output('eda-candidate-table', 'style_data_conditional')],
    [Input('queue-data', 'data'),
     Input('current-index', 'data'),
     Input('eda-x-metric', 'value'),
     Input('eda-y-metric', 'value'),
     Input('eda-color-metric', 'value'),
     Input('eda-symbol-metric', 'value'),
     Input('eda-log-flags', 'value'),
     Input('eda-selection-mode', 'value'),
     Input('eda-selection-candidate-ids', 'data'),
     Input('theme-mode-store', 'data'),
     Input('last-candidate-saved', 'data'),
     Input('import-trigger', 'data'),
     Input('review-db-scope', 'data')],
    prevent_initial_call=False,
)
def update_eda_panel(
    queue_data,
    current_index,
    x_metric,
    y_metric,
    color_metric,
    symbol_metric,
    log_flags,
    selection_mode,
    selection_candidate_ids,
    theme_mode,
    _last_saved,
    _import_trigger,
    _db_scope,
):
    theme = str(theme_mode or DEFAULT_THEME)
    try:
        frame = _current_eda_frame()
        queue_ids = _queue_candidate_ids(queue_data)
        queue_frame = queue_eda_frame(frame, queue_ids)
    except Exception as exc:
        return f"EDA unavailable: {exc}", eda_status_figure("EDA data unavailable.", theme=theme), [], []

    selected_candidate = selected_candidate_from_queue(queue_data, current_index)
    if not queue_ids:
        fig = eda_status_figure("Refresh the review queue to populate EDA.", theme=theme)
        return "No active review queue.", fig, [], []

    x_value, y_value, color_value, symbol_value = resolve_eda_metric_values(
        queue_frame if not queue_frame.empty else frame,
        x_metric=x_metric,
        y_metric=y_metric,
        color_metric=color_metric,
        symbol_metric=symbol_metric,
    )
    flags = set(log_flags or [])
    fig = eda_scatter_figure(
        queue_frame,
        x_metric=x_value,
        y_metric=y_value,
        color_metric=color_value,
        symbol_metric=symbol_value,
        selected_candidate_id=selected_candidate,
        log_x='logx' in flags,
        log_y='logy' in flags,
        theme=theme,
    )
    selection_values = [selection_mode] if isinstance(selection_mode, str) else list(selection_mode or [])
    selection_enabled = 'table' in {str(value) for value in selection_values}
    if selection_enabled:
        fig.update_layout(dragmode='select')
    else:
        fig.update_layout(dragmode='zoom')

    table_frame = queue_frame
    selection_ids = {str(value) for value in (selection_candidate_ids or [])} if selection_enabled else set()
    selection_active = bool(selection_ids)
    if selection_active:
        table_frame = queue_frame[queue_frame["candidate_id"].astype(str).isin(selection_ids)].copy()

    rows = eda_table_rows(table_frame)
    style = selected_row_style(rows, selected_candidate, theme=theme)
    selected_text = selected_candidate or 'none'
    counts = eda_plot_row_counts(
        queue_frame,
        x_metric=x_value,
        y_metric=y_value,
        log_x='logx' in flags,
        log_y='logy' in flags,
    )
    status_parts = [
        f"Queue rows: {len(queue_frame):,}/{len(frame):,}",
        f"Plotted: {int(counts.get('plottable_rows') or 0):,}",
    ]
    dropped_missing = int(counts.get('dropped_missing') or 0)
    dropped_nonpositive = int(counts.get('dropped_nonpositive') or 0)
    if dropped_missing:
        status_parts.append(f"Dropped missing: {dropped_missing:,}")
    if dropped_nonpositive:
        status_parts.append(f"Dropped log<=0: {dropped_nonpositive:,}")
    if selection_active:
        status_parts.append(f"Selected: {len(table_frame):,}")
    status_parts.append(f"Current: {selected_text}")
    status = " | ".join(status_parts)
    return status, fig, rows, style


def _eda_selection_enabled(selection_mode):
    selection_values = [selection_mode] if isinstance(selection_mode, str) else list(selection_mode or [])
    return 'table' in {str(value) for value in selection_values}


@app.callback(
    Output('eda-selection-candidate-ids', 'data'),
    Input('eda-custom-graph', 'selectedData'),
    State('eda-custom-graph', 'figure'),
    State('eda-selection-mode', 'value'),
    prevent_initial_call=True,
)
def capture_eda_selection(selected_data, figure, selection_mode):
    if not _eda_selection_enabled(selection_mode):
        raise dash.exceptions.PreventUpdate
    if not isinstance(selected_data, dict):
        raise dash.exceptions.PreventUpdate
    return candidate_ids_from_plotly_selection(selected_data, figure)


@app.callback(
    [Output('eda-custom-graph', 'selectedData'),
     Output('eda-selection-candidate-ids', 'data', allow_duplicate=True)],
    Input('eda-clear-selection-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def clear_eda_selection(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return None, []


@app.callback(
    Output('eda-selection-candidate-ids', 'data', allow_duplicate=True),
    Input('eda-selection-mode', 'value'),
    prevent_initial_call=True,
)
def clear_eda_selection_when_disabled(selection_mode):
    if _eda_selection_enabled(selection_mode):
        raise dash.exceptions.PreventUpdate
    return []


@app.callback(
    [Output('eda-plot-export-download', 'data'),
     Output('eda-export-status', 'children')],
    Input('eda-export-pdf-btn', 'n_clicks'),
    [State('queue-data', 'data'),
     State('current-index', 'data'),
     State('eda-x-metric', 'value'),
     State('eda-y-metric', 'value'),
     State('eda-color-metric', 'value'),
     State('eda-symbol-metric', 'value'),
     State('eda-log-flags', 'value')],
    prevent_initial_call=True,
)
def export_eda_plot_pdf(n_clicks, queue_data, current_index, x_metric, y_metric, color_metric, symbol_metric, log_flags):
    if not n_clicks:
        return no_update, no_update
    queue_ids = _queue_candidate_ids(queue_data)
    if not queue_ids:
        return no_update, "No active review queue to export."
    x_value = str(x_metric or "").strip()
    y_value = str(y_metric or "").strip()
    if not x_value or not y_value:
        return no_update, "Choose valid X and Y metrics before exporting."
    try:
        frame = _current_eda_frame()
        queue_frame = queue_eda_frame(frame, queue_ids)
    except Exception as exc:
        return no_update, f"EDA PDF export failed: {exc}"
    if queue_frame.empty:
        return no_update, "Current queue has no EDA rows to export."
    missing = [metric for metric in (x_value, y_value) if metric not in queue_frame.columns]
    if missing:
        return no_update, f"Missing EDA metric: {', '.join(missing)}"
    flags = set(log_flags or [])
    counts = eda_plot_row_counts(
        queue_frame,
        x_metric=x_value,
        y_metric=y_value,
        log_x='logx' in flags,
        log_y='logy' in flags,
    )
    if int(counts.get('plottable_rows') or 0) <= 0:
        if int(counts.get('dropped_nonpositive') or 0):
            return no_update, f"No queue rows remain after log-axis filtering for {y_value} vs {x_value}."
        return no_update, f"No queue rows have plottable {x_value} and {y_value} values."

    color_value = str(color_metric).strip() if color_metric and str(color_metric).strip() in queue_frame.columns else None
    symbol_value = str(symbol_metric).strip() if symbol_metric and str(symbol_metric).strip() in queue_frame.columns else None
    selected_candidate = selected_candidate_from_queue(queue_data, current_index)
    title = f"{y_value} vs {x_value}"
    try:
        export_fig = eda_publication_figure(
            queue_frame,
            x_metric=x_value,
            y_metric=y_value,
            color_metric=color_value,
            symbol_metric=symbol_value,
            selected_candidate_id=selected_candidate,
            log_x='logx' in flags,
            log_y='logy' in flags,
        )
        image_bytes = render_publication_pdf(
            export_fig,
            title=title,
            width=1200,
            height=820,
            legend_outside=True,
            style=False,
        )
    except Exception as exc:
        return no_update, f"EDA PDF export failed: {exc}"
    fname = f"review_eda_{slugify_token(y_value, fallback='y')}_vs_{slugify_token(x_value, fallback='x')}.pdf"
    return dcc.send_bytes(image_bytes, fname), f"Exported {fname}"


@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input('eda-custom-graph', 'clickData'),
     Input('eda-candidate-table', 'active_cell')],
    [State('eda-candidate-table', 'derived_viewport_data'),
     State('eda-candidate-table', 'derived_virtual_data'),
     State('eda-candidate-table', 'data'),
     State('eda-candidate-table', 'page_current'),
     State('eda-candidate-table', 'page_size'),
     State('queue-data', 'data'),
     State('current-index', 'data')],
    prevent_initial_call=True,
)
def navigate_from_eda(click_data, active_cell, viewport_table_data, visible_table_data, table_data, page_current, page_size, queue_data, current_index):
    triggered = callback_context.triggered_id
    candidate_id = ''
    candidate_ids = []
    if triggered == 'eda-custom-graph':
        points = (click_data or {}).get('points') or []
        if points:
            custom = points[0].get('customdata')
            if isinstance(custom, (list, tuple)) and custom:
                candidate_id = str(custom[0])
            elif custom:
                candidate_id = str(custom)
            if candidate_id:
                candidate_ids.append(candidate_id)
    elif triggered == 'eda-candidate-table' and isinstance(active_cell, dict):
        candidate_ids.extend(
            candidate_ids_from_eda_table_context(
                active_cell,
                viewport_table_data,
                visible_table_data,
                table_data,
                page_current,
                page_size,
            )
        )

    next_index = None
    for candidate_id in candidate_ids:
        next_index = candidate_index_in_queue(queue_data, candidate_id)
        if next_index is not None:
            break
    if next_index is None:
        raise dash.exceptions.PreventUpdate
    try:
        if int(current_index or 0) == int(next_index):
            raise dash.exceptions.PreventUpdate
    except dash.exceptions.PreventUpdate:
        raise
    except Exception:
        pass
    return next_index, f"EDA selected {candidate_id}."


# Toggle sidebar
