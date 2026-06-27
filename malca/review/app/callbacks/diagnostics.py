# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _matching_positive_click(
    triggered_id: object,
    button_ids: list[object] | tuple[object, ...] | None,
    clicks: list[object] | tuple[object, ...] | None,
) -> bool:
    if not isinstance(triggered_id, dict):
        return False
    target_panel = str(triggered_id.get("panel") or "")
    target_name = str(triggered_id.get("name") or "")
    for button_id, click_count in zip(button_ids or [], clicks or []):
        if not isinstance(button_id, dict):
            continue
        if str(button_id.get("panel") or "") != target_panel:
            continue
        if str(button_id.get("name") or "") != target_name:
            continue
        try:
            return int(click_count or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def _export_mini_plot_pdf_from_state(triggered_id, button_ids, clicks, graph_ids, figures, candidate_id):
    if not isinstance(triggered_id, dict):
        return no_update, no_update
    if not _matching_positive_click(triggered_id, button_ids, clicks):
        return no_update, no_update
    panel = str(triggered_id.get("panel") or "panel")
    name = str(triggered_id.get("name") or "plot")
    safe_id = slugify_token(candidate_id, fallback="candidate")
    if panel == "diagnostic":
        try:
            payload, _stored_lc_path, _source_path = _candidate_context(str(candidate_id or ""))
            signature = _diagnostic_background_signature(DB_PATH)
            background, _used_cache = _load_or_cache_diagnostic_background(signature)
            image_bytes = build_publication_diagnostic_pdf(name, payload, background)
        except Exception as exc:
            return no_update, f'Export failed (diagnostic PDF). {exc}'
        if image_bytes:
            fname = f"malca_{slugify_token(panel)}_{slugify_token(name)}_{safe_id}.pdf"
            return dcc.send_bytes(image_bytes, fname), f"Exported {fname}"

    selected = None
    for graph_id, figure in zip(graph_ids or [], figures or []):
        if not isinstance(graph_id, dict):
            continue
        if str(graph_id.get("panel")) == panel and str(graph_id.get("name")) == name:
            selected = figure
            break
    if not selected:
        return no_update, "No plot is available to export."
    title = _plot_title_text(selected, name.replace("-", " ").title())
    try:
        image_bytes = render_publication_pdf(
            selected,
            title=title,
            width=1200,
            height=820,
            legend_outside=True,
            right_margin=260,
        )
    except Exception as exc:
        return no_update, f'Export failed (plot PDF). {exc}'
    fname = f"malca_{slugify_token(panel)}_{slugify_token(name)}_{safe_id}.pdf"
    return dcc.send_bytes(image_bytes, fname), f"Exported {fname}"


@app.callback(
    [Output('mini-plot-export-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    Input({'type': 'mini-plot-export-btn', 'panel': ALL, 'name': ALL}, 'n_clicks'),
    [State({'type': 'mini-plot-export-graph', 'panel': ALL, 'name': ALL}, 'id'),
     State({'type': 'mini-plot-export-graph', 'panel': ALL, 'name': ALL}, 'figure'),
     State('current-candidate-id', 'data')],
    prevent_initial_call=True,
)
def export_mini_plot_pdf(_clicks, graph_ids, figures, candidate_id):
    button_inputs = callback_context.inputs_list[0] if callback_context.inputs_list else []
    if isinstance(button_inputs, dict):
        button_inputs = [button_inputs]
    button_ids = button_inputs
    button_ids = [item.get("id") for item in button_ids if isinstance(item, dict)]
    return _export_mini_plot_pdf_from_state(callback_context.triggered_id, button_ids, _clicks, graph_ids, figures, candidate_id)


def _render_diagnostic_plots(payload: dict, theme: str, background: dict | None = None) -> list:
    """Build diagnostic plot cards from candidate payload data."""
    theme_tokens = _external_followup_theme(theme)
    card_style = theme_tokens["card_style"]
    cards = []
    for builder in (
        build_cmd_figure,
        build_ir_colorcolor_figure,
        build_kiel_figure,
        build_teff_sed_alpha_figure,
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
            cards.append(_exportable_plot_card(
                fig,
                panel="diagnostic",
                name=str(getattr(builder, "__name__", "diagnostic")).replace("build_", "").replace("_figure", ""),
                card_style=card_style,
                height="280px",
            ))
    return cards


def _prepare_diagnostic_background(is_open, _import_trigger, _pipeline_progress, existing_state):
    """Load and cache diagnostic plot background data for the current review DB."""
    if not _details_open(is_open):
        raise dash.exceptions.PreventUpdate

    signature = _diagnostic_background_signature(DB_PATH)
    _background, used_cache = _load_or_cache_diagnostic_background(signature)

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


if _background_callback_manager is not None and _UI_BACKGROUND_CALLBACKS:
    @app.callback(
        Output('diagnostic-background-state', 'data'),
        [Input('diagnostic-plots-summary', 'n_clicks'),
         Input('import-trigger', 'data'),
         Input('pipeline-progress-trigger', 'data')],
        State('diagnostic-background-state', 'data'),
        background=True,
        running=[
            (Output('diagnostic-plots-status', 'children'), 'Loading population background...', ''),
        ],
        prevent_initial_call=True,
    )
    def prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state):
        return _prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state)
else:
    @app.callback(
        Output('diagnostic-background-state', 'data'),
        [Input('diagnostic-plots-summary', 'n_clicks'),
         Input('import-trigger', 'data'),
         Input('pipeline-progress-trigger', 'data')],
        State('diagnostic-background-state', 'data'),
        running=[
            (Output('diagnostic-plots-status', 'children'), 'Loading population background...', ''),
        ],
        prevent_initial_call=True,
    )
    def prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state):
        return _prepare_diagnostic_background(is_open, import_trigger, pipeline_progress, existing_state)


@app.callback(
    [Output('diagnostic-plots-panel', 'children'),
     Output('diagnostic-plots-status', 'children', allow_duplicate=True)],
    [Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data'),
     Input('diagnostic-background-state', 'data'),
     Input('diagnostic-plots-summary', 'n_clicks')],
    prevent_initial_call=True,
)
def update_diagnostic_plots(candidate_id, theme_mode, background_state, panel_requested=True):
    """Render diagnostic plots for the current candidate."""
    if not _details_open(panel_requested):
        return no_update, no_update
    if not candidate_id:
        return _lazy_panel_placeholder("No candidates loaded.", "error"), "No candidates loaded."

    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)

    signature = _diagnostic_background_signature(DB_PATH)
    cached_background = None
    if isinstance(background_state, dict) and background_state.get('ready') and background_state.get('signature') == signature:
        cached_background = _get_cached_diagnostic_background(signature)
    if cached_background is None:
        return _lazy_panel_placeholder("Loading population background for diagnostic plots."), "Loading population background..."

    panels = _render_diagnostic_plots(payload, str(theme_mode or DEFAULT_THEME), background=cached_background)
    return panels, ''


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
