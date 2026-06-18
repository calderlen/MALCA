# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output('plot-export-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    Input('export-plot', 'n_clicks'),
    [State('interactive-plot', 'figure'),
     State('plot-mode', 'value'),
     State('plot-image', 'src'),
     State('current-index', 'data'),
     State('current-candidate-id', 'data'),
     State('plot-render-request', 'data')],
    prevent_initial_call=True,
)
def export_active_plot(n_clicks, figure, plot_mode, plot_src, idx, candidate_id, render_request):
    """Export the currently shown plot.

    Native and PNG display modes export a Matplotlib-rendered PDF from the
    candidate light-curve data and current plot controls.
    """
    if not n_clicks:
        return no_update, no_update

    ordinal = int(idx) + 1 if idx is not None else 0
    plot_mode_value = str(plot_mode or '').strip().lower()

    if plot_mode_value in {'native', 'png'}:
        if not figure and not candidate_id:
            return no_update, 'No native plot is available to export.'

        # Get ASAS-SN ID for filename
        asas_sn_id = 'unknown'
        try:
            if candidate_id:
                payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
                asas_sn_id = str(payload.get('asas_sn_id') or payload.get('candidate_id') or 'unknown')
        except Exception:
            pass

        fname = f"malca_plot_{ordinal}_{asas_sn_id}.pdf"
        try:
            state = {}
            if isinstance(render_request, dict) and isinstance(render_request.get('state'), dict):
                state = dict(render_request.get('state') or {})
            export_candidate_id = str(candidate_id or state.get('candidate_id') or '').strip()
            if not export_candidate_id:
                return no_update, 'No candidate is selected.'
            payload, _stored_lc_path, source_path = _candidate_context(export_candidate_id)
            plot_dir_path = _review_plot_dir_for_context(source_path)
            run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
            overlays = set(state.get('overlay_values') or [])
            override_period = state.get('override_period')
            override_period_source = str(state.get('override_period_source') or 'manual/search')
            phase_period_pending = bool(state.get('phase_period_pending', False))
            suppress_catalog_phase_period = bool(state.get('suppress_catalog_phase_period', False))
            if override_period is not None:
                try:
                    override_period = float(override_period)
                    if override_period <= 0:
                        override_period = None
                except (TypeError, ValueError):
                    override_period = None
            residual_height = state.get('residual_height', DEFAULT_RESIDUAL_FRACTION)
            baseline_opacity = state.get('baseline_opacity', 0.5)
            try:
                residual_height = float(residual_height)
            except (TypeError, ValueError):
                residual_height = DEFAULT_RESIDUAL_FRACTION
            try:
                baseline_opacity = float(baseline_opacity)
            except (TypeError, ValueError):
                baseline_opacity = 0.5
            image_bytes = build_review_lightcurve_publication_pdf(
                payload,
                plot_dir=plot_dir_path,
                selected_cameras=list(state.get('selected_cameras') or []),
                selected_bands=list(state.get('selected_bands') or ['g', 'V']),
                filter_bad_cameras='filter_bad_cameras' in overlays,
                show_baseline=baseline_opacity > 0,
                show_event_markers='markers' in overlays,
                show_residuals='residuals' in overlays,
                show_phase_fold='phase' in overlays,
                phase_panel_mode=_coerce_choice(state.get('phase_panel_mode'), {'fold', 'time'}, 'fold'),
                show_raw_mag='raw' in overlays,
                override_period=override_period,
                override_period_source=override_period_source,
                phase_period_pending=phase_period_pending,
                suppress_catalog_phase_period=suppress_catalog_phase_period,
                show_diagnostics='diagnostics' in overlays,
                confidence_colors='confidence' in overlays,
                run_params=run_params or {},
                residual_fraction=residual_height,
                baseline_opacity=baseline_opacity,
                yaxis_mode='flux' if str(state.get('yaxis_mode') or 'mag') == 'flux' else 'mag',
                native_color_mode='band' if str(state.get('native_color_mode') or 'camera') == 'band' else 'camera',
                external_source_view=list(state.get('external_source_values') or ['asassn']),
                external_panel_mode=_coerce_choice(
                    state.get('external_source_layout'),
                    {'overlay', 'split'},
                    'overlay',
                ),
                candidate_id=export_candidate_id,
            )
        except Exception as exc:
            return no_update, f'Export failed (PDF). {exc}'
        return dcc.send_bytes(image_bytes, fname), f'Exported {fname}'

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
