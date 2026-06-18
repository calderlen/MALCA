# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _dash_triggered_id() -> object | None:
    try:
        return callback_context.triggered_id
    except Exception:
        return None


def _jsonish(value: object, default: object) -> object:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        parsed = json.loads(str(value))
    except Exception:
        return default
    return parsed


def _dustycult_float(value: object, digits: int = 4) -> str:
    return format_dustycult_float(value, digits)


def _dustycult_status_cards(fits: pd.DataFrame, theme: str | None) -> html.Div:
    spec = _external_followup_theme(theme)
    muted = str(spec["muted"])
    cards = []
    for card in dustycult_status_card_rows(fits):
        status = str(card.get("status") or "not run")
        detail = str(card.get("detail") or "")
        color = "#64c27b" if status == "ok" else ("#dd8080" if status == "failed" else muted)
        cards.append(html.Div([
            html.Div(str(card.get("label") or card.get("mode") or ""), style={'fontSize': '10px', 'color': muted}),
            html.Div(status, style={'fontSize': '13px', 'fontWeight': 600, 'color': color}),
            html.Div(detail, style={'fontSize': '10px', 'color': muted, 'overflowWrap': 'anywhere'}),
        ], style={
            'border': '1px solid rgba(125, 145, 166, 0.28)',
            'borderRadius': '6px',
            'padding': '6px 8px',
            'minWidth': '120px',
        }))
    return html.Div(cards, style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'})


def _select_dustycult_display_row(fits: pd.DataFrame) -> pd.Series | None:
    return select_dustycult_display_row(fits)


def _dustycult_result_figure(curves: pd.DataFrame, fit_row: pd.Series, theme: str | None) -> go.Figure:
    return build_dustycult_fit_figure(curves, fit_row, theme)


def _dustycult_parameter_table(fit_row: pd.Series, theme: str | None) -> html.Div:
    spec = _external_followup_theme(theme)
    rows = dustycult_posterior_rows(fit_row)
    if not rows:
        return html.Div("No posterior summary stored.", style={'fontSize': '11px', 'color': spec["muted"]})
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Parameter", style={'padding': '3px 6px'}),
            html.Th("Median", style={'padding': '3px 6px'}),
            html.Th("p16", style={'padding': '3px 6px'}),
            html.Th("p84", style={'padding': '3px 6px'}),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(parameter, style={'padding': '3px 6px', 'fontWeight': 600}),
                html.Td(median, style={'padding': '3px 6px'}),
                html.Td(p16, style={'padding': '3px 6px'}),
                html.Td(p84, style={'padding': '3px 6px'}),
            ])
            for parameter, median, p16, p84 in rows
        ]),
    ], className='dustycult-param-table', style={'width': '100%', 'fontSize': '11px', 'borderCollapse': 'collapse'})


def _dustycult_geometry_table(fit_row: pd.Series, theme: str | None) -> html.Div:
    spec = _external_followup_theme(theme)
    try:
        rows = dustycult_geometry_rows(fit_row)
    except Exception as exc:
        return html.Div(
            f"Circumstellar geometry unavailable: {exc}",
            style={'fontSize': '11px', 'color': spec["error"], 'overflowWrap': 'anywhere'},
        )
    return html.Div([
        html.Div("Circumstellar Dust Geometry", style={'fontSize': '11px', 'fontWeight': 700, 'color': spec["font"], 'marginBottom': '4px'}),
        html.Table(
            html.Tbody([
                html.Tr([
                    html.Td(label, style={'padding': '3px 6px', 'fontWeight': 600}),
                    html.Td(value, style={'padding': '3px 6px'}),
                ])
                for label, value in rows
            ]),
            className='dustycult-param-table',
            style={'width': '100%', 'fontSize': '11px', 'borderCollapse': 'collapse'},
        ),
    ])


def _render_dustycult_result_panel(candidate_id: str, theme_mode: str | None, _refresh_token: object = None) -> list:
    spec = _external_followup_theme(theme_mode)
    if not candidate_id:
        return [html.Div("No candidates loaded.", style={'fontSize': '11px', 'color': spec["error"]})]
    with closing(db_connect(Path(DB_PATH))) as conn:
        fits = load_dustycult_fits(conn, str(candidate_id))
    availability = check_dustycult_available()
    children: list = []
    if not availability.ok:
        children.append(html.Div(
            availability.message,
            style={'fontSize': '10px', 'color': spec["error"], 'overflowWrap': 'anywhere'},
        ))
    children.append(_dustycult_status_cards(fits, theme_mode))
    if fits is None or fits.empty:
        children.append(html.Div("No DustyCult fit has been run for this candidate.", style={'fontSize': '11px', 'color': spec["muted"]}))
        return children
    fit_row = _select_dustycult_display_row(fits)
    if fit_row is None:
        return children
    mode = str(fit_row.get("mode") or "quick")
    with closing(db_connect(Path(DB_PATH))) as conn:
        curves = load_dustycult_curve(conn, str(candidate_id), mode)
    if str(fit_row.get("status") or "").lower() == "ok":
        children.append(dcc.Graph(
            id='dustycult-fit-plot',
            figure=_dustycult_result_figure(curves, fit_row, theme_mode),
            mathjax=True,
            config=graph_config_without_image_export({'displayModeBar': False, 'responsive': True}),
            style={'height': '400px'},
        ))
        try:
            children.append(dcc.Graph(
                id='dustycult-occulter-plot',
                figure=build_dustycult_occulter_figure(fit_row, theme=theme_mode, grid_n=251),
                mathjax=True,
                config=graph_config_without_image_export({'displayModeBar': False, 'responsive': True}),
                style={'height': '440px'},
            ))
        except Exception as exc:
            children.append(html.Div(
                f"Occulter model unavailable: {exc}",
                style={'fontSize': '11px', 'color': spec["error"], 'overflowWrap': 'anywhere'},
            ))
    else:
        error = str(fit_row.get("error") or "DustyCult fit failed.")
        stderr = str(fit_row.get("stderr_tail") or "").strip()
        children.append(html.Div([
            html.Div(error, style={'fontSize': '11px', 'color': spec["error"], 'overflowWrap': 'anywhere'}),
            html.Pre(stderr[-1800:] if stderr else "", style={
                'display': 'block' if stderr else 'none',
                'fontSize': '10px',
                'whiteSpace': 'pre-wrap',
                'overflowWrap': 'anywhere',
                'maxHeight': '160px',
                'overflowY': 'auto',
                'marginTop': '6px',
            }),
        ]))
    children.append(html.Div(dustycult_fit_metadata_text(fit_row), style={'fontSize': '10px', 'color': spec["muted"], 'overflowWrap': 'anywhere'}))
    children.append(_dustycult_geometry_table(fit_row, theme_mode))
    children.append(_dustycult_parameter_table(fit_row, theme_mode))
    return children


def _dustycult_display_fit(conn, candidate_id: str) -> tuple[pd.Series | None, pd.DataFrame]:
    fits = load_dustycult_fits(conn, str(candidate_id))
    fit_row = _select_dustycult_display_row(fits)
    if fit_row is None:
        return None, pd.DataFrame()
    curves = load_dustycult_curve(conn, str(candidate_id), str(fit_row.get("mode") or "quick"))
    return fit_row, curves


def _dustycult_fit_publication_figure(conn, candidate_id: str) -> tuple[go.Figure, str]:
    fit_row, curves = _dustycult_display_fit(conn, candidate_id)
    if fit_row is None:
        raise ValueError("No DustyCult fit has been run for this candidate.")
    status = str(fit_row.get("status") or "").lower()
    if status != "ok":
        raise ValueError(f"DustyCult fit is not exportable because status is {status or 'unknown'}.")
    mode = str(fit_row.get("mode") or "quick")
    fig = _dustycult_result_figure(curves, fit_row, "white")
    export_fig = publication_figure(
        fig,
        title=f"DustyCult {mode.capitalize()} Fit",
        width=1200,
        height=820,
        legend_outside=True,
        right_margin=285,
        xaxis_title=r"$t\ [\mathrm{JD}]$",
        yaxis_title=r"$F/F_{\mathrm{GP}}$",
    )
    return export_fig, mode


def _dustycult_occulter_publication_figure(conn, candidate_id: str) -> tuple[go.Figure, str]:
    fit_row, _curves = _dustycult_display_fit(conn, candidate_id)
    if fit_row is None:
        raise ValueError("No DustyCult fit has been run for this candidate.")
    status = str(fit_row.get("status") or "").lower()
    if status != "ok":
        raise ValueError(f"DustyCult occulter is not exportable because fit status is {status or 'unknown'}.")
    mode = str(fit_row.get("mode") or "quick")
    fig = build_dustycult_occulter_figure(fit_row, theme="white", grid_n=501)
    export_fig = publication_figure(
        fig,
        title=f"DustyCult Occulter Model ({mode})",
        width=1200,
        height=620,
        legend_outside=False,
        right_margin=135,
        top_margin=86,
        bottom_margin=74,
        left_margin=86,
    )
    return export_fig, mode


def _phoebe_config_status_text() -> str:
    availability = check_phoebe_available()
    return availability.message if availability.ok else f"Unavailable: {availability.message}"


def _phoebe_row_json(fit_row: pd.Series | dict, column: str) -> dict:
    value = fit_row.get(column) if hasattr(fit_row, "get") else None
    parsed = parse_phoebe_json(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _phoebe_solver_status(fit_row: pd.Series | dict) -> str:
    params = _phoebe_row_json(fit_row, "params_json")
    metrics = _phoebe_row_json(fit_row, "metrics_json")
    return str(params.get("solver_status") or metrics.get("solver_status") or "").strip()


def _phoebe_display_status(fit_row: pd.Series | dict | None) -> str:
    if fit_row is None:
        return "not run"
    status = str(fit_row.get("status") or "unknown").strip().lower()
    solver_status = _phoebe_solver_status(fit_row)
    if status == "ok" and solver_status and solver_status != "ok":
        return "warning"
    return status


def _phoebe_warning_text(fit_row: pd.Series | dict) -> str:
    error = str(fit_row.get("error") or "").strip()
    if error:
        return error
    solver_status = _phoebe_solver_status(fit_row)
    if solver_status and solver_status != "ok":
        return f"PHOEBE solver did not complete; diagnostic model only. {solver_status}"
    return "PHOEBE solver did not complete; diagnostic model only."


def _phoebe_status_cards(fits: pd.DataFrame, theme: str | None) -> html.Div:
    spec = _external_followup_theme(theme)
    muted = str(spec["muted"])
    row = fits.iloc[-1] if fits is not None and not fits.empty else None
    status = _phoebe_display_status(row)
    detail = ""
    if row is not None:
        detail_parts = []
        runtime = row.get("runtime_sec")
        n_points = row.get("n_input_points")
        period = row.get("period_days")
        if runtime is not None and not pd.isna(runtime):
            detail_parts.append(f"{_dustycult_float(runtime, 3)} s")
        if period is not None and not pd.isna(period):
            detail_parts.append(f"P={_dustycult_float(period, 6)} d")
        if n_points is not None and not pd.isna(n_points):
            detail_parts.append(f"{int(float(n_points))} pts")
        error = str(row.get("error") or "").strip()
        if status != "ok" and error:
            detail_parts.append(error[:140])
        elif status == "warning":
            detail_parts.append(_phoebe_warning_text(row)[:140])
        detail = " | ".join(detail_parts)
    color = "#64c27b" if status == "ok" else ("#d99a28" if status == "warning" else ("#dd8080" if status == "failed" else muted))
    return html.Div([
        html.Div([
            html.Div("Latest", style={'fontSize': '10px', 'color': muted}),
            html.Div(status, style={'fontSize': '13px', 'fontWeight': 600, 'color': color}),
            html.Div(detail, style={'fontSize': '10px', 'color': muted, 'overflowWrap': 'anywhere'}),
        ], style={
            'border': '1px solid rgba(125, 145, 166, 0.28)',
            'borderRadius': '6px',
            'padding': '6px 8px',
            'minWidth': '160px',
        })
    ], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'})


def _phoebe_result_figure(fit_row: pd.Series, theme: str | None) -> go.Figure:
    spec = _external_followup_theme(theme)
    payload = parse_phoebe_json(fit_row.get("plot_json"), {})
    fig = go.Figure()
    try:
        phase = np.asarray(payload.get("phase") or [], dtype=float)
        flux = np.asarray(payload.get("flux") or [], dtype=float)
        flux_err = np.asarray(payload.get("flux_err") or [], dtype=float)
    except Exception:
        phase = np.asarray([], dtype=float)
        flux = np.asarray([], dtype=float)
        flux_err = np.asarray([], dtype=float)
    valid = np.isfinite(phase) & np.isfinite(flux)
    if bool(valid.any()):
        error_y = None
        if flux_err.size == phase.size and bool(np.isfinite(flux_err[valid]).any()):
            error_y = dict(type="data", array=flux_err[valid], visible=True, thickness=0.7)
        fig.add_trace(go.Scatter(
            x=phase[valid],
            y=flux[valid],
            mode="markers",
            name="observed",
            marker=dict(color="#69c779", size=5, opacity=0.78, line=dict(color="#111827", width=0.35)),
            error_y=error_y,
        ))
        model_payload = payload.get("model_flux")
        if isinstance(model_payload, list) and len(model_payload) == len(phase):
            model = np.asarray(model_payload, dtype=float)
            model_valid = valid & np.isfinite(model)
            if bool(model_valid.any()):
                order = np.argsort(phase[model_valid])
                fig.add_trace(go.Scatter(
                    x=phase[model_valid][order],
                    y=model[model_valid][order],
                    mode="lines",
                    name="PHOEBE model",
                    line=dict(color="#f2c86b", width=2),
                ))
    else:
        fig.add_annotation(text="No PHOEBE plot data stored for this fit.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    period = fit_row.get("period_days")
    title = "PHOEBE Fit" if _phoebe_display_status(fit_row) == "ok" else "PHOEBE Diagnostic Model"
    if period is not None and not pd.isna(period):
        title += f" (P={_dustycult_float(period, 6)} d)"
    fig.update_layout(
        template=None,
        title=title,
        paper_bgcolor=spec["paper_bg"],
        plot_bgcolor=spec["plot_bg"],
        font=dict(color=spec["font"], family=PUBLICATION_PLOTLY_FONT, size=11),
        margin=dict(l=56, r=18, t=42, b=48),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor=spec["legend_bg"],
            bordercolor=spec["legend_border"],
            borderwidth=1,
        ),
    )
    fig.update_xaxes(title="phase", gridcolor=spec["grid"], zeroline=False)
    fig.update_yaxes(title="relative flux", gridcolor=spec["grid"], zeroline=False)
    return fig


def _phoebe_parameter_table(fit_row: pd.Series, theme: str | None) -> html.Div:
    spec = _external_followup_theme(theme)
    metrics = _phoebe_row_json(fit_row, "metrics_json")
    params = _phoebe_row_json(fit_row, "params_json")
    rows = []
    for label, value in (
        ("period_source", fit_row.get("period_source")),
        ("model_kind", fit_row.get("model_kind")),
        ("reduced_chi2", metrics.get("reduced_chi2") if isinstance(metrics, dict) else None),
        ("rms_residual", metrics.get("rms_residual") if isinstance(metrics, dict) else None),
        ("model_flux_source", metrics.get("model_flux_source") if isinstance(metrics, dict) else None),
        ("model_flux_scale", metrics.get("model_flux_scale") if isinstance(metrics, dict) else None),
        ("solver_status", params.get("solver_status") if isinstance(params, dict) else None),
        ("compute_status", params.get("compute_status") if isinstance(params, dict) else None),
    ):
        rows.append(html.Tr([
            html.Td(str(label), style={'padding': '3px 6px', 'fontWeight': 600}),
            html.Td(_dustycult_float(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value or "-"), style={'padding': '3px 6px'}),
        ]))
    return html.Table(html.Tbody(rows), style={'width': '100%', 'fontSize': '11px', 'borderCollapse': 'collapse', 'color': spec["font"]})


def _render_phoebe_result_panel(candidate_id: str, theme_mode: str | None, _refresh_token: object = None) -> list:
    spec = _external_followup_theme(theme_mode)
    if not candidate_id:
        return [html.Div("No candidates loaded.", style={'fontSize': '11px', 'color': spec["error"]})]
    with closing(db_connect(Path(DB_PATH))) as conn:
        fits = load_phoebe_fits(conn, str(candidate_id))
    availability = check_phoebe_available()
    children: list = []
    if not availability.ok:
        children.append(html.Div(
            availability.message,
            style={'fontSize': '10px', 'color': spec["error"], 'overflowWrap': 'anywhere'},
        ))
    children.append(_phoebe_status_cards(fits, theme_mode))
    if fits is None or fits.empty:
        children.append(html.Div("No PHOEBE fit has been run for this candidate.", style={'fontSize': '11px', 'color': spec["muted"]}))
        return children
    fit_row = fits.iloc[-1]
    status = _phoebe_display_status(fit_row)
    if status in {"ok", "warning"}:
        if status == "warning":
            children.append(html.Div(
                _phoebe_warning_text(fit_row),
                style={'fontSize': '11px', 'color': '#d99a28', 'overflowWrap': 'anywhere'},
            ))
        children.append(dcc.Graph(
            id='phoebe-fit-plot',
            figure=_phoebe_result_figure(fit_row, theme_mode),
            mathjax=True,
            config=graph_config_without_image_export({'displayModeBar': True, 'responsive': True}),
            style={'height': '350px'},
        ))
    else:
        children.append(html.Div(
            str(fit_row.get("error") or "PHOEBE fit failed."),
            style={'fontSize': '11px', 'color': spec["error"], 'overflowWrap': 'anywhere'},
        ))
    meta = [
        f"status={status}",
        f"runtime={_dustycult_float(fit_row.get('runtime_sec'), 3)} s",
        f"version={fit_row.get('phoebe_version') or '-'}",
        f"input={fit_row.get('input_path') or '-'}",
    ]
    children.append(html.Div(" | ".join(meta), style={'fontSize': '10px', 'color': spec["muted"], 'overflowWrap': 'anywhere'}))
    children.append(_phoebe_parameter_table(fit_row, theme_mode))
    return children


def _dustycult_config_status_text() -> str:
    availability = check_dustycult_available()
    if availability.ok:
        return f"{availability.message}. Project: {availability.project_path}. Runner: {availability.script_path}."
    return f"Unavailable: {availability.message}"


@app.callback(
    [Output(field_id, 'value') for _key, field_id, _label, _step in _DUSTYCULT_CONTROL_FIELDS]
    + [Output('dustycult-defaults-status', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('dustycult-recompute-dip-btn', 'n_clicks'),
     Input('dustycult-summary', 'n_clicks')],
    prevent_initial_call=False,
)
def update_dustycult_controls(candidate_id, _recompute_clicks, panel_requested=True):
    """Populate editable DustyCult defaults for the current candidate."""
    if not _details_open(panel_requested):
        return tuple([no_update] * len(_DUSTYCULT_CONTROL_FIELDS) + [no_update])
    if not candidate_id:
        return tuple([None] * len(_DUSTYCULT_CONTROL_FIELDS) + ["No candidates loaded."])
    triggered_id = _dash_triggered_id()
    recompute = triggered_id == 'dustycult-recompute-dip-btn'
    try:
        payload, stored_lc_path, source_path = _candidate_context(candidate_id)
        lc_path = _effective_local_lc_path(payload, stored_lc_path=stored_lc_path, source_path=source_path)
        plot_dir_path = _review_plot_dir_for_context(source_path)
        run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
        with closing(db_connect(Path(DB_PATH))) as conn:
            defaults = control_defaults_for_candidate(
                conn,
                str(candidate_id),
                payload,
                lc_path=lc_path,
                plot_dir=str(plot_dir_path) if plot_dir_path else None,
                run_params=run_params,
                recompute=recompute,
            )
    except Exception as exc:
        return tuple([None] * len(_DUSTYCULT_CONTROL_FIELDS) + [f"DustyCult defaults failed: {exc}"])
    values = [defaults.get(key) for key, _field_id, _label, _step in _DUSTYCULT_CONTROL_FIELDS]
    source = str(defaults.get("source") or "defaults")
    message = str(defaults.get("message") or "")
    return tuple(values + [f"{source}: {message}"])


def _dustycult_fit_callback_impl(triggered_id, quick_clicks, full_clicks, candidate_id, refresh_token, *control_values):
    if triggered_id == 'dustycult-quick-fit-btn':
        mode = "quick"
        if not quick_clicks:
            raise dash.exceptions.PreventUpdate
    elif triggered_id == 'dustycult-full-fit-btn':
        mode = "full"
        if not full_clicks:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate
    if not candidate_id:
        return "No candidate selected.", refresh_token
    controls = _dustycult_control_values_from_states(control_values)
    try:
        payload, stored_lc_path, source_path = _candidate_context(candidate_id)
        lc_path = _effective_local_lc_path(payload, stored_lc_path=stored_lc_path, source_path=source_path)
        plot_dir_path = _review_plot_dir_for_context(source_path)
        run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
        with closing(db_connect(Path(DB_PATH))) as conn:
            row = run_dustycult_fit(
                conn,
                str(candidate_id),
                payload,
                db_path=Path(DB_PATH),
                controls=controls,
                mode=mode,
                lc_path=lc_path,
                plot_dir=str(plot_dir_path) if plot_dir_path else None,
                run_params=run_params,
            )
    except Exception as exc:
        row = {"status": "failed", "error": str(exc), "runtime_sec": None, "artifact_dir": ""}
    next_token = int(refresh_token or 0) + 1
    status = str(row.get("status") or "unknown")
    if status == "ok":
        return (
            f"DustyCult {mode} fit complete in {_dustycult_float(row.get('runtime_sec'), 3)} s. "
            f"Artifact: {row.get('artifact_dir') or ''}",
            next_token,
        )
    return (
        f"DustyCult {mode} fit failed: {row.get('error') or 'unknown error'}",
        next_token,
    )


if _background_callback_manager is not None:
    @app.callback(
        [Output('dustycult-run-status', 'children'),
         Output('dustycult-refresh-token', 'data')],
        [Input('dustycult-quick-fit-btn', 'n_clicks'),
         Input('dustycult-full-fit-btn', 'n_clicks')],
        [State('current-candidate-id', 'data'),
         State('dustycult-refresh-token', 'data')]
        + [State(field_id, 'value') for _key, field_id, _label, _step in _DUSTYCULT_CONTROL_FIELDS],
        background=True,
        running=[
            (Output('dustycult-quick-fit-btn', 'disabled'), True, False),
            (Output('dustycult-full-fit-btn', 'disabled'), True, False),
            (Output('dustycult-recompute-dip-btn', 'disabled'), True, False),
        ],
        prevent_initial_call=True,
    )
    def run_dustycult_fit_callback(quick_clicks, full_clicks, candidate_id, refresh_token, *control_values):
        return _dustycult_fit_callback_impl(
            _dash_triggered_id(),
            quick_clicks,
            full_clicks,
            candidate_id,
            refresh_token,
            *control_values,
        )
else:
    @app.callback(
        [Output('dustycult-run-status', 'children'),
         Output('dustycult-refresh-token', 'data')],
        [Input('dustycult-quick-fit-btn', 'n_clicks'),
         Input('dustycult-full-fit-btn', 'n_clicks')],
        [State('current-candidate-id', 'data'),
         State('dustycult-refresh-token', 'data')]
        + [State(field_id, 'value') for _key, field_id, _label, _step in _DUSTYCULT_CONTROL_FIELDS],
        prevent_initial_call=True,
    )
    def run_dustycult_fit_callback(quick_clicks, full_clicks, candidate_id, refresh_token, *control_values):
        return _dustycult_fit_callback_impl(
            _dash_triggered_id(),
            quick_clicks,
            full_clicks,
            candidate_id,
            refresh_token,
            *control_values,
        )


@app.callback(
    [Output('dustycult-result-panel', 'children'),
     Output('dustycult-config-status', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data'),
     Input('dustycult-refresh-token', 'data'),
     Input('dustycult-summary', 'n_clicks')],
    prevent_initial_call=False,
)
def update_dustycult_result_panel(candidate_id, theme_mode, refresh_token, panel_requested=True):
    """Render DustyCult status, predictive overlay, and posterior summary."""
    if not _details_open(panel_requested):
        return no_update, no_update
    if not candidate_id:
        return _render_dustycult_result_panel("", str(theme_mode or DEFAULT_THEME), refresh_token), "No candidates loaded."
    return (
        _render_dustycult_result_panel(str(candidate_id) if candidate_id else "", str(theme_mode or DEFAULT_THEME), refresh_token),
        _dustycult_config_status_text(),
    )


@app.callback(
    [Output('phoebe-period-days', 'value'),
     Output('phoebe-period-status', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('phoebe-summary', 'n_clicks')],
    prevent_initial_call=False,
)
def update_phoebe_period_control(candidate_id, panel_requested=True):
    """Populate the PHOEBE period field from the best available active-candidate period."""
    if not _details_open(panel_requested):
        return no_update, no_update
    if not candidate_id:
        return None, "No candidates loaded."
    try:
        payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
        period_days, source = infer_period_days(payload)
    except Exception as exc:
        return None, f"Period inference failed: {exc}"
    if period_days is None:
        return None, "No Gaia/VSX/ASAS-SN/ZTF/LS period found. Enter a period in days."
    return period_days, f"Using {source} period."


def _phoebe_fit_callback_impl(clicks, candidate_id, refresh_token, period_days, model_kind):
    if not clicks:
        raise dash.exceptions.PreventUpdate
    if not candidate_id:
        return "No candidate selected.", refresh_token
    try:
        payload, stored_lc_path, source_path = _candidate_context(candidate_id)
        lc_path = _effective_local_lc_path(payload, stored_lc_path=stored_lc_path, source_path=source_path)
        inferred_period, _period_source = infer_period_days(payload)
        manual_period = period_days
        try:
            if inferred_period is not None and period_days is not None and np.isclose(float(period_days), float(inferred_period), rtol=0, atol=1e-8):
                manual_period = None
        except Exception:
            pass
        with closing(db_connect(Path(DB_PATH))) as conn:
            row = run_phoebe_fit(
                conn,
                str(candidate_id),
                payload,
                lc_path=lc_path,
                manual_period_days=manual_period,
                model_kind=model_kind or "detached",
            )
    except Exception as exc:
        row = {"status": "failed", "error": str(exc), "runtime_sec": None}
    next_token = int(refresh_token or 0) + 1
    status = _phoebe_display_status(row)
    if status == "ok":
        return (
            f"PHOEBE fit complete in {_dustycult_float(row.get('runtime_sec'), 3)} s.",
            next_token,
        )
    if status == "warning":
        return (
            f"PHOEBE fit warning: {_phoebe_warning_text(row)}",
            next_token,
        )
    return f"PHOEBE fit failed: {row.get('error') or 'unknown error'}", next_token


if _background_callback_manager is not None:
    @app.callback(
        [Output('phoebe-run-status', 'children'),
         Output('phoebe-refresh-token', 'data')],
        Input('phoebe-fit-btn', 'n_clicks'),
        [State('current-candidate-id', 'data'),
         State('phoebe-refresh-token', 'data'),
         State('phoebe-period-days', 'value'),
         State('phoebe-model-kind', 'value')],
        background=True,
        running=[
            (Output('phoebe-fit-btn', 'disabled'), True, False),
        ],
        prevent_initial_call=True,
    )
    def run_phoebe_fit_callback(clicks, candidate_id, refresh_token, period_days, model_kind):
        return _phoebe_fit_callback_impl(clicks, candidate_id, refresh_token, period_days, model_kind)
else:
    @app.callback(
        [Output('phoebe-run-status', 'children'),
         Output('phoebe-refresh-token', 'data')],
        Input('phoebe-fit-btn', 'n_clicks'),
        [State('current-candidate-id', 'data'),
         State('phoebe-refresh-token', 'data'),
         State('phoebe-period-days', 'value'),
         State('phoebe-model-kind', 'value')],
        prevent_initial_call=True,
    )
    def run_phoebe_fit_callback(clicks, candidate_id, refresh_token, period_days, model_kind):
        return _phoebe_fit_callback_impl(clicks, candidate_id, refresh_token, period_days, model_kind)


@app.callback(
    [Output('phoebe-result-panel', 'children'),
     Output('phoebe-config-status', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('theme-mode-store', 'data'),
     Input('phoebe-refresh-token', 'data'),
     Input('phoebe-summary', 'n_clicks')],
    prevent_initial_call=False,
)
def update_phoebe_result_panel(candidate_id, theme_mode, refresh_token, panel_requested=True):
    """Render PHOEBE availability, fit status, and stored fit plot."""
    if not _details_open(panel_requested):
        return no_update, no_update
    if not candidate_id:
        return _render_phoebe_result_panel("", str(theme_mode or DEFAULT_THEME), refresh_token), "No candidates loaded."
    return (
        _render_phoebe_result_panel(str(candidate_id) if candidate_id else "", str(theme_mode or DEFAULT_THEME), refresh_token),
        _phoebe_config_status_text(),
    )


@app.callback(
    [Output('dustycult-export-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    [Input('dustycult-export-fit-btn', 'n_clicks'),
     Input('dustycult-export-occulter-btn', 'n_clicks')],
    State('current-candidate-id', 'data'),
    prevent_initial_call=True,
)
def export_dustycult_pdf(fit_clicks, occulter_clicks, candidate_id):
    """Export DustyCult fit or occulter model as publication PDF."""
    triggered = callback_context.triggered_id
    if not triggered or (triggered == 'dustycult-export-fit-btn' and not fit_clicks) or (triggered == 'dustycult-export-occulter-btn' and not occulter_clicks):
        return no_update, no_update
    if not candidate_id:
        return no_update, 'No candidate is selected.'
    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            if triggered == 'dustycult-export-occulter-btn':
                fig, mode = _dustycult_occulter_publication_figure(conn, str(candidate_id))
                kind = "occulter"
                width, height = 1200, 620
            else:
                fig, mode = _dustycult_fit_publication_figure(conn, str(candidate_id))
                kind = "fit"
                width, height = 1200, 820
            image_bytes = render_publication_pdf(
                fig,
                title=f"DustyCult {kind.capitalize()}",
                width=width,
                height=height,
                legend_outside=kind != "occulter",
                style=False,
            )
    except Exception as exc:
        return no_update, f'Export failed (DustyCult PDF). {exc}'
    safe_id = slugify_token(candidate_id, fallback="candidate")
    fname = f"malca_dustycult_{kind}_{safe_id}_{mode}.pdf"
    return dcc.send_bytes(image_bytes, fname), f'Exported {fname}'
