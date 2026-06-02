# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _load_sed_figure_for_candidate(candidate_id, extinction_mode, theme_mode):
    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
    with closing(db_connect(Path(DB_PATH))) as conn:
        external_rows = load_sed_rows(conn, str(candidate_id))
        model_curve_rows = load_sed_model_curves(conn, str(candidate_id))
        model_fit_rows = load_sed_model_fits(conn, str(candidate_id))
    return build_sed_figure(
        payload,
        candidate_id=str(candidate_id),
        external_rows=external_rows,
        model_curve_rows=model_curve_rows,
        model_fit_rows=model_fit_rows,
        extinction_mode=str(extinction_mode or "observed"),
        theme=str(theme_mode or DEFAULT_THEME),
    )


def _load_sed_source_status_for_candidate(candidate_id) -> list[dict[str, object]]:
    payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
    with closing(db_connect(Path(DB_PATH))) as conn:
        external_rows = load_sed_rows(conn, str(candidate_id))
    return sed_source_statuses(str(candidate_id), payload=payload, external_rows=external_rows)


def _format_sed_source_labels(labels: list[str], *, limit: int = 8) -> str:
    clean = [str(label) for label in labels if str(label).strip()]
    if not clean:
        return ""
    shown = ", ".join(clean[:limit])
    if len(clean) > limit:
        shown += f", +{len(clean) - limit} more"
    return shown


def _sed_status_labels_for(source_statuses: list[dict[str, object]] | None, status: str) -> list[str]:
    if not source_statuses:
        return []
    labels = []
    for item in source_statuses:
        key = str(item.get("key") or "").strip().lower()
        if key == "payload":
            continue
        if str(item.get("status") or "").strip().lower() == status:
            labels.append(str(item.get("label") or key))
    return labels


def _sed_status_text(
    rows: pd.DataFrame,
    warnings_list: list[str],
    source_statuses: list[dict[str, object]] | None = None,
) -> str:
    if rows is None or rows.empty:
        base = "No SED photometry available. Run `malca sed-photometry ... --review-db <db>` to add external SED catalogs."
    else:
        sources = sorted(str(x) for x in rows["source"].dropna().unique()) if "source" in rows.columns else []
        source_text = _format_sed_source_labels(sources)
        base = f"{len(rows)} SED points"
        if source_text:
            base += f" from {source_text}"
        if bool(rows.get("lambda_l_lambda", pd.Series(dtype=float)).isna().all()):
            base += "; luminosity unavailable without distance"
    miss_text = _format_sed_source_labels(_sed_status_labels_for(source_statuses, "miss"))
    if miss_text:
        base += f"; no match from {miss_text}"
    not_queried_text = _format_sed_source_labels(_sed_status_labels_for(source_statuses, "not_queried"))
    if not_queried_text:
        base += f"; not queried: {not_queried_text}"
    unknown_text = _format_sed_source_labels(_sed_status_labels_for(source_statuses, "unknown"))
    if unknown_text:
        base += f"; source status unknown: {unknown_text}"
    if warnings_list:
        return f"{base}. {' '.join(str(w) for w in warnings_list)}"
    return base


def _render_sed_fetch_provenance(source_statuses: list[dict[str, object]] | None):
    if not source_statuses:
        return html.Div()

    colors = {
        "hit": "#8ee0a1",
        "miss": "#d7b96f",
        "not_queried": "#7d91a6",
        "unknown": "#f59e9e",
    }
    rows = []
    for item in source_statuses:
        status = str(item.get("status") or "unknown").strip().lower()
        label = str(item.get("label") or item.get("key") or "source")
        n_rows = int(item.get("n_rows") or 0)
        source_names = item.get("source_names") or []
        bands = item.get("bands") or []
        storage = str(item.get("storage") or "").strip()
        message = str(item.get("message") or "").strip()
        if status == "hit":
            status_text = f"{n_rows} row" + ("" if n_rows == 1 else "s")
        elif status == "miss":
            status_text = "no match"
        elif status == "not_queried":
            status_text = "not queried"
        else:
            status_text = "unknown"
        detail_parts = []
        if source_names and source_names != [label]:
            detail_parts.append(_format_sed_source_labels([str(x) for x in source_names], limit=4))
        if bands:
            detail_parts.append(_format_sed_source_labels([str(x) for x in bands], limit=5))
        if storage:
            detail_parts.append(storage)
        if message:
            detail_parts.append(message)
        rows.append(html.Div([
            html.Span(label, style={'fontWeight': '600', 'color': '#c8d8e6'}),
            html.Span(status_text, style={'color': colors.get(status, '#f59e9e')}),
            html.Span(" | ".join(detail_parts), style={'color': '#7d91a6', 'minWidth': 0, 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '92px 72px minmax(0, 1fr)',
            'gap': '8px',
            'alignItems': 'baseline',
        }))
    return html.Details([
        html.Summary('SED Fetch Provenance', style={'cursor': 'pointer', 'color': '#9fb6cb'}),
        html.Div(rows, style={
            'display': 'grid',
            'gap': '3px',
            'marginTop': '6px',
            'fontSize': '10px',
            'lineHeight': '1.35',
        }),
    ], open=False, style={
        'border': '1px solid rgba(125, 145, 166, 0.25)',
        'borderRadius': '6px',
        'padding': '6px 8px',
        'fontSize': '10px',
        'background': 'rgba(8, 16, 24, 0.45)',
    })


@app.callback(
    [Output('sed-plot-panel', 'children'),
     Output('sed-status', 'children')],
    [Input('current-candidate-id', 'data'),
     Input('sed-extinction-mode', 'value'),
     Input('theme-mode-store', 'data'),
     Input('sed-details', 'open')],
    prevent_initial_call=False,
)
def update_sed_panel(candidate_id, extinction_mode, theme_mode, details_open=True):
    """Render SED photometry for the current candidate."""
    if not _details_open(details_open):
        return no_update, no_update
    if not candidate_id:
        return [], 'No candidates loaded.'
    try:
        fig, rows, warnings_list = _load_sed_figure_for_candidate(candidate_id, extinction_mode, theme_mode)
    except Exception as exc:
        return [], f"SED rendering failed: {exc}"
    try:
        source_statuses = _load_sed_source_status_for_candidate(candidate_id)
    except Exception as exc:
        source_statuses = []
        warnings_list = [*warnings_list, f"SED fetch provenance unavailable: {exc}"]
    graph = dcc.Graph(
        id='sed-plot',
        figure=fig,
        mathjax=True,
        config=graph_config_without_image_export({'displayModeBar': True, 'responsive': True}),
        style={'height': '420px'},
    )
    return [graph, _render_sed_fetch_provenance(source_statuses)], _sed_status_text(rows, warnings_list, source_statuses)


@app.callback(
    [Output('sed-export-download', 'data'),
     Output('notification', 'children', allow_duplicate=True)],
    Input('export-sed-plot', 'n_clicks'),
    [State('current-candidate-id', 'data'),
     State('sed-extinction-mode', 'value'),
     State('theme-mode-store', 'data')],
    prevent_initial_call=True,
)
def export_sed_plot(n_clicks, candidate_id, extinction_mode, theme_mode):
    """Export the current candidate SED as PDF."""
    if not n_clicks:
        return no_update, no_update
    if not candidate_id:
        return no_update, 'No candidate is selected.'
    try:
        fig, _rows, _warnings_list = _load_sed_figure_for_candidate(candidate_id, extinction_mode, "white")
        export_fig = go.Figure(fig)
        for trace in export_fig.data:
            trace_name = str(getattr(trace, "name", "") or "")
            if "Castelli/Kurucz" in trace_name and getattr(trace, "line", None) is not None:
                trace.line.color = "#111827"
                trace.line.width = 2.6
            if getattr(trace, "marker", None) is not None:
                trace.marker.size = 11.0
                trace.marker.opacity = 1.0
                if getattr(trace.marker, "line", None) is not None:
                    trace.marker.line.color = "#111827"
                    trace.marker.line.width = 0.7
        image_bytes = render_publication_pdf(
            export_fig,
            title='Spectral Energy Distribution',
            width=1200,
            height=820,
            legend_outside=True,
            right_margin=285,
            top_margin=92,
            bottom_margin=84,
            left_margin=96,
        )
    except Exception as exc:
        return no_update, f'Export failed (SED PDF). {exc}'
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_id)).strip("_") or "unknown"
    mode = str(extinction_mode or "observed").replace("-", "_")
    fname = f"malca_sed_{safe_id}_{mode}.pdf"
    return dcc.send_bytes(image_bytes, fname), f'Exported {fname}'

