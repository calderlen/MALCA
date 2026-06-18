# Spectrum viewer callbacks — exec'd into review app namespace.


def _spectrum_sources_for_candidate(candidate_id):
    """Return list of (survey, row_dict) tuples for spectra with fetchable backends."""
    from malca.enrich.spectrum_fetch import SURVEY_BACKEND_MAP, FetchBackend

    lookup_keys = _lookup_candidate_keys(candidate_id) if callable(globals().get('_lookup_candidate_keys')) else [str(candidate_id)]
    run_dir = _run_dir_from_source_path()
    for key in lookup_keys:
        rows = _load_spectra_rows(key, run_dir)
        if not rows.empty:
            break
    else:
        return []

    sources = []
    for _, r in rows.iterrows():
        survey = str(r.get("survey", ""))
        sources.append((survey, r.to_dict()))
    return sources


def _load_spectrum_for_row(row_dict, survey):
    """Fetch or load cached spectrum data for a single spectra_long row."""
    import pandas as pd
    from pathlib import Path
    from malca.enrich.spectrum_fetch import fetch_spectrum, FetchStatus

    run_dir = _run_dir_from_source_path()
    cache_dir = Path(run_dir) / "results" / "spectra_enrichment" / "spectra" if run_dir else None

    row = pd.Series(row_dict)
    result = fetch_spectrum(row, survey_key=survey, cache_dir=cache_dir)
    return result


@app.callback(
    Output('spectrum-source-dropdown', 'options'),
    Output('spectrum-source-dropdown', 'value'),
    Input('current-candidate-id', 'data'),
    Input('spectrum-summary', 'n_clicks'),
    prevent_initial_call=False,
)
def update_spectrum_sources(candidate_id, panel_clicks):
    if not _details_open(panel_clicks):
        return no_update, no_update
    if not candidate_id:
        return [], None

    sources = _spectrum_sources_for_candidate(candidate_id)
    if not sources:
        return [{'label': 'No spectra available', 'value': '__none__'}], '__none__'

    options = [{'label': survey, 'value': f"{i}:{survey}"} for i, (survey, _) in enumerate(sources)]
    return options, options[0]['value'] if options else None


@app.callback(
    Output('spectrum-plot-panel', 'children'),
    Output('spectrum-status', 'children'),
    Input('spectrum-source-dropdown', 'value'),
    Input('theme-mode-store', 'data'),
    Input('current-candidate-id', 'data'),
    Input('spectrum-summary', 'n_clicks'),
    prevent_initial_call=False,
)
def update_spectrum_panel(source_value, theme_mode, candidate_id, panel_clicks):
    if not _details_open(panel_clicks):
        return no_update, no_update
    if not candidate_id or not source_value or source_value == '__none__':
        return _lazy_panel_placeholder('Select a spectrum source.'), 'No spectrum selected.'

    sources = _spectrum_sources_for_candidate(candidate_id)
    if not sources:
        return _lazy_panel_placeholder('No spectra available for this candidate.'), 'No spectra available.'

    try:
        idx_str, survey = source_value.split(':', 1)
        idx = int(idx_str)
        if idx >= len(sources):
            return _lazy_panel_placeholder('Source no longer available.'), 'Source changed.'
        _, row_dict = sources[idx]
    except (ValueError, IndexError):
        return _lazy_panel_placeholder('Invalid source selection.'), 'Error parsing source.'

    from malca.enrich.spectrum_fetch import FetchStatus
    result = _load_spectrum_for_row(row_dict, survey)

    if result.status == FetchStatus.OK and result.data is not None:
        from malca.review.spectrum_plot import build_spectrum_figure
        import numpy as np

        redshift = row_dict.get("spectrum_redshift")
        if redshift is not None:
            try:
                redshift = float(redshift)
                if not np.isfinite(redshift):
                    redshift = None
            except (ValueError, TypeError):
                redshift = None

        fig = build_spectrum_figure(
            result.data.wavelength,
            result.data.flux,
            flux_err=result.data.flux_err,
            survey=survey,
            candidate_id=str(candidate_id),
            redshift=redshift,
            theme=str(theme_mode or DEFAULT_THEME),
        )
        graph = dcc.Graph(
            id='spectrum-plot',
            figure=fig,
            mathjax=True,
            config=graph_config_without_image_export({'displayModeBar': True, 'responsive': True}),
            style={'height': '420px'},
        )
        n_pts = len(result.data.wavelength)
        wl_min = float(result.data.wavelength.min())
        wl_max = float(result.data.wavelength.max())
        status = f"{survey}: {n_pts} points, {wl_min:.0f}–{wl_max:.0f} Å"
        return graph, status

    if result.status == FetchStatus.LINK_ONLY:
        link = result.link or row_dict.get("link", "")
        msg = result.message or "Flux data not available for this source."
        children = [html.Div(msg, style={'fontSize': '11px', 'padding': '8px'})]
        if link:
            children.append(html.A(
                'Open in archive',
                href=str(link),
                target='_blank',
                rel='noopener noreferrer',
                style={'fontSize': '11px', 'color': '#5eead4', 'padding': '0 8px'},
            ))
        return html.Div(children), f"Link only: {survey}"

    if result.status == FetchStatus.AUTH_REQUIRED:
        return html.Div(
            f"Authentication required for {survey}. Set ESO_USERNAME/ESO_PASSWORD environment variables.",
            style={'fontSize': '11px', 'padding': '8px', 'color': '#f59e0b'},
        ), f"Auth required: {survey}"

    return _lazy_panel_placeholder(f"Could not load spectrum: {result.message}"), f"Error: {result.message}"


@app.callback(
    Output('spectrum-export-download', 'data'),
    Output('notification', 'children', allow_duplicate=True),
    Input('export-spectrum-plot', 'n_clicks'),
    State('spectrum-source-dropdown', 'value'),
    State('current-candidate-id', 'data'),
    State('theme-mode-store', 'data'),
    prevent_initial_call=True,
)
def export_spectrum_pdf(n_clicks, source_value, candidate_id, theme_mode):
    if not n_clicks:
        return no_update, no_update
    if not candidate_id or not source_value or source_value == '__none__':
        return no_update, 'No spectrum selected.'

    sources = _spectrum_sources_for_candidate(candidate_id)
    try:
        idx_str, survey = source_value.split(':', 1)
        idx = int(idx_str)
        _, row_dict = sources[idx]
    except (ValueError, IndexError):
        return no_update, 'Source not found.'

    from malca.enrich.spectrum_fetch import FetchStatus
    result = _load_spectrum_for_row(row_dict, survey)
    if result.status != FetchStatus.OK or result.data is None:
        return no_update, f'No flux data to export for {survey}.'

    try:
        from malca.review.spectrum_plot import build_spectrum_figure
        fig = build_spectrum_figure(
            result.data.wavelength, result.data.flux, flux_err=result.data.flux_err,
            survey=survey, candidate_id=str(candidate_id), theme='white',
        )
        image_bytes = render_publication_pdf(fig, title='Spectrum', width=1200, height=820)
    except Exception as exc:
        return no_update, f'Export failed: {exc}'

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_id)).strip("_") or "unknown"
    return dcc.send_bytes(image_bytes, f"spectrum_{safe_id}_{survey}.pdf"), no_update


@app.callback(
    Output('spectrum-export-download', 'data', allow_duplicate=True),
    Output('notification', 'children', allow_duplicate=True),
    Input('export-spectrum-csv', 'n_clicks'),
    State('spectrum-source-dropdown', 'value'),
    State('current-candidate-id', 'data'),
    prevent_initial_call=True,
)
def export_spectrum_csv(n_clicks, source_value, candidate_id):
    if not n_clicks:
        return no_update, no_update
    if not candidate_id or not source_value or source_value == '__none__':
        return no_update, 'No spectrum selected.'

    sources = _spectrum_sources_for_candidate(candidate_id)
    try:
        idx_str, survey = source_value.split(':', 1)
        idx = int(idx_str)
        _, row_dict = sources[idx]
    except (ValueError, IndexError):
        return no_update, 'Source not found.'

    from malca.enrich.spectrum_fetch import FetchStatus
    result = _load_spectrum_for_row(row_dict, survey)
    if result.status != FetchStatus.OK or result.data is None:
        return no_update, f'No flux data to export for {survey}.'

    import pandas as pd
    df = pd.DataFrame({'wavelength': result.data.wavelength, 'flux': result.data.flux})
    if result.data.flux_err is not None:
        df['flux_err'] = result.data.flux_err

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_id)).strip("_") or "unknown"
    return dcc.send_data_frame(df.to_csv, f"spectrum_{safe_id}_{survey}.csv", index=False), no_update


@app.callback(
    Output('spectrum-export-download', 'data', allow_duplicate=True),
    Output('notification', 'children', allow_duplicate=True),
    Input('export-spectrum-fits', 'n_clicks'),
    State('spectrum-source-dropdown', 'value'),
    State('current-candidate-id', 'data'),
    prevent_initial_call=True,
)
def export_spectrum_fits(n_clicks, source_value, candidate_id):
    if not n_clicks:
        return no_update, no_update
    if not candidate_id or not source_value or source_value == '__none__':
        return no_update, 'No spectrum selected.'

    sources = _spectrum_sources_for_candidate(candidate_id)
    try:
        idx_str, survey = source_value.split(':', 1)
        idx = int(idx_str)
        _, row_dict = sources[idx]
    except (ValueError, IndexError):
        return no_update, 'Source not found.'

    from malca.enrich.spectrum_fetch import FetchStatus
    result = _load_spectrum_for_row(row_dict, survey)
    if result.status != FetchStatus.OK or result.data is None:
        return no_update, f'No flux data to export for {survey}.'

    try:
        import tempfile
        from astropy.io import fits
        from astropy.table import Table

        t = Table()
        t['wavelength'] = result.data.wavelength
        t['flux'] = result.data.flux
        if result.data.flux_err is not None:
            t['flux_err'] = result.data.flux_err

        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            t.write(tmp.name, format='fits', overwrite=True)
            with open(tmp.name, 'rb') as f:
                fits_bytes = f.read()

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_id)).strip("_") or "unknown"
        return dcc.send_bytes(fits_bytes, f"spectrum_{safe_id}_{survey}.fits"), no_update
    except ImportError:
        return no_update, 'astropy required for FITS export.'
    except Exception as exc:
        return no_update, f'FITS export failed: {exc}'
