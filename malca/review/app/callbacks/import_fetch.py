# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output('import-path', 'value'),
     Output('sidebar-status', 'children', allow_duplicate=True)],
    Input('keyboard-init', 'n_intervals'),
    State('import-path', 'value'),
    prevent_initial_call='initial_duplicate',
)
def auto_populate_detected_files(_n_intervals, current_import_path):
    """Auto-detect linked run files on app startup and fill import path."""
    if current_import_path:
        return no_update, no_update

    restored_path = None
    with closing(db_connect(Path(DB_PATH))) as conn:
        restored_path = str(load_app_state(conn, "last_input_file", "") or "").strip()

    run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
    if run_dir:
        try:
            detected = detect_run_directory_files(run_dir)
            detected_sources = _candidate_files_for_run_dir(run_dir)
        except Exception as e:
            detected = {'candidates': None, 'warnings': [f"Auto-detect failed: {str(e)}"]}
            detected_sources = []

        if detected_sources:
            resolved_candidates = "\n".join(str(p) for p in detected_sources)
            vetting_mode = _vetting_mode_for_sources(resolved_candidates)
            with closing(db_connect(Path(DB_PATH))) as conn:
                save_app_state(conn, "last_input_file", resolved_candidates)
            return resolved_candidates, (
                f"✓ Auto-detected {len(detected_sources)} candidates file(s): {_summarize_source_paths(detected_sources)} | "
                f"Vetting mode: {vetting_mode}"
            )

        warnings = detected.get('warnings') or []
        restored_sources = _resolve_import_sources(restored_path)
        if restored_sources:
            restored_mode = _vetting_mode_for_sources(restored_path)
            restored_text = "\n".join(str(p) for p in restored_sources)
            return restored_text, (
                f"⚠ {warnings[0]} | restored last candidates: {_summarize_source_paths(restored_sources)} | "
                f"Vetting mode: {restored_mode}"
                if warnings else (
                    f"✓ Restored last candidates: {_summarize_source_paths(restored_sources)} | "
                    f"Vetting mode: {restored_mode}"
                )
            )

        if warnings:
            return no_update, f"⚠ {warnings[0]}"

    restored_sources = _resolve_import_sources(restored_path)
    if restored_sources:
        restored_mode = _vetting_mode_for_sources(restored_path)
        restored_text = "\n".join(str(p) for p in restored_sources)
        return restored_text, (
            f"✓ Restored last candidates: {_summarize_source_paths(restored_sources)} | "
            f"Vetting mode: {restored_mode}"
        )

    return no_update, "⚠ Could not auto-detect candidates; set import path manually."


@app.callback(
    Output('queue-source-path', 'data'),
    [Input('keyboard-init', 'n_intervals'),
     Input('import-path', 'value')],
    prevent_initial_call=False,
)
def update_queue_source_scope(_n_intervals, import_path):
    """Scope queue to the active run bundle token when possible."""
    if _resolve_run_dir_from_db_path(DB_PATH) is not None:
        return ''
    # Prefer run-dir scope when --plot-dir is set so the queue loads (diagnostic/external panels need current-candidate-id).
    run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
    if run_dir is not None:
        return {'source_path_like_any': [run_dir.name], 'label': run_dir.name}
    # Standalone mode (no --plot-dir): don't scope queue to a restored import path, so merged DBs show full queue.
    if PLOT_DIR is None:
        return ''
    scope = _queue_scope_from_import_text(import_path)
    if scope:
        return scope
    return ''


@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('import-path', 'value'),
    prevent_initial_call=True,
)
def show_vetting_mode_status(import_path):
    """Show whether import will use vetted input, cache, or re-vetting."""
    if not import_path:
        return no_update
    mode = _vetting_mode_for_sources(import_path)
    return f"Vetting mode: {mode}"


def _import_sources_with_options(
    sources: list[Path],
    *,
    characterize_on,
    crossmatch,
    gaia_cache,
    chunk_size,
    dust_on,
    starhorse,
    vet_on,
    lc_mode,
    current_trigger,
) -> tuple[str, int]:
    """Import one or more candidate/light-curve sources into the review DB."""
    if not sources:
        raise ValueError("No import sources resolved")

    raw_lc_mode = 'yes' in (lc_mode or [])
    with closing(db_connect(Path(DB_PATH))) as conn:
        enable_characterize = 'yes' in (characterize_on or [])
        enable_vetting = 'yes' in (vet_on or [])

        if raw_lc_mode:
            total_rows = 0
            total_new = 0
            for src in sources:
                n_rows, n_new = import_lightcurve_files(
                    conn, src,
                    characterize=enable_characterize,
                    vet=enable_vetting,
                )
                total_rows += int(n_rows)
                total_new += int(n_new)
            save_app_state(conn, "last_input_file", "\n".join(str(p) for p in sources))
            return (
                f"✓ LC import: {total_rows} rows ({total_new} new) from {len(sources)} file(s)",
                (current_trigger or 0) + 1,
            )

        vetting_mode = _vetting_mode_for_sources("\n".join(str(p) for p in sources)) if enable_vetting else "vetting disabled"
        total_rows = 0
        total_new = 0
        for src in sources:
            df = load_candidates_file(src)
            n_rows, n_new = import_candidates(
                conn, df, str(src),
                characterize_before_import=enable_characterize,
                characterize_crossmatch=Path(crossmatch).expanduser() if enable_characterize and crossmatch else None,
                characterize_cache=Path(gaia_cache).expanduser() if enable_characterize and gaia_cache else None,
                characterize_chunk_size=int(chunk_size) if enable_characterize and chunk_size else GAIA_CHUNK_SIZE,
                characterize_dust='yes' in (dust_on or []) if enable_characterize else False,
                characterize_starhorse=starhorse.strip() if enable_characterize and starhorse and starhorse.strip() else None,
                vet_before_import=enable_vetting,
            )
            total_rows += int(n_rows)
            total_new += int(n_new)
        save_app_state(conn, "last_input_file", "\n".join(str(p) for p in sources))
        return (
            f"✓ Imported {total_rows} rows ({total_new} new) from {len(sources)} file(s) | Vetting mode: {vetting_mode}",
            (current_trigger or 0) + 1,
        )


# Import candidates
@app.callback(
    [Output('sidebar-status', 'children', allow_duplicate=True),
     Output('import-trigger', 'data')],
    Input('import-btn', 'n_clicks'),
    [State('import-path', 'value'),
     State('characterize-on-import', 'value'),
     State('characterize-crossmatch', 'value'),
     State('characterize-gaia-cache', 'value'),
     State('characterize-chunk-size', 'value'),
     State('characterize-dust', 'value'),
     State('characterize-starhorse', 'value'),
     State('vet-on-import', 'value'),
     State('import-lc-mode', 'value'),
     State('import-trigger', 'data')],
    prevent_initial_call=True
)
def import_candidates_callback(n_clicks, import_path, characterize_on,
                               crossmatch, gaia_cache, chunk_size, dust_on, starhorse, vet_on,
                               lc_mode, current_trigger):
    """Import candidates from file."""
    if not n_clicks or not import_path:
        return no_update, no_update

    raw_lc_mode = 'yes' in (lc_mode or [])
    sources = _resolve_import_sources(import_path, allow_run_dirs=not raw_lc_mode)
    if not sources:
        sources = [Path(str(import_path)).expanduser()]

    try:
        return _import_sources_with_options(
            sources,
            characterize_on=characterize_on,
            crossmatch=crossmatch,
            gaia_cache=gaia_cache,
            chunk_size=chunk_size,
            dust_on=dust_on,
            starhorse=starhorse,
            vet_on=vet_on,
            lc_mode=lc_mode,
            current_trigger=current_trigger,
        )
    except Exception as e:
        return f"✗ Import failed: {str(e)}", no_update


def _try_float(value) -> float | None:
    """Parse finite float values; return None on invalid input."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_first_nonempty_id(row: dict, keys: tuple[str, ...]) -> str | None:
    """Return first non-empty identifier found in row for candidate key names."""
    for key in keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        try:
            if pd.isna(raw):
                continue
        except Exception:
            pass
        text = _format_large_integer_like_display(raw).strip()
        if text and text.lower() not in {'nan', 'none'}:
            return text
    return None


def _nearest_catalog_match_by_coords(catalog_df: pd.DataFrame, ra_deg: float, dec_deg: float) -> tuple[dict | None, float | None]:
    """Return nearest cone-search row and separation (arcsec) for target coords."""
    if catalog_df is None or catalog_df.empty:
        return None, None

    ra_col = next((c for c in ('ra_deg', 'ra', 'RAJ2000', 'RA', 'raj2000') if c in catalog_df.columns), None)
    dec_col = next((c for c in ('dec_deg', 'dec', 'DEJ2000', 'DEC', 'Dec', 'dej2000') if c in catalog_df.columns), None)
    if ra_col is None or dec_col is None:
        return None, None

    ra_vals = pd.to_numeric(catalog_df[ra_col], errors='coerce').to_numpy(dtype=float)
    dec_vals = pd.to_numeric(catalog_df[dec_col], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(ra_vals) & np.isfinite(dec_vals)
    if not np.any(valid):
        return None, None

    ra0 = float(ra_deg) % 360.0
    dec0 = float(dec_deg)
    ra = np.mod(ra_vals[valid], 360.0)
    dec = dec_vals[valid]

    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    dra = np.deg2rad(((ra - ra0 + 180.0) % 360.0) - 180.0)
    dec_rad = np.deg2rad(dec)

    sin_ddec = np.sin((dec_rad - dec0_rad) / 2.0)
    sin_dra = np.sin(dra / 2.0)
    a = sin_ddec * sin_ddec + np.cos(dec0_rad) * np.cos(dec_rad) * sin_dra * sin_dra
    a = np.clip(a, 0.0, 1.0)
    sep_rad = 2.0 * np.arcsin(np.sqrt(a))
    sep_arcsec = np.rad2deg(sep_rad) * 3600.0

    nearest_local = int(np.argmin(sep_arcsec))
    valid_indices = np.flatnonzero(valid)
    nearest_idx = int(valid_indices[nearest_local])
    return catalog_df.iloc[nearest_idx].to_dict(), float(sep_arcsec[nearest_local])


def _fetch_candidate_impl(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger):
    """Core fetch logic; set_progress is optional for streaming status to GUI."""
    from malca.review.fetch import (
        fetch_and_analyze_by_gaia_id,
        fetch_and_analyze_by_id,
        fetch_cone_search,
    )

    def progress(msg):
        if set_progress and msg:
            try:
                set_progress(msg[:400] if len(msg) > 400 else msg)
            except Exception:
                pass

    if not n_clicks or not fetch_query:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    query = fetch_query.strip()
    if not query:
        return "✗ Query cannot be empty", no_update, no_update, no_update, no_update, no_update, no_update

    try:
        effective_backend = str(fetch_backend or DEFAULT_FETCH_BACKEND).strip().lower()
        if effective_backend not in {'skypatrol2', 'skypatrol1'}:
            effective_backend = DEFAULT_FETCH_BACKEND

        effective_fetch_type = str(fetch_type or 'asassn')
        effective_query = query
        cone_status_text = ''

        if effective_fetch_type == 'coords':
            progress('Searching SkyPatrol cone...')

            parts = [p for p in query.replace(',', ' ').split() if p]
            if len(parts) < 2:
                return "✗ Enter coordinates as 'RA Dec [radius_arcsec]'", no_update, no_update, '', None, no_update, no_update

            ra = _try_float(parts[0])
            dec = _try_float(parts[1])
            if ra is None or dec is None:
                return "✗ Invalid RA/Dec. Use decimal degrees: 'RA Dec [radius_arcsec]'", no_update, no_update, '', None, no_update, no_update
            if dec < -90.0 or dec > 90.0:
                return '✗ Dec must be between -90 and +90 degrees', no_update, no_update, '', None, no_update, no_update

            radius = 5.0
            if len(parts) > 2:
                radius = _try_float(parts[2])
                if radius is None or radius <= 0.0:
                    return '✗ Radius must be a positive number [arcsec]', no_update, no_update, '', None, no_update, no_update

            ra = ra % 360.0
            catalog_df = fetch_cone_search(ra, dec, radius_arcsec=radius, backend=effective_backend)
            if catalog_df.empty:
                return f'✗ No sources found within {radius:.2f}"', no_update, no_update, '', None, no_update, no_update

            nearest_row, sep_arcsec = _nearest_catalog_match_by_coords(catalog_df, ra, dec)
            if nearest_row is None or sep_arcsec is None:
                return '✗ Cone search returned rows without usable coordinates', no_update, no_update, '', None, no_update, no_update

            nearest_asas = _extract_first_nonempty_id(nearest_row, ('asas_sn_id', 'asassn_id'))
            nearest_gaia = _extract_first_nonempty_id(nearest_row, ('gaia_id', 'source_id'))

            if nearest_asas:
                effective_fetch_type = 'asassn'
                effective_query = nearest_asas
            elif nearest_gaia:
                effective_fetch_type = 'gaia'
                effective_query = nearest_gaia
            else:
                return '✗ Nearest cone match has no ASAS-SN ID or Gaia ID to fetch', no_update, no_update, '', None, no_update, no_update

            n_found = int(len(catalog_df))
            match_word = 'match' if n_found == 1 else 'matches'
            cone_status_text = (
                f' | cone nearest {effective_query} at {sep_arcsec:.2f}" '
                f'({n_found} {match_word} in {radius:.2f}")'
            )
            progress(f'SkyPatrol cone resolved: {effective_query} ({sep_arcsec:.2f}")')

        progress("Fetching light curve...")
        if effective_fetch_type == 'gaia':

            df, lc_path = fetch_and_analyze_by_gaia_id(effective_query, run_stats=True, backend=effective_backend)
        else:

            df, lc_path = fetch_and_analyze_by_id(effective_query, run_stats=True, backend=effective_backend)

        if df is None or df.empty:
            return f"✗ No data for {effective_query}", no_update, no_update, '', None, no_update, no_update

        progress("Importing basic light curve...")
        
        # We NEVER run characterization/vetting in the fetch callback anymore,
        # so the GUI can render the light curve IMMEDIATELY.
        with closing(db_connect(Path(DB_PATH))) as conn:
            n_rows, n_new = import_candidates(
                conn, df, source_path=f"fetch://{effective_backend}/{effective_fetch_type}/{effective_query}",
                characterize_before_import=False,
                vet_before_import=False,
            )

        cid = str(df.iloc[0]['candidate_id']) if 'candidate_id' in df.columns else effective_query
        _index_external_lc_paths.cache_clear()
        _index_external_lc_paths_from_root.cache_clear()
        
        auto_run = no_update
        if fetch_mode in ('full', 'full_ext'):
            auto_run = {'candidate_id': cid, 'mode': fetch_mode, 'ts': time.time()}

        status = f"✓ Added {effective_query} ({n_new} new) [{effective_backend}]{cone_status_text}"
        return (
            status,
            no_update,
            auto_run,
            '',
            None,
            {'candidate_ids': [cid], 'queue_size': 1, 'filter_hash': f'fetch:{cid}'},
            0,
        )

    except Exception as e:

        traceback.print_exc()
        return f"✗ Fetch failed: {str(e)}", no_update, no_update, '', None, no_update, no_update


# Fetch and Analyze Candidate
if _background_callback_manager is not None:
    @app.callback(
        [Output('fetch-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('pending-auto-run', 'data', allow_duplicate=True),
         Output('cone-results-container', 'children'),
         Output('cone-results-data', 'data'),
         Output('queue-data', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        Input('fetch-btn', 'n_clicks'),
        [State('fetch-type', 'value'),
         State('fetch-query', 'value'),
         State('fetch-mode', 'value'),
         State('fetch-backend', 'value'),
         State('import-trigger', 'data')],
        background=True,
        running=[
            (Output('fetch-btn', 'disabled'), True, False),
        ],
        progress=[Output('fetch-status', 'children')],
        prevent_initial_call=True,
    )
    def fetch_candidate_callback(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger):
        return _fetch_candidate_impl(set_progress, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger)
else:
    @app.callback(
        [Output('fetch-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('pending-auto-run', 'data', allow_duplicate=True),
         Output('cone-results-container', 'children'),
         Output('cone-results-data', 'data'),
         Output('queue-data', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        Input('fetch-btn', 'n_clicks'),
        [State('fetch-type', 'value'),
         State('fetch-query', 'value'),
         State('fetch-mode', 'value'),
         State('fetch-backend', 'value'),
         State('import-trigger', 'data')],
        prevent_initial_call=True,
    )
    def fetch_candidate_callback(n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger):
        return _fetch_candidate_impl(None, n_clicks, fetch_type, fetch_query, fetch_mode, fetch_backend, current_trigger)


def _pipeline_status_chip_elements(candidate_id) -> list:
    """Build pipeline status chips for the active candidate."""
    chips = []
    stage_labels = {
        'sed_photometry': 'SED',
        'sed_model_fit': 'SED model',
        'external_lcs': 'External LCs',
        'multi_survey_features': 'Multi-survey',
        'periodicity': 'Periodicity',
    }
    if candidate_id is None:
        return chips
    try:
        payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
        status = detect_pipeline_status(payload)
        with closing(db_connect(Path(DB_PATH))) as conn:
            status['sed_photometry'] = detect_sed_photometry_status(conn, str(candidate_id), payload)
            status['sed_model_fit'] = detect_sed_model_status(conn, str(candidate_id), payload)

        periodicity_sig_cols = (
            'periodicity_score',
            'lsp_period',
            'lsp_power',
            'lsp_is_significant',
        )
        periodicity_present = sum(
            1
            for col in periodicity_sig_cols
            if col in payload and payload[col] is not None
            and not (isinstance(payload[col], float) and np.isnan(payload[col]))
        )
        if periodicity_present == 0:
            status['periodicity'] = 'missing'
        elif periodicity_present == len(periodicity_sig_cols):
            status['periodicity'] = 'complete'
        else:
            status['periodicity'] = 'partial'

        color_map = {'complete': '#2d6a2d', 'partial': '#6a5c2d', 'missing': '#444'}
        for stage, state in status.items():
            chips.append(html.Span(
                f"{'●' if state == 'complete' else '○'} {stage_labels.get(stage, stage.capitalize())}",
                style={
                    'padding': '1px 6px',
                    'borderRadius': '8px',
                    'backgroundColor': color_map.get(state, '#444'),
                    'color': '#e0e0e0' if state != 'missing' else '#666',
                    'fontSize': '10px',
                },
            ))
    except Exception:
        return []
    return chips


# Pipeline status chips (updated when candidate changes and stages complete)
@app.callback(
    Output('pipeline-status-chips', 'children'),
    [Input('queue-data', 'modified_timestamp'),
     Input('current-candidate-id', 'data'),
     Input('pipeline-progress-trigger', 'data'),
     Input('pipeline-run-status', 'children')],
    prevent_initial_call=True
)
def update_pipeline_status_chips(_queue_data_ts, candidate_id, _pipeline_progress, _pipeline_status_text):
    """Show pipeline stage completion status for the current candidate."""
    return _pipeline_status_chip_elements(candidate_id)


# Cascade auto-run once queue updates for fetched candidate
@app.callback(
    Output('auto-run-pipeline-trigger', 'data', allow_duplicate=True),
    [Input('queue-data', 'modified_timestamp'),
     Input('current-candidate-id', 'data')],
    State('pending-auto-run', 'data'),
    prevent_initial_call=True,
)
def maybe_cascade_auto_run(_queue_data_ts, candidate_id, pending_auto_run):
    """Emit pending auto-run token when fetched candidate enters queue."""
    if candidate_id is None or not pending_auto_run:
        return no_update

    triggered_ids = {
        item['prop_id'].split('.')[0]
        for item in (callback_context.triggered or [])
        if item.get('prop_id')
    }
    if 'queue-data' not in triggered_ids:
        return no_update
    if str(pending_auto_run.get('candidate_id')) != str(candidate_id):
        return no_update

    try:
        payload, _stored_lc_path, _source_path = _candidate_context(candidate_id)
        status = detect_pipeline_status(payload)
        if any(state in {'missing', 'partial'} for state in status.values()):
            return pending_auto_run
    except Exception:
        return no_update
    return no_update


@app.callback(
    Output('pipeline-module-log-panel', 'children'),
    Input('pipeline-module-log', 'data'),
    prevent_initial_call=False,
)
def render_pipeline_module_log(log_data):
    """Render temporary in-GUI module run log lines."""
    lines = []
    if isinstance(log_data, dict):
        raw = log_data.get('lines')
        if isinstance(raw, list):
            lines = [str(x) for x in raw if x is not None]
    if not lines:
        return "No pipeline run log yet."
    return "\n".join(lines[-300:])


