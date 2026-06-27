# This file was mechanically split from malca.review.app; preserve behavior when editing.

@app.callback(
    Output('batch-export-status', 'children'),
    Input('batch-export-btn', 'n_clicks'),
    State('batch-export-target', 'value'),
    State('batch-export-path', 'value'),
    State('queue-data', 'data'),
    State('plot-render-request', 'data'),
    State('pdm-min-period', 'value'),
    State('pdm-max-period', 'value'),
    State('period-method', 'value'),
    background=True,
    running=[
        (Output('batch-export-btn', 'disabled'), True, False),
    ],
    progress=[Output('batch-export-status', 'children')],
    prevent_initial_call=True
)
def handle_batch_export(
    set_progress,
    n_clicks,
    target,
    out_path_str,
    queue_data,
    render_request,
    min_period,
    max_period,
    period_method,
):
    """Batch export candidate PDFs to a specified directory."""
    if not n_clicks:
        return ""
    if not out_path_str or not out_path_str.strip():
        return "Error: Please specify an export directory."

    try:
        out_dir = Path(out_path_str.strip()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"Error creating dir: {e}"

    if not isinstance(queue_data, dict) or not queue_data.get('candidate_ids'):
        return "Error: Queue is empty."

    candidate_ids = list(queue_data['candidate_ids'])

    # Optionally filter to unreviewed only
    if str(target) == 'unreviewed':
        set_progress("Fetching review status for queue...")
        unreviewed_ids = []
        try:
            with closing(db_connect(Path(DB_PATH))) as conn:
                chunk_size = 999
                for i in range(0, len(candidate_ids), chunk_size):
                    chunk = candidate_ids[i:i + chunk_size]
                    qmarks = ','.join('?' * len(chunk))
                    rows = conn.execute(
                        f"SELECT candidate_id, review_status FROM candidates WHERE candidate_id IN ({qmarks})",
                        chunk
                    ).fetchall()
                    for row in rows:
                        status = (row[1] or '').strip().lower()
                        if status != 'done':
                            unreviewed_ids.append(row[0])
            candidate_ids = [cid for cid in candidate_ids if cid in set(unreviewed_ids)]
        except Exception as e:
            return f"Error filtering queue: {e}"

    total = len(candidate_ids)
    if total == 0:
        return "No candidates match the selected criteria."

    set_progress(f"Exporting 0/{total} (PDF ok: 0, phase found: 0, phase skipped: 0, failed: 0)...")

    success = 0
    failed = 0
    phase_found = 0
    phase_skipped = 0

    # Parse settings from the current render request state
    state = {}
    if isinstance(render_request, dict) and isinstance(render_request.get('state'), dict):
        state = dict(render_request.get('state') or {})
    
    overlays = set(state.get('overlay_values') or [])
    override_period_source = str(state.get('override_period_source') or 'manual/search')
    phase_period_pending = bool(state.get('phase_period_pending', False))
    suppress_catalog_phase_period = bool(state.get('suppress_catalog_phase_period', False))
    phase_requested = 'phase' in overlays
    min_p, max_p = _normalize_period_search_bounds(min_period, max_period)
    search_method = str(period_method or AUTO_FALLBACK_PERIOD_METHOD).strip().lower()
    
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

    for i, cid in enumerate(candidate_ids):
        # Update progress text
        set_progress(
            f"Exporting {i+1}/{total} "
            f"(PDF ok: {success}, phase found: {phase_found}, phase skipped: {phase_skipped}, failed: {failed})..."
        )
        
        try:
            payload, _stored_lc_path, source_path = _candidate_context(cid)
            plot_dir_path = _review_plot_dir_for_context(source_path)
            run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
            show_phase_for_candidate = bool(phase_requested)
            candidate_override_period = None
            candidate_override_source = override_period_source
            candidate_phase_pending = phase_period_pending
            candidate_suppress_catalog_period = suppress_catalog_phase_period
            candidate_phase_status = "not_requested"

            if phase_requested:
                candidate_phase_pending = False
                candidate_suppress_catalog_period = False
                search_payload = dict(payload or {})
                if source_path and not search_payload.get("source_path"):
                    search_payload["source_path"] = str(source_path)
                stored_period, _stored_source = shared_resolve_stored_review_period(search_payload)
                if stored_period is not None and np.isfinite(stored_period) and stored_period > 0:
                    candidate_phase_status = "found"
                else:
                    result, _label = _run_period_search_for_payload(
                        search_payload,
                        min_period=min_p,
                        max_period=max_p,
                        method=search_method,
                    )
                    best_period = np.nan
                    if isinstance(result, dict):
                        try:
                            best_period = float(result.get("best_period"))
                        except (TypeError, ValueError):
                            best_period = np.nan
                    if np.isfinite(best_period) and best_period > 0:
                        candidate_override_period = float(best_period)
                        method_label = str((result or {}).get("method") or search_method.upper())
                        candidate_override_source = f"Batch auto-search ({method_label})"
                        candidate_phase_pending = False
                        candidate_suppress_catalog_period = False
                        candidate_phase_status = "found"
                    else:
                        show_phase_for_candidate = False
                        candidate_phase_pending = False
                        candidate_suppress_catalog_period = False
                        candidate_phase_status = "skipped"
            
            asas_sn_id = str(payload.get('asas_sn_id') or payload.get('candidate_id') or 'unknown')
            fname = f"malca_plot_{asas_sn_id}.pdf"
            out_file = out_dir / fname
            
            image_bytes = build_review_lightcurve_publication_pdf(
                payload,
                plot_dir=plot_dir_path,
                selected_cameras=list(state.get('selected_cameras') or []),
                selected_bands=list(state.get('selected_bands') or ['g', 'V']),
                filter_bad_cameras='filter_bad_cameras' in overlays,
                show_baseline=baseline_opacity > 0,
                show_event_markers='markers' in overlays,
                show_residuals='residuals' in overlays,
                show_phase_fold=show_phase_for_candidate,
                phase_panel_mode=_coerce_choice(state.get('phase_panel_mode'), {'fold', 'time'}, 'fold'),
                show_raw_mag='raw' in overlays,
                override_period=candidate_override_period,
                override_period_source=candidate_override_source,
                phase_period_pending=candidate_phase_pending,
                suppress_catalog_phase_period=candidate_suppress_catalog_period,
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
                candidate_id=cid,
            )
            
            out_file.write_bytes(image_bytes)
            success += 1
            if phase_requested:
                if candidate_phase_status == "found":
                    phase_found += 1
                else:
                    phase_skipped += 1
            
        except Exception as e:
            failed += 1
            print(f"Error exporting PDF for candidate {cid}: {e}")
            
    return (
        f"Export complete! PDF ok: {success}, phase found: {phase_found}, "
        f"phase skipped: {phase_skipped}, failed: {failed}. "
        f"Total: {total} PDFs saved to {out_dir}"
    )
