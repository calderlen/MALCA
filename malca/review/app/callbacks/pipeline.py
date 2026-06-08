# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _run_pipeline_impl(set_progress, run_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_sed_photometry_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, rerun_multi_survey_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick):

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if triggered_id == 'auto-run-pipeline-trigger' and not auto_trigger:
        raise dash.exceptions.PreventUpdate

    queue_size = int((queue_data or {}).get('queue_size') or 0) if isinstance(queue_data, dict) else 0
    print(f"[run_pipeline_callback] Triggered by: {triggered_id}, auto_trigger: {auto_trigger}, queue_size: {queue_size}, idx: {idx}")

    if (
        not run_clicks
        and not rerun_clicks
        and not rerun_stats_clicks
        and not rerun_events_clicks
        and not rerun_characterize_clicks
        and not rerun_sed_photometry_clicks
        and not rerun_vetting_clicks
        and not rerun_external_lcs_clicks
        and not rerun_multi_survey_clicks
        and not auto_trigger
    ):
        raise dash.exceptions.PreventUpdate
        
    candidate_id = None
    fetch_mode = None
    if auto_trigger:
        candidate_id = auto_trigger.get('candidate_id')
        fetch_mode = auto_trigger.get('mode')
    else:
        candidate_id = _queue_candidate_id(queue_data, idx)
        if candidate_id is None:
            return "No candidate selected", no_update, no_update
        
    if not candidate_id:
        return "No candidate_id", no_update, no_update

    try:
        try:
            progress_tick = int(current_progress_tick or 0)
        except Exception:
            progress_tick = 0
        log_lines: list[str] = []

        def append_log_line(text: str) -> None:
            nonlocal log_lines
            line = str(text or '').strip()
            if not line:
                return
            if log_lines and log_lines[-1] == line:
                return
            log_lines.append(line)
            if len(log_lines) > 300:
                log_lines = log_lines[-300:]

        def emit_progress(msg: str, *, bump_render: bool = False, reset_log: bool = False):
            nonlocal progress_tick
            text = str(msg or "")
            text = text[:300] if len(text) > 300 else text
            if reset_log:
                log_lines.clear()
            append_log_line(text)
            if bump_render:
                progress_tick += 1
            if set_progress:
                try:
                    set_progress((text, progress_tick, {'lines': list(log_lines)}))
                except Exception:
                    pass
            else:
                print(f"[pipeline] {text}")

        def p(msg):
            emit_progress(msg, bump_render=False)

        def on_stage_complete(stage_name: str):
            emit_progress(f"✓ {stage_name} complete", bump_render=True)

        emit_progress(f"Running pipeline for {candidate_id}...", reset_log=True)
            
        with closing(db_connect(Path(DB_PATH))) as conn:
            
            # Determine if this was an explicit deep fetch that should bypass caching
            force_stages = []
            force_only = False
            if triggered_id == 'rerun-pipeline-btn':
                force_stages = ['stats', 'events', 'characterize', 'sed_photometry', 'sed_model_fit', 'vetting', 'external_lcs', 'multi_survey_features']
            elif triggered_id == 'rerun-stage-stats-btn':
                force_stages = ['stats']
                force_only = True
            elif triggered_id == 'rerun-stage-events-btn':
                force_stages = ['events']
                force_only = True
            elif triggered_id == 'rerun-stage-characterize-btn':
                force_stages = ['characterize']
                force_only = True
            elif triggered_id == 'rerun-stage-sed-photometry-btn':
                force_stages = ['sed_photometry', 'sed_model_fit']
                force_only = True
            elif triggered_id == 'rerun-stage-vetting-btn':
                force_stages = ['vetting']
                force_only = True
            elif triggered_id == 'rerun-stage-external-lcs-btn':
                force_stages = ['external_lcs', 'multi_survey_features']
                force_only = True
            elif triggered_id == 'rerun-stage-multi-survey-btn':
                force_stages = ['multi_survey_features']
                force_only = True
            elif fetch_mode == 'full':
                force_stages = ['stats', 'events', 'characterize', 'sed_photometry', 'sed_model_fit', 'vetting']
            elif fetch_mode in ('full_ext', 'full_ext_crts'):
                force_stages = ['stats', 'events', 'characterize', 'sed_photometry', 'sed_model_fit', 'vetting', 'external_lcs', 'multi_survey_features']
                
            stages = run_missing_stages(
                conn,
                candidate_id,
                progress_callback=p,
                stage_complete_callback=on_stage_complete,
                force_stages=force_stages,
                only_force=force_only,
            )
            
            # If triggered by a full_ext fetch, ensure we run external LCs
            if fetch_mode == 'full_ext' and 'external_lcs' not in stages:
                p("Running external LCs...")
                from malca.vetting import fetch_external_lcs



                payload = get_candidate_payload(conn, candidate_id)
                df = pd.DataFrame([payload])
                run_dir = _resolve_run_dir_from_plot_dir(PLOT_DIR)
                ext_output = (
                    run_dir / "results"
                    if run_dir
                    else (_APP_REPO_ROOT / DEFAULT_OUTPUT_DIR / "results")
                )
                ext_output.mkdir(parents=True, exist_ok=True)
                df_ext = fetch_external_lcs(
                    df,
                    output_dir=ext_output,
                    run_atlas=False,
                    run_ztf=True,
                    run_gaia_epoch=True,
                    run_tess=True,
                    run_neowise=True,
                    run_kepler=True,
                    run_aavso=True,
                    run_ogle=True,
                    run_stripe82=True,
                    run_allwise_mep=True,
                    run_vvvx_virac=True,
                    run_ps1=True,
                    run_crts=True,
                    progress_callback=p,
                )
                if isinstance(df_ext, pd.DataFrame) and not df_ext.empty:
                    row = df_ext.iloc[0].to_dict()
                    update_candidate_payload(conn, candidate_id, row)
                    failures = list(getattr(df_ext, "attrs", {}).get("external_lc_failures") or [])
                    if failures:
                        p(f"External LCs finished with failures; leaving stage partial: {'; '.join(failures[:3])}")
                    else:
                        stages.append('external_lcs')
                        on_stage_complete('external_lcs')
                        p("Computing multi-survey features...")
                        from malca.review.pipeline import _run_multi_survey_features_stage

                        payload = get_candidate_payload(conn, candidate_id)
                        _run_multi_survey_features_stage(payload, ext_output, p)
                        update_candidate_payload(conn, candidate_id, payload)
                        stages.append('multi_survey_features')
                        on_stage_complete('multi_survey_features')

        refresh_idx = int(idx or 0) if idx is not None else 0
        if stages:
            _index_external_lc_paths.cache_clear()
            _index_external_lc_paths_from_root.cache_clear()
            return f"✓ Ran: {', '.join(stages)}", no_update, refresh_idx
        else:
            if triggered_id in {
                'rerun-pipeline-btn',
                'rerun-stage-stats-btn',
                'rerun-stage-events-btn',
                'rerun-stage-characterize-btn',
                'rerun-stage-sed-photometry-btn',
                'rerun-stage-vetting-btn',
                'rerun-stage-external-lcs-btn',
                'rerun-stage-multi-survey-btn',
            }:
                return "No stages could be rerun (missing requirements)", no_update, no_update
            return "All stages already complete (or missing requirements)", no_update, no_update
    except Exception as e:

        traceback.print_exc()
        return f"✗ Pipeline failed: {str(e)}", no_update, no_update

if _background_callback_manager is not None:
    @app.callback(
        [Output('pipeline-run-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        [Input('run-pipeline-btn', 'n_clicks'),
         Input('rerun-pipeline-btn', 'n_clicks'),
         Input('rerun-stage-stats-btn', 'n_clicks'),
         Input('rerun-stage-events-btn', 'n_clicks'),
         Input('rerun-stage-characterize-btn', 'n_clicks'),
         Input('rerun-stage-sed-photometry-btn', 'n_clicks'),
         Input('rerun-stage-vetting-btn', 'n_clicks'),
         Input('rerun-stage-external-lcs-btn', 'n_clicks'),
         Input('rerun-stage-multi-survey-btn', 'n_clicks'),
         Input('auto-run-pipeline-trigger', 'data')],
        [State('queue-data', 'data'),
         State('current-index', 'data'),
         State('import-trigger', 'data'),
         State('pipeline-progress-trigger', 'data')],
        background=True,
        running=[
            (Output('run-pipeline-btn', 'disabled'), True, False),
            (Output('rerun-pipeline-btn', 'disabled'), True, False),
            (Output('rerun-stage-stats-btn', 'disabled'), True, False),
            (Output('rerun-stage-events-btn', 'disabled'), True, False),
            (Output('rerun-stage-characterize-btn', 'disabled'), True, False),
            (Output('rerun-stage-sed-photometry-btn', 'disabled'), True, False),
            (Output('rerun-stage-vetting-btn', 'disabled'), True, False),
            (Output('rerun-stage-external-lcs-btn', 'disabled'), True, False),
            (Output('rerun-stage-multi-survey-btn', 'disabled'), True, False),
        ],
        progress=[
            Output('pipeline-run-status', 'children'),
            Output('pipeline-progress-trigger', 'data'),
            Output('pipeline-module-log', 'data'),
        ],
        prevent_initial_call=True
    )
    def run_pipeline_callback(set_progress, n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_sed_photometry_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, rerun_multi_survey_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick):
        return _run_pipeline_impl(set_progress, n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_sed_photometry_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, rerun_multi_survey_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick)
else:
    @app.callback(
        [Output('pipeline-run-status', 'children'),
         Output('import-trigger', 'data', allow_duplicate=True),
         Output('current-index', 'data', allow_duplicate=True)],
        [Input('run-pipeline-btn', 'n_clicks'),
         Input('rerun-pipeline-btn', 'n_clicks'),
         Input('rerun-stage-stats-btn', 'n_clicks'),
         Input('rerun-stage-events-btn', 'n_clicks'),
         Input('rerun-stage-characterize-btn', 'n_clicks'),
         Input('rerun-stage-sed-photometry-btn', 'n_clicks'),
         Input('rerun-stage-vetting-btn', 'n_clicks'),
         Input('rerun-stage-external-lcs-btn', 'n_clicks'),
         Input('rerun-stage-multi-survey-btn', 'n_clicks'),
         Input('auto-run-pipeline-trigger', 'data')],
        [State('queue-data', 'data'),
         State('current-index', 'data'),
         State('import-trigger', 'data'),
         State('pipeline-progress-trigger', 'data')],
        prevent_initial_call=True
    )
    def run_pipeline_callback(n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_sed_photometry_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, rerun_multi_survey_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick):
        return _run_pipeline_impl(None, n_clicks, rerun_clicks, rerun_stats_clicks, rerun_events_clicks, rerun_characterize_clicks, rerun_sed_photometry_clicks, rerun_vetting_clicks, rerun_external_lcs_clicks, rerun_multi_survey_clicks, auto_trigger, queue_data, idx, current_trigger, current_progress_tick)


# Export reviews
