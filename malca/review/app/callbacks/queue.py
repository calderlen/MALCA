# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    Output('queue-data', 'data'),
    [Input('refresh-btn', 'n_clicks'),
     Input('import-trigger', 'data'),
     Input('queue-source-path', 'data'),
     Input('restored-filter-applied', 'data'),
     Input('numeric-filter-bounds', 'data')],
    _queue_states + _TEXT_OPTION_STATES + _SELECT_OPTION_STATES,
    prevent_initial_call=False
)
def load_queue(refresh_clicks, import_trigger, queue_source_scope, restored_filters, numeric_bounds, *callback_states):
    """Load queue data from all sidebar filter states."""
    n_values = len(_queue_states)
    n_text_opts = len(_TEXT_OPTION_STATES)
    state_values = callback_states[:n_values]
    text_option_values = callback_states[n_values:n_values + n_text_opts]
    select_option_values = callback_states[n_values + n_text_opts:]

    with closing(db_connect(Path(DB_PATH))) as conn:
        ui_state = _queue_filter_ui_state_from_values(*state_values)
        ui_state = _merge_unhydrated_saved_queue_filter_ui_state(
            ui_state,
            restored_filters,
            text_option_values,
            select_option_values,
        )
        numeric_bounds = numeric_bounds or {}
        filter_params = _queue_filter_params_from_ui_state(
            ui_state,
            numeric_bounds,
            queue_source_scope,
        )

        queue_data = create_queue_data_dict(conn, filter_params)
        active_filters = {
            k: v for k, v in filter_params.items()
            if v not in (None, 'Any', [])
        }
        print(f"[queue] size={queue_data['queue_size']} active_filters={active_filters}")
        return queue_data


@app.callback(
    Output('queue-filter-provenance-panel', 'children'),
    Input('queue-data', 'data'),
    prevent_initial_call=False,
)
def render_queue_filter_provenance(queue_data):
    """Render queue-filter attrition details in the sidebar."""
    return _render_queue_filter_provenance_panel(queue_data)


@app.callback(
    Output('preload-trigger', 'data'),
    Input('current-index', 'data'),
    Input('queue-data', 'data'),
    State('plot-mode', 'value'),
    prevent_initial_call=False,
)
def preload_next_candidates(idx, queue_data, plot_mode):
    """Warm the next candidate's LC/baseline caches after a short navigation debounce."""
    generation = _next_preload_generation()
    if str(plot_mode or 'native').strip().lower() != 'native':
        return no_update
    if queue_data is None or not isinstance(queue_data, dict):
        return no_update
    candidate_ids = queue_data.get('candidate_ids')
    if not candidate_ids:
        return no_update
    idx = int(idx or 0)
    db_path = DB_PATH

    def do_preload():
        time.sleep(_PRELOAD_DELAY_SEC)
        if not _is_current_preload_generation(generation):
            return
        try:
            with closing(db_connect(Path(db_path))) as conn:
                for i in range(1, _PRELOAD_LOOKAHEAD + 1):
                    if not _is_current_preload_generation(generation):
                        return
                    if idx + i >= len(candidate_ids):
                        break
                    cid = candidate_ids[idx + i]
                    row = conn.execute(
                        "SELECT payload_json, source_path FROM candidates WHERE candidate_id = ?",
                        (str(cid),),
                    ).fetchone()
                    if not row:
                        continue
                    try:
                        payload = json.loads(row[0]) if row[0] else {}
                    except Exception:
                        continue
                    source_path = (row[1] or "").strip()
                    run_dir = _run_dir_from_source_path(source_path) if source_path else None
                    plot_dir = (run_dir / "plots") if run_dir and (run_dir / "plots").exists() else None
                    run_params = None
                    if run_dir and (run_dir / "run_params.json").exists():
                        try:
                            with open(run_dir / "run_params.json") as f:
                                run_params = json.load(f)
                        except Exception:
                            pass
                    try:
                        warm_caches_for_candidate(payload, plot_dir, run_params=run_params)
                    except Exception:
                        pass
        except Exception:
            pass

    t = Thread(target=do_preload, daemon=True)
    t.start()
    return no_update


def _queue_candidate_id(queue_data, idx) -> str | None:
    """Return the active candidate ID from queue-data dict storage."""
    if not isinstance(queue_data, dict):
        return None
    candidate_ids = queue_data.get('candidate_ids') or []
    if idx is None:
        return None
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(candidate_ids):
        return None
    return str(candidate_ids[idx])


def _normalized_queue_index(queue_data, idx) -> int | None:
    """Clamp a queue index to the currently visible queue, if needed."""
    if not isinstance(queue_data, dict):
        return None

    candidate_ids = queue_data.get('candidate_ids') or []
    try:
        current_idx = int(idx or 0)
    except (TypeError, ValueError):
        current_idx = 0

    if not candidate_ids:
        return 0 if current_idx != 0 else None

    clamped_idx = min(max(current_idx, 0), len(candidate_ids) - 1)
    if clamped_idx == current_idx:
        return None
    return clamped_idx


@app.callback(
    Output('current-index', 'data', allow_duplicate=True),
    [Input('queue-data', 'data'),
     Input('current-index', 'data')],
    prevent_initial_call='initial_duplicate',
)
def clamp_current_index_to_queue(queue_data, idx):
    """Keep the active index valid when filters change the visible queue."""
    normalized_idx = _normalized_queue_index(queue_data, idx)
    if normalized_idx is None:
        raise dash.exceptions.PreventUpdate
    return normalized_idx


def _has_external_period(payload: dict | None) -> bool:
    """Whether a payload already has a catalog or validated pipeline period."""
    return shared_has_external_period(payload)


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _score_period_harmonic_candidate(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    period: float,
    *,
    n_bins: int = 48,
    lag_weight: float = 0.1,
) -> dict[str, float]:
    if not np.isfinite(period) or period <= 0:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}

    all_jd = [jd for jd, _ in band_resid.values() if jd.size > 0]
    if not all_jd:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}
    jd0 = float(min(np.min(jd) for jd in all_jd))

    templates: dict[int, np.ndarray] = {}
    scatter_ratios: list[float] = []

    for band, (jd, resid) in band_resid.items():
        phase = np.mod((jd - jd0) / float(period), 1.0)
        template, _ = phase_template(phase, resid, n_bins=n_bins)
        templates[band] = template

        bin_idx = np.floor(phase * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        model = template[bin_idx]
        valid = np.isfinite(model) & np.isfinite(resid)
        if np.count_nonzero(valid) < 20:
            continue

        raw_sigma = _robust_sigma(resid[valid])
        folded_sigma = _robust_sigma(resid[valid] - model[valid])
        if np.isfinite(raw_sigma) and raw_sigma > 0 and np.isfinite(folded_sigma):
            scatter_ratios.append(float(folded_sigma / raw_sigma))

    if not scatter_ratios:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}

    scatter_ratio = float(np.mean(scatter_ratios))
    lag_phase = np.nan
    if 0 in templates and 1 in templates:
        lag_phase = template_phase_lag(templates[0], templates[1])
    lag_term = 0.0 if not np.isfinite(lag_phase) else float(lag_phase)

    objective = float(scatter_ratio + lag_weight * lag_term)
    return {"objective": objective, "scatter_ratio": scatter_ratio, "lag_phase": lag_phase}


def _arbitrate_harmonic_period(
    band_dfs: dict[int, pd.DataFrame],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
) -> tuple[float, float, dict[str, float]]:
    if not np.isfinite(base_period) or base_period <= 0:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    band_resid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for band in (0, 1):
        bdf = band_dfs.get(band)
        if bdf is None or bdf.empty or "resid" not in bdf.columns:
            continue
        jd = bdf["JD"].to_numpy(dtype=float)
        resid = bdf["resid"].to_numpy(dtype=float)
        mask = np.isfinite(jd) & np.isfinite(resid)
        if np.count_nonzero(mask) < 30:
            continue
        band_resid[band] = (jd[mask], resid[mask])

    if not band_resid:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    candidates: list[tuple[float, float, dict[str, float]]] = []
    for factor in (1.0, 2.0, 0.5):
        p = float(base_period) * float(factor)
        if not np.isfinite(p) or p <= 0:
            continue
        if p < float(min_period) or p > float(max_period):
            continue
        if any(abs(p - p_prev) <= 1e-10 * max(1.0, abs(p), abs(p_prev)) for _, p_prev, _ in candidates):
            continue
        score = _score_period_harmonic_candidate(band_resid, p)
        candidates.append((float(factor), p, score))

    if not candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    finite_candidates = [c for c in candidates if np.isfinite(c[2].get("objective", np.nan))]
    if not finite_candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    best_factor, best_period, best_score = min(finite_candidates, key=lambda x: float(x[2]["objective"]))
    base_entry = next((c for c in finite_candidates if abs(c[0] - 1.0) < 1e-12), None)
    base_objective = float(base_entry[2]["objective"]) if base_entry is not None else np.nan

    # Avoid jittery harmonic flips unless improvement is meaningful.
    if base_entry is not None and abs(best_factor - 1.0) > 1e-12:
        improvement = (base_objective - float(best_score["objective"])) / max(abs(base_objective), 1e-9)
        if not np.isfinite(improvement) or improvement < 0.02:
            best_factor, best_period, best_score = base_entry

    diag = {
        "objective": float(best_score.get("objective", np.nan)),
        "scatter_ratio": float(best_score.get("scatter_ratio", np.nan)),
        "lag_phase": float(best_score.get("lag_phase", np.nan)),
        "base_objective": base_objective,
    }
    return float(best_period), float(best_factor), diag


def _run_period_search_for_payload(
    payload: dict,
    *,
    min_period: float,
    max_period: float,
    method: str,
) -> tuple[dict | None, str]:
    """Run a period search against the current candidate payload."""
    plot_dir_path = _review_plot_dir_for_context(payload.get("source_path"))
    run_params = _load_run_params_for_plot_dir(str(plot_dir_path) if plot_dir_path else None)
    filter_bad_cameras = not bool(run_params.get('skip_camera_median', False))
    scatter_ratio = float(run_params.get('bad_camera_scatter_ratio', BAD_CAMERA_SCATTER_RATIO_THRESHOLD)) if run_params else BAD_CAMERA_SCATTER_RATIO_THRESHOLD
    clean_abs = float(run_params.get('clean_max_error_absolute', CLEAN_LC_MAX_ERROR_ABSOLUTE)) if run_params else CLEAN_LC_MAX_ERROR_ABSOLUTE
    clean_sig = float(run_params.get('clean_max_error_sigma', CLEAN_LC_MAX_ERROR_SIGMA)) if run_params else CLEAN_LC_MAX_ERROR_SIGMA
    baseline_name, baseline_kwargs, _baseline_warnings = _baseline_config_from_run_params(run_params)
    return shared_run_period_search_for_payload(
        payload,
        plot_dir=plot_dir_path,
        min_period=min_period,
        max_period=max_period,
        method=method,
        filter_bad_cameras=filter_bad_cameras,
        scatter_ratio=scatter_ratio,
        clean_max_error_absolute=clean_abs,
        clean_max_error_sigma=clean_sig,
        baseline_name=baseline_name,
        baseline_kwargs=baseline_kwargs,
    )



@app.callback(
    [Output('queue-data', 'data', allow_duplicate=True),
     Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input('candidate-search-btn', 'n_clicks'),
     Input('candidate-search-query', 'n_submit')],
    [State('candidate-search-query', 'value'),
     State('queue-data', 'data')],
    prevent_initial_call=True,
)
def _lookup_candidate_id_for_query(conn: sqlite3.Connection, query_text: str) -> tuple[str | None, str | None]:
    """Resolve a search query to a candidate_id and match label."""
    normalized = _format_large_integer_like_display(query_text).strip()
    exact_queries = (
        ("candidate_id", "SELECT candidate_id FROM candidates WHERE candidate_id = ? COLLATE NOCASE LIMIT 1"),
        ("ASAS-SN ID", "SELECT candidate_id FROM candidates WHERE asas_sn_id = ? COLLATE NOCASE LIMIT 1"),
        ("local LC path", "SELECT candidate_id FROM candidates WHERE lc_path = ? COLLATE NOCASE LIMIT 1"),
        ("Gaia ID", "SELECT candidate_id FROM candidates WHERE CAST(json_extract(payload_json, '$.gaia_id') AS TEXT) = ? COLLATE NOCASE LIMIT 1"),
        ("cluster LC path", "SELECT candidate_id FROM candidates WHERE CAST(json_extract(payload_json, '$.path') AS TEXT) = ? COLLATE NOCASE LIMIT 1"),
    )
    for label, query in exact_queries:
        row = conn.execute(query, (normalized,)).fetchone()
        if row is not None:
            return str(row[0]), label

    stem_query = normalized.lower()
    rows = conn.execute(
        "SELECT candidate_id, lc_path, source_path, "
        "CAST(json_extract(payload_json, '$.path') AS TEXT), "
        "CAST(json_extract(payload_json, '$.gaia_id') AS TEXT), "
        "payload_json "
        "FROM candidates"
    ).fetchall()
    for candidate_id, local_lc_path, source_path, cluster_lc_path, gaia_id, payload_json in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if candidate_id is not None and not payload.get("candidate_id"):
            payload["candidate_id"] = str(candidate_id)
        if cluster_lc_path and not payload.get("path"):
            payload["path"] = cluster_lc_path
        display_cluster_lc_path, effective_local_lc_path = _display_lc_paths(
            payload,
            stored_lc_path=local_lc_path,
            source_path=source_path,
        )
        for label, raw in (
            ("local LC stem", effective_local_lc_path),
            ("cluster LC stem", display_cluster_lc_path),
            ("Gaia ID", gaia_id),
        ):
            if not raw:
                continue
            text = _format_large_integer_like_display(raw).strip()
            if not text:
                continue
            if text.lower() == stem_query:
                return str(candidate_id), label
            try:
                path_obj = Path(text)
            except Exception:
                continue
            if path_obj.stem.lower() == stem_query or path_obj.name.lower() == stem_query:
                return str(candidate_id), label
    return None, None


def open_existing_candidate(n_clicks, n_submit, query, queue_data):
    """Jump to an existing candidate in the DB by common identifiers or LC stem."""
    _ = n_clicks, n_submit
    query_text = str(query or '').strip()
    if not query_text:
        raise dash.exceptions.PreventUpdate

    with closing(db_connect(Path(DB_PATH))) as conn:
        candidate_id, match_label = _lookup_candidate_id_for_query(conn, query_text)

    if candidate_id is None:
        return no_update, no_update, f"Candidate not found in DB: {query_text}"

    candidate_ids = list((queue_data or {}).get('candidate_ids') or []) if isinstance(queue_data, dict) else []
    if candidate_id in candidate_ids:
        return no_update, candidate_ids.index(candidate_id), f"Jumped to {candidate_id} in the current queue via {match_label or 'search'}."

    return (
        {'candidate_ids': [candidate_id], 'queue_size': 1, 'filter_hash': f'view:{candidate_id}'},
        0,
        f"Viewing {candidate_id} via {match_label or 'search'}. Refresh Queue to restore the filtered queue.",
    )


@app.callback(
    Output('last-candidate-saved', 'data'),
    Input('current-candidate-id', 'data'),
    State('startup-selection-applied', 'data'),
    prevent_initial_call=True,
)
def persist_last_candidate(current_candidate_id, startup_selection_applied):
    """Persist the most recently viewed candidate for this review DB."""
    if not startup_selection_applied:
        raise dash.exceptions.PreventUpdate
    candidate_id = str(current_candidate_id or '').strip()
    if not candidate_id:
        raise dash.exceptions.PreventUpdate
    with closing(db_connect(Path(DB_PATH))) as conn:
        save_app_state(conn, 'review_last_candidate', candidate_id)
    return int(time.time())


@app.callback(
    [Output('queue-data', 'data', allow_duplicate=True),
     Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('startup-selection-applied', 'data')],
    Input('queue-data', 'data'),
    State('startup-selection-applied', 'data'),
    prevent_initial_call='initial_duplicate',
)
def restore_startup_candidate(queue_data, already_applied):
    """Open CLI-requested candidate or last viewed candidate when the queue is ready."""
    if already_applied:
        raise dash.exceptions.PreventUpdate
    if queue_data is None:
        raise dash.exceptions.PreventUpdate

    explicit_query = str(INITIAL_CANDIDATE_QUERY or '').strip()
    candidate_ids = list((queue_data or {}).get('candidate_ids') or []) if isinstance(queue_data, dict) else []
    queue_size = int((queue_data or {}).get('queue_size') or 0) if isinstance(queue_data, dict) else 0

    # Wait for the real queue to materialize before consuming startup restore.
    # Queue initialization can briefly emit an empty queue before persisted filters
    # are restored, and marking startup selection as applied here would prevent
    # the saved candidate from reopening once the populated queue arrives.
    if not explicit_query and queue_size <= 0 and not candidate_ids:
        return no_update, no_update, no_update, no_update

    with closing(db_connect(Path(DB_PATH))) as conn:
        saved_candidate = str(load_app_state(conn, 'review_last_candidate', '') or '').strip()
        if explicit_query:
            candidate_id, match_label = _lookup_candidate_id_for_query(conn, explicit_query)
            if candidate_id is None:
                return no_update, no_update, f"Candidate not found in DB: {explicit_query}", True
            if candidate_id in candidate_ids:
                return no_update, candidate_ids.index(candidate_id), f"Opened {candidate_id} via {match_label or 'startup selection'}.", True
            return (
                {'candidate_ids': [candidate_id], 'queue_size': 1, 'filter_hash': f'view:{candidate_id}'},
                0,
                f"Opened {candidate_id} via {match_label or 'startup selection'}. Refresh Queue to restore the filtered queue.",
                True,
            )

    if saved_candidate and saved_candidate in candidate_ids:
        return no_update, candidate_ids.index(saved_candidate), f"Restored last candidate {saved_candidate}.", True
    return no_update, no_update, no_update, True


